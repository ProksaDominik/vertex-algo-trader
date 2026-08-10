"""Combine the two signals from this session that were CONSISTENTLY positive
(unlike everything else, which flipped sign under robustness checks):
  - COT contrarian positioning (scripts/cot_signal.py): fade extreme
    speculative positioning, weekly.
  - Cross-sectional currency momentum (scripts/fx_momentum_factor.py): long
    the strongest / short the weakest currency by trailing return, ~monthly.

These are structurally different information sources (positioning vs. price
momentum) -- if they're capturing complementary signal rather than the same
noise, requiring AGREEMENT between them before trading should improve on
either alone. If it doesn't, that's evidence they were both just noise.

Mechanics: at each momentum rebalance, for each candidate currency (top/
bottom-K by trailing return), only take the trade if the most recent COT
z-score for that SAME currency agrees with the momentum-implied direction
(momentum wants to SHORT a currency -> only do it if COT also reads that
currency as bearish-due-for-reversion, i.e. z > entry_z; and vice versa for
LONG). Currencies with no COT agreement are skipped, not traded anyway.

Usage: python scripts/combined_cot_momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

from metatraderbot.indicators import atr  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
COT_Z_ENTRY = 1.0  # the threshold that was robust in cot_signal.py

# currency -> (pair, invert vs USD)
CURRENCIES = {
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "JPY": ("USDJPY", True),
    "AUD": ("AUDUSD", False),
    "CAD": ("USDCAD", True),
    "CHF": ("USDCHF", True),
}
# NZD excluded -- no COT coverage post-2022 (see fetch_cot.py output)

COT_CODE = {"EUR": "eur", "GBP": "gbp", "JPY": "jpy", "AUD": "aud", "CAD": "cad", "CHF": "chf"}


def load_price(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{sym.lower()}_1d.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.dropna()


def load_cot(code: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"cot_{code}.parquet")
    df["net_pct_oi"] = (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / df["open_interest_all"]
    df["z"] = (df["net_pct_oi"] - df["net_pct_oi"].rolling(52).mean()) / df["net_pct_oi"].rolling(52).std()
    df["usable_from"] = df["date"] + pd.Timedelta(days=4)
    return df.dropna(subset=["z"]).sort_values("date").reset_index(drop=True)


def cot_z_as_of(cot_df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    """Most recent COT z-score usable (publication-lag safe) as of `date`."""
    usable = cot_df[cot_df["usable_from"] <= date]
    if usable.empty:
        return None
    return float(usable.iloc[-1]["z"])


def backtest(top_k: int, lookback_days: int, hold_days: int, sl_atr_mult: float,
             require_cot_agreement: bool, split_frac: float = 0.70) -> dict:
    bars_by_currency = {c: load_price(pair) for c, (pair, _) in CURRENCIES.items()}
    cot_by_currency = {c: load_cot(COT_CODE[c]) for c in CURRENCIES}

    all_dates = sorted(set.union(*[set(b.index) for b in bars_by_currency.values()]))
    closes = pd.DataFrame({c: b["close"].reindex(all_dates) for c, b in bars_by_currency.items()}).ffill()
    atrs = {c: atr(bars_by_currency[c], 14).reindex(all_dates).ffill() for c in bars_by_currency}

    rets = pd.DataFrame(index=closes.index)
    for c, (pair, invert) in CURRENCIES.items():
        r = closes[c].pct_change(lookback_days)
        rets[c] = -r if invert else r

    n = len(all_dates)
    train_rs, test_rs = [], []
    per_currency_log: list[tuple] = []
    skipped_no_agreement = 0
    taken = 0
    split_i = int(n * split_frac)

    i = lookback_days + 1
    while i < n - 1:
        date = all_dates[i]
        ranks = rets.iloc[i].dropna().sort_values()
        if len(ranks) < 2 * top_k:
            i += hold_days
            continue
        weakest = list(ranks.index[:top_k])
        strongest = list(ranks.index[-top_k:])

        exit_i = min(i + hold_days, n - 1)
        cycle_rs = []
        for c in strongest + weakest:
            pair, invert = CURRENCIES[c]
            momentum_dir = 1 if c in strongest else -1  # direction on the CURRENCY

            if require_cot_agreement:
                z = cot_z_as_of(cot_by_currency[c], date)
                if z is None:
                    continue
                # contrarian COT: z > entry -> currency due to WEAKEN (bearish);
                # z < -entry -> due to STRENGTHEN (bullish).
                cot_dir = None
                if z > COT_Z_ENTRY:
                    cot_dir = -1
                elif z < -COT_Z_ENTRY:
                    cot_dir = 1
                if cot_dir is None or cot_dir != momentum_dir:
                    skipped_no_agreement += 1
                    continue
            taken += 1

            direction = -momentum_dir if invert else momentum_dir  # direction on the PAIR
            spec = PAIR_SPECS[pair]
            cm, spread, comm = spec["contract_mult"], spec["spread_price"], spec["commission_per_lot"]

            entry_price = closes[c].iloc[i]
            a = atrs[c].iloc[i]
            if pd.isna(a) or a <= 0 or pd.isna(entry_price):
                continue
            sl = entry_price - direction * sl_atr_mult * a
            lots = RISK_USD / (sl_atr_mult * a * cm)

            path = closes[c].iloc[i + 1: exit_i + 1]
            exit_price = closes[c].iloc[exit_i]
            for px in path:
                if direction == 1 and px <= sl:
                    exit_price = sl
                    break
                if direction == -1 and px >= sl:
                    exit_price = sl
                    break

            pnl = direction * (exit_price - entry_price) * lots * cm
            cost = (spread * cm + comm) * lots * 2
            r = (pnl - cost) / RISK_USD
            cycle_rs.append(r)
            per_currency_log.append((date, c, r))

        (train_rs if i < split_i else test_rs).extend(cycle_rs)
        i += hold_days

    return {"train": train_rs, "test": test_rs, "skipped": skipped_no_agreement, "taken": taken,
            "per_currency_log": per_currency_log, "split_date": all_dates[split_i]}


def summarize(label: str, rs: list[float]) -> None:
    n = len(rs)
    if n < 5:
        print(f"  {label}: n={n} -- too few")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    wr = len(wins) / n
    e_r = sum(rs) / n
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    if rr > 0 and 0 < wr < 1:
        p_pass, _, _, _ = mc_sim(0.005, wr, rr, tpd=1)
    else:
        p_pass = float("nan")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
    print(f"  {label}: n={n}  win_rate={wr:.1%}  avg_RR={rr:.2f}  E_net={e_r:+.3f}R  PF={pf_s}  MC_P(pass)={pp_s}")


def main() -> None:
    print("=" * 96)
    print("COMBINED SIGNAL: cross-sectional momentum AND COT contrarian must AGREE")
    print("=" * 96)

    print("\n--- BASELINE: momentum alone (no COT filter) ---")
    base = backtest(top_k=2, lookback_days=21, hold_days=21, sl_atr_mult=2.0, require_cot_agreement=False)
    summarize("TRAIN", base["train"])
    summarize("TEST", base["test"])

    print("\n--- COMBINED: momentum + COT agreement required ---")
    comb = backtest(top_k=2, lookback_days=21, hold_days=21, sl_atr_mult=2.0, require_cot_agreement=True)
    print(f"  candidates: {comb['taken']} taken, {comb['skipped']} skipped (no COT agreement)")
    summarize("TRAIN", comb["train"])
    summarize("TEST", comb["test"])

    print("\n  per-currency breakdown of TEST trades (split date {}):".format(comb["split_date"].date()))
    by_currency: dict[str, list[float]] = {}
    for date, c, r in comb["per_currency_log"]:
        if date >= comb["split_date"]:
            by_currency.setdefault(c, []).append(r)
    for c, rs in sorted(by_currency.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {c}: n={len(rs)}  sum={sum(rs):+.2f}R  E_net={sum(rs)/len(rs):+.3f}R")

    print("\n--- robustness check: evaluate combined signal on first 30% directly ---")
    comb_rev = backtest(top_k=2, lookback_days=21, hold_days=21, sl_atr_mult=2.0,
                         require_cot_agreement=True, split_frac=0.30)
    summarize("first 30%", comb_rev["train"])


if __name__ == "__main__":
    main()
