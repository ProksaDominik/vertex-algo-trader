"""Cross-sectional currency momentum -- an academically documented FX factor
(Menkhoff, Sarno, Schmeling & Schrimpf, "Currency Momentum Strategies",
Journal of Financial Economics 2012; also QuantPedia's currency momentum
factor), structurally different from every time-series trend-following
strategy already tested in this project.

Time-series trend-following (already tested, all failed) asks: "is THIS
pair trending?" Cross-sectional momentum instead asks: "which currencies are
CURRENTLY STRONGEST/WEAKEST relative to each other?" -- ranks all available
currencies by trailing return, goes long the strongest, short the weakest,
rebalanced periodically. A currency can be flat in absolute terms and still
get a signal here if it's outperforming/underperforming its peers.

Mechanics:
  - 7 currencies vs USD: EUR, GBP, JPY, AUD, CAD, CHF, NZD (from the 7
    Dukascopy-major pairs already cached as daily bars). JPY/CAD/CHF returns
    are computed INVERTED (USDJPY falling = JPY strengthening).
  - At each ~21-trading-day rebalance: rank by trailing 21-day return.
    LONG the pair for the top-K strongest currencies, SHORT the pair for
    the bottom-K weakest. Hold to the next rebalance (or an ATR stop).
  - Real spread+commission cost; ATR-based risk sizing (RISK_USD split
    across however many positions are opened that cycle).

Usage: python scripts/fx_momentum_factor.py [--top-k N] [--lookback-days N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.indicators import atr  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0

# currency -> (pair, invert). invert=True when currency is the QUOTE (USD-base pairs).
CURRENCIES = {
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "JPY": ("USDJPY", True),
    "AUD": ("AUDUSD", False),
    "CAD": ("USDCAD", True),
    "CHF": ("USDCHF", True),
    "NZD": ("NZDUSD", False),
}


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{sym.lower()}_1d.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.dropna()


def backtest(top_k: int, lookback_days: int, hold_days: int, sl_atr_mult: float,
             bars_by_currency: dict, split_frac: float = 0.70) -> dict:
    all_dates = sorted(set.union(*[set(b.index) for b in bars_by_currency.values()]))
    closes = pd.DataFrame({c: b["close"].reindex(all_dates) for c, b in bars_by_currency.items()}).ffill()
    atrs = {c: atr(bars_by_currency[c], 14).reindex(all_dates).ffill() for c in bars_by_currency}

    # per-currency "strength" return series (inverted where needed)
    rets = pd.DataFrame(index=closes.index)
    for c, (pair, invert) in CURRENCIES.items():
        r = closes[c].pct_change(lookback_days)
        rets[c] = -r if invert else r

    n = len(all_dates)
    train_rs, test_rs = [], []
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
            direction = 1 if c in strongest else -1
            if invert:
                direction = -direction  # trading the PAIR: strong JPY (short USDJPY) etc.
            spec = PAIR_SPECS[pair]
            cm, spread, comm = spec["contract_mult"], spec["spread_price"], spec["commission_per_lot"]

            entry_price = closes[c].iloc[i]
            a = atrs[c].iloc[i]
            if pd.isna(a) or a <= 0 or pd.isna(entry_price):
                continue
            sl = entry_price - direction * sl_atr_mult * a
            lots = RISK_USD / (sl_atr_mult * a * cm)

            path = closes[c].iloc[i + 1: exit_i + 1]
            stopped = False
            exit_price = closes[c].iloc[exit_i]
            for px in path:
                if direction == 1 and px <= sl:
                    exit_price = sl
                    stopped = True
                    break
                if direction == -1 and px >= sl:
                    exit_price = sl
                    stopped = True
                    break

            pnl = direction * (exit_price - entry_price) * lots * cm
            cost = (spread * cm + comm) * lots * 2
            r = (pnl - cost) / RISK_USD
            cycle_rs.append(r)

        (train_rs if i < split_i else test_rs).extend(cycle_rs)
        i += hold_days

    return {"train": train_rs, "test": test_rs}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--lookback-days", type=int, default=21)
    ap.add_argument("--hold-days", type=int, default=21)
    ap.add_argument("--sl-atr-mult", type=float, default=2.0)
    args = ap.parse_args()

    bars_by_currency = {c: load(pair) for c, (pair, _) in CURRENCIES.items()}

    print("=" * 96)
    print(f"CROSS-SECTIONAL CURRENCY MOMENTUM -- top_k={args.top_k} lookback={args.lookback_days}d "
          f"hold={args.hold_days}d sl={args.sl_atr_mult}xATR")
    print("=" * 96)

    res = backtest(args.top_k, args.lookback_days, args.hold_days, args.sl_atr_mult, bars_by_currency)
    summarize("TRAIN (first 70%)", res["train"])
    summarize("TEST (last 30%)", res["test"])

    print("\n--- robustness check: evaluate on first 30% directly ---")
    res_rev = backtest(args.top_k, args.lookback_days, args.hold_days, args.sl_atr_mult,
                        bars_by_currency, split_frac=0.30)
    summarize("first 30% (train slot of a 30/70 split)", res_rev["train"])


if __name__ == "__main__":
    main()
