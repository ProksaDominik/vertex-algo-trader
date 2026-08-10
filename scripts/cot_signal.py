"""COT (Commitment of Traders) positioning signal backtest -- a genuinely
different information source from anything price-derived (TA, stat-arb, ML
on OHLCV all failed). Tests both the classic contrarian read (extreme
speculative positioning precedes reversal) and the momentum read (a rising
net position confirms/leads the trend), pooled across the 5 currencies with
full 2015-2026 COT coverage (EUR/JPY/AUD/CAD/CHF).

Mechanics:
  - net_pct_oi = (noncomm_long - noncomm_short) / open_interest, per weekly
    report. z-scored over a rolling 52-week window.
  - CONTRARIAN: |z| > entry_z -> fade it (bet on reversion).
  - MOMENTUM:   |z change from last week| > entry_dz -> follow it.
  - Publication-lag safe: CFTC's Tuesday report_date is publicly released
    the following Friday: the signal is only usable from the first trading
    day >= report_date + 4 calendar days (no lookahead).
  - Direction sign is flipped for USD-base pairs (USDJPY/USDCAD/USDCHF) --
    long JPY/CAD/CHF positioning is BEARISH for those pairs, since the
    non-USD currency is the quote currency there, not the base.
  - Entry/exit: enter at that trading day's close, hold exactly one COT
    cycle (until the next signal's entry day, ~1 week), sized off a 2.5xATR
    stop (checked daily through the hold) so risk is well-defined; real
    spread+commission cost from instruments.PAIR_SPECS.

Small-sample caveat, stated explicitly in the output: weekly resolution
means even 11 years of history is only ~600 observations per currency --
nowhere near the 200-OOS-trades-per-symbol bar the daily/intraday scripts
used. Pooled across 5 currencies partially compensates but the currencies'
weekly reports land on the same day, so pooled trades are not fully
independent -- treat any result here as a lower-confidence read than the
other three comparisons.

Usage: python scripts/cot_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.indicators import atr  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
SL_ATR_MULT = 2.5
ZWINDOW = 52  # weeks

# currency -> (pair, invert) -- invert=True when the COT currency is the
# QUOTE currency (USD-base pairs), so bullish-currency = bearish-pair.
CURRENCIES = {
    "eur": ("EURUSD", False),
    "jpy": ("USDJPY", True),
    "aud": ("AUDUSD", False),
    "cad": ("USDCAD", True),
    "chf": ("USDCHF", True),
}


def load_price(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{sym.lower()}_1d.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_cot(code: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"cot_{code}.parquet")
    df["net_pct_oi"] = (df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]) / df["open_interest_all"]
    df["z"] = (df["net_pct_oi"] - df["net_pct_oi"].rolling(ZWINDOW).mean()) / df["net_pct_oi"].rolling(ZWINDOW).std()
    df["dz"] = df["z"].diff()
    df["usable_from"] = df["date"] + pd.Timedelta(days=4)
    return df.dropna(subset=["z"])


def simulate(mode: str, entry_thresh: float, split_frac: float = 0.70) -> dict:
    all_rs_train: list[float] = []
    all_rs_test: list[float] = []
    per_currency = []

    for code, (pair, invert) in CURRENCIES.items():
        cot = load_cot(code)
        price = load_price(pair)
        spec = PAIR_SPECS[pair]
        cm, spread, comm = spec["contract_mult"], spec["spread_price"], spec["commission_per_lot"]
        a = atr(price, 14)

        split_idx = int(len(cot) * split_frac)
        rs_train: list[float] = []
        rs_test: list[float] = []

        for i in range(1, len(cot) - 1):
            row = cot.iloc[i]
            sig = row["z"] if mode == "contrarian" else row["dz"]
            if pd.isna(sig) or abs(sig) < entry_thresh:
                continue
            raw_dir = (-1 if sig > 0 else 1) if mode == "contrarian" else (1 if sig > 0 else -1)
            direction = -raw_dir if invert else raw_dir

            entry_dates = price.index[price.index >= row["usable_from"]]
            if len(entry_dates) == 0:
                continue
            entry_ts = entry_dates[0]
            entry_i = price.index.get_loc(entry_ts)

            next_row = cot.iloc[i + 1]
            exit_dates = price.index[price.index >= (next_row["date"] + pd.Timedelta(days=4))]
            if len(exit_dates) == 0:
                continue
            exit_ts = exit_dates[0]
            exit_i = price.index.get_loc(exit_ts)
            if exit_i <= entry_i:
                continue

            atr_i = a.loc[entry_ts] if entry_ts in a.index else np.nan
            if pd.isna(atr_i) or atr_i <= 0:
                continue
            entry_price = price["close"].loc[entry_ts]
            sl = entry_price - direction * SL_ATR_MULT * atr_i
            lots = RISK_USD / (SL_ATR_MULT * atr_i * cm)

            path = price.iloc[entry_i + 1: exit_i + 1]
            stopped_price = None
            for ts, bar in path.iterrows():
                if direction == 1 and bar["low"] <= sl:
                    stopped_price = sl
                    break
                if direction == -1 and bar["high"] >= sl:
                    stopped_price = sl
                    break
            exit_price = stopped_price if stopped_price is not None else price["close"].loc[exit_ts]

            pnl = direction * (exit_price - entry_price) * lots * cm
            cost = (spread * cm + comm) * lots * 2
            r = (pnl - cost) / RISK_USD
            (rs_train if row["date"] < cot.iloc[split_idx]["date"] else rs_test).append(r)

        all_rs_train += rs_train
        all_rs_test += rs_test
        if rs_test:
            per_currency.append((code.upper(), len(rs_test), sum(rs_test) / len(rs_test)))

    return {"train": all_rs_train, "test": all_rs_test, "per_currency": per_currency}


def report(mode: str, thresh: float) -> None:
    res = simulate(mode, thresh)
    rs = res["test"]
    n = len(rs)
    print(f"\n--- mode={mode} thresh={thresh} ---")
    for code, cn, ce in res["per_currency"]:
        print(f"    {code}  n={cn:4d}  E={ce:+.3f}R")
    if n < 5:
        print(f"  test n={n} -- too few to evaluate")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    wr = len(wins) / n
    e_r = sum(rs) / n
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    train_e = (sum(res["train"]) / len(res["train"])) if res["train"] else float("nan")
    if rr > 0 and 0 < wr < 1:
        p_pass, _, _, _ = mc_sim(0.005, wr, rr, tpd=1)  # ~1 pooled trade/day equivalent, weekly cadence
    else:
        p_pass = float("nan")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
    print(f"  TEST  n={n}  win_rate={wr:.1%}  avg_RR={rr:.2f}  E_net={e_r:+.3f}R  PF={pf_s}  "
          f"MC_P(pass)={pp_s}   (train E_net={train_e:+.3f}R, n={len(res['train'])})")


def report_reversed(mode: str, thresh: float) -> None:
    """Reverse the split: TEST on the first 30%, TRAIN on the last 70% --
    a cheap robustness check with zero new parameter fitting. A real,
    stable edge should not depend on which half is called 'test'."""
    res = simulate(mode, thresh, split_frac=0.30)
    # simulate() puts everything before the split into "train" -- so with
    # split_frac=0.30, "train" here is actually the FIRST 30% (what we want
    # to report as the reversed "test"), and "test" is the LAST 70%.
    rs = res["train"]
    n = len(rs)
    print(f"\n--- REVERSED mode={mode} thresh={thresh} (test=first 30%, train=last 70%) ---")
    if n < 5:
        print(f"  n={n} -- too few to evaluate")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    wr = len(wins) / n
    e_r = sum(rs) / n
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  n={n}  win_rate={wr:.1%}  E_net={e_r:+.3f}R  PF={pf_s}")


def main() -> None:
    print("=" * 96)
    print("COT POSITIONING SIGNAL -- 5 currencies (EUR/JPY/AUD/CAD/CHF), 2015-2026 weekly")
    print("CAVEAT: weekly resolution = small sample even over 11 years; treat as lower-confidence")
    print("=" * 96)
    for mode in ("contrarian", "momentum"):
        for thresh in (1.0, 1.5, 2.0) if mode == "contrarian" else (0.3, 0.5, 0.75):
            report(mode, thresh)

    print("\n" + "=" * 96)
    print("ROBUSTNESS CHECK: reversed split (does the contrarian edge hold both directions?)")
    print("=" * 96)
    for thresh in (1.0, 1.5, 2.0):
        report_reversed("contrarian", thresh)


if __name__ == "__main__":
    main()
