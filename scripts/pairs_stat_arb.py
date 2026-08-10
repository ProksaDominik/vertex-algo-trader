"""Pairs mean-reversion (stat-arb) research: a structurally DIFFERENT edge
source from the 18 single-symbol TA patterns in full_strategy_comparison.py
(which all showed negative OOS expectancy). Trades the price RATIO between
two historically-correlated symbols reverting to its own rolling mean --
this doesn't depend on either symbol's own technical pattern being
predictive, only on the pair's relationship being mean-reverting.

Mechanics per pair (A, B):
  ratio = close_A / close_B
  z = (ratio - rolling_mean(ratio, W)) / rolling_std(ratio, W)
  z > +entry_z  -> SHORT A / LONG B ("spread" too wide, expect reversion)
  z < -entry_z  -> LONG A / SHORT B
  exit WIN  when |z| reverts to <= exit_z
  exit LOSS when |z| extends to >= stop_z, or max_hold_bars timeout
Both legs sized inverse-ATR (equal $ volatility contribution per leg,
risk_usd split 50/50), costs = real spread+commission on BOTH legs, BOTH
open and close (4 total cost hits per round-trip trade).

Usage: python scripts/pairs_stat_arb.py [--symbols N]
"""
from __future__ import annotations

import argparse
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

# (sym_A, sym_B) -- diverse candidates from the correlation scan, all with
# TRAIN-half and OOS-half correlation of the same sign and reasonably stable.
PAIRS = [
    ("EURJPY", "GBPJPY"),
    ("NZDJPY", "AUDJPY"),
    ("AUDUSD", "NZDUSD"),
    ("EURUSD", "GBPUSD"),
    ("US30", "US500"),
    ("GBPCHF", "EURCHF"),
    ("GBPJPY", "CHFJPY"),
    ("EURJPY", "CHFJPY"),
    ("EURJPY", "NZDJPY"),
    ("NZDJPY", "CHFJPY"),
    ("AUDJPY", "CHFJPY"),
    ("GBPJPY", "NZDJPY"),
    ("EURJPY", "AUDJPY"),
    ("GBPJPY", "AUDJPY"),
    ("USDJPY", "CHFJPY"),
    ("USDJPY", "AUDJPY"),
    ("USDJPY", "NZDJPY"),
]


def load(sym: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{sym.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def backtest_pair(sym_a: str, sym_b: str, bars_a: pd.DataFrame, bars_b: pd.DataFrame,
                   lookback: int, entry_z: float, exit_z: float, stop_z: float,
                   max_hold_bars: int) -> dict:
    idx = bars_a.index.intersection(bars_b.index)
    a = bars_a.loc[idx]
    b = bars_b.loc[idx]
    ratio = a["close"] / b["close"]
    roll_mean = ratio.rolling(lookback).mean()
    roll_std = ratio.rolling(lookback).std()
    z = (ratio - roll_mean) / roll_std
    atr_a = atr(a, 14)
    atr_b = atr(b, 14)

    spec_a, spec_b = PAIR_SPECS[sym_a], PAIR_SPECS[sym_b]
    cm_a, cm_b = spec_a["contract_mult"], spec_b["contract_mult"]
    cost_a = spec_a["spread_price"] * cm_a + spec_a["commission_per_lot"]  # $ per lot round trip (one leg, one side)
    cost_b = spec_b["spread_price"] * cm_b + spec_b["commission_per_lot"]

    trades = []
    pos = None  # dict: direction (+1 = long A/short B, -1 = short A/long B), entry idx, lots
    n = len(a)
    for i in range(lookback + 1, n):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        if pos is None:
            atr_a_i, atr_b_i = atr_a.iloc[i], atr_b.iloc[i]
            if pd.isna(atr_a_i) or pd.isna(atr_b_i) or atr_a_i <= 0 or atr_b_i <= 0:
                continue
            direction = None
            if zi > entry_z:
                direction = -1  # spread too high -> short A, long B
            elif zi < -entry_z:
                direction = 1   # spread too low -> long A, short B
            if direction is None:
                continue
            lots_a = (RISK_USD / 2) / (atr_a_i * cm_a)
            lots_b = (RISK_USD / 2) / (atr_b_i * cm_b)
            pos = {
                "dir": direction, "entry_i": i,
                "entry_price_a": a["close"].iloc[i], "entry_price_b": b["close"].iloc[i],
                "lots_a": lots_a, "lots_b": lots_b,
            }
            continue

        held = i - pos["entry_i"]
        exit_now = False
        if abs(zi) <= exit_z:
            exit_now = True
        elif abs(zi) >= stop_z:
            exit_now = True
        elif held >= max_hold_bars:
            exit_now = True

        if exit_now:
            px_a, px_b = a["close"].iloc[i], b["close"].iloc[i]
            d = pos["dir"]
            pnl_a = d * (px_a - pos["entry_price_a"]) * pos["lots_a"] * cm_a
            pnl_b = -d * (px_b - pos["entry_price_b"]) * pos["lots_b"] * cm_b
            costs = (cost_a * pos["lots_a"] * 2) + (cost_b * pos["lots_b"] * 2)  # open+close, both legs
            pnl = pnl_a + pnl_b - costs
            trades.append(pnl / RISK_USD)
            pos = None

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    n_tr = len(trades)
    return {
        "trades": n_tr,
        "wins": len(wins),
        "rs": trades,
        "win_rate": (len(wins) / n_tr) if n_tr else 0.0,
        "e_r": (sum(trades) / n_tr) if n_tr else 0.0,
        "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=100)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=3.5)
    ap.add_argument("--max-hold", type=int, default=48)
    args = ap.parse_args()

    print("=" * 96)
    print(f"PAIRS STAT-ARB -- {len(PAIRS)} correlated pairs, lookback={args.lookback} "
          f"entry_z={args.entry_z} exit_z={args.exit_z} stop_z={args.stop_z} max_hold={args.max_hold}")
    print("70/30 train/OOS split, real spread+commission on BOTH legs")
    print("=" * 96)

    summary = []
    for sym_a, sym_b in PAIRS:
        bars_a, bars_b = load(sym_a), load(sym_b)
        if bars_a is None or bars_b is None:
            print(f"  [skip] {sym_a}/{sym_b}: missing data")
            continue
        idx = bars_a.index.intersection(bars_b.index)
        split = int(len(idx) * 0.70)
        split_ts = idx[split]

        train_a, oos_a = bars_a[bars_a.index < split_ts], bars_a[bars_a.index >= split_ts]
        train_b, oos_b = bars_b[bars_b.index < split_ts], bars_b[bars_b.index >= split_ts]

        r_train = backtest_pair(sym_a, sym_b, train_a, train_b, args.lookback,
                                 args.entry_z, args.exit_z, args.stop_z, args.max_hold)
        r_oos = backtest_pair(sym_a, sym_b, oos_a, oos_b, args.lookback,
                               args.entry_z, args.exit_z, args.stop_z, args.max_hold)

        wf_eff = (r_oos["e_r"] / r_train["e_r"]) if r_train["e_r"] and r_train["e_r"] > 0 else float("nan")
        win_rs = [x for x in r_oos["rs"] if x > 0]
        loss_rs = [x for x in r_oos["rs"] if x <= 0]
        avg_win = (sum(win_rs) / len(win_rs)) if win_rs else 0.0
        avg_loss = (-sum(loss_rs) / len(loss_rs)) if loss_rs else 0.0
        rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        if rr > 0 and 0 < r_oos["win_rate"] < 1:
            p_pass, _, _, _ = mc_sim(0.005, r_oos["win_rate"], rr, tpd=4)
        else:
            p_pass = float("nan")

        pf_s = f"{r_oos['pf']:.2f}" if r_oos["pf"] != float("inf") else "inf"
        wf_s = f"{wf_eff:.2f}" if wf_eff == wf_eff else "n/a"
        pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
        print(f"{sym_a}/{sym_b:<10} train n={r_train['trades']:4d} E={r_train['e_r']:+.3f}R   "
              f"OOS n={r_oos['trades']:4d} wr={r_oos['win_rate']:.1%} E={r_oos['e_r']:+.3f}R "
              f"PF={pf_s} wf_eff={wf_s} MC_P(pass)={pp_s}", flush=True)
        summary.append((f"{sym_a}/{sym_b}", r_oos["trades"], r_oos["e_r"], r_oos["pf"], p_pass, r_oos["rs"]))

    print("\n" + "=" * 96)
    print("RANKED (by OOS E_net per trade)")
    print("=" * 96)
    for name, n, e_r, pf, pp, _ in sorted(summary, key=lambda x: x[2], reverse=True):
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        pp_s = f"{pp:.1%}" if pp == pp else "n/a"
        flag = "  <-- POSITIVE EDGE" if e_r > 0 else ""
        print(f"{name:<16}{n:>6}{e_r:>+10.3f}R{pf_s:>8}{pp_s:>12}{flag}")

    print("\n" + "=" * 96)
    print("POOLED HONEST AGGREGATE (all pre-selected pairs combined -- guards against cherry-picking)")
    print("=" * 96)
    all_rs = [x for _, _, _, _, _, rs in summary for x in rs]
    if all_rs:
        wins = [x for x in all_rs if x > 0]
        losses = [x for x in all_rs if x <= 0]
        pf_all = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
        e_all = sum(all_rs) / len(all_rs)
        wr_all = len(wins) / len(all_rs)
        print(f"  ALL {len(PAIRS)} pairs pooled: n={len(all_rs)}  win_rate={wr_all:.1%}  "
              f"E_net={e_all:+.3f}R  PF={pf_all:.2f}")
    jpy_names = {"EURJPY/CHFJPY", "GBPJPY/CHFJPY", "NZDJPY/AUDJPY", "EURJPY/NZDJPY", "NZDJPY/CHFJPY",
                 "AUDJPY/CHFJPY", "GBPJPY/NZDJPY", "EURJPY/AUDJPY", "GBPJPY/AUDJPY",
                 "USDJPY/CHFJPY", "USDJPY/AUDJPY", "USDJPY/NZDJPY"}
    jpy_rs = [x for name, _, _, _, _, rs in summary if name in jpy_names for x in rs]
    if jpy_rs:
        wins = [x for x in jpy_rs if x > 0]
        losses = [x for x in jpy_rs if x <= 0]
        pf_j = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
        e_j = sum(jpy_rs) / len(jpy_rs)
        wr_j = len(wins) / len(jpy_rs)
        print(f"  JPY-cross-ratio cluster pooled: n={len(jpy_rs)}  win_rate={wr_j:.1%}  "
              f"E_net={e_j:+.3f}R  PF={pf_j:.2f}")


if __name__ == "__main__":
    main()
