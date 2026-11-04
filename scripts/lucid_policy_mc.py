"""Sizing POLICIES for the ORB stack, through LucidPro 50K Monte Carlo.

The funded account reaches its first payout only ~50% of the time at a fixed
size -- the weakest link. This compares ways of sizing day by day, using the
account state at the START of each day (known before trading):

  fixed-S       S x base stack every day
  cushion       half size while equity is < $1,000 above the MLL floor
  taper         size proportional to (equity - floor) / $2,000, 25%..100%
  lock-boost    base size until the MLL locks ($52,100 EOD), then 1.5x
  daily-stop    fixed, but the day's loss is capped at $600 (personal limit)

Base stack = 2 MNQ + 1 MGC ("unit"); multipliers scale it (fractional sizes
are an approximation of rounding to whole contracts). Same bootstrap and
pessimistic intraday reading as lucid_montecarlo.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_montecarlo as mc  # noqa: E402

UNIT = {"MNQ": 2, "MGC": 1}


def run_policy(P, L, mode, policy, horizon=120):
    n = P.shape[0]
    equity = np.full(n, mc.ANCHOR)
    peak = np.full(n, mc.ANCHOR)
    locked = np.zeros(n, bool)
    done = np.zeros(n, bool)
    outcome = np.zeros(n, np.int8)
    day_done = np.full(n, horizon)
    target = mc.EVAL_TARGET if mode == "eval" else mc.PAYOUT_LEVEL
    for d in range(horizon):
        live = ~done
        floor = np.where(locked, mc.LOCK_FLOOR, peak - mc.MLL)
        mult, day_cap = policy(equity, floor, peak, locked)
        pnl = np.maximum(P[:, d] * mult, day_cap)
        low = np.maximum(L[:, d] * mult, day_cap)
        fail = live & (equity + low <= floor)
        equity = np.where(live & ~fail, equity + pnl, equity)
        fail |= live & (equity <= floor)
        win = live & ~fail & (equity >= target)
        outcome[fail], outcome[win] = -1, 1
        day_done[fail | win] = d + 1
        done |= fail | win
        peak = np.where(done, peak, np.maximum(peak, equity))
        if mode == "funded":
            locked |= peak >= mc.LOCK_TRIGGER
        if done.all():
            break
    s = outcome == 1
    return s.mean(), (outcome == -1).mean(), float(np.median(day_done[s])) if s.any() else np.nan


def policies():
    cap = mc.DLL_CAP if mc.DLL_CAP is not None else -np.inf
    return {
        "fixed-1x (2 MNQ+1 MGC)": lambda e, f, p, lk: (1.0, cap),
        "fixed-2x (4 MNQ+2 MGC)": lambda e, f, p, lk: (2.0, cap),
        "fixed-3x (6 MNQ+3 MGC)": lambda e, f, p, lk: (3.0, cap),
        "cushion 1x->2x": lambda e, f, p, lk: (np.where(e - f < 1_000, 1.0, 2.0), cap),
        "taper 2x (25-100%)": lambda e, f, p, lk: (2.0 * np.clip((e - f) / 2_000, 0.25, 1.0), cap),
        "lock-boost 2x->3x": lambda e, f, p, lk: (np.where(lk, 3.0, 2.0), cap),
        "fixed-2x + $600 day stop": lambda e, f, p, lk: (2.0, -600.0),
        "cushion 1x->2x + $600 stop": lambda e, f, p, lk: (np.where(e - f < 1_000, 1.0, 2.0), -600.0),
    }


def main() -> int:
    trades = pd.read_parquet(ROOT / "data" / "futures" / "stack_orb_trend_trades.parquet")
    for label, start in (("2016-2026", "2016-01-01"), ("2021-2026", "2021-01-01")):
        t = trades[pd.to_datetime(trades["date"]) >= pd.Timestamp(start)]
        cal = pd.DatetimeIndex(pd.bdate_range(start, "2026-08-14"))
        pnl, low = mc.daily_series(t, UNIT, cal)
        pnl = np.maximum(pnl, -5_000.0)                  # undo the fixed cap; policies apply their own
        rng = np.random.default_rng(11)
        P, L = mc._paths(pnl, low, 20_000, 120, 5, rng)
        print(f"=== history {label}: unit stack earns ${pnl.mean() * 5:,.0f}/week at 1x ===")
        print(f"{'policy':<30} {'eval pass':>9} {'fail':>6} {'fails/pass':>10} {'days':>5} | "
              f"{'funded pay':>10} {'blow':>6} {'days':>5}")
        for name, pol in policies().items():
            ep, ef, ed = run_policy(P, L, "eval", pol)
            fp, fb, fd = run_policy(P, L, "funded", pol)
            print(f"{name:<30} {ep:>9.0%} {ef:>6.0%} {ef / max(ep, 1e-9):>10.2f} {ed:>5.0f} | "
                  f"{fp:>10.0%} {fb:>6.0%} {fd:>5.0f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
