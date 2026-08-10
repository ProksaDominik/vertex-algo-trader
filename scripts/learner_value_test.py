"""Would the self-improving layer actually improve the numbers over time?

The book ships three adaptive mechanisms:
  * SessionLearner   -- benches a (slot, session) pair once it has >= 6 trades
                        and net R <= -3
  * slot_health      -- quarantines a whole leg below a net-R floor over a
                        rolling window
  * auto_retune      -- shrinks risk on legs that bleed

All three are SELECTION ON PAST PERFORMANCE. This project has already measured
that idea once, in scripts/winner_picking_test.py: ranking symbols on the first
half of history and trading the winners in the second half found per-symbol
standard error (0.146R) as large as the spread across the whole basket
(0.156R), and 11 of 19 symbols FLIPPED sign between halves.

So the question is not whether the learner runs -- it is whether benching on a
small sample beats leaving the book alone. That is answerable directly: replay
history, apply the learner's exact rule as it would have fired at the time
(using only trades already closed), and compare against the unmodified book.

Two things are checked separately, because they fail differently:
  1. Does the rule improve expectancy on the trades it leaves behind?
  2. How long until it can fire at all, given real trade frequency?

Usage:
    python scripts/learner_value_test.py [--workers 5]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
           "AUDCHF", "EURAUD", "AUDCAD"]
LEGS = [("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 4.0),
        ("Boll20", BollingerBounceStrategy,
         {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0)]

# the shipped defaults
MIN_TRADES = 6
AVOID_BELOW = -3.0
SLOT_WINDOW = 12
SLOT_FLOOR = -6.0


def _cell(job):
    li, sym = job
    name, cls, params, tp = LEGS[li]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        strat = {"session": None,
                 "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym,
               "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                           "max_positions": 2, "daily_stop_loss_usd": -1e9},
               "strategy": strat}
        bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
        bt.run()
        out = []
        for t in bt.closed:
            o = pd.Timestamp(t.entered_at)
            out.append({"slot": f"{name}__{sym}", "sym": sym,
                        "open": o, "close": pd.Timestamp(t.exited_at),
                        "hour": o.hour, "r": t.pnl_usd / RISK_USD})
        return out
    except Exception as e:
        return {"error": repr(e)}


def session_of(h):
    if 0 <= h < 7:
        return "asian"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "ny_overlap"
    if 16 <= h < 21:
        return "ny_pm"
    return "late"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    jobs = [(i, s) for i in range(len(LEGS)) for s in SYMBOLS]
    trades = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            if isinstance(c, list):
                trades += c
    # CRITICAL: order by CLOSE time. The learner only knows a trade's outcome
    # once it has closed, so a rule keyed on entry order would be lookahead.
    trades.sort(key=lambda t: t["close"])
    print("=" * 96)
    print(f"LEARNER VALUE TEST — {len(trades)} trades, ordered by CLOSE time")
    print("=" * 96)

    # ---- 1. does the SessionLearner rule improve what it leaves behind? ----
    agg = defaultdict(lambda: [0, 0.0])       # (slot, session) -> [n, net_r]
    kept, benched = [], []
    for t in trades:
        key = (t["slot"], session_of(t["hour"]))
        n, net = agg[key]
        avoid = (n >= MIN_TRADES and net <= AVOID_BELOW)
        (benched if avoid else kept).append(t["r"])
        agg[key] = [n + 1, net + t["r"]]      # observed only after it closes

    def st(rs):
        if not rs:
            return "n=0"
        a = np.array(rs)
        return (f"n={len(a):<5} E={a.mean():+.3f}R  total={a.sum():+7.1f}R  "
                f"win={(a > 0).mean():.1%}")

    print("\n(1) SessionLearner rule (bench a slot+session after 6 trades at <= -3R)")
    print(f"    all trades, no learner : {st([t['r'] for t in trades])}")
    print(f"    trades the learner KEPT: {st(kept)}")
    print(f"    trades it BENCHED      : {st(benched)}")
    if benched and kept:
        base = np.mean([t["r"] for t in trades])
        print(f"\n    -> learner changed expectancy by "
              f"{np.mean(kept) - base:+.4f}R/trade")
        print(f"    -> it removed trades averaging {np.mean(benched):+.3f}R "
              f"({'good' if np.mean(benched) < base else 'BAD - it benched'} "
              f"{'' if np.mean(benched) < base else 'ABOVE-AVERAGE trades'})")

    # ---- 2. slot_health quarantine ----
    hist = defaultdict(list)
    kept2, benched2 = [], []
    for t in trades:
        h = hist[t["slot"]]
        window = h[-SLOT_WINDOW:]
        quarantined = (len(window) >= SLOT_WINDOW and sum(window) <= SLOT_FLOOR)
        (benched2 if quarantined else kept2).append(t["r"])
        h.append(t["r"])
    print("\n(2) slot_health quarantine (leg below -6R over its last 12 trades)")
    print(f"    kept    : {st(kept2)}")
    print(f"    benched : {st(benched2)}")
    if benched2:
        base = np.mean([t["r"] for t in trades])
        print(f"    -> changed expectancy by {np.mean(kept2) - base:+.4f}R/trade")

    # ---- 3. how long until it can fire at all? ----
    print("\n(3) TIME TO FIRE — how long before the rule has enough data")
    slots = {(t["slot"], session_of(t["hour"])) for t in trades}
    span_days = (trades[-1]["close"] - trades[0]["close"]).days
    per_day = len(trades) / max(span_days, 1)
    print(f"    {len(slots)} (slot, session) cells across {len(trades)} trades")
    print(f"    trade rate: {per_day:.2f}/day  ->  "
          f"{len(trades)/len(slots):.1f} trades per cell over "
          f"{span_days/365:.1f} years")
    reached = sum(1 for k, (n, _) in agg.items() if n >= MIN_TRADES)
    print(f"    cells that EVER reached {MIN_TRADES} trades: {reached}/{len(slots)}"
          f" ({reached/len(slots):.0%}) in {span_days/365:.1f} years")
    print(f"    -> a fresh book needs ~{MIN_TRADES*len(slots)/per_day/365:.1f} years"
          f" for every cell to become actionable")

    print("\n" + "=" * 96)
    print("The learner is selection on past performance. scripts/")
    print("winner_picking_test.py already showed per-symbol ranking does not")
    print("persist (11 of 19 symbols flipped sign between halves, SE 0.146R vs")
    print("a 0.156R spread). These numbers say whether the finer-grained")
    print("(slot, session) version fares any better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
