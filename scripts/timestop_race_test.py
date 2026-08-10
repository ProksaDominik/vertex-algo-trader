"""Does a FAST-RESOLVING roster beat a high-expectancy slow one in a 10-day race?

scripts/timebox_strategy_test.py found that RSI with a hard 1-2 day time stop
keeps positive expectancy in BOTH halves while cutting median hold from 18 days
to 1-2. That is a direct attack on the throughput bottleneck
(scripts/ftmo_race_backtest.py showed 2 concurrent positions with long holds
cycles only 4-6 trades/month, which cannot compound to +10%).

The trade is explicit:
    slow roster : +0.174R blended, 10-18 day holds
    fast roster : +0.054R to +0.077R, 1-2 day holds

Per trade the fast one earns a third as much; per unit of TIME it may earn far
more, and time is what a deadline measures. This settles it by replaying a real
challenge from every start date, marking equity to market daily from real
prices, with all three FTMO rules applied.

HONEST CAVEAT carried into the output: the fast roster's TRAIN expectancy is
+0.004R to +0.012R -- within noise of zero. Only the OOS half is clearly
positive. Costs are ~0.05R/trade before the 2-12x live slippage gap. Treat any
pass rate here as an optimistic bound, not a forecast.

Usage:
    python scripts/timestop_race_test.py [--workers 6]
"""
from __future__ import annotations

import argparse
import os
import sys

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

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]

# (label, class, params, tp, max_hold_days | None)
ROSTERS = {
    "SLOW (deployed): RSI@4R + Boll@4R, no time stop": [
        ("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 4.0, None),
        ("Boll20", BollingerBounceStrategy,
         {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0, None),
    ],
    "FAST: RSI@3R hold 2d": [
        ("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 2),
    ],
    "FAST: RSI@3R hold 1d": [
        ("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 1),
    ],
    "FAST MIX: RSI(14)@3R 2d + RSI(7)@3R 2d": [
        ("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 2),
        ("RSI7", RSIReversionStrategy,
         {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}, 3.0, 2),
    ],
    "HYBRID: fast RSI 2d + slow Boll@4R": [
        ("RSI14", RSIReversionStrategy,
         {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 2),
        ("Boll20", BollingerBounceStrategy,
         {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0, None),
    ],
}

import os as _os
# Phase 1 target 10%, Phase 2 (Verification) target 5%. The account RESETS to
# the starting balance between phases -- it is NOT a cumulative 15% on one
# balance, it is two separate races from a fresh $10,000.
TARGET = float(_os.environ.get("FTMO_TARGET", "0.10"))
DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZONS = (10, 20, 30, 60)


def _cell(job):
    """Trades for one (leg, symbol), each with a daily R path, time stop applied."""
    roster_key, li, sym = job
    label, cls, params, tp, max_hold = ROSTERS[roster_key][li]
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
        closes = bars["close"]
        out = []
        for t in bt.closed:
            o = pd.Timestamp(t.entered_at).normalize()
            c = pd.Timestamp(t.exited_at).normalize()
            risk_px = abs(t.entry_price - t.stop_loss)
            if risk_px <= 0 or c < o:
                continue
            seg = closes.loc[o:c]
            sign = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sign / risk_px
            final = t.pnl_usd / RISK_USD
            if len(path):
                path[-1] = final
            # Hard time stop: truncate at max_hold bars and mark to market
            # there. Costs still apply, so subtract the same round-trip drag
            # the engine charges (final - gross at exit) approximated by the
            # observed cost on the untruncated trade.
            if max_hold is not None and len(path) > max_hold + 1:
                cost_r = (path[-1] - (seg.values[-1] - t.entry_price)
                          * sign / risk_px)
                path = path[:max_hold + 1].copy()
                path[-1] = path[-1] + cost_r      # charge the round trip
            out.append({"open": o, "path": list(path), "final": float(path[-1])})
        return {"roster": roster_key, "trades": out}
    except Exception:
        return None


def replay(by_day, all_days, start_i, horizon, risk, max_conc, max_day):
    eq, realised, open_pos, tdays = 1.0, 0.0, [], 0
    for k in range(horizon):
        i = start_i + k
        if i >= len(all_days):
            break
        day0 = eq
        opened = 0
        for tr in by_day.get(all_days[i], []):
            if len(open_pos) >= max_conc or opened >= max_day:
                break
            open_pos.append([tr, 0])
            opened += 1
        if opened:
            tdays += 1
        floating, still = 0.0, []
        for tr, age in open_pos:
            age += 1
            if age < len(tr["path"]):
                floating += tr["path"][age] * risk
                still.append([tr, age])
            else:
                realised += tr["final"] * risk
        open_pos = still
        eq = 1.0 + realised + floating
        if eq <= day0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + TARGET and tdays >= MIN_DAYS:
            return "pass"
    return "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print("=" * 108)
    print("FAST (time-stopped) vs SLOW roster — historical FTMO replay, every start date")
    print("=" * 108, flush=True)

    jobs = [(k, i, s) for k in ROSTERS for i in range(len(ROSTERS[k]))
            for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if c:
                cells.append(c)

    combos = [(0.02, 6, 3), (0.03, 10, 4), (0.02, 12, 5), (0.015, 15, 6)]
    print(f"\n{'roster':<46}{'risk':>6}{'conc':>6}"
          + "".join(f"{str(h)+'d':>9}" for h in HORIZONS) + f"{'fail30':>9}")
    print("-" * 108)
    best = []
    for key in ROSTERS:
        trades = [t for c in cells if c["roster"] == key for t in c["trades"]]
        if not trades:
            continue
        by_day = {}
        for t in trades:
            by_day.setdefault(t["open"], []).append(t)
        all_days = sorted(by_day)
        starts = range(0, len(all_days) - 1, 4)
        for risk, conc, per_day in combos:
            res = {}
            for h in HORIZONS:
                out = [replay(by_day, all_days, i, h, risk, conc, per_day)
                       for i in starts]
                n = len(out)
                res[h] = (out.count("pass") / n, out.count("fail") / n)
            print(f"{key[:45]:<46}{risk*100:>5.1f}%{conc:>6}"
                  + "".join(f"{res[h][0]:>9.1%}" for h in HORIZONS)
                  + f"{res[30][1]:>9.1%}")
            best.append((res[10][0], res[30][0], key, risk, conc, per_day))
        print()

    print("=" * 108)
    for label, idx in (("BEST 10-DAY", 0), ("BEST 30-DAY", 1)):
        print(f"  {label}:")
        for r in sorted(best, key=lambda x: -x[idx])[:3]:
            print(f"    {r[2][:44]:<45} risk {r[3]*100:.1f}% conc {r[4]}"
                  f"  ->  10d {r[0]:.1%}   30d {r[1]:.1%}")
    print("\n  Real historical outcomes. The fast roster's TRAIN expectancy is")
    print("  +0.004R to +0.012R (near zero) -- only OOS is clearly positive, and")
    print("  live slippage (2-12x modelled) is not included. Optimistic bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
