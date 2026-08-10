"""Does the high-win-rate confluence setup actually pass FTMO more often?

scripts/confluence_strategy.py found that requiring RSI(14) and RSI(7) to
agree on the same symbol/day/direction produces a 65.6% OOS win rate at a
0.75R target -- the win rate the pass-rate curve says is needed for a 60%+
10-day result.

But that curve was built by flipping losers into winners at 3-4R payoffs.
This setup wins often at only 0.75R, so its EXPECTANCY (+0.121R) is actually
LOWER than the deployed roster's (+0.174R). What differs is the SHAPE: many
small wins instead of rare large ones, i.e. a much smoother equity path.

That shape is exactly what a 5%-daily-loss limit rewards, because the binding
constraint on this challenge was never expectancy -- it was that any strategy
volatile enough to make +10% quickly also produces 5% down days. So this is
a real question, not a formality: does smoothness buy more than expectancy
costs?

Settled the only honest way -- replaying a real challenge from every start
date, marking equity to market daily from real prices, all three FTMO rules.

Usage:
    python scripts/confluence_race_test.py [--workers 6] [--target 0.10]
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

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

# Each roster: list of (label, cls, params, tp, confluence_partner|None)
RSI14 = (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}})
RSI7 = (RSIReversionStrategy, {"rsi": {"period": 7, "oversold": 25, "overbought": 75}})
BOLL = (BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}})

ROSTERS = {
    "DEPLOYED: RSI@3R(2d) + Boll@4R": [("RSI14", RSI14, 3.0, None),
                                       ("Boll20", BOLL, 4.0, None)],
    "CONFLUENCE 0.75R (65.6% win)": [("CONF", RSI14, 0.75, RSI7)],
    "CONFLUENCE 1.0R (57.9% win)": [("CONF", RSI14, 1.0, RSI7)],
    "CONF 0.75R + Boll@4R": [("CONF", RSI14, 0.75, RSI7), ("Boll20", BOLL, 4.0, None)],
    "CONF 1.0R + Boll@4R": [("CONF", RSI14, 1.0, RSI7), ("Boll20", BOLL, 4.0, None)],
}

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZONS = (10, 20, 30)


def _run(cls_params, tp, sym, bars):
    cls, params = cls_params
    strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return bt.closed


def _cell(job):
    rkey, li, sym = job
    label, cls_params, tp, partner = ROSTERS[rkey][li]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        closed = _run(cls_params, tp, sym, bars)
        keep = None
        if partner is not None:
            # confluence: keep only setups the partner also fired on, same
            # day and direction. Both legs are computed from bars closed
            # before entry, so no lookahead is introduced by the agreement.
            other = _run(partner, tp, sym, bars)
            keep = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name)
                    for t in other}
        cl = bars["close"]
        out = []
        for t in closed:
            o = pd.Timestamp(t.entered_at).normalize()
            if keep is not None and (o, t.direction.name) not in keep:
                continue
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": o, "path": list(path), "final": float(path[-1])})
        return {"roster": rkey, "trades": out}
    except Exception:
        return None


def replay(by_day, days, i0, hz, risk, conc, per_day, target):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            op.append([tr, 0])
            opened += 1
        if opened:
            td += 1
        fl, still = 0.0, []
        for tr, age in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * risk
                still.append([tr, age])
            else:
                real += tr["final"] * risk
        op = still
        eq = 1.0 + real + fl
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + target and td >= MIN_DAYS:
            return "pass"
    return "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.10)
    args = ap.parse_args()

    print("=" * 104)
    print(f"CONFLUENCE RACE — target {args.target:.0%}, real replay every start date")
    print("=" * 104, flush=True)

    jobs = [(k, i, s) for k in ROSTERS for i in range(len(ROSTERS[k]))
            for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if c:
                cells.append(c)

    combos = [(0.02, 10, 5), (0.03, 10, 5), (0.02, 15, 6), (0.03, 15, 6),
              (0.04, 15, 8), (0.05, 20, 8)]
    print(f"\n{'roster':<34}{'risk':>6}{'conc':>6}"
          + "".join(f"{str(h)+'d':>9}" for h in HORIZONS) + f"{'fail30':>9}{'n':>8}")
    print("-" * 104)
    best = []
    for key in ROSTERS:
        trades = [t for c in cells if c["roster"] == key for t in c["trades"]]
        if not trades:
            continue
        by_day = {}
        for t in trades:
            by_day.setdefault(t["open"], []).append(t)
        days = sorted(by_day)
        starts = range(0, len(days) - 1, 4)
        for risk, conc, per_day in combos:
            res = {}
            for h in HORIZONS:
                out = [replay(by_day, days, i, h, risk, conc, per_day, args.target)
                       for i in starts]
                n = len(out)
                res[h] = (out.count("pass") / n, out.count("fail") / n)
            print(f"{key[:33]:<34}{risk*100:>5.1f}%{conc:>6}"
                  + "".join(f"{res[h][0]:>9.1%}" for h in HORIZONS)
                  + f"{res[30][1]:>9.1%}{len(trades):>8}")
            best.append((res[10][0], res[30][0], key, risk, conc, per_day))
        print()

    print("=" * 104)
    for lbl, idx in (("BEST 10-DAY", 0), ("BEST 30-DAY", 1)):
        print(f"  {lbl}:")
        for r in sorted(best, key=lambda x: -x[idx])[:3]:
            print(f"    {r[2][:36]:<37} risk {r[3]*100:.1f}% conc {r[4]}"
                  f"  ->  10d {r[0]:.1%}   30d {r[1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
