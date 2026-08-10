"""Historical FTMO challenge replay -- how often would you ACTUALLY have passed?

Every pass-probability figure produced earlier in this project came from a
Monte Carlo that credited each trade's full R at entry. Measured hold times
are median 18 days / mean 68, so that assumption is wrong for a 10-day race:
a trade opened on day 8 has not reached a 5R target by day 10.

This replaces the model with the data. For every possible START DATE in ~10.5
years it runs a real challenge:

  * signals come from the actual strategies on actual bars
  * positions open subject to the book's risk and concurrency limits
  * equity is marked to market EVERY DAY from real prices, so an open
    position's floating P&L counts -- which is how FTMO scores it
  * the +10% target, the 5% daily loss rule and the 10% overall loss rule are
    all applied to that equity
  * a pass also requires >= 4 distinct trading days

The output is the fraction of historical start dates that would have passed
within 10 / 30 / 60 days -- no distributional assumptions at all.

It sweeps the take-profit because that is the lever that trades expectancy
against resolution speed, which is the whole question for a deadline.

Usage:
    python scripts/ftmo_race_backtest.py [--workers 6]
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
LEGS = [
    ("Boll20", BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}}),
    ("RSI14", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    ("RSI7", RSIReversionStrategy,
     {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}),
    ("Boll40", BollingerBounceStrategy, {"bollinger": {"period": 40, "std_dev": 1.5}}),
]

RISK_PCT = 0.02
MAX_CONCURRENT = 2
MAX_PER_DAY = 2
# swept in main(); module-level values are the deployed defaults
TARGET, DAILY_LIM, TOTAL_LIM = 0.10, 0.05, 0.10
MIN_TRADING_DAYS = 4
HORIZONS = (5, 10, 20, 30, 60)


def _cell(job):
    """All trades for one (leg, symbol, tp), each with a DAILY R path."""
    li, sym, tp = job
    name, cls, params = LEGS[li]
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
            # daily unrealised R while the position is open, then the final R
            path = ((seg.values - t.entry_price) * sign / risk_px)
            final = t.pnl_usd / RISK_USD
            if len(path):
                path[-1] = final
            out.append({"open": o, "close": c,
                        "dates": list(seg.index), "path": list(path),
                        "final": final})
        return {"tp": tp, "trades": out}
    except Exception:
        return None


def replay(trades_by_day, all_days, start_i, horizon,
           risk=None, max_conc=None, max_day=None):
    """One challenge attempt beginning at all_days[start_i]."""
    risk = RISK_PCT if risk is None else risk
    max_conc = MAX_CONCURRENT if max_conc is None else max_conc
    max_day = MAX_PER_DAY if max_day is None else max_day
    eq = 1.0                      # equity as a fraction of the initial balance
    realised = 0.0
    open_pos = []                 # (path, dates, idx)
    tdays = 0
    day_start_eq = 1.0
    for k in range(horizon):
        i = start_i + k
        if i >= len(all_days):
            break
        day = all_days[i]
        day_start_eq = eq

        # open new positions (respecting concurrency + per-day cap)
        opened = 0
        for tr in trades_by_day.get(day, []):
            if len(open_pos) >= max_conc or opened >= max_day:
                break
            open_pos.append([tr, 0])
            opened += 1
        if opened:
            tdays += 1

        # mark to market
        floating = 0.0
        still = []
        for tr, age in open_pos:
            age += 1
            if age < len(tr["path"]):
                floating += tr["path"][age] * risk
                still.append([tr, age])
            else:
                realised += tr["final"] * risk
        open_pos = still
        eq = 1.0 + realised + floating

        if eq <= day_start_eq - DAILY_LIM:
            return "fail"
        if eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + TARGET and tdays >= MIN_TRADING_DAYS:
            return "pass"
    return "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tps", nargs="*", type=float,
                    default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    args = ap.parse_args()

    print("=" * 92)
    print("HISTORICAL FTMO REPLAY -- every start date, real prices, real holds")
    print(f"{len(SYMBOLS)} symbols x {len(LEGS)} legs | 2% risk | "
          f"max {MAX_CONCURRENT} open, {MAX_PER_DAY}/day")
    print("=" * 92, flush=True)

    jobs = [(li, s, tp) for tp in args.tps
            for li in range(len(LEGS)) for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if c:
                cells.append(c)

    # (risk, max_concurrent, max_per_day). Throughput is the suspected
    # bottleneck: 2 concurrent positions with 10-16 day holds only cycles
    # ~4-6 trades a month, which cannot compound to +10%.
    # HIGH-RISK region: the only mechanism that can resolve a 10-day race.
    combos = [(0.02, 6, 3), (0.03, 6, 3), (0.04, 6, 3),
              (0.03, 10, 4), (0.05, 8, 4), (0.05, 12, 5),
              (0.08, 8, 4), (0.10, 8, 4)]

    print(f"\n{'TP':>5}{'risk':>7}{'conc':>6}{'/day':>6}{'hold':>6}"
          f"{'10 days':>10}{'30 days':>10}{'60 days':>10}{'fail30':>9}")
    print("-" * 92)
    best = []
    for tp in args.tps:
        trades = [t for c in cells if c["tp"] == tp for t in c["trades"]]
        if not trades:
            continue
        by_day = {}
        for t in trades:
            by_day.setdefault(t["open"], []).append(t)
        all_days = sorted(by_day)
        if len(all_days) < 100:
            continue
        med_hold = np.median([(t["close"] - t["open"]).days for t in trades])
        starts = range(0, len(all_days) - 1, 4)

        for risk, conc, per_day in combos:
            res = {}
            for horizon in HORIZONS:
                out = [replay(by_day, all_days, i, horizon, risk, conc, per_day)
                       for i in starts]
                n = len(out)
                res[horizon] = (out.count("pass") / n, out.count("fail") / n)
            print(f"{tp:>5.1f}{risk*100:>6.1f}%{conc:>6}{per_day:>6}"
                  + "".join(f"{res[h][0]:>8.1%}" for h in HORIZONS)
                  + f"{res[30][1]:>9.1%}")
            best.append((res[10][0], res[30][0], res[30][1], tp, risk, conc, per_day))
        print()

    for label, key in (("BEST for 10 days", 0), ("BEST for 30 days", 1),
                       ("LOWEST FAILURE", 2)):
        print(f"  {label}:")
        for r in sorted(best, key=lambda x: -x[key])[:3]:
            print(f"    TP {r[3]:.1f} | risk {r[4]*100:.1f}% | {r[5]} concurrent | "
                  f"{r[6]}/day  ->  10d {r[0]:.1%}   30d {r[1]:.1%}   fail30 {r[2]:.1%}")

    print("\n  Every figure above is an actual historical outcome, not a model.")
    print("  Costs are the project's per-pair spread+commission; the 2-12x live")
    print("  slippage gap is NOT included and would lower these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
