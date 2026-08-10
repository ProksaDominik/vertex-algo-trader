"""YOLO daily profit-target + drawdown-pause overlay, measured on real history.

The user's instinct: "if we get to ~$150, stop there and start another day".
That half is mechanical and testable -- it is the same finish-line logic that
made the FTMO withdraw-and-reset plan work (a target converts an open-ended
compounding problem into a series of races, and races suit a high-variance
edge).

The other half -- "start another day that looks promising" -- is NOT
implementable. Predicting which day will pay has now failed four separate
tests in this project: day-of-week (flipped sign OOS), stop-after-a-loss (was
lookahead), prior-day momentum (unobservable in real time), volatility regime
(bootstrap CI straddled zero). So the resume rule here is mechanical: resume
the next session, or after a fixed cooldown. Nothing predictive.

What this measures, by replaying every day of ~10.5 years with equity marked
to market daily from real prices:
  * daily profit target: stop opening once the day is +$T
  * daily loss stop:     stop opening once the day is -$L
  * cooldown:            after a loss stop, skip N days entirely
against the baseline of trading every signal.

Usage:
    python scripts/yolo_daily_target_test.py [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
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

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
           "AUDCHF", "EURAUD", "AUDCAD"]
LEGS = [
    ("RSI14", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 4.0),
    ("Boll20", BollingerBounceStrategy,
     {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0),
    ("RSI7", RSIReversionStrategy,
     {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}, 4.0),
]
START = 200.0


def _cell(job):
    li, sym = job
    _, cls, params, tp = LEGS[li]
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
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = closes.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": o, "path": list(path), "final": float(path[-1])})
        return out
    except Exception:
        return None


def run_window(by_day, days, i0, horizon, risk, conc, target, loss_stop,
               cooldown):
    """Returns (end_equity, days_target_hit, days_traded)."""
    eq, realised, open_pos = START, 0.0, []
    cool, hits, traded = 0, 0, 0
    for k in range(horizon):
        i = i0 + k
        if i >= len(days):
            break
        eq_day_start = eq

        # Mark the book to market FIRST, so the target/stop decision is made
        # against the day's P&L as it actually stands -- the previous version
        # compared eq against itself before any update, so day_pnl was always
        # 0 and the overlay silently never fired.
        floating, still = 0.0, []
        for tr, age in open_pos:
            age += 1
            if age < len(tr["path"]):
                floating += tr["path"][age] * risk * START
                still.append([tr, age])
            else:
                realised += tr["final"] * risk * START
        open_pos = still
        eq = max(START + realised + floating, 0.0)
        day_pnl = eq - eq_day_start

        if target and day_pnl >= target:
            hits += 1

        opened = 0
        if cool > 0:
            cool -= 1
        elif not (target and day_pnl >= target) and \
                not (loss_stop and day_pnl <= -loss_stop):
            for tr in by_day.get(days[i], []):
                if len(open_pos) >= conc or opened >= 6:
                    break
                open_pos.append([tr, 0])
                opened += 1
        elif loss_stop and day_pnl <= -loss_stop:
            cool = cooldown
        if opened:
            traded += 1
    return eq, hits, traded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = [(i, s) for i in range(len(LEGS)) for s in SYMBOLS]
    trades = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if c:
                trades += c
    by_day = {}
    for t in trades:
        by_day.setdefault(t["open"], []).append(t)
    days = sorted(by_day)
    starts = list(range(0, len(days) - 6, 3))

    print("=" * 100)
    print(f"YOLO DAILY TARGET + PAUSE OVERLAY — ${START:.0f} book, one week "
          f"(5 trading days), {len(starts)} start weeks")
    print("=" * 100)
    print("  'target' = stop opening once the day is +$T. 'stop' = stop opening")
    print("  once the day is -$L, then sit out 'cool' days. Nothing predictive.\n")
    print(f"  {'risk':>6}{'target':>8}{'stop':>7}{'cool':>6}{'med wk':>9}"
          f"{'mean wk':>9}{'P(up wk)':>10}{'P(+$100 day)':>14}{'P(ruin)':>9}")
    print("-" * 100)

    rows = []
    for risk, target, loss_stop, cool in itertools.product(
            (0.05, 0.10, 0.20), (0, 50, 100, 150), (0, 40), (0, 2)):
        if loss_stop == 0 and cool:      # cooldown meaningless without a stop
            continue
        ends, hits = [], []
        for i in starts:
            e, h, _ = run_window(by_day, days, i, 5, risk, 6, target,
                                 loss_stop, cool)
            ends.append(e)
            hits.append(h)
        ends = np.array(ends)
        ret = (ends - START) / START
        rows.append((risk, target, loss_stop, cool, np.median(ret), ret.mean(),
                     (ret > 0).mean(), np.mean([h > 0 for h in hits]),
                     (ends <= START * 0.1).mean()))
        r = rows[-1]
        print(f"  {r[0]*100:>5.0f}%{('-' if not r[1] else '$'+str(r[1])):>8}"
              f"{('-' if not r[2] else '$'+str(r[2])):>7}{(r[3] or '-'):>6}"
              f"{r[4]*100:>8.2f}%{r[5]*100:>8.2f}%{r[6]:>10.1%}"
              f"{r[7]:>14.1%}{r[8]:>9.1%}")

    print("\n" + "=" * 100)
    print("BEST BY MEDIAN WEEKLY RETURN")
    for r in sorted(rows, key=lambda x: -x[4])[:5]:
        print(f"  risk {r[0]*100:.0f}%  target {('none' if not r[1] else '$'+str(r[1])):<6}"
              f"  stop {('none' if not r[2] else '$'+str(r[2])):<5}  cool {r[3] or '-'}"
              f"  ->  median {r[4]*100:+.2f}%/wk  mean {r[5]*100:+.2f}%  "
              f"P(up) {r[6]:.1%}")
    print("\n  P(+$100 day) is the share of weeks containing at least one day")
    print("  that made the target. A 50%-of-capital DAY is not in this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
