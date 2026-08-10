"""What the DEPLOYED YOLO book actually produces, week by week.

Earlier YOLO figures in this project were FX-only. The live book now runs two
uncorrelated streams:
    FX mean-reversion @4R  -- RSI(14) + Bollinger(20,2.0), 17 pairs
    NON-FX trend @6R       -- Donchian(20) on indices/oil/platinum/BTC/ETH

Measured by replaying every possible start week across ~10.5 years, marking
equity to market daily from real prices. NOT Monte Carlo: an MC that credits
each trade's full R at entry overstated a comparable figure ~30x earlier in
this project, because the median hold is days, not hours.

Reports the weekly DISTRIBUTION, not just an average, because the median and
the mean diverge sharply here -- a 4-6R book makes its money in rare large
weeks, so the mean flatters what a typical week looks like.

COST CAVEAT: non-FX spreads in PAIR_SPECS are ESTIMATES. Costs are the binding
constraint in this project, so the Donchian contribution is provisional until
verified against live fills.

Usage:
    python scripts/yolo_deployed_stats.py [--workers 5]
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
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
      "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
      "AUDCHF", "EURAUD", "AUDCAD"]
NONFX = ["US30", "NAS100", "DE40", "JP225", "WTI", "XPTUSD", "BTCUSD",
         "ETHUSD", "XAUUSD", "US500"]

# exactly the deployed legs
LEGS = [
    ("RSI_Daily", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 4.0, FX),
    ("Bollinger_Daily", BollingerBounceStrategy,
     {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0, FX),
    ("Donchian_Trend", DonchianBreakoutStrategy,
     {"donchian": {"period": 20}}, 6.0, NONFX),
]
START, RISK, CONC, PER_DAY, CCY_CAP = 200.0, 0.05, 6, 8, 2


def _cell(job):
    li, sym = job
    name, cls, params, tp, _ = LEGS[li]
    try:
        bars = load(sym, "1d")
        if bars is None or len(bars) < 400:
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
        cl = bars["close"]
        out = []
        for t in bt.closed:
            o = pd.Timestamp(t.entered_at).normalize()
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"leg": name, "sym": sym, "open": o,
                        "path": list(path), "final": float(path[-1])})
        return out
    except Exception as e:
        return {"error": f"{name}/{sym}: {e!r}"}


def week(by_day, days, i0, horizon):
    eq, real, op = START, 0.0, []
    for j in range(horizon):
        i = i0 + j
        if i >= len(days):
            break
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= CONC or opened >= PER_DAY:
                break
            s = tr["sym"]
            if len(s) == 6 and s.isalpha():
                cc = {s[:3], s[3:]}
                held = sum(1 for p, _ in op
                           if len(p["sym"]) == 6 and p["sym"].isalpha()
                           and cc & {p["sym"][:3], p["sym"][3:]})
                if held >= CCY_CAP:
                    continue
            op.append([tr, 0])
            opened += 1
        fl, still = 0.0, []
        for tr, age in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * RISK * START
                still.append([tr, age])
            else:
                real += tr["final"] * RISK * START
        op = still
        eq = max(START + real + fl, 0.0)
    return eq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    jobs = [(i, s) for i, (_, _, _, _, syms) in enumerate(LEGS) for s in syms]
    trades, errs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            if isinstance(c, list):
                trades += c
            elif c:
                errs.append(c["error"])
    if errs:
        print(f"{len(errs)} cells failed; first: {errs[0]}\n")

    by_leg = {}
    for t in trades:
        by_leg.setdefault(t["leg"], []).append(t["final"])
    print("=" * 92)
    print(f"DEPLOYED YOLO BOOK — ${START:.0f}, {RISK:.0%} risk, {CONC} concurrent")
    print("=" * 92)
    print(f"\n{'leg':<18}{'trades':>8}{'win%':>8}{'E_net':>9}{'total R':>10}")
    for name, rs in by_leg.items():
        a = np.array(rs)
        print(f"{name:<18}{len(a):>8}{(a > 0).mean()*100:>8.1f}"
              f"{a.mean():>+9.3f}{a.sum():>+10.1f}")
    allr = np.array([t["final"] for t in trades])
    print(f"{'COMBINED':<18}{len(allr):>8}{(allr > 0).mean()*100:>8.1f}"
          f"{allr.mean():>+9.3f}{allr.sum():>+10.1f}")

    by_day = {}
    for t in trades:
        by_day.setdefault(t["open"], []).append(t)
    days = sorted(by_day)
    starts = list(range(0, len(days) - 6, 2))

    print(f"\n{len(days)} signal days | {len(starts)} start weeks replayed")
    print("\n" + "=" * 92)
    print("WEEKLY OUTCOME DISTRIBUTION (5 trading days)")
    print("=" * 92)
    ends = np.array([week(by_day, days, i, 5) for i in starts])
    ret = (ends - START) / START
    print(f"  median week      {np.median(ret)*100:>+8.2f}%   "
          f"= ${np.median(ends)-START:>+8.2f}")
    print(f"  mean week        {ret.mean()*100:>+8.2f}%   = ${ends.mean()-START:>+8.2f}")
    print(f"  P(up week)       {(ret > 0).mean():>8.1%}")
    print(f"  25th / 75th pct  {np.percentile(ret,25)*100:>+7.2f}% / "
          f"{np.percentile(ret,75)*100:>+.2f}%")
    print(f"  5th / 95th pct   {np.percentile(ret,5)*100:>+7.2f}% / "
          f"{np.percentile(ret,95)*100:>+.2f}%")
    print(f"  P(lose half)     {(ends <= START*0.5).mean():>8.1%}")
    print(f"  best / worst     {ret.max()*100:>+7.1f}% / {ret.min()*100:>+.1f}%")

    print("\n" + "=" * 92)
    print("LONGER HORIZONS")
    print("=" * 92)
    print(f"  {'horizon':>10}{'median':>11}{'mean':>11}{'P(up)':>9}{'P(lose half)':>14}")
    for hz, lbl in ((5, "1 week"), (10, "2 weeks"), (21, "1 month"), (63, "3 months")):
        e = np.array([week(by_day, days, i, hz) for i in starts])
        r = (e - START) / START
        print(f"  {lbl:>10}{np.median(r)*100:>+10.2f}%{r.mean()*100:>+10.2f}%"
              f"{(r > 0).mean():>9.1%}{(e <= START*0.5).mean():>14.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
