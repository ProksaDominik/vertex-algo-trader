"""Race-test the two-stream book: FX mean-reversion + non-FX trend.

scripts/nonfx_trend_test.py established the pieces:
  * Trend-following works on indices/commodities/crypto and survives BOTH
    halves -- Donchian20@6R at +0.219R train / +0.340R OOS, the strongest
    in-sample number in this project (RSI daily is +0.044R train).
  * The two streams are essentially UNCORRELATED (+0.047 and -0.002).
  * A 50/50 blend cut volatility 35% (2.61 -> 1.69) while raising mu/sigma
    above either stream alone -- the first time in this project that cutting
    variance did not cut drift with it.

50/50 was arbitrary. This sweeps the allocation between streams, and race-tests
the result against the real FTMO rules from every historical start date.

Why the weighting matters: the streams have different volatilities, so equal
RISK is not equal VARIANCE contribution. A risk-parity-style tilt toward the
lower-volatility stream should beat 50/50 if the independence is real.

COST CAVEAT: non-FX spreads in PAIR_SPECS are ESTIMATES (several symbols
returned zero ticks when queried). Costs are the binding constraint in this
project, so treat these numbers as provisional pending verification against
real fills.

Usage:
    python scripts/two_stream_race.py [--workers 5]
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
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
      "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
      "AUDCHF", "EURAUD", "AUDCAD"]
# tradeable on the live account only (BRENT/COPPER excluded -- not offered)
NONFX = ["US30", "NAS100", "DE40", "JP225", "WTI", "XPTUSD", "BTCUSD",
         "ETHUSD", "XAUUSD", "US500"]

RSI14 = {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}
RSI7 = {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}

TREND_LEGS = [("Donch20_6R", DonchianBreakoutStrategy,
               {"donchian": {"period": 20}}, 6.0),
              ("MACD_4R", MACDTrendStrategy, {"macd": {}}, 4.0)]

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4


def _bt(cls, params, tp, sym, bars):
    strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return bt.closed


def _paths(closed, bars, keep=None):
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
    return out


def _cell(job):
    kind, sym = job
    try:
        bars = load(sym, "1d")
        if bars is None or len(bars) < 600:
            return None
        bars = bars[bars.index.dayofweek != 6]
        if kind == "fx":
            agree = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name)
                     for t in _bt(RSIReversionStrategy, RSI7, 1.0, sym, bars)}
            tr = _paths(_bt(RSIReversionStrategy, RSI14, 1.0, sym, bars),
                        bars, agree)
            return {"stream": "fx", "trades": tr}
        out = []
        for _, cls, params, tp in TREND_LEGS:
            out += _paths(_bt(cls, params, tp, sym, bars), bars)
        return {"stream": "trend", "trades": out}
    except Exception as e:
        return {"error": f"{kind}/{sym}: {e!r}"}


def replay(fx_day, tr_day, days, i0, hz, risk_fx, risk_tr, conc, per_day, target):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        opened = 0
        for src, day_map, rk in (("fx", fx_day, risk_fx), ("tr", tr_day, risk_tr)):
            for t in day_map.get(days[i], []):
                if len(op) >= conc or opened >= per_day:
                    break
                op.append([t, 0, rk])
                opened += 1
        if opened:
            td += 1
        fl, still = 0.0, []
        for t, age, rk in op:
            age += 1
            if age < len(t["path"]):
                fl += t["path"][age] * rk
                still.append([t, age, rk])
            else:
                real += t["final"] * rk
        op = still
        eq = 1.0 + real + fl
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + target and td >= MIN_DAYS:
            return "pass"
    return "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.10)
    args = ap.parse_args()

    print("=" * 100)
    print(f"TWO-STREAM RACE — FX mean-reversion + non-FX trend, target "
          f"{args.target:.0%}")
    print("=" * 100, flush=True)

    jobs = [("fx", s) for s in FX] + [("trend", s) for s in NONFX]
    fx_tr, tr_tr, errs = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            if not c:
                continue
            if "error" in c:
                errs.append(c["error"])
            elif c["stream"] == "fx":
                fx_tr += c["trades"]
            else:
                tr_tr += c["trades"]
    if errs:
        print(f"{len(errs)} cells failed; first: {errs[0]}")
    print(f"FX trades {len(fx_tr)} | trend trades {len(tr_tr)}\n")

    fx_day, tr_day = {}, {}
    for t in fx_tr:
        fx_day.setdefault(t["open"], []).append(t)
    for t in tr_tr:
        tr_day.setdefault(t["open"], []).append(t)
    days = sorted(set(fx_day) | set(tr_day))
    starts = range(0, len(days) - 1, 4)

    print(f"{'fx risk':>9}{'trend risk':>12}{'conc':>6}{'10d':>9}{'20d':>9}"
          f"{'30d':>9}{'fail30':>9}")
    print("-" * 100)
    best = []
    for rfx, rtr in ((0.04, 0.00), (0.00, 0.04), (0.03, 0.01), (0.02, 0.02),
                     (0.01, 0.03), (0.03, 0.03), (0.02, 0.04), (0.04, 0.02),
                     (0.04, 0.04), (0.05, 0.03), (0.03, 0.05)):
        for conc, per_day in ((15, 8), (20, 10)):
            res = {}
            for hz in (10, 20, 30):
                out = [replay(fx_day, tr_day, days, i, hz, rfx, rtr, conc,
                              per_day, args.target) for i in starts]
                n = len(out)
                res[hz] = (out.count("pass") / n, out.count("fail") / n)
            print(f"{rfx*100:>8.1f}%{rtr*100:>11.1f}%{conc:>6}"
                  + "".join(f"{res[h][0]:>9.1%}" for h in (10, 20, 30))
                  + f"{res[30][1]:>9.1%}")
            best.append((res[10][0], res[30][0], rfx, rtr, conc))
        print()

    print("=" * 100)
    for lbl, idx in (("BEST 10-DAY", 0), ("BEST 30-DAY", 1)):
        print(f"  {lbl}:")
        for r in sorted(best, key=lambda x: -x[idx])[:3]:
            print(f"    fx {r[2]*100:.1f}% + trend {r[3]*100:.1f}%  conc {r[4]}"
                  f"  ->  10d {r[0]:.1%}   30d {r[1]:.1%}")
    print("\n  Compare against FX-only best: 39.1% (10d) / 42.9% (30d).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
