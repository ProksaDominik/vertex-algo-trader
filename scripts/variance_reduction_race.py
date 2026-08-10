"""Cut volatility at constant drift -- the lever the ceiling grid identified.

scripts/theoretical_ceiling.py mapped the 10-day pass rate against daily drift
(mu) and daily volatility (sigma), sweeping far past anything real:

    mu 1.0%/day, sigma 0.5%  ->  49.9%
    mu 1.0%/day, sigma 1.5%  ->  53.1%
    mu 2.0%/day, sigma 0.5%  ->  100%

The confluence roster already carries roughly mu ~ 1%/day at 4% risk, which
the grid says should yield ~50%. It measured 39.1%. The gap is sigma: 15
concurrent positions on a basket worth only ~4.9 INDEPENDENT bets
(scripts/correlation_audit.py) move together, so realised volatility is far
higher than the position count implies.

So this does not search for a new strategy. It measures, for each portfolio
construction, the REALISED daily mu and sigma of the equity curve, and finds
which construction sits closest to the grid's good region. Levers:

  * risk per position (down) x position count (up) -- same gross exposure,
    more diversification, IF the extra positions are genuinely independent
  * hard per-currency caps -- the correlation audit showed a single currency
    can carry 7 of 11 simultaneous signals
  * symbol universe size

Reports mu, sigma, mu/sigma and the actual pass rate side by side, so it is
visible whether a construction moved sigma without killing mu.

Usage:
    python scripts/variance_reduction_race.py [--workers 3]
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
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
           "AUDCHF", "EURAUD", "AUDCAD"]
RSI14 = {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}
RSI7 = {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}
TP = 1.0            # the confluence geometry that survived both halves
DAILY_LIM, TOTAL_LIM, MIN_DAYS, TARGET = 0.05, 0.10, 4, 0.10


def _run(params, sym, bars):
    strat = {"session": None, "exit": {"tp_r_multiple": TP, "sl_swing_lookback": 10}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(cls_of(params)(cfg)))
    bt.run()
    return bt.closed


def cls_of(_params):
    return RSIReversionStrategy


def _cell(sym):
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        base = _run(RSI14, sym, bars)
        agree = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name)
                 for t in _run(RSI7, sym, bars)}
        cl = bars["close"]
        out = []
        for t in base:
            o = pd.Timestamp(t.entered_at).normalize()
            if (o, t.direction.name) not in agree:
                continue
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": o, "sym": sym, "path": list(path),
                        "final": float(path[-1])})
        return out
    except Exception:
        return None


def replay(by_day, days, i0, hz, risk, conc, per_day, cap, want_path=False):
    eq, real, op, td = 1.0, 0.0, [], 0
    daily = []
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            if cap:
                cc = {tr["sym"][:3], tr["sym"][3:]}
                held = sum(1 for p, _ in op if cc & {p["sym"][:3], p["sym"][3:]})
                if held >= cap:
                    continue
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
        daily.append(eq - d0)
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return ("fail", daily) if want_path else "fail"
        if eq >= 1.0 + TARGET and td >= MIN_DAYS:
            return ("pass", daily) if want_path else "pass"
    return ("unresolved", daily) if want_path else "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    print("=" * 104)
    print("VARIANCE REDUCTION — measured mu/sigma vs actual pass rate")
    print("=" * 104, flush=True)

    trades = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, SYMBOLS):
            if c:
                trades += c
    by_day = {}
    for t in trades:
        by_day.setdefault(t["open"], []).append(t)
    days = sorted(by_day)
    starts = list(range(0, len(days) - 1, 4))
    print(f"{len(trades)} confluence trades, {len(days)} signal days\n")

    # (risk, conc, per_day, currency_cap) -- same gross exposure at different
    # granularity, so any change in sigma is a diversification effect.
    combos = [
        (0.040, 15, 8, 0), (0.040, 15, 8, 2), (0.040, 15, 8, 1),
        (0.030, 20, 8, 0), (0.030, 20, 8, 2), (0.030, 20, 8, 1),
        (0.020, 30, 10, 0), (0.020, 30, 10, 2), (0.020, 30, 10, 1),
        (0.015, 40, 12, 2), (0.010, 60, 15, 2),
        (0.050, 12, 6, 2), (0.060, 10, 5, 2),
    ]
    print(f"{'risk':>7}{'conc':>6}{'/day':>6}{'ccap':>6}{'mu/day':>9}"
          f"{'sigma':>9}{'mu/sig':>8}{'10d':>8}{'30d':>8}{'fail30':>9}")
    print("-" * 104)
    best = []
    for risk, conc, per_day, cap in combos:
        # measure realised daily mu/sigma over 30-day windows
        steps = []
        for i in starts[::3]:
            _, dl = replay(by_day, days, i, 30, risk, conc, per_day, cap, True)
            steps += dl
        mu = float(np.mean(steps)) if steps else 0.0
        sd = float(np.std(steps)) if steps else 0.0
        o10 = [replay(by_day, days, i, 10, risk, conc, per_day, cap) for i in starts]
        o30 = [replay(by_day, days, i, 30, risk, conc, per_day, cap) for i in starts]
        p10 = o10.count("pass") / len(o10)
        p30 = o30.count("pass") / len(o30)
        f30 = o30.count("fail") / len(o30)
        print(f"{risk*100:>6.1f}%{conc:>6}{per_day:>6}{(cap or '-'):>6}"
              f"{mu*100:>8.2f}%{sd*100:>8.2f}%{(mu/sd if sd else 0):>8.3f}"
              f"{p10:>8.1%}{p30:>8.1%}{f30:>9.1%}")
        best.append((p10, p30, risk, conc, per_day, cap, mu, sd))

    print("\n" + "=" * 104)
    for lbl, idx in (("BEST 10-DAY", 0), ("BEST 30-DAY", 1)):
        print(f"  {lbl}:")
        for r in sorted(best, key=lambda x: -x[idx])[:3]:
            print(f"    risk {r[2]*100:.1f}% conc {r[3]} ccap {r[5] or '-'}"
                  f"  mu {r[6]*100:+.2f}%/d  sigma {r[7]*100:.2f}%"
                  f"  ->  10d {r[0]:.1%}  30d {r[1]:.1%}")
    print("\n  The ceiling grid says mu 1%/day at sigma 0.5-1.5% is worth 50-56%")
    print("  at 10 days. Compare the mu/sigma columns above against that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
