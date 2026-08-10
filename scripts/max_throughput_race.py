"""Push the proven lever: many fast legs + high concurrency.

Established so far (scripts/ftmo_race_backtest.py, timestop_race_test.py):
  * Throughput was the real bottleneck -- 2 concurrent with long holds cycles
    only 4-6 trades/month and cannot compound to +10%.
  * A hard 1-2 day time stop on RSI keeps positive expectancy in BOTH halves
    while cutting median hold 18d -> 1-2d.
  * Best so far: 32.3% at 10 days, 43.1% at 30 (Phase 1).

Untested and cheap: with 1-day holds a position frees its slot the next day,
so far MORE simultaneous positions are affordable than with 18-day holds. This
sweeps concurrency to 30 and stacks every time-stopped RSI/Bollinger variant
that passed the both-halves gate, plus the 3 extra symbols already on disk but
not in the deployed basket.

Everything is a real historical replay from every start date, equity marked to
market daily from real prices, all three FTMO rules enforced, >=4 trading days
required. No Monte Carlo -- that flaw already cost this project a 30x
overstatement.

Usage:
    python scripts/max_throughput_race.py [--workers 6] [--target 0.10]
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
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

# 17 deployed + any extra majors already downloaded. More symbols = more
# throughput, which is the lever that actually moved the number.
BASE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
        "AUDCHF", "EURAUD", "AUDCAD"]
EXTRA = [s for s in ("EURGBP", "GBPAUD", "EURNZD", "CADJPY", "AUDNZD")
         if s in PAIR_SPECS and (ROOT / "data" / f"{s.lower()}_15m_long.parquet").exists()]
SYMBOLS = BASE + EXTRA

# Every time-stopped variant that passed the both-halves gate in
# scripts/timebox_strategy_test.py, plus the slow Bollinger for diversity.
LEGS = [
    ("RSI14_3R_1d", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 1),
    ("RSI14_3R_2d", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 3.0, 2),
    ("RSI14_5R_1d", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 5.0, 1),
    ("RSI7_3R_2d", RSIReversionStrategy,
     {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}, 3.0, 2),
    ("Boll20_4R_slow", BollingerBounceStrategy,
     {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0, None),
]

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZONS = (10, 20, 30)


def _cell(job):
    li, sym = job
    label, cls, params, tp, max_hold = LEGS[li]
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
            if max_hold is not None and len(path) > max_hold + 1:
                cost_r = path[-1] - (seg.values[-1] - t.entry_price) * sign / risk_px
                path = path[:max_hold + 1].copy()
                path[-1] = path[-1] + cost_r        # still pay the round trip
            out.append({"leg": label, "open": o, "path": list(path),
                        "final": float(path[-1]), "sym": sym})
        return out
    except Exception:
        return None


def replay(by_day, all_days, start_i, horizon, risk, max_conc, max_day,
           target, cap_ccy=0):
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
            if cap_ccy:
                sym = tr["sym"]
                ccy = {sym[:3], sym[3:]}
                held = sum(1 for p, _ in open_pos
                           if ccy & {p["sym"][:3], p["sym"][3:]})
                if held >= cap_ccy:
                    continue
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
        if eq >= 1.0 + target and tdays >= MIN_DAYS:
            return "pass"
    return "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.10)
    args = ap.parse_args()

    print("=" * 104)
    print(f"MAX-THROUGHPUT RACE — target {args.target:.0%} | {len(SYMBOLS)} symbols "
          f"({len(EXTRA)} extra) | {len(LEGS)} legs")
    print("=" * 104, flush=True)

    jobs = [(i, s) for i in range(len(LEGS)) for s in SYMBOLS]
    trades = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if c:
                trades += c
    print(f"{len(trades)} trades pooled")

    by_day = {}
    for t in trades:
        by_day.setdefault(t["open"], []).append(t)
    all_days = sorted(by_day)
    starts = range(0, len(all_days) - 1, 4)
    print(f"{len(all_days)} signal days, {len(starts)} challenge start dates\n")

    combos = []
    for risk in (0.015, 0.02, 0.03):
        for conc in (10, 15, 20, 30):
            for per_day in (5, 8):
                combos.append((risk, conc, per_day, 0))
    # correlation guard on the widest ones
    for risk in (0.02, 0.03):
        for conc in (20, 30):
            combos.append((risk, conc, 8, 2))

    print(f"{'risk':>7}{'conc':>6}{'/day':>6}{'ccap':>6}"
          + "".join(f"{str(h)+'d':>9}" for h in HORIZONS) + f"{'fail30':>9}")
    print("-" * 104)
    best = []
    for risk, conc, per_day, cap in combos:
        res = {}
        for h in HORIZONS:
            out = [replay(by_day, all_days, i, h, risk, conc, per_day,
                          args.target, cap) for i in starts]
            n = len(out)
            res[h] = (out.count("pass") / n, out.count("fail") / n)
        print(f"{risk*100:>6.1f}%{conc:>6}{per_day:>6}{(cap or '-'):>6}"
              + "".join(f"{res[h][0]:>9.1%}" for h in HORIZONS)
              + f"{res[30][1]:>9.1%}")
        best.append((res[10][0], res[30][0], risk, conc, per_day, cap))

    print("\n" + "=" * 104)
    for label, idx in (("BEST 10-DAY", 0), ("BEST 30-DAY", 1)):
        print(f"  {label}:")
        for r in sorted(best, key=lambda x: -x[idx])[:3]:
            print(f"    risk {r[2]*100:.1f}%  conc {r[3]}  {r[4]}/day  "
                  f"ccap {r[5] or '-'}  ->  10d {r[0]:.1%}   30d {r[1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
