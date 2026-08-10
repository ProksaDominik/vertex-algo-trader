"""Pick the most profitable AGGRESSIVE YOLO roster, by measurement.

Goal: more trades/day and higher growth, without tipping into the region
where the median week goes negative (this book's payoff is skewed, so raising
risk past a point makes the TYPICAL week worse while the mean keeps climbing).

Rosters compared, all daily bars:
  A  deployed: RSI14@4R + Boll20@4R (FX) + Donchian20@6R (non-FX)
  B  A + RSI7@4R + Boll40@4R  -- the variant legs
  C  B + Donchian55@6R + MACD@4R on non-FX -- more trend breadth
  D  C at wider targets on the trend side

NOTE ON THE VARIANT LEGS: portfolio_runner._build_strategy_config used to drop
the `rsi` key entirely, so a leg configured RSI(7) was built with RSI(14)
defaults -- an exact duplicate. That is fixed, so RSI7 is genuinely a second
signal now. The offline measurement was always valid (params are passed
directly here); only the live book was affected.

Sweeps risk and concurrency for each roster and reports the weekly
distribution -- median AND mean, because they diverge sharply.

Usage:
    python scripts/yolo_roster_compare.py [--workers 5]
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
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
      "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
      "AUDCHF", "EURAUD", "AUDCAD"]
NONFX = ["US30", "NAS100", "DE40", "JP225", "WTI", "XPTUSD", "BTCUSD",
         "ETHUSD", "XAUUSD", "US500"]

LEG_DEFS = {
    "RSI14":  (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 4.0, FX),
    "RSI7":   (RSIReversionStrategy, {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}, 4.0, FX),
    "Boll20": (BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0, FX),
    "Boll40": (BollingerBounceStrategy, {"bollinger": {"period": 40, "std_dev": 1.5}}, 4.0, FX),
    "Donch20": (DonchianBreakoutStrategy, {"donchian": {"period": 20}}, 6.0, NONFX),
    "Donch55": (DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 6.0, NONFX),
    "MACDnf": (MACDTrendStrategy, {"macd": {}}, 4.0, NONFX),
}
ROSTERS = {
    "A deployed (3 legs)": ["RSI14", "Boll20", "Donch20"],
    "B +variants (5)":     ["RSI14", "RSI7", "Boll20", "Boll40", "Donch20"],
    "C +trend breadth (7)": ["RSI14", "RSI7", "Boll20", "Boll40", "Donch20",
                             "Donch55", "MACDnf"],
}
START = 200.0


def _cell(job):
    key, sym = job
    cls, params, tp, _ = LEG_DEFS[key]
    try:
        bars = load(sym, "1d")
        if bars is None or len(bars) < 400:
            return None
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        strat = {"session": None,
                 "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym,
               "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                           "max_positions": 2, "daily_stop_loss_usd": -1e9},
               "strategy": strat}
        bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
        bt.run()
        cutoff = bars.index[split]
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
            out.append({"leg": key, "sym": sym, "open": o, "path": list(path),
                        "final": float(path[-1]),
                        "half": "train" if o < cutoff else "oos"})
        return out
    except Exception as e:
        return {"error": f"{key}/{sym}: {e!r}"}


def run_window(by_day, days, i0, hz, risk, conc, per_day, ccy_cap=2):
    eq, real, op = START, 0.0, []
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            s = tr["sym"]
            if ccy_cap and len(s) == 6 and s.isalpha():
                cc = {s[:3], s[3:]}
                if sum(1 for p, _ in op if len(p["sym"]) == 6 and p["sym"].isalpha()
                       and cc & {p["sym"][:3], p["sym"][3:]}) >= ccy_cap:
                    continue
            op.append([tr, 0])
            opened += 1
        fl, still = 0.0, []
        for tr, age in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * risk * START
                still.append([tr, age])
            else:
                real += tr["final"] * risk * START
        op = still
        eq = max(START + real + fl, 0.0)
    return eq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    jobs = [(k, s) for k, (_, _, _, syms) in LEG_DEFS.items() for s in syms]
    trades, errs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            if isinstance(c, list):
                trades += c
            elif c:
                errs.append(c["error"])
    if errs:
        print(f"{len(errs)} cells failed; first: {errs[0]}\n")

    print("=" * 100)
    print("LEG QUALITY (both halves must be positive to deserve a place)")
    print("=" * 100)
    print(f"{'leg':<10}{'n':>7}{'/day':>7}{'win%':>7}{'E_train':>10}{'E_oos':>10}")
    span = (max(t["open"] for t in trades) - min(t["open"] for t in trades)).days
    for k in LEG_DEFS:
        rs = [t for t in trades if t["leg"] == k]
        if not rs:
            continue
        tr = [t["final"] for t in rs if t["half"] == "train"]
        oo = [t["final"] for t in rs if t["half"] == "oos"]
        if not tr or not oo:
            continue
        ok = np.mean(tr) > 0 and np.mean(oo) > 0
        print(f"{k:<10}{len(rs):>7}{len(rs)/(span*5/7):>7.2f}"
              f"{np.mean([t['final'] > 0 for t in rs])*100:>7.1f}"
              f"{np.mean(tr):>+10.3f}{np.mean(oo):>+10.3f}   {'KEEP' if ok else 'reject'}")

    print("\n" + "=" * 100)
    print("ROSTER x RISK — weekly distribution on a $200 book")
    print("=" * 100)
    print(f"{'roster':<22}{'risk':>6}{'conc':>6}{'/day':>7}{'med wk':>9}"
          f"{'mean wk':>9}{'P(up)':>8}{'med mo':>9}{'P(ruin)':>9}")
    print("-" * 100)
    best = []
    for rname, keys in ROSTERS.items():
        sel = [t for t in trades if t["leg"] in keys]
        by_day = {}
        for t in sel:
            by_day.setdefault(t["open"], []).append(t)
        days = sorted(by_day)
        starts = list(range(0, len(days) - 6, 3))
        per_day_rate = len(sel) / (span * 5 / 7)
        for risk, conc, pday in ((0.05, 6, 8), (0.08, 8, 10), (0.10, 10, 12),
                                 (0.15, 10, 12)):
            wk = np.array([run_window(by_day, days, i, 5, risk, conc, pday)
                           for i in starts])
            mo = np.array([run_window(by_day, days, i, 21, risk, conc, pday)
                           for i in starts])
            rw, rm = (wk - START) / START, (mo - START) / START
            print(f"{rname:<22}{risk*100:>5.0f}%{conc:>6}{per_day_rate:>7.2f}"
                  f"{np.median(rw)*100:>+8.2f}%{rw.mean()*100:>+8.2f}%"
                  f"{(rw > 0).mean():>8.1%}{np.median(rm)*100:>+8.2f}%"
                  f"{(mo <= START*0.1).mean():>9.1%}")
            best.append((np.median(rm), np.median(rw), rname, risk, conc, pday,
                         per_day_rate))
        print()

    print("=" * 100)
    print("BEST BY MEDIAN MONTH (the typical outcome, not the average):")
    for r in sorted(best, key=lambda x: -x[0])[:5]:
        print(f"  {r[2]:<22} risk {r[3]*100:.0f}%  conc {r[4]}  "
              f"{r[6]:.2f} trades/day  ->  median month {r[0]*100:+.2f}%  "
              f"median week {r[1]*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
