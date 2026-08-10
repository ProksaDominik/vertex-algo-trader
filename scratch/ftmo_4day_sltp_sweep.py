"""SL distance / TP multiple grid for the CONFLUENCE (RSI14+RSI7) winner.

scratch/ftmo_4day_search.py established CONFLUENCE 1.0R (RSI14+RSI7) as the
best raw-pass 4-day roster, always run at the textbook sl_swing_lookback=10
and a single shared TP (0.75R or 1.0R). This script asks the question that
sweep never asked: does tuning stop distance and target change the 4-day
pass/fail picture, holding strategy and timeframe fixed?

sl_swing_lookback controls how far back the stop-placement swing search
looks (bars) -- a SMALLER lookback -> tighter stop -> smaller price move
needed to get stopped out (more whipsaws, but also faster resolution,
which matters directly for a 4-day deadline: a trade still open when the
challenge horizon ends is marked to floating equity, not realized).
A LARGER lookback -> wider stop -> fewer whipsaws, slower resolution.

Same real-replay engine as ftmo_4day_search.py: real daily bars, equity
marked to market day by day, all three FTMO rules, MIN_DAYS=4, walking
every possible start date. No Monte Carlo.

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_sltp_sweep.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(r"C:\Users\proksa\OneDrive\Počítač\New folder\ruflo\ruflo\metatraderbot")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

BASE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
        "AUDCHF", "EURAUD", "AUDCAD"]
EXTRA = [s for s in ("EURGBP", "GBPAUD", "EURNZD", "CADJPY", "AUDNZD")
         if s in PAIR_SPECS and (ROOT / "data" / f"{s.lower()}_15m_long.parquet").exists()]
SYMBOLS = BASE + EXTRA

RSI14 = {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}
RSI7 = {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}

SL_GRID = [5, 10, 20]          # bars, swing-lookback for the SL
TP_GRID = [0.75, 1.0, 1.5]      # R multiple, shared by both confluence legs

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZON = 4


def _run(params, tp, sl_lookback, sym, bars):
    strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": sl_lookback}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(RSIReversionStrategy(cfg)))
    bt.run()
    return bt.closed


def _cell(job):
    sl_lookback, tp, sym = job
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        closed = _run(RSI14, tp, sl_lookback, sym, bars)
        other = _run(RSI7, tp, sl_lookback, sym, bars)
        keep = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name) for t in other}
        cl = bars["close"]
        out = []
        for t in closed:
            o = pd.Timestamp(t.entered_at).normalize()
            if (o, t.direction.name) not in keep:
                continue
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": o, "path": list(path), "final": float(path[-1]), "sym": sym})
        return {"combo": (sl_lookback, tp), "trades": out}
    except Exception:
        return None


def replay(by_day, days, i0, hz, risk, conc, per_day, target, cap_ccy=0):
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
            if cap_ccy:
                sym = tr["sym"]
                ccy = {sym[:3], sym[3:]}
                held = sum(1 for p, _ in op if ccy & {p["sym"][:3], p["sym"][3:]})
                if held >= cap_ccy:
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
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + target and td >= MIN_DAYS:
            return "pass"
    return "unresolved"


def pass_fail(by_day, days, starts, risk, conc, per_day, target, cap_ccy=0):
    out = [replay(by_day, days, i, HORIZON, risk, conc, per_day, target, cap_ccy)
           for i in starts]
    n = len(out)
    if n == 0:
        return 0.0, 0.0, 0
    return out.count("pass") / n, out.count("fail") / n, n


CACHE = ROOT / "scratch" / "cache_sltp_trades.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--csv-out", default=str(ROOT / "scratch" / "ftmo_4day_sltp_grid.csv"))
    args = ap.parse_args()

    print("=" * 108)
    print("SL/TP GRID for CONFLUENCE 1.0R (RSI14+RSI7) -- real replay, horizon=4")
    print("=" * 108, flush=True)

    if args.use_cache and CACHE.exists():
        with open(CACHE, "rb") as f:
            by_combo = pickle.load(f)
        print(f"Loaded cache {CACHE}", flush=True)
    else:
        jobs = [(sl, tp, s) for sl in SL_GRID for tp in TP_GRID for s in SYMBOLS]
        cells = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for c in ex.map(_cell, jobs, chunksize=8):
                if c:
                    cells.append(c)
        by_combo = {}
        for sl in SL_GRID:
            for tp in TP_GRID:
                key = (sl, tp)
                trades = [t for c in cells if c["combo"] == key for t in c["trades"]]
                if not trades:
                    continue
                by_day = {}
                for t in trades:
                    by_day.setdefault(t["open"], []).append(t)
                days = sorted(by_day)
                by_combo[key] = (trades, by_day, days)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(by_combo, f)
        print(f"Cached -> {CACHE}", flush=True)

    risks = [0.03, 0.05, 0.06, 0.08, 0.10]
    concs = [15, 20, 25, 30]
    per_days = [6, 10]

    all_results = []
    print(f"\n{'sl_lookback':>12}{'tp':>6}{'risk':>7}{'conc':>6}{'/day':>6}"
          f"{'pass4d':>9}{'fail4d':>9}{'n':>8}")
    print("-" * 108)
    for (sl, tp), (trades, by_day, days) in by_combo.items():
        starts = range(0, len(days) - 1, 4)
        n_trades = len(trades)
        best = None
        for risk in risks:
            for conc in concs:
                for per_day in per_days:
                    p, f, n = pass_fail(by_day, days, starts, risk, conc, per_day, args.target)
                    rec = (p, f, sl, tp, risk, conc, per_day, n_trades)
                    all_results.append(rec)
                    if best is None or p > best[0]:
                        best = rec
        p, f, sl_, tp_, risk, conc, per_day, n_trades = best
        print(f"{sl:>12}{tp:>6.2f}{risk*100:>6.1f}%{conc:>6}{per_day:>6}"
              f"{p:>9.1%}{f:>9.1%}{n_trades:>8}  <- best raw pass for this SL/TP")

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sl_lookback", "tp", "risk", "conc", "per_day", "pass4d", "fail4d", "n_trades"])
        for p, fl, sl, tp, risk, conc, per_day, n_trades in all_results:
            w.writerow([sl, tp, risk, conc, per_day, p, fl, n_trades])
    print(f"\nWrote {len(all_results)} rows -> {args.csv_out}")

    print("\n" + "=" * 108)
    print("TOP 10 BY RAW PASS RATE:")
    top = sorted(all_results, key=lambda x: (-x[0], x[1]))[:10]
    for p, f, sl, tp, risk, conc, per_day, n_trades in top:
        print(f"  pass {p:>6.1%}  fail {f:>6.1%}  n={n_trades:<6}  sl={sl:<3} tp={tp:<5.2f} "
              f"risk {risk*100:>4.1f}%  conc {conc:>2}  {per_day}/day")

    print("\n" + "=" * 108)
    for fail_cap in (0.50, 0.55, 0.60):
        eligible = [r for r in all_results if r[1] <= fail_cap and r[7] >= 300]
        if not eligible:
            print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): no eligible combo")
            continue
        best_def = sorted(eligible, key=lambda x: -x[0])[0]
        p, f, sl, tp, risk, conc, per_day, n_trades = best_def
        print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): pass {p:.1%}  fail {f:.1%}  "
              f"n={n_trades}  sl={sl} tp={tp}  risk {risk*100:.1f}%  conc {conc}  {per_day}/day")

    # half-split for the raw-pass winner and the fail<=50% defensible pick
    print("\n" + "=" * 108)
    print("FIRST-HALF vs SECOND-HALF VALIDATION")
    print("-" * 108)
    seen = set()
    picks = [("RAW-PASS WINNER", top[0])]
    elig50 = [r for r in all_results if r[1] <= 0.50 and r[7] >= 300]
    if elig50:
        picks.append(("DEFENSIBLE fail<=50%", sorted(elig50, key=lambda x: -x[0])[0]))
    for label, (p, f, sl, tp, risk, conc, per_day, n_trades) in picks:
        combo_id = (sl, tp, risk, conc, per_day)
        if combo_id in seen:
            continue
        seen.add(combo_id)
        trades, by_day, days = by_combo[(sl, tp)]
        mid = len(days) // 2
        first_days, second_days = set(days[:mid]), set(days[mid:])
        starts_first = [i for i in range(0, len(days) - 1, 4) if days[i] in first_days]
        starts_second = [i for i in range(0, len(days) - 1, 4) if days[i] in second_days]
        pf1 = pass_fail(by_day, days, starts_first, risk, conc, per_day, args.target)
        pf2 = pass_fail(by_day, days, starts_second, risk, conc, per_day, args.target)
        print(f"  [{label}] sl={sl} tp={tp} risk {risk*100:.1f}% conc {conc} {per_day}/day  n={n_trades}")
        print(f"      1st half: pass {pf1[0]:>6.1%} fail {pf1[1]:>6.1%} (n={pf1[2]})   "
              f"2nd half: pass {pf2[0]:>6.1%} fail {pf2[1]:>6.1%} (n={pf2[2]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
