"""Other popular strategies, tested for 4-day FTMO PASS RATE (not raw
expectancy) on 1d bars: MACD, Donchian/Turtle breakout, EMA crossover,
Ichimoku. canonical_strategy_test.py already found these marginal-to-weak
on raw OOS expectancy at 1d -- but the RSI confluence + SL-tightening work
in this investigation already showed pass-rate-for-a-deadline is a
different question from raw expectancy (smoothness/throughput matter as
much as edge). So: test them standalone AND stacked with the established
CONFLUENCE (RSI14+RSI7) winner for throughput diversity, at horizon=4,
same real-replay engine as scratch/ftmo_4day_search.py.

No Monte Carlo. Real daily bars, mark-to-market daily, all three FTMO
rules, MIN_DAYS=4, every possible start date.

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_other_strategies.py --workers 8
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
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

BASE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
        "AUDCHF", "EURAUD", "AUDCAD"]
EXTRA = [s for s in ("EURGBP", "GBPAUD", "EURNZD", "CADJPY", "AUDNZD")
         if s in PAIR_SPECS and (ROOT / "data" / f"{s.lower()}_15m_long.parquet").exists()]
SYMBOLS = BASE + EXTRA

RSI14 = (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}})
RSI7 = (RSIReversionStrategy, {"rsi": {"period": 7, "oversold": 25, "overbought": 75}})
DONCHIAN = (DonchianBreakoutStrategy, {"donchian": {"period": 20}})
MACD = (MACDTrendStrategy, {"macd": {}})
EMAX = (EMACrossoverStrategy, {"ema_cross": {"fast_period": 20, "slow_period": 50}})
ICHI = (IchimokuStrategy, {"ichimoku": {}})

# roster: list of (label, cls_params, tp, confluence_partner|None)
ROSTERS: dict[str, list] = {
    "Donchian20_turtle alone": [("Donch", DONCHIAN, 2.0, None)],
    "MACD alone": [("MACD", MACD, 2.0, None)],
    "EMACross_20_50 alone": [("EMAX", EMAX, 2.0, None)],
    "Ichimoku alone": [("Ichi", ICHI, 2.0, None)],
    "CONF 1.0R + Donchian20@2R": [("CONF", RSI14, 1.0, RSI7), ("Donch", DONCHIAN, 2.0, None)],
    "CONF 1.0R + MACD@2R": [("CONF", RSI14, 1.0, RSI7), ("MACD", MACD, 2.0, None)],
    "CONF 1.0R + EMAX@2R": [("CONF", RSI14, 1.0, RSI7), ("EMAX", EMAX, 2.0, None)],
    "CONF 1.0R + Ichimoku@2R": [("CONF", RSI14, 1.0, RSI7), ("Ichi", ICHI, 2.0, None)],
}

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZON = 4


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
            out.append({"open": o, "path": list(path), "final": float(path[-1]), "sym": sym})
        return {"roster": rkey, "trades": out}
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


def pass_fail(by_day, days, starts, risk, conc, per_day, target):
    out = [replay(by_day, days, i, HORIZON, risk, conc, per_day, target) for i in starts]
    n = len(out)
    if n == 0:
        return 0.0, 0.0, 0
    return out.count("pass") / n, out.count("fail") / n, n


CACHE = ROOT / "scratch" / "cache_other_strategies.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    print("=" * 108)
    print("OTHER POPULAR STRATEGIES -- 4-day pass rate, real replay")
    print("=" * 108, flush=True)

    if args.use_cache and CACHE.exists():
        with open(CACHE, "rb") as f:
            by_roster = pickle.load(f)
    else:
        jobs = [(k, i, s) for k in ROSTERS for i in range(len(ROSTERS[k])) for s in SYMBOLS]
        cells = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for c in ex.map(_cell, jobs, chunksize=8):
                if c:
                    cells.append(c)
        by_roster = {}
        for key in ROSTERS:
            trades = [t for c in cells if c["roster"] == key for t in c["trades"]]
            if not trades:
                continue
            by_day = {}
            for t in trades:
                by_day.setdefault(t["open"], []).append(t)
            days = sorted(by_day)
            by_roster[key] = (trades, by_day, days)
        with open(CACHE, "wb") as f:
            pickle.dump(by_roster, f)
        print(f"Cached -> {CACHE}", flush=True)

    risks = [0.04, 0.05, 0.06, 0.08, 0.10]
    concs = [15, 20, 30]
    per_days = [6, 10]

    all_results = []
    print(f"\n{'roster':<32}{'risk':>6}{'conc':>6}{'/day':>6}{'pass4d':>9}{'fail4d':>9}{'n':>8}")
    print("-" * 108)
    for key, (trades, by_day, days) in by_roster.items():
        starts = range(0, len(days) - 1, 4)
        n_trades = len(trades)
        best = None
        for risk in risks:
            for conc in concs:
                for per_day in per_days:
                    p, f, n = pass_fail(by_day, days, starts, risk, conc, per_day, args.target)
                    rec = (p, f, key, risk, conc, per_day, n_trades)
                    all_results.append(rec)
                    if best is None or p > best[0]:
                        best = rec
        p, f, key_, risk, conc, per_day, n_trades = best
        print(f"{key[:31]:<32}{risk*100:>5.1f}%{conc:>6}{per_day:>6}"
              f"{p:>9.1%}{f:>9.1%}{n_trades:>8}  <- best raw pass")

    csv_out = ROOT / "scratch" / "ftmo_4day_other_strategies_grid.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roster", "risk", "conc", "per_day", "pass4d", "fail4d", "n_trades"])
        for p, fl, key, risk, conc, per_day, n_trades in all_results:
            w.writerow([key, risk, conc, per_day, p, fl, n_trades])
    print(f"\nWrote {len(all_results)} rows -> {csv_out}")

    print("\n" + "=" * 108)
    print("TOP 10 BY RAW PASS RATE:")
    top = sorted(all_results, key=lambda x: (-x[0], x[1]))[:10]
    for p, f, key, risk, conc, per_day, n_trades in top:
        print(f"  pass {p:>6.1%}  fail {f:>6.1%}  n={n_trades:<6}  risk {risk*100:>4.1f}%  "
              f"conc {conc:>2}  {per_day}/day  {key}")

    print("\n" + "=" * 108)
    for fail_cap in (0.50, 0.55, 0.60):
        eligible = [r for r in all_results if r[1] <= fail_cap and r[6] >= 300]
        if not eligible:
            print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): no eligible combo")
            continue
        best_def = sorted(eligible, key=lambda x: -x[0])[0]
        p, f, key, risk, conc, per_day, n_trades = best_def
        print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): pass {p:.1%}  fail {f:.1%}  "
              f"n={n_trades}  risk {risk*100:.1f}%  conc {conc}  {per_day}/day  {key}")

    # half-split validation on anything that looks surprisingly good (>=35%
    # pass) as a sample-collapse / methodology-bug red flag check, plus the
    # raw-pass winner and the fail<=55% defensible pick regardless.
    print("\n" + "=" * 108)
    print("FIRST-HALF vs SECOND-HALF VALIDATION")
    print("-" * 108)
    flagged = [r for r in all_results if r[0] >= 0.35 and r[6] >= 300]
    picks = {top[0][2] + str(top[0][3:6]): top[0]}
    elig55 = [r for r in all_results if r[1] <= 0.55 and r[6] >= 300]
    if elig55:
        b = sorted(elig55, key=lambda x: -x[0])[0]
        picks[b[2] + str(b[3:6])] = b
    for r in flagged:
        picks[r[2] + str(r[3:6]) + "_FLAGGED"] = r
    for label, (p, f, key, risk, conc, per_day, n_trades) in picks.items():
        trades, by_day, days = by_roster[key]
        mid = len(days) // 2
        first_days, second_days = set(days[:mid]), set(days[mid:])
        starts_first = [i for i in range(0, len(days) - 1, 4) if days[i] in first_days]
        starts_second = [i for i in range(0, len(days) - 1, 4) if days[i] in second_days]
        pf1 = pass_fail(by_day, days, starts_first, risk, conc, per_day, args.target)
        pf2 = pass_fail(by_day, days, starts_second, risk, conc, per_day, args.target)
        print(f"  [{label}] risk {risk*100:.1f}% conc {conc} {per_day}/day  n={n_trades}  "
              f"(full: pass {p:.1%} fail {f:.1%})")
        print(f"      1st half: pass {pf1[0]:>6.1%} fail {pf1[1]:>6.1%} (n={pf1[2]})   "
              f"2nd half: pass {pf2[0]:>6.1%} fail {pf2[1]:>6.1%} (n={pf2[2]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
