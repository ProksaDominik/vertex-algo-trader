"""Intraday (1h/4h) real-replay for the 4-day FTMO question.

This is the genuinely untested corner: every prior FTMO pass-rate result in
this repo used 1d bars, marking equity once per calendar day. That misses
two things a shorter timeframe could plausibly change:

  1. Faster trade resolution -> more capital velocity within a 4-day
     window (the same lever that made tightening the SL lookback help on
     1d -- see ftmo_4day_sltp_sweep.py).
  2. The -5%/day rule needs to see the actual INTRADAY equity path, not
     just the close-to-close daily move, or it understates how often a
     day gets breached.

Engine changes from the validated 1d engine (ftmo_4day_search.py):
  - Trades keep their exact entry BAR timestamp (not normalized to the
    calendar day), and per-trade "path" is now indexed bar-by-bar at the
    trading timeframe (1h or 4h), not day-by-day.
  - The portfolio clock steps over every bar timestamp at which ANY pooled
    trade opens (the same "event clock" the 1d engine already uses --
    there it steps over days with >=1 entry; here it steps over bars with
    >=1 entry). With 15-20+ FX symbols and several legs, this is dense
    within a day even at 1h resolution, but it is still an event clock,
    not a literal continuous per-bar walk -- see caveats in the report.
  - A calendar-day boundary is detected on every step (ts.normalize()
    changing); the day's floating-vs-open anchor (d0) resets there, and
    that is when the -5%/day and MIN_DAYS bookkeeping happens -- so the
    -5% check now fires on ANY bar within the day where floating equity
    breaches it, not just at the day's last bar.
  - horizon is still 4 TRADING DAYS (calendar dates elapsed), not 4 bars.

No Monte Carlo. Same three FTMO rules, same MIN_DAYS=4 gate, walking every
possible start date (start = the first bar of each calendar day).

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_intraday.py --workers 8
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
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_orb import ORBStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402
from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy  # noqa: E402

MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY"]
BASE17 = MAJORS + ["GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
                    "AUDCHF", "EURAUD", "AUDCAD"]

RSI14 = (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}})
RSI7 = (RSIReversionStrategy, {"rsi": {"period": 7, "oversold": 25, "overbought": 75}})
BOLL = (BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}})
SUPER = (SupertrendStrategy, {"supertrend": {"period": 10, "multiplier": 3.0}})
ORB = (ORBStrategy, {"orb": {}})
LONDON = (LondonBreakoutStrategy, {"london_breakout": {}})
VWAP = (VWAPReversionStrategy, {"vwap": {}})

# Trimmed to fit a bounded single run: the core hypothesis (does the
# already-proven-best 1d strategy do better at finer resolution?) on 4h
# with the full symbol set, plus a light 1h check (CONFLUENCE + the
# cheapest session-native strategy, ORB) on the majors only.
JOBSETS = {
    "4h": {
        "symbols": BASE17,
        "rosters": {
            "CONFLUENCE 1.0R (RSI14+RSI7)": [("CONF", RSI14, 1.0, RSI7)],
        },
    },
    "1h": {
        "symbols": MAJORS,
        "rosters": {
            "CONFLUENCE 1.0R (RSI14+RSI7)": [("CONF", RSI14, 1.0, RSI7)],
            "ORB alone": [("ORB", ORB, 2.0, None)],
        },
    },
}

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
HORIZON_DAYS = 4


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
    tf, rkey, li, sym = job
    label, cls_params, tp, partner = JOBSETS[tf]["rosters"][rkey][li]
    try:
        bars = load(sym, tf)
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        closed = _run(cls_params, tp, sym, bars)
        keep = None
        if partner is not None:
            other = _run(partner, tp, sym, bars)
            keep = {(t.entered_at, t.direction.name) for t in other}
        cl = bars["close"]
        out = []
        for t in closed:
            o = t.entered_at  # exact bar timestamp -- NOT normalized to day
            if keep is not None and (o, t.direction.name) not in keep:
                continue
            c = t.exited_at
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            if len(seg) < 1:
                continue
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": pd.Timestamp(o), "path": list(path),
                        "final": float(path[-1]), "sym": sym})
        return {"tf": tf, "roster": rkey, "trades": out}
    except Exception:
        return None


def intraday_replay(by_bar, bars_clock, i0, horizon_days, risk, conc, per_day, target):
    """Same portfolio mechanics as the 1d engine's replay(), but the clock
    steps over BARS (event clock: every timestamp where >=1 pooled trade
    opens), and a calendar-day boundary (not a bar boundary) is what resets
    the daily-loss anchor and advances the MIN_DAYS / horizon counters."""
    eq, real, op = 1.0, 0.0, []
    td = 0
    current_date = None
    days_covered = 0
    opened_today = 0
    opened_any_today = False
    d0 = eq
    i = i0
    n = len(bars_clock)
    while days_covered < horizon_days and i < n:
        ts = bars_clock[i]
        date = ts.normalize()
        if date != current_date:
            if opened_any_today:
                td += 1
            current_date = date
            d0 = eq
            days_covered += 1
            opened_today = 0
            opened_any_today = False
            if days_covered > horizon_days:
                break
        for tr in by_bar.get(ts, []):
            if len(op) >= conc or opened_today >= per_day:
                break
            op.append([tr, 0])
            opened_today += 1
            opened_any_today = True
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
        i += 1
    return "unresolved"


def pass_fail(by_bar, bars_clock, starts, risk, conc, per_day, target):
    out = [intraday_replay(by_bar, bars_clock, i, HORIZON_DAYS, risk, conc, per_day, target)
           for i in starts]
    n = len(out)
    if n == 0:
        return 0.0, 0.0, 0
    return out.count("pass") / n, out.count("fail") / n, n


CACHE = ROOT / "scratch" / "cache_intraday_trades.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    print("=" * 108)
    print("INTRADAY (1h/4h) 4-DAY FTMO SEARCH -- bar-level equity marking, real replay")
    print("=" * 108, flush=True)

    if args.use_cache and CACHE.exists():
        with open(CACHE, "rb") as f:
            by_key = pickle.load(f)
    else:
        jobs = [(tf, rk, li, s) for tf, cfg in JOBSETS.items()
                for rk in cfg["rosters"] for li in range(len(cfg["rosters"][rk]))
                for s in cfg["symbols"]]
        print(f"{len(jobs)} (tf, roster, leg, symbol) cells to backtest", flush=True)
        cells = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            done = 0
            for c in ex.map(_cell, jobs, chunksize=4):
                if c:
                    cells.append(c)
                done += 1
                if done % 20 == 0:
                    print(f"  ... {done}/{len(jobs)} cells", flush=True)
        by_key = {}
        for tf, cfg in JOBSETS.items():
            for rk in cfg["rosters"]:
                trades = [t for c in cells if c["tf"] == tf and c["roster"] == rk
                          for t in c["trades"]]
                if not trades:
                    continue
                by_bar = {}
                for t in trades:
                    by_bar.setdefault(t["open"], []).append(t)
                bars_clock = sorted(by_bar)
                by_key[(tf, rk)] = (trades, by_bar, bars_clock)
        with open(CACHE, "wb") as f:
            pickle.dump(by_key, f)
        print(f"Cached -> {CACHE}", flush=True)

    risks = [0.04, 0.05, 0.06, 0.08, 0.10]
    concs = [15, 20, 30]
    per_days = [6, 10]

    all_results = []
    print(f"\n{'tf':>4}{'roster':<32}{'risk':>6}{'conc':>6}{'/day':>6}"
          f"{'pass4d':>9}{'fail4d':>9}{'n':>8}")
    print("-" * 108)
    for (tf, rk), (trades, by_bar, bars_clock) in by_key.items():
        n_trades = len(trades)
        # start = first bar of each distinct calendar date, strided by 4
        # dates for comparability with the 1d engine's cadence
        seen_dates, starts = set(), []
        for idx, ts in enumerate(bars_clock):
            d = ts.normalize()
            if d not in seen_dates:
                seen_dates.add(d)
                starts.append(idx)
        starts = starts[::4]
        best = None
        for risk in risks:
            for conc in concs:
                for per_day in per_days:
                    p, f, n = pass_fail(by_bar, bars_clock, starts, risk, conc, per_day, args.target)
                    rec = (p, f, tf, rk, risk, conc, per_day, n_trades)
                    all_results.append(rec)
                    if best is None or p > best[0]:
                        best = rec
        p, f, tf_, rk_, risk, conc, per_day, n_trades = best
        print(f"{tf:>4}{rk[:31]:<32}{risk*100:>5.1f}%{conc:>6}{per_day:>6}"
              f"{p:>9.1%}{f:>9.1%}{n_trades:>8}  <- best raw pass")

    csv_out = ROOT / "scratch" / "ftmo_4day_intraday_grid.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tf", "roster", "risk", "conc", "per_day", "pass4d", "fail4d", "n_trades"])
        for p, fl, tf, rk, risk, conc, per_day, n_trades in all_results:
            w.writerow([tf, rk, risk, conc, per_day, p, fl, n_trades])
    print(f"\nWrote {len(all_results)} rows -> {csv_out}")

    print("\n" + "=" * 108)
    print("TOP 10 BY RAW PASS RATE:")
    top = sorted(all_results, key=lambda x: (-x[0], x[1]))[:10]
    for p, f, tf, rk, risk, conc, per_day, n_trades in top:
        print(f"  pass {p:>6.1%}  fail {f:>6.1%}  n={n_trades:<6}  tf={tf:<3} "
              f"risk {risk*100:>4.1f}%  conc {conc:>2}  {per_day}/day  {rk}")

    print("\n" + "=" * 108)
    for fail_cap in (0.50, 0.55, 0.60):
        eligible = [r for r in all_results if r[1] <= fail_cap and r[7] >= 300]
        if not eligible:
            print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): no eligible combo")
            continue
        best_def = sorted(eligible, key=lambda x: -x[0])[0]
        p, f, tf, rk, risk, conc, per_day, n_trades = best_def
        print(f"DEFENSIBLE (fail<={fail_cap:.0%}, n>=300): pass {p:.1%}  fail {f:.1%}  "
              f"n={n_trades}  tf={tf}  {rk}  risk {risk*100:.1f}%  conc {conc}  {per_day}/day")

    # half-split for the raw-pass winner + anything that looks surprisingly
    # good (>=35% pass) -- explicit red-flag check for sample collapse
    print("\n" + "=" * 108)
    print("FIRST-HALF vs SECOND-HALF VALIDATION")
    print("-" * 108)
    flagged = [r for r in all_results if r[0] >= 0.35 and r[7] >= 300]
    picks = {(top[0][2], top[0][3]) + tuple(top[0][4:7]): top[0]}
    for r in flagged:
        picks[(r[2], r[3]) + tuple(r[4:7]) + ("FLAGGED",)] = r
    for label, (p, f, tf, rk, risk, conc, per_day, n_trades) in picks.items():
        trades, by_bar, bars_clock = by_key[(tf, rk)]
        seen_dates, starts = set(), []
        for idx, ts in enumerate(bars_clock):
            d = ts.normalize()
            if d not in seen_dates:
                seen_dates.add(d)
                starts.append(idx)
        mid = len(starts) // 2
        starts_first, starts_second = starts[:mid][::4], starts[mid:][::4]
        pf1 = pass_fail(by_bar, bars_clock, starts_first, risk, conc, per_day, args.target)
        pf2 = pass_fail(by_bar, bars_clock, starts_second, risk, conc, per_day, args.target)
        print(f"  [{label}] risk {risk*100:.1f}% conc {conc} {per_day}/day  n={n_trades}  "
              f"(full: pass {p:.1%} fail {f:.1%})")
        print(f"      1st half: pass {pf1[0]:>6.1%} fail {pf1[1]:>6.1%} (n={pf1[2]})   "
              f"2nd half: pass {pf2[0]:>6.1%} fail {pf2[1]:>6.1%} (n={pf2[2]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
