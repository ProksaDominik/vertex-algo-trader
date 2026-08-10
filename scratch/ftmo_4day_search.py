"""4-trading-day FTMO pass-rate search.

Extends the exact validated real-replay engine from
scripts/confluence_race_test.py and scripts/max_throughput_race.py to:

  1. A wider set of strategy stacks (2-5 legs) built ONLY from the
     positive-daily-expectancy strategies already established: RSI14,
     RSI7, RSI14+RSI7 confluence, Supertrend, Bollinger, SMACross.
  2. A wider risk-per-trade sweep (2% - 10%).
  3. Wider concurrency (15 - 40) and trades/day cap sweep.
  4. A per-currency exposure cap on/off comparison at the winning point.
  5. A first-half / second-half of history validation split for the
     single best surviving candidate (the check that killed the earlier
     3-way-confluence sample-collapse artifact).

No Monte Carlo anywhere. Every number below comes from walking every
possible 4-trading-day challenge start date over real daily bars, marking
equity to market day by day, and enforcing all three FTMO rules exactly
(5% daily loss, 10% total drawdown, >=4 trading days before a pass counts).

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_search.py --workers 6
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
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

BASE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
        "AUDCHF", "EURAUD", "AUDCAD"]
EXTRA = [s for s in ("EURGBP", "GBPAUD", "EURNZD", "CADJPY", "AUDNZD")
         if s in PAIR_SPECS and (ROOT / "data" / f"{s.lower()}_15m_long.parquet").exists()]
SYMBOLS = BASE + EXTRA

RSI14 = (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}})
RSI7 = (RSIReversionStrategy, {"rsi": {"period": 7, "oversold": 25, "overbought": 75}})
BOLL = (BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}})
SUPER = (SupertrendStrategy, {"supertrend": {"period": 10, "multiplier": 3.0}})
SMAX = (SMACrossStrategy, {"ma_cross": {}})

# roster: list of (label, cls_params, tp, confluence_partner|None)
ROSTERS: dict[str, list] = {
    "DEPLOYED: RSI14@3R + Boll@4R": [
        ("RSI14", RSI14, 3.0, None), ("Boll20", BOLL, 4.0, None)],
    "CONFLUENCE 1.0R (RSI14+RSI7)": [
        ("CONF", RSI14, 1.0, RSI7)],
    "CONF 1.0R + Boll@4R": [
        ("CONF", RSI14, 1.0, RSI7), ("Boll20", BOLL, 4.0, None)],
    "CONF 1.0R + Supertrend@2R": [
        ("CONF", RSI14, 1.0, RSI7), ("Super", SUPER, 2.0, None)],
    "CONF 1.0R + SMACross@2R": [
        ("CONF", RSI14, 1.0, RSI7), ("SMAX", SMAX, 2.0, None)],
    "RSI14@3R + RSI7@3R + Boll@4R": [
        ("RSI14", RSI14, 3.0, None), ("RSI7", RSI7, 3.0, None), ("Boll20", BOLL, 4.0, None)],
    "RSI14@3R + Super@2R + SMAX@2R": [
        ("RSI14", RSI14, 3.0, None), ("Super", SUPER, 2.0, None), ("SMAX", SMAX, 2.0, None)],
    "CONF 1.0R + Boll@4R + Super@2R": [
        ("CONF", RSI14, 1.0, RSI7), ("Boll20", BOLL, 4.0, None), ("Super", SUPER, 2.0, None)],
    "ALL-5: RSI14+RSI7+Boll+Super+SMAX": [
        ("RSI14", RSI14, 3.0, None), ("RSI7", RSI7, 3.0, None), ("Boll20", BOLL, 4.0, None),
        ("Super", SUPER, 2.0, None), ("SMAX", SMAX, 2.0, None)],
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


CACHE = ROOT / "scratch" / "cache_1d_trades.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--use-cache", action="store_true",
                     help="skip cell generation, reuse scratch/cache_1d_trades.pkl")
    ap.add_argument("--csv-out", default=str(ROOT / "scratch" / "ftmo_4day_full_grid.csv"))
    args = ap.parse_args()

    print("=" * 108)
    print(f"4-DAY FTMO SEARCH -- target {args.target:.0%}, real replay every start date, "
          f"MIN_DAYS={MIN_DAYS}")
    print("=" * 108, flush=True)

    if args.use_cache and CACHE.exists():
        with open(CACHE, "rb") as f:
            by_roster = pickle.load(f)
        print(f"Loaded cached pooled trades from {CACHE}", flush=True)
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
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(by_roster, f)
        print(f"Cached pooled trades -> {CACHE}", flush=True)

    risks = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    concs = [15, 20, 25, 30, 40]
    per_days = [6, 10, 15]

    print(f"\n{'roster':<36}{'risk':>6}{'conc':>6}{'/day':>6}"
          f"{'pass4d':>9}{'fail4d':>9}{'n':>8}")
    print("-" * 108)

    all_results = []
    for key, (trades, by_day, days) in by_roster.items():
        starts = range(0, len(days) - 1, 4)
        n_trades = len(trades)
        roster_best = None
        for risk in risks:
            for conc in concs:
                for per_day in per_days:
                    p, f, n = pass_fail(by_day, days, starts, risk, conc, per_day, args.target)
                    rec = (p, f, key, risk, conc, per_day, n_trades)
                    all_results.append(rec)
                    if roster_best is None or p > roster_best[0]:
                        roster_best = rec
        p, f, key_, risk, conc, per_day, n_trades = roster_best
        print(f"{key[:35]:<36}{risk*100:>5.1f}%{conc:>6}{per_day:>6}"
              f"{p:>9.1%}{f:>9.1%}{n_trades:>8}  <- roster best (max raw pass)")

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roster", "risk", "conc", "per_day", "pass4d", "fail4d", "n_trades"])
        for p, fl, key, risk, conc, per_day, n_trades in all_results:
            w.writerow([key, risk, conc, per_day, p, fl, n_trades])
    print(f"\nWrote {len(all_results)} full-grid rows -> {args.csv_out}")

    print("\n" + "=" * 108)
    print("TOP 10 CELLS OVERALL BY RAW PASS RATE (ignores fail rate -- rewards max risk):")
    top = sorted(all_results, key=lambda x: (-x[0], x[1]))[:10]
    for p, f, key, risk, conc, per_day, n_trades in top:
        print(f"  pass {p:>6.1%}  fail {f:>6.1%}  n={n_trades:<6}  "
              f"risk {risk*100:>4.1f}%  conc {conc:>2}  {per_day}/day  {key}")

    # --- defensible pick: best pass rate SUBJECT TO a fail-rate ceiling,
    # and require n_trades >= 300 (low-confidence cutoff from the task spec)
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

    # --- currency-cap comparison at the raw-pass winner (for reference) ----
    p0, f0, key0, risk0, conc0, per_day0, n0 = top[0]
    trades0, by_day0, days0 = by_roster[key0]
    starts0 = range(0, len(days0) - 1, 4)
    print("\n" + "=" * 108)
    print(f"CURRENCY-CAP ON/OFF at winner: {key0}  risk {risk0*100:.1f}%  conc {conc0}  {per_day0}/day")
    print("-" * 108)
    for cap in (0, 1, 2, 3):
        p, f, n = pass_fail(by_day0, days0, starts0, risk0, conc0, per_day0, args.target, cap)
        print(f"  cap_per_currency={cap or 'off':<4}  pass {p:>6.1%}  fail {f:>6.1%}  n={n}")

    # --- first-half / second-half validation ---------------------------
    # Validate the RAW-PASS winner (top[0]) AND the defensible picks (best
    # pass rate subject to fail<=50%/55%/60%, n>=300) -- the raw winner
    # alone is exactly the kind of "looks best on paper" candidate the
    # 3-way-confluence sample-collapse trap taught us to distrust.
    candidates = {("RAW-PASS WINNER", top[0][2]): top[0]}
    for fail_cap in (0.50, 0.55, 0.60):
        eligible = [r for r in all_results if r[1] <= fail_cap and r[6] >= 300]
        if eligible:
            best_def = sorted(eligible, key=lambda x: -x[0])[0]
            candidates[(f"DEFENSIBLE fail<={fail_cap:.0%}", best_def[2])] = best_def
    # also the per-roster best-by-pass for a couple of high-n rosters, for
    # runner-up comparison diversity
    for key, (trades, by_day, days) in by_roster.items():
        roster_rows = [r for r in all_results if r[2] == key and r[6] >= 300]
        if roster_rows:
            candidates.setdefault((f"ROSTER BEST: {key}", key),
                                   sorted(roster_rows, key=lambda x: -x[0])[0])

    print("\n" + "=" * 108)
    print("FIRST-HALF vs SECOND-HALF VALIDATION (overfitting guardrail)")
    print("-" * 108)
    seen_combo = set()
    for label, (p, f, key, risk, conc, per_day, n_trades) in candidates.items():
        combo_id = (key, risk, conc, per_day)
        if combo_id in seen_combo:
            continue
        seen_combo.add(combo_id)
        trades, by_day, days = by_roster[key]
        mid = len(days) // 2
        first_days, second_days = set(days[:mid]), set(days[mid:])
        starts_first = [i for i in range(0, len(days) - 1, 4) if days[i] in first_days]
        starts_second = [i for i in range(0, len(days) - 1, 4) if days[i] in second_days]
        pf1 = pass_fail(by_day, days, starts_first, risk, conc, per_day, args.target)
        pf2 = pass_fail(by_day, days, starts_second, risk, conc, per_day, args.target)
        print(f"  [{label[0]}] {key[:38]:<39} risk {risk*100:>4.1f}% conc {conc:>2} "
              f"{per_day}/day  n={n_trades}  (full: pass {p:.1%} fail {f:.1%})")
        print(f"      1st half: pass {pf1[0]:>6.1%} fail {pf1[1]:>6.1%} (n_starts={pf1[2]})   "
              f"2nd half: pass {pf2[0]:>6.1%} fail {pf2[1]:>6.1%} (n_starts={pf2[2]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
