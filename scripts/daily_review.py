"""Daily analysis of every trade, with statistically-gated auto-adaptation.

WHY THIS IS SPLIT INTO TWO CLASSES OF ACTION
--------------------------------------------
"Analyse the day and improve automatically" is only safe for the things a
handful of trades can actually measure. With ~1 trade/day spread over ~120
(symbol, session) slots, ANY strategy-level adaptation is fitting noise -- and
that is precisely how this project previously produced edges that evaporated
(see the +$374 day that was one correlated bet, then -$832 six days later).

So this splits actions by how fast the underlying quantity converges:

  AUTO-APPLIED (measurement, converges in ~10-20 observations)
    - execution cost calibration: real slippage and realised spread per symbol,
      measured from requested-vs-filled prices. This is a MEASUREMENT of the
      broker, not an estimate of edge, so a small sample is legitimate.
    - execution alarms: signals that never become fills, a gate rejecting
      nearly everything, systematic one-sided slippage. Actionable on day one
      because they are deterministic faults, not statistical claims.

  PROPOSED ONLY (edge estimates, need hundreds of trades)
    - dropping a symbol, session, or leg
    - changing RSI thresholds, stops or targets
    For each, the report states the sample size required and how far away the
    current data is, so "not yet decidable" is visible rather than silently
    acted upon.

Usage:
    python scripts/daily_review.py                      # report only
    python scripts/daily_review.py --apply              # + write cost calibration
    python scripts/daily_review.py --book yolo --days 7
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.instruments import PAIR_SPECS  # noqa: E402

BOOKS = {
    "ftmo": {"ledger": "data/paper_ledger.sqlite", "risk_pct": 0.04,
             "account": 10_000.0, "expect_e": 0.274, "expect_win": 0.331},
    "yolo": {"ledger": "data/paper_ledger_yolo.sqlite", "risk_pct": 0.10,
             "account": 200.0, "expect_e": 0.274, "expect_win": 0.331},
}
CALIB_PATH = ROOT / "data" / "cost_calibration.json"


def rows(con, sql, args=()):
    """Query helper that tolerates older ledgers missing newer tables.

    Archived ledgers predate the `rejections`/`trade_reviews` tables, and a
    review of a historical run should still produce the sections it CAN.
    """
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def required_n(effect: float, sd: float, power_z: float = 2.8) -> int:
    """Trades needed to resolve `effect` at ~80% power, 95% confidence."""
    if effect <= 0 or sd <= 0:
        return 10 ** 9
    return int(math.ceil((power_z * sd / effect) ** 2))


# ----------------------------------------------------------------- sections

def section_trades(con, book, since):
    print("=" * 92)
    print("1. TRADES")
    print("=" * 92)
    cp = rows(con, "SELECT * FROM closed_positions WHERE closed_ts >= ? "
                   "ORDER BY closed_ts", (since,))
    if not cp:
        print("  no closed trades in the window.")
        return []

    risk_usd = book["account"] * book["risk_pct"]
    recs = []
    for r in cp:
        sl_dist = abs((r["entry_price"] or 0) - (r["sl"] or 0))
        rm = (r["pnl_usd"] or 0.0) / risk_usd if risk_usd else 0.0
        recs.append({**r, "r": rm, "sl_dist": sl_dist,
                     "day": str(r["closed_ts"])[:10]})

    by_day = defaultdict(list)
    for r in recs:
        by_day[r["day"]].append(r)

    print(f"  {'day':<12}{'n':>4}{'wins':>6}{'net R':>9}{'net $':>11}{'detail'}")
    for day in sorted(by_day):
        d = by_day[day]
        w = sum(1 for x in d if x["r"] > 0)
        detail = ", ".join(f"{x['symbol']}{x['r']:+.2f}R" for x in d[:4])
        print(f"  {day:<12}{len(d):>4}{w:>6}{sum(x['r'] for x in d):>+9.2f}"
              f"{sum(x['pnl_usd'] or 0 for x in d):>+11.2f}  {detail}")

    rs = [r["r"] for r in recs]
    wins = [x for x in rs if x > 0]
    print(f"\n  TOTAL n={len(rs)}  win={len(wins)/len(rs):.1%}  "
          f"E_net={mean(rs):+.3f}R  net={sum(rs):+.2f}R")
    return recs


def section_vs_backtest(recs, book):
    print("\n" + "=" * 92)
    print("2. LIVE vs BACKTEST -- is the edge behaving as measured?")
    print("=" * 92)
    if len(recs) < 2:
        print(f"  n={len(recs)}: too few to say anything. Not a bad sign, just no signal.")
        return
    rs = [r["r"] for r in recs]
    m, sd = mean(rs), stdev(rs)
    se = sd / math.sqrt(len(rs))
    exp = book["expect_e"]
    print(f"  backtested E_net : {exp:+.3f}R")
    print(f"  live E_net       : {m:+.3f}R  +/- {se:.3f} SE  (n={len(rs)})")
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print(f"  live 95% CI      : [{lo:+.3f}R, {hi:+.3f}R]")
    if lo <= exp <= hi:
        print("  -> consistent with the backtest (the CI contains it). Nothing to act on.")
    elif hi < exp:
        print("  -> live is running BELOW backtest, but check section 3 before blaming")
        print("     the strategy: execution costs explain most such gaps.")
    else:
        print("  -> live is running ABOVE backtest. Almost certainly luck at this n.")
    need = required_n(abs(exp), sd if sd > 0 else 1.0)
    print(f"\n  trades needed to confirm/deny a {exp:+.3f}R edge: ~{need}")
    print(f"  collected so far: {len(rs)}  ({len(rs)/need:.1%} of the way)")


def section_execution(con, since, apply_calib):
    print("\n" + "=" * 92)
    print("3. EXECUTION QUALITY -- the part a small sample CAN measure")
    print("=" * 92)
    orders = rows(con, "SELECT * FROM orders WHERE ts >= ?", (since,))
    filled = [o for o in orders if o.get("filled_price") and o.get("requested_price")]
    calib = {}
    if not filled:
        print("  no filled orders in the window.")
    else:
        per_sym = defaultdict(list)
        for o in filled:
            slip = abs(float(o["filled_price"]) - float(o["requested_price"]))
            per_sym[o["symbol"]].append(slip)
        print(f"  {'symbol':<9}{'fills':>6}{'mean slip':>13}{'assumed spread':>16}"
              f"{'ratio':>8}  verdict")
        for sym, slips in sorted(per_sym.items()):
            spec = PAIR_SPECS.get(sym, {})
            assumed = float(spec.get("spread_price", 0) or 0)
            ms = mean(slips)
            ratio = (ms / assumed) if assumed else float("nan")
            verdict = ("OK" if ratio == ratio and ratio <= 1.5
                       else "WORSE THAN MODELLED" if ratio == ratio else "no spec")
            print(f"  {sym:<9}{len(slips):>6}{ms:>13.6f}{assumed:>16.6f}"
                  f"{ratio:>8.2f}  {verdict}")
            if len(slips) >= 5 and assumed:
                calib[sym] = {"observed_slip": ms, "assumed_spread": assumed,
                              "ratio": ratio, "n": len(slips)}
        if calib and apply_calib:
            CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"updated": datetime.now(timezone.utc).isoformat(),
                       "note": "measured slippage vs PAIR_SPECS assumption; "
                               "symbols with n>=5 fills only",
                       "symbols": calib}
            CALIB_PATH.write_text(json.dumps(payload, indent=2))
            print(f"\n  [APPLIED] cost calibration written -> {CALIB_PATH.name} "
                  f"({len(calib)} symbols)")
        elif calib:
            print(f"\n  [would apply] {len(calib)} symbols have >=5 fills; "
                  f"re-run with --apply to write the calibration file")

    # Signal -> order funnel: the fastest way to spot a config that is
    # generating signals but never trading them.
    sig = rows(con, "SELECT COUNT(*) c FROM signals WHERE ts >= ?", (since,))[0]["c"]
    rej = rows(con, "SELECT gate, COUNT(*) c FROM rejections WHERE ts >= ? "
                    "GROUP BY gate ORDER BY c DESC", (since,))
    print(f"\n  funnel: {sig} signals -> {len(orders)} orders -> {len(filled)} fills")
    if rej:
        total_rej = sum(r["c"] for r in rej)
        print(f"  {total_rej} rejections by gate:")
        for r in rej:
            share = r["c"] / total_rej
            flag = "  <-- DOMINANT, check this gate" if share > 0.5 else ""
            print(f"     {r['gate']:<22}{r['c']:>5}  ({share:.0%}){flag}")
    if sig and len(filled) / max(sig, 1) < 0.2:
        print("\n  ALARM: under 20% of signals became fills. That is an execution or")
        print("  gating fault, not a strategy result -- fix before judging the edge.")


def section_proposals(recs, book):
    print("\n" + "=" * 92)
    print("4. PROPOSED STRATEGY CHANGES -- reported, deliberately NOT auto-applied")
    print("=" * 92)
    if not recs:
        print("  nothing to propose without trades.")
        return
    rs = [r["r"] for r in recs]
    sd = stdev(rs) or 1.0

    for key, label in (("symbol", "SYMBOL"), ("config_name", "LEG")):
        agg = defaultdict(list)
        for r in recs:
            agg[str(r.get(key))].append(r["r"])
        worst = sorted(agg.items(), key=lambda kv: sum(kv[1]))[:3]
        print(f"\n  worst {label}s so far:")
        for name, xs in worst:
            need = required_n(abs(mean(xs)) or 0.05, sd)
            decidable = "DECIDABLE" if len(xs) >= need else f"need ~{need}, have {len(xs)}"
            print(f"     {name[:34]:<35} n={len(xs):<3} net={sum(xs):+6.2f}R  -> {decidable}")

    print("\n  None of the above are acted on automatically. With ~1 trade/day across")
    print("  ~120 (symbol, session) slots, a week yields ~0.04 trades per slot -- the")
    print("  SessionLearner's own 6-trade threshold is ~144 weeks away. That threshold")
    print("  is a FEATURE: adapting on this little data is how the previous rosters")
    print("  were fooled. Re-run the offline sweeps instead once real data accumulates:")
    print("     python scripts/canonical_strategy_test.py")
    print("     python scripts/rsi_geometry_sweep.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=list(BOOKS) + ["both"], default="both")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--apply", action="store_true",
                    help="write the execution cost calibration (safe class only)")
    ap.add_argument("--ledger", help="override the ledger path (e.g. review an archive)")
    args = ap.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    books = list(BOOKS) if args.book == "both" else [args.book]

    for name in books:
        book = BOOKS[name]
        path = Path(args.ledger) if args.ledger else ROOT / book["ledger"]
        if args.ledger and not path.is_absolute():
            path = ROOT / args.ledger
        print("\n" + "#" * 92)
        print(f"# DAILY REVIEW -- {name.upper()}  (last {args.days} day(s))")
        print(f"# ledger: {path.name}")
        print("#" * 92)
        if not path.exists():
            print("  ledger not found -- has the bot run yet?")
            continue
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            recs = section_trades(con, book, since)
            section_vs_backtest(recs, book)
            section_execution(con, since, args.apply)
            section_proposals(recs, book)
        finally:
            con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
