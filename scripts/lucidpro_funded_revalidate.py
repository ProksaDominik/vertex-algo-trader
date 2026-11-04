"""Periodic re-validation of the LucidPro FUNDED-STAGE roster
(run_lucidpro_funded.py) -- the disciplined alternative to "self-improving
after each day", which this project already measured and rejected (see
scripts/learner_value_test.py: benching legs on recent results cut FTMO
expectancy ~60% because short-window results are noise, not signal, for a
book trading well under 1x/week per leg -- see this script's own docstring
for the exact per-leg trade-frequency numbers that make daily adaptation
statistically meaningless here).

Same two-tier split as scripts/daily_review.py, which already established
this pattern for the FTMO/YOLO books:

  AUTO-CHECKED, report only (measurement, converges reasonably fast):
    - live trade count and MLL-lock status (has equity crossed the Initial
      Trail Balance? see scripts/lucidpro_mll_lock_taper.py) if the funded
      bot has actually been running
    - whether refreshed price data still confirms each deployed leg's edge
      at the SAME parameters already chosen (e_net > 0, both-halves-
      positive, n >= 100, survives 5x modelled slippage) -- this catches
      "the edge quietly died" without re-running the full strategy search

  FLAGGED ONLY, never auto-applied (edge estimates, need real time to pass):
    - "a full re-screen (new strategies/timeframes/targets, like the
      2026-08-14 research that found the current roster) is due" -- fires
      once >= FULL_RESCREEN_DAYS have passed since the roster was last
      substantively changed, because that kind of search only makes sense
      periodically, not on every run

This script NEVER edits run_lucidpro_funded.py. It writes a dated report to
logs/lucidpro_funded_review/ and prints a summary. A human (or a future
Claude session) reads the report and decides whether to act.

Usage:
    .venv/Scripts/python.exe scripts/lucidpro_funded_revalidate.py
    .venv/Scripts/python.exe scripts/lucidpro_funded_revalidate.py --skip-download
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scratch"))

from canonical_strategy_test import RISK_USD, Windowed  # noqa: E402
from lucidpro_sameday_search import load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

# Mirrors run_lucidpro_funded.py's ROSTER exactly -- keep in sync by hand;
# importing it directly would require MT5 to be importable, which this
# report-only script should not need.
ROSTER = [
    ("Donchian_BTCUSD_1R", "BTCUSD", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 25}}, 1.0),
    ("Donchian_NAS100_1R", "NAS100", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 40}}, 1.0),
    ("Donchian_XAUUSD_1R", "XAUUSD", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 10}}, 1.0),
    ("Supertrend_JP225_1R", "JP225", "4h", SupertrendStrategy,
        {"supertrend": {"period": 14, "multiplier": 3.5}}, 1.0),
    ("Donchian_XAUUSD_6R", "XAUUSD", "4h", DonchianBreakoutStrategy, {"donchian": {}}, 6.0),
    ("Donchian_BTCUSD_3R", "BTCUSD", "4h", DonchianBreakoutStrategy, {"donchian": {}}, 3.0),
    ("MACD_XAUUSD_2R", "XAUUSD", "1d", MACDTrendStrategy, {"macd": {}}, 2.0),
    ("Ichimoku_BTCUSD_2R", "BTCUSD", "1d", IchimokuStrategy, {"ichimoku": {}}, 2.0),
]
BASKET_SYMBOLS = ["US500", "NAS100", "WTI", "XAUUSD", "DE40", "JP225", "BTCUSD", "ETHUSD"]

ROSTER_LAST_CHANGED = date(2026, 8, 14)  # bump this by hand whenever the roster is edited
FULL_RESCREEN_DAYS = 90
MIN_N = 100
SLIPPAGE_GATE_MULT = 5

VIRTUAL_ACCOUNT = 50_000.0
MLL_USD = 2_000.0
INITIAL_TRAIL_USD = 52_100.0  # start + MLL + $100 buffer, see run_lucidpro_funded.py docstring

STATE_PATH = ROOT / "data" / "state_lucidpro_funded.json"
TRADES_LOG_PATH = ROOT / "data" / "lucidpro_funded_trades.jsonl"
REPORT_DIR = ROOT / "logs" / "lucidpro_funded_review"


def refresh_data() -> None:
    print(f"[1/3] Refreshing price data for {BASKET_SYMBOLS}...")
    cmd = [str(ROOT / ".venv" / "Scripts" / "python.exe"),
           str(ROOT / "scripts" / "download_history.py"), *BASKET_SYMBOLS]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"  [WARN] download_history.py exited {result.returncode} -- "
              f"continuing with existing on-disk data. stderr tail:\n{result.stderr[-500:]}")
    else:
        print("  done.")


def stats(closed) -> dict:
    rs = [t.pnl_usd / RISK_USD for t in closed]
    n = len(rs)
    if n == 0:
        return {"n": 0, "e_net": 0.0, "half1": None, "half2": None}
    e_net = sum(rs) / n
    mid_ts = sorted(t.entered_at for t in closed)[n // 2]
    h1 = [t.pnl_usd / RISK_USD for t in closed if t.entered_at < mid_ts]
    h2 = [t.pnl_usd / RISK_USD for t in closed if t.entered_at >= mid_ts]
    e1 = sum(h1) / len(h1) if h1 else None
    e2 = sum(h2) / len(h2) if h2 else None
    return {"n": n, "e_net": e_net, "half1": e1, "half2": e2}


def check_leg(label: str, sym: str, tf: str, cls, params: dict, tp: float) -> dict:
    bars = load(sym, tf)
    if bars is None:
        return {"label": label, "status": "NO_DATA"}
    specs = PAIR_SPECS[sym]
    strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
    strat.update(params)
    base_cfg = {"symbol": sym, "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                                            "max_positions": 3, "daily_stop_loss_usd": -1e9},
                "strategy": strat}
    bt = Backtest(base_cfg, bars, strategy=Windowed(cls(base_cfg)))
    bt.run()
    base_stats = stats(bt.closed)

    stress_cfg = dict(base_cfg)
    stress_cfg["backtest"] = {"spread_price": specs["spread_price"] * SLIPPAGE_GATE_MULT,
                               "commission_per_lot": specs["commission_per_lot"] * SLIPPAGE_GATE_MULT}
    bts = Backtest(stress_cfg, bars, strategy=Windowed(cls(stress_cfg)))
    bts.run()
    stress_e = stats(bts.closed)["e_net"]

    both_pos = base_stats["half1"] is not None and base_stats["half1"] > 0 and base_stats["half2"] > 0
    ok = (base_stats["n"] >= MIN_N and both_pos and base_stats["e_net"] > 0 and stress_e > 0)
    status = "OK" if ok else "DEGRADED"
    return {"label": label, "sym": sym, "tf": tf, "tp": tp, "status": status,
            "n": base_stats["n"], "e_net": base_stats["e_net"],
            "half1": base_stats["half1"], "half2": base_stats["half2"], "x5_e_net": stress_e}


def check_live_status() -> dict:
    out = {"has_state": False, "trades": 0, "mll_locked": False, "current_equity": None}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            out["has_state"] = True
            out["halted"] = state.get("halted", False)
            out["halt_reason"] = state.get("halt_reason")
            out["peak_equity"] = state.get("peak_equity")
            out["realized_usd"] = state.get("realized_usd")
            if state.get("peak_equity") is not None:
                out["mll_locked"] = state["peak_equity"] * VIRTUAL_ACCOUNT >= INITIAL_TRAIL_USD \
                    if state["peak_equity"] < 100 else state["peak_equity"] >= INITIAL_TRAIL_USD
        except (OSError, ValueError):
            pass
    if TRADES_LOG_PATH.exists():
        out["trades"] = sum(1 for _ in open(TRADES_LOG_PATH, encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true",
                     help="Skip refreshing price data (use existing on-disk parquet files)")
    args = ap.parse_args()

    lines = []
    lines.append(f"=== LucidPro funded-roster re-validation -- {datetime.now(timezone.utc).isoformat()} ===")

    if not args.skip_download:
        refresh_data()
    else:
        lines.append("[1/3] Skipped data refresh (--skip-download)")

    lines.append("[2/3] Re-checking each deployed leg against current data...")
    print(lines[-1])
    results = []
    for label, sym, tf, cls, params, tp in ROSTER:
        r = check_leg(label, sym, tf, cls, params, tp)
        results.append(r)
        if r["status"] == "OK":
            line = (f"  OK       {label:<20}{sym:>8}{tf:>4} TP={tp}R  "
                     f"e_net={r['e_net']:+.3f}  n={r['n']}  x5={r['x5_e_net']:+.3f}")
        elif r["status"] == "NO_DATA":
            line = f"  NO_DATA  {label:<20}{sym:>8}{tf:>4} -- could not load bars"
        else:
            line = (f"  DEGRADED {label:<20}{sym:>8}{tf:>4} TP={tp}R  "
                     f"e_net={r.get('e_net', 0):+.3f}  n={r.get('n', 0)}  "
                     f"half1={r.get('half1')}  half2={r.get('half2')}  x5={r.get('x5_e_net')}")
        print(line)
        lines.append(line)

    degraded = [r for r in results if r["status"] != "OK"]
    if degraded:
        lines.append(f"\n{len(degraded)}/{len(results)} legs no longer clear the bar "
                      f"(n>={MIN_N}, both-halves-positive, survives {SLIPPAGE_GATE_MULT}x slippage). "
                      f"This does NOT mean auto-remove them -- {MIN_N}-trade statistics move slowly; "
                      f"review before touching the deployed roster.")
    else:
        lines.append(f"\nAll {len(results)} legs still clear the bar on refreshed data.")
    print(lines[-1])

    lines.append("\n[3/3] Live status + full-rescreen due date...")
    print(lines[-1])
    live = check_live_status()
    if live["has_state"]:
        line = (f"  Live state found: {live['trades']} closed trades, halted={live.get('halted')}, "
                f"peak_equity={live.get('peak_equity')}")
        if live.get("halt_reason"):
            line += f", halt_reason={live['halt_reason']}"
        print(line)
        lines.append(line)
        line = f"  MLL-lock threshold ({INITIAL_TRAIL_USD:,.0f}): " + \
               ("CLEARED -- floor should be locked flat per lucidpro_mll_lock_taper.py's mechanic"
                if live["mll_locked"] else "not yet cleared -- MLL still trailing")
        print(line)
        lines.append(line)
    else:
        line = "  No live state file yet -- funded bot has not been started."
        print(line)
        lines.append(line)

    days_since_rescreen = (date.today() - ROSTER_LAST_CHANGED).days
    if days_since_rescreen >= FULL_RESCREEN_DAYS:
        line = (f"  FLAG: {days_since_rescreen} days since the roster was last substantively "
                 f"re-screened (last: {ROSTER_LAST_CHANGED}, threshold: {FULL_RESCREEN_DAYS}d). "
                 f"A full re-run of the 2026-08-14 research (new strategies/timeframes/targets, "
                 f"not just re-checking the current 8 legs) may be due -- this is a multi-hour job, "
                 f"run it deliberately, not automatically.")
    else:
        line = (f"  {days_since_rescreen}/{FULL_RESCREEN_DAYS} days since last full re-screen -- "
                 f"not due yet.")
    print(line)
    lines.append(line)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"review_{date.today().isoformat()}.log"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    # Keep 24 reports (2 years at monthly cadence) -- trend matters, not one snapshot.
    for old in sorted(REPORT_DIR.glob("review_*.log"), reverse=True)[24:]:
        old.unlink(missing_ok=True)

    return 1 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
