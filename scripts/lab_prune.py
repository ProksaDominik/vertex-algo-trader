"""Remove genuinely dead lab strategies -- on FREQUENCY, never on performance.

TWO KINDS OF "USELESS", ONLY ONE OF WHICH IS KNOWABLE EARLY
-----------------------------------------------------------
A strategy can be useless because it never fires, or because it fires and
loses. These need completely different evidence.

  NEVER FIRES  is a mechanical fact. Once the lab has covered every session and
               every day of the week a few times over, a strategy with no
               signals will never produce any, and keeping it costs compute
               while adding nothing. This is safe to prune, and this script
               does exactly that.

  FIRES AND    is a statistical claim, and an early one is almost always wrong.
  LOSES        Per-trade noise is ~1.0R against edges around 0.09R, so a
               strategy needs thousands of trades before its mean means
               anything. Pruning the "bad" ones after days would delete
               genuinely good strategies having an unlucky week and keep bad
               ones having a lucky one -- and worse, it turns the survivors into
               a selected set, which is exactly how a backtest manufactures a
               winner out of noise. This script therefore REFUSES to prune on
               performance at all.

THE SESSION TRAP
----------------
The obvious frequency rule -- "no signals, delete it" -- is wrong if applied
too early. ORB_LDNopen fires once a day at the 07:00 London open; ORB_NYopen at
13:00. Start the lab at 11:50 and both look stone dead, along with every 1h
strategy that has seen two bars. Forty minutes in, 152 of 296 strategies had no
signal and essentially none of them were actually dead.

So pruning requires MIN_DAYS of observation (default 7), which covers every
session window and every weekday several times. Below that the script exits
without touching anything, whatever the flags say.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "lab_state.json"

MIN_DAYS = 7.0          # must cover every session window several times over


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually write the pruned state (default is a dry run)")
    ap.add_argument("--min-days", type=float, default=MIN_DAYS)
    ap.add_argument("--min-signals", type=int, default=1,
                    help="keep a strategy if it produced at least this many signals")
    args = ap.parse_args()

    if not STATE_PATH.exists():
        print("No lab state. Nothing to prune.")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    stats = state.get("stats", {})
    started = state.get("started")
    try:
        days = (datetime.now(timezone.utc)
                - datetime.fromisoformat(started)).total_seconds() / 86400
    except Exception:
        days = 0.0

    print("LAB PRUNE")
    print("  strategies : %d" % len(stats))
    print("  running    : %.2f days (minimum for pruning: %.1f)" % (days, args.min_days))
    print()

    if days < args.min_days:
        print("REFUSING TO PRUNE -- only %.2f days of observation." % days)
        print()
        print("  Session strategies fire in one window per day. ORB_LDNopen needs the")
        print("  07:00 UTC open, ORB_NYopen the 13:00 open, and a 1h strategy sees 24")
        print("  bars a day. Before %.0f days, 'no signals' mostly means 'the lab has"
              % args.min_days)
        print("  not been running long enough to see its window', not 'this is dead'.")
        print()
        dead_now = sum(1 for v in stats.values()
                       if v.get("n", 0) + len(v.get("open_signals", [])) == 0)
        print("  (%d strategies currently show no signal. Come back in %.1f days.)"
              % (dead_now, args.min_days - days))
        return 0

    dead, alive = [], []
    for cid, v in stats.items():
        total = v.get("n", 0) + len(v.get("open_signals", []))
        (alive if total >= args.min_signals else dead).append(cid)

    print("  alive : %d" % len(alive))
    print("  dead  : %d  (zero signals in %.1f days)" % (len(dead), days))
    if dead:
        print()
        print("  dead by family:")
        for fam, n in Counter(c.split("|")[0] for c in dead).most_common():
            print("     %-18s %d" % (fam, n))
        print()
        print("  first 15:")
        for c in dead[:15]:
            print("     %s" % c)

    if not args.apply:
        print()
        print("DRY RUN -- nothing written. Re-run with --apply to remove the dead set.")
        return 0
    if not dead:
        print("\nNothing to remove.")
        return 0

    backup = STATE_PATH.with_suffix(".prebackup_%s.json"
                                    % datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(STATE_PATH, backup)
    for cid in dead:
        stats.pop(cid, None)
    state["stats"] = stats
    state.setdefault("pruned", []).append(
        {"at": datetime.now(timezone.utc).isoformat(), "removed": len(dead),
         "after_days": round(days, 2)})
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    print()
    print("Removed %d strategies. Backup: %s" % (len(dead), backup.name))
    print("Restart run_lab.py so the cohort matches -- it rebuilds from the registry,")
    print("so also drop them from lab_registry.FAMILIES or they will simply return.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
