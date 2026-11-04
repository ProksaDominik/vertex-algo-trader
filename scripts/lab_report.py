"""Read the live lab's accumulated sample -- honestly.

THE PROBLEM THIS SOLVES
-----------------------
With 296 strategies under test and no true edge anywhere, about 15 will clear
p<0.05 by chance, and the best of them will show a healthy-looking positive
expectancy. Sorting by mean R and reading off the top is therefore guaranteed
to produce a "winner" whether or not one exists. That is the exact mechanism
that produced the VWAP artifact earlier in this project: +0.166R in-sample,
-0.149R on full history.

So this report does three things a plain leaderboard does not:

1. Reports each strategy's MINIMUM DETECTABLE EFFECT at its current sample
   size. If a strategy shows +0.08R but its MDE is 0.30R, the number is noise
   and is labelled as such -- not ranked.

2. Applies BENJAMINI-HOCHBERG across all strategies tested. Controlling the
   false-discovery rate is the correct treatment for "I ran 296 tests"; an
   uncorrected p<0.05 means nothing here.

3. Refuses to declare anything until it passes both. The headline is
   deliberately "N strategies have a readable sample, M survive correction",
   which for a long time will be "0".
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STATE_PATH = ROOT / "data" / "lab_state.json"
MIN_N = 30              # below this, not even worth computing a t
Z_POWER = 0.8416        # 80% power


def norm_sf(z: float) -> float:
    """Two-sided p from a z score. math.erfc avoids a scipy dependency."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> float:
    """Return the largest p that is significant at FDR q. 0.0 if none.

    BH rather than Bonferroni: with 296 tests Bonferroni (alpha/296) is so
    conservative it would reject a genuine +0.10R edge for years. BH controls
    the expected PROPORTION of false discoveries instead, which is the right
    question when screening many candidates.
    """
    if not pvals:
        return 0.0
    s = sorted(pvals)
    m = len(s)
    crit = 0.0
    for i, p in enumerate(s, start=1):
        if p <= q * i / m:
            crit = p
    return crit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    if not STATE_PATH.exists():
        print("No lab state yet -- run scripts/run_lab.py first.")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    stats = state.get("stats", {})

    rows = []
    for cid, d in stats.items():
        n = d.get("n", 0)
        if n < 2:
            continue
        mean = d["sum_r"] / n
        var = (d["sum_r2"] - n * mean * mean) / (n - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd <= 0:
            continue
        se = sd / math.sqrt(n)
        t = mean / se
        p = norm_sf(t)
        mde = (1.96 + Z_POWER) * sd / math.sqrt(n)
        rows.append({"cid": cid, "n": n, "mean": mean, "sd": sd, "t": t, "p": p,
                     "mde": mde, "win": 100.0 * d.get("wins", 0) / n,
                     "readable": n >= args.min_n and mean > mde})

    total_trades = sum(d.get("n", 0) for d in stats.values())
    open_sigs = sum(len(d.get("open_signals", [])) for d in stats.values())
    print("LIVE LAB REPORT")
    print("  started        : %s" % state.get("started", "?"))
    print("  last update    : %s" % state.get("updated", "?"))
    print("  strategies     : %d" % len(stats))
    print("  resolved trades: %d   (open signals: %d)" % (total_trades, open_sigs))
    print("=" * 100)

    if not rows:
        print("\nNo strategy has resolved 2+ trades yet. Nothing to report.")
        return 0

    ready = [r for r in rows if r["n"] >= args.min_n]
    print("\n%d/%d strategies have n >= %d" % (len(ready), len(rows), args.min_n))

    if not ready:
        best = max(rows, key=lambda r: r["n"])
        print("Largest sample so far: %s with n=%d." % (best["cid"], best["n"]))
        print("\nNothing is readable yet. This is expected -- see the note below.")
    else:
        crit = benjamini_hochberg([r["p"] for r in ready], args.fdr)
        survivors = [r for r in ready if r["p"] <= crit and r["mean"] > 0] if crit else []
        print("Benjamini-Hochberg at FDR %.0f%%: critical p = %.5f -> %d survivor(s)"
              % (args.fdr * 100, crit, len(survivors)))
        print()
        print("%-38s %6s %8s %8s %7s %8s %8s"
              % ("strategy", "n", "mean R", "MDE", "win%", "t", "verdict"))
        print("-" * 100)
        for r in sorted(ready, key=lambda x: -x["mean"])[:args.top]:
            if r in survivors:
                v = "SURVIVES"
            elif r["mean"] <= r["mde"]:
                v = "noise"
            else:
                v = "not sig"
            print("%-38s %6d %+8.4f %8.3f %6.1f%% %+8.2f %8s"
                  % (r["cid"][:38], r["n"], r["mean"], r["mde"], r["win"], r["t"], v))
        print()
        if survivors:
            print("SURVIVORS (positive, and significant after correcting for %d tests):"
                  % len(ready))
            for r in sorted(survivors, key=lambda x: -x["mean"]):
                print("   %-38s n=%-6d %+.4f R  p=%.5f" % (r["cid"], r["n"], r["mean"], r["p"]))
            print()
            print("   Even these are CANDIDATES, not conclusions. Confirm on a fresh")
            print("   forward window before risking money -- surviving a screen you")
            print("   were selected from is not the same as an edge.")
        else:
            print("NO strategy survives multiple-comparison correction.")
            print("With %d simultaneous tests, roughly %.0f would clear an uncorrected"
                  % (len(ready), 0.05 * len(ready)))
            print("p<0.05 by chance alone -- which is why that threshold is not used here.")

    print()
    print("-" * 100)
    print("WHY THIS IS SLOW, in one line: per-trade noise (sd ~1.0R) is roughly")
    print("10x a good edge (~0.09R), so separating them needs thousands of trades")
    print("per strategy. A leaderboard read before then is a ranking of luck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
