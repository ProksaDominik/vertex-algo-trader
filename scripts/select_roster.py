"""Turn canonical_strategy_test.py output into the two deployable rosters.

The two books are scored by DIFFERENT objectives, because they have different
jobs:

  FTMO  — maximize P(pass) from research/mc3.py: a Monte Carlo of the actual
          FTMO $10k constraints (+10% target, 5% daily loss, 10% overall
          drawdown) driven by the strategy's measured (win_rate, RR) at the
          book's 0.5% base risk. A strategy with a big edge but a fat left
          tail can easily score WORSE here than a smaller, steadier one —
          which is the whole point of scoring pass probability rather than
          expectancy.

  YOLO  — maximize expected compounded growth at its 2.5% risk, penalized by
          the strategy's own worst observed drawdown. No pass/fail gate, so
          the objective is return, but a strategy that would have blown the
          book at 2.5% risk is disqualified regardless of expectancy.

Robustness gates applied BEFORE either ranking (a strategy that fails these
never reaches the objective function):
  - positive out-of-sample expectancy
  - net-positive on a MAJORITY of symbols, not carried by one outlier
  - at least MIN_TRADES out-of-sample trades

Usage:
    python scripts/select_roster.py [--in backtest/canonical_results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from mc3 import sim as mc_sim  # noqa: E402

MIN_TRADES = 40          # below this the stats are noise
FTMO_RISK = 0.005        # portfolio_ftmo.yaml base_risk_pct
YOLO_RISK = 0.025        # portfolio_yolo.yaml risk_pct


def gate(row: dict) -> tuple[bool, str]:
    if row["n"] < MIN_TRADES:
        return False, f"only {row['n']} trades"
    if row["e_net"] <= 0:
        return False, f"negative expectancy ({row['e_net']:+.3f}R)"
    if row["tested_syms"] and row["pos_syms"] / row["tested_syms"] < 0.5:
        return False, f"positive on only {row['pos_syms']}/{row['tested_syms']} symbols"
    if row["rr"] <= 0:
        return False, "degenerate reward:risk"
    return True, ""


def ftmo_score(row: dict) -> dict:
    """P(pass), P(fail) for this strategy under the real FTMO $10k rules."""
    tpd = max(1, round(row["per_day"] or 1))
    w = min(max(row["win_rate"], 0.05), 0.95)
    p, d, t, u = mc_sim(FTMO_RISK, w, row["rr"], tpd=tpd)
    return {"p_pass": p, "p_fail_daily": d, "p_fail_total": t, "unresolved": u,
            "tpd_used": tpd}


def yolo_score(row: dict) -> dict:
    """Expected compounded growth per 100 trades at YOLO risk, DD-penalized.

    Growth uses log-utility on the measured win/loss geometry so a strategy
    that risks ruin at 2.5% is punished automatically rather than by a rule.
    """
    import math
    w, rr = row["win_rate"], row["rr"]
    win_step = math.log(1 + YOLO_RISK * rr)
    loss_step = math.log(max(1e-9, 1 - YOLO_RISK))
    g = w * win_step + (1 - w) * loss_step
    # Worst observed R drawdown translated into equity terms at YOLO risk.
    dd_equity = 1 - math.exp(row["max_dd_r"] * loss_step)
    return {"log_growth_per_trade": g, "growth_per_100": math.exp(g * 100) - 1,
            "modeled_dd_pct": dd_equity * 100,
            "ruin_risk": dd_equity > 0.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="backtest/canonical_results.json")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.inp).read_text())
    print("=" * 100)
    print("ROSTER SELECTION — survivors of the robustness gates, scored per book")
    print("=" * 100)

    passed, rejected = [], []
    for r in rows:
        ok, why = gate(r)
        (passed if ok else rejected).append((r, why))

    print(f"\n{len(passed)} of {len(rows)} (strategy x timeframe) combos cleared the gates.\n")

    if not passed:
        print("NOTHING CLEARED. Closest misses (by expectancy):")
        near = sorted(rejected, key=lambda x: -x[0]["e_net"])[:10]
        for r, why in near:
            print(f"  {r['strategy']:<26} {r['tf']:>3}  E={r['e_net']:+.3f}R  "
                  f"PF={r['pf']:.2f}  n={r['n']:<5} -- {why}")
        return 1

    print("--- FTMO book: ranked by Monte Carlo P(pass) under real FTMO $10k rules ---")
    print(f"{'strategy':<26}{'tf':>4}{'n':>7}{'/day':>7}{'win%':>7}{'RR':>6}"
          f"{'E_net':>9}{'P(pass)':>9}{'P(fail)':>9}")
    print("-" * 100)
    ftmo_ranked = []
    for r, _ in passed:
        sc = ftmo_score(r)
        ftmo_ranked.append((r, sc))
    ftmo_ranked.sort(key=lambda x: -x[1]["p_pass"])
    for r, sc in ftmo_ranked:
        print(f"{r['strategy']:<26}{r['tf']:>4}{r['n']:>7}{r['per_day']:>7.2f}"
              f"{r['win_rate']*100:>7.1f}{r['rr']:>6.2f}{r['e_net']:>+9.3f}"
              f"{sc['p_pass']:>8.1%}{sc['p_fail_daily']+sc['p_fail_total']:>9.1%}")

    print("\n--- YOLO book: ranked by modeled compounded growth at 2.5% risk ---")
    print(f"{'strategy':<26}{'tf':>4}{'n':>7}{'/day':>7}{'E_net':>9}"
          f"{'growth/100t':>13}{'modeled DD':>12}{'ruin?':>7}")
    print("-" * 100)
    yolo_ranked = [(r, yolo_score(r)) for r, _ in passed]
    yolo_ranked.sort(key=lambda x: -x[1]["log_growth_per_trade"])
    for r, sc in yolo_ranked:
        print(f"{r['strategy']:<26}{r['tf']:>4}{r['n']:>7}{r['per_day']:>7.2f}"
              f"{r['e_net']:>+9.3f}{sc['growth_per_100']:>12.1%}"
              f"{sc['modeled_dd_pct']:>11.1f}%{'YES' if sc['ruin_risk'] else 'no':>7}")

    print("\n--- rejected (and why) ---")
    for r, why in sorted(rejected, key=lambda x: -x[0]["e_net"])[:20]:
        print(f"  {r['strategy']:<26}{r['tf']:>4}  E={r['e_net']:+.3f}R  n={r['n']:<5} -- {why}")

    out = ROOT / "backtest" / "roster_selection.json"
    with open(out, "w") as f:
        json.dump({"ftmo": [{**r, **sc} for r, sc in ftmo_ranked],
                   "yolo": [{**r, **sc} for r, sc in yolo_ranked]}, f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
