"""LucidPro 50K FUNDED account: which strategy gets the first $2,000 payout
fastest with the lowest blow rate, and the most $ per week? (2026-10-01)

Rules (lucid_pass_research.FundedRules): equity-based $2,000 MLL trailing the
best daily close, locked at $50,100 once a close reaches $52,100; payout 1 cap
$2,000 then $2,500 (5 payouts, then live review); $52,100 must remain after a
payout; each cycle >= $500 profit, >= 3 days, largest winning day <= 40% of
the cycle's profit; 90/10 split. Rolling starts 2019-26 (full night data),
120-trading-day horizon per funded account.

Candidates: the daily race (long NQ from 19:00 ET, a daily take-profit sized
for the 40% rule, optional own daily stop, out 16:00 ET) and ORB + trend.

Usage:  .venv/Scripts/python.exe scripts/lucid_funded_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402

S = lp.S
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)      # first evaluation + resets until a pass, list prices
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5            # ~1.5 trading days per try


def main() -> int:
    G = {m: lp.load_grid(m) for m in ("NQ", "GC")}
    nq = G["NQ"]
    n = len(nq.dates)
    ok = nq.has[:, S(19)] & nq.has[:, lp.RTH_LAST]
    race = lp.build_leg(nq, "race", np.where(ok, 1, 0), np.full(n, S(19)), np.full(n, lp.RTH_LAST),
                        stop=nq.O[:, S(19)] - 400.0)
    orb_nq, orb_gc = lp.leg_orb_trend(nq), lp.leg_orb_trend(G["GC"])
    dates, al = lp.align([race, orb_nq, orb_gc])
    keep = pd.to_datetime(dates).year >= 2019
    starts = np.arange(0, keep.sum() - 120)
    rows = []

    def run(label, wv, tp=None, stop=None, policy="full"):
        c, w, b = lp.combine(al, wv, with_best=True)
        c, w, b = c[keep], w[keep], b[keep]
        b = b - 0.5 * max(wv)                       # a limit fill needs a tick through
        want = (2000.0, 2500.0, 2500.0, 2500.0, 2500.0) if policy == "full" else (500.0,) * 5
        rules = lp.FundedRules(want=want)
        res = lp.replay_funded(c, w, b, rules, starts, day_tp=tp, day_stop=stop)
        s = lp.summarise_funded(res)
        funded_weeks = s["avg_days_alive"] / 5.0
        s["net_after_eval_cost"] = s["avg_paid_net"] - EVAL_COST
        s["per_week_pipeline"] = s["net_after_eval_cost"] / (EVAL_WEEKS + funded_weeks)
        rows.append({"strategy": label, "policy": policy, **s})

    for size in (10, 20, 30, 40):
        for tp in (500.0, 700.0, 1000.0, 1400.0):
            for stop in (None, 600.0, 1000.0):
                lab = f"race {size // 10} NQ, take ${tp:,.0f}/day" + (f", stop ${stop:,.0f}" if stop else "")
                run(lab, [size, 0, 0], tp=tp, stop=stop)
    for size in (10, 20, 30, 40):
        run(f"race {size // 10} NQ, take $1,000/day", [size, 0, 0], tp=1000.0, policy="any $500+")
    for wv, lab in (([0, 2, 1], "ORB 2 MNQ + 1 MGC (current funded)"), ([0, 6, 3], "ORB 6 MNQ + 3 MGC"),
                    ([0, 12, 6], "ORB 12 MNQ + 6 MGC"), ([0, 20, 10], "ORB 20 MNQ + 10 MGC")):
        run(lab, wv)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "futures" / "funded_study.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["strategy", "policy", "first_payout", "blown_before_payout", "median_days_to_payout", "avg_payouts",
            "avg_paid_net", "net_after_eval_cost", "avg_days_alive", "per_week_pipeline"]
    print(res.sort_values("per_week_pipeline", ascending=False)[cols].head(30).to_string(
        index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nBy highest chance of the first $2,000 payout:")
    print(res.sort_values("first_payout", ascending=False)[cols].head(10).to_string(
        index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nORB (current funded bot) rows:")
    print(res[res.strategy.str.startswith("ORB")][cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
