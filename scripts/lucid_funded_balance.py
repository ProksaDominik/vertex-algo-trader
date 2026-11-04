"""Funded account, owner's brief 2026-10-01: "the perfect balance -- at least a
$500 withdrawal in a small amount of time, one account first, then more".

Searches the daily race (sizes, daily take-profit caps, own stops, entry
times), the adaptive take-profit (only what the next payout still needs) with
a pause once the money is there, payout plans (first $500 / $1,000, then
$500 / $1,000 / $2,500) and race + ORB stacks, under the LucidPro 50K funded
rules (lucid_pass_research.FundedRules). Rolling starts 2019-26, 120 trading
days per funded account. Scored by the chance of a $500+ payout within 10
trading days, then $ per week.

Usage:  .venv/Scripts/python.exe scripts/lucid_funded_balance.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402

S = lp.S
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5
H = 120


def main() -> int:
    G = {m: lp.load_grid(m) for m in ("NQ", "GC")}
    nq = G["NQ"]
    n = len(nq.dates)
    legs = []
    for a in (S(19), S(2), lp.RTH_OPEN):
        ok = nq.has[:, a] & nq.has[:, lp.RTH_LAST]
        legs.append(lp.build_leg(nq, f"race {a}", np.where(ok, 1, 0), np.full(n, a), np.full(n, lp.RTH_LAST),
                                 stop=nq.O[:, a] - 400.0))
    legs += [lp.leg_orb_trend(nq), lp.leg_orb_trend(G["GC"])]
    names = ["race19", "race02", "race0930", "orbNQ", "orbGC"]
    dates, al = lp.align(legs)
    keep = pd.to_datetime(dates).year >= 2019
    starts = np.arange(0, keep.sum() - H)
    rows = []

    def run(label, w: dict, tp=None, stop=None, adaptive=False, pause=False, w1=500.0, w2=500.0):
        wv = [w.get(k, 0) for k in names]
        c, wo, b = lp.combine(al, wv, with_best=True)
        c, wo, b = c[keep], wo[keep], b[keep] - 0.5 * max(wv)
        rules = lp.FundedRules(want=(w1, w2, w2, w2, w2), max_days=H)
        res = lp.replay_funded(c, wo, b, rules, starts, day_tp=tp, day_stop=stop, adaptive=adaptive,
                               pause_when_met=pause)
        s = lp.summarise_funded(res)
        got = res.payouts >= 1
        rows.append({
            "strategy": label, "first/later payout": f"${w1:,.0f}/${w2:,.0f}",
            "payout in 5d": (got & (res.first_day <= 5)).mean(),
            "payout in 10d": (got & (res.first_day <= 10)).mean(),
            "payout in 20d": (got & (res.first_day <= 20)).mean(),
            "any payout": got.mean(),
            "blown before payout": ((res.status == 2) & ~got).mean(),
            "median days": s["median_days_to_payout"],
            "avg payouts": s["avg_payouts"],
            "avg paid (90%)": s["avg_paid_net"],
            "$/week/account": (s["avg_paid_net"] - EVAL_COST) / (EVAL_WEEKS + s["avg_days_alive"] / 5),
        })

    entries = {"race19": "19:00", "race02": "02:00", "race0930": "09:30"}
    for leg, size, cap, stop, ad, w1, w2 in itertools.product(
            ("race19", "race02", "race0930"), (5, 10, 15, 20), (700.0, 1040.0, 1400.0), (None, 600.0),
            ((False, False), (True, True)), (500.0, 1000.0), (500.0, 1000.0, 2500.0)):
        if leg != "race19" and (size not in (10,) or stop):
            continue
        adaptive, pause = ad
        lab = (f"race {size / 10:g} NQ @{entries[leg]}, take<= ${cap:,.0f}" + (" adaptive+pause" if adaptive else "")
               + (f", stop ${stop:,.0f}" if stop else ""))
        run(lab, {leg: size}, tp=cap, stop=stop, adaptive=adaptive, pause=pause, w1=w1, w2=w2)
    for orb, lab in (({"orbNQ": 2, "orbGC": 1}, "ORB 2 MNQ + 1 MGC"), ({"orbNQ": 4, "orbGC": 2}, "ORB 4 MNQ + 2 MGC")):
        for w1, w2 in ((500.0, 500.0), (500.0, 2500.0)):
            run(lab, orb, w1=w1, w2=w2)
            run(lab + " + pause", orb, tp=1e9, adaptive=True, pause=True, w1=w1, w2=w2)
            for size in (5, 10):
                run(f"stack: race {size / 10:g} NQ + {lab}, take<= $1,040 adaptive+pause",
                    {"race19": size, **orb}, tp=1040.0, adaptive=True, pause=True, w1=w1, w2=w2)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "futures" / "funded_balance.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)
    f = lambda v: f"{v:,.2f}"                                              # noqa: E731
    print("Top 20 by chance of a $500+ payout within 10 trading days:")
    print(res.sort_values(["payout in 10d", "$/week/account"], ascending=False).head(20).to_string(
        index=False, float_format=f))
    print("\nTop 10 by $ per week:")
    print(res.sort_values("$/week/account", ascending=False).head(10).to_string(index=False, float_format=f))
    print(f"\n{len(res)} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
