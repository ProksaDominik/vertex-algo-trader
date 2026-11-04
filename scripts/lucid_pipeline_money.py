"""Is the whole Lucid pipeline profitable, and by how much? (2026-10-01)

One "account slot" run for a year: buy a LucidPro 50K evaluation, race it
(2 NQ, confirmed rules) with resets after fails until it passes, then run the
funded race (1 NQ from 19:00 ET, take what the next payout still needs <= $1,040,
pause once met; first payout $500, later $2,500, 90/10 split) until the funded
account blows or takes 5 payouts -- then buy a new evaluation. Outcomes of each
evaluation attempt and each funded account are drawn from the replays of every
start day 2019-26. Fees: list prices and ~30% off (discount codes).

Usage:  .venv/Scripts/python.exe scripts/lucid_pipeline_money.py
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
YEAR = 252
RNG = np.random.default_rng(7)


def outcomes():
    g = lp.load_grid("NQ")
    n = len(g.dates)
    a, b = S(19), lp.RTH_LAST
    ok = g.has[:, a] & g.has[:, b]
    leg = lp.build_leg(g, "race", np.where(ok, 1, 0), np.full(n, a), np.full(n, b), stop=g.O[:, a] - 400.0)
    keep = pd.to_datetime(g.dates).year >= 2019
    starts = np.arange(0, keep.sum() - 120)
    # evaluation attempts: 2 NQ, Lucid's confirmed rules, MLL lock
    c, w, bb = 20 * leg.close[keep], 20 * leg.worst[keep], 20 * leg.best[keep] - 10
    ev = lp.replay(c, w, lp.Rules(max_days=120, pass_on="best", mll_on="equity", dll_on="balance", lock=100.0,
                                  **lp.PLANS["LucidPro 50K"]), starts, best=bb)
    # funded accounts: 1 NQ, adaptive take <= $1,040 + pause, $500 then $2,500
    c, w, bb = 10 * leg.close[keep], 10 * leg.worst[keep], 10 * leg.best[keep] - 5
    fr = lp.replay_funded(c, w, bb, lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0), max_days=120),
                          starts, day_tp=1040.0, adaptive=True, pause_when_met=True)
    return ev, fr


def simulate_year(ev, fr, price_eval, price_reset, split=0.9):
    t, fees, paid, payouts, funded = 0, 0.0, 0.0, 0, 0
    fees += price_eval
    while t < YEAR:
        e = ev.iloc[RNG.integers(len(ev))]
        t += max(1, int(e.days))
        if e.status != 1:                       # failed (or never resolved): reset and race again
            if t < YEAR:
                fees += price_reset
            continue
        funded += 1
        f = fr.iloc[RNG.integers(len(fr))]
        # payouts land inside the year only up to the days left
        days_left = YEAR - t
        if f.payouts > 0 and f.first_day <= days_left:
            paid += f.paid * split if f.days <= days_left else f.paid * split * min(1.0, days_left / max(f.days, 1))
            payouts += int(f.payouts)
        t += max(1, int(f.days))
        if t < YEAR:
            fees += price_eval                  # funded account over -> a new evaluation
    return paid - fees, paid, fees, payouts, funded


def main() -> int:
    ev, fr = outcomes()
    print(f"evaluation attempt: pass {(ev.status == 1).mean():.1%}, avg {ev.days.mean():.1f} trading days")
    got = fr.payouts >= 1
    print(f"funded account: gets paid {got.mean():.1%}, avg paid (gross) ${fr.paid.mean():,.0f}, "
          f"avg life {fr.days.mean():.1f} trading days, payouts when paid: {fr.payouts[got].mean():.2f}")
    for label, pe, pr in (("list prices ($185 eval, $120 reset)", 185.0, 120.0),
                          ("~30% off ($130 eval, $84 reset)", 130.0, 84.0)):
        res = np.array([simulate_year(ev, fr, pe, pr) for _ in range(20000)])
        net, paid, fees, npay, nfund = res.T
        month = net / 12
        print(f"\n=== one account slot for a year, {label} ===")
        print(f"  net profit: average ${net.mean():,.0f} (= ${net.mean() / 52:,.0f}/week, ${month.mean():,.0f}/month)")
        print(f"  middle outcome (median) ${np.median(net):,.0f} | worst 10% below ${np.percentile(net, 10):,.0f}"
              f" | best 10% above ${np.percentile(net, 90):,.0f}")
        print(f"  chance the year ends in a loss: {(net < 0).mean():.1%}")
        print(f"  per year on average: {nfund.mean():.1f} funded accounts, {npay.mean():.1f} payouts, "
              f"fees ${fees.mean():,.0f}, paid to you ${paid.mean():,.0f}")
    # a single month for one slot (cash-flow feel)
    res_m = []
    for _ in range(20000):
        global YEAR
        YEAR = 21
        res_m.append(simulate_year(ev, fr, 185.0, 120.0)[0])
    YEAR = 252
    res_m = np.array(res_m)
    print(f"\none month, one slot, list prices: average ${res_m.mean():,.0f}, "
          f"loss in {(res_m < 0).mean():.0%} of months, median ${np.median(res_m):,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
