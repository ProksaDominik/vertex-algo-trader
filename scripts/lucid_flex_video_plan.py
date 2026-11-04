"""The owner's twelfth video (2026-10-08): a LucidFlex 50K risk plan, replayed against our LucidPro pipeline.

His plan, made exact (the same race leg as every Lucid study: long NQ from 19:00 New York, out 16:00, the
"stop" is Lucid's own loss line = the full $2,000 drawdown):
  evaluation  risk the full drawdown for +$1,500 a day (Flex's 50% consistency rule caps a day at
              $1,500 of the $3,000 target) -> two winning days pass. (= lucid_flex_study.flex_eval)
  funded      first go for +$3,400 (if a day ends short, the rest is carried to the next day), then
              +$150 a day until the cycle has 5 days of >= $150; Flex then pays 50% of the cycle profit,
              max $2,000 ($3,400 + 4 x $150 = $4,000 -> the max $2,000 payout); repeat.
Contract size is not in the video (only the dollar risk and target), so sizes from 0.5 to 4 NQ are swept.
Money per month for one account slot as in lucid_flex_study.month_money (12 months simulated, 90% split).
Replays of every start day 2019-2026.

Usage:  .venv/Scripts/python.exe scripts/lucid_flex_video_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_flex_study as fs  # noqa: E402


def funded_schedule(c, w, b, starts, micros, big=3400.0, small=150.0, good=150.0, mll=2000.0, lock=100.0,
                    need_days=5, share=0.5, cap=2000.0, max_payouts=5):
    m, n = len(starts), c.shape[0]
    bal, hwm = np.zeros(m), np.zeros(m)
    cyc_pnl, cyc_good = np.zeros(m), np.zeros(m, int)
    paid, npay = np.zeros(m), np.zeros(m, int)
    status, days, first_day = np.zeros(m, int), np.zeros(m, int), np.full(m, -1)
    for step in range(fs.H):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W = micros * c[d[ia]], micros * w[d[ia]]
        B = micros * b[d[ia]] - fs.TICK_PER_MICRO * micros
        floor = np.minimum(hwm[ia] - mll, lock)
        tp = np.where(cyc_pnl[ia] < big, np.maximum(big - cyc_pnl[ia], small), small)
        pnl, blown, _ = fs.day(bal[ia], floor, tp, C, W, B)
        days[ia] += 1
        status[ia[blown]] = 2
        bal[ia] += pnl
        hwm[ia] = np.maximum(hwm[ia], bal[ia])
        cyc_pnl[ia] += pnl
        cyc_good[ia] += (pnl >= good).astype(int)
        ok = (status[ia] == 0) & (cyc_good[ia] >= need_days) & (cyc_pnl[ia] > 0)
        if ok.any():
            j = ia[ok]
            amt = np.minimum(share * cyc_pnl[j], cap)
            bal[j] -= amt
            paid[j] += amt
            npay[j] += 1
            first_day[j] = np.where(first_day[j] < 0, days[j], first_day[j])
            cyc_pnl[j], cyc_good[j] = 0.0, 0
            status[j[npay[j] >= max_payouts]] = 3
    return pd.DataFrame({"status": status, "days": days, "paid": paid, "payouts": npay, "first_day": first_day})


def main() -> int:
    c, w, b, yrs = fs.race_leg()
    starts = np.arange(0, len(c) - fs.H)
    ev_pro, fr_pro = fs.pro_outcomes(c, w, b, starts)
    pd.set_option("display.width", 220)
    rows = []
    evs = {}
    for micros in (5, 10, 20, 40):
        ev = fs.flex_eval(c, w, b, starts, micros)
        evs[micros] = ev
        rows.append({"his Flex evaluation, size": f"{micros / 10:g} NQ", "pass %": (ev.status == 1).mean(),
                     "avg days to pass": ev.days[ev.status == 1].mean()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"(our LucidPro race evaluation: pass {(ev_pro.status == 1).mean():.1%})")
    fr_rows = []
    frs = {}
    for micros in (5, 10, 20, 40):
        fr = funded_schedule(c, w, b, starts, micros)
        frs[micros] = fr
        fr_rows.append({"his Flex funded, size": f"{micros / 10:g} NQ", **fs.summary(fr)})
    fr_rows.append({"his Flex funded, size": "OUR LucidPro funded bot", **fs.summary(fr_pro)})
    print("\n" + pd.DataFrame(fr_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    best_ev = max(evs, key=lambda k: (evs[k].status == 1).mean())
    best_fr = max(frs, key=lambda k: frs[k].paid.mean())
    res = []
    for label, ev, fr, pe, pr in (
            ("OUR LucidPro race, list ($185 eval, $120 reset)", ev_pro, fr_pro, 185.0, 120.0),
            ("OUR LucidPro race, ~30% off ($130 / $84)", ev_pro, fr_pro, 130.0, 84.0),
            (f"HIS LucidFlex plan, list ($150 a try), eval {best_ev / 10:g} NQ, funded {best_fr / 10:g} NQ",
             evs[best_ev], frs[best_fr], 150.0, 150.0),
            ("HIS LucidFlex plan, ~30% off ($105 a try)", evs[best_ev], frs[best_fr], 105.0, 105.0)):
        mm = fs.month_money(ev, fr, pe, pr)
        res.append({"plan": label, "avg $/month": mm.mean(), "median $/month": np.median(mm),
                    "months positive %": (mm > 0).mean()})
    print("\nmoney per month, one account slot (after fees, 90% split):")
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:,.0f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
