"""The owner's fourteenth video (2026-10-08): a LucidBlack 50K risk plan vs our LucidPro race.

His plan (the same race leg as every Lucid study: long NQ 19:00 -> 16:00 New York; the "stop" is Lucid's
own loss line = the full $2,000 drawdown):
  evaluation  LucidBlack 50K: target $3,000, $2,000 end-of-day trailing loss limit, 60% consistency (no
              day above 60% of the profit). He risks the drawdown for +$1,515 a day -> two wins pass.
  funded      40% consistency, a payout needs $3,000 profit in the cycle; the first payout is capped at
              $1,500, later ones at $3,000. He goes for +$1,050 a day (three wins = $3,150). ASSUMED (the
              video is not precise): a payout is half the cycle's profit up to the cap ("make 3,000,
              withdraw half"), the loss limit trails end-of-day balances and locks at +$100 like LucidPro,
              at most 5 payouts.
Prices: $126 with the code shown in the video, ~$180 list, per try (no separate reset price given).
Contract size is not given (only dollars), so 0.5-4 NQ are swept. Money per month as in
lucid_flex_study.month_money. Replays of every start day 2019-2026.

Usage:  .venv/Scripts/python.exe scripts/lucid_black_video_plan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_flex_study as fs  # noqa: E402
import lucid_flex_video_plan as fv  # noqa: E402


def black_eval(c, w, b, starts, micros, take=1515.0, target=3000.0, cons=0.6, mll=2000.0, lock=100.0):
    m, n = len(starts), c.shape[0]
    bal, hwm, best = np.zeros(m), np.zeros(m), np.zeros(m)
    status, days = np.zeros(m, int), np.zeros(m, int)
    for step in range(fs.H):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W = micros * c[d[ia]], micros * w[d[ia]]
        B = micros * b[d[ia]] - fs.TICK_PER_MICRO * micros
        floor = np.minimum(hwm[ia] - mll, lock)
        need = np.maximum(target - bal[ia], np.where(best[ia] > 0, best[ia] / cons - bal[ia], 0.0))
        tp = np.clip(need, 150.0, take)
        pnl, blown, _ = fs.day(bal[ia], floor, tp, C, W, B)
        days[ia] += 1
        nb, nbest = bal[ia] + pnl, np.maximum(best[ia], pnl)
        passed = ~blown & (nb >= target) & (nbest <= cons * nb)
        status[ia[blown]] = 2
        status[ia[passed]] = 1
        bal[ia], best[ia] = nb, nbest
        hwm[ia] = np.maximum(hwm[ia], nb)
    return pd.DataFrame({"status": status, "days": days})


def black_funded(c, w, b, starts, micros, take=1050.0, need=3000.0, cons=0.4, caps=(1500.0, 3000.0),
                 share=0.5, mll=2000.0, lock=100.0, max_payouts=5):
    m, n = len(starts), c.shape[0]
    bal, hwm = np.zeros(m), np.zeros(m)
    cyc, best = np.zeros(m), np.zeros(m)
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
        tp = np.full(len(ia), take)
        pnl, blown, _ = fs.day(bal[ia], floor, tp, C, W, B)
        days[ia] += 1
        status[ia[blown]] = 2
        bal[ia] += pnl
        hwm[ia] = np.maximum(hwm[ia], bal[ia])
        cyc[ia] += pnl
        best[ia] = np.maximum(best[ia], pnl)
        ok = (status[ia] == 0) & (cyc[ia] >= need) & (best[ia] <= cons * cyc[ia])
        if ok.any():
            j = ia[ok]
            cap = np.where(npay[j] == 0, caps[0], caps[1])
            amt = np.minimum(share * cyc[j], cap)
            bal[j] -= amt
            paid[j] += amt
            npay[j] += 1
            first_day[j] = np.where(first_day[j] < 0, days[j], first_day[j])
            cyc[j], best[j] = 0.0, 0.0
            status[j[npay[j] >= max_payouts]] = 3
    return pd.DataFrame({"status": status, "days": days, "paid": paid, "payouts": npay, "first_day": first_day})


def main() -> int:
    c, w, b, _ = fs.race_leg()
    starts = np.arange(0, len(c) - fs.H)
    ev_pro, fr_pro = fs.pro_outcomes(c, w, b, starts)
    pd.set_option("display.width", 220)
    evs, frs, rows, frows = {}, {}, [], []
    for micros in (5, 10, 20, 40):
        evs[micros] = ev = black_eval(c, w, b, starts, micros)
        rows.append({"his LucidBlack evaluation, size": f"{micros / 10:g} NQ", "pass %": (ev.status == 1).mean(),
                     "avg days to pass": ev.days[ev.status == 1].mean()})
        frs[micros] = fr = black_funded(c, w, b, starts, micros)
        frows.append({"his LucidBlack funded, size": f"{micros / 10:g} NQ", **fs.summary(fr)})
    frows.append({"his LucidBlack funded, size": "OUR LucidPro funded bot", **fs.summary(fr_pro)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"(our LucidPro race evaluation: pass {(ev_pro.status == 1).mean():.1%})\n")
    print(pd.DataFrame(frows).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    be = max(evs, key=lambda k: (evs[k].status == 1).mean())
    bf = max(frs, key=lambda k: frs[k].paid.mean())
    flex_ev = fs.flex_eval(c, w, b, starts, 10)
    flex_fr = fv.funded_schedule(c, w, b, starts, 10)
    res = []
    for label, ev, fr, pe, pr in (
            ("OUR LucidPro race, list ($185 / $120 reset)", ev_pro, fr_pro, 185.0, 120.0),
            ("OUR LucidPro race, ~30% off ($130 / $84)", ev_pro, fr_pro, 130.0, 84.0),
            (f"HIS LucidBlack plan, list (~$180 a try), eval {be / 10:g} NQ, funded {bf / 10:g} NQ", evs[be], frs[bf], 180.0, 180.0),
            ("HIS LucidBlack plan, with his code ($126 a try)", evs[be], frs[bf], 126.0, 126.0),
            ("video 12's LucidFlex plan, ~30% off ($105 a try)", flex_ev, flex_fr, 105.0, 105.0)):
        mm = fs.month_money(ev, fr, pe, pr)
        res.append({"plan": label, "avg $/month": mm.mean(), "median $/month": np.median(mm),
                    "years ending in profit %": (mm > 0).mean()})
    print("\nmoney per month, one account slot (after fees, 90% split):")
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:,.0f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
