"""LucidFlex vs LucidPro for the race: which plan pays more per month? (2026-10-07)

Owner: "now we need to get to better numbers and higher %". Within LucidPro the
race is at its ceiling, so this compares Lucid's LucidFlex 50K plan (official
help pages + plan listing, 2026-10-07):
  evaluation  $3,000 target, $2,000 EOD-trailing MLL (open equity assumed to
              count, as Lucid confirmed for Pro), largest day <= 50% of the
              profit (so >= 2 days), no daily loss limit; $150 list / ~$105 code
  funded      NO buffer, NO consistency rule; a payout needs 5 days with >= $150
              profit in the cycle and positive cycle profit; payout = 50% of the
              cycle profit, max $2,000; 5 payouts, then live; 90% to the trader
against LucidPro 50K as deployed (evaluation race 2 NQ, funded race 1 NQ with
the $52,100 buffer and 40% rule).

Both run the same race leg (long from 19:00 ET, out 16:00 ET, 400-pt safety
stop) on the 2016-2026 proxy history; starts 2019-2026, 120 trading days per
account; adverse move first inside a bar; the take-profit needs a one-tick
trade-through. Flex funded: race a fixed daily take T (grid), stop racing in a
cycle once it has 5 good days, payout at that day's close.

Usage:  .venv/Scripts/python.exe scripts/lucid_flex_study.py
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
import lucid_pipeline_money as pm  # noqa: E402

K = lp.K
H = 120
TICK_PER_MICRO = 0.5            # 0.25 pt x $2
RNG = np.random.default_rng(17)


def race_leg():
    g = lp.load_grid("NQ")
    n = len(g.dates)
    a, b = lp.S(19), lp.RTH_LAST
    ok = g.has[:, a] & g.has[:, b]
    leg = lp.build_leg(g, "race", np.where(ok, 1, 0), np.full(n, a), np.full(n, b), stop=g.O[:, a] - 400.0)
    keep = np.asarray(pd.to_datetime(g.dates).year >= 2019)
    yrs = np.asarray(pd.to_datetime(g.dates).year)[keep]
    return leg.close[keep], leg.worst[keep], leg.best[keep], yrs


def first(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(1), mask.argmax(1), K)


def day(b0, floor, tp, C, W, B):
    """One race day per active start: (pnl, blown, hit_take)."""
    f_m = first(b0[:, None] + W <= floor[:, None])
    f_t = first(B >= tp[:, None])
    blown = (f_m <= f_t) & (f_m < K)
    hit = ~blown & (f_t < K)
    return np.where(blown, floor - b0, np.where(hit, tp, C[:, -1])), blown, hit


def flex_eval(c, w, b, starts, micros, day_cap=1500.0, target=3000.0, mll=2000.0, lock=100.0, floor_tp=150.0):
    m = len(starts)
    n = c.shape[0]
    bal, hwm, best = np.zeros(m), np.zeros(m), np.zeros(m)
    status, days = np.zeros(m, int), np.zeros(m, int)
    for step in range(H):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W = micros * c[d[ia]], micros * w[d[ia]]
        B = micros * b[d[ia]] - TICK_PER_MICRO * micros
        floor = np.minimum(hwm[ia] - mll, lock)
        need = np.maximum(target - bal[ia], np.where(best[ia] > 0, best[ia] / 0.5 - bal[ia], 0.0))
        tp = np.clip(need, floor_tp, day_cap)
        pnl, blown, hit = day(bal[ia], floor, tp, C, W, B)
        days[ia] += 1
        nb = bal[ia] + pnl
        nbest = np.maximum(best[ia], pnl)
        passed = ~blown & (nb >= target) & (nbest <= 0.5 * nb)
        status[ia[blown]] = 2
        status[ia[passed]] = 1
        bal[ia], best[ia] = nb, nbest
        hwm[ia] = np.maximum(hwm[ia], nb)
    return pd.DataFrame({"status": status, "days": days})


def flex_funded(c, w, b, starts, micros, take, good=150.0, mll=2000.0, lock=100.0, need_days=5,
                share=0.5, cap=2000.0, max_payouts=5):
    m = len(starts)
    n = c.shape[0]
    bal, hwm = np.zeros(m), np.zeros(m)
    cyc_pnl, cyc_good = np.zeros(m), np.zeros(m, int)
    paid, npay = np.zeros(m), np.zeros(m, int)
    status, days, first_day = np.zeros(m, int), np.zeros(m, int), np.full(m, -1)
    for step in range(H):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W = micros * c[d[ia]], micros * w[d[ia]]
        B = micros * b[d[ia]] - TICK_PER_MICRO * micros
        floor = np.minimum(hwm[ia] - mll, lock)
        tp = np.full(len(ia), float(take))
        pnl, blown, _ = day(bal[ia], floor, tp, C, W, B)
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


def pro_outcomes(c, w, b, starts):
    ev = lp.replay(20 * c, 20 * w, lp.Rules(max_days=H, pass_on="best", mll_on="equity", dll_on="balance",
                                            lock=100.0, **lp.PLANS["LucidPro 50K"]), starts, best=20 * b - 10)
    fr = lp.replay_funded(10 * c, 10 * w, 10 * b - 5, lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0),
                                                                    max_days=H),
                          starts, day_tp=1040.0, adaptive=True, pause_when_met=True)
    return ev, fr


def month_money(ev, fr, price_eval, price_retry, months=12, split=0.9, sims=20000):
    """Steady-state money per month for one account slot (a year simulated, divided by 12)."""
    E_s, E_d = ev.status.to_numpy(), ev.days.to_numpy()
    F_p, F_d, F_f = fr.paid.to_numpy(), fr.days.to_numpy(), fr.first_day.to_numpy()
    year = 252
    out = np.empty(sims)
    for k in range(sims):
        t, fees, paid, fresh = 0, price_eval, 0.0, True
        while t < year:
            if not fresh:
                fees += price_retry
            fresh = False
            i = RNG.integers(len(E_s))
            t += max(1, int(E_d[i]))
            if E_s[i] != 1 or t >= year:
                continue
            j = RNG.integers(len(F_p))
            left = year - t
            if F_p[j] > 0 and F_f[j] <= left:
                paid += F_p[j] * split * (1.0 if F_d[j] <= left else min(1.0, left / max(F_d[j], 1)))
            t += max(1, int(F_d[j]))
            if t < year:
                fees += price_eval
                fresh = True
        out[k] = (paid - fees) / months
    return out


def summary(fr: pd.DataFrame) -> dict:
    got = fr.payouts >= 1
    return {"payout %": got.mean(), "within 10 days %": (got & (fr.first_day <= 10)).mean(),
            "blown before payout %": ((fr.status == 2) & ~got).mean(),
            "median days to 1st": float(fr.first_day[got].median()) if got.any() else np.nan,
            "avg paid (gross)": fr.paid.mean(), "avg to you": 0.9 * fr.paid.mean()}


def main() -> int:
    c, w, b, yrs = race_leg()
    starts = np.arange(0, len(c) - H)
    sy = yrs[starts]
    halves = {"2019-22": sy <= 2022, "2023-26": sy >= 2023}
    ev_pro, fr_pro = pro_outcomes(c, w, b, starts)
    print("LucidPro 50K (deployed): evaluation pass {:.1%} | funded: {}".format(
        (ev_pro.status == 1).mean(), {k: round(v, 3) for k, v in summary(fr_pro).items()}))
    rows = []
    for micros in (10, 20):
        ev = flex_eval(c, w, b, starts, micros)
        rows.append({"what": f"Flex evaluation, {micros // 10} NQ", "pass %": (ev.status == 1).mean(),
                     **{f"pass % {h}": (ev.status[mk] == 1).mean() for h, mk in halves.items()},
                     "avg days": ev.days[ev.status == 1].mean()})
    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fl = []
    for micros, take in itertools.product((5, 10), (160, 200, 250, 300, 400, 500, 750, 1000)):
        fr = flex_funded(c, w, b, starts, micros, take)
        s = summary(fr)
        fl.append({"size": f"{micros / 10:g} NQ", "take $/day": take, **s,
                   **{f"payout % {h}": (fr.payouts[mk] >= 1).mean() for h, mk in halves.items()},
                   "_fr": fr})
    ft = pd.DataFrame(fl)
    pd.set_option("display.width", 250)
    print("\nLucidFlex funded, by daily take (5 good days of >= $150 per payout, 50% of the cycle profit paid):")
    print(ft.drop(columns="_fr").to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    # money per month: the deployed Pro pipeline vs Flex with its best funded setting and best eval size
    ev_flex = flex_eval(c, w, b, starts, 20)
    best = ft.iloc[ft["avg to you"].idxmax()]
    print(f"\nbest Flex funded setting by money: {best['size']}, take ${best['take $/day']}")
    res = []
    for label, ev, fr, pe, pr in (("LucidPro 50K, list ($185 / $120 reset)", ev_pro, fr_pro, 185.0, 120.0),
                                  ("LucidPro 50K, ~30% off ($130 / $84)", ev_pro, fr_pro, 130.0, 84.0),
                                  ("LucidFlex 50K, list ($150 each try)", ev_flex, best["_fr"], 150.0, 150.0),
                                  ("LucidFlex 50K, code ($105 each try)", ev_flex, best["_fr"], 105.0, 105.0)):
        mm = month_money(ev, fr, pe, pr)
        res.append({"plan": label, "avg $/month": mm.mean(), "median $/month": np.median(mm),
                    "months positive %": (mm > 0).mean()})
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:,.0f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
