"""Big funded-account search (owner, 2026-10-01: "make a big research, do anything
you can"): the fastest $500+ withdrawal with the lowest blow rate.

Grid: markets NQ / ES / GC x direction long / trend x 8 entry times x 4 sizes x
daily take-profit cap $700 / $1,040 (adaptive: only what the next payout still
needs, pause once met) x intraday profit lock on/off x day filters (all, turn of
the month, skip Mondays, after a down day), plus NQ+GC two-market races.
LucidPro 50K funded rules (lucid_pass_research.FundedRules), first payout $500
then $2,500, 120 trading days per funded account.

Anti-overfitting: configurations are RANKED on funded accounts started in
2019-2022 and then judged on 2023-2026 starts they were never ranked on; the
pick is re-checked on real NinjaTrader futures bars (Jun 2025 - Sep 2026).

Usage:  .venv/Scripts/python.exe scripts/lucid_funded_big.py
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402

S = lp.S
H = 120
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5
RULES = lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0), max_days=H)
ENTRIES = {"18:15": S(18, 15), "19:00": S(19), "21:00": S(21), "00:00": S(0), "02:00": S(2), "04:00": S(4),
           "08:00": S(8), "09:30": lp.RTH_OPEN}
SIZES = {"NQ": (5, 10, 15, 20), "ES": (10, 20, 30, 40), "GC": (5, 10, 15, 20)}   # micros (10 = 1 mini)
TICK_USD = {"NQ": 0.5, "ES": 1.25, "GC": 1.0}


def exit_slot(m):
    return lp.RTH_LAST if m in ("NQ", "ES") else S(16, 15)


def day_filters(g: lp.Grid) -> dict:
    first, last = lp.month_position(g.dates)
    dow = pd.to_datetime(g.dates).dayofweek
    down = (g.daily["prev_rth_ret"] < 0).to_numpy()
    return {"all days": np.ones(len(g.dates), bool), "turn of month": np.isin(first, (1, 2, 3)) | (last == -1),
            "skip Mondays": dow != 0, "after a down day": down}


def sides(g: lp.Grid, how: str) -> np.ndarray:
    if how == "long":
        return np.ones(len(g.dates), int)
    t = g.daily["trend"].to_numpy()
    return np.where(np.isnan(t), 0, t).astype(int)


def score(res: pd.DataFrame, years: np.ndarray, period) -> dict:
    sel = (years >= period[0]) & (years <= period[1])
    r = res[sel]
    got = r.payouts >= 1
    s = lp.summarise_funded(r)
    return {"p10": (got & (r.first_day <= 10)).mean(), "any": got.mean(),
            "blown": ((r.status == 2) & ~got).mean(), "med_days": s["median_days_to_payout"],
            "paid": s["avg_paid_net"],
            "per_week": (s["avg_paid_net"] - EVAL_COST) / (EVAL_WEEKS + s["avg_days_alive"] / 5)}


def run_config(c, w, b, starts, years, tp, lock):
    res = lp.replay_funded(c, w, b, RULES, starts, day_tp=tp, adaptive=True, pause_when_met=True,
                           lock_at=(0.6 if lock else None), lock_keep=0.2)
    return score(res, years, (2019, 2022)), score(res, years, (2023, 2026)), score(res, years, (2019, 2026))


def main() -> int:
    t0 = time.time()
    G = {m: lp.load_grid(m) for m in ("NQ", "ES", "GC")}
    rows = []
    for m, g in G.items():
        n = len(g.dates)
        yrs_all = pd.to_datetime(g.dates).year
        keep = yrs_all >= 2019
        starts = np.arange(0, keep.sum() - H)
        years = np.asarray(yrs_all[keep][starts])
        filt = day_filters(g)
        for (ename, a), how, (fname, fmask) in itertools.product(ENTRIES.items(), ("long", "trend"), filt.items()):
            ok = g.has[:, a] & g.has[:, exit_slot(m)] & fmask
            sd = np.where(ok, sides(g, how), 0)
            stop = np.where(sd >= 0, g.O[:, a] - 400.0 * (1 if m != "GC" else 0.25),
                            g.O[:, a] + 400.0 * (1 if m != "GC" else 0.25))
            leg = lp.build_leg(g, "x", sd, np.full(n, a), np.full(n, exit_slot(m)), stop=stop)
            for size, tp, lock in itertools.product(SIZES[m], (700.0, 1040.0), (False, True)):
                c, w = size * leg.close[keep], size * leg.worst[keep]
                b = size * leg.best[keep] - TICK_USD[m] * size
                a_, b_, f_ = run_config(c, w, b, starts, years, tp, lock)
                rows.append({"market": m, "entry": ename, "direction": how, "days": fname, "micros": size,
                             "take<=": tp, "lock": lock,
                             **{f"A_{k}": v for k, v in a_.items()}, **{f"B_{k}": v for k, v in b_.items()},
                             **{f"all_{k}": v for k, v in f_.items()}})
        print(f"{m}: {len(rows)} configurations so far ({time.time() - t0:.0f}s)", flush=True)

    # two-market race: NQ + GC long, same evening, one account-level daily take-profit
    legs = []
    for m in ("NQ", "GC"):
        g = G[m]
        n = len(g.dates)
        ok = g.has[:, S(19)] & g.has[:, exit_slot(m)]
        legs.append(lp.build_leg(g, m, np.where(ok, 1, 0), np.full(n, S(19)), np.full(n, exit_slot(m)),
                                 stop=g.O[:, S(19)] - (400.0 if m == "NQ" else 100.0)))
    dates, al = lp.align(legs)
    yrs_all = pd.to_datetime(dates).year
    keep = yrs_all >= 2019
    starts = np.arange(0, keep.sum() - H)
    years = np.asarray(yrs_all[keep][starts])
    for (s1, s2), tp, lock in itertools.product(((5, 5), (10, 5), (5, 10), (10, 10)), (700.0, 1040.0),
                                                (False, True)):
        c, w, b = lp.combine(al, [s1, s2], with_best=True)
        a_, b_, f_ = run_config(c[keep], w[keep], b[keep] - 1.0 * (s1 + s2), starts, years, tp, lock)
        rows.append({"market": "NQ+GC", "entry": "19:00", "direction": "long", "days": "all days",
                     "micros": f"{s1}+{s2}", "take<=": tp, "lock": lock,
                     **{f"A_{k}": v for k, v in a_.items()}, **{f"B_{k}": v for k, v in b_.items()},
                     **{f"all_{k}": v for k, v in f_.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "futures" / "funded_big.csv", index=False)
    pd.set_option("display.width", 280)
    f = lambda v: f"{v:,.3f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:,.0f}" if isinstance(v, float) else v)  # noqa: E731
    cols = ["market", "entry", "direction", "days", "micros", "take<=", "lock", "A_p10", "A_blown",
            "B_p10", "B_any", "B_blown", "B_med_days", "B_paid", "B_per_week"]
    print(f"\n{len(res)} configurations in {time.time() - t0:.0f}s")
    print("\nTop 15 RANKED ON 2019-22 starts, judged on 2023-26 starts (B_ = never used for ranking):")
    print(res.sort_values(["A_p10", "A_per_week"], ascending=False).head(15)[cols].to_string(index=False,
                                                                                          formatters={c: f for c in cols}))
    base = res[(res.market == "NQ") & (res.entry == "19:00") & (res.direction == "long") & (res.days == "all days")
               & (res.micros == 10) & (res["take<="] == 1040.0) & (~res.lock)]
    print("\nCurrent pick (race 1 NQ @19:00 long, all days, take<= $1,040, no lock):")
    print(base[cols].to_string(index=False, formatters={c: f for c in cols}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
