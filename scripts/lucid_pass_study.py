"""LucidPro evaluation pass-rate study (2026-09-30) -- reproduces docs/LUCIDPRO_PASS_RESEARCH.md.

  1. edge stacks (ORB-trend NQ/GC, turn of month, night drift) vs their zero-edge twins
  2. the same strategies under three readings of Lucid's MLL/DLL wording
  3. the target race (long NQ, limit at the target, no stop, flat 16:00 ET): variants,
     real NinjaTrader MNQ bars vs the proxy, year by year, and a stress test

Usage:
    .venv/Scripts/python.exe scripts/lucid_pass_study.py
Writes data/futures/pass_*.csv.
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
OUT = lp.OUT
H = 120                                           # evaluation horizon, trading days
READINGS = {
    "strict": dict(mll_on="equity", dll_on="equity"),          # open P&L counts
    "middle": dict(mll_on="balance", dll_on="equity"),         # DLL flattens on open P&L
    "lenient": dict(mll_on="balance", dll_on="balance"),       # closed balance only
}


def zero_edge(a):
    """Same paths with the leg's mean P&L per trade removed along the day."""
    c, w, p, b = a
    traded = p != 0
    mu = p[traded].mean() if traded.any() else 0.0
    live = (c != 0) | (w != 0)
    frac = np.cumsum(live, axis=1) / np.maximum(live.sum(1, keepdims=True), 1)
    sh = np.where(traded, mu, 0.0)[:, None] * frac
    return c - sh, w - sh, p - np.where(traded, mu, 0.0), b - sh


def evaluate(c, w, b, plan, reading, starts, pass_on="best"):
    rules = lp.Rules(max_days=H, pass_on=pass_on, **lp.PLANS[plan], **READINGS[reading])
    return lp.replay(c, w, rules, starts, best=b)


def part1_2(G) -> pd.DataFrame:
    legs = {"NQ_ORB": lp.leg_orb_trend(G["NQ"]), "GC_ORB": lp.leg_orb_trend(G["GC"]),
            "NQ_TOM": lp.leg_tom(G["NQ"]), "GC_TOM": lp.leg_tom(G["GC"]),
            "NQ_NIGHT": lp.leg_window(G["NQ"], "NQ night", S(19), S(9, 15)),
            "NQ_RTH": lp.leg_window(G["NQ"], "NQ RTH long", lp.RTH_OPEN, lp.RTH_LAST)}
    names = list(legs)
    dates, al = lp.align([legs[k] for k in names])
    al0 = [zero_edge(a) for a in al]
    starts = np.arange(0, len(dates) - H)
    stacks = {"ORB NQ+GC 6+3 (old trial)": {"NQ_ORB": 6, "GC_ORB": 3},
              "ORB+TOM+night x3": {"NQ_ORB": 6, "GC_ORB": 3, "NQ_TOM": 3, "GC_TOM": 3, "NQ_NIGHT": 3},
              "NQ ORB 20": {"NQ_ORB": 20}, "NQ RTH race 20": {"NQ_RTH": 20}}
    rows = []
    for plan in ("LucidPro 50K", "LucidPro 25K"):
        for label, wts in stacks.items():
            if sum(wts.values()) > lp.MAX_UNITS[plan]:
                continue
            wv = [wts.get(k, 0) for k in names]
            single = len(wts) == 1
            for zero in (False, True):
                c, w, b = lp.combine(al0 if zero else al, wv, with_best=True)
                for rd in READINGS:
                    s = lp.summarise(evaluate(c, w, b, plan, rd, starts, "best" if single else "close"))
                    rows.append({"plan": plan, "stack": label + (" [zero edge]" if zero else ""), "reading": rd,
                                 "pass": s["pass_resolved"], "fails_per_pass": s["fails_per_pass"],
                                 "pass_day1": s["pass_day1"], "med_days": s["med_days"]})
    return pd.DataFrame(rows)


def race(g, a=S(19), b=lp.RTH_LAST, name="race"):
    return lp.leg_window(g, name, a, b)


def part3(G) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nq = G["NQ"]
    # (a) real futures vs proxy on the same days
    real = lp.load_grid("NQ", ROOT / "data" / "futures" / "MNQ_15m.parquet")
    common = np.intersect1d(real.dates, nq.dates)
    rows = []
    for gname, g in (("real MNQ futures", real), ("proxy", nq)):
        m = np.isin(g.dates, common)
        lg = race(g)
        st = np.arange(0, m.sum() - 40)
        for plan, size in (("LucidPro 25K", 20), ("LucidPro 50K", 40)):
            c, w, b = size * lg.close[m], size * lg.worst[m], size * lg.best[m] - 0.5 * size
            for rd in ("strict", "lenient"):
                rules = lp.Rules(max_days=40, pass_on="best", **lp.PLANS[plan], **READINGS[rd])
                s = lp.summarise(lp.replay(c, w, rules, st, best=b))
                rows.append({"data": gname, "days": f"{common[0]}..{common[-1]}", "plan": plan, "micros": size,
                             "reading": rd, "pass": s["pass_resolved"], "pass_day1": s["pass_day1"]})
    real_vs_proxy = pd.DataFrame(rows)

    # (b) year by year (proxy), 25K 20 MNQ
    lg = race(nq)
    st = np.arange(0, len(nq.dates) - H)
    yrs = pd.to_datetime(nq.dates[st]).year
    by_year = {}
    for rd in ("strict", "lenient"):
        res = evaluate(20 * lg.close, 20 * lg.worst, 20 * lg.best - 10, "LucidPro 25K", rd, st)
        res["year"] = yrs
        by_year[rd] = res[res.status > 0].groupby("year").status.apply(lambda s: (s == 1).mean())
    by_year = pd.DataFrame(by_year)

    # (c) stress (2019+ = full night data)
    keep = pd.to_datetime(nq.dates).year >= 2019
    st = np.arange(0, keep.sum() - H)
    rows = []

    def one(label, a, b, size=20, plan="LucidPro 25K", extra=0.0, fill_ticks=1):
        leg = race(nq, a, b)
        tr = leg.traded[keep][:, None]
        c = size * leg.close[keep] - extra * size * tr
        w = size * leg.worst[keep] - extra * size * tr
        bb = size * leg.best[keep] - extra * size * tr - 0.5 * fill_ticks * size
        out = {"variant": label, "plan": plan, "micros": size}
        for rd in ("strict", "lenient"):
            s = lp.summarise(evaluate(c, w, bb, plan, rd, st))
            out[rd] = s["pass_resolved"]
            out[rd + "_day1"] = s["pass_day1"]
            out[rd + "_fails_per_pass"] = s["fails_per_pass"]
        rows.append(out)

    one("base: long 19:00 -> 16:00", S(19), lp.RTH_LAST)
    one("costs x2 + 4-tick limit fills", S(19), lp.RTH_LAST, extra=2.0, fill_ticks=4)
    for h, mm in ((18, 15), (21, 0), (0, 0), (2, 0), (8, 0)):
        one(f"entry {h:02d}:{mm:02d}", S(h, mm), lp.RTH_LAST)
    one("exit 15:15", S(19), S(15, 0))
    for size in (10, 15):
        one(f"{size} MNQ", S(19), lp.RTH_LAST, size=size)
    for size in (20, 40):
        one(f"50K, {size} MNQ", S(19), lp.RTH_LAST, size=size, plan="LucidPro 50K")
    return real_vs_proxy, by_year, pd.DataFrame(rows)


def main() -> int:
    fh_awake = getattr(lp.fh, "keep_awake", None)
    if fh_awake:
        fh_awake()
    G = {m: lp.load_grid(m) for m in ("NQ", "GC")}
    pd.set_option("display.width", 250)
    fmt = {"float_format": lambda v: f"{v:.2f}", "index": False}
    t12 = part1_2(G)
    t12.to_csv(OUT / "pass_stacks_readings.csv", index=False)
    print("=== 1-2: stacks x rule readings ===")
    print(t12.to_string(**fmt))
    rv, by, stress = part3(G)
    rv.to_csv(OUT / "pass_race_real_vs_proxy.csv", index=False)
    by.to_csv(OUT / "pass_race_by_year.csv")
    stress.to_csv(OUT / "pass_race_stress.csv", index=False)
    print("\n=== 3a: race on real MNQ futures vs proxy ===")
    print(rv.to_string(**fmt))
    print("\n=== 3b: race by start year (25K, 20 MNQ) ===")
    print(by.round(2).to_string())
    print("\n=== 3c: stress ===")
    print(stress.to_string(**fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
