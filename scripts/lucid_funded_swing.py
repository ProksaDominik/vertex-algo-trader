"""Funded account: MULTI-DAY swing edges, made Lucid-legal by re-entering daily
(owner, 2026-10-01 -- the old MT5 funded bot's numbers came from multi-day CFD
swing trades incl. Bitcoin).

Lucid's trading day runs 18:00 -> 16:45 ET, so a multi-day position is held as
a chain of one-day positions: flat at 16:30/16:45, back in at 19:00 ET. Each
day pays a full round trip (2 ticks + Lucid commission). Signals use only the
previous RTH close (16:00 ET), i.e. what is known when the evening session
opens.

Legs (long-only unless stated; from futures_multiday.py's most robust effects):
  TOM      turn of the month: the last trading day + the first 3
  RSI2     Connors pullback: close > SMA200 and RSI(2) < 10 -> hold until
           close > SMA5, at most 5 days
  DON20    20-day closing-high breakout -> hold until a 10-day closing low
  ORB      the intraday ORB + trend leg (for stacking)

Each leg and stack is put through the funded replay (FundedRules: equity MLL
$2,000 trailing the best close / lock $50,100; first payout $500 then $2,500;
40% consistency; >= 3 days; 90/10), rolling starts 2019-26, 120 days, and
ranked on 2019-22 starts / judged on 2023-26 starts.

Usage:  .venv/Scripts/python.exe scripts/lucid_funded_swing.py
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
H = 120
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5
RULES = lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0), max_days=H)


def rsi(c: pd.Series, n: int = 2) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def daily_positions(g: lp.Grid, rule: str) -> np.ndarray:
    """+1/0 per Lucid day, decided from closes up to the PREVIOUS RTH close."""
    dd = g.daily
    c = dd["close"].ffill()
    n = len(c)
    pos = np.zeros(n, int)
    if rule == "TOM":
        first, last = lp.month_position(g.dates)
        return np.where(np.isin(first, (1, 2, 3)) | (last == -1), 1, 0)
    if rule == "RSI2":
        sma200, sma5, r2 = c.rolling(200).mean(), c.rolling(5).mean(), rsi(c, 2)
        held = 0
        for i in range(1, n):
            j = i - 1                                            # yesterday's close decides today
            if held:
                held += 1
                if c.iloc[j] > sma5.iloc[j] or held > 5:
                    held = 0
            if not held and c.iloc[j] > sma200.iloc[j] and r2.iloc[j] < 10:
                held = 1
            pos[i] = 1 if held else 0
        return pos
    if rule == "DON20":
        hi20, lo10 = c.rolling(20).max().shift(1), c.rolling(10).min().shift(1)
        on = False
        for i in range(1, n):
            j = i - 1
            if not on and c.iloc[j] > hi20.iloc[j]:
                on = True
            elif on and c.iloc[j] < lo10.iloc[j]:
                on = False
            pos[i] = 1 if on else 0
        return pos
    raise ValueError(rule)


def swing_leg(g: lp.Grid, rule: str) -> lp.Leg:
    a, b = S(19), (lp.RTH_LAST if g.market in ("NQ", "ES") else S(16, 15))
    n = len(g.dates)
    ok = g.has[:, a] & g.has[:, b]
    side = np.where(ok, daily_positions(g, rule), 0)
    return lp.build_leg(g, f"{g.market} {rule}", side, np.full(n, a), np.full(n, b))


def score(res, years, lo, hi):
    sel = (years >= lo) & (years <= hi)
    r = res[sel]
    got = r.payouts >= 1
    s = lp.summarise_funded(r)
    return {"p10": (got & (r.first_day <= 10)).mean(), "p20": (got & (r.first_day <= 20)).mean(),
            "any": got.mean(), "blown": ((r.status == 2) & ~got).mean(),
            "med_days": s["median_days_to_payout"], "paid": s["avg_paid_net"],
            "per_week": (s["avg_paid_net"] - EVAL_COST) / (EVAL_WEEKS + s["avg_days_alive"] / 5)}


def main() -> int:
    G = {m: lp.load_grid(m) for m in ("NQ", "ES", "GC")}
    legs = {}
    for m in ("NQ", "ES", "GC"):
        for rule in ("TOM", "RSI2", "DON20"):
            legs[f"{m} {rule}"] = swing_leg(G[m], rule)
    legs["NQ ORB"] = lp.leg_orb_trend(G["NQ"])
    legs["GC ORB"] = lp.leg_orb_trend(G["GC"])
    names = list(legs)
    print("per-leg stats (1 micro, 2016-26, daily re-entry costs included):")
    print(pd.DataFrame([lp.leg_stats(legs[k]) for k in names]).to_string(index=False,
                                                                          float_format=lambda v: f"{v:.2f}"))
    dates, al = lp.align([legs[k] for k in names])
    yrs = pd.to_datetime(dates).year
    keep = yrs >= 2019
    starts = np.arange(0, keep.sum() - H)
    years = np.asarray(yrs[keep][starts])
    daily = pd.DataFrame({k: a[2] for k, a in zip(names, al)}, index=pd.to_datetime(dates))
    print("\nannualised Sharpe per leg (daily P&L incl. flat days):")
    print((daily.mean() / daily.std() * np.sqrt(252)).round(2).to_string())

    stacks = {
        "TOM NQ+ES+GC": {"NQ TOM": 1, "ES TOM": 1, "GC TOM": 1},
        "RSI2 NQ+ES": {"NQ RSI2": 1, "ES RSI2": 1},
        "DON20 NQ+GC": {"NQ DON20": 1, "GC DON20": 1},
        "swing all (TOM+RSI2+DON20, NQ/ES/GC)": {k: 1 for k in names if "ORB" not in k},
        "swing all + ORB": {k: 1 for k in names},
        "ORB NQ+GC": {"NQ ORB": 2, "GC ORB": 1},
    }
    rows = []
    for (label, w), mult in itertools.product(stacks.items(), (1, 2, 3, 5, 8)):
        wv = [w.get(k, 0) * mult for k in names]
        c, wo, b = lp.combine(al, wv, with_best=True)
        res = lp.replay_funded(c[keep], wo[keep], b[keep], RULES, starts)
        a_, b_ = score(res, years, 2019, 2022), score(res, years, 2023, 2026)
        rows.append({"stack": label, "size x": mult, **{f"A_{k}": v for k, v in a_.items()},
                     **{f"B_{k}": v for k, v in b_.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "futures" / "funded_swing.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["stack", "size x", "A_any", "A_blown", "A_per_week", "B_p20", "B_any", "B_blown", "B_med_days",
            "B_paid", "B_per_week"]
    print("\nfunded replay, ranked on 2019-22 (A_) and judged on 2023-26 (B_):")
    print(res.sort_values("A_per_week", ascending=False)[cols].to_string(index=False,
                                                                         float_format=lambda v: f"{v:,.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
