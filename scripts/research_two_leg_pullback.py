"""Two-legged pullback to the 21 EMA in a trend, strong signal bar, stop entry (owner's fifth video, 2026-10-08).

The video's checklist made exact (longs; shorts mirrored):
  0. Trends only, no ranges: the last two confirmed swing highs AND lows rising (swing = extreme of 2
     bars each side, known 2 bars later) and the 21 EMA rising over the last 5 bars.
  1. Pullback to a key entry point: after a new 20-bar high, the pullback touches the 21 EMA (*or: no
     touch required -- the video's trend line cannot be drawn reliably by a program).
  2. Two legs: leg 1 down, then a bar trades above the previous bar's high (the first attempt up),
     then leg 2 makes a new pullback low.
  3. A strong signal bar with the trend: a bull bar closing in its top third with a body >= 40% of
     its range. Buy stop 1 tick above it (valid for the next bar), stop loss 1 tick below it.
     Target (* the video leaves it open): 1R, 1.5R or 2R. A new 20-bar high before entry resets.
Sessions (*): entries 09:30-15:30 New York or any hour; flat by 16:40 New York (Lucid). One position at
a time. Costs: $3.50 round trip + 0.25 pt spread; 1 tick slippage on the stop entry and on stops; a bar
reaching stop and target counts as the stop; a stop on the entry bar is taken, a target is not.
Data ("works on any time frame"): NQ / ES 15-min stand-ins 2016-2026 (10 years; choose 2016-2022, judge
2023-2026), NinjaTrader NQ 5-min 2021-2026 (choose 2021-2023) and NQ 1-min 2024-2026 (choose to Jun 2025).

Usage:  .venv/Scripts/python.exe scripts/research_two_leg_pullback.py
"""
from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICK, SPREAD = 0.25, 0.25
DATA = {   # name -> (file, commission round trip in points, split date, years)
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01", 10.6),
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01", 10.6),
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01", 5.65),
    "NQ 1-min 2024-26": (ROOT / "data" / "futures" / "NQ_1m_long_nt.parquet", 0.175, "2025-07-01", 2.55),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def run(args) -> pd.DataFrame:
    name, rr, need_ema, ny_only = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    ema = pd.Series(c).ewm(span=21, adjust=False).mean().to_numpy()
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)          # CME day: 18:00 starts the next one
    hh20 = pd.Series(h).rolling(20).max().to_numpy()
    ll20 = pd.Series(l).rolling(20).min().to_numpy()
    p = 2
    shs, sls = [], []
    out = []
    pos = None
    setup = {1: None, -1: None}
    pending = None                      # (side, trigger, stop, valid_bar)
    for i in range(30, n):
        # --- swings (confirmed p bars later) ---
        j = i - p
        if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:i + 1].max():
            shs = (shs + [h[j]])[-2:]
        if l[j] < l[j - p:j].min() and l[j] <= l[j + 1:i + 1].min():
            sls = (sls + [l[j]])[-2:]
        trend = 0
        if len(shs) == 2 and len(sls) == 2:
            if shs[1] > shs[0] and sls[1] > sls[0] and ema[i] > ema[i - 5]:
                trend = 1
            elif shs[1] < shs[0] and sls[1] < sls[0] and ema[i] < ema[i - 5]:
                trend = -1
        # --- the pending stop entry (valid on this bar only) ---
        if pending is not None and pending[3] == i and pos is None:
            side, trig, stp, _ = pending
            if (h[i] >= trig) if side > 0 else (l[i] <= trig):
                ent = (max(o[i], trig) if side > 0 else min(o[i], trig)) + side * TICK      # slippage
                risk = side * (ent - stp)
                if risk > 0:
                    pos = (side, ent, stp, ent + side * rr * risk, i)
        pending = None
        # --- the open position ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if 1000 <= mins[i] < 1080:
                px, how = o[i], "16:40 close"
            elif (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                px, how = tgt, "target"
            elif i + 1 >= n or day[i + 1] != day[i] or 1000 <= mins[i + 1] < 1080:
                px, how = c[i], "16:40 close"                     # some data has no bars 16:15-18:00
            if px is not None:
                risk = abs(ent - stp)
                out.append((df.index[ie], df.index[i], side, risk, (side * (px - ent) - comm - SPREAD) / risk, how))
                pos = None
        # --- two-legged pullback state, per side ---
        for side in (1, -1):
            new_ext = (h[i] >= hh20[i]) if side > 0 else (l[i] <= ll20[i])
            if trend != side:
                setup[side] = None
                continue
            if new_ext:
                setup[side] = {"leg": 0, "pull_ext": l[i] if side > 0 else h[i], "touch": False, "i0": i}
                continue
            s = setup[side]
            if s is None:
                continue
            if (l[i] <= ema[i]) if side > 0 else (h[i] >= ema[i]):
                s["touch"] = True
            deeper = (l[i] < s["pull_ext"]) if side > 0 else (h[i] > s["pull_ext"])
            if deeper:
                s["pull_ext"] = l[i] if side > 0 else h[i]
            attempt = (h[i] > h[i - 1]) if side > 0 else (l[i] < l[i - 1])
            if s["leg"] == 0 and attempt and i - s["i0"] >= 2:
                s["leg"] = 1                                   # first leg ended: one try with the trend
            elif s["leg"] == 1 and deeper:
                s["leg"] = 2                                   # second leg: a new pullback extreme
            if s["leg"] == 2 and (s["touch"] or not need_ema):
                rng = h[i] - l[i]
                body = side * (c[i] - o[i])
                strong = rng > 0 and body >= 0.4 * rng and (
                    (c[i] >= l[i] + 2 * rng / 3) if side > 0 else (c[i] <= h[i] - 2 * rng / 3))
                window_ok = (570 <= mins[i] < 930) if ny_only else not (990 <= mins[i] < 1080)
                if strong and window_ok and pos is None and i + 1 < n:
                    trig = (h[i] + TICK) if side > 0 else (l[i] - TICK)
                    stp = (l[i] - TICK) if side > 0 else (h[i] + TICK)
                    pending = (side, trig, stp, i + 1)
                    setup[side] = None
    return pd.DataFrame(out, columns=["opened", "closed", "side", "risk", "R", "how"])


def summary(t: pd.DataFrame, lo=None, hi=None) -> tuple[int, float, float, float]:
    if lo is not None:
        t = t[(t["opened"] >= pd.Timestamp(lo, tz="UTC")) & (t["opened"] < pd.Timestamp(hi, tz="UTC"))]
    n = len(t)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    r = t["R"].to_numpy()
    return n, float((t["how"] == "target").mean()), float(r.mean()), float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))


def main() -> int:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    grid = list(itertools.product(DATA, (1.0, 1.5, 2.0), (True, False), (True, False)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        rows.append({"data": g[0], "target": f"{g[1]:g}R", "EMA touch": "yes" if g[2] else "no",
                     "hours": "NY 09:30-15:30" if g[3] else "any", "trades": n, "per year": n / years, "win%": w,
                     "avg R": r, "older R": ra, "z older": za, "newer R": rb, "z newer": zb, "_t": t})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    for name in DATA:
        part = res[res["data"] == name].sort_values("older R", ascending=False)
        print(f"\n=== {name}: older years choose, newer years judge (split {DATA[name][2]}) ===")
        print(part.drop(columns=["_t", "data"]).to_string(index=False, formatters=fm))
        b = part.iloc[0]
        t = b["_t"]
        months = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m"))["R"].sum()
        print(f"  all {len(part)} versions: average {part['avg R'].mean():+.3f} R; newer years positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R). "
              f"Chosen ({b['target']}, EMA touch {b['EMA touch']}, {b['hours']}): older {b['older R']:+.3f} -> newer "
              f"{b['newer R']:+.3f} R (z {b['z newer']:+.2f}); win {b['win%']:.1%}; winning months {np.mean(months > 0):.0%}; "
              f"median risk {t['risk'].median():.1f} pts")
        if "15-min" in name:
            by = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean"])
            print("  by year (trades / avg R): " + ", ".join(f"{y}: {int(v['count'])}/{v['mean']:+.2f}" for y, v in by.iterrows()))
    res.drop(columns=["_t"]).to_csv(ROOT / "data" / "research_two_leg_pullback.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
