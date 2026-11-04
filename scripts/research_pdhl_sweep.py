"""Previous-day high/low sweep -> break of structure -> fair value gap (owner's third video, 2026-10-08).

Rules as described (long side; shorts mirrored):
  1. Mark the previous day's high (PDH) and low (PDL) -- *the full CME day (18:00-17:00 New York) or
     *New York hours only (09:30-16:00).
  2. Price trades below the PDL (the sweep). Then, "immediately" (*within 12 or 36 five-minute bars of
     the sweep's lowest point), a candle CLOSES above the last swing high (*swing = extreme of 1 or 2
     bars on each side, known only later), and that swing high is ABOVE the PDL ("break of structure
     above the liquidity"; wicks don't count).
  3. The move up leaves a bullish fair value gap (3 candles: the 3rd's low above the 1st's high) between
     the sweep low and the break; the most recent one is used. Limit entry when price comes back to it
     (*at the gap's top edge or its middle), stop 1 tick below the sweep low, target the PDH. Taken
     only if target >= 2.5 x risk. One setup per side per day; the setup is dropped if price hits the
     stop level or the target before filling, or at 15:30 New York.
Sweeps and entries (*): any hour of the CME day, or New York hours only. Everything flat by 16:40 New
York (Lucid). One position at a time. Costs as before: $3.50 round trip + 0.25 pt spread, 1 tick
slippage on stops; a bar touching stop and target counts as the stop; no target on the entry bar.
Data: NinjaTrader 5-minute NQ Feb 2021 - Oct 2026 (choose on 2021-2023, judge on 2024-2026),
5-minute ES Dec 2023 - Oct 2026 as a check.

Usage:  .venv/Scripts/python.exe scripts/research_pdhl_sweep.py
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
F = ROOT / "data" / "futures"
TICK, SPREAD, RR_MIN = 0.25, 0.25, 2.5
COMM = {"NQ": 0.175, "ES": 0.07}


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(F / f"{sym}_5m_long_nt.parquet")[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def prev_day_levels(df: pd.DataFrame, rth_range: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)          # CME day: 18:00 belongs to the next date
    src = df if not rth_range else df[(mins >= 570) & (mins < 960)]
    sday = day if not rth_range else day[(mins >= 570) & (mins < 960)]
    g = pd.DataFrame({"h": src["high"].to_numpy(), "l": src["low"].to_numpy(), "d": sday}).groupby("d").agg(
        h=("h", "max"), l=("l", "min"))
    prev = g.shift(1)
    pdh = pd.Series(day).map(prev["h"]).to_numpy(dtype=float)
    pdl = pd.Series(day).map(prev["l"]).to_numpy(dtype=float)
    return pdh, pdl, day, mins


def run(args) -> pd.DataFrame:
    sym, rth_range, rth_only, p, bos_win, mid, flat = args
    df = load(sym)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    pdh, pdl, day, mins = prev_day_levels(df, rth_range)
    comm = COMM[sym]
    sh: list[tuple[float, int]] = []          # confirmed swing highs (level, index)
    sl: list[tuple[float, int]] = []
    out = []
    pos = None
    cur_day = None
    st = {}
    for i in range(10, len(c)):
        if day[i] != cur_day:
            cur_day = day[i]
            st = {1: {"s": "idle"}, -1: {"s": "idle"}}
        # --- swings confirmed p bars later ---
        j = i - p
        if j >= p:
            if h[j] == h[j - p:i + 1].max() and h[j] > h[j - p:j].max():
                sh.append((h[j], j))
                sh = sh[-60:]
            if l[j] == l[j - p:i + 1].min() and l[j] < l[j - p:j].min():
                sl.append((l[j], j))
                sl = sl[-60:]
        # --- the open position ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if flat and 1000 <= mins[i] < 1080:
                px, how = o[i], "16:40 close"
            elif (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                px, how = tgt, "target"
            if px is not None:
                risk = abs(ent - stp)
                out.append((df.index[ie], df.index[i], side, risk, (side * (px - ent) - comm - SPREAD) / risk, how))
                pos = None
        if np.isnan(pdh[i]) or np.isnan(pdl[i]):
            continue
        in_window = (570 <= mins[i] < 930) if rth_only else not (930 <= mins[i] < 1080)
        for side in (1, -1):
            s = st[side]
            liq, opp = (pdl[i], pdh[i]) if side > 0 else (pdh[i], pdl[i])
            if s["s"] == "idle":
                if in_window and ((l[i] < liq) if side > 0 else (h[i] > liq)):
                    s.update(s="swept", ext=(l[i] if side > 0 else h[i]), ext_i=i)
                continue
            if s["s"] == "swept":
                if (l[i] < s["ext"]) if side > 0 else (h[i] > s["ext"]):
                    s.update(ext=(l[i] if side > 0 else h[i]), ext_i=i)
                if i - s["ext_i"] > bos_win or not in_window:
                    s["s"] = "done"
                    continue
                # break of structure: close beyond the last swing on the far side of the liquidity
                swings = sh if side > 0 else sl
                lvl = next((v for v, k in reversed(swings) if k < i and ((v > liq) if side > 0 else (v < liq))), None)
                if lvl is None or not ((c[i] > lvl) if side > 0 else (c[i] < lvl)):
                    continue
                gap = None
                for k in range(i, s["ext_i"] + 1, -1):           # most recent fair value gap in the move
                    if side > 0 and l[k] > h[k - 2]:
                        gap = (h[k - 2], l[k])
                        break
                    if side < 0 and h[k] < l[k - 2]:
                        gap = (h[k], l[k - 2])
                        break
                if gap is None:
                    s["s"] = "done"
                    continue
                g_lo, g_hi = gap
                ent = (g_lo + g_hi) / 2 if mid else (g_hi if side > 0 else g_lo)
                stp = s["ext"] - side * TICK
                risk = side * (ent - stp)
                if risk <= 0 or side * (opp - ent) < RR_MIN * risk:
                    s["s"] = "done"
                    continue
                s.update(s="pending", ent=ent, stp=stp, tgt=opp, made=i)
                continue
            if s["s"] == "pending":
                if i == s["made"]:
                    continue
                if not in_window or mins[i] >= 930 and mins[i] < 1080:
                    s["s"] = "done"
                    continue
                if (l[i] <= s["stp"]) if side > 0 else (h[i] >= s["stp"]):
                    if (l[i] <= s["ent"]) if side > 0 else (h[i] >= s["ent"]):
                        if pos is None:                                    # filled and stopped in one bar
                            risk = abs(s["ent"] - s["stp"])
                            out.append((df.index[i], df.index[i], side, risk, (-risk - TICK - comm - SPREAD) / risk, "stop"))
                    s["s"] = "done"
                    continue
                if (h[i] >= s["tgt"]) if side > 0 else (l[i] <= s["tgt"]):
                    s["s"] = "done"                                        # ran to the target without us
                    continue
                if (l[i] <= s["ent"]) if side > 0 else (h[i] >= s["ent"]):
                    if pos is None:
                        ent = min(o[i], s["ent"]) if side > 0 else max(o[i], s["ent"])
                        pos = (side, ent, s["stp"], s["tgt"], i)
                    s["s"] = "done"
    return pd.DataFrame(out, columns=["opened", "closed", "side", "risk", "R", "how"])


def summary(t: pd.DataFrame, lo: int | None = None, hi: int | None = None) -> tuple[int, float, float, float]:
    if lo is not None:
        y = t["opened"].dt.year
        t = t[(y >= lo) & (y <= hi)]
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
    grid = list(itertools.product(["NQ"], (False, True), (True, False), (1, 2), (12, 36), (False, True), (True, False)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, 2021, 2023)
        _, _, rb, zb = summary(t, 2024, 2026)
        rows.append({"prev day": "NY hours" if g[1] else "full day", "trade hours": "NY 09:30-15:30" if g[2] else "any",
                     "swing": g[3], "within bars": g[4], "entry": "gap middle" if g[5] else "gap edge",
                     "16:40 flat": "yes (Lucid)" if g[6] else "no (hold)",
                     "trades": n, "per year": n / 5.65, "win%": w, "avg R": r, "2021-23 R": ra, "z A": za,
                     "2024-26 R": rb, "z B": zb, "_t": t, "_g": g})
    res = pd.DataFrame(rows).sort_values("2021-23 R", ascending=False)
    pd.set_option("display.width", 230)
    fm = {c: (lambda v: f"{v:+.3f}") for c in ("avg R", "2021-23 R", "2024-26 R")}
    fm.update({"z A": lambda v: f"{v:+.2f}", "z B": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    cols = ["prev day", "trade hours", "swing", "within bars", "entry", "16:40 flat", "trades", "per year", "win%", "avg R",
            "2021-23 R", "z A", "2024-26 R", "z B"]
    print("=== Previous-day high/low sweep + break of structure + fair value gap, NQ 5-minute Feb 2021 - Oct 2026 ===")
    print("(targets >= 2.5R, so ~29% wins break even before costs)")
    print(res[cols].to_string(index=False, formatters=fm))
    for fl in ("yes (Lucid)", "no (hold)"):
        part = res[res["16:40 flat"] == fl]
        print(f"16:40 flat = {fl}: {len(part)} versions, average {part['avg R'].mean():+.3f} R; 2024-26 positive in "
              f"{(part['2024-26 R'] > 0).mean():.0%} (average {part['2024-26 R'].mean():+.3f} R); target reached "
              f"{np.mean([(t['how'] == 'target').mean() for t in part['_t']]):.0%} of trades")
    print(f"\nall {len(res)} versions: average {res['avg R'].mean():+.3f} R/trade; 2024-26 positive in "
          f"{(res['2024-26 R'] > 0).mean():.0%} of versions (average {res['2024-26 R'].mean():+.3f} R)")
    best = res.iloc[0]
    t = best["_t"]
    print(f"\nCHOSEN on 2021-2023: previous day = {best['prev day']}, hours {best['trade hours']}, swing {best['swing']}, "
          f"within {best['within bars']} bars, entry at the {best['entry']}, 16:40 flat {best['16:40 flat']}")
    print(f"  NQ: 2021-23 {best['2021-23 R']:+.3f} R (z {best['z A']:+.2f}) -> 2024-26 {best['2024-26 R']:+.3f} R "
          f"(z {best['z B']:+.2f}); win {best['win%']:.1%}, {best['per year']:.0f} trades a year")
    by = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean", "sum"])
    print("  by year (trades / avg R / total R): " + ", ".join(f"{y}: {int(v['count'])}/{v['mean']:+.2f}/{v['sum']:+.0f}"
                                                            for y, v in by.iterrows()))
    months = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m"))["R"].sum()
    streak = worst = 0
    for wn in (t["how"] == "target"):
        streak = 0 if wn else streak + 1
        worst = max(worst, streak)
    print(f"  exits: {t['how'].value_counts().to_dict()}; profitable months {np.mean(months > 0):.0%} of {len(months)}; "
          f"longest losing streak {worst}; median risk {t['risk'].median():.1f} pts (${2 * t['risk'].median():,.0f} per MNQ)")
    es = run(("ES",) + tuple(best["_g"][1:]))
    n, w, r, z = summary(es)
    print(f"  ES check (Dec 2023 - Oct 2026, never used for choosing): {n} trades, win {w:.1%}, {r:+.3f} R (z {z:+.2f})")
    res.drop(columns=["_t", "_g"]).to_csv(ROOT / "data" / "research_pdhl_sweep.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
