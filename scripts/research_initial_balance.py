"""Initial Balance Breakout (owner's ninth video, Czech, 2026-10-08): first check its statistics, then trade it.

Initial balance (IB) = the first New York hour, 09:30-10:30 (15:30-16:30 Prague).
Part 1 -- the video's statistics (it claims 16 years of data): one side of the IB breaks by 16:00 in 95.7% of
  days, only one side 76.4%, both ~20%, none 3.5%; if the IB made its LOW first and HIGH second, the HIGH
  breaks first in 75.7%; high first -> the LOW breaks first in 63.4%. Checked on 10 years of 15-minute
  NQ/ES, with a control: where price CLOSED inside the IB at 10:30 (top half / bottom half) -- if the
  first extreme only says "price ended near the other side", the edge is just distance.
Part 2 -- the trade, as described: direction = the side the first extreme points to; wait for a pullback
  (before either side of the IB breaks, by 15:30) into a zone: *a fair value gap formed in the IB in that
  direction, *the New York VWAP (from 09:30), or *either; entry *limit at the zone (FVG near edge / the
  VWAP) or *a confirmation candle (touched the zone, closed back in our direction); stop 1 tick beyond the
  FVG's far edge (VWAP: 1 ATR beyond it); target = the expected IB extreme + 1 tick (the break); *only if
  target >= 1R or 1.5R (or no minimum); stops closer than 4 ticks or a quarter ATR are skipped. One trade a day, flat by 16:40. Costs as before ($3.50 round trip
  + 0.25 pt spread, 1 tick slippage on stops; stop first if a bar reaches both; no target on the entry bar).
Data: NQ / ES 15-min stand-ins 2016-2026 (10 years; choose 2016-2022, judge 2023-2026) and NinjaTrader NQ
5-min 2021-2026 (choose 2021-2023, judge 2024-2026).

Usage:  .venv/Scripts/python.exe scripts/research_initial_balance.py
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
DATA = {
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01", 10.6),
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01", 10.6),
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01", 5.65),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def days_of(df: pd.DataFrame):
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray(et.date)
    return mins, day


# ---------------------------------------------------------------- part 1: the statistics
def ib_stats(name: str) -> pd.DataFrame:
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    mins, day = days_of(df)
    rows = []
    for d, idx in pd.Series(np.arange(len(c))).groupby(day):
        idx = idx.to_numpy()
        m = mins[idx]
        ib = idx[(m >= 570) & (m < 630)]
        after = idx[(m >= 630) & (m < 960)]
        if len(ib) < 2 or len(after) < 4:
            continue
        ihi, ilo = h[ib].max(), l[ib].min()
        t_hi, t_lo = ib[np.argmax(h[ib])], ib[np.argmin(l[ib])]
        if t_hi == t_lo:
            first = "same bar"
        else:
            first = "low first" if t_lo < t_hi else "high first"
        up = np.flatnonzero(h[after] > ihi)
        dn = np.flatnonzero(l[after] < ilo)
        fu, fd = (up[0] if len(up) else 10**9), (dn[0] if len(dn) else 10**9)
        if fu == fd == 10**9:
            first_break = "none"
        elif fu == fd:
            first_break = "both in one bar"
        else:
            first_break = "high" if fu < fd else "low"
        close_pos = (c[ib[-1]] - ilo) / (ihi - ilo) if ihi > ilo else 0.5
        # the naive trade: at 10:30 go the predicted way, target that IB extreme, stop the other one
        side = 1 if first == "low first" else (-1 if first == "high first" else 0)
        naive = np.nan
        if side and len(after):
            ent = o[after[0]]
            tgt, stp = (ihi + TICK, ilo - TICK) if side > 0 else (ilo - TICK, ihi + TICK)
            risk, rew = side * (ent - stp), side * (tgt - ent)
            if risk > 0 and rew > 0:
                res = None
                for k in after:
                    if (l[k] <= stp) if side > 0 else (h[k] >= stp):
                        res = -risk - TICK
                        break
                    if (h[k] >= tgt) if side > 0 else (l[k] <= tgt):
                        res = rew
                        break
                if res is None:
                    res = side * (c[after[-1]] - ent)
                naive = (res - comm - SPREAD) / risk
        rows.append({"day": pd.Timestamp(d), "first": first, "first_break": first_break,
                     "broke_hi": len(up) > 0, "broke_lo": len(dn) > 0, "close_pos": close_pos, "naive_R": naive})
    return pd.DataFrame(rows)


def print_stats(name: str, s: pd.DataFrame) -> None:
    n = len(s)
    any_b = (s.broke_hi | s.broke_lo).mean()
    one = (s.broke_hi ^ s.broke_lo).mean()
    both = (s.broke_hi & s.broke_lo).mean()
    none = (~s.broke_hi & ~s.broke_lo).mean()
    print(f"\n=== {name}: {n} days ===")
    print(f"  a side breaks by 16:00: {any_b:.1%} (video 95.7%) | only one side {one:.1%} (76.4%) | "
          f"both {both:.1%} (~20%) | none {none:.1%} (3.5%)")
    for first, target, claim in (("low first", "high", "75.7%"), ("high first", "low", "63.4%")):
        x = s[(s["first"] == first) & s["first_break"].isin(["high", "low"])]
        print(f"  {first}: the {target} breaks first in {(x['first_break'] == target).mean():.1%} of {len(x)} days "
              f"(video {claim})")
    x = s[s["first_break"].isin(["high", "low"])]
    top, bot = x[x.close_pos >= 0.5], x[x.close_pos < 0.5]
    print(f"  control -- closed in the TOP half of the IB at 10:30: high breaks first {(top['first_break'] == 'high').mean():.1%};"
          f" bottom half: low breaks first {(bot['first_break'] == 'low').mean():.1%}")
    for first, half in (("low first", "top"), ("low first", "bottom"), ("high first", "top"), ("high first", "bottom")):
        y = x[(x["first"] == first) & ((x.close_pos >= 0.5) if half == "top" else (x.close_pos < 0.5))]
        if len(y):
            print(f"    {first}, closed in the {half} half: high breaks first {(y['first_break'] == 'high').mean():.1%} ({len(y)} days)")
    r = s["naive_R"].dropna()
    print(f"  naive trade at 10:30 (predicted side, target that IB extreme, stop the other): {len(r)} trades, "
          f"average {r.mean():+.3f} R, winning {np.mean(r > 0):.0%}")


# ---------------------------------------------------------------- part 2: the trade
def run(args) -> pd.DataFrame:
    name, zone, entry, min_rr = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    mins, day = days_of(df)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    tpx = (h + l + c) / 3
    out = []
    pos = None
    cur = None
    for i in range(20, n):
        if day[i] != cur:
            cur, ib, st, done, vw_num, vw_den = day[i], [], None, False, 0.0, 0
        m = mins[i]
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] >= tgt) if side > 0 else (l[i] <= tgt)):
                px, how = tgt, "target"
            elif i + 1 >= n or day[i + 1] != day[i] or mins[i + 1] >= 1000:
                px, how = c[i], "16:40 close"
            if px is not None:
                out.append((df.index[ie], df.index[i], side, abs(ent - stp), (side * (px - ent) - comm - SPREAD) / abs(ent - stp), how))
                pos = None
        if m >= 570:
            vw_num += tpx[i]
            vw_den += 1                                             # (no reliable volume: equal weights)
        vwap = vw_num / vw_den if vw_den else np.nan
        if 570 <= m < 630:
            ib.append(i)
            continue
        if done or pos is not None or m < 630 or m >= 930 or len(ib) < 2:
            continue
        ib_a = np.array(ib)
        ihi, ilo = h[ib_a].max(), l[ib_a].min()
        if st is None:
            t_hi, t_lo = ib_a[np.argmax(h[ib_a])], ib_a[np.argmin(l[ib_a])]
            if t_hi == t_lo:
                done = True
                continue
            side = 1 if t_lo < t_hi else -1
            fvg = None
            for k in ib_a[2:]:                                      # the last gap of the hour in our direction
                if side > 0 and l[k] > h[k - 2]:
                    fvg = (h[k - 2], l[k])
                if side < 0 and h[k] < l[k - 2]:
                    fvg = (h[k], l[k - 2])
            st = {"side": side, "fvg": fvg}
        side = st["side"]
        # the IB broke before our pullback: the setup is over
        if h[i] > ihi or l[i] < ilo:
            done = True
            continue
        zones = []
        if zone in ("fair value gap", "either") and st["fvg"] is not None:
            g_lo, g_hi = st["fvg"]
            zones.append(((g_hi if side > 0 else g_lo), (g_lo - TICK if side > 0 else g_hi + TICK)))
        if zone in ("VWAP", "either") and not np.isnan(vwap) and not np.isnan(atr[i]):
            zones.append((vwap, vwap - side * atr[i]))
        for near, stp in zones:
            touched = (l[i] <= near) if side > 0 else (h[i] >= near)
            if not touched:
                continue
            if entry == "limit":
                ent = min(o[i], near) if side > 0 else max(o[i], near)
            else:
                if not (side * (c[i] - o[i]) > 0 and side * (c[i] - near) > 0):
                    continue
                ent = c[i]
            tgt = (ihi + TICK) if side > 0 else (ilo - TICK)
            risk, rew = side * (ent - stp), side * (tgt - ent)
            done = True
            if risk < max(4 * TICK, 0.25 * atr[i]) or rew <= 0 or rew < min_rr * risk:
                break                                  # no realistic stop closer than 4 ticks / a quarter ATR
            if entry == "limit" and ((l[i] <= stp) if side > 0 else (h[i] >= stp)):
                out.append((df.index[i], df.index[i], side, risk, (-risk - TICK - comm - SPREAD) / risk, "stop"))
                break
            pos = (side, ent, stp, tgt, i)
            break
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
    for name in DATA:
        print_stats(name, ib_stats(name))
    grid = list(itertools.product(DATA, ("fair value gap", "VWAP", "either"), ("limit", "confirmation"), (0.0, 1.0, 1.5)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        rows.append({"data": g[0], "zone": g[1], "entry": g[2], "min target": f"{g[3]:g}R", "trades": n,
                     "per year": n / years, "win%": w, "avg R": r, "older R": ra, "z older": za, "newer R": rb,
                     "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    for name in DATA:
        part = res[res["data"] == name].sort_values("older R", ascending=False)
        print(f"\n=== the trade, {name} (older choose, newer judge; split {DATA[name][2]}) ===")
        print(part.drop(columns=["data"]).to_string(index=False, formatters=fm))
        b = part.iloc[0]
        print(f"  all {len(part)} versions: average {part['avg R'].mean():+.3f} R; newer positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R). Chosen: {b['zone']}, "
              f"{b['entry']}, min {b['min target']}: older {b['older R']:+.3f} -> newer {b['newer R']:+.3f} R (z {b['z newer']:+.2f})")
    res.to_csv(ROOT / "data" / "research_initial_balance.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
