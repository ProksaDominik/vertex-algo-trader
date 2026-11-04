"""Volume profile value-area fades + order-flow 'orbs' (owner's thirteenth video, 2026-10-08).

Shorts at the value-area high (longs mirrored at the value-area low):
  Levels: value-area high/low (70% of volume around the point of control) of the OVERNIGHT session
  (18:00-09:30 New York) and of the PREVIOUS New York session (09:30-16:00), built from 1-minute bars
  (each bar's volume spread evenly over its range, 1-tick bins).
  1. In New York hours (09:30-15:30) price trades up to a value-area high.
  2. Within 15 minutes, confirmation (*):
       'orbs'  -- big sell orders (trades of 10+ contracts, from the NinjaTrader order flow) outweigh big
                  buys at least 2:1 over the last 5 minutes, and a candle closes back below the level;
       'delta' -- more volume sold than bought over the last 5 minutes, and the close back below;
       'none'  -- just a close back below the level.
  3. Enter at that close; stop 2 ticks above the high since the touch (at least 4 points); target (*)
     1R (his rule) or 1.4R (his example: 8 points risked for 11).
Up to 2 trades a day, one at a time, flat by 16:40. Costs: $3.50 round trip (0.175 pt) + 0.25 pt spread,
1 tick slippage on stops; a bar reaching stop and target counts as the stop; no target on the entry bar.
Data: NinjaTrader NQ 1-minute Mar 2024 - Oct 2026; order flow Oct 2025 - Oct 2026 (the orbs versions run
only there). Each period is split in half by date: choose on the first, judge on the second.

Usage:  .venv/Scripts/python.exe scripts/research_volume_profile.py
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
TICK, SPREAD, COMM = 0.25, 0.25, 0.175


def load() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "futures" / "NQ_1m_long_nt.parquet")[["open", "high", "low", "close", "volume"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    df = df[~df.index.duplicated()].sort_index()
    flow = []
    for f in sorted((ROOT / "data" / "orderflow").glob("NQ_*.parquet")):
        flow.append(pd.read_parquet(f))
    fl = pd.concat(flow).sort_index() if flow else pd.DataFrame()
    fl.index = pd.DatetimeIndex(fl.index).tz_convert("UTC").as_unit("ns")
    fl = fl[~fl.index.duplicated()]
    df = df.join(fl[["buy", "sell", "big_buy", "big_sell"]], how="left")
    return df


def value_area(lo: np.ndarray, hi: np.ndarray, vol: np.ndarray) -> tuple[float, float, float]:
    """(value-area low, point of control, value-area high), 70% of the volume, 1-tick bins."""
    b0 = int(np.floor(lo.min() / TICK))
    s = np.floor(lo / TICK).astype(int) - b0
    e = np.ceil(hi / TICK).astype(int) - b0
    nb = int(e.max()) + 2
    rate = vol / np.maximum(e - s + 1, 1)
    diff = np.zeros(nb + 1)
    np.add.at(diff, s, rate)
    np.add.at(diff, e + 1, -rate)
    prof = np.cumsum(diff)[:nb]
    poc = int(np.argmax(prof))
    total, acc, a, b = prof.sum(), prof[poc], poc, poc
    while acc < 0.7 * total and (a > 0 or b < nb - 1):
        up = prof[b + 1] if b < nb - 1 else -1
        dn = prof[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            acc += up
        else:
            a -= 1
            acc += dn
    return (b0 + a) * TICK, (b0 + poc) * TICK, (b0 + b) * TICK


def levels(df: pd.DataFrame, mins: np.ndarray, day: np.ndarray) -> dict:
    lo, hi, vol = df["low"].to_numpy(), df["high"].to_numpy(), df["volume"].to_numpy(dtype=float)
    idx = pd.Series(np.arange(len(df)))
    out, prev_rth = {}, None
    for d, ix in idx.groupby(day):
        ix = ix.to_numpy()
        m = mins[ix]
        on = ix[(m >= 1080) | (m < 570)]
        rth = ix[(m >= 570) & (m < 960)]
        lv = {}
        if len(on) > 60 and vol[on].sum() > 0:
            lv["overnight"] = value_area(lo[on], hi[on], vol[on])
        if prev_rth is not None:
            lv["previous day"] = prev_rth
        out[d] = lv
        if len(rth) > 60 and vol[rth].sum() > 0:
            prev_rth = value_area(lo[rth], hi[rth], vol[rth])
    return out


def run(args) -> pd.DataFrame:
    which, confirm, rr, flow_only = args
    df = load()
    if flow_only:
        df = df[df.index >= df["big_sell"].first_valid_index()]
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    bb, bs = df["big_buy"].fillna(0).to_numpy(), df["big_sell"].fillna(0).to_numpy()
    by, se = df["buy"].fillna(0).to_numpy(), df["sell"].fillna(0).to_numpy()
    n = len(c)
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    lv_all = levels(df, mins, day)
    roll = lambda x: pd.Series(x).rolling(5, min_periods=1).sum().to_numpy()  # noqa: E731
    bb5, bs5, by5, se5 = roll(bb), roll(bs), roll(by), roll(se)
    out = []
    pos = None
    cur = None
    for i in range(1, n):
        if day[i] != cur:
            cur, trades_today = day[i], 0
            lv = lv_all.get(cur, {})
            names = ["overnight", "previous day"] if which == "both" else [which]
            lines = [(nm, -1, lv[nm][2]) for nm in names if nm in lv] + [(nm, 1, lv[nm][0]) for nm in names if nm in lv]
            armed = {}
            used = set()
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                px, how = tgt, "target"
            elif i + 1 >= n or day[i + 1] != day[i] or mins[i + 1] >= 1000:
                px, how = c[i], "16:40 close"
            if px is not None:
                out.append((df.index[ie], df.index[i], side, abs(ent - stp), (side * (px - ent) - COMM - SPREAD) / abs(ent - stp), how))
                pos = None
        if not (570 <= mins[i] < 930) or trades_today >= 2:
            continue
        for k, (nm, side, lvl) in enumerate(lines):
            if k in used:
                continue
            if k not in armed:
                if (h[i] >= lvl) if side < 0 else (l[i] <= lvl):
                    armed[k] = {"t": i, "ext": h[i] if side < 0 else l[i]}
                continue
            a = armed[k]
            a["ext"] = max(a["ext"], h[i]) if side < 0 else min(a["ext"], l[i])
            if i - a["t"] > 15:
                used.add(k)
                continue
            back = (c[i] < lvl) if side < 0 else (c[i] > lvl)
            if not back:
                continue
            if confirm == "orbs":
                ok = (bs5[i] >= 2 * max(bb5[i], 1) and bs5[i] > 0) if side < 0 else (bb5[i] >= 2 * max(bs5[i], 1) and bb5[i] > 0)
            elif confirm == "delta":
                ok = (se5[i] > by5[i]) if side < 0 else (by5[i] > se5[i])
            else:
                ok = True
            if not ok or pos is not None:
                continue
            ent = c[i]
            stp = (a["ext"] + 2 * TICK) if side < 0 else (a["ext"] - 2 * TICK)
            if side * (ent - stp) < 4.0:
                stp = ent - side * 4.0
            risk = side * (ent - stp)
            pos = (side, ent, stp, ent + side * rr * risk, i)
            used.add(k)
            trades_today += 1
            break
    return pd.DataFrame(out, columns=["opened", "closed", "side", "risk", "R", "how"])


def summary(t: pd.DataFrame, lo=None, hi=None) -> tuple[int, float, float, float]:
    if lo is not None:
        t = t[(t["opened"] >= pd.Timestamp(lo, tz="UTC")) & (t["opened"] < pd.Timestamp(hi, tz="UTC"))]
    n = len(t)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    r = t["R"].to_numpy()
    return n, float((r > 0).mean()), float(r.mean()), float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))


def main() -> int:
    grid = [(w, cf, rr, True) for w, cf, rr in itertools.product(("overnight", "previous day", "both"),
                                                                 ("orbs", "delta", "none"), (1.0, 1.4))]
    grid += [(w, "none", rr, False) for w, rr in itertools.product(("overnight", "previous day", "both"), (1.0, 1.4))]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        split = "2026-04-10" if g[3] else "2025-07-01"
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        rows.append({"period": "Oct 2025-Oct 2026 (with order flow)" if g[3] else "Mar 2024-Oct 2026", "levels": g[0],
                     "confirmation": g[1], "target": f"{g[2]:g}R", "trades": n, "winning%": w, "avg R": r,
                     "first half R": ra, "z first": za, "second half R": rb, "z second": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "first half R", "second half R")}
    fm.update({"z first": lambda v: f"{v:+.2f}", "z second": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}"})
    print(res.to_string(index=False, formatters=fm))
    for cf in ("orbs", "delta", "none"):
        part = res[(res["confirmation"] == cf) & res["period"].str.contains("order flow")]
        print(f"order-flow year, confirmation {cf}: average {part['avg R'].mean():+.3f} R, win {part['winning%'].mean():.0%}")
    print(f"all {len(res)} versions: average {res['avg R'].mean():+.3f} R; second half positive in "
          f"{(res['second half R'] > 0).mean():.0%}")
    res.to_csv(ROOT / "data" / "research_volume_profile.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
