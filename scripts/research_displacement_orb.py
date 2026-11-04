"""'Displacement ORB' + session analysis (owner's eighth video, 2026-10-08), as described.

Longs (shorts mirrored):
  1. Opening range = high/low of 09:30-09:45 New York.
  2. Displacement: a candle CLOSES above the range as part of >= 3 bullish candles in a row, each with a
     body >= 0.3 x ATR(14) and together >= 1.5 x ATR (the "3-4 big green candles"); checked until 11:00.
  3. Zone (*): the last bearish candle before the run (demand / order block, its full range) or the
     first fair value gap inside the run.
  4. Entry (*), after price comes back into the zone (within 2 hours, before 12:00): "holding" = a
     bullish candle that touched the zone closes above it; or only a bullish engulfing candle that
     touched the zone. Entry at that close; the zone is dead if a candle closes below it.
  5. Stop 1 tick below the zone; target (*) 1.5R or 2R.
  Session filter (* on/off): Asia range 20:00-24:00 (the evening before), London 02:00-08:00. London
     took only the Asia low -> New York longs only; only the high -> shorts only; both -> London's own
     direction (08:00 close vs 02:00 open); neither -> no trade.
One trade a day, flat by 16:40 New York. Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt
spread; 1 tick slippage on stops; a bar reaching stop and target counts as the stop; no target on the
entry bar.
Data: NinjaTrader NQ 5-min 2021-2026 (choose 2021-2023, judge 2024-2026), ES 5-min Dec 2023 - Oct 2026
(check), and a 10-year check with 15-minute candles (NQ / ES stand-ins 2016-2026, choose 2016-2022).

Usage:  .venv/Scripts/python.exe scripts/research_displacement_orb.py
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
DATA = {   # name -> (file, commission round trip in points, split, years, minutes per bar)
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01", 5.65, 5),
    "ES 5-min 2023-26": (ROOT / "data" / "futures" / "ES_5m_long_nt.parquet", 0.07, "2025-04-01", 2.8, 5),
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01", 10.6, 15),
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01", 10.6, 15),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def session_bias(df: pd.DataFrame, mins: np.ndarray, day: np.ndarray) -> dict:
    """CME day -> +1 (New York longs), -1 (shorts) or 0 (no trade) from the Asia / London pattern."""
    f = pd.DataFrame({"o": df["open"].to_numpy(), "h": df["high"].to_numpy(), "l": df["low"].to_numpy(),
                      "c": df["close"].to_numpy(), "m": mins, "d": day})
    asia = f[f.m >= 1200].groupby("d").agg(h=("h", "max"), l=("l", "min"))
    lon = f[(f.m >= 120) & (f.m < 480)].groupby("d").agg(h=("h", "max"), l=("l", "min"), o=("o", "first"), c=("c", "last"))
    out = {}
    for d in lon.index:
        if d not in asia.index:
            continue
        hi_taken, lo_taken = lon.loc[d, "h"] > asia.loc[d, "h"], lon.loc[d, "l"] < asia.loc[d, "l"]
        if lo_taken and not hi_taken:
            out[d] = 1
        elif hi_taken and not lo_taken:
            out[d] = -1
        elif hi_taken and lo_taken:
            out[d] = 1 if lon.loc[d, "c"] > lon.loc[d, "o"] else -1
        else:
            out[d] = 0
    return out


def run(args) -> pd.DataFrame:
    name, zone_kind, trigger, rr, use_filter = args
    path, comm, _, _, bar_min = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    bias = session_bias(df, mins, day) if use_filter else {}
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    out = []
    pos = None
    cur = None
    for i in range(20, n):
        if day[i] != cur:
            cur, or_hi, or_lo, st, done = day[i], -np.inf, np.inf, None, False
        # --- open position: stop first, then target; flat on the session's last bar before 16:40 ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                px, how = tgt, "target"
            elif i + 1 >= n or day[i + 1] != day[i] or 1000 <= mins[i + 1] < 1080:
                px, how = c[i], "16:40 close"
            if px is not None:
                out.append((df.index[ie], df.index[i], side, abs(ent - stp),
                            (side * (px - ent) - comm - SPREAD) / abs(ent - stp), how))
                pos = None
        m = mins[i]
        if 570 <= m < 585:                                       # the opening range: 09:30-09:45
            or_hi, or_lo = max(or_hi, h[i]), min(or_lo, l[i])
            continue
        if done or pos is not None or not np.isfinite(or_hi) or m < 585 or m >= 720:
            continue
        allowed = bias.get(cur, 0) if use_filter else None
        # --- 2. displacement out of the range ---
        if st is None and m < 660 and not np.isnan(atr[i]):
            for side in (1, -1):
                if allowed is not None and allowed != side:
                    continue
                if not ((c[i] > or_hi) if side > 0 else (c[i] < or_lo)):
                    continue
                k = i
                while k > 0 and side * (c[k] - o[k]) >= 0.3 * atr[i] and i - k < 8:
                    k -= 1
                first = k + 1                                     # first candle of the run
                if i - first + 1 < 3 or side * (c[i] - o[first]) < 1.5 * atr[i]:
                    continue
                if zone_kind == "order block":
                    ob = next((j for j in range(first - 1, max(first - 6, 0), -1) if side * (c[j] - o[j]) < 0), None)
                    if ob is None:
                        continue
                    z_lo, z_hi = l[ob], h[ob]
                else:
                    fvg = None
                    for j in range(first + 2, i + 1):
                        if side > 0 and l[j] > h[j - 2]:
                            fvg = (h[j - 2], l[j])
                            break
                        if side < 0 and h[j] < l[j - 2]:
                            fvg = (h[j], l[j - 2])
                            break
                    if fvg is None:
                        continue
                    z_lo, z_hi = fvg
                st = {"side": side, "lo": z_lo, "hi": z_hi, "made": i}
                break
            continue
        if st is None:
            continue
        # --- 3-4. back into the zone, then the entry candle ---
        side, z_lo, z_hi = st["side"], st["lo"], st["hi"]
        if (i - st["made"]) * bar_min > 120:
            done = True
            continue
        if (c[i] < z_lo) if side > 0 else (c[i] > z_hi):
            done = True                                           # zone broken
            continue
        touched = (l[i] <= z_hi) if side > 0 else (h[i] >= z_lo)
        if not touched:
            continue
        bull = side * (c[i] - o[i]) > 0
        if trigger == "holding":
            ok = bull and ((c[i] > z_hi) if side > 0 else (c[i] < z_lo))
        else:
            prev_opp = side * (c[i - 1] - o[i - 1]) < 0
            ok = bull and prev_opp and abs(c[i] - o[i]) > abs(c[i - 1] - o[i - 1]) and (
                (c[i] >= o[i - 1] and o[i] <= c[i - 1]) if side > 0 else (c[i] <= o[i - 1] and o[i] >= c[i - 1]))
        if not ok:
            continue
        ent = c[i]
        stp = (z_lo - TICK) if side > 0 else (z_hi + TICK)
        risk = side * (ent - stp)
        done = True
        if risk > 0:
            pos = (side, ent, stp, ent + side * rr * risk, i)
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
    grid = list(itertools.product(DATA, ("order block", "fair value gap"), ("holding", "engulfing"), (1.5, 2.0), (True, False)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years, _ = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        rows.append({"data": g[0], "zone": g[1], "entry": g[2], "target": f"{g[3]:g}R",
                     "session filter": "on" if g[4] else "off", "trades": n, "per year": n / years, "win%": w,
                     "avg R": r, "older R": ra, "z older": za, "newer R": rb, "z newer": zb, "_t": t})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    for name in DATA:
        part = res[res["data"] == name].sort_values("older R", ascending=False)
        print(f"\n=== {name}: older choose, newer judge (split {DATA[name][2]}); the video claims a 79.45% win rate ===")
        print(part.drop(columns=["_t", "data"]).to_string(index=False, formatters=fm))
        b = part.iloc[0]
        print(f"  all {len(part)} versions: average {part['avg R'].mean():+.3f} R; newer positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R); best win rate of any "
              f"version {part['win%'].max():.0%}. Chosen: {b['zone']}, {b['entry']}, {b['target']}, filter "
              f"{b['session filter']}: older {b['older R']:+.3f} -> newer {b['newer R']:+.3f} R (z {b['z newer']:+.2f})")
    res.drop(columns=["_t"]).to_csv(ROOT / "data" / "research_displacement_orb.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
