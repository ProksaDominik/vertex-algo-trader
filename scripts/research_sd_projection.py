"""ICT 'standard deviation' reversal (owner's eleventh video, 2026-10-08): buy at -1 SD of a manipulation leg.

The video's trade, made exact (longs; shorts mirrored):
  * Manipulation leg: confirmed swings (extreme of 2 bars each side, known 2 bars later) in the order
    high H1, low L1, a HIGHER high H2 > H1, then a LOWER low L2 < L1. Leg size S = H2 - L2.
  * -1 standard deviation = L2 - S (the leg projected once more below its low). Buy there (limit) while
    the leg's high H2 has not been taken (the "draw on liquidity" above).
  * Stop (* the video gives none): -2 SD (L2 - 2S) or -2.5 SD (L2 - 2.5S).
  * Target (*): the leg's high H2 (the highs he aims for) or the leg's low L2.
  Setups expire after 48 bars or when price trades above H2 first.
Sessions: any hour (he traded in Asia); flat on the session's last bar before 16:40 New York (Lucid).
One position at a time. Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt spread; 1 tick slippage
on stops; stop first if a bar reaches both; no target on the entry bar; stops under 4 ticks / a quarter
ATR are skipped.
Data: NQ / ES 15-min stand-ins 2016-2026 (choose 2016-2022, judge 2023-2026), NinjaTrader NQ 5-min
2021-2026 (choose 2021-2023, judge 2024-2026).

Usage:  .venv/Scripts/python.exe scripts/research_sd_projection.py
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


def run(args) -> pd.DataFrame:
    name, stop_sd, target = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    p = 2
    swings: list[tuple[str, float, int]] = []          # ('H'|'L', level, index), in order of confirmation
    setups: list[dict] = []
    out = []
    pos = None
    for i in range(30, n):
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
                out.append((df.index[ie], df.index[i], side, abs(ent - stp), (side * (px - ent) - comm - SPREAD) / abs(ent - stp), how))
                pos = None
        j = i - p
        new = None
        if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:i + 1].max():
            new = ("H", h[j], j)
        elif l[j] < l[j - p:j].min() and l[j] <= l[j + 1:i + 1].min():
            new = ("L", l[j], j)
        if new is not None:
            if swings and swings[-1][0] == new[0]:          # keep alternating highs and lows: keep the extreme
                if (new[0] == "H" and new[1] >= swings[-1][1]) or (new[0] == "L" and new[1] <= swings[-1][1]):
                    swings[-1] = new
            else:
                swings.append(new)
            swings = swings[-6:]
            if len(swings) >= 4:
                a, b, cc, d = swings[-4:]
                if (a[0], b[0], cc[0], d[0]) == ("H", "L", "H", "L") and cc[1] > a[1] and d[1] < b[1]:
                    s_ = cc[1] - d[1]                         # bullish leg: higher high, then lower low
                    setups.append({"side": 1, "lvl": d[1] - s_, "stp": d[1] - stop_sd * s_,
                                   "tgt": cc[1] if target == "leg high" else d[1], "draw": cc[1], "made": i})
                elif (a[0], b[0], cc[0], d[0]) == ("L", "H", "L", "H") and cc[1] < a[1] and d[1] > b[1]:
                    s_ = d[1] - cc[1]                         # bearish leg: lower low, then higher high
                    setups.append({"side": -1, "lvl": d[1] + s_, "stp": d[1] + stop_sd * s_,
                                   "tgt": cc[1] if target == "leg high" else d[1], "draw": cc[1], "made": i})
                setups = setups[-4:]
        keep = []
        for s in setups:
            side = s["side"]
            if i - s["made"] > 48 or ((h[i] > s["draw"]) if side > 0 else (l[i] < s["draw"])):
                continue                                      # expired, or the draw was taken without us
            if (l[i] <= s["lvl"]) if side > 0 else (h[i] >= s["lvl"]):
                if pos is None and not (1000 <= mins[i] < 1080):
                    ent = min(o[i], s["lvl"]) if side > 0 else max(o[i], s["lvl"])
                    risk = side * (ent - s["stp"])
                    if not np.isnan(atr[i]) and risk >= max(4 * TICK, 0.25 * atr[i]) and side * (s["tgt"] - ent) > 0:
                        if (l[i] <= s["stp"]) if side > 0 else (h[i] >= s["stp"]):
                            out.append((df.index[i], df.index[i], side, risk, (-risk - TICK - comm - SPREAD) / risk, "stop"))
                        else:
                            pos = (side, ent, s["stp"], s["tgt"], i)
                continue                                      # used
            keep.append(s)
        setups = keep
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
    grid = list(itertools.product(DATA, (2.0, 2.5), ("leg high", "leg low")))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        rows.append({"data": g[0], "stop": f"-{g[1]:g} SD", "target": g[2], "trades": n, "per year": n / years,
                     "winning%": w, "avg R": r, "older R": ra, "z older": za, "newer R": rb, "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    print(res.to_string(index=False, formatters=fm))
    print(f"\nall {len(res)} versions: average {res['avg R'].mean():+.3f} R; newer years positive in "
          f"{(res['newer R'] > 0).mean():.0%} (average {res['newer R'].mean():+.3f} R)")
    res.to_csv(ROOT / "data" / "research_sd_projection.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
