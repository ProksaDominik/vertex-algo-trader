"""30-minute opening range: break, retest, go (owner's tenth video, 2026-10-08), as described.

  1. Range = high / low of 09:30-10:00 New York.
  2. A candle CLOSES above the range high (longs; shorts mirrored at the low).
  3. Later price comes back to that level and a candle that touched it closes back above it, in the top
     40% of its range ("made it support") -> enter at that close. Retests count until 15:00; the setup
     ends if a candle closes back inside the range by more than the level's own candle would allow
     (a close below the level that is not a touch-and-hold).
  4. Stop 1 tick below that candle's low (stops under 4 ticks or a quarter ATR are skipped as unrealistic).
  5. Exit (*): all at 1R; all at 2R; or half at 1R, stop to break-even, the rest to the end of the day.
One trade a day, flat on the session's last bar before 16:40 New York (Lucid). Costs: $3.50 round trip
(0.175 pt NQ, 0.07 ES) + 0.25 pt spread; 1 tick slippage on stops; a bar reaching stop and target counts
as the stop; no target on the entry bar.
Data: NQ / ES 15-min stand-ins 2016-2026 (10 years; choose 2016-2022, judge 2023-2026), NinjaTrader NQ
5-min 2021-2026 (choose 2021-2023) and ES 5-min Dec 2023 - Oct 2026 (choose to Mar 2025).

Usage:  .venv/Scripts/python.exe scripts/research_orb30_retest.py
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
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01", 10.6),
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01", 10.6),
    "ES 5-min 2023-26": (ROOT / "data" / "futures" / "ES_5m_long_nt.parquet", 0.07, "2025-04-01", 2.8),
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01", 5.65),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def run(args) -> pd.DataFrame:
    name, exit_style = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray(et.date)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    out = []
    pos = None
    cur = None
    for i in range(20, n):
        if day[i] != cur:
            cur, r_hi, r_lo, st = day[i], -np.inf, np.inf, {"s": "range"}
        if pos is not None:
            p_ = pos
            side, ent = p_["side"], p_["ent"]
            px = how = None
            if (l[i] <= p_["stp"]) if side > 0 else (h[i] >= p_["stp"]):
                px, how = p_["stp"] - side * TICK, "break-even" if p_["half"] else "stop"
            elif i > p_["ie"] and not p_["half"] and ((h[i] > p_["tgt"]) if side > 0 else (l[i] < p_["tgt"])):
                if exit_style == "half":
                    p_.update(half=True, banked=0.5 * side * (p_["tgt"] - ent), stp=ent)
                else:
                    px, how = p_["tgt"], "target"
            if px is None and (i + 1 >= n or day[i + 1] != day[i] or mins[i + 1] >= 1000):
                px, how = c[i], "16:40 close"
            if px is not None:
                frac = 0.5 if p_["half"] else 1.0
                pts = p_.get("banked", 0.0) + frac * side * (px - ent) - comm - SPREAD
                out.append((df.index[p_["ie"]], df.index[i], side, p_["risk"], pts / p_["risk"], how))
                pos = None
        m = mins[i]
        if 570 <= m < 600:
            r_hi, r_lo = max(r_hi, h[i]), min(r_lo, l[i])
            continue
        if pos is not None or st["s"] == "done" or m < 600 or m >= 900 or not np.isfinite(r_hi):
            continue
        if st["s"] == "range":
            if c[i] > r_hi:
                st = {"s": "broken", "side": 1, "lvl": r_hi}
            elif c[i] < r_lo:
                st = {"s": "broken", "side": -1, "lvl": r_lo}
            continue
        side, lvl = st["side"], st["lvl"]
        touched = (l[i] <= lvl) if side > 0 else (h[i] >= lvl)
        rng = h[i] - l[i]
        if not touched:
            continue
        holds = rng > 0 and ((c[i] > lvl and c[i] >= l[i] + 0.6 * rng) if side > 0 else (c[i] < lvl and c[i] <= h[i] - 0.6 * rng))
        if not holds:
            if (c[i] < lvl) if side > 0 else (c[i] > lvl):
                st = {"s": "done"}                               # closed back inside: the break failed
            continue
        ent = c[i]
        stp = (l[i] - TICK) if side > 0 else (h[i] + TICK)
        risk = side * (ent - stp)
        st = {"s": "done"}
        if np.isnan(atr[i]) or risk < max(4 * TICK, 0.25 * atr[i]):
            continue
        mult = 2.0 if exit_style == "2R" else 1.0
        pos = {"side": side, "ent": ent, "stp": stp, "tgt": ent + side * mult * risk, "risk": risk, "ie": i, "half": False}
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
    grid = list(itertools.product(DATA, ("1R", "2R", "half")))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        usd = (t["R"] * t["risk"]).mean() * (5.0 if g[0].startswith("ES") else 2.0) if len(t) else np.nan
        rows.append({"data": g[0], "exit": {"half": "half at 1R + runner"}.get(g[1], g[1]), "trades": n,
                     "per year": n / years, "winning%": w, "avg R": r, "$/trade per micro": usd,
                     "older R": ra, "z older": za, "newer R": rb, "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}", "$/trade per micro": lambda v: f"{v:+.2f}"})
    print(res.to_string(index=False, formatters=fm))
    print(f"\nall {len(res)} versions: average {res['avg R'].mean():+.3f} R; newer years positive in "
          f"{(res['newer R'] > 0).mean():.0%} (average {res['newer R'].mean():+.3f} R)")
    res.to_csv(ROOT / "data" / "research_orb30_retest.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
