"""200 DEMA + Supertrend(12, 3) (owner's seventh video, 2026-10-08), as described.

  Long when the close is above the 200-period double EMA and the Supertrend (ATR 12, multiplier 3,
  TradingView's formula) flips to buy -- entry at that candle's close. Stop: the Supertrend line at
  entry (hard, 1 tick slippage). Exit: when the Supertrend flips to sell (at that close). Shorts
  mirrored.
  Video tips (*): also enter when the Supertrend is already on buy and the close crosses above the
  DEMA; exit early when price touches the outer Fibonacci Bollinger band (200-period average of the
  typical price +- 3 standard deviations; a simple average -- the 10-year data has no reliable volume).
  Holding (*): until the flip, as in the video, or flat on the session's last bar before 16:40 New York
  (Lucid).
Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt spread. R = entry to the stop.
Data: NQ / ES 15-min stand-ins 2016-2026 (10 years; choose 2016-2022, judge 2023-2026); NinjaTrader NQ
5-min 2021-2026 (choose 2021-2023, judge 2024-2026). (The video's DOGE / LTC are not traded at Lucid.)

Usage:  .venv/Scripts/python.exe scripts/research_dema_supertrend.py
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


def supertrend(h, l, c, period=12, mult=3.0):
    n = len(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()       # TradingView's RMA
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    trend = np.ones(n, int)
    for i in range(1, n):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        if trend[i - 1] == -1 and c[i] > dn[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and c[i] < up[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    line = np.where(trend == 1, up, dn)
    return trend, line


def run(args) -> pd.DataFrame:
    name, late_entry, fbb, flat = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    e1 = pd.Series(c).ewm(span=200, adjust=False).mean()
    dema = (2 * e1 - e1.ewm(span=200, adjust=False).mean()).to_numpy()
    trend, line = supertrend(h, l, c)
    tp = (h + l + c) / 3
    basis = pd.Series(tp).rolling(200).mean().to_numpy()
    sd = pd.Series(tp).rolling(200).std().to_numpy()
    upper, lower = basis + 3 * sd, basis - 3 * sd
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    out = []
    pos = None
    for i in range(400, n):
        if pos is not None:
            side, ent, stp, ie, risk = pos
            px = how = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = (min(o[i], stp) if side > 0 else max(o[i], stp)) - side * TICK, "stop"
            elif fbb and ((h[i] >= upper[i]) if side > 0 else (l[i] <= lower[i])):
                px, how = (max(o[i], upper[i]) if side > 0 else min(o[i], lower[i])), "Bollinger band"
            elif trend[i] != side:
                px, how = c[i], "Supertrend flip"
            elif flat and (i + 1 >= n or day[i + 1] != day[i] or 1000 <= mins[i + 1] < 1080):
                px, how = c[i], "16:40 close"
            if px is not None:
                out.append((df.index[ie], df.index[i], side, risk, (side * (px - ent) - comm - SPREAD) / risk, how))
                pos = None
        if pos is not None or np.isnan(dema[i]):
            continue
        if flat and (1000 <= mins[i] < 1080 or i + 1 >= n or day[i + 1] != day[i] or 1000 <= mins[i + 1] < 1080):
            continue                                            # no new trade on the session's last bar
        flip = trend[i] != trend[i - 1]
        side = trend[i]
        above = (c[i] > dema[i]) if side > 0 else (c[i] < dema[i])
        crossed = above and ((c[i - 1] <= dema[i - 1]) if side > 0 else (c[i - 1] >= dema[i - 1]))
        if (flip and above) or (late_entry and crossed):
            stp = line[i]
            risk = side * (c[i] - stp)
            if risk > 0:
                pos = (side, c[i], stp, i, risk)
    return pd.DataFrame(out, columns=["opened", "closed", "side", "risk", "R", "how"])


def streaks(r: np.ndarray) -> tuple[int, int]:
    best_w = best_l = w = lo = 0
    for x in r:
        w, lo = (w + 1, 0) if x > 0 else (0, lo + 1)
        best_w, best_l = max(best_w, w), max(best_l, lo)
    return best_w, best_l


def summary(t: pd.DataFrame, lo=None, hi=None) -> tuple[int, float, float, float]:
    if lo is not None:
        t = t[(t["opened"] >= pd.Timestamp(lo, tz="UTC")) & (t["opened"] < pd.Timestamp(hi, tz="UTC"))]
    n = len(t)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    r = t["R"].to_numpy()
    return n, float((r > 0).mean()), float(r.mean()), float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))


def main() -> int:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    grid = list(itertools.product(DATA, (False, True), (False, True), (True, False)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        sw, sl = streaks(t["R"].to_numpy())
        rows.append({"data": g[0], "late DEMA entry": "yes" if g[1] else "no", "Bollinger exit": "yes" if g[2] else "no",
                     "hold": "flat 16:40 (Lucid)" if g[3] else "until flip", "trades": n, "per year": n / years,
                     "winning%": w, "avg R": r, "older R": ra, "z older": za, "newer R": rb, "z newer": zb,
                     "max wins/losses in a row": f"{sw}/{sl}", "_t": t})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    for name in DATA:
        part = res[res["data"] == name].sort_values("older R", ascending=False)
        print(f"\n=== {name}: older years choose, newer judge (split {DATA[name][2]}) ===")
        print(part.drop(columns=["_t", "data"]).to_string(index=False, formatters=fm))
        b = part.iloc[0]
        t = b["_t"]
        print(f"  all {len(part)} versions: average {part['avg R'].mean():+.3f} R; newer years positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R); exits of the chosen "
              f"version {t['how'].value_counts().to_dict()}")
        if "15-min" in name:
            by = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean"])
            print("  chosen by year (trades / avg R): " + ", ".join(f"{y}: {int(v['count'])}/{v['mean']:+.2f}"
                                                               for y, v in by.iterrows()))
    res.drop(columns=["_t"]).to_csv(ROOT / "data" / "research_dema_supertrend.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
