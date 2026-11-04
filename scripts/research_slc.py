"""'SLC' -- Structure, Level, Confirmation (owner's second video, 2026-10-08), tested as described.

  S  4-hour structure: the last two confirmed swing highs AND lows both rising = uptrend (longs only),
     both falling = downtrend (shorts only), anything else = consolidation (no trades). Swings are
     4-hour highs/lows that are the extreme of 2 bars on each side, known only 2 bars later; a
     5-minute bar uses only 4-hour bars already closed.
  L  5-minute supply/demand levels: the candle right before a sharp move (body >= *k x ATR(14),
     k = 1.5 / 2 / 2.5), from its low to its high; levels from the last 24 hours. Scenario 1: a fresh
     level's first retest. Scenario 2 (*on/off): a level broken once the other way, then broken back
     cleanly and retested from the other side; chopped through twice = dead.
  C  Stochastic (*%K 5 / 9 / 14, smoothed 3, levels 80/20 -- the video's "custom settings" are not
     shown): while price is at the level (*within 6 or 12 bars of the touch), the %K line goes above
     80 and then crosses back below it (shorts; mirrored below 20 for longs). Entry at that bar's
     close, stop 1 tick beyond the level, target 2R.
Sessions (*): entries 09:30-15:30 New York only, or any hour; everything flat by 16:40 New York
(Lucid). One position at a time. Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt spread,
1 tick slippage on stops; a bar touching stop and target counts as the stop.
Data: NinjaTrader 5-minute NQ Feb 2021 - Oct 2026 (choose settings on 2021-2023, judge on 2024-2026),
5-minute ES Dec 2023 - Oct 2026 as a cross-check.

Usage:  .venv/Scripts/python.exe scripts/research_slc.py
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
TICK, SPREAD, RR, MAX_AGE = 0.25, 0.25, 2.0, 288          # 288 five-minute bars = 24 h
COMM = {"NQ": 0.175, "ES": 0.07}


def load(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(F / f"{sym}_5m_long_nt.parquet")[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def htf_structure(df: pd.DataFrame, p: int = 2) -> np.ndarray:
    b = df.resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h, l = b["high"].to_numpy(), b["low"].to_numpy()
    state = np.zeros(len(b))
    shs, sls = [], []
    for i in range(len(b)):
        j = i - p
        if j >= p:
            if h[j] == h[j - p:i + 1].max() and h[j] > h[j - p:j].max():
                shs.append(h[j])
            if l[j] == l[j - p:i + 1].min() and l[j] < l[j - p:j].min():
                sls.append(l[j])
        if len(shs) >= 2 and len(sls) >= 2:
            if shs[-1] > shs[-2] and sls[-1] > sls[-2]:
                state[i] = 1
            elif shs[-1] < shs[-2] and sls[-1] < sls[-2]:
                state[i] = -1
    closes = pd.DataFrame({"t": (b.index + pd.Timedelta(hours=4)).as_unit("ns"), "v": state})
    m = pd.merge_asof(pd.DataFrame({"t": df.index}), closes, on="t", direction="backward")
    return m["v"].fillna(0).to_numpy()


def stoch_k(h, l, c, n: int, s: int = 3) -> np.ndarray:
    hh, ll = pd.Series(h).rolling(n).max(), pd.Series(l).rolling(n).min()
    raw = 100 * (pd.Series(c) - ll) / (hh - ll).replace(0, np.nan)
    return raw.rolling(s).mean().to_numpy()


def run(args) -> pd.DataFrame:
    sym, k, n_st, win, scen2, rth_only = args
    df = load(sym)
    htf = htf_structure(df)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    K = stoch_k(h, l, c, n_st)
    comm = COMM[sym]
    zones: list[dict] = []
    pos = None
    out = []
    for i in range(20, len(c)):
        # --- the open position: forced flat 16:40-18:00 New York, else stop first, then target ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if 1000 <= mins[i] < 1080:
                px, how = o[i], "16:40 close"
            elif (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif (h[i] > tgt) if side > 0 else (l[i] < tgt):
                px, how = tgt, "target"
            if px is not None:
                risk = abs(ent - stp)
                out.append((df.index[ie], df.index[i], side, risk, (side * (px - ent) - comm - SPREAD) / risk, how))
                pos = None
        # --- levels: touch -> confirmation window; break states for scenario 2 ---
        can_enter = (570 <= mins[i] < 930) if rth_only else not (990 <= mins[i] < 1080)
        keep = []
        for z in zones:
            if i - z["made"] > MAX_AGE:
                continue
            side, lo, hi = z["side"], z["lo"], z["hi"]
            beyond_far = (c[i] > hi) if side < 0 else (c[i] < lo)      # closed through the level, against
            beyond_near = (c[i] < lo) if side < 0 else (c[i] > hi)     # closed cleanly through, our way
            touched = (h[i] >= lo) if side < 0 else (l[i] <= hi)
            st = z["state"]
            if st in ("fresh", "rebroken") and i > z["made"] and touched:
                if st == "rebroken" and beyond_far:
                    continue                                           # chopped through twice: dead
                z.update(state="armed", t_arm=i, ext=False)
                st = "armed"
            elif st == "fresh" and beyond_far:
                if not scen2:
                    continue
                z["state"] = st = "broken1"
            elif st == "broken1":
                if beyond_near:
                    z["state"] = st = "rebroken"
                keep.append(z)
                continue
            elif st == "rebroken":
                if beyond_far:
                    continue
            if st == "armed":
                if beyond_far:
                    if scen2 and not z.get("second"):
                        z.update(state="broken1", second=True)
                        keep.append(z)
                    continue
                if i - z["t_arm"] > win:
                    continue                                           # this retest is over
                if not np.isnan(K[i]):
                    if (K[i] >= 80) if side < 0 else (K[i] <= 20):
                        z["ext"] = True
                    crossed = (K[i - 1] >= 80 > K[i]) if side < 0 else (K[i - 1] <= 20 < K[i])
                    if z["ext"] and crossed:
                        if pos is None and can_enter and htf[i] == side:
                            ent = c[i]
                            stp = (hi + TICK) if side < 0 else (lo - TICK)
                            risk = side * (ent - stp)
                            if risk > 0:
                                pos = (side, ent, stp, ent + side * RR * risk, i)
                        continue                                       # used
            keep.append(z)
        zones = keep[-40:]
        # --- a new level: the candle before a sharp move ---
        if not np.isnan(atr[i]) and abs(c[i] - o[i]) >= k * atr[i]:
            side = 1 if c[i] > o[i] else -1
            if (side > 0 and c[i] > h[i - 1]) or (side < 0 and c[i] < l[i - 1]):
                zones.append({"side": side, "lo": l[i - 1], "hi": h[i - 1], "made": i, "state": "fresh"})
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
    grid = list(itertools.product(["NQ"], (1.5, 2.0, 2.5), (5, 9, 14), (6, 12), (False, True), (True, False)))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        n, w, r, z = summary(t)
        na, wa, ra, za = summary(t, 2021, 2023)
        nb, wb, rb, zb = summary(t, 2024, 2026)
        rows.append({"sharp k": g[1], "stoch": g[2], "window": g[3], "broken-retest": g[4],
                     "hours": "NY 09:30-15:30" if g[5] else "any", "trades": n, "per year": n / 5.65,
                     "win%": w, "avg R": r, "2021-23 R": ra, "z A": za, "2024-26 R": rb, "z B": zb, "_t": t, "_g": g})
    res = pd.DataFrame(rows).sort_values("2021-23 R", ascending=False)
    pd.set_option("display.width", 230)
    fm = {c: (lambda v: f"{v:+.3f}") for c in ("avg R", "2021-23 R", "2024-26 R")}
    fm.update({"z A": lambda v: f"{v:+.2f}", "z B": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    cols = ["sharp k", "stoch", "window", "broken-retest", "hours", "trades", "per year", "win%", "avg R",
            "2021-23 R", "z A", "2024-26 R", "z B"]
    print("=== SLC on NQ 5-minute, Feb 2021 - Oct 2026 (break-even win rate at 2R ~ 33% before costs) ===")
    print(res[cols].to_string(index=False, formatters=fm))
    print(f"\nall {len(res)} versions: average {res['avg R'].mean():+.3f} R/trade; later years (2024-26) positive in "
          f"{(res['2024-26 R'] > 0).mean():.0%} of versions, average {res['2024-26 R'].mean():+.3f} R")
    best = res.iloc[0]
    t = best["_t"]
    print(f"\nCHOSEN on 2021-2023: sharp {best['sharp k']}x ATR, stochastic {best['stoch']}, window {best['window']} bars, "
          f"broken-retest {best['broken-retest']}, hours {best['hours']}")
    print(f"  NQ: 2021-23 {best['2021-23 R']:+.3f} R (z {best['z A']:+.2f}) -> 2024-26 {best['2024-26 R']:+.3f} R (z {best['z B']:+.2f})")
    by = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean", "sum"])
    print("  by year (trades / avg R / total R): " + ", ".join(f"{y}: {int(v['count'])}/{v['mean']:+.2f}/{v['sum']:+.0f}"
                                                            for y, v in by.iterrows()))
    days = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.date)["R"].sum()
    months = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m"))["R"].sum()
    streak = worst = 0
    for wn in (t["how"] == "target"):
        streak = 0 if wn else streak + 1
        worst = max(worst, streak)
    print(f"  profitable days {np.mean(days > 0):.0%} of {len(days)}, profitable months {np.mean(months > 0):.0%} of "
          f"{len(months)}, longest losing streak {worst}, median risk {t['risk'].median():.1f} pts "
          f"(${2 * t['risk'].median():,.0f} per MNQ, ${20 * t['risk'].median():,.0f} per NQ)")
    es = run(("ES",) + tuple(best["_g"][1:]))
    n, w, r, z = summary(es)
    print(f"  ES cross-check (Dec 2023 - Oct 2026, never used for choosing): {n} trades, win {w:.1%}, {r:+.3f} R (z {z:+.2f})")
    res.drop(columns=["_t", "_g"]).to_csv(ROOT / "data" / "research_slc.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
