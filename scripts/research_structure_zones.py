"""'The only trading strategy you'll ever need' (owner's video, 2026-10-08), tested as written.

The video's three steps, made exact (choices the video leaves open are swept, marked *):
  1. Market structure. Swing highs/lows = a bar's high (low) is the extreme of the p bars on each side
     (*p = 2 or 3), known only p bars later. Uptrend: when price breaks the last swing high (*by close
     or by wick), the lowest low since that swing high becomes the VALID LOW; the trend stays up until
     price breaks the valid low. Then downtrend, mirrored: the valid high is the highest high since the
     valid low was set, and so on.
  2. Supply & demand. In an uptrend, an impulse candle (body >= *k x ATR(14), k = 1, 1.5, 2) going up
     marks a DEMAND zone = high to low of the candle right before it. Downtrend: supply, mirrored.
     Only zones in the trend's direction; a zone is used once (first return), dropped when a candle
     closes through it or the trend flips.
  3. Entry when price comes back into the zone (limit at its near edge), stop 1 tick beyond the far
     edge, target = the recent high (highest high since the zone; recent low for shorts). Taken only if
     target >= 2.5 x risk. One position at a time.
Timeframes (*): 15 min, 1 h, 4 h. Costs: NQ $3.50 commission round trip (0.175 pt) + 0.25 pt spread on
every trade + 1 tick slippage on stops (ES: 0.07 pt commission). A bar touching stop and target counts
as the stop; no target on the entry bar.
Data: nas100 / us500 15-min stand-ins 2016-2026 (choose settings on 2016-2022, judge on 2023-2026) and
the real NinjaTrader NQ / ES futures bars 2021-2026.

Usage:  .venv/Scripts/python.exe scripts/research_structure_zones.py
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICK = 0.25
MARKETS = {   # name -> (file, commission round trip in points)
    "NQ stand-in 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175),
    "NQ futures 2021-26": (ROOT / "data" / "futures" / "NQ_15m_long_nt.parquet", 0.175),
    "ES stand-in 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07),
    "ES futures 2021-26": (ROOT / "data" / "futures" / "ES_15m_long_nt.parquet", 0.07),
}
SPREAD = 0.25
RR_MIN = 2.5


def bars(path: Path, tf: str) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df[~df.index.duplicated()].sort_index()
    if tf != "15min":
        df = df.resample(tf, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return df


def run(df: pd.DataFrame, k: float, wick: bool, p: int, comm: float) -> pd.DataFrame:
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    n = len(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    trend = 0
    sh = sl = None            # last confirmed swing high / low: (level, index)
    sh_used = sl_used = True  # a swing level breaks the structure only once
    valid_low = valid_high = np.nan
    vl_idx = vh_idx = 0
    zones: list[list] = []    # [side, near, far, created, tp_from]
    pos = None                # (side, entry, stop, target, i_entry, zone_created)
    out = []
    for i in range(30, n):
        # --- a position: stop first, then target (never on the entry bar) ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            hit_stop = (l[i] <= stp) if side > 0 else (h[i] >= stp)
            hit_tgt = i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt))
            if hit_stop or hit_tgt:
                px = stp - side * TICK if hit_stop else tgt
                risk = abs(ent - stp)
                pts = side * (px - ent) - comm - SPREAD
                out.append((df.index[ie], df.index[i], side, ent, stp, tgt, px, pts / risk, "stop" if hit_stop else "target"))
                pos = None
        # --- swings confirmed p bars later ---
        j = i - p
        if j >= p:
            if h[j] == h[j - p:i + 1].max() and h[j] > h[j - p:j].max():
                sh, sh_used = (h[j], j), False
            if l[j] == l[j - p:i + 1].min() and l[j] < l[j - p:j].min():
                sl, sl_used = (l[j], j), False
        up_brk = lambda lvl: (h[i] > lvl) if wick else (c[i] > lvl)   # noqa: E731
        dn_brk = lambda lvl: (l[i] < lvl) if wick else (c[i] < lvl)   # noqa: E731
        # --- structure ---
        if trend >= 0 and not np.isnan(valid_low) and dn_brk(valid_low):
            trend, valid_high, vh_idx = -1, h[vl_idx:i + 1].max(), vl_idx + int(np.argmax(h[vl_idx:i + 1]))
            valid_low = np.nan
            zones = [z for z in zones if z[0] < 0]
        elif trend <= 0 and not np.isnan(valid_high) and up_brk(valid_high):
            trend, valid_low, vl_idx = 1, l[vh_idx:i + 1].min(), vh_idx + int(np.argmin(l[vh_idx:i + 1]))
            valid_high = np.nan
            zones = [z for z in zones if z[0] > 0]
        if sh is not None and not sh_used and up_brk(sh[0]) and trend >= 0:
            seg = l[sh[1]:i + 1]
            valid_low, vl_idx, sh_used = seg.min(), sh[1] + int(np.argmin(seg)), True
            if trend == 0:
                trend = 1
        if sl is not None and not sl_used and dn_brk(sl[0]) and trend <= 0:
            seg = h[sl[1]:i + 1]
            valid_high, vh_idx, sl_used = seg.max(), sl[1] + int(np.argmax(seg)), True
            if trend == 0:
                trend = -1
        # --- entries: first return into a zone of the trend's side ---
        keep = []
        for z in zones:
            side, near, far, made, _ = z
            if side != trend:
                continue
            if (side > 0 and c[i] < far) or (side < 0 and c[i] > far):
                continue                                  # closed through: zone gone
            touched = (l[i] <= near) if side > 0 else (h[i] >= near)
            if i > made and touched:
                if pos is None:
                    ent = min(o[i], near) if side > 0 else max(o[i], near)
                    stp = far - side * TICK
                    tgt = (h[made:i].max() if side > 0 else l[made:i].min())
                    risk = side * (ent - stp)
                    if risk > 0 and side * (tgt - ent) >= RR_MIN * risk:
                        if (l[i] <= stp) if side > 0 else (h[i] >= stp):          # stopped on the entry bar
                            pts = -risk - TICK - comm - SPREAD
                            out.append((df.index[i], df.index[i], side, ent, stp, tgt, stp - side * TICK, pts / risk, "stop"))
                        else:
                            pos = (side, ent, stp, tgt, i)
                continue                                  # used once, traded or not
            keep.append(z)
        zones = keep
        # --- new zone: impulse candle in the trend's direction, zone = the candle before it ---
        if trend != 0 and not np.isnan(atr[i]) and trend * (c[i] - o[i]) >= k * atr[i]:
            near, far = (h[i - 1], l[i - 1]) if trend > 0 else (l[i - 1], h[i - 1])
            if trend * (c[i] - near) > 0:
                zones.append([trend, near, far, i, i])
    return pd.DataFrame(out, columns=["opened", "closed", "side", "entry", "stop", "target", "exit", "R", "how"])


def stats(t: pd.DataFrame, years: tuple[int, int] | None = None) -> dict:
    if years:
        y = t["opened"].dt.year
        t = t[(y >= years[0]) & (y <= years[1])]
    n = len(t)
    if not n:
        return {"trades": 0, "win%": np.nan, "avg R": np.nan, "z": np.nan}
    r = t["R"].to_numpy()
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    return {"trades": n, "win%": float((t["how"] == "target").mean()), "avg R": float(r.mean()),
            "z": float(r.mean() / se) if se and se > 0 else np.nan}


def main() -> int:
    pd.set_option("display.width", 220)
    grid = list(itertools.product(("15min", "1h", "4h"), (1.0, 1.5, 2.0), (False, True), (2, 3)))
    rows = []
    cache = {}
    for name, (path, comm) in MARKETS.items():
        for tf, k, wick, p in grid:
            key = (name, tf)
            if key not in cache:
                cache[key] = bars(path, tf)
            t = run(cache[key], k, wick, p, comm)
            first = "2016-2022" if "2016" in name else "2021-2023"
            a = stats(t, (2016, 2022) if "2016" in name else (2021, 2023))
            b = stats(t, (2023, 2026) if "2016" in name else (2024, 2026))
            rows.append({"data": name, "tf": tf, "impulse k": k, "break": "wick" if wick else "close", "swing p": p,
                         "trades": len(t), "win% all": stats(t)["win%"], "avg R all": stats(t)["avg R"],
                         "A trades": a["trades"], "A avg R": a["avg R"], "A z": a["z"],
                         "B trades": b["trades"], "B avg R": b["avg R"], "B z": b["z"], "_t": t, "_first": first})
    res = pd.DataFrame(rows)
    f = {c: (lambda v: f"{v:+.3f}") for c in ("avg R all", "A avg R", "B avg R")}
    f.update({c: (lambda v: f"{v:+.2f}") for c in ("A z", "B z")})
    f["win% all"] = lambda v: f"{v:.1%}"
    cols = ["data", "tf", "impulse k", "break", "swing p", "trades", "win% all", "avg R all",
            "A trades", "A avg R", "A z", "B trades", "B avg R", "B z"]
    for name in MARKETS:
        part = res[res["data"] == name].sort_values("A avg R", ascending=False)
        print(f"\n=== {name}: A = {part['_first'].iloc[0]} (choose), B = later years (judge); "
              f"break-even win rate at 2.5R+ targets ~ 29% before costs ===")
        print(part[cols].to_string(index=False, formatters=f))
    # the setting chosen on the NQ stand-in's 2016-2022 years, judged everywhere else
    nq = res[res["data"] == "NQ stand-in 2016-26"].sort_values("A avg R", ascending=False).iloc[0]
    print(f"\nCHOSEN on NQ 2016-2022: {nq['tf']}, impulse {nq['impulse k']}x ATR, break by {nq['break']}, "
          f"swing p={nq['swing p']}: A {nq['A avg R']:+.3f} R/trade over {nq['A trades']} trades")
    for name in MARKETS:
        r = res[(res["data"] == name) & (res["tf"] == nq["tf"]) & (res["impulse k"] == nq["impulse k"])
                & (res["break"] == nq["break"]) & (res["swing p"] == nq["swing p"])].iloc[0]
        t = r["_t"]
        by_year = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean", "sum"])
        months = t.groupby(t["opened"].dt.to_period("M"))["R"].sum()
        print(f"  {name}: {r['trades']} trades, win {r['win% all']:.1%}, avg {r['avg R all']:+.3f} R; "
              f"later years {r['B avg R']:+.3f} R (z {r['B z']:+.2f}); profitable months {np.mean(months > 0):.0%}"
              f" of {len(months)}")
        print("    by year (trades / avg R / total R): " + ", ".join(
            f"{y}: {int(v['count'])}/{v['mean']:+.2f}/{v['sum']:+.0f}" for y, v in by_year.iterrows()))
    res.drop(columns=["_t", "_first"]).to_csv(ROOT / "data" / "research_structure_zones.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
