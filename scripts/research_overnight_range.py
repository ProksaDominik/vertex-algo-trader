"""Overnight-range break (owner's sixth video, 2026-10-08): trade the break of the 18:00-09:30 high/low.

Rules as described:
  1. Overnight range: highest high and lowest low from 18:00 New York (futures open) to 09:30.
  2. After 09:30, a candle CLOSES above the overnight high (longs) or below the overnight low (shorts).
  3. Entry (*):  breakout -- at the close of that candle; or
                 retest   -- price comes back to the level and a bullish candle forms there (low within
                             1 tick of the level, close above it, close in the top 40% of its range) ->
                             at its close (mirrored for shorts).
  4. Stop (*):   "mental" -- out at the close of the first candle back inside the range (he uses no hard
                             stop), or a hard stop 1 tick beyond the signal candle.
     The risk unit R = entry to the far end of the signal candle (+1 tick) in both cases.
  5. Exit (*):   all at 1R; all at 2R; or his way -- half at 1R, stop to break-even, the rest runs to
                 the stop or the end of the day.
One trade a day, signals until 12:00 New York, everything flat by 16:40 (Lucid). Costs: $3.50 round trip
(0.175 pt NQ, 0.07 ES) + 0.25 pt spread; 1 tick slippage on hard stops; a bar reaching stop and target
counts as the stop; no target on the entry bar.
Data: NQ / ES 15-min stand-ins 2016-2026 (10 years; choose 2016-2022, judge 2023-2026) and NinjaTrader
NQ 5-min 2021-2026 (choose 2021-2023, judge 2024-2026).

Usage:  .venv/Scripts/python.exe scripts/research_overnight_range.py
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
DATA = {   # name -> (file, commission round trip in points, split, years)
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01", 10.6),
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01", 10.6),
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01", 5.65),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def run(args) -> pd.DataFrame:
    name, mode, stop_style, exit_style = args
    path, comm, _, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    out = []
    cur = None
    on_hi = on_lo = np.nan
    state = None
    pos = None
    n = len(c)
    for i in range(1, n):
        if day[i] != cur:
            cur, on_hi, on_lo, state = day[i], -np.inf, np.inf, {"s": "range", "done": False}
        # --- the open position (checked on every bar; flat on the session's last bar before 16:40) ---
        if pos is not None:
            p_ = pos
            side, ent, stp, lvl, risk = p_["side"], p_["ent"], p_["stp"], p_["lvl"], p_["risk"]
            exit_px = how = None
            hard_hit = (l[i] <= stp) if side > 0 else (h[i] >= stp)
            if 1000 <= mins[i] < 1080:
                exit_px, how = o[i], "16:40 close"
            elif hard_hit and (stop_style == "hard" or p_["half"]):
                exit_px, how = stp - side * TICK, "stop" if not p_["half"] else "break-even"
            elif i > p_["ie"] and not p_["half"] and ((h[i] > p_["tgt"]) if side > 0 else (l[i] < p_["tgt"])):
                if exit_style == "half":
                    p_["half"] = True                               # half off at 1R, rest to break-even
                    p_["banked"] = 0.5 * side * (p_["tgt"] - ent)
                    p_["stp"] = ent
                else:
                    exit_px, how = p_["tgt"], "target"
            elif stop_style == "mental" and not p_["half"] and ((c[i] < lvl) if side > 0 else (c[i] > lvl)):
                exit_px, how = c[i], "closed back inside"
            if exit_px is None and (i + 1 >= n or day[i + 1] != day[i] or mins[i + 1] >= 1000):
                exit_px, how = c[i], "16:40 close"                # some data has no bars 16:15-18:00
            if exit_px is not None:
                frac = 0.5 if p_["half"] else 1.0
                pts = p_.get("banked", 0.0) + frac * side * (exit_px - ent) - comm - SPREAD
                out.append((df.index[p_["ie"]], df.index[i], side, risk, pts / risk, how))
                pos = None
        if mins[i] >= 1080 or mins[i] < 570:                       # overnight: build the range
            on_hi, on_lo = max(on_hi, h[i]), min(on_lo, l[i])
            continue
        if not np.isfinite(on_hi):
            continue
        if state["done"] or pos is not None or mins[i] >= 720:
            continue
        # --- signals ---
        if state["s"] == "range":
            side = 1 if c[i] > on_hi else (-1 if c[i] < on_lo else 0)
            if side == 0:
                continue
            lvl = on_hi if side > 0 else on_lo
            if mode == "breakout":
                ent, far = c[i], (l[i] if side > 0 else h[i])
            else:
                state.update(s="broken", side=side, lvl=lvl)
                continue
        else:                                                       # waiting for the retest
            side, lvl = state["side"], state["lvl"]
            rng = h[i] - l[i]
            touched = (l[i] <= lvl + TICK) if side > 0 else (h[i] >= lvl - TICK)
            if not touched:
                continue
            good = rng > 0 and ((c[i] > lvl and c[i] >= l[i] + 0.6 * rng) if side > 0 else
                                (c[i] < lvl and c[i] <= h[i] - 0.6 * rng))
            if not good:
                if (c[i] < lvl) if side > 0 else (c[i] > lvl):
                    state["done"] = True                            # failed the retest: back inside
                continue
            ent, far = c[i], (l[i] if side > 0 else h[i])
        stp = far - side * TICK
        risk = side * (ent - stp)
        state["done"] = True
        if risk <= 0:
            continue
        mult = 2.0 if exit_style == "2R" else 1.0
        pos = {"side": side, "ent": ent, "stp": stp, "lvl": lvl, "risk": risk, "tgt": ent + side * mult * risk,
               "ie": i, "half": False}
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
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    grid = list(itertools.product(DATA, ("breakout", "retest"), ("mental", "hard"), ("1R", "2R", "half")))
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(run, grid))
    rows = []
    for g, t in zip(grid, results):
        _, _, split, years = DATA[g[0]]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        pts_mnq = (t["R"] * t["risk"]).mean() * 2 if len(t) else np.nan          # $ per trade per MNQ
        rows.append({"data": g[0], "entry": g[1], "stop": g[2], "exit": {"half": "half at 1R + runner"}.get(g[3], g[3]),
                     "trades": n, "per year": n / years, "winning%": w, "avg R": r, "$/trade 1 MNQ": pts_mnq,
                     "older R": ra, "z older": za, "newer R": rb, "z newer": zb, "_t": t})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z older": lambda v: f"{v:+.2f}", "z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}", "$/trade 1 MNQ": lambda v: f"{v:+.1f}"})
    for name in DATA:
        part = res[res["data"] == name].sort_values("older R", ascending=False)
        print(f"\n=== {name}: older years choose, newer judge (split {DATA[name][2]}) ===")
        print(part.drop(columns=["_t", "data"]).to_string(index=False, formatters=fm))
        b = part.iloc[0]
        t = b["_t"]
        print(f"  all {len(part)} versions: average {part['avg R'].mean():+.3f} R; newer years positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R). Chosen ({b['entry']}, "
              f"{b['stop']} stop, {b['exit']}): older {b['older R']:+.3f} -> newer {b['newer R']:+.3f} R "
              f"(z {b['z newer']:+.2f}); exits {t['how'].value_counts().to_dict()}")
        if "15-min" in name:
            by = t.groupby(t["opened"].dt.year)["R"].agg(["count", "mean"])
            print("  by year (trades / avg R): " + ", ".join(f"{y}: {int(v['count'])}/{v['mean']:+.2f}" for y, v in by.iterrows()))
    res.drop(columns=["_t"]).to_csv(ROOT / "data" / "research_overnight_range.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
