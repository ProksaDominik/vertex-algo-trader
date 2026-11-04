"""Three classic intraday setups (owner's screenshot, 2026-10-08), each with its stated rules.

A. 30-minute ORB breakout: range = 09:30-10:00 New York. The first candle (10:00-12:00) that CLOSES beyond
   it -- *with volume >= 1.5x the average of the previous 20 candles, or without that filter -- enter at
   the close. Stop *at the range midpoint or 1 tick inside the opposite side. Target *2R or 3R.
B. VWAP trend pullback: session VWAP from 09:30 (volume weighted) with standard-deviation bands. Uptrend =
   VWAP higher than 30 minutes ago and the previous close above it; a candle that touched VWAP closes back
   above it -> long at the close (shorts mirrored). Entries 10:00-15:00, up to 2 a day. Stop *1 tick under
   the lowest low of the last 6 candles or at the -2 SD band. Exit *2R, or when a candle closes back
   through VWAP ("ride the trend").
C. Pre-market reclaim (fakeout fade): levels = *the overnight session's (18:00-09:30) or the pre-market's
   (04:00-09:30) high and low. From 09:30 to 11:30 price trades above the high, then a candle closes back
   below it -- *with volume >= 1.5x the 20-candle average, or without -- short at that close (mirrored at
   the low). Stop 1 tick above the highest high since the poke. Target *1.5R or 2R. One trade per side.
All: flat on the session's last candle before 16:40 New York (Lucid), one position at a time, stops under
4 ticks or a quarter ATR skipped. Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt spread; 1 tick
slippage on stops; stop first if a candle reaches both; no target on the entry candle.
Data: NQ / ES 15-min stand-ins 2016-2026 (choose 2016-2022, judge 2023-2026), NinjaTrader NQ 5-min
2021-2026 (choose 2021-2023) and ES 5-min Dec 2023 - Oct 2026 (choose to Mar 2025).

Usage:  .venv/Scripts/python.exe scripts/research_three_setups.py
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
    "NQ 15-min 2016-26": (ROOT / "data" / "nas100_15m_long.parquet", 0.175, "2023-01-01"),
    "ES 15-min 2016-26": (ROOT / "data" / "us500_15m_long.parquet", 0.07, "2023-01-01"),
    "NQ 5-min 2021-26": (ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet", 0.175, "2024-01-01"),
    "ES 5-min 2023-26": (ROOT / "data" / "futures" / "ES_5m_long_nt.parquet", 0.07, "2025-04-01"),
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


class Book:
    """Open position bookkeeping shared by the three setups."""

    def __init__(self, df, comm):
        self.df, self.comm, self.pos, self.out = df, comm, None, []

    def step(self, i, o, h, l, c, last_bar, vwap=None):
        p = self.pos
        if p is None:
            return
        side = p["side"]
        px = how = None
        if (l[i] <= p["stp"]) if side > 0 else (h[i] >= p["stp"]):
            px, how = p["stp"] - side * TICK, "stop"
        elif i > p["ie"] and p["tgt"] is not None and ((h[i] > p["tgt"]) if side > 0 else (l[i] < p["tgt"])):
            px, how = p["tgt"], "target"
        elif p["tgt"] is None and vwap is not None and i > p["ie"] and ((c[i] < vwap) if side > 0 else (c[i] > vwap)):
            px, how = c[i], "closed through VWAP"
        elif last_bar:
            px, how = c[i], "16:40 close"
        if px is not None:
            risk = abs(p["ent"] - p["stp"])
            self.out.append((self.df.index[p["ie"]], self.df.index[i], side, risk,
                             (side * (px - p["ent"]) - self.comm - SPREAD) / risk, how))
            self.pos = None

    def enter(self, side, ent, stp, rr, i, atr_i):
        risk = side * (ent - stp)
        if np.isnan(atr_i) or risk < max(4 * TICK, 0.25 * atr_i):
            return False
        self.pos = {"side": side, "ent": ent, "stp": stp, "tgt": None if rr is None else ent + side * rr * risk, "ie": i}
        return True

    def frame(self):
        return pd.DataFrame(self.out, columns=["opened", "closed", "side", "risk", "R", "how"])


def prep(name):
    path, comm, _ = DATA[name]
    df = load(path)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    cme = np.asarray((et + pd.Timedelta(hours=6)).date)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    vavg = pd.Series(v).shift(1).rolling(20, min_periods=10).mean().to_numpy()
    n = len(c)
    last = np.zeros(n, bool)
    last[:-1] = (cme[1:] != cme[:-1]) | (mins[1:] >= 1000) & (mins[1:] < 1080)
    last[-1] = True
    return df, comm, o, h, l, c, v, mins, cme, atr, vavg, last


def run_orb(args):
    name, stop_kind, vol_filter, rr = args
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = prep(name)
    bk = Book(df, comm)
    cur = None
    for i in range(20, len(c)):
        if cme[i] != cur:
            cur, hi, lo, done = cme[i], -np.inf, np.inf, False
        bk.step(i, o, h, l, c, last[i])
        if 570 <= mins[i] < 600:
            hi, lo = max(hi, h[i]), min(lo, l[i])
            continue
        if done or bk.pos is not None or not (600 <= mins[i] < 720) or not np.isfinite(hi):
            continue
        side = 1 if c[i] > hi else (-1 if c[i] < lo else 0)
        if side == 0:
            continue
        if vol_filter and not (v[i] >= 1.5 * vavg[i]):
            continue
        stp = (hi + lo) / 2 if stop_kind == "midpoint" else ((lo + TICK) if side > 0 else (hi - TICK))
        done = True
        bk.enter(side, c[i], stp, rr, i, atr[i])
    return bk.frame()


def run_vwap(args):
    name, stop_kind, exit_kind = args
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = prep(name)
    tp = (h + l + c) / 3
    bar_min = int(round((df.index[1] - df.index[0]).total_seconds() / 60)) or 5
    look = max(1, 30 // bar_min)
    bk = Book(df, comm)
    cur = None
    vw = np.full(len(c), np.nan)
    sd = np.full(len(c), np.nan)
    for i in range(20, len(c)):
        if cme[i] != cur:
            cur, sv, svp, svp2, k_today, today0 = cme[i], 0.0, 0.0, 0.0, 0, None
        if mins[i] >= 570 and mins[i] < 1000:
            w = max(v[i], 1.0)
            sv, svp, svp2 = sv + w, svp + w * tp[i], svp2 + w * tp[i] ** 2
            vw[i] = svp / sv
            sd[i] = math.sqrt(max(svp2 / sv - vw[i] ** 2, 0.0))
            if today0 is None:
                today0 = i
        bk.step(i, o, h, l, c, last[i], vwap=vw[i] if not np.isnan(vw[i]) else None)
        if bk.pos is not None or k_today >= 2 or not (600 <= mins[i] < 900) or today0 is None or i - look < today0:
            continue
        rising, falling = vw[i] > vw[i - look], vw[i] < vw[i - look]
        side = 0
        if rising and c[i - 1] > vw[i - 1] and l[i] <= vw[i] and c[i] > vw[i]:
            side = 1
        elif falling and c[i - 1] < vw[i - 1] and h[i] >= vw[i] and c[i] < vw[i]:
            side = -1
        if side == 0:
            continue
        if stop_kind == "swing":
            stp = (l[i - 5:i + 1].min() - TICK) if side > 0 else (h[i - 5:i + 1].max() + TICK)
        else:
            stp = vw[i] - side * 2 * sd[i]
        if bk.enter(side, c[i], stp, 2.0 if exit_kind == "2R" else None, i, atr[i]):
            k_today += 1
    return bk.frame()


def run_reclaim(args):
    name, start, vol_filter, rr = args
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = prep(name)
    bk = Book(df, comm)
    cur = None
    for i in range(20, len(c)):
        if cme[i] != cur:
            cur, phi, plo, st = cme[i], -np.inf, np.inf, {1: None, -1: None}
            done = {1: False, -1: False}
        bk.step(i, o, h, l, c, last[i])
        m = mins[i]
        in_pre = (m >= 1080 or m < 570) if start == "overnight" else (240 <= m < 570)
        if in_pre:
            phi, plo = max(phi, h[i]), min(plo, l[i])
            continue
        if not (570 <= m < 690) or not np.isfinite(phi):
            continue
        for side, lvl in ((-1, phi), (1, plo)):                 # side = the trade: short after a poke above
            if done[side]:
                continue
            s = st[side]
            if s is None:
                if (h[i] > lvl) if side < 0 else (l[i] < lvl):
                    st[side] = {"ext": h[i] if side < 0 else l[i], "t": i}
                    s = st[side]
                else:
                    continue
            s["ext"] = max(s["ext"], h[i]) if side < 0 else min(s["ext"], l[i])
            back = (c[i] < lvl) if side < 0 else (c[i] > lvl)
            if not back or bk.pos is not None:
                continue
            if vol_filter and not (v[i] >= 1.5 * vavg[i]):
                continue
            stp = (s["ext"] + TICK) if side < 0 else (s["ext"] - TICK)
            done[side] = True
            bk.enter(side, c[i], stp, rr, i, atr[i])
    return bk.frame()


def summary(t, lo=None, hi=None):
    if lo is not None:
        t = t[(t["opened"] >= pd.Timestamp(lo, tz="UTC")) & (t["opened"] < pd.Timestamp(hi, tz="UTC"))]
    n = len(t)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    r = t["R"].to_numpy()
    return n, float((r > 0).mean()), float(r.mean()), float(r.mean() / (r.std(ddof=1) / math.sqrt(n)))


def dispatch(job):
    kind, args = job
    return {"A": run_orb, "B": run_vwap, "C": run_reclaim}[kind](args)


def main() -> int:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    jobs = []
    for d in DATA:
        jobs += [("A", (d, s, vf, rr)) for s, vf, rr in itertools.product(("midpoint", "opposite side"), (False, True), (2.0, 3.0))]
        jobs += [("B", (d, s, e)) for s, e in itertools.product(("swing", "2 SD band"), ("2R", "VWAP close"))]
        jobs += [("C", (d, st, vf, rr)) for st, vf, rr in itertools.product(("overnight", "04:00"), (False, True), (1.5, 2.0))]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(dispatch, jobs))
    rows = []
    for (kind, args), t in zip(jobs, results):
        split = DATA[args[0]][2]
        n, w, r, z = summary(t)
        _, _, ra, za = summary(t, "2000-01-01", split)
        _, _, rb, zb = summary(t, split, "2100-01-01")
        label = {"A": "ORB 30-min breakout", "B": "VWAP trend pullback", "C": "pre-market reclaim"}[kind]
        rows.append({"setup": label, "data": args[0], "settings": " / ".join(str(a) for a in args[1:]),
                     "trades": n, "winning%": w, "avg R": r, "older R": ra, "newer R": rb, "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}"})
    for setup in ("ORB 30-min breakout", "VWAP trend pullback", "pre-market reclaim"):
        part = res[res["setup"] == setup]
        print(f"\n=== {setup} ===")
        print(part.drop(columns=["setup"]).to_string(index=False, formatters=fm))
        print(f"  {len(part)} versions: average {part['avg R'].mean():+.3f} R per trade; newer years positive in "
              f"{(part['newer R'] > 0).mean():.0%} (average {part['newer R'].mean():+.3f} R)")
        for d in DATA:
            q = part[part["data"] == d]
            print(f"    {d}: average {q['avg R'].mean():+.3f} R, best on older years {q.sort_values('older R').iloc[-1]['settings']}"
                  f" -> newer {q.sort_values('older R').iloc[-1]['newer R']:+.3f} R")
    res.to_csv(ROOT / "data" / "research_three_setups.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
