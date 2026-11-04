"""New twists from the Wealthsimple '11 proven techniques' article (owner, 2026-10-08) on rules tested before.

 a 60-minute ORB breakout: range 09:30-10:30; the first candle 10:30-12:00 that closes beyond it -> enter at
   the close; stop *at the midpoint or 1 tick inside the other side; target *2R or 3R. One a day.
 b Swing breakout, measured move: swings = extreme of 3 candles each side; a break of the last swing high
   (stop order + 1 tick) -> long; stop 1 tick under the last swing low; target = the height of the previous
   swing (last swing high - the swing low before it) projected from the entry. Shorts mirrored.
 c Fibonacci 38.2%: research_ten_classics '2 Fibonacci pullback' with the entry at 38.2% (stop past 78.6%,
   target the swing's end).
 d Range + RSI: no trend (ADX(14) < 20, band width under its 120-candle median); limit buy at the lower
   Bollinger band only when RSI(14) < 30 (sell at the upper band when RSI > 70); stop half an ATR outside;
   target the middle band.
 e Daily 20/50 EMA trend + pullback (holds for days -- NOT allowed on LucidPro; reference): EMA20 > EMA50 =
   uptrend; buy at the close of a day that touched EMA20 and closed above it; out when EMA20 crosses under
   EMA50; first stop under EMA50.
Intraday ones: entries 09:30-15:30 New York, flat by 16:40, costs and fills as in research_three_setups.
Data: NQ / ES 15-min stand-ins 2016-2026 (choose 2016-2022, judge 2023-2026), NQ 5-min 2021-2026, ES 5-min
2023-2026; the daily test on daily bars built from the 10-year data.

Usage:  .venv/Scripts/python.exe scripts/research_article_variants.py
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_ten_classics as tc  # noqa: E402
import research_three_setups as base  # noqa: E402

TICK = base.TICK


def orb60(dataset, stop_kind, rr):
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = base.prep(dataset)
    bk = base.Book(df, comm)
    cur = None
    for i in range(20, len(c)):
        if cme[i] != cur:
            cur, hi, lo, done = cme[i], -np.inf, np.inf, False
        bk.step(i, o, h, l, c, last[i])
        if 570 <= mins[i] < 630:
            hi, lo = max(hi, h[i]), min(lo, l[i])
            continue
        if done or bk.pos is not None or not (630 <= mins[i] < 720) or not np.isfinite(hi):
            continue
        side = 1 if c[i] > hi else (-1 if c[i] < lo else 0)
        if side:
            stp = (hi + lo) / 2 if stop_kind == "midpoint" else ((lo + TICK) if side > 0 else (hi - TICK))
            done = True
            bk.enter(side, c[i], stp, rr, i, atr[i])
    return bk.frame()


def swing_measured(dataset, _):
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = base.prep(dataset)
    bk = base.Book(df, comm)
    p, highs, lows, used_h, used_l = 3, [], [], True, True
    for i in range(30, len(c)):
        bk.step(i, o, h, l, c, last[i])
        j = i - p
        if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:i + 1].max():
            highs, used_h = (highs + [h[j]])[-2:], False
        if l[j] < l[j - p:j].min() and l[j] <= l[j + 1:i + 1].min():
            lows, used_l = (lows + [l[j]])[-2:], False
        if bk.pos is not None or not tc.in_window(mins[i]) or len(highs) < 1 or len(lows) < 1:
            continue
        sh, sl = highs[-1], lows[-1]
        if not used_h and h[i] >= sh + TICK:
            used_h = True
            ent = max(o[i], sh + TICK) + TICK
            height = sh - sl
            stp = sl - TICK
            if height > 0 and ent - stp > 0:
                bk.enter(1, ent, stp, height / (ent - stp), i, atr[i])
        elif not used_l and l[i] <= sl - TICK:
            used_l = True
            ent = min(o[i], sl - TICK) - TICK
            height = sh - sl
            stp = sh + TICK
            if height > 0 and stp - ent > 0:
                bk.enter(-1, ent, stp, height / (stp - ent), i, atr[i])
    return bk.frame()


def fib382(dataset, _):
    return tc.strat("2 Fibonacci pullback", 0.382, dataset)


def range_rsi(dataset, _):
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = base.prep(dataset)
    mid, upb, lob = tc.bands(c)
    bw = (upb - lob) / mid
    med = pd.Series(bw).rolling(120, min_periods=60).median().to_numpy()
    ax, r = tc.adx(h, l, c), tc.rsi(c)
    bk = base.Book(df, comm)
    for i in range(130, len(c)):
        bk.step(i, o, h, l, c, last[i])
        if bk.pos is not None or not tc.in_window(mins[i]) or np.isnan(atr[i]):
            continue
        if not (ax[i - 1] < 20 and bw[i - 1] < med[i - 1]):
            continue
        for side, band, ok in ((1, lob[i - 1], r[i - 1] < 30), (-1, upb[i - 1], r[i - 1] > 70)):
            if ok and ((l[i] <= band) if side > 0 else (h[i] >= band)):
                ent = min(o[i], band) if side > 0 else max(o[i], band)
                stp = band - side * 0.5 * atr[i]
                tgt = mid[i - 1]
                risk = side * (ent - stp)
                if risk > 0 and side * (tgt - ent) > 0:
                    bk.enter(side, ent, stp, side * (tgt - ent) / risk, i, atr[i])
                    if bk.pos is not None and ((l[i] <= stp) if side > 0 else (h[i] >= stp)):
                        bk.step(i, o, h, l, c, False)
                break
    return bk.frame()


def daily_ema(dataset):
    path, comm, _ = base.DATA[dataset]
    df = base.load(path)
    et = df.index.tz_convert("America/New_York")
    day = (et + pd.Timedelta(hours=6)).date
    d = df.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    h, l, c = (d[x].to_numpy() for x in ("high", "low", "close"))
    e20, e50 = tc.ema(c, 20), tc.ema(c, 50)
    out, pos = [], None
    for i in range(60, len(c)):
        if pos is not None:
            side, ent, stp, ie = pos
            px = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px = stp - side * TICK
            elif (e20[i] < e50[i]) if side > 0 else (e20[i] > e50[i]):
                px = c[i]
            if px is not None:
                risk = abs(ent - stp)
                out.append((pd.Timestamp(d.index[ie], tz="UTC"), side, risk, (side * (px - ent) - comm - base.SPREAD) / risk))
                pos = None
        if pos is None:
            if e20[i] > e50[i] and l[i] <= e20[i] < c[i] and c[i] > e50[i]:
                pos = (1, c[i], e50[i] - TICK, i)
            elif e20[i] < e50[i] and h[i] >= e20[i] > c[i] and c[i] < e50[i]:
                pos = (-1, c[i], e50[i] + TICK, i)
    t = pd.DataFrame(out, columns=["opened", "side", "risk", "R"])
    t["how"] = ""
    return t


JOBS = [("a ORB 60-min breakout", (s, rr)) for s, rr in itertools.product(("midpoint", "other side"), (2.0, 3.0))] + \
       [("b swing breakout, measured move", (None,)), ("c Fibonacci 38.2%", (None,)), ("d range + RSI 30/70", (None,))]
FUN = {"a ORB 60-min breakout": lambda d, a: orb60(d, *a), "b swing breakout, measured move": lambda d, a: swing_measured(d, *a),
       "c Fibonacci 38.2%": lambda d, a: fib382(d, *a), "d range + RSI 30/70": lambda d, a: range_rsi(d, *a)}


def job(x):
    name, args, d = x
    return FUN[name](d, args)


def main() -> int:
    grid = [(nm, a, d) for nm, a in JOBS for d in base.DATA]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(job, grid))
    rows = []
    for (nm, a, d), t in zip(grid, results):
        split = base.DATA[d][2]
        n, w, r, _ = base.summary(t)
        _, _, ra, _ = base.summary(t, "2000-01-01", split)
        _, _, rb, zb = base.summary(t, split, "2100-01-01")
        rows.append({"strategy": nm, "settings": " / ".join(str(x) for x in a if x is not None) or "-", "data": d,
                     "trades": n, "winning%": w, "avg R": r, "older R": ra, "newer R": rb, "z newer": zb})
    for d in ("NQ 15-min 2016-26", "ES 15-min 2016-26"):
        t = daily_ema(d)
        n, w, r, _ = base.summary(t)
        _, _, ra, _ = base.summary(t, "2000-01-01", "2023-01-01")
        _, _, rb, zb = base.summary(t, "2023-01-01", "2100-01-01")
        rows.append({"strategy": "e daily 20/50 EMA + pullback (not Lucid)", "settings": "-", "data": d.replace("15-min", "daily"),
                     "trades": n, "winning%": w, "avg R": r, "older R": ra, "newer R": rb, "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}"})
    print(res.to_string(index=False, formatters=fm))
    s = res.groupby("strategy").agg(avg_R=("avg R", "mean"), newer_avg=("newer R", "mean"),
                                    newer_positive=("newer R", lambda x: (x > 0).mean()))
    print("\n" + s.sort_values("avg_R", ascending=False).to_string(float_format=lambda v: f"{v:+.3f}"))
    res.to_csv(ROOT / "data" / "research_article_variants.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
