"""Ten classic futures strategies from an Optimus Futures article (owner, 2026-10-08), each made exact.

Intraday ones: entries 09:30-15:30 New York (the news straddle: at its release), one position at a time,
flat on the session's last candle before 16:40 (Lucid), stops under 4 ticks / a quarter ATR skipped,
$3.50 round trip + 0.25 pt spread, 1 tick slippage on stops and stop entries, stop first if a candle
reaches both, no target on the entry candle. Longs shown; shorts mirrored.
 1 ZigZag breakout     buy-stop 1 tick above the last swing high (extreme of 3 candles each side);
                       stop 1 tick under the last swing low; target *0.6R or 1R ("60-100% of the risk").
 2 Fibonacci pullback  after a swing low -> swing high leg of >= 3 ATR: limit buy at the *50% or 61.8%
                       retracement; stop past 78.6%; target the swing high.
 3 MA ribbon           EMAs 8/13/21/34/55 line up upward, price above them and the ribbon widening (first
                       candle of that state) -> buy at the close; stop under the last 10 candles' low;
                       exit *1R or when EMA 8 falls under EMA 13 (the ribbon contracts / crosses).
 4 Bollinger squeeze   band width in its lowest 20% of the last 120 candles within the last 5 candles,
                       then a close above the upper band on above-average volume -> buy at the close;
                       stop under the last 20 candles' low; exit *on a close back inside the band or 2R.
 5 Bollinger reversion a close above the upper band, then a close back inside -> short at that close;
                       target the middle band; stop above the high of the last *3 or 5 candles.
 7 Box breakout        the last *12 or 24 candles form a box no taller than 3 ATR; a close above it ->
                       buy; target = box height (measured move); stop 1 tick under the box.
 8 Inside the bands    no trend (ADX(14) < 20, band width under its 120-candle median): limit buy at the
                       lower band; stop half an ATR outside it; target *the middle or the opposite band.
 9 EMA 9/21 + RSI      EMA 9 crosses above EMA 21, close above EMA 9, RSI(14) > 50 -> buy at the close;
                       stop under the entry candle; exit *on a close under EMA 9 or 1.5R.
10 News straddle       CPI / jobs report (08:30) and Fed decisions (14:00): stop orders 1 tick beyond the
                       15 minutes before the release, valid 30 minutes; stop at the other side; target 2x.
 6 Donchian 20-day     (daily, holds for weeks -- not tradable on Lucid, shown for completeness): buy at a
                       close above the previous 20 days' high, out at a close below the channel middle;
                       first stop under the entry day's low.
Data: NQ / ES 15-min stand-ins 2016-2026 (choose 2016-2022, judge 2023-2026), NinjaTrader NQ 5-min
2021-2026 and ES 5-min 2023-2026; Donchian on daily bars built from the 10-year data.

Usage:  .venv/Scripts/python.exe scripts/research_ten_classics.py
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import research_three_setups as base  # noqa: E402

TICK = base.TICK
DATA = base.DATA


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def wilder(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), n), wilder(np.maximum(-d, 0), n)
    return 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))


def adx(h, l, c, n=14):
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    s_tr = wilder(tr, n)
    pdi, mdi = 100 * wilder(pdm, n) / s_tr, 100 * wilder(mdm, n) / s_tr
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
    return wilder(np.nan_to_num(dx), n)


def bands(c, n=20, k=2.0):
    s = pd.Series(c)
    mid, sd = s.rolling(n).mean().to_numpy(), s.rolling(n).std(ddof=0).to_numpy()
    return mid, mid + k * sd, mid - k * sd


def in_window(m):
    return 570 <= m < 930


def strat(name, variant, dataset):
    df, comm, o, h, l, c, v, mins, cme, atr, vavg, last = base.prep(dataset)
    n = len(c)
    bk = base.Book(df, comm)
    if name == "1 ZigZag breakout":
        p, sh, sl, used_h, used_l = 3, None, None, True, True
        for i in range(30, n):
            bk.step(i, o, h, l, c, last[i])
            j = i - p
            if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:i + 1].max():
                sh, used_h = h[j], False
            if l[j] < l[j - p:j].min() and l[j] <= l[j + 1:i + 1].min():
                sl, used_l = l[j], False
            if bk.pos is not None or not in_window(mins[i]) or sh is None or sl is None:
                continue
            if not used_h and h[i] >= sh + TICK:
                used_h = True
                bk.enter(1, max(o[i], sh + TICK) + TICK, sl - TICK, variant, i, atr[i])
            elif not used_l and l[i] <= sl - TICK:
                used_l = True
                bk.enter(-1, min(o[i], sl - TICK) - TICK, sh + TICK, variant, i, atr[i])
    elif name == "2 Fibonacci pullback":
        p, swings, setup = 3, [], None
        for i in range(30, n):
            bk.step(i, o, h, l, c, last[i])
            j = i - p
            if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:i + 1].max():
                swings.append(("H", h[j], j))
            elif l[j] < l[j - p:j].min() and l[j] <= l[j + 1:i + 1].min():
                swings.append(("L", l[j], j))
            swings = swings[-4:]
            if len(swings) >= 2 and (setup is None or setup["made"] != swings[-1][2]):
                a, b = swings[-2], swings[-1]
                leg = abs(b[1] - a[1])
                if a[0] != b[0] and not np.isnan(atr[i]) and leg >= 3 * atr[i]:
                    side = 1 if b[0] == "H" else -1
                    ext, start = b[1], a[1]
                    setup = {"side": side, "ent": ext - side * variant * leg, "stp": ext - side * 0.786 * leg - side * TICK,
                             "tgt": ext, "made": b[2], "t": i}
            if setup is None or bk.pos is not None:
                continue
            s = setup
            if i - s["t"] > 48 or ((h[i] > s["tgt"]) if s["side"] > 0 else (l[i] < s["tgt"])):
                setup = None
                continue
            if in_window(mins[i]) and ((l[i] <= s["ent"]) if s["side"] > 0 else (h[i] >= s["ent"])):
                ent = min(o[i], s["ent"]) if s["side"] > 0 else max(o[i], s["ent"])
                risk = s["side"] * (ent - s["stp"])
                rr = s["side"] * (s["tgt"] - ent) / risk if risk > 0 else 0
                if rr > 0 and bk.enter(s["side"], ent, s["stp"], rr, i, atr[i]):
                    if (l[i] <= s["stp"]) if s["side"] > 0 else (h[i] >= s["stp"]):
                        bk.step(i, o, h, l, c, False)
                setup = None
    elif name == "3 MA ribbon":
        es = [ema(c, k) for k in (8, 13, 21, 34, 55)]
        up = (es[0] > es[1]) & (es[1] > es[2]) & (es[2] > es[3]) & (es[3] > es[4]) & (c > es[0])
        dn = (es[0] < es[1]) & (es[1] < es[2]) & (es[2] < es[3]) & (es[3] < es[4]) & (c < es[0])
        spread = es[0] - es[4]
        for i in range(60, n):
            if bk.pos is not None and variant == "contracts":
                side = bk.pos["side"]
                if i > bk.pos["ie"] and ((es[0][i] < es[1][i]) if side > 0 else (es[0][i] > es[1][i])):
                    bk.pos["tgt"] = c[i]                       # leave at this close
                    bk.pos["stp"] = bk.pos["stp"]
                    risk = abs(bk.pos["ent"] - bk.pos["stp"])
                    bk.out.append((df.index[bk.pos["ie"]], df.index[i], side, risk,
                                   (side * (c[i] - bk.pos["ent"]) - comm - base.SPREAD) / risk, "ribbon contracts"))
                    bk.pos = None
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]):
                continue
            for side, st in ((1, up), (-1, dn)):
                widening = (spread[i] > spread[i - 3]) if side > 0 else (spread[i] < spread[i - 3])
                if st[i] and not st[i - 1] and widening:
                    stp = (l[i - 9:i + 1].min() - TICK) if side > 0 else (h[i - 9:i + 1].max() + TICK)
                    bk.enter(side, c[i], stp, 1.0 if variant == "1R" else None, i, atr[i])
                    if variant == "contracts" and bk.pos is not None:
                        bk.pos["tgt"] = None
                    break
    elif name == "4 Bollinger squeeze":
        mid, upb, lob = bands(c)
        bw = (upb - lob) / mid
        q20 = pd.Series(bw).rolling(120, min_periods=60).quantile(0.2).to_numpy()
        sq = bw <= q20
        recent = pd.Series(sq.astype(float)).rolling(5).max().to_numpy() > 0
        for i in range(130, n):
            if bk.pos is not None and variant == "re-entry":
                side = bk.pos["side"]
                if i > bk.pos["ie"] and ((c[i] < upb[i]) if side > 0 else (c[i] > lob[i])):
                    risk = abs(bk.pos["ent"] - bk.pos["stp"])
                    bk.out.append((df.index[bk.pos["ie"]], df.index[i], side, risk,
                                   (side * (c[i] - bk.pos["ent"]) - comm - base.SPREAD) / risk, "back inside the band"))
                    bk.pos = None
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]) or not recent[i - 1] or not (v[i] > vavg[i]):
                continue
            if c[i] > upb[i]:
                bk.enter(1, c[i], l[i - 19:i + 1].min() - TICK, 2.0 if variant == "2R" else None, i, atr[i])
            elif c[i] < lob[i]:
                bk.enter(-1, c[i], h[i - 19:i + 1].max() + TICK, 2.0 if variant == "2R" else None, i, atr[i])
    elif name == "5 Bollinger reversion":
        mid, upb, lob = bands(c)
        for i in range(30, n):
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]):
                continue
            k = variant
            if c[i - 1] > upb[i - 1] and c[i] < upb[i]:
                stp = h[i - k + 1:i + 1].max() + TICK
                risk = stp - c[i]
                if risk > 0 and mid[i] < c[i]:
                    bk.enter(-1, c[i], stp, (c[i] - mid[i]) / risk, i, atr[i])
            elif c[i - 1] < lob[i - 1] and c[i] > lob[i]:
                stp = l[i - k + 1:i + 1].min() - TICK
                risk = c[i] - stp
                if risk > 0 and mid[i] > c[i]:
                    bk.enter(1, c[i], stp, (mid[i] - c[i]) / risk, i, atr[i])
    elif name == "7 Box breakout":
        k = variant
        hh = pd.Series(h).shift(1).rolling(k).max().to_numpy()
        ll = pd.Series(l).shift(1).rolling(k).min().to_numpy()
        for i in range(k + 20, n):
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]) or np.isnan(atr[i]):
                continue
            height = hh[i] - ll[i]
            if height <= 0 or height > 3 * atr[i]:
                continue
            if c[i] > hh[i]:
                stp = ll[i] - TICK
                bk.enter(1, c[i], stp, height / (c[i] - stp), i, atr[i])
            elif c[i] < ll[i]:
                stp = hh[i] + TICK
                bk.enter(-1, c[i], stp, height / (stp - c[i]), i, atr[i])
    elif name == "8 Inside the bands":
        mid, upb, lob = bands(c)
        bw = (upb - lob) / mid
        med = pd.Series(bw).rolling(120, min_periods=60).median().to_numpy()
        ax = adx(h, l, c)
        for i in range(130, n):
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]) or np.isnan(atr[i]):
                continue
            if not (ax[i - 1] < 20 and bw[i - 1] < med[i - 1]):
                continue
            for side, band, tgt in ((1, lob[i - 1], mid[i - 1] if variant == "middle" else upb[i - 1]),
                                    (-1, upb[i - 1], mid[i - 1] if variant == "middle" else lob[i - 1])):
                if (l[i] <= band) if side > 0 else (h[i] >= band):
                    ent = min(o[i], band) if side > 0 else max(o[i], band)
                    stp = band - side * 0.5 * atr[i]
                    risk = side * (ent - stp)
                    if risk > 0 and side * (tgt - ent) > 0:
                        bk.enter(side, ent, stp, side * (tgt - ent) / risk, i, atr[i])
                        if bk.pos is not None and ((l[i] <= stp) if side > 0 else (h[i] >= stp)):
                            bk.step(i, o, h, l, c, False)
                    break
    elif name == "9 EMA 9/21 + RSI":
        e9, e21, r = ema(c, 9), ema(c, 21), rsi(c)
        for i in range(30, n):
            if bk.pos is not None and variant == "close under EMA 9":
                side = bk.pos["side"]
                if i > bk.pos["ie"] and ((c[i] < e9[i]) if side > 0 else (c[i] > e9[i])):
                    risk = abs(bk.pos["ent"] - bk.pos["stp"])
                    bk.out.append((df.index[bk.pos["ie"]], df.index[i], side, risk,
                                   (side * (c[i] - bk.pos["ent"]) - comm - base.SPREAD) / risk, "close through EMA 9"))
                    bk.pos = None
            bk.step(i, o, h, l, c, last[i])
            if bk.pos is not None or not in_window(mins[i]):
                continue
            if e9[i] > e21[i] and e9[i - 1] <= e21[i - 1] and c[i] > e9[i] and r[i] > 50:
                bk.enter(1, c[i], l[i] - TICK, 1.5 if variant == "1.5R" else None, i, atr[i])
            elif e9[i] < e21[i] and e9[i - 1] >= e21[i - 1] and c[i] < e9[i] and r[i] < 50:
                bk.enter(-1, c[i], h[i] + TICK, 1.5 if variant == "1.5R" else None, i, atr[i])
    elif name == "10 News straddle":
        ev = json.loads((ROOT / "data" / "futures" / "event_dates.json").read_text(encoding="utf-8"))
        times = {}
        for k, hhmm in (("cpi", 510), ("nfp", 510), ("fomc", 840)):
            for d in ev[k]:
                times.setdefault(pd.Timestamp(d).date(), []).append(hhmm)
        et_date = np.asarray(df.index.tz_convert("America/New_York").date)
        bar_min = int(round((df.index[1] - df.index[0]).total_seconds() / 60)) or 5
        pre = max(1, 15 // bar_min)
        arm = None
        for i in range(30, n):
            bk.step(i, o, h, l, c, last[i])
            rel = times.get(et_date[i], [])
            if mins[i] in rel and arm is None and bk.pos is None:
                arm = {"hi": h[i - pre:i].max(), "lo": l[i - pre:i].min(), "until": mins[i] + 30}
            if arm is None:
                continue
            if mins[i] >= arm["until"] or bk.pos is not None:
                arm = None if mins[i] >= arm["until"] else arm
                continue
            up_t, dn_t = arm["hi"] + TICK, arm["lo"] - TICK
            hit_up, hit_dn = h[i] >= up_t, l[i] <= dn_t
            if hit_up and hit_dn:                               # both in one candle: assume the worse order
                risk = up_t - dn_t
                bk.out.append((df.index[i], df.index[i], 1, risk, (-risk - 2 * TICK - comm - base.SPREAD) / risk, "both sides"))
                arm = None
            elif hit_up:
                bk.enter(1, max(o[i], up_t) + TICK, dn_t, 2.0, i, atr[i])
                arm = None
            elif hit_dn:
                bk.enter(-1, min(o[i], dn_t) - TICK, up_t, 2.0, i, atr[i])
                arm = None
    return bk.frame()


def donchian(dataset):
    path, comm, _ = DATA[dataset]
    df = base.load(path)
    et = df.index.tz_convert("America/New_York")
    day = (et + pd.Timedelta(hours=6)).date
    d = df.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    hh = pd.Series(h).shift(1).rolling(20).max().to_numpy()
    ll = pd.Series(l).shift(1).rolling(20).min().to_numpy()
    out, pos = [], None
    for i in range(21, len(c)):
        midl = (hh[i] + ll[i]) / 2
        if pos is not None:
            side, ent, stp, ie = pos
            px = None
            if (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px = stp - side * TICK
            elif (c[i] < midl) if side > 0 else (c[i] > midl):
                px = c[i]
            if px is not None:
                risk = abs(ent - stp)
                out.append((pd.Timestamp(d.index[ie]), pd.Timestamp(d.index[i]), side, risk, (side * (px - ent) - comm - base.SPREAD) / risk, ""))
                pos = None
        if pos is None:
            if c[i] > hh[i] and c[i] > l[i]:
                pos = (1, c[i], l[i] - TICK, i)
            elif c[i] < ll[i] and c[i] < h[i]:
                pos = (-1, c[i], h[i] + TICK, i)
    t = pd.DataFrame(out, columns=["opened", "closed", "side", "risk", "R", "how"])
    t["opened"] = pd.to_datetime(t["opened"]).dt.tz_localize("UTC")
    return t


JOBS = [("1 ZigZag breakout", v) for v in (0.6, 1.0)] + [("2 Fibonacci pullback", v) for v in (0.5, 0.618)] + \
       [("3 MA ribbon", v) for v in ("1R", "contracts")] + [("4 Bollinger squeeze", v) for v in ("re-entry", "2R")] + \
       [("5 Bollinger reversion", v) for v in (3, 5)] + [("7 Box breakout", v) for v in (12, 24)] + \
       [("8 Inside the bands", v) for v in ("middle", "opposite")] + [("9 EMA 9/21 + RSI", v) for v in ("close under EMA 9", "1.5R")] + \
       [("10 News straddle", "2x")]


def job(args):
    name, variant, dataset = args
    return strat(name, variant, dataset)


def main() -> int:
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    grid = [(nm, v, d) for nm, v in JOBS for d in DATA]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        results = list(ex.map(job, grid))
    rows = []
    for (nm, v, d), t in zip(grid, results):
        split = DATA[d][2]
        n, w, r, z = base.summary(t)
        _, _, ra, _ = base.summary(t, "2000-01-01", split)
        _, _, rb, zb = base.summary(t, split, "2100-01-01")
        rows.append({"strategy": nm, "variant": str(v), "data": d, "trades": n, "winning%": w, "avg R": r,
                     "older R": ra, "newer R": rb, "z newer": zb})
    for d in ("NQ 15-min 2016-26", "ES 15-min 2016-26"):
        t = donchian(d)
        n, w, r, z = base.summary(t)
        _, _, ra, _ = base.summary(t, "2000-01-01", "2023-01-01")
        _, _, rb, zb = base.summary(t, "2023-01-01", "2100-01-01")
        rows.append({"strategy": "6 Donchian 20-day (daily, not Lucid)", "variant": "-", "data": d.replace("15-min", "daily"),
                     "trades": n, "winning%": w, "avg R": r, "older R": ra, "newer R": rb, "z newer": zb})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    fm = {k: (lambda v: f"{v:+.3f}") for k in ("avg R", "older R", "newer R")}
    fm.update({"z newer": lambda v: f"{v:+.2f}", "winning%": lambda v: f"{v:.1%}"})
    print(res.to_string(index=False, formatters=fm))
    print("\nSUMMARY per strategy (all its versions and data sets):")
    s = res.groupby("strategy").agg(versions=("avg R", "size"), avg_R=("avg R", "mean"),
                                    newer_avg_R=("newer R", "mean"), newer_positive=("newer R", lambda x: (x > 0).mean()),
                                    trades=("trades", "mean"))
    print(s.sort_values("avg_R", ascending=False).to_string(float_format=lambda v: f"{v:+.3f}"))
    res.to_csv(ROOT / "data" / "research_ten_classics.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
