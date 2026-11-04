"""TJR's strategy (owner's fourth video, 2026-10-08): liquidity sweep -> 5-min reversal -> 1-min continuation.

Rules as described (shorts; longs mirrored):
  1. Draws on liquidity, fixed before the 09:30 New York open: Asia session high/low (20:00-24:00),
     London high/low (02:00-05:00), the previous New York session's high/low (09:30-16:00), and the
     1-hour swing highs/lows of the last 24 hours that price has not traded through yet.
  2. Manipulation: inside the trading window (*09:30-12:00 or 09:30-15:30 New York) price trades above
     one of those highs.
  3. Reversal within 30 minutes of the sweep's top: a completed 5-minute candle closes below the last
     5-minute swing low (break of structure) -- *or also below a bullish 5-minute fair value gap of the
     last hour (an "inverse fair value gap").
  4. Continuation on the entry chart (*1-minute, as in the video; or 5-minute for a longer history):
     a break UP (a close above the last swing high since the reversal = the retrace), then a break back
     DOWN (a close below the last swing low since the reversal) -> sell at that close. Stop 1 tick above
     the retrace high ("the second high"); dropped if price goes above the sweep's top first or after
     60 bars. Target: the nearest liquidity low below that pays at least *1 / 1.5 / 2 x the risk (no
     such draw -> no trade). Swings: *1 or 2 bars each side, known only later.
One trade a day; flat by 16:40 New York (Lucid). Costs: $3.50 round trip (0.175 pt) + 0.25 pt spread,
1 tick slippage on stops; a bar touching stop and target counts as the stop; no target on the entry bar.
Data: NinjaTrader NQ 1-minute Mar 2024 - Oct 2026 (choose on Mar 2024 - Jun 2025, judge Jul 2025 -
Oct 2026) and NQ 5-minute Feb 2021 - Oct 2026 (choose 2021-2023, judge 2024-2026).

Usage:  .venv/Scripts/python.exe scripts/research_tjr.py
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
TICK, SPREAD, COMM = 0.25, 0.25, 0.175


def load(tf: str) -> pd.DataFrame:
    df = pd.read_parquet(F / f"NQ_{tf}_long_nt.parquet")[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    return df[~df.index.duplicated()].sort_index()


def pivots(h: np.ndarray, l: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """is_sh[j] / is_sl[j]: bar j is a swing high/low (extreme of p bars each side); known at j + p."""
    n = len(h)
    is_sh, is_sl = np.zeros(n, bool), np.zeros(n, bool)
    for j in range(p, n - p):
        if h[j] > h[j - p:j].max() and h[j] >= h[j + 1:j + p + 1].max():
            is_sh[j] = True
        if l[j] < l[j - p:j].min() and l[j] <= l[j + 1:j + p + 1].min():
            is_sl[j] = True
    return is_sh, is_sl


def day_levels(df: pd.DataFrame) -> dict:
    """CME day -> (highs, lows) known before 09:30 New York."""
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    frame = pd.DataFrame({"h": h, "l": l, "d": day, "m": mins})
    asia = frame[(frame.m >= 1200)].groupby("d").agg(h=("h", "max"), l=("l", "min"))          # 20:00-24:00 (prev date)
    london = frame[(frame.m >= 120) & (frame.m < 300)].groupby("d").agg(h=("h", "max"), l=("l", "min"))
    ny = frame[(frame.m >= 570) & (frame.m < 960)].groupby("d").agg(h=("h", "max"), l=("l", "min"))
    prev_ny = ny.shift(1)
    hb = df.resample("1h", label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
    sh, sl = pivots(hb["high"].to_numpy(), hb["low"].to_numpy(), 1)
    hb_close = hb.index + pd.Timedelta(hours=1)
    out = {}
    days = sorted(set(day))
    open_ts = {}
    for d in days:
        open_ts[d] = pd.Timestamp(f"{d} 09:30", tz="America/New_York").tz_convert("UTC")
    hh, hl = hb["high"].to_numpy(), hb["low"].to_numpy()
    for d in days:
        highs, lows = [], []
        for src in (asia, london, prev_ny):
            if d in src.index and not np.isnan(src.loc[d, "h"]):
                highs.append(float(src.loc[d, "h"]))
                lows.append(float(src.loc[d, "l"]))
        t_open = open_ts[d]
        lo_i = hb_close.searchsorted(t_open - pd.Timedelta(hours=24))
        last_i = hb_close.searchsorted(t_open, side="right") - 1     # last 1-hour bar closed by 09:30
        for j in range(lo_i, last_i):                               # j + 1 <= last_i: confirmed by 09:30
            after_hi = hh[j + 1:last_i + 1].max(initial=-np.inf)
            after_lo = hl[j + 1:last_i + 1].min(initial=np.inf)
            if sh[j] and after_hi < hh[j]:
                highs.append(float(hh[j]))
            if sl[j] and after_lo > hl[j]:
                lows.append(float(hl[j]))
        out[d] = (highs, lows)
    return out


def run(args) -> pd.DataFrame:
    tf, am_only, ifvg, min_rr, pe = args
    df = load(tf)
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    day = np.asarray((et + pd.Timedelta(hours=6)).date)
    levels = day_levels(df)
    # 5-minute reversal chart, and which entry bar completes each 5-minute bar
    if tf == "1m":
        f5 = df.resample("5min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        end5 = (f5.index + pd.Timedelta(minutes=4)).as_unit("ns")
        pos5 = pd.Series(np.arange(len(f5)), index=end5)
        comp5 = pd.Series(df.index).map(pos5).fillna(-1).astype(int).to_numpy()
    else:
        f5 = df
        comp5 = np.arange(len(df))
    h5, l5, c5 = f5["high"].to_numpy(), f5["low"].to_numpy(), f5["close"].to_numpy()
    sh5, sl5 = pivots(h5, l5, 1)
    sh_e, sl_e = pivots(h, l, pe)
    win_end = 720 if am_only else 930
    out = []
    pos = None
    cur, st, alive_h, alive_l, traded = None, None, [], [], False
    for i in range(1, len(c)):
        if day[i] != cur:
            cur = day[i]
            hs, ls = levels.get(cur, ([], []))
            alive_h, alive_l = list(hs), list(ls)
            st, traded = None, False
        # --- open position ---
        if pos is not None:
            side, ent, stp, tgt, ie = pos
            px = how = None
            if 1000 <= mins[i] < 1080:
                px, how = o[i], "16:40 close"
            elif (l[i] <= stp) if side > 0 else (h[i] >= stp):
                px, how = stp - side * TICK, "stop"
            elif i > ie and ((h[i] > tgt) if side > 0 else (l[i] < tgt)):
                px, how = tgt, "target"
            if px is not None:
                risk = abs(ent - stp)
                out.append((df.index[ie], df.index[i], side, risk, (side * (px - ent) - COMM - SPREAD) / risk, how))
                pos = None
        in_win = 570 <= mins[i] < win_end
        if not in_win:
            if mins[i] >= win_end or mins[i] < 570:
                st = None if mins[i] >= win_end else st
            # liquidity taken outside the window is gone too (from 09:30 on)
            continue
        swept_h = [v for v in alive_h if h[i] > v]
        swept_l = [v for v in alive_l if l[i] < v]
        alive_h = [v for v in alive_h if h[i] <= v]
        alive_l = [v for v in alive_l if l[i] >= v]
        if traded:
            continue
        # --- 2. manipulation ---
        if st is None:
            if swept_h:
                st = {"side": -1, "s": "swept", "ext": h[i], "ext_i": i}
            elif swept_l:
                st = {"side": 1, "s": "swept", "ext": l[i], "ext_i": i}
            continue
        side = st["side"]
        if st["s"] == "swept":
            if (h[i] > st["ext"]) if side < 0 else (l[i] < st["ext"]):
                st.update(ext=(h[i] if side < 0 else l[i]), ext_i=i)
            k = comp5[i]
            if k >= 2:
                rev = False
                # break of structure: close beyond the last confirmed 5-min swing on our side
                for j in range(k - 2, max(k - 40, 0), -1):
                    if side < 0 and sl5[j]:
                        rev = c5[k] < l5[j]
                        break
                    if side > 0 and sh5[j]:
                        rev = c5[k] > h5[j]
                        break
                if not rev and ifvg:                   # close through a 5-min gap left by the push
                    for j in range(k, max(k - 12, 2), -1):
                        if side < 0 and l5[j] > h5[j - 2] and c5[k] < h5[j - 2] and j < k:
                            rev = True
                            break
                        if side > 0 and h5[j] < l5[j - 2] and c5[k] > l5[j - 2] and j < k:
                            rev = True
                            break
                if rev:
                    st.update(s="reversed", t_rev=i, retr=False, rext=None)
                    continue
            span = (i - st["ext_i"]) if tf == "1m" else (i - st["ext_i"]) * 5
            if span > 30:
                st = None                              # no reversal in 30 minutes: forget it
            continue
        # --- 4. continuation on the entry chart ---
        if (h[i] > st["ext"]) if side < 0 else (l[i] < st["ext"]):
            st = None                                  # went beyond the manipulation's extreme
            continue
        if i - st["t_rev"] > 60:
            st = None
            continue
        t0 = st["t_rev"]
        hi_since = h[t0:i + 1].max()
        lo_since = l[t0:i + 1].min()
        last_sh = next((h[j] for j in range(i - pe, t0 - 1, -1) if sh_e[j]), None)
        last_sl = next((l[j] for j in range(i - pe, t0 - 1, -1) if sl_e[j]), None)
        if not st["retr"]:
            lvl = last_sh if side < 0 else last_sl
            if lvl is not None and ((c[i] > lvl) if side < 0 else (c[i] < lvl)):
                st["retr"] = True
            continue
        lvl = last_sl if side < 0 else last_sh
        if lvl is None or not ((c[i] < lvl) if side < 0 else (c[i] > lvl)):
            continue
        ent = c[i]
        stp = (hi_since + TICK) if side < 0 else (lo_since - TICK)
        risk = side * (ent - stp)
        st = None
        if risk <= 0:
            continue
        draws = sorted((v for v in alive_l if v < ent), reverse=True) if side < 0 else sorted(v for v in alive_h if v > ent)
        tgt = next((v for v in draws if side * (v - ent) >= min_rr * risk), None)
        if tgt is None or pos is not None:
            continue
        pos = (side, ent, stp, tgt, i)
        traded = True
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
    pd.set_option("display.width", 230)
    fm = {c: (lambda v: f"{v:+.3f}") for c in ("avg R", "first R", "later R")}
    fm.update({"z first": lambda v: f"{v:+.2f}", "z later": lambda v: f"{v:+.2f}", "win%": lambda v: f"{v:.1%}",
               "per year": lambda v: f"{v:.0f}"})
    for tf, years, a_end, label in (("1m", 2.55, "2025-07-01", "1-minute entries (as in the video), Mar 2024 - Oct 2026"),
                                    ("5m", 5.65, "2024-01-01", "5-minute entries (longer history), Feb 2021 - Oct 2026")):
        grid = list(itertools.product([tf], (True, False), (False, True), (1.0, 1.5, 2.0), (1, 2)))
        with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
            results = list(ex.map(run, grid))
        rows = []
        for g, t in zip(grid, results):
            n, w, r, z = summary(t)
            _, _, ra, za = summary(t, "2000-01-01", a_end)
            _, _, rb, zb = summary(t, a_end, "2100-01-01")
            rows.append({"window": "09:30-12:00" if g[1] else "09:30-15:30", "reversal": "BOS or inverse gap" if g[2] else "BOS",
                         "min target": f"{g[3]:g}R", "swing": g[4], "trades": n, "per year": n / years, "win%": w,
                         "avg R": r, "first R": ra, "z first": za, "later R": rb, "z later": zb, "_t": t})
        res = pd.DataFrame(rows).sort_values("first R", ascending=False)
        print(f"\n=== TJR, {label}; 'first' = up to {a_end} (choose), 'later' = after (judge) ===")
        print(res.drop(columns=["_t"]).to_string(index=False, formatters=fm))
        print(f"all {len(res)} versions: average {res['avg R'].mean():+.3f} R/trade; later part positive in "
              f"{(res['later R'] > 0).mean():.0%} of versions (average {res['later R'].mean():+.3f} R)")
        b = res.iloc[0]
        t = b["_t"]
        days = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.date)["R"].sum()
        months = t.groupby(t["opened"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m"))["R"].sum()
        print(f"CHOSEN: window {b['window']}, reversal {b['reversal']}, min target {b['min target']}, swing {b['swing']}: "
              f"first {b['first R']:+.3f} R (z {b['z first']:+.2f}) -> later {b['later R']:+.3f} R (z {b['z later']:+.2f}); "
              f"win {b['win%']:.1%}, {b['per year']:.0f} trades a year, winning days {np.mean(days > 0):.0%}, "
              f"winning months {np.mean(months > 0):.0%} of {len(months)}; exits {t['how'].value_counts().to_dict()}")
        res.drop(columns=["_t"]).to_csv(ROOT / "data" / f"research_tjr_{tf}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
