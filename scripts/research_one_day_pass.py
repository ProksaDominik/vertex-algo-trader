"""'LucidPro 50K one-day pass' plan (owner, 2026-10-08) -- tested exactly as its own spec asks:
historical one-day simulations (500+ setups), the full evaluation, and a 10,000-run Monte Carlo.

Setup (longs; shorts mirrored), NQ 5-minute bars (= MNQ prices), window 09:35-11:00 New York:
  levels   overnight low (18:00-09:30), previous New York session low, the 09:30-09:35 candle's low, and
           15-minute swing lows (2 bars each side) of the last 24 h that price has not traded through.
  sweep    a candle trades below a level; REJECTION = a candle closes back above it within 3 candles.
  displace within 6 candles: a bullish candle, body >= *0.8 or 1.2 x ATR(14), closing in the top quarter of
           its range, closing above the last 5-minute swing high (1 bar each side) AND above the session VWAP.
  retest   limit entry at *the broken swing high, or *the displacement candle's midpoint, until 11:00;
           cancelled if price trades below the sweep low first.
  stop     1 tick below the sweep low. Room filter (*on/off): the nearest opposing level (overnight high,
           previous day high, opening candle high, 15-minute swing highs) at least 2R above the entry.
  size     MNQ = floor($1,500 / (stop x $2)), max 40; target = +$3,000 net ($ fixed -> the plan's one-trade pass).
  news     no trade on CPI / jobs report / Fed days (dates in data/futures/event_dates.json).
  second   after a loss, a second qualifying setup the same day ONLY if Lucid allows it: its $1,200 daily loss
           limit counts realized losses, so a -$1,200 or worse first loss ends the day. Second trade: $350
           risk, 2R target.
Costs: $0.50 per MNQ per side commission; 1 tick slippage on stops (limit entries and targets fill at price);
a candle reaching stop and target counts as the stop; no target on the entry candle; out at 16:00.
Evaluation: start $0, target +$3,000, loss limit $2,000 below the best end-of-day balance (locks at +$100),
open trades count. One-day pass = +$3,000 the same day. Full evaluation: the plan repeated each day (first-
trade risk = min($1,500, room left - $100), target = what is still needed, at least 2R) until pass or breach.
Data: NinjaTrader NQ 5-min Feb 2021 - Oct 2026 (choose 2021-2023, judge 2024-2026).

Usage:  .venv/Scripts/python.exe scripts/research_one_day_pass.py
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICK, USD_PT, COMM_SIDE = 0.25, 2.0, 0.50
RNG = np.random.default_rng(11)


def load():
    df = pd.read_parquet(ROOT / "data" / "futures" / "NQ_5m_long_nt.parquet")[["open", "high", "low", "close", "volume"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    return df[~df.index.duplicated()].sort_index()


def news_days():
    ev = json.loads((ROOT / "data" / "futures" / "event_dates.json").read_text(encoding="utf-8"))
    return {pd.Timestamp(d).date() for k in ("cpi", "nfp", "fomc") for d in ev[k]}


def setups(df: pd.DataFrame, body_k: float, entry_kind: str, room_filter: bool):
    """Every qualifying setup (in time order): its entry, stop and the price path after the entry, so that
    any size / target can be evaluated later."""
    o, h, l, c, v = (df[x].to_numpy() for x in ("open", "high", "low", "close", "volume"))
    n = len(c)
    mins = np.asarray(df.index.hour * 60 + df.index.minute)
    cme = np.asarray((df.index + pd.Timedelta(hours=6)).date)
    cal = np.asarray(df.index.date)
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    # 15-minute swings
    b15 = df.resample("15min", label="left", closed="left").agg({"high": "max", "low": "min"}).dropna()
    h15, l15 = b15["high"].to_numpy(), b15["low"].to_numpy()
    t15 = b15.index + pd.Timedelta(minutes=15)
    sh15 = [(t15[j + 2], h15[j]) for j in range(2, len(h15) - 2) if h15[j] > h15[j - 2:j].max() and h15[j] >= h15[j + 1:j + 3].max()]
    sl15 = [(t15[j + 2], l15[j]) for j in range(2, len(l15) - 2) if l15[j] < l15[j - 2:j].min() and l15[j] <= l15[j + 1:j + 3].min()]
    sh15_t = pd.DatetimeIndex([t for t, _ in sh15])
    sl15_t = pd.DatetimeIndex([t for t, _ in sl15])
    skip = news_days()
    out = []
    day_idx = pd.Series(np.arange(n)).groupby(cme).apply(lambda s: s.to_numpy())
    prev_rth = None
    for d, ix in day_idx.items():
        m = mins[ix]
        rth = ix[(m >= 570) & (m < 960)]
        on = ix[(m >= 1080) | (m < 570)]
        if len(rth) < 30 or len(on) < 30:
            prev_rth = (h[rth].max(), l[rth].min()) if len(rth) else prev_rth
            continue
        today = cal[rth[0]]
        open_bar = rth[0]
        t_open = df.index[open_bar]
        if prev_rth is not None and today not in skip:
            onh, onl = h[on].max(), l[on].min()
            pdh, pdl = prev_rth
            orh, orl = h[open_bar], l[open_bar]
            # 15-minute swings of the last 24 h, untouched until 09:35
            lo_t, hi_t = t_open - pd.Timedelta(hours=24), t_open + pd.Timedelta(minutes=5)
            highs15 = [p for t, p in sh15[sh15_t.searchsorted(lo_t):sh15_t.searchsorted(hi_t)]]
            lows15 = [p for t, p in sl15[sl15_t.searchsorted(lo_t):sl15_t.searchsorted(hi_t)]]
            pre = ix[(ix < open_bar + 1)]
            highs15 = [p for p in highs15 if h[open_bar] <= p]
            lows15 = [p for p in lows15 if l[open_bar] >= p]
            lv_lo = [onl, pdl, orl] + lows15
            lv_hi = [onh, pdh, orh] + highs15
            vw_n = vw_d = 0.0
            win = rth[(mins[rth] >= 575) & (mins[rth] < 660)]
            for i in rth[:1]:
                vw_n += (h[i] + l[i] + c[i]) / 3 * max(v[i], 1)
                vw_d += max(v[i], 1)
            states = {1: None, -1: None}
            for i in win:
                vw_n += (h[i] + l[i] + c[i]) / 3 * max(v[i], 1)
                vw_d += max(v[i], 1)
                vwap = vw_n / vw_d
                for side in (1, -1):
                    s = states[side]
                    levels = lv_lo if side > 0 else lv_hi
                    if s is None or s["s"] == "dead":
                        hit = [L for L in levels if ((l[i] < L <= h[i - 1] + 1e9) if side > 0 else (h[i] > L))
                               and ((l[i - 1] >= L) if side > 0 else (h[i - 1] <= L))]
                        if hit:
                            L = max(hit) if side > 0 else min(hit)
                            states[side] = {"s": "swept", "L": L, "ext": l[i] if side > 0 else h[i], "t": i}
                        continue
                    if s["s"] in ("swept", "rejected", "displaced"):
                        s["ext"] = min(s["ext"], l[i]) if side > 0 else max(s["ext"], h[i])
                    if s["s"] == "swept":
                        if (c[i] > s["L"]) if side > 0 else (c[i] < s["L"]):
                            s.update(s="rejected", tr=i)
                        elif i - s["t"] >= 3:
                            s["s"] = "dead"
                        continue
                    if s["s"] == "rejected":
                        if i - s["tr"] > 6:
                            s["s"] = "dead"
                            continue
                        rng = h[i] - l[i]
                        body = side * (c[i] - o[i])
                        strong = rng > 0 and not np.isnan(atr[i]) and body >= body_k * atr[i] and (
                            (c[i] >= l[i] + 0.75 * rng) if side > 0 else (c[i] <= h[i] - 0.75 * rng))
                        if not strong or not ((c[i] > vwap) if side > 0 else (c[i] < vwap)):
                            continue
                        # the last confirmed 5-minute swing against us (lower high for longs)
                        struct = None
                        for j in range(i - 2, max(s["t"] - 12, rth[0]), -1):
                            if side > 0 and h[j] > h[j - 1] and h[j] >= h[j + 1]:
                                struct = h[j]
                                break
                            if side < 0 and l[j] < l[j - 1] and l[j] <= l[j + 1]:
                                struct = l[j]
                                break
                        if struct is None or not ((c[i] > struct) if side > 0 else (c[i] < struct)):
                            continue
                        ent = struct if entry_kind == "broken level" else (o[i] + c[i]) / 2
                        stp = s["ext"] - side * TICK
                        if side * (ent - stp) <= 2 * TICK:
                            s["s"] = "dead"
                            continue
                        s.update(s="displaced", td=i, ent=ent, stp=stp)
                        continue
                    if s["s"] == "displaced":
                        if (l[i] <= s["stp"] + side * TICK) if side > 0 else (h[i] >= s["stp"] + side * TICK):
                            if not ((l[i] <= s["ent"]) if side > 0 else (h[i] >= s["ent"])):
                                s["s"] = "dead"                     # back beyond the sweep before the retest
                                continue
                        if (l[i] <= s["ent"]) if side > 0 else (h[i] >= s["ent"]):
                            ent, stp = s["ent"], s["stp"]
                            ent = min(o[i], ent) if side > 0 else max(o[i], ent)
                            risk_pts = side * (ent - stp)
                            if risk_pts <= 0:
                                s["s"] = "dead"
                                continue
                            opp = [L for L in (lv_hi if side > 0 else lv_lo) if side * (L - ent) > 0]
                            room = min(side * (L - ent) for L in opp) if opp else np.inf
                            if room_filter and room < 2 * risk_pts:
                                s["s"] = "dead"
                                continue
                            # the path after the entry: favourable excursion before the stop, stopped?, 16:00 result
                            rest = ix[(ix >= i) & (mins[ix] < 960)]
                            mfe, stopped, end_pts = 0.0, False, 0.0
                            for k in rest:
                                adverse = (l[k] <= stp) if side > 0 else (h[k] >= stp)
                                if adverse:
                                    stopped = True
                                    break
                                if k > i:
                                    mfe = max(mfe, side * ((h[k] if side > 0 else l[k]) - ent))
                                end_pts = side * (c[k] - ent)
                            out.append({"day": pd.Timestamp(today), "time": df.index[i], "side": side, "ent": ent,
                                        "risk_pts": risk_pts, "mfe": mfe, "stopped": stopped, "end_pts": end_pts})
                            s["s"] = "dead"
        prev_rth = (h[rth].max(), l[rth].min())
    return pd.DataFrame(out)


def trade_usd(row, risk_usd, target_usd, max_k=40):
    """$ result of one setup at a given dollar risk and dollar target (net of costs); None if it cannot be sized."""
    k = int(min(max_k, math.floor(risk_usd / (row.risk_pts * USD_PT))))
    if k < 1:
        return None, 0
    cost = 2 * COMM_SIDE * k
    tgt_pts = (target_usd + cost) / (k * USD_PT)
    if row.mfe >= tgt_pts:
        return target_usd, k
    if row.stopped:
        return -(row.risk_pts + TICK) * k * USD_PT - cost, k
    return row.end_pts * k * USD_PT - cost, k


def simulate_eval(days, by_day, first_risk=1500.0, second_risk=350.0, dll=1200.0, target=3000.0, mll=2000.0, max_days=252):
    """One evaluation over consecutive trading days: (passed, days used, first-day pass?)."""
    bal = peak_eod = 0.0
    for di, d in enumerate(days[:max_days]):
        floor = min(peak_eod - mll, 100.0)
        room = bal - floor
        trades = by_day.get(d, [])
        day_real = 0.0
        for ti, row in enumerate(trades[:2]):
            if ti == 0:
                risk = min(first_risk, room - 100.0)
                need = target - bal
                tgt = max(need, 2 * risk)
            else:
                if day_real <= -dll or day_real >= 0:
                    break
                risk = min(second_risk, room - 100.0)
                tgt = 2 * risk
            if risk < 50:
                break
            res, k = trade_usd(row, risk, min(tgt, target - bal) if ti == 0 else tgt)
            if res is None:
                continue
            # open equity reached the stop level before closing? (counted at the stop)
            bal += res
            day_real += res
            room = bal - floor
            if bal <= floor:
                return False, di + 1, False
            if bal >= target:
                return True, di + 1, di == 0
            if res > 0:
                break
        peak_eod = max(peak_eod, bal)
    return None, max_days, False


def main() -> int:
    df = load()
    trading_days = sorted({pd.Timestamp(d) for d in df.index[(df.index.hour * 60 + df.index.minute >= 570) &
                                                               (df.index.hour * 60 + df.index.minute < 960)].date})
    rows = []
    keep = {}
    for body_k, entry_kind, room_filter in itertools.product((0.8, 1.2), ("broken level", "candle midpoint"), (True, False)):
        st = setups(df, body_k, entry_kind, room_filter)
        if st.empty:
            continue
        first = st.groupby("day").head(1)
        res_1 = [trade_usd(r, 1500.0, 3000.0)[0] for r in first.itertuples()]
        res_1 = np.array([x for x in res_1 if x is not None])
        by_day = {d: list(g.itertuples()) for d, g in st.groupby("day")}
        hist = [simulate_eval(trading_days[s:], by_day) for s in range(0, len(trading_days) - 260, 3)]
        label = f"body {body_k} ATR / entry at {entry_kind} / room filter {'on' if room_filter else 'off'}"
        a = first[first.day < pd.Timestamp("2024-01-01")]
        b = first[first.day >= pd.Timestamp("2024-01-01")]
        win_a = np.mean([trade_usd(r, 1500, 3000)[0] >= 3000 for r in a.itertuples() if trade_usd(r, 1500, 3000)[0] is not None])
        win_b = np.mean([trade_usd(r, 1500, 3000)[0] >= 3000 for r in b.itertuples() if trade_usd(r, 1500, 3000)[0] is not None])
        rows.append({"version": label, "setups": len(st), "days with a setup": first.day.nunique(),
                     "setup days / year": first.day.nunique() / 5.65,
                     "ONE-DAY PASS (1st trade hits +$3,000)": np.mean(res_1 >= 3000),
                     "2021-23": win_a, "2024-26": win_b,
                     "avg 1st trade $": res_1.mean(),
                     "FULL EVAL pass % (within a year)": np.mean([x[0] is True for x in hist]),
                     "failed %": np.mean([x[0] is False for x in hist]),
                     "still open after a year %": np.mean([x[0] is None for x in hist]),
                     "median days to pass": np.median([x[1] for x in hist if x[0]]) if any(x[0] for x in hist) else np.nan})
        keep[label] = (st, by_day, res_1)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("=== historical simulation, NQ/MNQ 5-min Feb 2021 - Oct 2026 (the plan needs >= 60% one-day pass) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:,.0f}"))
    for label in [v for v in keep if len(keep[v][0]) >= 100] + [v for v in keep if len(keep[v][0]) < 100][:1]:
        st, by_day, res_1 = keep[label]
        days_sorted = sorted(by_day)
        setup_share = len(days_sorted) / len(trading_days)
        sims = []
        for _ in range(10000):
            fake = {fd: by_day[days_sorted[RNG.integers(len(days_sorted))]] for fd in range(252) if RNG.random() < setup_share}
            sims.append(simulate_eval(list(range(252)), fake))
        one_day = RNG.choice(res_1, size=10000)
        streak = worst = 0
        for x in res_1:
            streak = streak + 1 if x < 0 else 0
            worst = max(worst, streak)
        passed = [x for x in sims if x[0] is True]
        tag = " -- TOO FEW to judge" if len(st) < 100 else ""
        print(f"\n=== Monte Carlo, 10,000 evaluations: {label} ({len(st)} setups{tag}) ===")
        print(f"  ONE-DAY pass probability on a setup day: {np.mean(one_day >= 3000):.1%}   (the plan requires >= 60%)")
        print(f"  full evaluation within a year: passed {np.mean([x[0] is True for x in sims]):.1%} | failed (loss limit) "
              f"{np.mean([x[0] is False for x in sims]):.1%} | still open {np.mean([x[0] is None for x in sims]):.1%}")
        med = np.median([x[1] for x in passed]) if passed else float("nan")
        print(f"  median trading days to pass: {med:.0f}; first trade: average ${res_1.mean():+,.0f}, 5th / 95th percentile "
              f"${np.percentile(res_1, 5):+,.0f} / ${np.percentile(res_1, 95):+,.0f}; worst losing streak {worst}")
    print("\n(our race: ~41% of evaluations pass, most in 1-2 days)")
    res.to_csv(ROOT / "data" / "research_one_day_pass.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
