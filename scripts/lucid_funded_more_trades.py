"""Funded account: does trading MORE times a day help? (owner, 2026-10-08: "we need more trades per day on funded")

A minute-by-minute replay of a LucidPro 50K funded account on real NQ bars (NinjaTrader 1-minute 2024-26 and
5-minute 2021-26), every start day, up to 120 trading days per account, under the rules the funded bot uses:
  loss limit (MLL) $2,000 below the best end-of-day balance, counted on open equity, locked at $50,100 once an
  end-of-day balance reaches $52,100; payouts at the day's close after >= 3 trading days, >= $500 cycle profit,
  best day <= 40% of the cycle, $52,100 kept -- first $500 (cap $2,000), then $2,500; five payouts = done.
Every version races the side of the 200-day trend (as deployed), pays the spread on entry, 1 tick on stops,
$3.50 commission, and assumes the worse of stop / target when one bar touches both.

Versions:
  current      one race from 19:00 ET, take-profit = what the next payout still needs ($150-$1,040), no stop but the
               loss line; pause once the payout needs are met (what runs now)
  3 races      the same day take split into 3 smaller races in a row
  scalps       many trades a day with their own stop and target (1:1 $400, 2:1 $600/$300, 1:1 $200, and 1:1 $400 in
               New York hours only), until the day take is reached or the day is down $1,000
Scored per funded account and per week of one account slot (evaluation fees from the race's 41% pass rate).

Usage:  .venv/Scripts/python.exe scripts/lucid_funded_more_trades.py            (more trades a day)
        .venv/Scripts/python.exe scripts/lucid_funded_more_trades.py --safer    (one race, own stop / smaller take)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ET = "America/New_York"
PV, TICK, COMM = 20.0, 0.25, 3.50
KEEP, LOCK_FLOOR, MLL = 2100.0, 100.0, 2000.0
WANT = np.array([500.0, 2500.0, 2500.0, 2500.0, 2500.0])
CAP = np.array([2000.0, 2500.0, 2500.0, 2500.0, 2500.0])
H = 120
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5
DATA = {"NQ 1-min 2024-26": "data/futures/NQ_1m_long_nt.parquet", "NQ 5-min 2021-26": "data/futures/NQ_5m_long_nt.parquet"}
VARIANTS = {
    "current: 1 race, take = payout need": dict(kind="race", split=1),
    "3 smaller races a day": dict(kind="race", split=3),
    "scalps 1:1 $400/$400": dict(kind="scalp", tp=400.0, sl=400.0),
    "scalps 2:1 $600/$300": dict(kind="scalp", tp=600.0, sl=300.0),
    "scalps 1:1 $200/$200": dict(kind="scalp", tp=200.0, sl=200.0),
    "scalps 1:1 $400/$400 NY hours": dict(kind="scalp", tp=400.0, sl=400.0, rth=True),
}
SAFER = {   # owner, 2026-10-08: "we need a strategy for the funded bot" -- one race a day, but survive a loss
    "current: 1 race, take = payout need": dict(kind="race", split=1),
    "1 race, own stop $1,000 (2 lives)": dict(kind="scalp", tp=1040.0, sl=1000.0, once=True),
    "1 race, own stop $650 (3 lives)": dict(kind="scalp", tp=1040.0, sl=650.0, once=True),
    "1 race, own stop $400": dict(kind="scalp", tp=1040.0, sl=400.0, once=True),
    "1 race, take at most $520 a day": dict(kind="race", split=1, cap=520.0),
    "1 race, own stop $1,000, take at most $520": dict(kind="scalp", tp=520.0, sl=1000.0, once=True),
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / path)
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(ET)
    return df.sort_index()


def trend_sides() -> dict:
    df = load(DATA["NQ 5-min 2021-26"])
    m = df.index.hour * 60 + df.index.minute
    rth = df[(m >= 570) & (m < 960)]
    day = (rth.index + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    dc = rth["close"].groupby(day).last()
    side = np.sign(dc - dc.rolling(200).mean()).shift(1).dropna()
    return {d: (1 if s >= 0 else -1) for d, s in side.items()}


def sessions(df: pd.DataFrame, sides: dict) -> list:
    m = df.index.hour * 60 + df.index.minute
    df = df[(m >= 19 * 60) | (m < 16 * 60)]
    m = df.index.hour * 60 + df.index.minute
    day = (df.index + pd.Timedelta(hours=6)).normalize().tz_localize(None)
    step = int(np.median(np.diff(m[:2000]))) if len(m) > 1 else 1
    step = 5 if step >= 5 else 1
    full = 21 * 60 // step
    out = []
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    d = day.to_numpy()
    cut = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        k = pd.Timestamp(d[a])
        if k not in sides or b - a < 0.7 * full or not (m[a] >= 19 * 60):
            continue
        out.append((k, sides[k], o[a:b], h[a:b], l[a:b], c[a:b], np.asarray(m[a:b])))
    return out


def simulate(sess: list, var: dict) -> dict:
    n_days = len(sess)
    starts = np.arange(0, n_days - H)
    A = len(starts)
    bal, peak, paid = np.zeros(A), np.zeros(A), np.zeros(A)
    npay, cyc_days, alive, trades = (np.zeros(A, int) for _ in range(4))
    cyc_pnl, cyc_best = np.zeros(A), np.zeros(A)
    first_day = np.full(A, -1)
    blown, fin = np.zeros(A, bool), np.zeros(A, bool)
    race = var["kind"] == "race"
    for d in range(n_days):
        act = np.flatnonzero((starts <= d) & (d < starts + H) & ~fin)
        if not len(act):
            continue
        _, side, o, h, l, c, mins = sess[d]
        k = np.minimum(npay[act], 4)
        need = np.maximum.reduce([KEEP + WANT[k] - bal[act],
                                  np.where(cyc_best[act] > 0, cyc_best[act] / 0.4 - cyc_pnl[act], 0.0),
                                  500.0 - cyc_pnl[act]])
        cap = np.clip(need, 150.0, var.get("cap", 1040.0))
        floor = np.minimum(peak[act] - MLL, LOCK_FLOOR)
        b0 = bal[act].copy()
        n = len(act)
        dp = np.zeros(n)
        stop_day = need <= 0                      # payout needs met: pause until the cycle has 3 days
        dead = np.zeros(n, bool)
        inpos = np.zeros(n, bool)
        ent, tp, sl, line = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        ntr = np.zeros(n, int)
        last = len(o) - 1
        for i in range(len(o)):
            mm = mins[i]
            if var.get("rth"):
                window = 570 <= mm < 930
            elif (race and var["split"] == 1) or var.get("once"):
                window = mm >= 19 * 60 or mm < 12 * 60
            else:
                window = mm >= 19 * 60 or mm < 930
            if window and i < last:
                e = ~inpos & ~stop_day & ~dead
                if e.any():
                    px = o[i] + side * TICK
                    left = cap[e] - dp[e]
                    cushion = b0[e] + dp[e] - floor[e]
                    if race:
                        t_usd = np.minimum(np.maximum(cap[e] / var["split"], 150.0), left)
                        s_usd = cushion + 1e6                      # no own stop: only the loss line
                    else:
                        t_usd = np.minimum(var["tp"], left)
                        s_usd = np.minimum(var["sl"], np.minimum(1000.0 + dp[e], cushion - 50.0))
                    ok = (t_usd >= 50.0) & (s_usd >= 50.0)
                    ee = np.flatnonzero(e)[ok]
                    stop_day[np.flatnonzero(e)[~ok]] = True
                    # too close to the loss line to place even a 2.5-point stop: the account is finished --
                    # start a new one (counted as blown) instead of leaving it idle for months
                    dead[np.flatnonzero(e)[~ok & (cushion < 100.0)]] = True
                    tpts = np.maximum(np.round(t_usd[ok] / PV / TICK) * TICK, TICK)
                    spts = np.maximum(np.round(s_usd[ok] / PV / TICK) * TICK, TICK)
                    ent[ee] = px
                    tp[ee] = px + side * tpts
                    sl[ee] = px - side * spts
                    line[ee] = px - side * cushion[ok] / PV
                    inpos[ee] = True
            if not inpos.any():
                continue
            adv, fav = (l[i], h[i]) if side > 0 else (h[i], l[i])
            p = inpos.copy()
            hit_line = p & (side * (adv - line) <= 0)
            hit_sl = p & ~hit_line & (side * (adv - sl) <= 0)
            hit_tp = p & ~hit_line & ~hit_sl & (side * (fav - tp) >= 0)
            eod = p & ~hit_line & ~hit_sl & ~hit_tp & (i == last)
            pnl = np.zeros(n)
            pnl[hit_line] = side * (line[hit_line] - side * TICK - ent[hit_line]) * PV - COMM
            pnl[hit_sl] = side * (sl[hit_sl] - side * TICK - ent[hit_sl]) * PV - COMM
            pnl[hit_tp] = side * (tp[hit_tp] - ent[hit_tp]) * PV - COMM
            pnl[eod] = side * (c[i] - side * TICK - ent[eod]) * PV - COMM
            out = hit_line | hit_sl | hit_tp | eod
            if out.any():
                dp += pnl
                ntr += out
                inpos[out] = False
                dead |= hit_line
                if (race and var["split"] == 1) or var.get("once"):
                    stop_day |= out
                else:
                    stop_day |= out & ((dp >= cap - 1e-6) | (dp <= -1000.0 + 1e-6))
        bal[act] = b0 + dp
        trades[act] += ntr
        alive[act] += 1
        blown[act[dead]] = True
        fin[act[dead]] = True
        live = act[~dead]
        dpl = dp[~dead]
        peak[live] = np.maximum(peak[live], bal[live])
        cyc_days[live] += 1
        cyc_pnl[live] += dpl
        cyc_best[live] = np.maximum(cyc_best[live], dpl)
        kk = np.minimum(npay[live], 4)
        avail = bal[live] - KEEP
        due = (cyc_days[live] >= 3) & (cyc_pnl[live] >= 500) & (cyc_best[live] <= 0.4 * cyc_pnl[live]) & (avail >= WANT[kk])
        if due.any():
            w = live[due]
            amt = np.minimum(CAP[kk[due]], avail[due])
            bal[w] -= amt
            paid[w] += amt
            npay[w] += 1
            first_day[w] = np.where(first_day[w] < 0, alive[w], first_day[w])
            cyc_days[w], cyc_pnl[w], cyc_best[w] = 0, 0.0, 0.0
            fin[w[npay[w] >= 5]] = True
    got = npay >= 1
    net = 0.9 * paid.mean()
    return {"accounts": A, "payout reached %": got.mean(),
            "median days to 1st payout": float(np.median(first_day[got])) if got.any() else np.nan,
            "blown %": blown.mean(), "paid per account $": paid.mean(), "to you per account $": net,
            "trades per day": trades.sum() / max(1, alive.sum()), "days per account": alive.mean(),
            "$ per week per slot": (net - EVAL_COST) / (EVAL_WEEKS + alive.mean() / 5)}


def job(x):
    dname, vname = x
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    sides = trend_sides()
    sess = sessions(load(DATA[dname]), sides)
    r = simulate(sess, {**VARIANTS, **SAFER}[vname])
    return {"data": dname, "version": vname, **r}


def main() -> int:
    grid = [(d, v) for d in DATA for v in (SAFER if "--safer" in sys.argv else VARIANTS)]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 2)) as ex:
        rows = list(ex.map(job, grid))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    fm = {"payout reached %": "{:.1%}".format, "blown %": "{:.1%}".format, "trades per day": "{:.2f}".format,
          "median days to 1st payout": "{:.0f}".format, "days per account": "{:.0f}".format}
    for col in ("paid per account $", "to you per account $", "$ per week per slot"):
        fm[col] = (lambda v: f"{v:,.0f}")
    print(res.to_string(index=False, formatters=fm))
    res.to_csv(ROOT / "data" / ("lucid_funded_safer.csv" if "--safer" in sys.argv else "lucid_funded_more_trades.csv"),
               index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
