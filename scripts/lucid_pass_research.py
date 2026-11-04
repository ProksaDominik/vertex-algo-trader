"""LucidPro 50K evaluation: which strategy / stack gives the highest PASS RATE?

Owner's brief (2026-09-30): pass the LucidPro 50K evaluation -- ideally in one
day -- with a 60-70% pass rate; several strategies may be stacked.

Rules replayed (LucidPro 50K evaluation, strict readings)
---------------------------------------------------------
* start $50,000, pass at $53,000. The bot flattens everything the moment its
  equity reaches the target (account-level take profit, checked at 15m bar
  closes -- never at the bar's best tick).
* MLL: floor = highest END-OF-DAY balance - $2,000 (trailing, no lock), checked
  against INTRADAY equity including open positions, using the worst price of
  every bar for every leg at the same time (pessimistic).
* DLL $1,200: when the day's P&L (open positions included) touches -$1,200,
  everything is flattened and nothing else trades until the next session.
* The Lucid trading day runs 18:00 ET -> 16:45 ET. Positions may be held
  through the Globex night session; only 16:45-18:00 ET must be flat. This is
  what makes night-session strategies legal on LucidPro.

Data and costs
--------------
10.5 years (2016-2026) of 15m proxies (data/*_15m_long.parquet) laid on a
91-slot grid per Lucid day (slot 0 = the 18:00 ET bar, slot 62 = 09:30, slot
87 = 15:45, slot 90 = 16:30). Every trade pays 2 ticks of slippage plus the
Lucid round-trip commission of the contract, charged at entry.

The evaluation is replayed from EVERY historical start day (rolling starts,
real day order, no resampling) until pass / fail / horizon.

This module is the engine (grid, legs, replay); scripts/lucid_pass_study.py
runs the study and prints every table of docs/LUCIDPRO_PASS_RESEARCH.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402
from metatraderbot import futures  # noqa: E402

OUT = ROOT / "data" / "futures"
K = 91                                    # 15m slots per Lucid day: 18:00 .. 16:30 ET


def S(h: int, m: int = 0) -> int:
    """Grid slot of the 15m bar that OPENS at h:m ET (18:00 -> 0, 09:30 -> 62)."""
    mins = h * 60 + m
    return (mins - 1080) // 15 if mins >= 1080 else (mins + 360) // 15


RTH_OPEN, RTH_LAST = S(9, 30), S(15, 45)

# research market -> (proxy file, micro/cheapest Lucid contract, inverted quote)
MARKETS: dict[str, tuple[str, str, bool]] = {
    "ES": ("us500", "MES", False),
    "NQ": ("nas100", "MNQ", False),
    "YM": ("us30", "MYM", False),
    "GC": ("xauusd", "MGC", False),
    "CL": ("wti", "MCL", False),
    "HG": ("copper", "HG", False),
    "NKD": ("jp225", "NKD", False),
    "6E": ("eurusd", "6E", False),
    "6B": ("gbpusd", "6B", False),
    "6A": ("audusd", "6A", False),
    "6J": ("usdjpy", "6J", True),
    "6C": ("usdcad", "6C", True),
    "6S": ("usdchf", "6S", True),
}


# --- data ----------------------------------------------------------------------

@dataclass
class Grid:
    market: str
    contract: str
    dates: np.ndarray            # Lucid trading dates (datetime64[D])
    O: np.ndarray                # (n, K) forward-filled inside the day
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    has: np.ndarray              # (n, K) bar really exists
    pv: float = 0.0
    tick: float = 0.0
    cost_usd: float = 0.0        # round trip per contract: 2 ticks + commission both ways
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_grid(market: str, path: Path | None = None) -> Grid:
    """10-year proxy by default; `path` = real futures bars (data/futures/<root>_15m.parquet)."""
    proxy, contract, inverted = MARKETS[market]
    if path is not None:
        inverted = False                 # futures bars are already quoted the futures way
    df = fh.bars_from_parquet(path or ROOT / "data" / f"{proxy}_15m_long.parquet")
    if inverted:                                   # USDJPY -> 6J (JPY in USD)
        df = pd.DataFrame({"open": 1 / df["open"], "high": 1 / df["low"], "low": 1 / df["high"],
                           "close": 1 / df["close"]}, index=df.index)
    idx = df.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    keep = (mins < 16 * 60 + 45) | (mins >= 18 * 60)
    df, mins = df[keep], mins[keep]
    base = np.asarray(df.index.tz_localize(None).normalize().values.astype("datetime64[D]"))
    lucid = base + (mins >= 1080).astype("timedelta64[D]")
    slot = np.where(mins >= 1080, (mins - 1080) // 15, (mins + 360) // 15)
    wd = (lucid.astype("datetime64[D]").view("int64") - 4) % 7     # 0 = Monday
    ok = wd < 5
    lucid, slot, df = lucid[ok], slot[ok], df[ok]
    dates, row = np.unique(lucid, return_inverse=True)
    n = len(dates)
    arrs = {}
    for col in ("open", "high", "low", "close"):
        a = np.full((n, K), np.nan)
        a[row, slot] = df[col].to_numpy()
        arrs[col] = a
    has = ~np.isnan(arrs["close"])
    C = pd.DataFrame(arrs["close"]).ffill(axis=1).to_numpy()
    prevC = np.concatenate([np.full((n, 1), np.nan), C[:, :-1]], axis=1)
    O = np.where(has, arrs["open"], prevC)
    H = np.where(has, arrs["high"], C)
    L = np.where(has, arrs["low"], C)
    s = futures.spec(contract)
    g = Grid(market, contract, dates, O, H, L, C, has, s["point_value"], s["tick_size"],
             2 * s["tick_size"] * s["point_value"] + 2 * s["commission"])
    # RTH daily frame (09:30-16:00 ET), same definitions as the earlier research
    rth_ok = has[:, RTH_OPEN] & has[:, RTH_LAST]
    dd = pd.DataFrame({"open": O[:, RTH_OPEN], "high": np.nanmax(H[:, RTH_OPEN:RTH_LAST + 1], axis=1),
                       "low": np.nanmin(L[:, RTH_OPEN:RTH_LAST + 1], axis=1), "close": C[:, RTH_LAST]},
                      index=pd.to_datetime(dates))
    dd[~rth_ok] = np.nan
    dd["rth_ok"] = rth_ok
    closes = dd["close"].where(rth_ok)
    dd["prev_close"] = closes.ffill().shift(1)
    dd["sma20"] = closes.rolling(20, min_periods=15).mean().shift(1)
    dd["trend"] = np.sign(dd["prev_close"] - dd["sma20"])
    dd["range"] = dd["high"] - dd["low"]
    dd["atr14"] = dd["range"].rolling(14, min_periods=10).mean().shift(1)
    dd["prev_rth_ret"] = (dd["close"] / dd["open"] - 1).shift(1)          # yesterday 09:30->16:00
    dd["prev_cc_ret"] = closes.pct_change().shift(1)
    g.daily = dd
    return g


# --- legs: one trade per day at most -> bar-by-bar P&L paths ----------------------

@dataclass
class Leg:
    name: str
    contract: str
    dates: np.ndarray
    close: np.ndarray            # (n, K) cumulative $ per 1 contract at each bar close
    worst: np.ndarray            # (n, K) worst point inside each bar
    best: np.ndarray             # (n, K) best point inside each bar
    pnl: np.ndarray              # (n,) final $ per contract (0 = no trade)
    traded: np.ndarray           # (n,) bool
    risk: np.ndarray             # (n,) $ at risk to the stop (nan = no stop)


def build_leg(g: Grid, name: str, side: np.ndarray, a: np.ndarray, b: np.ndarray,
              entry: np.ndarray | None = None, stop: np.ndarray | None = None,
              target: np.ndarray | None = None) -> Leg:
    """side (n,) in {-1,0,1}; entry at the OPEN of slot a (or `entry` price), time
    exit at the CLOSE of slot b; optional stop / target prices (nan = none)."""
    n = len(g.dates)
    kk = np.arange(K)[None, :]
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    e = g.O[np.arange(n), np.clip(a, 0, K - 1)] if entry is None else np.asarray(entry, float)
    side = np.where(np.isnan(e), 0, side).astype(int)
    sd = side[:, None].astype(float)
    stop = np.full(n, np.nan) if stop is None else np.asarray(stop, float)
    target = np.full(n, np.nan) if target is None else np.asarray(target, float)
    live = (kk >= a[:, None]) & (kk <= b[:, None]) & (sd != 0)
    adverse = np.where(sd > 0, g.L, g.H)
    favour = np.where(sd > 0, g.H, g.L)
    with np.errstate(invalid="ignore"):
        hs = live & (sd * (adverse - stop[:, None]) <= 0)
        ht = live & (sd * (favour - target[:, None]) >= 0)
    fs = np.where(hs.any(1), hs.argmax(1), K)
    ft = np.where(ht.any(1), ht.argmax(1), K)
    ex = np.minimum(np.minimum(fs, ft), b)
    by_stop = (fs <= ft) & (fs <= b)
    by_tgt = ~by_stop & (ft <= b)
    exit_px = np.where(by_stop, stop, np.where(by_tgt, target, g.C[np.arange(n), np.clip(b, 0, K - 1)]))
    cost = g.cost_usd
    close = np.where(live, sd * (g.C - e[:, None]) * g.pv - cost, 0.0)
    worst = np.where(live, sd * (adverse - e[:, None]) * g.pv - cost, 0.0)
    best = np.where(live, sd * (favour - e[:, None]) * g.pv - cost, 0.0)
    final = np.where(side != 0, side * (exit_px - e) * g.pv - cost, 0.0)
    at_exit = kk == ex[:, None]
    after = kk > ex[:, None]
    fin = final[:, None]
    close = np.where(at_exit & (sd != 0), fin, close)
    worst = np.where(at_exit & (sd != 0), np.minimum(worst, fin), worst)
    best = np.where(at_exit & (sd != 0), np.where(by_tgt[:, None], fin, np.maximum(best, fin)), best)
    close = np.where(after & (sd != 0), fin, close)
    worst = np.where(after & (sd != 0), fin, worst)
    best = np.where(after & (sd != 0), fin, best)
    close = np.nan_to_num(close)
    worst = np.minimum(np.nan_to_num(worst), close)
    best = np.maximum(np.nan_to_num(best), close)
    risk = np.where(side != 0, np.abs(e - stop) * g.pv, np.nan)
    return Leg(name, g.contract, g.dates, close, worst, best, np.nan_to_num(final), side != 0, risk)


def leg_window(g: Grid, name: str, a: int, b: int, side: int = 1, cond: np.ndarray | None = None,
               stop_atr: float | None = None) -> Leg:
    """Hold from the open of slot a to the close of slot b (both bars must exist)."""
    n = len(g.dates)
    ok = g.has[:, a] & g.has[:, b]
    if cond is not None:
        ok &= np.asarray(cond, bool)
    sides = np.where(ok, side, 0)
    stop = None
    if stop_atr is not None:
        e = g.O[:, a]
        stop = e - side * stop_atr * g.daily["atr14"].to_numpy()
    return build_leg(g, name, sides, np.full(n, a), np.full(n, b), stop=stop)


def leg_orb_trend(g: Grid, or_bars: int = 3, last_signal: int = S(12, 0), target_r: float = 2.0,
                  exit_slot: int = RTH_LAST, use_trend: bool = True) -> Leg:
    """Deployed ORB + trend: range of the first or_bars RTH bars, first bar CLOSE
    outside it decides (against the trend -> no trade), entry next bar open,
    stop at the far side, target_r x risk, out at the 15:45 bar close."""
    n = len(g.dates)
    dd = g.daily
    trend = dd["trend"].to_numpy()
    side = np.zeros(n, int)
    a = np.zeros(n, int)
    stop = np.full(n, np.nan)
    tgt = np.full(n, np.nan)
    e = np.full(n, np.nan)
    r0 = RTH_OPEN
    for i in range(n):
        if not dd["rth_ok"].iloc[i] or (use_trend and np.isnan(trend[i])):
            continue
        orh, orl = g.H[i, r0:r0 + or_bars].max(), g.L[i, r0:r0 + or_bars].min()
        if orh <= orl:
            continue
        for k in range(r0 + or_bars, last_signal):
            c = g.C[i, k]
            sd = 1 if c > orh else (-1 if c < orl else 0)
            if sd == 0:
                continue
            if use_trend and sd != trend[i]:
                break
            en = g.O[i, k + 1]
            st = orl if sd > 0 else orh
            if (sd > 0 and st >= en) or (sd < 0 and st <= en):
                break
            side[i], a[i], e[i], stop[i] = sd, k + 1, en, st
            tgt[i] = en + sd * target_r * abs(en - st) if target_r else np.nan
            break
    return build_leg(g, f"{g.market} ORB-trend", side, a, np.full(n, exit_slot), entry=e, stop=stop,
                     target=tgt)


def month_position(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trading-day number inside the month (1, 2, ...) and from its end (-1 = last)."""
    d = pd.to_datetime(dates)
    key = np.asarray(d.year * 12 + d.month)
    first = pd.Series(key).groupby(key).cumcount().to_numpy() + 1
    last = -(pd.Series(key[::-1]).groupby(key[::-1]).cumcount().to_numpy()[::-1] + 1)
    return first, last


def leg_tom(g: Grid, days_after: tuple[int, ...] = (1, 2, 3), days_before: tuple[int, ...] = (-1,),
            last_slot: int | None = None) -> Leg:
    """Turn of the month, held one Lucid day at a time: long from the first bar of
    each window day (18:00 ET when it exists) to the last flat-safe bar."""
    first, last = month_position(g.dates)
    inwin = np.isin(first, days_after) | np.isin(last, days_before)
    n = len(g.dates)
    b = last_slot if last_slot is not None else (RTH_LAST if g.market in ("ES", "NQ", "YM", "NKD") else S(16, 15))
    a = np.where(g.has.any(1), g.has.argmax(1), K)
    ok = inwin & (a < b) & g.has[:, b]
    return build_leg(g, f"{g.market} TOM", np.where(ok, 1, 0), np.minimum(a, K - 1), np.full(n, b))


# --- evaluation replay -------------------------------------------------------------

@dataclass
class Rules:
    target: float = 3000.0
    mll: float = 2000.0
    dll: float | None = 1200.0        # None = no daily loss limit (LucidPro 25K)
    own_stop: float | None = None     # self-imposed daily stop ($, positive), <= dll
    own_take: float | None = None     # self-imposed daily profit lock ($, positive)
    slip: float = 0.0                 # extra $ lost when a flatten is forced
    max_days: int = 60
    lock: float | None = None         # MLL locks here (relative to start) once hwm - mll reaches it; LucidPro: +100
    mll_on: str = "equity"            # "equity": open P&L counts (strict) | "balance": closed balance only
    dll_on: str = "equity"            # "equity": flatten when open P&L touches it | "balance": realised only
    pass_on: str = "close"            # "close": bar close | "best": limit order at the target price


PLANS = {
    "LucidPro 25K": dict(target=1250.0, mll=1000.0, dll=None),
    "LucidPro 50K": dict(target=3000.0, mll=2000.0, dll=1200.0),
    "LucidPro 100K": dict(target=6000.0, mll=3000.0, dll=1800.0),
    "LucidPro 150K": dict(target=9000.0, mll=4500.0, dll=2700.0),
}
MAX_UNITS = {"LucidPro 25K": 20, "LucidPro 50K": 40, "LucidPro 100K": 60, "LucidPro 150K": 100}


def replay(close: np.ndarray, worst: np.ndarray, rules: Rules, starts: np.ndarray,
           best: np.ndarray | None = None) -> pd.DataFrame:
    """Rolling-start replay of the evaluation. close/worst/best: (n, K) combined $ paths.

    With mll_on="balance" / dll_on="balance" open-position losses are ignored
    until positions close (every leg is flat by 16:45 ET, so the end of day is
    the latest check) -- the LENIENT reading of Lucid's wording, unverified."""
    n = close.shape[0]
    Sn = len(starts)
    bal = np.zeros(Sn)
    hwm = np.zeros(Sn)
    status = np.zeros(Sn, int)            # 0 open, 1 pass, 2 fail
    days = np.zeros(Sn, int)
    lim = [x for x in (rules.dll, rules.own_stop) if x is not None]
    stop_lvl = min(lim) if lim else None
    if rules.dll_on == "balance" and rules.own_stop is None:
        stop_lvl = None                   # a realised-only DLL never flattens a single daily trade
    elif rules.dll_on == "balance":
        stop_lvl = rules.own_stop
    for step in range(rules.max_days):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W = close[d[ia]], worst[d[ia]]
        B = best[d[ia]] if (best is not None and rules.pass_on == "best") else C
        b0, fl = bal[ia], hwm[ia] - rules.mll
        if rules.lock is not None:
            fl = np.minimum(fl, rules.lock)
        days[ia] += 1

        def first(mask):
            return np.where(mask.any(1), mask.argmax(1), K)

        if rules.mll_on == "equity":
            f_mll = first(b0[:, None] + W <= fl[:, None])
        else:
            f_mll = np.full(len(ia), K)
        f_stop = first(W <= -stop_lvl) if stop_lvl is not None else np.full(len(ia), K)
        f_pass = first(b0[:, None] + B >= rules.target)
        f_take = first(C >= rules.own_take) if rules.own_take else np.full(len(ia), K)
        # adverse first inside a bar: a stop-out / DLL at the same bar beats the pass
        stop_pnl = -(stop_lvl or 0.0) - rules.slip
        ev_stop = (f_stop <= f_mll) & (f_stop <= f_pass) & (f_stop < K)
        # the flatten itself can still breach the MLL if the floor is above it
        ev_stop_fail = ev_stop & (b0 + stop_pnl <= fl)
        ev_mll = (f_mll < f_stop) & (f_mll <= f_pass) & (f_mll < K) | ev_stop_fail
        ev_pass = ~ev_mll & ~ev_stop & (f_pass < K)
        ev_take = ~ev_mll & ~ev_stop & ~ev_pass & (f_take < K)
        end = C[:, -1]
        day_pnl = np.where(ev_stop, stop_pnl, np.where(ev_take, C[np.arange(len(ia)), np.minimum(f_take, K - 1)], end))
        new_bal = b0 + day_pnl
        st = np.where(ev_mll, 2, np.where(ev_pass | (new_bal >= rules.target), 1, 0))
        st = np.where((st == 0) & (new_bal <= fl), 2, st)
        status[ia] = st
        bal[ia] = new_bal
        hwm[ia] = np.maximum(hwm[ia], new_bal)
    return pd.DataFrame({"start": starts, "status": status, "days": days, "bal": bal})


@dataclass
class FundedRules:
    """LucidPro 50K FUNDED account (support.lucidtrading.com + owner's screenshot,
    2026-09-30; equity-based MLL confirmed by Lucid support). All $ relative to
    the $50,000 start."""
    mll: float = 2000.0               # open equity counts; trails the best END-OF-DAY balance ...
    lock: float = 100.0               # ... and locks at +$100 ($50,100) once that reaches +$2,100
    keep: float = 2100.0              # balance that must remain after a payout ($52,100)
    caps: tuple = (2000.0, 2500.0, 2500.0, 2500.0, 2500.0)   # payout 1, then 2-5 (then live review)
    want: tuple = (2000.0, 2500.0, 2500.0, 2500.0, 2500.0)   # request only once this much is available
    min_cycle_profit: float = 500.0
    consistency: float = 0.40         # largest winning day <= 40% of the cycle's profit
    min_cycle_days: int = 3           # >= 3 calendar days between requests (trading days: stricter)
    max_days: int = 120


def replay_funded(close: np.ndarray, worst: np.ndarray, best: np.ndarray, rules: FundedRules,
                  starts: np.ndarray, day_tp: float | None = None, day_stop: float | None = None,
                  slip: float = 0.0, adaptive: bool = False, tp_floor: float = 150.0,
                  pause_when_met: bool = False, lock_at: float | None = None,
                  lock_keep: float = 0.2) -> pd.DataFrame:
    """Rolling-start replay of a funded account: one row per start day with the
    payouts taken, when the first one came, and whether / when the MLL ended it.
    day_tp: the race's take-profit for the day (limit at that $ gain; None = the
    leg's own exits). day_stop: an own daily stop in $ (None = none).
    adaptive: the day's take-profit is only what the next payout still needs
    (money above the $52,100 keep, the 40% rule, the $500 cycle minimum), between
    tp_floor and day_tp. pause_when_met: no trade once the money side of the
    next payout is met -- just wait out the 3-day minimum.
    lock_at / lock_keep: once the day is up lock_at x its take-profit, a stop at
    lock_keep x the take-profit protects a winning day (from the next bar)."""
    n = close.shape[0]
    m = len(starts)
    bal = np.zeros(m)
    hwm = np.zeros(m)
    status = np.zeros(m, int)                   # 0 alive, 2 blown, 3 five payouts done
    days = np.zeros(m, int)
    paid = np.zeros(m)
    npay = np.zeros(m, int)
    first_day = np.full(m, -1)
    blown_day = np.full(m, -1)
    cyc_days = np.zeros(m, int)
    cyc_pnl = np.zeros(m)
    cyc_max = np.zeros(m)
    caps = np.array(rules.caps)
    want = np.array(rules.want)
    for step in range(rules.max_days):
        d = starts + step
        act = (status == 0) & (d < n)
        if not act.any():
            break
        ia = np.flatnonzero(act)
        C, W, B = close[d[ia]], worst[d[ia]], best[d[ia]]
        b0 = bal[ia]
        floor = np.minimum(hwm[ia] - rules.mll, rules.lock)
        days[ia] += 1
        kk = np.minimum(npay[ia], len(caps) - 1)
        need = np.maximum.reduce([rules.keep + want[kk] - b0,
                                  np.where(cyc_max[ia] > 0, cyc_max[ia] / rules.consistency - cyc_pnl[ia], 0.0),
                                  rules.min_cycle_profit - cyc_pnl[ia]])
        trade = ~(pause_when_met & (need <= 0)) if pause_when_met else np.ones(len(ia), bool)
        if day_tp and adaptive:
            tp = np.clip(need, tp_floor, day_tp)
        else:
            tp = np.full(len(ia), day_tp if day_tp else np.inf)

        def first(mask):
            return np.where(mask.any(1), mask.argmax(1), K)

        f_mll = first((b0[:, None] + W <= floor[:, None]) & trade[:, None])
        f_stop = first((W <= -day_stop) & trade[:, None]) if day_stop else np.full(len(ia), K)
        f_tp = first((B >= tp[:, None]) & trade[:, None]) if day_tp else np.full(len(ia), K)
        if lock_at is not None and day_tp:
            f_trig = first((B >= (lock_at * tp)[:, None]) & trade[:, None])
            later = np.arange(K)[None, :] > f_trig[:, None]
            f_lock = first((W <= (lock_keep * tp)[:, None]) & later & trade[:, None])
        else:
            f_lock = np.full(len(ia), K)
        ev_mll = (f_mll <= f_stop) & (f_mll <= f_tp) & (f_mll <= f_lock) & (f_mll < K)
        ev_stop = ~ev_mll & (f_stop <= f_tp) & (f_stop <= f_lock) & (f_stop < K)
        ev_lock = ~ev_mll & ~ev_stop & (f_lock < f_tp) & (f_lock < K)
        ev_tp = ~ev_mll & ~ev_stop & ~ev_lock & (f_tp < K)
        pnl = np.where(ev_mll, floor - b0, np.where(ev_stop, -day_stop - slip if day_stop else 0.0,
                                                    np.where(ev_lock, lock_keep * np.where(np.isfinite(tp), tp, 0.0),
                                                             np.where(ev_tp, np.where(np.isfinite(tp), tp, 0.0),
                                                                      C[:, -1]))))
        pnl = np.where(trade, pnl, 0.0)
        nb = b0 + pnl
        blown = ev_mll | (nb <= floor)
        status[ia[blown]] = 2
        blown_day[ia[blown]] = days[ia[blown]]
        bal[ia] = nb
        hwm[ia] = np.maximum(hwm[ia], nb)
        cyc_days[ia] += 1
        cyc_pnl[ia] += pnl
        cyc_max[ia] = np.maximum(cyc_max[ia], pnl)
        # payout request at the end of the day
        k = np.minimum(npay[ia], len(caps) - 1)
        avail = bal[ia] - rules.keep
        ok = ((status[ia] == 0) & (cyc_days[ia] >= rules.min_cycle_days) & (cyc_pnl[ia] >= rules.min_cycle_profit)
              & (cyc_max[ia] <= rules.consistency * cyc_pnl[ia]) & (avail >= want[k]))
        if ok.any():
            j = ia[ok]
            amt = np.minimum(caps[k[ok]], avail[ok])
            bal[j] -= amt
            paid[j] += amt
            first_day[j] = np.where(first_day[j] < 0, days[j], first_day[j])
            npay[j] += 1
            cyc_days[j], cyc_pnl[j], cyc_max[j] = 0, 0.0, 0.0
            status[j[npay[j] >= len(caps)]] = 3
    return pd.DataFrame({"start": starts, "status": status, "days": days, "paid": paid, "payouts": npay,
                         "first_day": first_day, "blown_day": blown_day, "bal": bal})


def summarise_funded(res: pd.DataFrame, split: float = 0.9) -> dict:
    got = res.payouts >= 1
    return {
        "first_payout": got.mean(),
        "blown_before_payout": ((res.status == 2) & ~got).mean(),
        "median_days_to_payout": float(res.first_day[got].median()) if got.any() else np.nan,
        "avg_paid_net": (res.paid * split).mean(),
        "avg_payouts": res.payouts.mean(),
        "avg_days_alive": res.days.mean(),
        "net_per_week_alive": (res.paid * split).sum() / max(res.days.sum() / 5.0, 1e-9),
        "five_payouts": (res.status == 3).mean(),
    }


def summarise(res: pd.DataFrame) -> dict:
    p = (res.status == 1).mean()
    f = (res.status == 2).mean()
    o = (res.status == 0).mean()
    passed = res[res.status == 1]
    resolved = p + f
    pr = p / resolved if resolved else np.nan
    return {"pass": p, "fail": f, "open": o, "pass_resolved": pr,
            "fails_per_pass": (1 - pr) / pr if pr and pr > 0 else np.inf,
            "pass_day1": ((res.status == 1) & (res.days == 1)).mean(),
            "pass_5d": ((res.status == 1) & (res.days <= 5)).mean(),
            "med_days": float(passed.days.median()) if len(passed) else np.nan}


# --- helpers ---------------------------------------------------------------------

def align(legs: list[Leg]) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
    """Common calendar (union of dates); a leg is flat on days it has no data.
    Each entry: (close, worst, pnl, best)."""
    dates = np.unique(np.concatenate([lg.dates for lg in legs]))
    out = []
    for lg in legs:
        pos = np.searchsorted(dates, lg.dates)
        c = np.zeros((len(dates), K))
        w = np.zeros((len(dates), K))
        b = np.zeros((len(dates), K))
        p = np.zeros(len(dates))
        c[pos], w[pos], p[pos], b[pos] = lg.close, lg.worst, lg.pnl, lg.best
        out.append((c, w, p, b))
    return dates, out


def combine(aligned, weights, with_best: bool = False):
    c = sum(wt * a[0] for wt, a in zip(weights, aligned) if wt)
    w = sum(wt * a[1] for wt, a in zip(weights, aligned) if wt)
    if with_best:
        b = sum(wt * a[3] for wt, a in zip(weights, aligned) if wt)
        return c, w, b
    return c, w


def leg_stats(lg: Leg) -> dict:
    t = lg.traded
    r = lg.pnl[t]
    d = pd.to_datetime(lg.dates[t])
    if len(r) < 10:
        return {"leg": lg.name, "n": int(len(r))}
    yrs = pd.Series(r).groupby(d.year).sum()
    half = len(r) // 2
    return {"leg": lg.name, "n": int(len(r)), "per_yr": len(r) / 10.6, "usd": r.mean(),
            "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), "win": (r > 0).mean(),
            "h1": r[:half].mean(), "h2": r[half:].mean(), "yrs_pos": (yrs > 0).mean(),
            "worst_intraday": float(lg.worst[t].min(1).mean())}


if __name__ == "__main__":
    print("engine module -- run scripts/lucid_pass_study.py")
