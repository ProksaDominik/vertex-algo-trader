"""Does TIGHT-TARGET intraday mean reversion give a high-win-rate, fast, shallow
strategy good enough for a 10-day FTMO challenge?

WHY THIS EXISTS
---------------
The FTMO profile we need is NOT the profile the daily book has. Daily RSI/Bollinger
at 3R-6R win ~30% of the time and hold for weeks; that cannot produce +10% inside
10 trading days without a tail-risk bet. What the challenge actually wants is:

    win rate  > 65%      (so equity grinds up, not lurches)
    median hold < 5 days (so 10 days is enough bars for the edge to express)
    no -5% day           (FTMO's hard daily limit)

The textbook way to buy a high win rate is a TIGHT take-profit: at TP = 0.5R you
win far more often than you lose. The counter-argument already established in this
repo is that transaction cost is a FIXED price distance, so shrinking R makes cost
a BIGGER fraction of every trade (measured: stop exits realise -1.054R, not -1.000R).
Whether the win-rate gain beats the cost drag is an empirical question, not an
assumption -- so this script measures it directly.

WHAT IS TESTED
--------------
Timeframes 15m / 30m / 1h (the intraday band, where nothing has yet been tested at
sub-1R targets), 20-symbol basket, 2016-2026, targets 0.3 / 0.5 / 0.75 / 1.0 / 1.5 R:

  Bollinger      BB(20,2) band rejection            -- canonical class, tight TP is new
  RSI            RSI(14) 30/70 cross-back           -- canonical class, tight TP is new
  BB_RSI         BB rejection AND RSI confluence    -- new "scalp" confluence variant
  VWAP           session-VWAP stretch fade          -- existing class, never swept
  ORFade_move    fade the London-open 07:00 thrust  -- new
  ORFade_break   fade a false break of the 07:00 opening range -- new

VALIDATION GATE
---------------
Each symbol's history is split 70% train / 30% OOS chronologically. A config only
counts as a survivor if expectancy is POSITIVE IN BOTH HALVES. Train and OOS are
printed side by side for everything, including the failures. Sunday bars
(index.dayofweek == 6, the ~2h Sunday-evening reopen stub) are dropped.

HOW IT RUNS FAST ENOUGH
-----------------------
Two optimisations, both exactness-preserving and both verified by `--validate`:

1. Signals are generated VECTORISED over the whole series once per
   (strategy, symbol, timeframe) and reused across all five targets. Entry price,
   direction and stop are independent of the take-profit, so this changes nothing.
2. Fills are simulated by `simulate()` -- a scalar re-implementation of
   metatraderbot.backtest.Backtest restricted to the case these strategies use
   (no trailing stop, no skyrocket exit, daily stop disabled). `--validate`
   asserts it reproduces the real engine trade-for-trade against the canonical
   strategy classes. This is ~50x faster than driving the pandas engine, which
   is what makes a 3,600-backtest grid finish in minutes instead of hours.

Cost model, sizing, max_positions and SL-before-TP tie-breaking are all the
project's existing ones (instruments.PAIR_SPECS via the engine's formulas).

Usage:
    python scripts/intraday_tight_target_sweep.py --validate     # prove the fast path
    python scripts/intraday_tight_target_sweep.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import BASKET, RISK_USD, load  # noqa: E402
from metatraderbot.instruments import get_specs  # noqa: E402

TIMEFRAMES = ["15m", "30m", "1h"]
TP_RS = [0.3, 0.5, 0.75, 1.0, 1.5]
OOS_FRAC = 0.30
MAX_POSITIONS = 2

# Hold-time histogram, in DAYS. Fine bins where the mass is expected to be
# (intraday to a few days), coarse in the tail. Pooling histograms instead of
# raw trade lists is what keeps the cross-process payload small; median
# resolution is 7.2 minutes below 3 days.
HOLD_BINS = np.concatenate([
    np.arange(0.0, 3.0, 0.005),
    np.arange(3.0, 15.0, 0.05),
    np.arange(15.0, 400.0, 1.0),
    [1e9],
])


# --------------------------------------------------------------------------
# Indicators (full-series equivalents of the per-window versions the canonical
# strategy classes compute inside the 400-bar Windowed wrapper)
# --------------------------------------------------------------------------
def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().to_numpy()


def _rsi(closes: pd.Series, period: int = 14) -> np.ndarray:
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).to_numpy()


def bars_for(sym: str, tf: str) -> pd.DataFrame | None:
    """Load `tf` bars, Sundays removed.

    `load()` passes the timeframe straight to DataFrame.resample, and pandas 3
    rejects a lowercase "m" offset ("'m' is no longer supported ... use 'ME'"),
    so the intraday labels used everywhere else in this repo have to be
    translated to minute aliases here rather than in the shared loader.
    """
    freq = {"15m": "15min", "30m": "30min"}.get(tf, tf)
    b = load(sym, freq)
    if b is None:
        return None
    return b[b.index.dayofweek != 6]      # ~2h Sunday-evening reopen stub


def _day_groups(index: pd.DatetimeIndex) -> list[np.ndarray]:
    """Positional indices of the bars belonging to each UTC calendar day."""
    day = index.normalize().asi8
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], len(day)]
    return [np.arange(s, e) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------
# Vectorised signal generators.
# Each returns (dir, entry, sl) arrays aligned to `df`. dir is 0/+1/-1.
# --------------------------------------------------------------------------
Signals = tuple[np.ndarray, np.ndarray, np.ndarray]


def _blank(n: int) -> Signals:
    return (np.zeros(n, np.int8), np.full(n, np.nan), np.full(n, np.nan))


def _warmup(sig: Signals, k: int) -> Signals:
    """Suppress the first `k` bars.

    The canonical classes each open with a `if bar_index < k: return []` guard.
    Vectorised indicators have no such guard, so without this the fast path
    fires a handful of extra trades at the very start of history that the real
    engine never takes. k must match the class it mirrors or `--validate` fails.
    """
    d, e, s = sig
    d[:k] = 0
    e[:k] = np.nan
    s[:k] = np.nan
    return d, e, s


def sig_bollinger(df: pd.DataFrame, rsi_confluence: bool = False) -> Signals:
    """BollingerBounceStrategy, vectorised. period=20, std=2.0, buffer=20% of width."""
    n = len(df)
    d, ent, sl = _blank(n)
    close = df["close"]
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = (sma + 2.0 * std).to_numpy()
    lower = (sma - 2.0 * std).to_numpy()
    buf = (upper - lower) * 0.20
    c = close.to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()

    long_c = (l <= lower) & (c > lower)
    short_c = (h >= upper) & (c < upper)
    if rsi_confluence:
        r = _rsi(close)
        long_c &= r < 35.0
        short_c &= r > 65.0
    # The class uses if/elif, so a bar that satisfies both resolves LONG.
    short_c &= ~long_c
    ok = np.isfinite(upper) & np.isfinite(lower) & (std.to_numpy() > 0)
    long_c &= ok
    short_c &= ok

    d[long_c] = 1
    ent[long_c] = c[long_c]
    sl[long_c] = l[long_c] - buf[long_c]
    d[short_c] = -1
    ent[short_c] = c[short_c]
    sl[short_c] = h[short_c] + buf[short_c]
    return _warmup((d, ent, sl), 22)          # class guard: bar_index < period + 2


def sig_bb_rsi(df: pd.DataFrame) -> Signals:
    return sig_bollinger(df, rsi_confluence=True)


def sig_rsi(df: pd.DataFrame) -> Signals:
    """RSIReversionStrategy, vectorised. period=14, 30/70, swing stop over 10 bars."""
    n = len(df)
    d, ent, sl = _blank(n)
    r = _rsi(df["close"])
    r_prev = np.r_[np.nan, r[:-1]]
    c = df["close"].to_numpy()
    lo10 = df["low"].rolling(10).min().to_numpy()
    hi10 = df["high"].rolling(10).max().to_numpy()

    long_c = (r_prev <= 30.0) & (r > 30.0)
    short_c = (r_prev >= 70.0) & (r < 70.0) & ~long_c

    d[long_c] = 1
    ent[long_c] = c[long_c]
    sl[long_c] = lo10[long_c]
    d[short_c] = -1
    ent[short_c] = c[short_c]
    sl[short_c] = hi10[short_c]
    return _warmup((d, ent, sl), 19)          # class guard: bar_index < period + 5


def sig_vwap(df: pd.DataFrame) -> Signals:
    """VWAPReversionStrategy, vectorised. band=2.0*ATR14, stop buffer 0.25*ATR,
    rejection candle required. VWAP is anchored to the start of each UTC day."""
    n = len(df)
    d, ent, sl = _blank(n)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    day = pd.Series(df.index.normalize().asi8, index=df.index)
    cum_pv = (typical * vol).groupby(day).cumsum().to_numpy()
    cum_v = vol.groupby(day).cumsum().to_numpy()
    bar_of_day = vol.groupby(day).cumcount().to_numpy()  # 0-based

    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cum_v > 0, cum_pv / cum_v, c)
    a = _atr(df, 14)
    ok = np.isfinite(a) & (a > 0) & (bar_of_day >= 2)   # class needs >=3 same-day bars
    band = 2.0 * a
    buf = 0.25 * a
    dev = c - vwap
    bull = c > o

    long_c = ok & (dev < -band) & bull
    short_c = ok & (dev > band) & ~bull & ~long_c

    d[long_c] = 1
    ent[long_c] = c[long_c]
    sl[long_c] = l[long_c] - buf[long_c]
    d[short_c] = -1
    ent[short_c] = c[short_c]
    sl[short_c] = h[short_c] + buf[short_c]
    return _warmup((d, ent, sl), 20)          # class guard: bar_index < 20


def sig_orfade_move(df: pd.DataFrame, start_hour: int = 7, or_minutes: int = 60,
                    min_atr: float = 0.5, buf_atr: float = 0.10) -> Signals:
    """NEW: fade the first move after the London open.

    Build the 07:00-08:00 UTC opening range. On the first bar after it closes,
    if the range travelled more than 0.5*ATR(14) in one direction, take the
    OTHER side -- betting the open thrust is an over-reaction that retraces.
    Stop beyond the opening range's extreme. One trade per day, maximum.
    """
    n = len(df)
    d, ent, sl = _blank(n)
    a = _atr(df, 14)
    idx = df.index
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    start, end = start_hour * 60, start_hour * 60 + or_minutes

    for pos in _day_groups(idx):
        m = minutes[pos]
        orp = pos[(m >= start) & (m < end)]
        if len(orp) == 0:
            continue
        aft = pos[m >= end]
        if len(aft) == 0:
            continue
        i = aft[0]
        av = a[i]
        if not np.isfinite(av) or av <= 0:
            continue
        move = c[orp[-1]] - o[orp[0]]
        if move > min_atr * av:
            d[i] = -1
            ent[i] = c[i]
            sl[i] = max(h[orp].max(), h[i]) + buf_atr * av
        elif move < -min_atr * av:
            d[i] = 1
            ent[i] = c[i]
            sl[i] = min(l[orp].min(), l[i]) - buf_atr * av
    return d, ent, sl


def sig_orfade_break(df: pd.DataFrame, start_hour: int = 7, or_minutes: int = 60,
                     window_hours: float = 6.0, buf_atr: float = 0.10) -> Signals:
    """NEW: fade a FALSE break of the London opening range.

    Same 07:00-08:00 UTC range, but wait for price to poke through it and close
    back INSIDE within the next 6 hours -- the liquidity-grab shape, anchored to
    the session open rather than to the Asian range. First occurrence only.
    """
    n = len(df)
    d, ent, sl = _blank(n)
    a = _atr(df, 14)
    idx = df.index
    minutes = (idx.hour * 60 + idx.minute).to_numpy()
    h, l, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    start, end = start_hour * 60, start_hour * 60 + or_minutes
    stop_min = end + window_hours * 60

    for pos in _day_groups(idx):
        m = minutes[pos]
        orp = pos[(m >= start) & (m < end)]
        if len(orp) == 0:
            continue
        or_hi, or_lo = h[orp].max(), l[orp].min()
        for i in pos[(m >= end) & (m < stop_min)]:
            av = a[i]
            if not np.isfinite(av) or av <= 0:
                continue
            if l[i] < or_lo and c[i] > or_lo:
                d[i] = 1
                ent[i] = c[i]
                sl[i] = l[i] - buf_atr * av
                break
            if h[i] > or_hi and c[i] < or_hi:
                d[i] = -1
                ent[i] = c[i]
                sl[i] = h[i] + buf_atr * av
                break
    return d, ent, sl


SIGGENS = {
    "Bollinger": sig_bollinger,
    "RSI": sig_rsi,
    "BB_RSI": sig_bb_rsi,
    "VWAP": sig_vwap,
    "ORFade_move": sig_orfade_move,
    "ORFade_break": sig_orfade_break,
}


# --------------------------------------------------------------------------
# Fill simulator -- exact scalar equivalent of metatraderbot.backtest.Backtest
# for the no-trail / no-skyrocket / daily-stop-disabled case.
# --------------------------------------------------------------------------
def simulate(df: pd.DataFrame, sig: Signals, tp_r: float, mult: float,
             spread: float, comm: float, risk: float = RISK_USD,
             max_pos: int = MAX_POSITIONS) -> dict:
    d_arr, e_arr, s_arr = sig
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    # Unit-agnostic on purpose: pandas 3 hands back datetime64[ms] here, not
    # [ns], so `.asi8 / 86400e9` silently reports hold times 1e6x too small.
    t_day = ((df.index - df.index[0]) / pd.Timedelta(days=1)).to_numpy()
    day_key = df.index.normalize().values.astype("datetime64[D]").astype("int64")
    n = len(c)

    rs: list[float] = []
    holds: list[float] = []
    reasons: list[str] = []
    exit_days: list[int] = []
    open_pos: list[tuple] = []

    for i in range(n):
        if open_pos:
            hi, lo = h[i], l[i]
            still = []
            for p in open_pos:
                pd_, entry, stop, tp, lots, ie = p
                if pd_ > 0:
                    hit_sl, hit_tp = lo <= stop, hi >= tp
                else:
                    hit_sl, hit_tp = hi >= stop, lo <= tp
                if hit_sl:                      # engine assumes SL first on a tie
                    px, why = stop, "sl"
                elif hit_tp:
                    px, why = tp, "tp"
                else:
                    still.append(p)
                    continue
                pnl = (px - entry) * pd_ * lots * mult - (comm * lots * 2 + spread * lots * mult)
                rs.append(pnl / risk)
                holds.append(t_day[i] - t_day[ie])
                reasons.append(why)
                exit_days.append(day_key[i])
            open_pos = still

        sd = d_arr[i]
        if sd and len(open_pos) < max_pos:
            entry, stop = e_arr[i], s_arr[i]
            dist = (entry - stop) if sd > 0 else (stop - entry)
            if dist > 0:
                lots = risk / (dist * mult)
                if lots > 0:
                    open_pos.append((int(sd), entry, stop,
                                     entry + sd * dist * tp_r, lots, i))

    for p in open_pos:                                   # engine force-closes
        pd_, entry, stop, tp, lots, ie = p
        pnl = (c[n - 1] - entry) * pd_ * lots * mult - (comm * lots * 2 + spread * lots * mult)
        rs.append(pnl / risk)
        holds.append(t_day[n - 1] - t_day[ie])
        reasons.append("end-of-data")
        exit_days.append(day_key[n - 1])

    return {"rs": np.asarray(rs), "holds": np.asarray(holds),
            "reasons": np.asarray(reasons), "exit_days": np.asarray(exit_days, np.int64)}


def _pack(res: dict, n_days: int) -> dict:
    """Compress a simulate() result into a poolable summary (see HOLD_BINS)."""
    rs = res["rs"]
    if len(rs) == 0:
        return {"n": 0, "days": n_days}
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    sl_mask = res["reasons"] == "sl"
    hist, _ = np.histogram(res["holds"], bins=HOLD_BINS)
    daily = {}
    for day, r in zip(res["exit_days"], rs):
        daily[int(day)] = daily.get(int(day), 0.0) + float(r)
    return {
        "n": int(len(rs)), "days": n_days,
        "n_win": int(len(wins)),
        "sum_r": float(rs.sum()), "sum_win": float(wins.sum()),
        "sum_loss": float(losses.sum()), "sum_sq": float((rs ** 2).sum()),
        "n_sl": int(sl_mask.sum()), "sum_sl_r": float(rs[sl_mask].sum()),
        "n_tp": int((res["reasons"] == "tp").sum()),
        "hold_hist": hist.tolist(),
        "daily": daily,
    }


# --------------------------------------------------------------------------
# Worker: one (strategy, symbol, timeframe) cell -> all targets x both halves
# --------------------------------------------------------------------------
_ZERO_COST = False       # rebound in each worker by _init_worker


def _init_worker(zero_cost: bool) -> None:
    global _ZERO_COST
    _ZERO_COST = zero_cost


def _cell(job: tuple[str, str, str]) -> dict | None:
    name, sym, tf = job
    try:
        bars = bars_for(sym, tf)
        if bars is None or len(bars) < 2000:
            return None
        sig_full = SIGGENS[name](bars)
        specs = get_specs(sym)
        mult, spread, comm = (specs["contract_mult"], specs["spread_price"],
                              specs["commission_per_lot"])
        if _ZERO_COST:
            # Diagnostic mode: same signals, frictionless fills. Separates
            # "this pattern has no edge" from "the edge is real but smaller
            # than the spread" -- only the latter is worth chasing further.
            spread, comm = 0.0, 0.0
        split = int(len(bars) * (1 - OOS_FRAC))
        out: dict = {"name": name, "sym": sym, "tf": tf, "cells": {}}
        for half, sl_ in (("train", slice(0, split)), ("oos", slice(split, len(bars)))):
            b = bars.iloc[sl_]
            if len(b) < 1000:
                continue
            sub_sig = tuple(a[sl_] for a in sig_full)
            n_days = len(np.unique(b.index.normalize().asi8))
            for tp in TP_RS:
                res = simulate(b, sub_sig, tp, mult, spread, comm)
                out["cells"][f"{half}|{tp}"] = _pack(res, n_days)
        return out
    except Exception as e:  # noqa: BLE001 - a broken cell must not kill the sweep
        return {"name": name, "sym": sym, "tf": tf, "cells": {}, "error": repr(e)}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def _median_from_hist(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return float("nan")
    cum = np.cumsum(hist)
    k = int(np.searchsorted(cum, total / 2.0))
    k = min(k, len(HOLD_BINS) - 2)
    return float((HOLD_BINS[k] + HOLD_BINS[k + 1]) / 2.0)


def _pct_from_hist(hist: np.ndarray, q: float) -> float:
    total = hist.sum()
    if total == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(hist), total * q))
    k = min(k, len(HOLD_BINS) - 2)
    return float((HOLD_BINS[k] + HOLD_BINS[k + 1]) / 2.0)


def aggregate(packs: list[dict]) -> dict | None:
    packs = [p for p in packs if p and p.get("n", 0) > 0]
    if not packs:
        return None
    n = sum(p["n"] for p in packs)
    n_win = sum(p["n_win"] for p in packs)
    sum_r = sum(p["sum_r"] for p in packs)
    sum_win = sum(p["sum_win"] for p in packs)
    sum_loss = sum(p["sum_loss"] for p in packs)
    n_sl = sum(p["n_sl"] for p in packs)
    sum_sl = sum(p["sum_sl_r"] for p in packs)
    days = max(p["days"] for p in packs)
    hist = np.sum([p["hold_hist"] for p in packs], axis=0)

    # Portfolio view: the basket traded together, chronologically. Worst day in R
    # is the number that maps onto FTMO's 5% daily loss limit.
    daily: dict[int, float] = {}
    for p in packs:
        for d, r in p["daily"].items():
            daily[int(d)] = daily.get(int(d), 0.0) + r
    order = sorted(daily)
    dr = np.array([daily[d] for d in order]) if order else np.array([0.0])
    eq = np.cumsum(dr)
    max_dd = float(np.max(np.maximum.accumulate(np.r_[0.0, eq]) - np.r_[0.0, eq]))

    return {
        "n": n, "per_day": n / days if days else 0.0,
        "win": n_win / n, "e": sum_r / n, "total_r": sum_r,
        "pf": (sum_win / -sum_loss) if sum_loss < 0 else float("inf"),
        "med_hold_d": _median_from_hist(hist),
        "p90_hold_d": _pct_from_hist(hist, 0.90),
        "avg_sl_r": (sum_sl / n_sl) if n_sl else float("nan"),
        "worst_day_r": float(dr.min()), "best_day_r": float(dr.max()),
        "max_dd_r": max_dd, "n_days": len(order),
        "syms_pos": sum(1 for p in packs if p["sum_r"] > 0), "syms": len(packs),
    }


# --------------------------------------------------------------------------
# Validation: fast path must equal the real engine + canonical classes
# --------------------------------------------------------------------------
def validate() -> int:
    from metatraderbot.backtest import Backtest
    from canonical_strategy_test import Windowed
    from metatraderbot.strategies_bollinger import BollingerBounceStrategy
    from metatraderbot.strategies_rsi import RSIReversionStrategy
    from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy

    cases = [
        ("Bollinger", BollingerBounceStrategy, {"bollinger": {}}),
        ("RSI", RSIReversionStrategy, {"rsi": {}}),
        ("VWAP", VWAPReversionStrategy, {"vwap": {}}),
    ]
    fails = 0
    for sym, tf, nbars in (("EURUSD", "1h", 20_000), ("XAUUSD", "30m", 12_000),
                           ("GBPJPY", "15m", 12_000)):
        bars = bars_for(sym, tf).iloc[:nbars]
        specs = get_specs(sym)
        for name, cls, params in cases:
            for tp in (0.5, 1.0):
                strat = {"session": None,
                         "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
                strat.update(params)
                cfg = {"symbol": sym,
                       "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                                   "max_positions": MAX_POSITIONS,
                                   "daily_stop_loss_usd": -1e9},
                       "strategy": strat}
                bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
                bt.run()
                ref = np.array([t.pnl_usd / RISK_USD for t in bt.closed])
                fast = simulate(bars, SIGGENS[name](bars), tp,
                                specs["contract_mult"], specs["spread_price"],
                                specs["commission_per_lot"])["rs"]
                same_n = len(ref) == len(fast)
                dev = float(np.abs(ref - fast).max()) if same_n and len(ref) else float("nan")
                # Tolerance is 1e-6 R = $1e-4 at a $100 risk unit. Residuals at
                # that scale are float64 summation-order noise between the
                # pandas rolling path and the numpy one, not behavioural drift;
                # the trade COUNT must match exactly.
                ok = same_n and (len(ref) == 0 or dev < 1e-6)
                fails += 0 if ok else 1
                print(f"  {'OK ' if ok else 'FAIL'} {sym:<7}{tf:<5}{name:<11}TP{tp:<5}"
                      f"engine n={len(ref):<6} fast n={len(fast):<6} maxdiff={dev:.2e}")
    print("\nvalidation:", "PASS" if fails == 0 else f"{fails} MISMATCHES")
    return 1 if fails else 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tf", nargs="*", default=TIMEFRAMES)
    ap.add_argument("--out", default="backtest/intraday_tight_targets.json")
    args = ap.parse_args()

    if args.validate:
        print("Validating fast simulator against metatraderbot.backtest.Backtest "
              "+ canonical strategy classes\n")
        return validate()

    jobs = [(n, s, tf) for tf in args.tf for n in SIGGENS for s in BASKET]
    print("=" * 118)
    print("INTRADAY TIGHT-TARGET SWEEP — high win rate / fast resolution hunt for the FTMO challenge")
    print(f"{len(SIGGENS)} strategies x {len(BASKET)} symbols x {len(args.tf)} timeframes "
          f"x {len(TP_RS)} targets x 2 halves = {len(jobs) * len(TP_RS) * 2} backtests")
    print("Sundays excluded. train = first 70%, OOS = last 30%, chronological.")
    print("=" * 118, flush=True)

    cells, done = [], 0
    with ProcessPoolExecutor(max_workers=min(args.workers, os.cpu_count() or 4)) as ex:
        for c in ex.map(_cell, jobs, chunksize=1):
            done += 1
            if c:
                cells.append(c)
            if done % 20 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} cells", flush=True)

    for e in [c for c in cells if c.get("error")][:5]:
        print(f"  [ERROR] {e['name']} {e['sym']} {e['tf']}: {e['error']}")

    results = []
    for tf in args.tf:
        print(f"\n{'=' * 118}\n### TIMEFRAME {tf}\n{'=' * 118}")
        print(f"{'strategy':<14}{'TP':>5}{'n_oos':>8}{'/day':>7}{'win%':>7}"
              f"{'E_oos':>9}{'E_train':>9}{'PF':>7}{'medHold':>9}{'p90H':>8}"
              f"{'avgSL':>8}{'wrstDay':>9}{'maxDD':>8}  verdict")
        print("-" * 118)
        for name in SIGGENS:
            for tp in TP_RS:
                mine = [c for c in cells if c["name"] == name and c["tf"] == tf]
                o = aggregate([c["cells"].get(f"oos|{tp}") for c in mine])
                t = aggregate([c["cells"].get(f"train|{tp}") for c in mine])
                if not o or not t:
                    continue
                both = o["e"] > 0 and t["e"] > 0
                verdict = "SURVIVOR" if both else ("oos+ only" if o["e"] > 0 else "")
                pf = f"{o['pf']:.2f}" if np.isfinite(o["pf"]) else "inf"
                print(f"{name:<14}{tp:>5.2f}{o['n']:>8}{o['per_day']:>7.2f}"
                      f"{o['win'] * 100:>7.1f}{o['e']:>+9.3f}{t['e']:>+9.3f}{pf:>7}"
                      f"{o['med_hold_d']:>9.2f}{o['p90_hold_d']:>8.2f}"
                      f"{o['avg_sl_r']:>+8.3f}{o['worst_day_r']:>+9.1f}"
                      f"{o['max_dd_r']:>8.1f}  {verdict}")
                results.append({"tf": tf, "name": name, "tp_r": tp,
                                "oos": o, "train": t, "both_positive": both})
            print()

    keep = [r for r in results if r["both_positive"] and r["oos"]["n"] >= 100]
    print("=" * 118)
    print("SURVIVORS — expectancy positive in BOTH halves, n_oos >= 100")
    print("=" * 118)
    if not keep:
        print("  NONE. Every intraday tight-target config tested is negative in at "
              "least one half of the sample.")
    for r in sorted(keep, key=lambda x: -x["oos"]["e"]):
        o = r["oos"]
        print(f"  {r['name']:<14}{r['tf']:<5}TP {r['tp_r']:<5} "
              f"win {o['win'] * 100:.1f}%  E_oos {o['e']:+.4f}R  "
              f"E_train {r['train']['e']:+.4f}R  medHold {o['med_hold_d']:.2f}d  "
              f"{o['per_day']:.2f}/day")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
