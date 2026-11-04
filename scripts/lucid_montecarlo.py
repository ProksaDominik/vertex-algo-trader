"""Monte Carlo of LucidPro 50K rules for a strategy stack.

Takes the TRADES of a stack (data/futures/hypotheses_trades.parquet), turns
them into daily account P&L at a given size, and replays that history in
random 5-day blocks (keeping the week-scale clustering of good and bad days)
thousands of times through Lucid's rules:

EVALUATION  start $50,000; PASS at an end-of-day balance of $53,000; FAIL when
            the day's worst point touches the Max Loss Limit -- $2,000 under
            the best end-of-day balance, trailing (no lock assumed in the
            evaluation: the stricter reading).
FUNDED      same MLL, but it LOCKS at $50,100 once an end-of-day balance
            reaches $52,100; a run "pays" at an end-of-day balance of $52,600
            (a $500 withdrawal above Lucid's $52,100 buffer).
BOTH        the $1,200 daily loss limit is soft: it only stops NEW trades. A
            day's loss is NOT capped -- an open trade's stop can lose far more
            than $1,200 in one fill (a wide-range MNQ stop at 6 contracts can
            exceed $2,000). Capping the day at -$1,300, as this file did at
            first, hid exactly the worst days and flattered big fixed sizes
            (fixed 6 MNQ + 3 MGC showed $339/week where the trades add up to
            ~$160) -- corrected 2026-09-30.

The day's worst point is taken as start-of-day equity plus ALL of that day's
losing trades, as if every loser came before every winner -- a deliberately
pessimistic reading of intraday drawdown, since only trade results (not
intra-trade paths) are known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR = 50_000.0
MLL = 2_000.0
EVAL_TARGET = 53_000.0
LOCK_TRIGGER, LOCK_FLOOR = 52_100.0, 50_100.0
PAYOUT_LEVEL = 52_600.0
DLL_CAP = None              # no cap: the soft DLL cannot truncate a stop already in the market


def daily_series(trades: pd.DataFrame, contracts: dict[str, int], calendar: pd.DatetimeIndex
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(day P&L, day worst point relative to the open) on every calendar day."""
    t = trades.copy()
    t["usd_sized"] = t["usd"] * t["leg"].map(contracts)
    t["loss"] = t["usd_sized"].clip(upper=0.0)
    g = t.groupby(pd.to_datetime(t["date"]))
    pnl = g["usd_sized"].sum().reindex(calendar, fill_value=0.0).to_numpy()
    low = g["loss"].sum().reindex(calendar, fill_value=0.0).to_numpy()
    if DLL_CAP is not None:
        pnl = np.maximum(pnl, DLL_CAP)
        low = np.maximum(low, DLL_CAP)
    return pnl, low


def _paths(pnl: np.ndarray, low: np.ndarray, n_sims: int, horizon: int, block: int, rng) -> tuple:
    n = len(pnl)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, n - block, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    return pnl[idx], low[idx]


def simulate(pnl: np.ndarray, low: np.ndarray, mode: str, n_sims: int = 20_000, horizon: int = 120,
             block: int = 5, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    P, L = _paths(pnl, low, n_sims, horizon, block, rng)
    equity = np.full(n_sims, ANCHOR)
    peak = np.full(n_sims, ANCHOR)
    locked = np.zeros(n_sims, dtype=bool)
    done = np.zeros(n_sims, dtype=bool)
    outcome = np.zeros(n_sims, dtype=np.int8)           # 1 success, -1 fail, 0 still running
    day_done = np.full(n_sims, horizon, dtype=np.int32)
    for d in range(horizon):
        live = ~done
        floor = np.where(locked, LOCK_FLOOR, peak - MLL)
        worst = equity + L[:, d]
        fail = live & (worst <= floor)
        equity = np.where(live & ~fail, equity + P[:, d], equity)
        fail |= live & (equity <= floor)
        target = EVAL_TARGET if mode == "eval" else PAYOUT_LEVEL
        win = live & ~fail & (equity >= target)
        outcome[fail] = -1
        outcome[win] = 1
        day_done[fail | win] = d + 1
        done |= fail | win
        peak = np.where(done, peak, np.maximum(peak, equity))
        if mode == "funded":
            locked |= peak >= LOCK_TRIGGER
        if done.all():
            break
    succ = outcome == 1
    return {"p_success": succ.mean(), "p_fail": (outcome == -1).mean(), "p_open": (outcome == 0).mean(),
            "median_days": float(np.median(day_done[succ])) if succ.any() else np.nan,
            "p75_days": float(np.percentile(day_done[succ], 75)) if succ.any() else np.nan}
