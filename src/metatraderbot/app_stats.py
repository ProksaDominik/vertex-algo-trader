"""Performance statistics for the Vertex app.

Every figure the UI shows is computed here, from one normalised trade frame, so
the same number cannot disagree between two panels.

Two rules this module enforces, both learned from real bugs in this project:

1. **Win rate is over RESOLVED trades only.** The bot's state file counts a
   trade as soon as it OPENS but only records a win/loss when it CLOSES, so
   dividing wins by total trades made two wins and two open positions read as a
   50% win rate instead of 100%. Open positions are reported separately and
   never enter a ratio.
2. **Real and simulated results are kept apart.** `TradeStats.source` marks
   which is which, and `combine()` will not silently merge them -- reconstructed
   history is not money, and a dashboard that blurs the two misleads its owner
   about their own account.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Source = Literal["real", "simulated", "combined"]


@dataclass
class TradeStats:
    source: Source = "real"
    n_closed: int = 0
    n_open: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float | None = None          # over resolved trades only
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    profit_factor: float | None = None
    expectancy: float | None = None        # $ per trade
    expectancy_r: float | None = None      # R per trade
    total_r: float = 0.0
    avg_win: float | None = None
    avg_loss: float | None = None
    largest_win: float | None = None
    largest_loss: float | None = None
    win_loss_ratio: float | None = None
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0                # +n winning, -n losing
    max_drawdown: float = 0.0
    max_drawdown_pct: float | None = None
    peak_equity: float | None = None
    avg_hold_hours: float | None = None
    floating_pnl: float = 0.0
    equity_curve: pd.DataFrame | None = field(default=None, repr=False)
    by_strategy: pd.DataFrame | None = field(default=None, repr=False)
    by_symbol: pd.DataFrame | None = field(default=None, repr=False)
    daily: pd.DataFrame | None = field(default=None, repr=False)


def _streaks(signs: list[int]) -> tuple[int, int, int]:
    """(longest win streak, longest loss streak, current streak signed)."""
    if not signs:
        return 0, 0, 0
    best_w = best_l = run = 0
    last = 0
    for s in signs:
        if s == 0:
            run, last = 0, 0
            continue
        run = run + 1 if s == last else 1
        last = s
        if s > 0:
            best_w = max(best_w, run)
        else:
            best_l = max(best_l, run)
    return best_w, best_l, (run if last > 0 else -run)


def compute(trades: pd.DataFrame | None, starting_equity: float,
            source: Source = "real", n_open: int = 0,
            floating_pnl: float = 0.0) -> TradeStats:
    """Full statistics from a normalised closed-trade frame.

    Expects the columns produced by the dashboard's `_normalize_trades`:
    Closed, P&L, R, Strategy, Symbol, HoldH. Missing columns degrade to None
    rather than raising -- a partial ledger should still render a partial panel.
    """
    st = TradeStats(source=source, n_open=n_open, floating_pnl=floating_pnl,
                    peak_equity=starting_equity)
    if trades is None or trades.empty or "P&L" not in trades.columns:
        return st

    t = trades.dropna(subset=["P&L"]).copy()
    if t.empty:
        return st
    if "Closed" in t.columns:
        t = t.sort_values("Closed")

    pnl = t["P&L"].astype(float)
    st.n_closed = len(t)
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    st.wins, st.losses = len(w), len(l)
    st.scratches = st.n_closed - st.wins - st.losses
    resolved = st.wins + st.losses
    st.win_rate = (st.wins / resolved) if resolved else None

    st.gross_profit = float(w.sum())
    st.gross_loss = float(abs(l.sum()))
    st.net_pnl = float(pnl.sum())
    st.profit_factor = (st.gross_profit / st.gross_loss) if st.gross_loss > 0 else None
    st.expectancy = float(pnl.mean())
    st.avg_win = float(w.mean()) if st.wins else None
    st.avg_loss = float(abs(l.mean())) if st.losses else None
    st.largest_win = float(w.max()) if st.wins else None
    st.largest_loss = float(abs(l.min())) if st.losses else None
    st.win_loss_ratio = (st.avg_win / st.avg_loss) if (st.avg_win and st.avg_loss) else None

    if "R" in t.columns and t["R"].notna().any():
        r = t["R"].astype(float).dropna()
        st.total_r = float(r.sum())
        st.expectancy_r = float(r.mean())
    if "HoldH" in t.columns and t["HoldH"].notna().any():
        st.avg_hold_hours = float(t["HoldH"].astype(float).mean())

    st.max_win_streak, st.max_loss_streak, st.current_streak = _streaks(
        [1 if x > 0 else (-1 if x < 0 else 0) for x in pnl])

    eq = starting_equity + pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    st.max_drawdown = float(dd.min()) if len(dd) else 0.0
    st.peak_equity = float(peak.max()) if len(peak) else starting_equity
    st.max_drawdown_pct = (st.max_drawdown / st.peak_equity * 100) if st.peak_equity else None

    if "Closed" in t.columns:
        st.equity_curve = pd.DataFrame({"ts": t["Closed"].values,
                                        "equity": eq.values,
                                        "pnl": pnl.values})
        d = t.set_index("Closed")["P&L"].astype(float)
        if not d.empty:
            daily = d.resample("D").agg(["sum", "count"])
            daily = daily[daily["count"] > 0].reset_index()
            daily.columns = ["date", "pnl", "trades"]
            st.daily = daily

    for col, attr in (("Strategy", "by_strategy"), ("Symbol", "by_symbol")):
        if col in t.columns:
            g = t.groupby(col)["P&L"].agg(
                trades="size", net="sum",
                wins=lambda s: int((s > 0).sum())).reset_index()
            g["win_rate"] = np.where(g["trades"] > 0, g["wins"] / g["trades"] * 100, np.nan)
            g["avg"] = g["net"] / g["trades"]
            setattr(st, attr, g.sort_values("net", ascending=False))
    return st


def sparkline_points(curve: pd.DataFrame | None, n: int = 60) -> list[float]:
    """Downsampled equity series for an inline sparkline."""
    if curve is None or curve.empty:
        return []
    v = curve["equity"].to_numpy(dtype=float)
    if len(v) <= n:
        return v.tolist()
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    return v[idx].tolist()
