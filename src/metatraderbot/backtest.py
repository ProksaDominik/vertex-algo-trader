"""
Bar-close backtest engine for the no-wick strategy.

Fill model:
- Entry fills at signal candle's flat side (limit) on the bar that first crosses it.
- Exit fills at SL or TP whichever the bar's high/low hits.
- If both SL and TP are inside the same bar's range, assume SL first (conservative).
- Spread + commission applied per round-trip in dollars.

Extend for Phase 2 with real MT5 fills instead of these idealizations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from metatraderbot.instruments import get_specs
from metatraderbot.strategy import (
    Direction,
    NoWickStrategy,
    TradeSpec,
)


@dataclass
class OpenPosition:
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    entered_at: pd.Timestamp
    orig_sl: float = 0.0

    def __post_init__(self):
        if not self.orig_sl:
            self.orig_sl = self.stop_loss


@dataclass
class ClosedPosition:
    direction: Direction
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lots: float
    entered_at: pd.Timestamp
    exited_at: pd.Timestamp
    exit_reason: str
    pnl_usd: float


class Backtest:
    def __init__(self, config: dict, bars: pd.DataFrame, strategy=None):
        self.config = config
        self.bars = bars.reset_index()
        if "timestamp" not in self.bars.columns:
            self.bars = self.bars.rename(columns={self.bars.columns[0]: "timestamp"})

        cap = config["capital"]
        self.initial_capital = float(cap["initial_usd"])
        self.max_positions = int(cap["max_positions"])
        self.daily_stop_usd = float(cap["daily_stop_loss_usd"])

        # Per-pair contract specs (mult + default spread/commission).
        specs = get_specs(config)
        self.mult = specs["contract_mult"]

        # Config's backtest section overrides pair defaults if explicitly set.
        # For multi-pair sweeps, callers can pop these keys to force per-pair specs.
        bt = config.get("backtest", {})
        self.spread_price = float(bt.get("spread_price", specs["spread_price"]))
        self.commission_per_lot = float(bt.get("commission_per_lot", specs["commission_per_lot"]))

        # Live exit-stack modeling (Phase 5): the runner's ratchet trail and
        # skyrocket exit, approximated at BAR-CLOSE granularity (live
        # re-evaluates every tick, i.e. trails tighter than this model —
        # the approximation is conservative for the trail's benefit).
        # Defaults 0 = off = identical to the legacy fill model, so the
        # original sweep results stay reproducible.
        self.trail_start_r = float(cap.get("trail_start_r", 0.0))
        self.trail_lock_fraction = float(cap.get("trail_lock_fraction", 0.5))
        self.skyrocket_multiple = float(cap.get("skyrocket_multiple", 0.0))

        # Optional TIME STOP: force a position out at market once it has been
        # held this long, whatever the price. Default 0 = off, so every
        # existing result stays reproducible.
        #
        # Exists to test "keep the strong daily entries but cap how long
        # capital is committed" -- this roster's daily legs hold 22 days median
        # and up to 265, which is what makes a $2,000 realised payout take
        # months. Capping the hold is not free: a trade cut at 2 days exits
        # wherever price happens to be, so winners that needed a week to reach
        # target become partial gains or small losses. Whether that trade is
        # worth making is an empirical question, not an obvious one.
        self.max_hold_hours = float(cap.get("max_hold_hours", 0.0))

        self.strategy = strategy if strategy is not None else NoWickStrategy(config)
        self.equity = self.initial_capital
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

        self.open: list[OpenPosition] = []
        self.closed: list[ClosedPosition] = []
        self.daily_pnl: dict[date, float] = {}
        self.sub_min_lot_count: int = 0  # trades whose ideal lots < 0.01 (unfilling live)

    def _raw_pnl(self, pos: OpenPosition, exit_price: float) -> float:
        move = (exit_price - pos.entry_price) if pos.direction == Direction.LONG \
            else (pos.entry_price - exit_price)
        return move * pos.lots * self.mult

    def _costs(self, lots: float) -> float:
        # Round-turn commission + spread (spread hit once on entry).
        return (self.commission_per_lot * lots * 2) + (self.spread_price * lots * self.mult)

    def _check_exits(self, bar: pd.Series, bar_ts: pd.Timestamp, avg_range: float = 0.0):
        still_open = []
        for pos in self.open:
            hit_sl = False
            hit_tp = False
            if pos.direction == Direction.LONG:
                if bar["low"] <= pos.stop_loss:
                    hit_sl = True
                if bar["high"] >= pos.take_profit:
                    hit_tp = True
            else:
                if bar["high"] >= pos.stop_loss:
                    hit_sl = True
                if bar["low"] <= pos.take_profit:
                    hit_tp = True

            # SL/TP take precedence over the time stop: if this bar's range
            # reached either level, that exit happened first intrabar.
            timed_out = (
                self.max_hold_hours > 0
                and (bar_ts - pos.entered_at).total_seconds() / 3600.0 >= self.max_hold_hours
            )

            if hit_sl:
                exit_price, reason = pos.stop_loss, "sl"
            elif hit_tp:
                exit_price, reason = pos.take_profit, "tp"
            elif timed_out:
                # Out at market on this bar's close, wherever price is.
                exit_price, reason = float(bar["close"]), "time"
            else:
                # Skyrocket exit at bar close (mirrors runner logic).
                closed_now = False
                if self.skyrocket_multiple > 0 and avg_range > 0:
                    rng = float(bar["high"] - bar["low"])
                    bullish = bar["close"] > bar["open"]
                    favorable = bullish if pos.direction == Direction.LONG else (not bullish)
                    move = (bar["close"] - pos.entry_price) if pos.direction == Direction.LONG \
                        else (pos.entry_price - bar["close"])
                    if rng >= self.skyrocket_multiple * avg_range and favorable and move > 0:
                        exit_price, reason = float(bar["close"]), "skyrocket"
                        closed_now = True
                if not closed_now:
                    # Ratchet trail update from this bar's close (mirrors the
                    # runner: engage at trail_start_r, lock a fraction of the
                    # reached gain, only ever tighten).
                    if self.trail_start_r > 0:
                        r_dist = abs(pos.entry_price - pos.orig_sl)
                        if r_dist > 0:
                            profit_dist = (bar["close"] - pos.entry_price) \
                                if pos.direction == Direction.LONG \
                                else (pos.entry_price - bar["close"])
                            profit_r = profit_dist / r_dist
                            if profit_r >= self.trail_start_r:
                                lock_r = profit_r * self.trail_lock_fraction
                                cand = pos.entry_price + lock_r * r_dist \
                                    if pos.direction == Direction.LONG \
                                    else pos.entry_price - lock_r * r_dist
                                improves = cand > pos.stop_loss if pos.direction == Direction.LONG \
                                    else cand < pos.stop_loss
                                if improves:
                                    pos.stop_loss = cand
                    still_open.append(pos)
                    continue

            pnl = self._raw_pnl(pos, exit_price) - self._costs(pos.lots)
            self.equity += pnl
            self.closed.append(
                ClosedPosition(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    lots=pos.lots,
                    entered_at=pos.entered_at,
                    exited_at=bar_ts,
                    exit_reason=reason,
                    pnl_usd=pnl,
                )
            )
            d = bar_ts.date()
            self.daily_pnl[d] = self.daily_pnl.get(d, 0.0) + pnl

        self.open = still_open

    def _try_enter(self, trade: TradeSpec, bar_ts: pd.Timestamp):
        if len(self.open) >= self.max_positions:
            return
        if self.daily_pnl.get(bar_ts.date(), 0.0) <= self.daily_stop_usd:
            return
        # Note: no equity-based skip. Phase 1 measures the strategy's raw edge
        # across the whole dataset. Capital survival is a Phase 4 concern
        # (kill switch on real blowup). Keeping the equity gate here masked
        # the strategy's real behavior by force-closing open positions at
        # end-of-data with phantom losses.
        if trade.lots < 0.01:
            self.sub_min_lot_count += 1  # would be rejected on a real retail broker
        self.open.append(
            OpenPosition(
                direction=trade.direction,
                entry_price=trade.entry_price,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                lots=trade.lots,
                entered_at=bar_ts,
            )
        )

    def run(self) -> None:
        need_range = self.skyrocket_multiple > 0
        ranges = (self.bars["high"] - self.bars["low"]).rolling(20).mean() if need_range else None
        for i in range(len(self.bars)):
            bar = self.bars.iloc[i]
            bar_ts = pd.Timestamp(bar["timestamp"])

            avg_range = float(ranges.iloc[i - 1]) if (need_range and i > 0 and pd.notna(ranges.iloc[i - 1])) else 0.0
            self._check_exits(bar, bar_ts, avg_range)
            new_trades = self.strategy.on_bar(i, self.bars)
            for t in new_trades:
                self._try_enter(t, bar_ts)

            self.equity_curve.append((bar_ts, self.equity))

        # Force-close any leftovers at last close.
        if self.open and len(self.bars):
            last = self.bars.iloc[-1]
            last_ts = pd.Timestamp(last["timestamp"])
            for pos in self.open:
                pnl = self._raw_pnl(pos, last["close"]) - self._costs(pos.lots)
                self.equity += pnl
                self.closed.append(
                    ClosedPosition(
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=last["close"],
                        stop_loss=pos.stop_loss,
                        take_profit=pos.take_profit,
                        lots=pos.lots,
                        entered_at=pos.entered_at,
                        exited_at=last_ts,
                        exit_reason="end-of-data",
                        pnl_usd=pnl,
                    )
                )
            self.open = []

    def report(self) -> dict:
        wins = [t for t in self.closed if t.pnl_usd > 0]
        losses = [t for t in self.closed if t.pnl_usd <= 0]
        total_pnl = sum(t.pnl_usd for t in self.closed)

        equity_vals = [e[1] for e in self.equity_curve]
        eq = pd.Series(equity_vals) if equity_vals else pd.Series([self.initial_capital])
        peak = eq.cummax()
        drawdown_series = eq - peak
        max_dd_usd = float(drawdown_series.min())
        max_dd_pct = float((drawdown_series / peak).min() * 100) if peak.max() > 0 else 0.0

        by_reason: dict[str, int] = {}
        pnl_by_reason: dict[str, float] = {}
        for t in self.closed:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
            pnl_by_reason[t.exit_reason] = pnl_by_reason.get(t.exit_reason, 0.0) + t.pnl_usd

        return {
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(self.closed) * 100) if self.closed else 0.0,
            "total_pnl_usd": total_pnl,
            "initial_capital": self.initial_capital,
            "final_equity": self.equity,
            "return_pct": (self.equity / self.initial_capital - 1) * 100,
            "avg_win_usd": (sum(t.pnl_usd for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss_usd": (sum(t.pnl_usd for t in losses) / len(losses)) if losses else 0.0,
            "max_drawdown_usd": max_dd_usd,
            "max_drawdown_pct": max_dd_pct,
            "expectancy_usd_per_trade": (total_pnl / len(self.closed)) if self.closed else 0.0,
            "blown_up": self.equity <= 0,
            "sub_min_lot_trades": self.sub_min_lot_count,
            "sub_min_lot_pct": (self.sub_min_lot_count / len(self.closed) * 100) if self.closed else 0.0,
            "by_exit_reason": by_reason,
            "pnl_by_exit_reason": pnl_by_reason,
        }
