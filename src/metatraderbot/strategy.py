"""
No-wick strategy state machine.

Detects wickless-with-trend candles, waits N bars for price to retrace to the
flat side, then emits a TradeSpec for the backtest engine (or the live paper
trader in Phase 2) to execute.

Rules (per the video's description):
- Bullish trend (price > 50-EMA, EMA slope up) + bullish wickless (flat bottom)
    -> mark the candle's LOW as the retrace level; enter LONG when hit
- Bearish trend (price < 50-EMA, EMA slope down) + bearish wickless (flat top)
    -> mark the candle's HIGH as the retrace level; enter SHORT when hit
- SL = swing high (short) / swing low (long) over lookback window
- TP = 1:1 R:R (configurable multiplier)
- Signal expires if not retraced within `retrace_bars` bars
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from metatraderbot.indicators import (
    ema,
    is_wickless_bearish,
    is_wickless_bullish,
    swing_high,
    swing_low,
)
from metatraderbot.instruments import contract_mult


# Kept for callers that still import the old name; new code uses contract_mult().
USD_PER_LOT_PER_DOLLAR_MOVE = 100.0


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class PendingSignal:
    direction: Direction
    entry_level: float
    signal_bar_index: int
    signal_bar_high: float
    signal_bar_low: float
    bars_since_signal: int = 0


@dataclass
class TradeSpec:
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    signal_bar_index: int


class NoWickStrategy:
    # Tells portfolio_runner._restore/_persist_runtime_state which dataclass
    # `self.pending` holds, so restart-restore can reconstruct it generically
    # instead of hardcoding one strategy's pending-signal shape.
    PENDING_CLS = PendingSignal

    def __init__(self, config: dict):
        s = config["strategy"]
        self.ema_period = int(s["trend"]["ema_period"])
        self.wick_max_pct = float(s["signal"]["wick_max_pct"])
        self.body_min_pct = float(s["signal"]["body_min_pct"])
        self.retrace_bars = int(s["entry"]["retrace_bars"])
        self.sl_lookback = int(s["exit"]["sl_swing_lookback"])
        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        # Optional quality filters (all default to disabled).
        self.min_ema_slope_pct = float(s["trend"].get("min_ema_slope_pct", 0.0))
        self.min_ema_distance_pct = float(s["trend"].get("min_ema_distance_pct", 0.0))
        sess = s.get("session") or {}
        self.session_start_hour = sess.get("start_hour_utc")
        self.session_end_hour = sess.get("end_hour_utc")

        self.pending: list[PendingSignal] = []

    def _trend(self, closes: pd.Series) -> str:
        e = ema(closes, self.ema_period)
        if len(e) < 2:
            return "none"
        cur, prev = e.iloc[-1], e.iloc[-2]
        if cur == 0:
            return "none"
        slope_pct = (cur - prev) / cur
        price = closes.iloc[-1]
        distance_pct = (price - cur) / cur
        if (price > cur
                and slope_pct >= self.min_ema_slope_pct
                and distance_pct >= self.min_ema_distance_pct):
            return "up"
        if (price < cur
                and slope_pct <= -self.min_ema_slope_pct
                and distance_pct <= -self.min_ema_distance_pct):
            return "down"
        return "none"

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        """Called at the close of each bar. Returns trades to open on the *next* bar."""
        if bar_index < max(self.ema_period, self.sl_lookback) + 2:
            return []

        recent = bars.iloc[: bar_index + 1]
        current_bar = recent.iloc[-1]
        trend = self._trend(recent["close"])

        # 1) Check pending signals from prior bars for retrace on this bar.
        trades: list[TradeSpec] = []
        still_pending: list[PendingSignal] = []

        for sig in self.pending:
            sig.bars_since_signal += 1
            if sig.bars_since_signal > self.retrace_bars:
                continue  # expired without retrace

            hit = False
            entry_price = sig.entry_level

            if sig.direction == Direction.SHORT and current_bar["high"] >= sig.entry_level:
                hit = True
            elif sig.direction == Direction.LONG and current_bar["low"] <= sig.entry_level:
                hit = True

            if not hit:
                still_pending.append(sig)
                continue

            # Compute SL over swing lookback, then lot size to risk exactly risk_usd.
            sl_slice = recent.iloc[-self.sl_lookback :]
            if sig.direction == Direction.SHORT:
                sl = swing_high(sl_slice["high"], self.sl_lookback)
                if sl <= entry_price:
                    sl = entry_price + (entry_price - sig.signal_bar_low)
                sl_distance = sl - entry_price
                tp = entry_price - sl_distance * self.tp_r_multiple
            else:
                sl = swing_low(sl_slice["low"], self.sl_lookback)
                if sl >= entry_price:
                    sl = entry_price - (sig.signal_bar_high - entry_price)
                sl_distance = entry_price - sl
                tp = entry_price + sl_distance * self.tp_r_multiple

            if sl_distance <= 0:
                continue

            # Fractional lots: measure the strategy's TRUE edge, unclamped by
            # broker minimums. The backtest engine records what fraction of
            # trades would be sub-0.01 (unfilling on a real retail account).
            lots = self.risk_usd / (sl_distance * self.contract_mult)
            if lots <= 0:
                continue

            trades.append(
                TradeSpec(
                    direction=sig.direction,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    lots=lots,
                    signal_bar_index=sig.signal_bar_index,
                )
            )

        self.pending = still_pending

        # 2a) Session filter: only *detect* new signals during the allowed
        #     window. Retrace-based entries above still fire any time.
        if self.session_start_hour is not None and self.session_end_hour is not None:
            try:
                hour = pd.Timestamp(current_bar["timestamp"]).hour
                if not (self.session_start_hour <= hour < self.session_end_hour):
                    return trades
            except (KeyError, ValueError):
                pass  # no timestamp column — don't filter

        # 2b) Detect NEW wickless-with-trend signal on the current bar.
        #     Added after retrace check so it can't fire on its own signal bar.
        if trend == "down" and is_wickless_bearish(current_bar, self.wick_max_pct, self.body_min_pct):
            self.pending.append(
                PendingSignal(
                    direction=Direction.SHORT,
                    entry_level=current_bar["high"],
                    signal_bar_index=bar_index,
                    signal_bar_high=current_bar["high"],
                    signal_bar_low=current_bar["low"],
                )
            )
        elif trend == "up" and is_wickless_bullish(current_bar, self.wick_max_pct, self.body_min_pct):
            self.pending.append(
                PendingSignal(
                    direction=Direction.LONG,
                    entry_level=current_bar["low"],
                    signal_bar_index=bar_index,
                    signal_bar_high=current_bar["high"],
                    signal_bar_low=current_bar["low"],
                )
            )

        return trades
