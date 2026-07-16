"""SMA Crossover — the most iconic retail trend-following strategy.

Rules:
- Fast SMA (default 10) and Slow SMA (default 30) on close prices.
- LONG when fast crosses UP through slow AND close is above slow.
- SHORT when fast crosses DOWN through slow AND close is below slow.
- Entry: current bar's close.
- SL: swing low/high over lookback window.
- TP: risk_distance * tp_r_multiple.

Simple, well-known, works well in trending markets. Adds a different signal
shape to the portfolio: pure trend detection via MA relationship rather than
breakout (Donchian) or momentum (MACD).
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class SMACrossStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        m = (s.get("ma_cross") or {})
        self.fast_period = int(m.get("fast_period", 10))
        self.slow_period = int(m.get("slow_period", 30))

        self.sl_lookback = int(s["exit"]["sl_swing_lookback"])
        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        sess = s.get("session") or {}
        self.session_start_hour = sess.get("start_hour_utc")
        self.session_end_hour = sess.get("end_hour_utc")

    def _session_ok(self, current: pd.Series) -> bool:
        if self.session_start_hour is None or self.session_end_hour is None:
            return True
        try:
            hour = pd.Timestamp(current["timestamp"]).hour
            return self.session_start_hour <= hour < self.session_end_hour
        except (KeyError, ValueError):
            return True

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < self.slow_period + 5:
            return []

        recent = bars.iloc[: bar_index + 1]
        closes = recent["close"]

        fast = closes.rolling(self.fast_period).mean()
        slow = closes.rolling(self.slow_period).mean()

        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or pd.isna(fast.iloc[-2]):
            return []

        fast_now, fast_prev = fast.iloc[-1], fast.iloc[-2]
        slow_now, slow_prev = slow.iloc[-1], slow.iloc[-2]

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        # Bullish cross: fast SMA crosses up through slow SMA + price above slow
        bull_cross = fast_prev <= slow_prev and fast_now > slow_now
        bear_cross = fast_prev >= slow_prev and fast_now < slow_now

        if bull_cross and entry > slow_now:
            sl_slice = recent.iloc[-self.sl_lookback :]
            sl = float(sl_slice["low"].min())
            sl_distance = entry - sl
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry + sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.LONG,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        elif bear_cross and entry < slow_now:
            sl_slice = recent.iloc[-self.sl_lookback :]
            sl = float(sl_slice["high"].max())
            sl_distance = sl - entry
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry - sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.SHORT,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        return trades
