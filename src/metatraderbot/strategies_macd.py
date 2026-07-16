"""MACD Trend Follower — classic momentum crossover strategy.

Rules:
- Compute MACD (12/26 EMAs) and its signal line (9-EMA of MACD).
- LONG when MACD line crosses UP through its signal line AND MACD is positive (uptrend).
- SHORT when MACD line crosses DOWN through its signal line AND MACD is negative.
- Entry: current bar's close.
- SL: swing low/high over lookback window.
- TP: risk_distance * tp_r_multiple.

Fires medium frequency. Different signal shape from Donchian/NoWick — captures
momentum shifts rather than breakouts, so it diversifies the trend-following bucket.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class MACDTrendStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        m = (s.get("macd") or {})
        self.fast = int(m.get("fast_period", 12))
        self.slow = int(m.get("slow_period", 26))
        self.signal_period = int(m.get("signal_period", 9))
        self.require_zero_side = bool(m.get("require_zero_side", True))

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
        min_bars = self.slow + self.signal_period + 5
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        closes = recent["close"]

        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        if len(macd) < 2 or macd.isna().iloc[-1] or signal.isna().iloc[-1]:
            return []

        macd_now, macd_prev = macd.iloc[-1], macd.iloc[-2]
        sig_now, sig_prev = signal.iloc[-1], signal.iloc[-2]

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        # Bullish cross: MACD crosses ABOVE signal line
        bull_cross = macd_prev <= sig_prev and macd_now > sig_now
        # Bearish cross
        bear_cross = macd_prev >= sig_prev and macd_now < sig_now

        if bull_cross and (not self.require_zero_side or macd_now > 0):
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
        elif bear_cross and (not self.require_zero_side or macd_now < 0):
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
