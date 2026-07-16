"""Bollinger Bounce — mean-reversion strategy.

Rules:
- Compute BB(period, std_dev) on close prices.
- LONG when a bar dips through the lower band and closes back above it (rejection).
- SHORT when a bar spikes through the upper band and closes back below it.
- Entry: the current bar's close.
- SL: opposite side of the trigger bar (low - buffer for LONG, high + buffer for SHORT).
- TP: risk_distance * tp_r_multiple (1:1 default).

Fires more often than NoWick — potentially several signals per day per pair —
which suits high-frequency trading. Ranges well, but gets stopped out in strong trends.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class BollingerBounceStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        bb = (s.get("bollinger") or {})
        self.period = int(bb.get("period", 20))
        self.std_dev = float(bb.get("std_dev", 2.0))
        self.buffer_pct = float(bb.get("sl_buffer_pct", 0.20))

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
        if bar_index < self.period + 2:
            return []

        recent = bars.iloc[: bar_index + 1]
        closes = recent["close"]
        sma = closes.rolling(self.period).mean().iloc[-1]
        std = closes.rolling(self.period).std().iloc[-1]
        if pd.isna(sma) or pd.isna(std) or std == 0:
            return []

        upper = float(sma + self.std_dev * std)
        lower = float(sma - self.std_dev * std)
        band_width = upper - lower
        buffer = band_width * self.buffer_pct

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        if current["low"] <= lower and entry > lower:
            sl = float(current["low"]) - buffer
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
        elif current["high"] >= upper and entry < upper:
            sl = float(current["high"]) + buffer
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
