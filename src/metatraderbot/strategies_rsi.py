"""RSI Mean Reversion — oscillator-based reversal strategy.

Rules:
- Compute RSI(period=14) on closes.
- LONG when RSI crosses UP through the oversold threshold (30) — bounce off oversold.
- SHORT when RSI crosses DOWN through the overbought threshold (70).
- Entry: current bar's close.
- SL: swing low/high over lookback window.
- TP: risk_distance * tp_r_multiple.

Pure oscillator — completely different signal shape from Bollinger (which uses bands).
Fires on outright oversold/overbought conditions, so signals cluster around strong
counter-trend swings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class RSIReversionStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        r = (s.get("rsi") or {})
        self.period = int(r.get("period", 14))
        self.oversold = float(r.get("oversold", 30))
        self.overbought = float(r.get("overbought", 70))

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

    def _rsi(self, closes: pd.Series) -> pd.Series:
        # Wilder's RSI (EWM alpha = 1/period)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < self.period + 5:
            return []

        recent = bars.iloc[: bar_index + 1]
        rsi = self._rsi(recent["close"])

        if len(rsi) < 2:
            return []
        rsi_now, rsi_prev = rsi.iloc[-1], rsi.iloc[-2]
        if pd.isna(rsi_now) or pd.isna(rsi_prev):
            return []

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        # LONG: RSI was oversold, now crossed back up
        if rsi_prev <= self.oversold and rsi_now > self.oversold:
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
        # SHORT: RSI was overbought, now crossed back down
        elif rsi_prev >= self.overbought and rsi_now < self.overbought:
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
