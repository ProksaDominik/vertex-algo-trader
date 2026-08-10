"""Sweep Reclaim — a liquidity-grab fade.

Idea: resting stop-loss orders pool just beyond obvious swing highs/lows.
Price often spikes THROUGH such a level (a "sweep" / stop hunt), triggers
those stops, then snaps back — because the move had no real intent, it was
just grabbing liquidity. This strategy fades that: it waits for a bar that
pierces a recent swing low but CLOSES back above it (the "reclaim"), and
goes long — betting the sweep is exhausted. Mirror for shorts (pierce a
swing high, close back below).

- Bullish: current low < recent swing low, but current close > that swing
  low -> LONG on the reclaim. Stop below the sweep's low.
- Bearish: current high > recent swing high, but close < it -> SHORT.
- TP: risk_distance * tp_r_multiple.

Mean-reversion in character, so it diversifies the trend-following bucket
(Donchian/MACD/BOS). Best in ranges and at session extremes; gets run over
in a strong one-way trend (the sweep just keeps going).
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.indicators import atr
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class SweepReclaimStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        p = (s.get("sweep") or {})
        self.lookback = int(p.get("lookback", 20))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.25))
        self.min_pierce_atr = float(p.get("min_pierce_atr", 0.0))  # optional: sweep must pierce by >= this*ATR

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
        if bar_index < self.lookback + 15:
            return []
        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        prev = recent.iloc[-(self.lookback + 1):-1]      # exclude current bar
        swing_low = float(prev["low"].min())
        swing_high = float(prev["high"].max())

        atr_val = float(atr(recent, 14).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return []
        buf = self.sl_buffer_atr * atr_val
        min_pierce = self.min_pierce_atr * atr_val

        entry = float(current["close"])
        low = float(current["low"])
        high = float(current["high"])
        trades: list[TradeSpec] = []

        # Bullish sweep + reclaim: dipped below the swing low, closed above it.
        if low < swing_low - min_pierce and entry > swing_low:
            sl = low - buf
            sl_dist = entry - sl
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.LONG, entry_price=entry, stop_loss=sl,
                        take_profit=entry + sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
        # Bearish sweep + reclaim.
        elif high > swing_high + min_pierce and entry < swing_high:
            sl = high + buf
            sl_dist = sl - entry
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.SHORT, entry_price=entry, stop_loss=sl,
                        take_profit=entry - sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
        return trades
