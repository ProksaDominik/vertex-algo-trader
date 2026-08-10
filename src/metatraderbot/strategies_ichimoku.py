"""Ichimoku Kinko Hyo ("one glance equilibrium chart") -- a complete
Japanese trend-following system, extremely popular especially on JPY pairs
where it originated. Five lines, three used here as an entry filter stack:

  - Tenkan-sen (conversion, 9):  (9-high + 9-low) / 2
  - Kijun-sen   (base, 26):      (26-high + 26-low) / 2
  - Senkou Span A/B (the "cloud"): projected forward 26 bars on a real
    chart. For a CAUSAL signal (no lookahead), "the cloud at bar i" uses
    the span values computed 26 bars ago -- exactly what a chart actually
    displays as the cloud sitting under/over bar i right now.
  - Chikou confirmation: current close vs close 26 bars ago.

LONG: price above the cloud, Tenkan > Kijun, Chikou (current close) above
the close from 26 bars ago -- full bullish alignment. SHORT is the mirror.
SL beyond the Kijun-sen; TP a fixed R multiple.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class IchimokuStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("ichimoku") or {})
        self.tenkan_period = int(d.get("tenkan_period", 9))
        self.kijun_period = int(d.get("kijun_period", 26))
        self.senkou_b_period = int(d.get("senkou_b_period", 52))
        self.displacement = int(d.get("displacement", 26))
        self.sl_buffer_atr_mult = float(d.get("sl_buffer_atr_mult", 0.5))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

    @staticmethod
    def _mid(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        return (high.rolling(period).max() + low.rolling(period).min()) / 2.0

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = self.senkou_b_period + self.displacement + 2
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        high, low, close = recent["high"], recent["low"], recent["close"]

        tenkan = self._mid(high, low, self.tenkan_period)
        kijun = self._mid(high, low, self.kijun_period)
        senkou_a_raw = (tenkan + kijun) / 2.0
        senkou_b_raw = self._mid(high, low, self.senkou_b_period)

        # Cloud "visible" at the current bar was computed `displacement`
        # bars ago -- causal, matches what a real chart shows.
        if len(senkou_a_raw) <= self.displacement:
            return []
        cloud_a = senkou_a_raw.iloc[-1 - self.displacement]
        cloud_b = senkou_b_raw.iloc[-1 - self.displacement]
        if pd.isna(cloud_a) or pd.isna(cloud_b):
            return []
        cloud_top, cloud_bot = max(cloud_a, cloud_b), min(cloud_a, cloud_b)

        tenkan_now, kijun_now = tenkan.iloc[-1], kijun.iloc[-1]
        if pd.isna(tenkan_now) or pd.isna(kijun_now):
            return []

        if len(close) <= self.displacement:
            return []
        chikou_ref = close.iloc[-1 - self.displacement]
        price = float(close.iloc[-1])

        bullish = price > cloud_top and tenkan_now > kijun_now and price > chikou_ref
        bearish = price < cloud_bot and tenkan_now < kijun_now and price < chikou_ref

        from metatraderbot.indicators import atr as _atr
        a = float(_atr(recent, 14).iloc[-1])
        if pd.isna(a) or a <= 0:
            return []
        buf = self.sl_buffer_atr_mult * a

        trades: list[TradeSpec] = []
        if bullish:
            entry = price
            sl = float(kijun_now) - buf
            sl_distance = entry - sl
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry + sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.LONG, entry_price=entry,
                        stop_loss=sl, take_profit=tp, lots=lots,
                        signal_bar_index=bar_index,
                    ))
        elif bearish:
            entry = price
            sl = float(kijun_now) + buf
            sl_distance = sl - entry
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry - sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.SHORT, entry_price=entry,
                        stop_loss=sl, take_profit=tp, lots=lots,
                        signal_bar_index=bar_index,
                    ))
        return trades
