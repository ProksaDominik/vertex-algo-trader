"""VWAP Reversion — fade a stretch away from the session VWAP.

VWAP (Volume-Weighted Average Price) is the average price paid over the
session, weighted by volume — institutions treat it as "fair value" for the
day. Price that runs far from VWAP tends to get pulled back toward it. This
strategy anchors VWAP to the start of each UTC day and, when price is
stretched a configurable distance from it AND the current bar shows a
rejection back toward it, enters a reversion trade targeting VWAP.

- Anchored VWAP = Σ(typical_price × volume) / Σ(volume) from the day's open,
  typical_price = (high + low + close) / 3.
- Long when close < VWAP − band and the bar closes up (rejection); short
  when close > VWAP + band and the bar closes down. band = k × ATR(14).
- Stop just beyond the current bar's extreme; TP = risk × tp_r_multiple.

Mean-reverting and volume-aware — the only strategy here that uses volume.
Strong in balanced/rotational sessions; struggles on trend days that walk
away from VWAP and never come back.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.indicators import atr
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec, bar_times


class VWAPReversionStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        p = (s.get("vwap") or {})
        self.band_atr = float(p.get("band_atr", 2.0))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.25))
        self.require_rejection = bool(p.get("require_rejection", True))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

    def _session_vwap(self, day_bars: pd.DataFrame) -> float:
        typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3.0
        vol = day_bars["volume"].astype(float)
        tot = float(vol.sum())
        if tot <= 0:
            return float(day_bars["close"].iloc[-1])
        return float((typical * vol).sum() / tot)

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < 20:
            return []
        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        # Bars belonging to the CURRENT UTC day (anchor point for VWAP).
        try:
            times = bar_times(recent)
            day = times[-1].date()
            same_day = recent[times.date == day]
        except (KeyError, ValueError, TypeError):
            return []
        if len(same_day) < 3:
            return []

        vwap = self._session_vwap(same_day)
        atr_val = float(atr(recent, 14).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return []
        band = self.band_atr * atr_val

        entry = float(current["close"])
        dev = entry - vwap
        buf = self.sl_buffer_atr * atr_val
        bull_bar = current["close"] > current["open"]
        trades: list[TradeSpec] = []

        # Stretched BELOW vwap -> long back toward it (rejection = up bar).
        if dev < -band and (bull_bar or not self.require_rejection):
            sl = float(current["low"]) - buf
            sl_dist = entry - sl
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.LONG, entry_price=entry, stop_loss=sl,
                        take_profit=entry + sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
        elif dev > band and (not bull_bar or not self.require_rejection):
            sl = float(current["high"]) + buf
            sl_dist = sl - entry
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.SHORT, entry_price=entry, stop_loss=sl,
                        take_profit=entry - sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
        return trades
