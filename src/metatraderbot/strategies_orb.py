"""Opening Range Breakout — session-anchored breakout strategy.

Rules:
- Mark the high/low of the first `range_minutes` of each day starting at
  `range_start_hour_utc` (a session open — London/NY etc.).
- Once that window closes, watch for price to break above the range high
  (long) or below the range low (short), within `trade_window_hours` of the
  range closing. After that window, or once a trade's been taken, stand
  down until the next day's range forms.
- Entry: the breaking bar's close. SL: the opposite side of the range.
- TP: risk_distance * tp_r_multiple (same convention as every other
  strategy here).

Unlike Donchian/MACD/SMACross (all pattern-anchored — they fire whenever
their condition happens to occur), this is time-anchored: at most one
signal per symbol per day, tied to a specific session window instead of a
rolling lookback. Different signal shape, not a variant of the existing
trend-following family.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec, bar_time


class ORBStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        o = (s.get("orb") or {})
        self.range_start_hour = int(o.get("range_start_hour_utc", 8))
        self.range_minutes = int(o.get("range_minutes", 30))
        self.trade_window_hours = float(o.get("trade_window_hours", 4))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        self._tracked_day = None
        self._range_high: float | None = None
        self._range_low: float | None = None
        self._traded_today = False

    def _reset_for_new_day(self, day) -> None:
        self._tracked_day = day
        self._range_high = None
        self._range_low = None
        self._traded_today = False

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        current = bars.iloc[bar_index]
        ts = bar_time(bars, bar_index)
        day = ts.date()
        if day != self._tracked_day:
            self._reset_for_new_day(day)

        bar_minutes = ts.hour * 60 + ts.minute
        range_start_min = self.range_start_hour * 60
        range_end_min = range_start_min + self.range_minutes

        # Still inside the opening-range window: accumulate high/low, no trades yet.
        if range_start_min <= bar_minutes < range_end_min:
            high, low = float(current["high"]), float(current["low"])
            self._range_high = high if self._range_high is None else max(self._range_high, high)
            self._range_low = low if self._range_low is None else min(self._range_low, low)
            return []

        # Range hasn't formed yet today, already traded it, or we're past
        # the trade window (don't chase a stale breakout hours later).
        if self._range_high is None or self._range_low is None or self._traded_today:
            return []
        trade_window_end_min = range_end_min + self.trade_window_hours * 60
        if bar_minutes >= trade_window_end_min:
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        if float(current["high"]) >= self._range_high and entry > self._range_high:
            sl = self._range_low
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
                    self._traded_today = True
        elif float(current["low"]) <= self._range_low and entry < self._range_low:
            sl = self._range_high
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
                    self._traded_today = True
        return trades
