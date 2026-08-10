"""Time-of-Day (TOD) Reversion — fade an intraday over-extension at a set hour.

Idea: some hours of the day tend to mean-revert — a market that has run
hard away from where it opened often drifts back toward it later in the
session (position squaring, end-of-session flows). This strategy waits for
a specific hour, and IF price is stretched far from the day's opening price,
it fades the move — betting on a pull back toward the open.

- At `revert_hour_utc`, measure deviation = close − day_open.
- If deviation > stretch_atr × ATR(14) (stretched UP), go SHORT.
- If deviation < −stretch_atr × ATR (stretched DOWN), go LONG.
- Stop just beyond the current bar's extreme; TP = risk × tp_r_multiple.
- At most one trade per day.

Time-anchored and mean-reverting — a different shape from the rolling
trend systems. Works when the day is range-bound and over-extended; loses
when a genuine trend keeps running past the fade.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.indicators import atr
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class TODReversionStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        p = (s.get("tod") or {})
        self.revert_hour = int(p.get("revert_hour_utc", 20))
        self.stretch_atr = float(p.get("stretch_atr", 1.5))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.25))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        self._day = None
        self._day_open = None
        self._traded_today = False

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < 20:
            return []
        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        try:
            ts = pd.Timestamp(current["timestamp"])
        except (KeyError, ValueError):
            return []

        day = ts.date()
        if day != self._day:
            self._day = day
            self._day_open = float(current["open"])
            self._traded_today = False
        if self._traded_today or self._day_open is None:
            return []
        if ts.hour != self.revert_hour:
            return []

        atr_val = float(atr(recent, 14).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return []

        entry = float(current["close"])
        deviation = entry - self._day_open
        buf = self.sl_buffer_atr * atr_val
        trades: list[TradeSpec] = []

        if deviation > self.stretch_atr * atr_val:
            # Stretched up -> fade short toward the open.
            sl = float(current["high"]) + buf
            sl_dist = sl - entry
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.SHORT, entry_price=entry, stop_loss=sl,
                        take_profit=entry - sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
                    self._traded_today = True
        elif deviation < -self.stretch_atr * atr_val:
            sl = float(current["low"]) - buf
            sl_dist = entry - sl
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.LONG, entry_price=entry, stop_loss=sl,
                        take_profit=entry + sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index))
                    self._traded_today = True
        return trades
