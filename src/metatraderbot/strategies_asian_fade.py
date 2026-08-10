"""Module A: Asian Range Fade.

NOTE: this project's docs/strategy/FOREX_STRATEGY_STACK_SPEC.md was not
available when this was written -- only the module NAME and a few filter
hints (research/freq.py, SPRINT_CONFIG_ADDENDUM.md) were. This is a
reasonable, self-contained interpretation of "Asian fade", not a
transcription of that spec.

Idea: the Asian session is thin and range-bound. Its high/low becomes a
level where late/impatient participants get trapped on a false break before
London volume arrives. Build the session's range while it's forming, then
during the fade window (after the session, before/into London), fade a
wick that pierces the range and closes back inside it -- betting the poke
was a liquidity grab, not the start of a real move.

- Bullish fade: a bar's low dips below the Asian low but its close is back
  inside the range -> LONG, stop below the wick's low.
- Bearish fade: a bar's high pokes above the Asian high but closes back
  inside -> SHORT, stop above the wick's high.
- No fade if the day's Asian range is too small to be a real level (thin
  holiday sessions produce noise, not a level worth trading against).
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


def _timestamps(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Works whether `bars` carries real datetimes as its INDEX (live bars
    -- mt5_engine.get_bars does bars.set_index("timestamp")) or as a
    "timestamp" COLUMN (metatraderbot.backtest.Backtest resets the index to
    a plain range and keeps timestamp as a column instead)."""
    if "timestamp" in bars.columns:
        return pd.DatetimeIndex(bars["timestamp"])
    return bars.index


class AsianFadeStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("asian_fade") or {})
        self.session_start_hour = int(d.get("session_start_hour_utc", 0))
        self.session_end_hour = int(d.get("session_end_hour_utc", 7))
        self.fade_start_hour = int(d.get("fade_start_hour_utc", 7))
        self.fade_end_hour = int(d.get("fade_end_hour_utc", 16))
        self.min_range_pips = float(d.get("min_range_pips", 15.0))
        self.sl_buffer_pips = float(d.get("sl_buffer_pips", 3.0))
        self.lookback_bars = int(d.get("lookback_bars", 96))  # bars searched for "today's" session

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

    def _session_range(self, recent: pd.DataFrame, current_ts: pd.Timestamp) -> tuple[float, float] | None:
        """Today's Asian-session high/low, from bars whose timestamp falls
        in [session_start_hour, session_end_hour) on the SAME UTC date as
        the current bar (the Asian window is early enough UTC that it never
        crosses into the next calendar day for the fade window that follows
        it)."""
        window = recent.tail(self.lookback_bars)
        ts = _timestamps(window)
        same_day = ts.date == current_ts.date()
        in_session = (ts.hour >= self.session_start_hour) & (ts.hour < self.session_end_hour)
        session_bars = window[same_day & in_session]
        if session_bars.empty:
            return None
        return float(session_bars["low"].min()), float(session_bars["high"].max())

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < self.lookback_bars:
            return []
        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        current_ts = _timestamps(recent)[-1]

        hour = current_ts.hour
        if not (self.fade_start_hour <= hour < self.fade_end_hour):
            return []

        rng = self._session_range(recent, current_ts)
        if rng is None:
            return []
        session_low, session_high = rng
        if (session_high - session_low) < self.min_range_pips * self.pip:
            return []  # no real range today -- nothing to fade

        buf = self.sl_buffer_pips * self.pip
        entry = float(current["close"])
        low = float(current["low"])
        high = float(current["high"])
        trades: list[TradeSpec] = []

        if low < session_low and entry > session_low:
            sl = low - buf
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
        elif high > session_high and entry < session_high:
            sl = high + buf
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
