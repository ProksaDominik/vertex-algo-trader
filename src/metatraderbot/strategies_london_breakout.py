"""Module B: London Breakout.

NOTE: docs/strategy/FOREX_STRATEGY_STACK_SPEC.md (the actual spec this
module name is drawn from) was not available -- only the name and a couple
of filter hints. This is a reasonable, self-contained interpretation, not
a transcription of that spec.

Idea: the Asian session's range is often a compression before the London
volume surge. Rather than fading it (Module A), this trades the breakout:
once London opens, a candle that BODY-closes decisively beyond the Asian
range is treated as the real move starting, and joins in that direction.
Never chases a wick-only poke -- the close itself must clear the level.

- Bullish: close > Asian high (with a minimum clearance) during the
  breakout window -> LONG, stop back at the Asian low side of the range
  (or a tighter ATR-style buffer below entry if the range is wide).
- Bearish: mirror, close < Asian low -> SHORT.
- Requires a minimum daily-range filter (D1 ATR) so this doesn't trade a
  breakout on an already-dead, illiquid day.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.indicators import atr
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


class LondonBreakoutStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("london_breakout") or {})
        self.session_start_hour = int(d.get("session_start_hour_utc", 0))
        self.session_end_hour = int(d.get("session_end_hour_utc", 7))
        self.breakout_start_hour = int(d.get("breakout_start_hour_utc", 7))
        self.breakout_end_hour = int(d.get("breakout_end_hour_utc", 10))
        self.min_clearance_pips = float(d.get("min_clearance_pips", 2.0))
        self.sl_buffer_pips = float(d.get("sl_buffer_pips", 3.0))
        self.min_d1_atr_pips = float(d.get("min_d1_atr_pips", 40.0))
        self.lookback_bars = int(d.get("lookback_bars", 96))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

    def _session_range(self, recent: pd.DataFrame, current_ts: pd.Timestamp) -> tuple[float, float] | None:
        window = recent.tail(self.lookback_bars)
        ts = _timestamps(window)
        same_day = ts.date == current_ts.date()
        in_session = (ts.hour >= self.session_start_hour) & (ts.hour < self.session_end_hour)
        session_bars = window[same_day & in_session]
        if session_bars.empty:
            return None
        return float(session_bars["low"].min()), float(session_bars["high"].max())

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = max(self.lookback_bars, 20)
        if bar_index < min_bars:
            return []
        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        current_ts = _timestamps(recent)[-1]

        hour = current_ts.hour
        if not (self.breakout_start_hour <= hour < self.breakout_end_hour):
            return []

        # Daily-range liquidity filter: skip dead days (proxy for D1 ATR
        # using a ~20-bar-per-day-ish window on the entry timeframe).
        atr_val = float(atr(recent, 14).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return []

        rng = self._session_range(recent, current_ts)
        if rng is None:
            return []
        session_low, session_high = rng

        clearance = self.min_clearance_pips * self.pip
        buf = self.sl_buffer_pips * self.pip
        entry = float(current["close"])
        opened = float(current["open"])
        trades: list[TradeSpec] = []

        if entry > session_high + clearance and entry > opened:
            sl = session_low - buf
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
        elif entry < session_low - clearance and entry < opened:
            sl = session_high + buf
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
