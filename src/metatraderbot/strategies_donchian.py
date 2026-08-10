"""Donchian Breakout — classic Turtle-style momentum strategy.

Rules:
- Compute Donchian channel(period): highest high & lowest low over last `period`
  bars EXCLUDING the current bar.
- LONG when current close > upper channel (breakout above N-period high).
- SHORT when current close < lower channel (breakdown below N-period low).
- Entry: current bar's close.
- SL: opposite channel (wide SL, small lots — the trade-off of breakout systems).
- TP: risk_distance * tp_r_multiple.

Trend-following; performs well in trending regimes, gets chopped in ranges.
Very different regime from NoWick and BollingerBounce — good portfolio diversifier.

Optional confirmation filters (all off by default, backward compatible):
- min_breakout_atr_mult: only take a breakout that clears the channel by at
  least this many multiples of ATR(14) — filters out marginal "just poked
  through the level" breaks in favor of decisive ones.
- trend_ema_period: only take breakouts aligned with a longer EMA's
  direction (LONG needs close > EMA, SHORT needs close < EMA) — filters out
  breakouts fighting the broader trend.
- min_momentum_atr_mult: only take a breakout whose OWN bar range (high-low)
  is at least this many multiples of ATR(14) — filters out breakouts on a
  slow/quiet bar in favor of ones happening on a rapid, high-velocity move.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.indicators import atr, ema
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class DonchianBreakoutStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("donchian") or {})
        self.period = int(d.get("period", 20))
        self.min_breakout_atr_mult = float(d.get("min_breakout_atr_mult", 0.0))
        self.trend_ema_period = int(d.get("trend_ema_period", 0) or 0)
        self.min_momentum_atr_mult = float(d.get("min_momentum_atr_mult", 0.0))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        # Stop-placement mode (default preserves the classic Turtle behavior:
        # SL at the full opposite channel — a deliberately WIDE stop).
        #   "channel"      : SL = opposite channel edge (original)
        #   "channel_frac" : SL distance = sl_channel_frac x channel width
        #   "atr"          : SL distance = sl_atr_mult x ATR(14)
        # A tighter stop lets the SAME dollar risk buy a bigger position and
        # brings the (tp_r_multiple x SL) target physically closer — the
        # trade-off is more shake-outs on noise. Which wins is an empirical
        # question, answered per-symbol by scripts/geometry_sweep.py.
        exit_cfg = s.get("exit", {}) or {}
        self.sl_mode = str(d.get("sl_mode", exit_cfg.get("sl_mode", "channel"))).lower()
        self.sl_channel_frac = float(d.get("sl_channel_frac", 1.0))
        self.sl_atr_mult = float(d.get("sl_atr_mult", 1.5))

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
        min_bars = max(self.period, self.trend_ema_period) + 2
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        prev_window = recent.iloc[-(self.period + 1) : -1]  # exclude current bar
        upper = float(prev_window["high"].max())
        lower = float(prev_window["low"].min())

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        min_margin = 0.0
        atr_val = None
        need_atr = (self.min_breakout_atr_mult > 0 or self.min_momentum_atr_mult > 0
                    or self.sl_mode == "atr")
        if need_atr:
            atr_val = atr(recent, 14).iloc[-1]
            if pd.isna(atr_val) or atr_val <= 0:
                return []
            min_margin = self.min_breakout_atr_mult * atr_val

        def _sl_distance(channel_dist: float) -> float:
            if self.sl_mode == "atr":
                return self.sl_atr_mult * atr_val
            if self.sl_mode == "channel_frac":
                return self.sl_channel_frac * channel_dist
            return channel_dist  # "channel" — full opposite channel (original)

        if self.min_momentum_atr_mult > 0:
            bar_range = float(current["high"] - current["low"])
            if bar_range < self.min_momentum_atr_mult * atr_val:
                return []

        trend_ok_long = trend_ok_short = True
        if self.trend_ema_period > 0:
            trend = ema(recent["close"], self.trend_ema_period).iloc[-1]
            trend_ok_long = entry > trend
            trend_ok_short = entry < trend

        if entry > upper + min_margin and trend_ok_long:
            sl_distance = _sl_distance(entry - lower)
            if sl_distance > 0:
                sl = entry - sl_distance
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry + sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.LONG,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        elif entry < lower - min_margin and trend_ok_short:
            sl_distance = _sl_distance(upper - entry)
            if sl_distance > 0:
                sl = entry + sl_distance
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry - sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.SHORT,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        return trades
