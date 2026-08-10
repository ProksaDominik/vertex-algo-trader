"""Module D: H4 Pullback.

NOTE: docs/strategy/FOREX_STRATEGY_STACK_SPEC.md (the actual spec this
module name is drawn from) was not available -- only the name and a
filter hint (swap-cost gate). This is a reasonable, self-contained
interpretation, not a transcription of that spec.

Idea: trade WITH the higher-timeframe (H4) trend, entered on a pullback to
a fast EMA on the entry timeframe rather than chasing continuation moves.
Unlike Module C's EMA-crossover version (a discrete cross event), this is
continuous: as long as H4 bias holds, ANY pullback tag of the fast EMA with
a same-direction confirmation candle is actionable -- which is why this
module fires more often than a fresh-cross strategy (research/freq.py's
own filter-cascade estimate has this module's trigger stage firing on
~85% of regime-qualifying days).

- H4 bias: resample the entry-timeframe bars into htf_multiplier-bar
  buckets (default 8 x 30m = H4) and require higher-high+higher-low (up)
  or lower-high+lower-low (down) over the last 3 buckets.
- Entry: price pulls back to tag the fast EMA while staying on the biased
  side of the slow EMA, confirmed by a decisive same-direction candle.
- SL: beyond the slow EMA or the recent swing extreme, whichever is
  further (mirrors the EMA-crossover module's stop logic).
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


class H4PullbackStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("h4_pullback") or {})
        self.htf_multiplier = int(d.get("htf_multiplier", 8))   # e.g. 8 x 30m = H4
        self.htf_lookback = int(d.get("htf_lookback", 8))
        self.fast_period = int(d.get("fast_period", 21))
        self.slow_period = int(d.get("slow_period", 50))
        self.pullback_tolerance_pips = float(d.get("pullback_tolerance_pips", 3.0))
        self.sl_swing_lookback = int(d.get("sl_swing_lookback", 10))
        self.min_body_pct = float(d.get("confirmation_min_body_pct", 0.4))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

    def _htf_trend(self, bars: pd.DataFrame) -> str:
        n = self.htf_multiplier
        need = n * (self.htf_lookback + 1)
        window = bars.tail(need)
        if len(window) < n * 3:
            return "none"
        usable = len(window) - (len(window) % n)
        window = window.tail(usable)
        groups = [window.iloc[i: i + n] for i in range(0, usable, n)]
        htf_highs = [g["high"].max() for g in groups]
        htf_lows = [g["low"].min() for g in groups]
        if len(htf_highs) < 3:
            return "none"
        h1, h2, h3 = htf_highs[-3], htf_highs[-2], htf_highs[-1]
        l1, l2, l3 = htf_lows[-3], htf_lows[-2], htf_lows[-1]
        if h3 > h2 > h1 and l3 > l2 > l1:
            return "up"
        if h3 < h2 < h1 and l3 < l2 < l1:
            return "down"
        return "none"

    @staticmethod
    def _is_decisive(row: pd.Series, direction: Direction, min_body_pct: float) -> bool:
        rng = row["high"] - row["low"]
        if rng <= 0:
            return False
        body = abs(row["close"] - row["open"])
        if direction == Direction.LONG:
            return row["close"] > row["open"] and (body / rng) >= min_body_pct
        return row["close"] < row["open"] and (body / rng) >= min_body_pct

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = max(
            self.htf_multiplier * (self.htf_lookback + 1),
            self.slow_period + self.sl_swing_lookback + 2,
        )
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        htf = self._htf_trend(recent)
        if htf == "none":
            return []

        closes = recent["close"]
        ema_fast = closes.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow_period, adjust=False).mean()
        if ema_fast.isna().iloc[-1] or ema_slow.isna().iloc[-1]:
            return []

        current = recent.iloc[-1]
        fast_now, slow_now = float(ema_fast.iloc[-1]), float(ema_slow.iloc[-1])
        price = float(current["close"])
        tol = self.pullback_tolerance_pips * self.pip
        trades: list[TradeSpec] = []

        if htf == "up" and price > slow_now:
            tagged = current["low"] <= fast_now + tol
            confirmed = tagged and self._is_decisive(current, Direction.LONG, self.min_body_pct)
            if confirmed:
                swing_low = float(recent["low"].tail(self.sl_swing_lookback).min())
                sl = min(slow_now, swing_low)
                sl_distance = price - sl
                if sl_distance > 0:
                    lots = self.risk_usd / (sl_distance * self.contract_mult)
                    if lots > 0:
                        tp = price + sl_distance * self.tp_r_multiple
                        trades.append(TradeSpec(
                            direction=Direction.LONG, entry_price=price,
                            stop_loss=sl, take_profit=tp, lots=lots,
                            signal_bar_index=bar_index,
                        ))
        elif htf == "down" and price < slow_now:
            tagged = current["high"] >= fast_now - tol
            confirmed = tagged and self._is_decisive(current, Direction.SHORT, self.min_body_pct)
            if confirmed:
                swing_high = float(recent["high"].tail(self.sl_swing_lookback).max())
                sl = max(slow_now, swing_high)
                sl_distance = sl - price
                if sl_distance > 0:
                    lots = self.risk_usd / (sl_distance * self.contract_mult)
                    if lots > 0:
                        tp = price - sl_distance * self.tp_r_multiple
                        trades.append(TradeSpec(
                            direction=Direction.SHORT, entry_price=price,
                            stop_loss=sl, take_profit=tp, lots=lots,
                            signal_bar_index=bar_index,
                        ))
        return trades
