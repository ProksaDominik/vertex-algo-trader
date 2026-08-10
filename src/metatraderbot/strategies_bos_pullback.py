"""Trend Pullback Strategy (BOS Continuation).

Institutions move price in waves: a Break of Structure (BOS) — a candle BODY
closing past the prior swing high/low — consumes liquidity and confirms the
next wave's direction. Instead of chasing that breakout (where retail traders
get trapped), this waits for price to pull back into the 50%-61.8% Fibonacci
"discount zone" of the move that caused the BOS, then joins on a confirmation
candle (pinbar or engulfing) inside that zone.

Two timeframes, one bar stream: the entry-timeframe bars are resampled
in-strategy into higher-timeframe (HTF) bars to confirm a genuine HH/HL (or
LH/LL) trend before taking a BOS on the entry timeframe — approximating the
"1H bias, 15m entry" split from a single bar series (htf_multiplier bars per
HTF bar, e.g. 4 x 15m = 1H).

Never chases a wick-only break: the swing level must be broken by a candle
BODY close, not a high/low poke (wicks are frequently liquidity sweeps).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from metatraderbot.indicators import is_bearish_engulfing, is_bearish_pinbar, is_bullish_engulfing, is_bullish_pinbar
from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


@dataclass
class PendingBOS:
    direction: Direction
    bos_level: float          # the swing high/low that was broken
    swing_extreme: float      # the swing low/high that started the move (fib anchor)
    bars_since_bos: int = 0


class TrendPullbackBOSStrategy:
    PENDING_CLS = PendingBOS  # see NoWickStrategy.PENDING_CLS

    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("bos_pullback") or {})
        self.htf_multiplier = int(d.get("htf_multiplier", 4))   # e.g. 4 x 15m = 1H
        self.htf_lookback = int(d.get("htf_lookback", 20))
        self.swing_lookback = int(d.get("swing_lookback", 20))
        self.fib_low = float(d.get("fib_low", 0.5))
        self.fib_high = float(d.get("fib_high", 0.618))
        self.sl_buffer_pips = float(d.get("sl_buffer_pips", 4.0))
        self.max_wait_bars = int(d.get("max_wait_bars", 20))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

        self.pending: list[PendingBOS] = []

    def _htf_trend(self, bars: pd.DataFrame) -> str:
        """Resamples the tail of `bars` into htf_multiplier-bar buckets and
        checks the last few buckets for higher-highs+higher-lows (up) or
        lower-highs+lower-lows (down). Needs at least 3 HTF bars to compare."""
        n = self.htf_multiplier
        need = n * (self.htf_lookback + 1)
        window = bars.tail(need)
        if len(window) < n * 3:
            return "none"
        # Drop a partial leading bucket so every group has exactly n bars.
        usable = len(window) - (len(window) % n)
        window = window.tail(usable)
        groups = [window.iloc[i : i + n] for i in range(0, usable, n)]
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

    def _fib_zone(self, move_start: float, move_end: float) -> tuple[float, float]:
        """50%-61.8% retracement zone of the move from move_start to
        move_end, measured back from move_end. move_start->move_end is the
        impulse leg that caused the BOS (swing_extreme -> bos_level for
        BOTH directions). Returns (zone_lo, zone_hi) sorted so the same call
        shape works for a downward retrace (LONG) or an upward one (SHORT)."""
        rng = move_end - move_start
        a = move_end - rng * self.fib_low
        b = move_end - rng * self.fib_high
        return (a, b) if a <= b else (b, a)

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = max(self.htf_multiplier * (self.htf_lookback + 1), self.swing_lookback + 2)
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        prev = recent.iloc[-2]
        trades: list[TradeSpec] = []

        # 1) Advance pending BOS setups: waiting for a pullback into the fib
        #    discount/premium zone + a confirmation candle.
        still_pending: list[PendingBOS] = []
        for sig in self.pending:
            sig.bars_since_bos += 1
            if sig.bars_since_bos > self.max_wait_bars:
                continue  # expired without a pullback entry

            zone_lo, zone_hi = self._fib_zone(sig.swing_extreme, sig.bos_level)

            if sig.direction == Direction.LONG:
                # Structure failure: price already broke back below the move's origin.
                if current["close"] < sig.swing_extreme:
                    continue
                in_zone = zone_lo <= current["low"] <= zone_hi
                confirmed = in_zone and (
                    is_bullish_pinbar(current) or is_bullish_engulfing(prev, current)
                )
                if confirmed:
                    entry = float(current["close"])
                    sl = sig.swing_extreme - self.sl_buffer_pips * self.pip
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
                    continue  # setup used, don't keep it pending
            else:
                if current["close"] > sig.swing_extreme:
                    continue
                in_zone = zone_lo <= current["high"] <= zone_hi
                confirmed = in_zone and (
                    is_bearish_pinbar(current) or is_bearish_engulfing(prev, current)
                )
                if confirmed:
                    entry = float(current["close"])
                    sl = sig.swing_extreme + self.sl_buffer_pips * self.pip
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
                    continue

            still_pending.append(sig)
        self.pending = still_pending

        # 2) Detect a NEW BOS on this bar: a candle BODY close past the prior
        #    swing high/low (never a wick-only poke), aligned with the HTF trend.
        swing_window = recent.iloc[-(self.swing_lookback + 1) : -1]  # excludes current bar
        prior_swing_high = float(swing_window["high"].max())
        prior_swing_low = float(swing_window["low"].min())
        htf = self._htf_trend(recent)

        body_close = float(current["close"])
        if htf == "up" and body_close > prior_swing_high:
            self.pending.append(PendingBOS(
                direction=Direction.LONG,
                bos_level=prior_swing_high,
                swing_extreme=prior_swing_low,
            ))
        elif htf == "down" and body_close < prior_swing_low:
            self.pending.append(PendingBOS(
                direction=Direction.SHORT,
                bos_level=prior_swing_low,
                swing_extreme=prior_swing_high,
            ))

        return trades
