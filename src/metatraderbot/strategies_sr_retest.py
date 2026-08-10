"""Support & Resistance Breakout + Retest.

A horizontal level that price has tested and bounced off repeatedly is a
zone where trapped traders sit. When that level is forcefully breached, the
trapped side gets squeezed on the retest — Role Reversal (broken support
becomes resistance and vice versa) — creating double-sided pressure: trapped
traders closing at breakeven plus fresh breakout traders stepping in.

Rules:
- Scan a lookback window for a horizontal level touched >= min_touches times
  (clustered swing highs/lows within a pip tolerance).
- Wait for a strong candle to close clearly through the level (a real body
  break, not a wick).
- Wait for price to return and tag the broken level from the other side.
- Trigger only on a rejection candle (pinbar / shooting star) at the retest.
- Optional session filter — off-hours breakouts (thin Asian session) fake
  out far more often than London/NY-session ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from metatraderbot.indicators import is_bearish_pinbar, is_bullish_pinbar
from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


@dataclass
class PendingRetest:
    direction: Direction
    level: float
    bars_since_break: int = 0


def _cluster_touches(prices: list[float], tolerance: float) -> float | None:
    """Groups nearby swing prices into clusters and returns the mean of the
    largest cluster with >= 2 members, or None if nothing clusters. Simple
    O(n^2) grouping — lookback windows here are small (tens of bars)."""
    best_cluster: list[float] = []
    for p in prices:
        cluster = [q for q in prices if abs(q - p) <= tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    if len(best_cluster) < 2:
        return None
    return sum(best_cluster) / len(best_cluster)


class SRBreakoutRetestStrategy:
    PENDING_CLS = PendingRetest  # see NoWickStrategy.PENDING_CLS

    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("sr_retest") or {})
        self.level_lookback = int(d.get("level_lookback", 80))
        self.swing_arm = int(d.get("swing_arm", 2))  # bars each side to confirm a local extreme
        self.min_touches = int(d.get("min_touches", 2))
        self.retest_tolerance_pips = float(d.get("retest_tolerance_pips", 5.0))
        self.sl_buffer_pips = float(d.get("sl_buffer_pips", 7.0))
        self.max_wait_bars = int(d.get("max_wait_bars", 30))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])
        self.touch_tolerance = float(d.get("level_touch_tolerance_pips", 3.0)) * self.pip

        sess = s.get("session") or {}
        self.session_start_hour = sess.get("start_hour_utc")
        self.session_end_hour = sess.get("end_hour_utc")

        self.pending: list[PendingRetest] = []

    def _session_ok(self, current: pd.Series) -> bool:
        if self.session_start_hour is None or self.session_end_hour is None:
            return True
        try:
            hour = pd.Timestamp(current["timestamp"]).hour
            return self.session_start_hour <= hour < self.session_end_hour
        except (KeyError, ValueError):
            return True

    def _find_swing_lows(self, window: pd.DataFrame) -> list[float]:
        lows = window["low"].values
        arm = self.swing_arm
        out = []
        for i in range(arm, len(lows) - arm):
            seg = lows[i - arm : i + arm + 1]
            if lows[i] == seg.min():
                out.append(float(lows[i]))
        return out

    def _find_swing_highs(self, window: pd.DataFrame) -> list[float]:
        highs = window["high"].values
        arm = self.swing_arm
        out = []
        for i in range(arm, len(highs) - arm):
            seg = highs[i - arm : i + arm + 1]
            if highs[i] == seg.max():
                out.append(float(highs[i]))
        return out

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = self.level_lookback + self.swing_arm + 2
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        trades: list[TradeSpec] = []

        # 1) Advance pending retests: waiting for price to tag the broken
        #    level from the other side and reject with a pinbar.
        still_pending: list[PendingRetest] = []
        for sig in self.pending:
            sig.bars_since_break += 1
            if sig.bars_since_break > self.max_wait_bars:
                continue

            if sig.direction == Direction.SHORT:
                # Broken support, now resistance — wait for a tag from below.
                tagged = current["high"] >= sig.level - self.retest_tolerance_pips * self.pip
                if tagged and is_bearish_pinbar(current):
                    entry = float(current["close"])
                    sl = sig.level + self.sl_buffer_pips * self.pip
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
                # Structure failure: price never came back down through the level again.
            else:
                tagged = current["low"] <= sig.level + self.retest_tolerance_pips * self.pip
                if tagged and is_bullish_pinbar(current):
                    entry = float(current["close"])
                    sl = sig.level - self.sl_buffer_pips * self.pip
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
                    continue

            still_pending.append(sig)
        self.pending = still_pending

        if not self._session_ok(current):
            return trades

        # 2) Detect a NEW breakout on this bar, off a level built from
        #    clustered swing highs/lows in the lookback window (excludes the
        #    current bar so the level can't be built from the breakout itself).
        window = recent.iloc[-(self.level_lookback + self.swing_arm + 1) : -1]
        is_strong_bear = current["close"] < current["open"]
        is_strong_bull = current["close"] > current["open"]

        support = _cluster_touches(self._find_swing_lows(window), self.touch_tolerance)
        if (
            support is not None
            and is_strong_bear
            and current["close"] < support - self.touch_tolerance
        ):
            self.pending.append(PendingRetest(direction=Direction.SHORT, level=support))
            return trades  # one new setup per bar is plenty

        resistance = _cluster_touches(self._find_swing_highs(window), self.touch_tolerance)
        if (
            resistance is not None
            and is_strong_bull
            and current["close"] > resistance + self.touch_tolerance
        ):
            self.pending.append(PendingRetest(direction=Direction.LONG, level=resistance))

        return trades
