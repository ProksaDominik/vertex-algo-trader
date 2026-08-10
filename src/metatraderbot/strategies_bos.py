"""Market-Open Break of Structure (BOS).

Idea: liquidity and direction often get set right after a session open
(London ~07:00 UTC, New York ~13:00 UTC). This strategy waits for the open,
reads the most recent *market structure* — the last confirmed swing high and
swing low (fractal pivots), not just a range — and enters on the first clean
BREAK of that structure within a trade window. Break of the swing high =
long; break of the swing low = short. Stop sits at the opposing structure.

Different signal shape from ORB (which trades a fixed opening-range box) and
from Donchian (rolling channel): BOS anchors to actual pivots and to a
session clock, at most one trade per symbol per day.

Causality: a pivot at bar i is only *confirmed* `swing_strength` bars later
(it must be the extreme of the window on both sides), so the structure used
for a decision never peeks at future bars — no repainting, no lookahead.

Params (config['strategy']['bos']):
  session_open_hour_utc : session open hour, UTC (default 13 = NY)
  trade_window_hours    : hours after open to hunt the break (default 4)
  swing_lookback        : bars scanned back for the structure (default 40)
  swing_strength        : fractal strength k; pivot = extreme of 2k+1 window (default 2)
Exit: config['strategy']['exit']['tp_r_multiple'].
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


def last_confirmed_swings(highs, lows, k: int):
    """Most recent CONFIRMED swing high and swing low (fractal strength k),
    scanning newest→oldest. A bar is a confirmed pivot only if it's the
    extreme of the k bars on EACH side, so the newest k bars can't be
    pivots yet (not enough right-side confirmation) — that's what keeps the
    signal causal. Returns (swing_high, swing_low), either may be None."""
    n = len(highs)
    sh = sl = None
    for i in range(n - 1 - k, k - 1, -1):
        if sh is None and highs[i] == max(highs[i - k:i + k + 1]):
            sh = float(highs[i])
        if sl is None and lows[i] == min(lows[i - k:i + k + 1]):
            sl = float(lows[i])
        if sh is not None and sl is not None:
            break
    return sh, sl


class MarketOpenBOSStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        b = (s.get("bos") or {})
        self.session_open_hour = int(b.get("session_open_hour_utc", 13))
        self.trade_window_hours = float(b.get("trade_window_hours", 4))
        self.swing_lookback = int(b.get("swing_lookback", 40))
        self.swing_strength = int(b.get("swing_strength", 2))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

        self._tracked_day = None
        self._traded_today = False

    def _reset_day(self, day) -> None:
        self._tracked_day = day
        self._traded_today = False

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < self.swing_lookback + self.swing_strength + 2:
            return []
        current = bars.iloc[bar_index]
        try:
            ts = pd.Timestamp(current["timestamp"])
        except (KeyError, ValueError):
            return []

        day = ts.date()
        if day != self._tracked_day:
            self._reset_day(day)
        if self._traded_today:
            return []

        # Only inside [open, open + trade_window].
        bar_min = ts.hour * 60 + ts.minute
        open_min = self.session_open_hour * 60
        if bar_min < open_min or bar_min >= open_min + self.trade_window_hours * 60:
            return []

        window = bars.iloc[bar_index - self.swing_lookback: bar_index + 1]
        highs = window["high"].tolist()
        lows = window["low"].tolist()
        swing_high, swing_low = last_confirmed_swings(highs, lows, self.swing_strength)
        if swing_high is None or swing_low is None:
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        # Break ABOVE the last swing high -> long; break BELOW swing low -> short.
        if entry > swing_high:
            sl = swing_low
            sl_dist = entry - sl
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.LONG, entry_price=entry, stop_loss=sl,
                        take_profit=entry + sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index,
                    ))
                    self._traded_today = True
        elif entry < swing_low:
            sl = swing_high
            sl_dist = sl - entry
            if sl_dist > 0:
                lots = self.risk_usd / (sl_dist * self.contract_mult)
                if lots > 0:
                    trades.append(TradeSpec(
                        direction=Direction.SHORT, entry_price=entry, stop_loss=sl,
                        take_profit=entry - sl_dist * self.tp_r_multiple,
                        lots=lots, signal_bar_index=bar_index,
                    ))
                    self._traded_today = True
        return trades
