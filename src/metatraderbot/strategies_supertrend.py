"""Supertrend -- a simple ATR-band trailing indicator, one of the most
widely used trend indicators in retail/crypto trading for how easy it is to
read (one flipping line on the chart).

  mid = (high + low) / 2
  upper = mid + multiplier * ATR(period)
  lower = mid - multiplier * ATR(period)
  The "final" bands only move IN the trend's favor (ratchet, same idea as
  this project's own trailing stop): final_upper only decreases while price
  stays below it, final_lower only increases while price stays above it.
  Direction flips (and the line jumps to the other side) the bar price
  closes through it.

LONG on a flip to uptrend, SHORT on a flip to downtrend. SL at the
Supertrend line itself; TP a fixed R multiple.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metatraderbot.indicators import atr as _atr
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class SupertrendStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("supertrend") or {})
        self.period = int(d.get("period", 10))
        self.multiplier = float(d.get("multiplier", 3.0))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

    def _compute(self, bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (supertrend_line, direction) as numpy arrays, direction +1/-1.
        Plain-array inner loop -- pandas .iloc scalar access in a per-bar
        Python loop is too slow for backtesting thousands of bars."""
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()
        a = _atr(bars, self.period).to_numpy()
        mid = (high + low) / 2.0
        upper = mid + self.multiplier * a
        lower = mid - self.multiplier * a

        n = len(bars)
        final_upper = upper.copy()
        final_lower = lower.copy()
        direction = np.ones(n, dtype=np.int8)

        for i in range(1, n):
            if np.isnan(upper[i]) or np.isnan(final_upper[i - 1]):
                continue
            final_upper[i] = min(upper[i], final_upper[i - 1]) if close[i - 1] <= final_upper[i - 1] else upper[i]
            final_lower[i] = max(lower[i], final_lower[i - 1]) if close[i - 1] >= final_lower[i - 1] else lower[i]
            prev_dir = direction[i - 1]
            if prev_dir == 1:
                direction[i] = -1 if close[i] < final_lower[i] else 1
            else:
                direction[i] = 1 if close[i] > final_upper[i] else -1

        line = np.where(direction == 1, final_lower, final_upper)
        return line, direction

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = self.period + 5
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        line, direction = self._compute(recent)
        if len(direction) < 2:
            return []
        cur_dir, prev_dir = direction[-1], direction[-2]
        if cur_dir == prev_dir:
            return []  # only trade on the flip bar itself

        price = float(recent["close"].iloc[-1])
        sl = float(line[-1])
        trades: list[TradeSpec] = []

        if cur_dir == 1:
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
        else:
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
