"""Smart-Money-Concepts BOS: Liquidity Sweep -> Break of Structure -> Order
Block retest.

The popular retail "BOS" playbook (ICT/SMC style), distinct from this
project's other two BOS-named strategies:
  - strategies_bos.py (MarketOpenBOSStrategy): session-anchored breakout of
    a swing level, no sweep/reversal concept.
  - strategies_bos_pullback.py (TrendPullbackBOSStrategy): HTF-trend-aligned
    CONTINUATION break + Fibonacci retracement zone entry.

This one is a REVERSAL pattern, the classic SMC narrative:
  1. LIQUIDITY SWEEP: a bar's wick pokes past a recent swing high/low (where
     retail stop-losses/breakout orders cluster) but the bar's BODY closes
     back on the other side -- the "stop hunt" that traps breakout traders.
  2. BREAK OF STRUCTURE (BOS): within a few bars, price reverses and a
     candle's BODY closes beyond a nearer, opposite-side swing point --
     confirming the sweep was a trap, not the start of a real move.
  3. ORDER BLOCK (OB): the last opposite-colored candle before the impulse
     leg that caused the BOS -- e.g. for a bearish BOS, the last bullish
     candle before the down-move. Institutional narrative: that candle is
     where "smart money" built the position now driving the reversal.
  4. ENTRY: price retraces back into the OB's body range -> enter in the
     BOS direction. SL beyond the sweep's extreme; TP a fixed R multiple.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


@dataclass
class PendingOB:
    direction: Direction
    ob_low: float
    ob_high: float
    sweep_extreme: float   # beyond this = invalidated
    bars_since_bos: int = 0


class SMCBOSStrategy:
    PENDING_CLS = PendingOB

    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("smc_bos") or {})
        self.swing_lookback = int(d.get("swing_lookback", 15))
        self.sweep_buffer_pips = float(d.get("sweep_buffer_pips", 1.0))
        self.bos_lookback = int(d.get("bos_lookback", 8))       # nearer swing used for the BOS itself
        self.max_wait_bars = int(d.get("max_wait_bars", 15))
        self.sl_buffer_pips = float(d.get("sl_buffer_pips", 3.0))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

        self.pending: list[PendingOB] = []
        self._last_sweep: dict | None = None  # {"direction", "extreme", "bars_ago"}

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        min_bars = self.swing_lookback + 2
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        current = recent.iloc[-1]
        trades: list[TradeSpec] = []

        # 1) Advance pending order-block setups.
        still_pending: list[PendingOB] = []
        for sig in self.pending:
            sig.bars_since_bos += 1
            if sig.bars_since_bos > self.max_wait_bars:
                continue
            if sig.direction == Direction.LONG:
                if current["close"] < sig.sweep_extreme:
                    continue  # structure re-invalidated
                in_ob = sig.ob_low <= current["low"] <= sig.ob_high
                if in_ob:
                    entry = float(current["close"])
                    sl = sig.sweep_extreme - self.sl_buffer_pips * self.pip
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
            else:
                if current["close"] > sig.sweep_extreme:
                    continue
                in_ob = sig.ob_low <= current["high"] <= sig.ob_high
                if in_ob:
                    entry = float(current["close"])
                    sl = sig.sweep_extreme + self.sl_buffer_pips * self.pip
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

        # 2) Detect a liquidity sweep on this bar (wick beyond swing, body back inside).
        swing_window = recent.iloc[-(self.swing_lookback + 1): -1]
        swing_high = float(swing_window["high"].max())
        swing_low = float(swing_window["low"].min())
        buf = self.sweep_buffer_pips * self.pip

        if float(current["high"]) > swing_high + buf and float(current["close"]) < swing_high:
            self._last_sweep = {"direction": "down", "extreme": float(current["high"]), "bars_ago": 0}
        elif float(current["low"]) < swing_low - buf and float(current["close"]) > swing_low:
            self._last_sweep = {"direction": "up", "extreme": float(current["low"]), "bars_ago": 0}
        elif self._last_sweep is not None:
            self._last_sweep["bars_ago"] += 1
            if self._last_sweep["bars_ago"] > self.max_wait_bars:
                self._last_sweep = None

        # 3) Given a live sweep, look for the BOS confirmation + build the OB.
        if self._last_sweep is not None and bar_index >= self.bos_lookback + 1:
            bos_window = recent.iloc[-(self.bos_lookback + 1): -1]
            near_low = float(bos_window["low"].min())
            near_high = float(bos_window["high"].max())

            if self._last_sweep["direction"] == "down" and float(current["close"]) < near_low:
                # bearish BOS after a high-side sweep -> find the OB: last
                # bullish candle before this down-impulse.
                impulse = recent.iloc[-(self.bos_lookback + 1):]
                ob = None
                for j in range(len(impulse) - 2, -1, -1):
                    row = impulse.iloc[j]
                    if row["close"] > row["open"]:
                        ob = row
                        break
                if ob is not None:
                    self.pending.append(PendingOB(
                        direction=Direction.SHORT,
                        ob_low=float(min(ob["open"], ob["close"])),
                        ob_high=float(max(ob["open"], ob["close"])),
                        sweep_extreme=self._last_sweep["extreme"],
                    ))
                self._last_sweep = None
            elif self._last_sweep["direction"] == "up" and float(current["close"]) > near_high:
                impulse = recent.iloc[-(self.bos_lookback + 1):]
                ob = None
                for j in range(len(impulse) - 2, -1, -1):
                    row = impulse.iloc[j]
                    if row["close"] < row["open"]:
                        ob = row
                        break
                if ob is not None:
                    self.pending.append(PendingOB(
                        direction=Direction.LONG,
                        ob_low=float(min(ob["open"], ob["close"])),
                        ob_high=float(max(ob["open"], ob["close"])),
                        sweep_extreme=self._last_sweep["extreme"],
                    ))
                self._last_sweep = None

        return trades
