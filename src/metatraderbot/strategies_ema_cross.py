"""Dual Moving Average Crossover (20 / 50 EMA).

EMAs smooth price into a read on momentum: the fast (20) EMA tracks
short-term momentum, the slow (50) tracks medium-term. A crossover confirms
a shift in control. Rather than buying/selling right on the cross (chasing),
this waits for a pullback to tag the fast EMA and a confirmation candle,
joining the move with a pre-defined, structurally-placed stop.

Rules:
- 20 EMA crosses above/below the 50 EMA -> arm a pending setup.
- Price must hold on the correct side of both EMAs while pending (an
  immediate re-cross the other way invalidates the setup — this is what
  ranging/whipsaw markets look like, and the strategy's own known weakness).
- Wait for price to pull back and tag the 20 EMA, then a same-direction
  confirmation candle (a decisive close, not just a touch) to enter.
- SL: whichever is FURTHER from entry of {50 EMA, recent swing low/high} —
  the wider of the two, since either alone can be too tight in a fast move.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from metatraderbot.instruments import contract_mult, pip_size
from metatraderbot.strategy import Direction, TradeSpec


@dataclass
class PendingCross:
    direction: Direction
    bars_since_cross: int = 0


class EMACrossoverStrategy:
    PENDING_CLS = PendingCross  # see NoWickStrategy.PENDING_CLS

    def __init__(self, config: dict):
        s = config["strategy"]
        d = (s.get("ema_cross") or {})
        self.fast_period = int(d.get("fast_period", 20))
        self.slow_period = int(d.get("slow_period", 50))
        self.pullback_tolerance_pips = float(d.get("pullback_tolerance_pips", 3.0))
        self.sl_swing_lookback = int(d.get("sl_swing_lookback", 10))
        self.max_wait_bars = int(d.get("max_wait_bars", 15))
        self.min_body_pct = float(d.get("confirmation_min_body_pct", 0.4))

        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self.pip = pip_size(config["symbol"])

        self.pending: list[PendingCross] = []

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
        min_bars = self.slow_period + self.sl_swing_lookback + 2
        if bar_index < min_bars:
            return []

        recent = bars.iloc[: bar_index + 1]
        closes = recent["close"]
        ema_fast = closes.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow_period, adjust=False).mean()
        if len(ema_fast) < 2 or ema_fast.isna().iloc[-1] or ema_slow.isna().iloc[-1]:
            return []

        current = recent.iloc[-1]
        fast_now, slow_now = ema_fast.iloc[-1], ema_slow.iloc[-1]
        fast_prev, slow_prev = ema_fast.iloc[-2], ema_slow.iloc[-2]
        trades: list[TradeSpec] = []

        # 1) Advance a pending crossover: alignment must hold, then wait for
        #    a pullback that TAGS the fast EMA, confirmed by a decisive candle.
        # At most one setup is ever pending (LONG/SHORT are mutually
        # exclusive), but `pending` is a list -- like every other stateful
        # strategy here -- so the generic persist/restore-across-restart
        # path (portfolio_runner._persist_runtime_state) can treat it the
        # same as BOS/SRRetest/NoWick instead of needing a special case.
        still_pending: list[PendingCross] = []
        for sig in self.pending:
            sig.bars_since_cross += 1
            price = float(current["close"])

            alignment_broken = (
                (sig.direction == Direction.LONG and price < slow_now)
                or (sig.direction == Direction.SHORT and price > slow_now)
            )
            if alignment_broken or sig.bars_since_cross > self.max_wait_bars:
                continue  # expired / invalidated, don't keep it

            tol = self.pullback_tolerance_pips * self.pip
            if sig.direction == Direction.LONG:
                tagged = current["low"] <= fast_now + tol
                confirmed = tagged and self._is_decisive(current, Direction.LONG, self.min_body_pct)
                if confirmed:
                    entry = price
                    swing_low = float(recent["low"].tail(self.sl_swing_lookback).min())
                    sl = min(slow_now, swing_low)  # whichever is FURTHER below entry
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
                tagged = current["high"] >= fast_now - tol
                confirmed = tagged and self._is_decisive(current, Direction.SHORT, self.min_body_pct)
                if confirmed:
                    entry = price
                    swing_high = float(recent["high"].tail(self.sl_swing_lookback).max())
                    sl = max(slow_now, swing_high)  # whichever is FURTHER above entry
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

        # 2) Detect a NEW crossover on this bar (only if nothing's already pending).
        if not self.pending:
            bull_cross = fast_prev <= slow_prev and fast_now > slow_now
            bear_cross = fast_prev >= slow_prev and fast_now < slow_now
            if bull_cross and current["close"] > slow_now:
                self.pending.append(PendingCross(direction=Direction.LONG))
            elif bear_cross and current["close"] < slow_now:
                self.pending.append(PendingCross(direction=Direction.SHORT))

        return trades
