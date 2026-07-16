"""Stochastic Oscillator — retail scalping favorite.

Rules:
- %K = 100 * (close - low_N) / (high_N - low_N) over N=14 bars.
- %D = 3-bar SMA of %K.
- LONG when %K crosses UP through %D in the oversold zone (<20).
- SHORT when %K crosses DOWN through %D in the overbought zone (>80).
- Entry: current bar's close.
- SL: swing low/high over lookback window.
- TP: risk_distance * tp_r_multiple.

Different signal shape from RSI (which is a single line): the %K/%D crossover
gives a two-line confirmation. Good for mean-reversion on ranging pairs.
"""
from __future__ import annotations

import pandas as pd

from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec


class StochasticStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        st = (s.get("stochastic") or {})
        self.k_period = int(st.get("k_period", 14))
        self.d_period = int(st.get("d_period", 3))
        self.oversold = float(st.get("oversold", 20))
        self.overbought = float(st.get("overbought", 80))

        self.sl_lookback = int(s["exit"]["sl_swing_lookback"])
        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)

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

    def _stochastic(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        highs = bars["high"].rolling(self.k_period).max()
        lows = bars["low"].rolling(self.k_period).min()
        rng = (highs - lows).replace(0, 1e-12)
        k = 100 * (bars["close"] - lows) / rng
        d = k.rolling(self.d_period).mean()
        return k, d

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        if bar_index < self.k_period + self.d_period + 5:
            return []

        recent = bars.iloc[: bar_index + 1]
        k, d = self._stochastic(recent)

        if len(k) < 2 or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]) or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]):
            return []

        k_now, k_prev = k.iloc[-1], k.iloc[-2]
        d_now, d_prev = d.iloc[-1], d.iloc[-2]

        current = recent.iloc[-1]
        if not self._session_ok(current):
            return []

        entry = float(current["close"])
        trades: list[TradeSpec] = []

        # Bull cross in oversold: %K crosses ABOVE %D while below 20 zone
        bull_cross = k_prev <= d_prev and k_now > d_now
        bear_cross = k_prev >= d_prev and k_now < d_now

        if bull_cross and k_now < self.oversold + 10 and d_now < self.oversold + 10:
            sl_slice = recent.iloc[-self.sl_lookback :]
            sl = float(sl_slice["low"].min())
            sl_distance = entry - sl
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry + sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.LONG,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        elif bear_cross and k_now > self.overbought - 10 and d_now > self.overbought - 10:
            sl_slice = recent.iloc[-self.sl_lookback :]
            sl = float(sl_slice["high"].max())
            sl_distance = sl - entry
            if sl_distance > 0:
                lots = self.risk_usd / (sl_distance * self.contract_mult)
                if lots > 0:
                    tp = entry - sl_distance * self.tp_r_multiple
                    trades.append(TradeSpec(
                        direction=Direction.SHORT,
                        entry_price=entry, stop_loss=sl, take_profit=tp,
                        lots=lots, signal_bar_index=bar_index,
                    ))
        return trades
