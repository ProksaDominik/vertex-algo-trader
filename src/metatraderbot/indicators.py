"""Technical indicators + candle helpers for the no-wick strategy."""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def is_wickless_bullish(row: pd.Series, wick_max_pct: float, body_min_pct: float) -> bool:
    """Bullish wickless: green candle with (near-)flat bottom + solid body."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return False
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return (
        row["close"] > row["open"]
        and (lower_wick / rng) <= wick_max_pct
        and (body / rng) >= body_min_pct
    )


def is_wickless_bearish(row: pd.Series, wick_max_pct: float, body_min_pct: float) -> bool:
    """Bearish wickless: red candle with (near-)flat top + solid body."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return False
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    return (
        row["close"] < row["open"]
        and (upper_wick / rng) <= wick_max_pct
        and (body / rng) >= body_min_pct
    )


def swing_high(highs: pd.Series, lookback: int) -> float:
    return float(highs.tail(lookback).max())


def swing_low(lows: pd.Series, lookback: int) -> float:
    return float(lows.tail(lookback).min())


def is_bullish_pinbar(row: pd.Series, wick_min_pct: float = 0.5, body_max_pct: float = 0.35) -> bool:
    """Long lower wick + small body near the top — rejection of lower prices."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return False
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    return (lower_wick / rng) >= wick_min_pct and (body / rng) <= body_max_pct


def is_bearish_pinbar(row: pd.Series, wick_min_pct: float = 0.5, body_max_pct: float = 0.35) -> bool:
    """Long upper wick + small body near the bottom — rejection of higher
    prices (a "shooting star" is a bearish pinbar by this same definition)."""
    rng = row["high"] - row["low"]
    if rng <= 0:
        return False
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    return (upper_wick / rng) >= wick_min_pct and (body / rng) <= body_max_pct


def is_bullish_engulfing(prev: pd.Series, cur: pd.Series) -> bool:
    return (
        prev["close"] < prev["open"]
        and cur["close"] > cur["open"]
        and cur["close"] >= prev["open"]
        and cur["open"] <= prev["close"]
    )


def is_bearish_engulfing(prev: pd.Series, cur: pd.Series) -> bool:
    return (
        prev["close"] > prev["open"]
        and cur["close"] < cur["open"]
        and cur["close"] <= prev["open"]
        and cur["open"] >= prev["close"]
    )


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range — used as a volatility yardstick for "how decisive
    is this move" signal-quality filters (a breakout that only just pokes
    through a level isn't the same conviction as one that clears it by a
    multiple of the recent typical range)."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
