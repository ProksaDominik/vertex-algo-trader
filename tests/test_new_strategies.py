"""Signal-logic tests for the 3 strategies added to replace the roster:
Trend Pullback (BOS Continuation), S/R Breakout + Retest, and 20/50 EMA
Crossover. Each test hand-builds bars that satisfy the strategy's own entry
conditions (verified against the real on_bar output, not assumed) and checks
that the emitted TradeSpec has a sane direction/SL/TP/lots relationship.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.strategies_bos_pullback import TrendPullbackBOSStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_sr_retest import SRBreakoutRetestStrategy  # noqa: E402
from metatraderbot.strategy import Direction  # noqa: E402

TS = pd.date_range("2026-01-01", periods=200, freq="15min", tz="UTC")


def _cfg(symbol, exit_params, strategy_key, strategy_params):
    return {
        "symbol": symbol,
        "capital": {"risk_usd_per_trade": 100.0},
        "strategy": {"exit": exit_params, strategy_key: strategy_params},
    }


def _mk_bars(rows):
    df = pd.DataFrame(rows, index=TS[: len(rows)])
    df.index.name = "timestamp"
    return df


# --------------------------------------------------------------------------
# Strategy 1: Trend Pullback (BOS Continuation)
# --------------------------------------------------------------------------

def test_bos_pullback_long_entry_on_fib_zone_confirmation():
    cfg = _cfg(
        "EURUSD", {"tp_r_multiple": 2.0}, "bos_pullback",
        {"htf_multiplier": 2, "htf_lookback": 3, "swing_lookback": 6,
         "fib_low": 0.5, "fib_high": 0.618, "sl_buffer_pips": 4.0,
         "max_wait_bars": 10},
    )
    strat = TrendPullbackBOSStrategy(cfg)

    # 2 padding bars (outside the swing window) + 6 bars forming 3 rising
    # HTF buckets (htf_multiplier=2) -> HTF trend reads "up". These 6 bars
    # also double as the swing_lookback window for the BOS bar that follows.
    rows = [
        dict(open=1.0990, high=1.0995, low=1.0985, close=1.0992),
        dict(open=1.0992, high=1.0998, low=1.0988, close=1.0995),
        dict(open=1.1000, high=1.1010, low=1.0995, close=1.1008),
        dict(open=1.1008, high=1.1012, low=1.1000, close=1.1010),
        dict(open=1.1010, high=1.1022, low=1.1005, close=1.1020),
        dict(open=1.1018, high=1.1025, low=1.1010, close=1.1023),
        dict(open=1.1020, high=1.1035, low=1.1015, close=1.1033),
        dict(open=1.1030, high=1.1040, low=1.1020, close=1.1038),
    ]
    prior_swing_high = max(r["high"] for r in rows[2:])
    # BOS bar: body close well above the prior swing high (real impulse leg).
    bos_close = prior_swing_high + 0.0040
    rows.append(dict(open=prior_swing_high + 0.0002, high=bos_close + 0.0002,
                      low=prior_swing_high, close=bos_close))
    bos_index = len(rows) - 1
    bars = _mk_bars(rows)

    assert strat._htf_trend(bars.iloc[: bos_index + 1]) == "up"
    trades = strat.on_bar(bos_index, bars)
    assert trades == []
    assert len(strat.pending) == 1  # BOS armed, waiting for pullback

    sig = strat.pending[0]
    zone_lo, zone_hi = strat._fib_zone(sig.swing_extreme, sig.bos_level)
    mid = (zone_lo + zone_hi) / 2

    # Pullback + confirmation bar: low dips into the fib zone, bullish
    # pinbar shape (long lower wick, small body near the top).
    body_top = mid + 0.0008
    rows.append(dict(open=body_top - 0.0001, high=body_top + 0.0001,
                      low=mid, close=body_top))
    confirm_index = len(rows) - 1
    bars = _mk_bars(rows)

    trades = strat.on_bar(confirm_index, bars)
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0
    assert strat.pending == []  # consumed


def test_bos_pullback_no_trade_without_confirmation_candle():
    """A pullback into the fib zone with a plain (non-pinbar/engulfing)
    candle must NOT trigger an entry -- confirmation is required, not just
    proximity to the zone."""
    cfg = _cfg(
        "EURUSD", {"tp_r_multiple": 2.0}, "bos_pullback",
        {"htf_multiplier": 2, "htf_lookback": 3, "swing_lookback": 6,
         "fib_low": 0.5, "fib_high": 0.618, "sl_buffer_pips": 4.0,
         "max_wait_bars": 10},
    )
    strat = TrendPullbackBOSStrategy(cfg)
    rows = [
        dict(open=1.0990, high=1.0995, low=1.0985, close=1.0992),
        dict(open=1.0992, high=1.0998, low=1.0988, close=1.0995),
        dict(open=1.1000, high=1.1010, low=1.0995, close=1.1008),
        dict(open=1.1008, high=1.1012, low=1.1000, close=1.1010),
        dict(open=1.1010, high=1.1022, low=1.1005, close=1.1020),
        dict(open=1.1018, high=1.1025, low=1.1010, close=1.1023),
        dict(open=1.1020, high=1.1035, low=1.1015, close=1.1033),
        dict(open=1.1030, high=1.1040, low=1.1020, close=1.1038),
    ]
    prior_swing_high = max(r["high"] for r in rows[2:])
    bos_close = prior_swing_high + 0.0040
    rows.append(dict(open=prior_swing_high + 0.0002, high=bos_close + 0.0002,
                      low=prior_swing_high, close=bos_close))
    bos_index = len(rows) - 1
    bars = _mk_bars(rows)
    strat.on_bar(bos_index, bars)
    sig = strat.pending[0]
    zone_lo, zone_hi = strat._fib_zone(sig.swing_extreme, sig.bos_level)
    mid = (zone_lo + zone_hi) / 2

    # Dips into the zone but is a plain small-range doji-ish bar, not a
    # pinbar (wick too short) or engulfing candle.
    rows.append(dict(open=mid, high=mid + 0.0003, low=mid - 0.0002, close=mid + 0.0001))
    confirm_index = len(rows) - 1
    bars = _mk_bars(rows)

    trades = strat.on_bar(confirm_index, bars)
    assert trades == []
    assert len(strat.pending) == 1  # still waiting


# --------------------------------------------------------------------------
# Strategy 2: Support & Resistance Breakout + Retest
# --------------------------------------------------------------------------

def _sr_cfg():
    return _cfg(
        "EURUSD", {"tp_r_multiple": 2.0}, "sr_retest",
        {"level_lookback": 10, "swing_arm": 2, "min_touches": 2,
         "retest_tolerance_pips": 5.0, "sl_buffer_pips": 7.0,
         "max_wait_bars": 30, "level_touch_tolerance_pips": 3.0},
    )


def _flat_bar(mid, spread=0.0006):
    return dict(open=mid - 0.0001, high=mid + 0.0001, low=mid - spread, close=mid)


def test_sr_retest_long_entry_after_resistance_breakout_and_reject():
    strat = SRBreakoutRetestStrategy(_sr_cfg())

    # 2 padding bars (bar_index must be >= level_lookback+swing_arm+2=14)
    # + 12 bars with 2 clustered swing highs at 1.1050 (positions 3 and 8,
    # each flanked by 2 lower bars so the arm=2 local-extreme check fires).
    pattern = [
        1.1005, 1.1008,
        1.1010, 1.1020, 1.1030, 1.1050, 1.1030, 1.1015,
        1.1020, 1.1035, 1.1050, 1.1030, 1.1015, 1.1010,
    ]
    rows = [_flat_bar(p) for p in pattern]

    touch_tol = 3.0 * 0.0001
    breakout_close = 1.1050 + 0.0001 + touch_tol + 0.0010
    rows.append(dict(open=1.1050, high=breakout_close + 0.0002,
                      low=1.1049, close=breakout_close))
    breakout_index = len(rows) - 1
    bars = _mk_bars(rows)

    trades = strat.on_bar(breakout_index, bars)
    assert trades == []
    assert len(strat.pending) == 1
    sig = strat.pending[0]
    assert sig.direction == Direction.LONG
    level = sig.level

    # Retest bar: low tags back down near the broken resistance (now
    # support), bullish pinbar rejection.
    tag_low = level + 0.0002
    body_top = tag_low + 0.0025
    rows.append(dict(open=body_top - 0.0002, high=body_top + 0.0001,
                      low=tag_low, close=body_top))
    retest_index = len(rows) - 1
    bars = _mk_bars(rows)

    trades = strat.on_bar(retest_index, bars)
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0


def test_sr_retest_no_trade_without_clustered_level():
    """A breakout with no repeated touches beforehand (no S/R level to
    break) must not arm a retest setup."""
    strat = SRBreakoutRetestStrategy(_sr_cfg())
    # Monotonically rising bars: no local extrema cluster at all.
    pattern = [1.1000 + i * 0.0006 for i in range(14)]
    rows = [_flat_bar(p) for p in pattern]
    rows.append(dict(open=1.1100, high=1.1130, low=1.1099, close=1.1125))
    bars = _mk_bars(rows)

    trades = strat.on_bar(len(rows) - 1, bars)
    assert trades == []
    assert strat.pending == []


# --------------------------------------------------------------------------
# Strategy 3: Dual 20/50 EMA Crossover (pullback entry)
# --------------------------------------------------------------------------

def _ema_cfg():
    return _cfg(
        "EURUSD", {"tp_r_multiple": 2.0}, "ema_cross",
        {"fast_period": 5, "slow_period": 10, "pullback_tolerance_pips": 3.0,
         "sl_swing_lookback": 3, "max_wait_bars": 15,
         "confirmation_min_body_pct": 0.4},
    )


def test_ema_crossover_long_entry_on_pullback_to_fast_ema():
    strat = EMACrossoverStrategy(_ema_cfg())

    closes = [1.1000] * 15
    for i in range(6):
        closes.append(1.1000 + 0.0015 * (i + 1))  # rally: fast EMA crosses above slow
    closes.append(1.108)  # pullback/confirmation bar
    closes.append(1.107)  # filler

    rows = []
    for i, c in enumerate(closes):
        if i == 21:
            rows.append(dict(open=1.1066, high=1.1082, low=1.1065, close=1.1080))
        else:
            rows.append(dict(open=c - 0.0001, high=c + 0.0002, low=c - 0.0002, close=c))
    bars = _mk_bars(rows)

    trades = []
    for i in range(14, len(bars)):
        trades = strat.on_bar(i, bars)
        if trades:
            break

    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0


def test_ema_crossover_invalidated_by_recross_before_pullback():
    """If price re-crosses back below the slow EMA before ever tagging the
    fast EMA, the LONG pending setup must be dropped (not carried forward
    stale) -- the sharp reversal here also arms a fresh bearish setup,
    which is the correct behavior, but it must be a NEW short pending
    (bars_since_cross reset), never the invalidated long one surviving."""
    strat = EMACrossoverStrategy(_ema_cfg())

    closes = [1.1000] * 15
    for i in range(6):
        closes.append(1.1000 + 0.0015 * (i + 1))  # bull cross at bar 15
    # Immediately dump back below the flat base -- alignment break.
    closes += [1.0950, 1.0900, 1.0850]

    rows = [dict(open=c - 0.0001, high=c + 0.0002, low=c - 0.0002, close=c) for c in closes]
    bars = _mk_bars(rows)

    for i in range(14, len(bars)):
        strat.on_bar(i, bars)

    assert strat.pending == [] or strat.pending[0].direction == Direction.SHORT
