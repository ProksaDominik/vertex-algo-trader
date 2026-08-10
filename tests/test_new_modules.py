"""Signal-logic tests for the 4 CLAUDE.md strategy "modules" (A/B/C/D):
Asian Range Fade, London Breakout, Liquidity Sweep (reuses the existing
SweepReclaimStrategy directly -- see portfolio_runner.STRATEGY_CLASSES'
"LiquiditySweep" entry, already covered by test_phase8.py), and H4
Pullback. docs/strategy/FOREX_STRATEGY_STACK_SPEC.md (the actual spec
these module names are drawn from) was not available when these were
written -- only the names and a few filter hints -- so these are this
project's own interpretation of each module, verified against real
on_bar output rather than assumed correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.strategies_asian_fade import AsianFadeStrategy  # noqa: E402
from metatraderbot.strategies_h4_pullback import H4PullbackStrategy  # noqa: E402
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy  # noqa: E402
from metatraderbot.strategy import Direction  # noqa: E402

PAD = 24  # bars entirely on the PREVIOUS UTC day, just to clear the warmup gate
ASIAN_PRICES = [1.1005, 1.1008, 1.1012, 1.1015, 1.1010, 1.1018, 1.1020,
                 1.1016, 1.1012, 1.1008, 1.1005, 1.1010, 1.1014, 1.1009]


def _base_cfg(strategy_key, params, tp=2.0):
    return {
        "symbol": "EURUSD",
        "capital": {"risk_usd_per_trade": 100.0},
        "strategy": {"exit": {"tp_r_multiple": tp}, strategy_key: params},
    }


def _mk_bars(rows, start="2026-01-04 12:00"):
    ts = pd.date_range(start, periods=len(rows), freq="30min", tz="UTC")
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "timestamp"
    return df


def _pad_and_session_rows():
    """PAD bars on 01-04, then a 14-bar Asian session (00:00-06:30 UTC on
    01-05) with a real ~21-pip range. Returns (rows, session_low, session_high)."""
    rows = [dict(open=1.0990, high=1.0995, low=1.0985, close=1.0992) for _ in range(PAD)]
    for p in ASIAN_PRICES:
        rows.append(dict(open=p - 0.0002, high=p + 0.0003, low=p - 0.0003, close=p))
    session_low = min(r["low"] for r in rows[PAD:PAD + 14])
    session_high = max(r["high"] for r in rows[PAD:PAD + 14])
    return rows, session_low, session_high


# --------------------------------------------------------------------------
# Module A: Asian Range Fade
# --------------------------------------------------------------------------

def _asian_fade_cfg():
    return _base_cfg("asian_fade", {
        "session_start_hour_utc": 0, "session_end_hour_utc": 7,
        "fade_start_hour_utc": 7, "fade_end_hour_utc": 16,
        "min_range_pips": 15.0, "sl_buffer_pips": 3.0, "lookback_bars": 20,
    })


def test_asian_fade_long_on_wick_below_session_low_that_reclaims():
    strat = AsianFadeStrategy(_asian_fade_cfg())
    rows, session_low, session_high = _pad_and_session_rows()
    for _ in range(3):
        rows.append(dict(open=1.1010, high=1.1013, low=1.1007, close=1.1011))
    fade_close = session_low + 0.0003  # wicks below, closes back inside
    rows.append(dict(open=session_low - 0.0002, high=session_low + 0.0004,
                      low=session_low - 0.0008, close=fade_close))
    bars = _mk_bars(rows)

    trades = strat.on_bar(len(rows) - 1, bars)
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0


def test_asian_fade_skips_a_too_small_range():
    """A quiet/holiday Asian session with a tiny range isn't a real level
    -- min_range_pips must gate it out even if the wick+reclaim shape
    would otherwise qualify."""
    cfg = _asian_fade_cfg()
    cfg["strategy"]["asian_fade"]["min_range_pips"] = 50.0  # bigger than the ~21-pip test range
    strat = AsianFadeStrategy(cfg)
    rows, session_low, session_high = _pad_and_session_rows()
    for _ in range(3):
        rows.append(dict(open=1.1010, high=1.1013, low=1.1007, close=1.1011))
    fade_close = session_low + 0.0003
    rows.append(dict(open=session_low - 0.0002, high=session_low + 0.0004,
                      low=session_low - 0.0008, close=fade_close))
    bars = _mk_bars(rows)

    trades = strat.on_bar(len(rows) - 1, bars)
    assert trades == []


# --------------------------------------------------------------------------
# Module B: London Breakout
# --------------------------------------------------------------------------

def _london_breakout_cfg():
    return _base_cfg("london_breakout", {
        "session_start_hour_utc": 0, "session_end_hour_utc": 7,
        "breakout_start_hour_utc": 7, "breakout_end_hour_utc": 10,
        "min_clearance_pips": 2.0, "sl_buffer_pips": 3.0, "min_d1_atr_pips": 0.0,
        "lookback_bars": 20,
    })


def test_london_breakout_long_on_decisive_close_above_session_high():
    strat = LondonBreakoutStrategy(_london_breakout_cfg())
    rows, session_low, session_high = _pad_and_session_rows()
    for _ in range(3):
        rows.append(dict(open=1.1010, high=1.1013, low=1.1007, close=1.1011))
    breakout_close = session_high + 0.0006  # clears high + min_clearance
    rows.append(dict(open=session_high + 0.0001, high=breakout_close + 0.0002,
                      low=session_high - 0.0001, close=breakout_close))
    bars = _mk_bars(rows)

    trades = strat.on_bar(len(rows) - 1, bars)
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0


def test_london_breakout_ignores_a_wick_only_poke():
    """A bar that WICKS above the session high but closes back inside must
    not trigger -- only a body close beyond the level counts."""
    strat = LondonBreakoutStrategy(_london_breakout_cfg())
    rows, session_low, session_high = _pad_and_session_rows()
    for _ in range(3):
        rows.append(dict(open=1.1010, high=1.1013, low=1.1007, close=1.1011))
    # High pokes well above the session high, but close stays inside it.
    rows.append(dict(open=1.1011, high=session_high + 0.0010,
                      low=1.1009, close=session_high - 0.0002))
    bars = _mk_bars(rows)

    trades = strat.on_bar(len(rows) - 1, bars)
    assert trades == []


# --------------------------------------------------------------------------
# Module D: H4 Pullback
# --------------------------------------------------------------------------

def _h4_pullback_cfg():
    return _base_cfg("h4_pullback", {
        "htf_multiplier": 2, "htf_lookback": 3, "fast_period": 5, "slow_period": 10,
        "pullback_tolerance_pips": 3.0, "sl_swing_lookback": 3,
        "confirmation_min_body_pct": 0.4,
    })


def _h4_pullback_bars():
    closes = [1.1000] * 15
    for i in range(6):
        closes.append(1.1000 + 0.0015 * (i + 1))  # rally establishes an up H4 bias
    closes.append(1.108)   # pullback/confirmation bar
    closes.append(1.107)   # filler
    rows = []
    for i, c in enumerate(closes):
        if i == 21:
            rows.append(dict(open=1.1066, high=1.1082, low=1.1065, close=1.1080))
        else:
            rows.append(dict(open=c - 0.0001, high=c + 0.0002, low=c - 0.0002, close=c))
    return _mk_bars(rows, start="2026-01-05 00:00")


def test_h4_pullback_long_entry_when_htf_up_and_price_tags_fast_ema():
    strat = H4PullbackStrategy(_h4_pullback_cfg())
    bars = _h4_pullback_bars()

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


def test_h4_pullback_no_trade_against_the_htf_bias():
    """The same pullback-to-EMA shape must NOT fire as a SHORT while the
    H4 bias is still up -- module D only trades WITH the higher-timeframe
    trend, never against it."""
    strat = H4PullbackStrategy(_h4_pullback_cfg())
    bars = _h4_pullback_bars()

    for i in range(14, len(bars)):
        trades = strat.on_bar(i, bars)
        for spec in trades:
            assert spec.direction != Direction.SHORT
