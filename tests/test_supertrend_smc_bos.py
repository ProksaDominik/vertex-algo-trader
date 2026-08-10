"""Signal-logic tests for the two Phase 11 roster strategies: Supertrend
and SMC_BOS (liquidity sweep + break of structure + order block). Verified
against real on_bar output, not assumed correct -- same discipline as
test_new_modules.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402
from metatraderbot.strategies_smc_bos import SMCBOSStrategy  # noqa: E402
from metatraderbot.strategy import Direction  # noqa: E402


def _base_cfg(strategy_key, params, tp=2.0):
    return {
        "symbol": "EURUSD",
        "capital": {"risk_usd_per_trade": 100.0},
        "strategy": {"exit": {"tp_r_multiple": tp}, strategy_key: params},
    }


def _mk_bars(rows, start="2026-01-05 00:00"):
    ts = pd.date_range(start, periods=len(rows), freq="30min", tz="UTC")
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "timestamp"
    return df


# --------------------------------------------------------------------------
# Supertrend
# --------------------------------------------------------------------------

def _supertrend_cfg(period=5, multiplier=2.0):
    return _base_cfg("supertrend", {"period": period, "multiplier": multiplier})


def _supertrend_flip_up_bars():
    """A decisive DOWNTREND first (establishes direction=-1 -- the
    Supertrend line sits above price), then a sharp reversal rally that
    closes back through the (downtrend) upper band -- forces a genuine
    flip to uptrend partway through the rally."""
    rows = []
    price = 1.1200
    for _ in range(12):
        price -= 0.0060
        rows.append(dict(open=price + 0.0060, high=price + 0.0065, low=price - 0.0010, close=price))
    for _ in range(10):
        price += 0.0060
        rows.append(dict(open=price - 0.0060, high=price + 0.0010, low=price - 0.0065, close=price))
    return _mk_bars(rows)


def test_supertrend_long_on_flip_to_uptrend():
    strat = SupertrendStrategy(_supertrend_cfg())
    bars = _supertrend_flip_up_bars()

    trades = []
    fired_at = None
    for i in range(10, len(bars)):
        trades = strat.on_bar(i, bars)
        if trades:
            fired_at = i
            break

    assert trades, "expected a flip signal somewhere in the rally"
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    assert spec.lots > 0

    # Not still flipping every subsequent bar -- direction should hold once
    # trend is established, so the very next bar (still rallying) must NOT
    # re-fire (only the flip bar itself trades).
    if fired_at is not None and fired_at + 1 < len(bars):
        next_trades = strat.on_bar(fired_at + 1, bars)
        assert next_trades == []


def test_supertrend_no_signal_without_enough_bars():
    strat = SupertrendStrategy(_supertrend_cfg(period=10))
    bars = _mk_bars([dict(open=1.1, high=1.101, low=1.099, close=1.1) for _ in range(5)])
    assert strat.on_bar(4, bars) == []


# --------------------------------------------------------------------------
# SMC_BOS: liquidity sweep -> break of structure -> order block retest
# --------------------------------------------------------------------------

def _smc_bos_cfg():
    return _base_cfg("smc_bos", {
        "swing_lookback": 10, "sweep_buffer_pips": 1.0, "bos_lookback": 5,
        "max_wait_bars": 10, "sl_buffer_pips": 3.0,
    }, tp=1.5)


def test_smc_bos_short_after_sweep_high_then_bearish_bos_then_ob_retest():
    strat = SMCBOSStrategy(_smc_bos_cfg())
    rows = []
    # 1) build a ranging swing high/low so a later wick can "sweep" it
    for _ in range(12):
        rows.append(dict(open=1.1000, high=1.1015, low=1.0985, close=1.1000))
    swing_high = 1.1015

    # 2) a BULLISH candle (this becomes the order block) right before the
    #    down-impulse that creates the BOS
    ob_low, ob_high = 1.0998, 1.1006
    rows.append(dict(open=ob_low, high=ob_high + 0.0002, low=ob_low - 0.0002, close=ob_high))

    # 3) the sweep bar: wicks above the swing high, closes back below it
    rows.append(dict(open=1.1002, high=swing_high + 0.0006, low=1.0996, close=1.1000))

    # 4) impulsive down move that closes below the nearer swing low ->
    #    bearish BOS confirmation
    near_low = min(r["low"] for r in rows[-6:])
    rows.append(dict(open=1.0995, high=1.0997, low=1.0965, close=near_low - 0.0010))

    bars = _mk_bars(rows)
    n = len(rows)

    # Feed the sweep + BOS bars through on_bar to build the pending order block.
    for i in range(11, n):
        strat.on_bar(i, bars)

    # 5) now price retraces back UP into the order block's body range.
    retest_price = (ob_low + ob_high) / 2
    bars2 = pd.concat([bars, _mk_bars(
        [dict(open=retest_price - 0.0002, high=retest_price + 0.0003,
              low=retest_price - 0.0003, close=retest_price)],
        start=bars.index[-1] + pd.Timedelta(minutes=30),
    )])

    trades = strat.on_bar(n, bars2)
    assert len(trades) == 1
    spec = trades[0]
    assert spec.direction == Direction.SHORT
    assert spec.take_profit < spec.entry_price < spec.stop_loss
    assert spec.lots > 0


def test_smc_bos_no_signal_without_a_sweep():
    """A plain range with no wick-beyond-swing sweep must never arm a
    pending order block, regardless of later price action."""
    strat = SMCBOSStrategy(_smc_bos_cfg())
    rows = [dict(open=1.1000, high=1.1010, low=1.0990, close=1.1000) for _ in range(25)]
    bars = _mk_bars(rows)
    for i in range(11, len(rows)):
        trades = strat.on_bar(i, bars)
        assert trades == []
    assert strat.pending == []
