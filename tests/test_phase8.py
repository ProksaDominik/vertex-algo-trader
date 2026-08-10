"""Phase 8 tests — per-strategy config expansion, per-slot exit overrides,
and the three new strategies (SweepReclaim, TODReversion, VWAPReversion).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.portfolio_runner import expand_configs, validate_portfolio_cfg  # noqa: E402
from metatraderbot.strategies_sweep_reclaim import SweepReclaimStrategy  # noqa: E402
from metatraderbot.strategies_tod_reversion import TODReversionStrategy  # noqa: E402
from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy  # noqa: E402
from metatraderbot.strategy import Direction  # noqa: E402


# --- config expansion ----------------------------------------------------

def test_expand_symbols_into_legs():
    cfg = [{"name": "Donchian", "strategy": "DonchianBreakout", "timeframe": "30m",
            "magic_base": 2100, "symbols": ["XAUUSD", "US30", "EURUSD"],
            "strategy_params": {"exit": {"tp_r_multiple": 2.0}}}]
    legs = expand_configs(cfg)
    assert [l["name"] for l in legs] == ["Donchian__XAUUSD", "Donchian__US30", "Donchian__EURUSD"]
    assert [l["magic_number"] for l in legs] == [2100, 2101, 2102]
    assert all(l["strategy_group"] == "Donchian" for l in legs)
    assert [l["symbol"] for l in legs] == ["XAUUSD", "US30", "EURUSD"]


def test_expand_legacy_single_symbol_passthrough():
    cfg = [{"name": "old", "strategy": "DonchianBreakout", "timeframe": "30m",
            "symbol": "EURUSD", "magic_number": 9, "strategy_params": {"exit": {"tp_r_multiple": 2.0}}}]
    legs = expand_configs(cfg)
    assert len(legs) == 1 and legs[0]["symbol"] == "EURUSD" and legs[0]["magic_number"] == 9
    assert legs[0]["strategy_group"] == "old"


def test_symbols_without_magic_base_rejected():
    cfg = {"capital": {"risk_usd_per_trade": 100, "daily_stop_loss_usd": -1,
                       "max_total_drawdown_usd": -1, "max_positions_per_config": 1},
           "configs": [{"name": "Donchian", "strategy": "DonchianBreakout", "timeframe": "30m",
                        "symbols": ["EURUSD"], "strategy_params": {"exit": {"tp_r_multiple": 2.0}}}]}
    assert any("magic_base" in e for e in validate_portfolio_cfg(cfg))


def test_expanded_magic_collision_detected():
    cfg = {"capital": {"risk_usd_per_trade": 100, "daily_stop_loss_usd": -1,
                       "max_total_drawdown_usd": -1, "max_positions_per_config": 1},
           "configs": [
               {"name": "A", "strategy": "DonchianBreakout", "timeframe": "30m",
                "magic_base": 100, "symbols": ["EURUSD", "US30"],
                "strategy_params": {"exit": {"tp_r_multiple": 2.0}}},
               {"name": "B", "strategy": "MACDTrend", "timeframe": "30m",
                "magic_base": 101, "symbols": ["GBPUSD"],
                "strategy_params": {"exit": {"tp_r_multiple": 2.0}}}]}
    # A -> 100,101 ; B -> 101  => collision on 101
    assert any("already used" in e for e in validate_portfolio_cfg(cfg))


def test_reversion_exit_override_passes_validation():
    cfg = {"capital": {"risk_usd_per_trade": 100, "daily_stop_loss_usd": -400,
                       "max_total_drawdown_usd": -800, "max_positions_per_config": 2,
                       "trail_start_r": 1.0},
           "configs": [{"name": "VWAPReversion", "strategy": "VWAPReversion", "timeframe": "30m",
                        "magic_base": 2400, "symbols": ["EURUSD"],
                        "strategy_params": {"vwap": {"band_atr": 2.0},
                                            "exit": {"tp_r_multiple": 1.0, "trail_start_r": 0.0}}}]}
    # tp 1.0 with the leg's own trail 0.0 must be OK even though book trail is 1.0.
    assert validate_portfolio_cfg(cfg) == []


@pytest.mark.parametrize("name", ["portfolio_ftmo.yaml", "portfolio_yolo.yaml"])
def test_shipped_per_strategy_configs_valid(name):
    yaml = pytest.importorskip("yaml")
    with open(ROOT / "config" / name) as f:
        cfg = yaml.safe_load(f)
    assert validate_portfolio_cfg(cfg) == []


# --- new strategies fire the right way -----------------------------------

def _bars(rows, start="2026-07-21 00:00"):
    ts = pd.date_range(start, periods=len(rows), freq="30min", tz="UTC")
    df = pd.DataFrame(rows)
    df["timestamp"] = ts
    return df


def _cfg(key, params, tp=1.0):
    strat = {k: {} for k in ["trend", "signal", "entry", "bollinger", "donchian",
                             "orb", "macd", "bos", "sweep", "tod", "vwap"]}
    strat["session"] = None
    strat["exit"] = {"tp_r_multiple": tp}
    strat[key] = params
    return {"symbol": "EURUSD", "capital": {"risk_usd_per_trade": 100}, "strategy": strat}


def test_sweep_reclaim_fires_long_on_reclaim():
    # 40 flat bars around 1.10, then a bar that pierces below the swing low
    # but closes back above it -> bullish sweep + reclaim.
    rows = [{"open": 1.10, "high": 1.101, "low": 1.099, "close": 1.10, "volume": 100} for _ in range(40)]
    rows.append({"open": 1.10, "high": 1.101, "low": 1.090, "close": 1.1005, "volume": 100})  # sweep+reclaim
    bars = _bars(rows)
    strat = SweepReclaimStrategy(_cfg("sweep", {"lookback": 20, "sl_buffer_atr": 0.25}))
    specs = strat.on_bar(len(bars) - 1, bars)
    assert specs and specs[0].direction == Direction.LONG
    assert specs[0].stop_loss < specs[0].entry_price < specs[0].take_profit


def test_vwap_reversion_fires_when_stretched_below():
    # Build a day where price drifts far below VWAP then prints an up bar.
    rows = []
    px = 1.10
    for i in range(30):
        rows.append({"open": px, "high": px + 0.0005, "low": px - 0.0005, "close": px, "volume": 100})
        px -= 0.0006  # steady drift down builds a VWAP well above price
    # final bar: still below vwap, closes UP (rejection)
    rows.append({"open": px, "high": px + 0.0015, "low": px - 0.0003, "close": px + 0.0012, "volume": 100})
    bars = _bars(rows)
    strat = VWAPReversionStrategy(_cfg("vwap", {"band_atr": 1.0}))
    specs = strat.on_bar(len(bars) - 1, bars)
    assert specs and specs[0].direction == Direction.LONG


def test_tod_reversion_fires_only_at_target_hour_when_stretched():
    # Day opens at 1.10, runs UP hard; at hour 20 UTC it should fade short.
    rows = []
    px = 1.10
    for i in range(48):
        rows.append({"open": px if i else 1.10, "high": px + 0.003, "low": px - 0.001,
                     "close": px, "volume": 100})
        px += 0.0015
    bars = _bars(rows, start="2026-07-21 00:00")   # 48 x 30m = full day; hour 20 = index 40
    strat = TODReversionStrategy(_cfg("tod", {"revert_hour_utc": 20, "stretch_atr": 1.0}))
    fired = [t for i in range(len(bars)) for t in strat.on_bar(i, bars)]
    assert fired and fired[0].direction == Direction.SHORT
    assert len(fired) == 1  # one per day
