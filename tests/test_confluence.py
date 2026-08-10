"""Confluence must fire ONLY when its legs agree.

This is the roster that measured best (39.1% 10-day FTMO pass vs 32.3%), and
its whole value is selectivity: RSI(14) and RSI(7) overlap on ~10% of signal
days, so agreement is genuinely rare. A bug that let it fire on a single leg
would silently turn it back into a plain RSI book at the wrong target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.strategies_confluence import ConfluenceStrategy  # noqa: E402
from metatraderbot.strategy import Direction, TradeSpec  # noqa: E402


def _spec(direction=Direction.LONG, entry=1.1000):
    return TradeSpec(direction=direction, entry_price=entry,
                     stop_loss=entry - 0.002, take_profit=entry + 0.004,
                     lots=0.1, signal_bar_index=1)


class FakeLeg:
    """Stands in for a registered strategy; returns whatever it is told to."""
    def __init__(self, out):
        self.out = out
        self.calls = 0

    def on_bar(self, bar_index, bars):
        self.calls += 1
        return list(self.out)


class BoomLeg:
    def on_bar(self, bar_index, bars):
        raise RuntimeError("leg exploded")


def _mk(legs):
    """Build a ConfluenceStrategy with pre-made legs, bypassing config wiring."""
    s = ConfluenceStrategy.__new__(ConfluenceStrategy)
    s.legs = legs
    s.require_same_direction = True
    return s


BARS = pd.DataFrame({"close": [1.0, 1.1]})


def test_fires_when_both_legs_agree():
    s = _mk([FakeLeg([_spec()]), FakeLeg([_spec()])])
    out = s.on_bar(1, BARS)
    assert len(out) == 1
    assert out[0].direction is Direction.LONG


def test_silent_when_one_leg_is_silent():
    s = _mk([FakeLeg([_spec()]), FakeLeg([])])
    assert s.on_bar(1, BARS) == []
    # and the reverse order
    s2 = _mk([FakeLeg([]), FakeLeg([_spec()])])
    assert s2.on_bar(1, BARS) == []


def test_silent_when_legs_disagree_on_direction():
    """Two mean-reversion signals pointing opposite ways is a disagreement,
    not confluence — picking either side would be arbitrary."""
    s = _mk([FakeLeg([_spec(Direction.LONG)]),
             FakeLeg([_spec(Direction.SHORT)])])
    assert s.on_bar(1, BARS) == []


def test_first_leg_owns_the_trade():
    """Levels must come from ONE leg. Blending stops across legs would create
    a trade neither strategy actually signalled."""
    a, b = _spec(entry=1.2000), _spec(entry=1.9999)
    s = _mk([FakeLeg([a]), FakeLeg([b])])
    out = s.on_bar(1, BARS)
    assert out[0].entry_price == 1.2000
    assert out[0].stop_loss == a.stop_loss


def test_a_broken_leg_does_not_take_the_book_down():
    s = _mk([FakeLeg([_spec()]), BoomLeg()])
    assert s.on_bar(1, BARS) == []      # no trade, no exception


def test_rejects_a_single_leg_config():
    cfg = {"strategy": {"confluence": {"legs": [{"strategy": "RSIReversion"}]},
                        "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10}},
           "capital": {"risk_usd_per_trade": 100.0},
           "symbol": "EURUSD"}
    with pytest.raises(ValueError, match="at least 2 legs"):
        ConfluenceStrategy(cfg)


def test_rejects_unknown_leg_strategy():
    cfg = {"strategy": {"confluence": {"legs": [{"strategy": "RSIReversion"},
                                                {"strategy": "NoSuchThing"}]},
                        "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10}},
           "capital": {"risk_usd_per_trade": 100.0},
           "symbol": "EURUSD"}
    with pytest.raises(ValueError, match="not a registered strategy"):
        ConfluenceStrategy(cfg)


def test_runner_passes_unlisted_strategy_params_through():
    """Regression: _build_strategy_config used a hardcoded key whitelist that
    silently DROPPED `rsi`, `confluence` and others.

    Consequence in production: a leg configured as RSI(7) 25/75 was built with
    RSIReversionStrategy's defaults (14, 30/70), making it an exact duplicate
    of the RSI(14) leg. Two supposedly independent legs fired the identical
    signal.
    """
    from metatraderbot.portfolio_runner import PortfolioRunner
    from metatraderbot.strategies_rsi import RSIReversionStrategy

    r = PortfolioRunner.__new__(PortfolioRunner)
    r.risk_usd = 100.0
    params = {
        "rsi": {"period": 7, "oversold": 25, "overbought": 75},
        "confluence": {"legs": [{"strategy": "RSIReversion"},
                                {"strategy": "RSIReversion"}]},
        "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10},
    }
    cfg = r._build_strategy_config("EURUSD", params)
    assert "rsi" in cfg["strategy"], "rsi params must survive"
    assert "confluence" in cfg["strategy"], "confluence params must survive"

    inst = RSIReversionStrategy(cfg)
    assert (inst.period, inst.oversold, inst.overbought) == (7, 25.0, 75.0)

    # keys the whitelist DOES name must still get their defaults
    assert cfg["strategy"]["trend"]["ema_period"] == 50
    assert cfg["strategy"]["exit"]["sl_swing_lookback"] == 10


def test_real_legs_build_from_config():
    """End-to-end: the documented config shape must actually construct."""
    cfg = {
        "symbol": "EURUSD",
        "capital": {"risk_usd_per_trade": 100.0},
        "strategy": {
            "confluence": {"legs": [
                {"strategy": "RSIReversion",
                 "rsi": {"period": 14, "oversold": 30, "overbought": 70}},
                {"strategy": "RSIReversion",
                 "rsi": {"period": 7, "oversold": 25, "overbought": 75}},
            ]},
            "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10},
        },
    }
    s = ConfluenceStrategy(cfg)
    assert len(s.legs) == 2
    # the legs must get their OWN rsi params, not share one
    assert s.legs[0].period == 14 and s.legs[1].period == 7
    assert s.legs[1].oversold == 25
