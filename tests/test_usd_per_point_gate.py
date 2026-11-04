"""usd_per_point() must not size off a broker figure the frozen table disputes.

THE INCIDENT THIS PINS (2026-08-28)
-----------------------------------
The broker reported trade_tick_value=0.1 / trade_tick_size=0.01 for XAUUSD,
so usd_per_point() returned 10.0 USD per price unit per lot. Live P&L on four
open positions proved the true figure was 100.0 -- a -1350.30 loss over a 192.90
move on 0.07 lots is exactly 100/point/lot. Gold was therefore sized TEN TIMES
too large: ~4,722 USD of risk per position where 500 was intended, ~19,188 USD
in total against a 2,000 USD daily loss limit.

tests/test_sizing_safety.py already covered this class of bug -- but only inside
PortfolioRunner. run_lucidpro_funded.py calls engine.usd_per_point() directly:

    mult = engine.usd_per_point(sym) or contract_mult(sym)

and because 10.0 is truthy, the correct contract_mult fallback never ran. The
gate existed one layer above the code that actually needed it, so the tests
passed while real money was mis-sized. These tests pin the gate at the engine
layer, where every caller inherits it.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import mt5_engine as eng_mod  # noqa: E402


class _Info:
    def __init__(self, tick_value, tick_size):
        self.trade_tick_value = tick_value
        self.trade_tick_size = tick_size


@pytest.fixture
def engine(monkeypatch):
    e = eng_mod.MT5Engine.__new__(eng_mod.MT5Engine)
    monkeypatch.setattr(e, "resolve_symbol", lambda s: s, raising=False)
    return e


def _patch_info(monkeypatch, info):
    monkeypatch.setattr(eng_mod, "mt5",
                        types.SimpleNamespace(symbol_info=lambda s: info),
                        raising=False)


def test_gold_broker_understatement_is_corrected(engine, monkeypatch):
    """The exact live failure: broker says 10, frozen table says 100."""
    _patch_info(monkeypatch, _Info(0.1, 0.01))          # -> 10.0
    got = engine.usd_per_point("XAUUSD")
    assert got == pytest.approx(100.0), (
        "must not size gold off the broker's 10.0 when contract_mult says 100.0")


def test_disagreement_always_resolves_to_the_smaller_position(engine, monkeypatch):
    """A LARGER multiplier means a SMALLER position. On drift, take the larger.

    Direction matters: picking the smaller multiplier would double down on the
    original bug. Under-sizing costs a little profit, over-sizing loses accounts.
    """
    _patch_info(monkeypatch, _Info(100.0, 0.01))        # -> 10_000, way above 100
    got = engine.usd_per_point("XAUUSD")
    assert got == pytest.approx(10_000.0)


def test_agreeing_sources_pass_through_untouched(engine, monkeypatch):
    """No gate when the two agree -- the live value must stay authoritative."""
    _patch_info(monkeypatch, _Info(1.0, 0.01))          # -> 100.0, matches frozen
    assert engine.usd_per_point("XAUUSD") == pytest.approx(100.0)


def test_minor_drift_keeps_the_live_value(engine, monkeypatch):
    """Small differences are real (contract sizes vary); only material drift fires."""
    _patch_info(monkeypatch, _Info(1.2, 0.01))          # -> 120 vs 100, 1.2x
    assert engine.usd_per_point("XAUUSD") == pytest.approx(120.0)


def test_unusable_broker_specs_still_return_none(engine, monkeypatch):
    """None lets the caller fall back explicitly rather than size off garbage."""
    _patch_info(monkeypatch, _Info(0.0, 0.01))
    assert engine.usd_per_point("XAUUSD") is None
    _patch_info(monkeypatch, None)
    assert engine.usd_per_point("XAUUSD") is None


def test_fx_is_not_disturbed_by_the_gate(engine, monkeypatch):
    """EURUSD's true figure is 100_000/point/lot; the gate must not clamp it."""
    _patch_info(monkeypatch, _Info(1.0, 0.00001))       # -> 100_000
    got = engine.usd_per_point("EURUSD")
    assert got is not None and got >= 1000.0, (
        "gate must not drag a legitimately large FX multiplier down")
