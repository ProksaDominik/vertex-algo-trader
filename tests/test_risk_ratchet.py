"""CLAUDE.md's non-negotiable risk ratchet (spec §3.3) -- values are copied
verbatim from that spec, so these tests pin the exact thresholds/outputs
rather than re-deriving them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.risk_ratchet import current_risk_pct  # noqa: E402


def test_base_risk_at_breakeven():
    assert current_risk_pct(10_000.0, 10_000.0) == 0.0050


def test_shrinks_approaching_target():
    assert current_risk_pct(10_500.0, 10_000.0) == 0.0040  # +5%
    assert current_risk_pct(10_800.0, 10_000.0) == 0.0025  # +8%
    assert current_risk_pct(11_000.0, 10_000.0) == 0.0025  # beyond +8% still protects


def test_shrinks_in_drawdown():
    assert current_risk_pct(9_800.0, 10_000.0) == 0.0040   # -2%
    assert current_risk_pct(9_600.0, 10_000.0) == 0.0025   # -4%
    assert current_risk_pct(9_400.0, 10_000.0) == 0.0015   # -6%, floor
    assert current_risk_pct(8_000.0, 10_000.0) == 0.0015   # deep DD still floors, never goes negative/0


def test_boundaries_are_inclusive_ge():
    # p >= 0.08 uses the 0.08 bucket exactly at the boundary
    assert current_risk_pct(10_800.0, 10_000.0) == 0.0025
    assert current_risk_pct(10_799.0, 10_000.0) == 0.0040
    assert current_risk_pct(10_500.0, 10_000.0) == 0.0040
    assert current_risk_pct(10_499.0, 10_000.0) == 0.0050


def test_never_returns_a_higher_risk_than_base_no_matter_the_profit():
    """The whole point of the ratchet: risk NEVER increases to chase
    anything, in either direction away from breakeven."""
    for equity in [10_000, 10_300, 10_600, 10_900, 11_500, 15_000]:
        assert current_risk_pct(equity, 10_000.0) <= 0.0050


def test_zero_initial_is_safe():
    assert current_risk_pct(100.0, 0.0) == 0.0
