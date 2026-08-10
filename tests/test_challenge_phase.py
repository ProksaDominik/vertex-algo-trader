"""FTMO phase tracking: target reached vs actually PASSED."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.challenge_phase import evaluate, next_phase  # noqa: E402

ACCOUNT = 10_000.0


def test_below_target_reports_progress():
    s = evaluate("challenge", profit_usd=400.0, account_size=ACCOUNT, trading_days=2)
    assert not s.target_reached and not s.passed and not s.banked
    assert s.target_usd == 1_000.0
    assert "4.00% of 10.0%" in s.headline
    assert "600" in s.detail  # remaining profit needed


def test_target_hit_but_not_enough_days_is_NOT_a_pass():
    """The trap this module exists to prevent: +10% on day one passes nothing."""
    s = evaluate("challenge", profit_usd=1_200.0, account_size=ACCOUNT, trading_days=1)
    assert s.target_reached is True
    assert s.passed is False, "4 trading days are still required"
    assert s.banked is True, "risk must be cut in this window"
    assert "3 more trading day" in s.headline


def test_target_and_days_is_a_pass():
    s = evaluate("challenge", profit_usd=1_050.0, account_size=ACCOUNT, trading_days=4)
    assert s.passed is True
    assert s.banked is False
    assert "PASSED" in s.headline
    assert "Verification" in s.detail


def test_verification_target_is_five_percent():
    below = evaluate("verification", profit_usd=400.0, account_size=ACCOUNT, trading_days=5)
    assert below.target_usd == 500.0 and not below.passed
    ok = evaluate("verification", profit_usd=520.0, account_size=ACCOUNT, trading_days=5)
    assert ok.passed and "FUNDED" in ok.detail


def test_funded_phase_has_no_target_and_says_real_money():
    s = evaluate("funded", profit_usd=-250.0, account_size=ACCOUNT, trading_days=99)
    assert s.target_pct is None and s.target_usd is None
    assert s.passed is False and s.banked is False
    assert "REAL MONEY" in s.headline.upper()
    assert "10% overall" in s.detail


def test_losing_book_never_looks_passed():
    s = evaluate("challenge", profit_usd=-800.0, account_size=ACCOUNT, trading_days=9)
    assert not s.target_reached and not s.passed and not s.banked
    assert s.profit_pct == -8.0


def test_phase_progression():
    assert next_phase("challenge") == "verification"
    assert next_phase("verification") == "funded"
    assert next_phase("funded") is None


def test_unknown_phase_defaults_to_challenge():
    s = evaluate("", profit_usd=0.0, account_size=ACCOUNT, trading_days=0)
    assert s.phase == "challenge" and s.target_pct == 10.0
