"""Phase 6 tests — trade-review learning, session learner, ledger reviews,
and the Market-Open BOS strategy. Pure Python, no MT5.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.strategies_bos import MarketOpenBOSStrategy, last_confirmed_swings  # noqa: E402
from metatraderbot.strategy import Direction  # noqa: E402
from metatraderbot.trade_review import (  # noqa: E402
    SessionLearner,
    classify_session,
    review_trade,
)


# --- session classification ---------------------------------------------

@pytest.mark.parametrize("hour,expected", [
    (0, "asian"), (3, "asian"), (7, "london"), (11, "london"),
    (12, "ny_overlap"), (16, "ny_overlap"), (17, "ny_pm"), (20, "ny_pm"),
    (21, "late"), (23, "late"),
])
def test_classify_session(hour, expected):
    assert classify_session(hour) == expected


# --- review_trade --------------------------------------------------------

def _row(**kw):
    base = dict(config_name="s1", symbol="EURUSD", direction="long",
                entry_price=1.1000, exit_price=1.1100, sl=1.0900, pnl_usd=100.0,
                exit_reason="tp",
                opened_ts=datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc).isoformat(),
                closed_ts=datetime(2026, 7, 21, 15, 30, tzinfo=timezone.utc).isoformat())
    base.update(kw); return base


def test_review_tp_win_geometry_and_session():
    rv = review_trade(_row(), risk_usd=100)
    assert rv.was_win is True
    assert rv.r_multiple == pytest.approx(1.0, abs=1e-9)   # 100 profit / 100 risk dist
    assert rv.session == "ny_overlap"                       # 14:00 UTC
    assert rv.hold_minutes == pytest.approx(90.0)
    assert "take-profit" in rv.why.lower()


def test_review_sl_loss_reasoning():
    rv = review_trade(_row(exit_price=1.0900, pnl_usd=-100.0, exit_reason="sl"), risk_usd=100)
    assert rv.was_win is False
    assert rv.r_multiple == pytest.approx(-1.0, abs=1e-9)
    assert "stopped out" in rv.why.lower()


def test_review_trailed_stop_in_profit_is_explained():
    rv = review_trade(_row(exit_price=1.1050, pnl_usd=50.0, exit_reason="sl"), risk_usd=100)
    assert rv.was_win is True
    assert "trailing stop locked" in rv.why.lower()


def test_review_falls_back_to_pnl_when_no_sl_distance():
    rv = review_trade(_row(sl=1.1000, pnl_usd=30.0), risk_usd=60)
    assert rv.r_multiple == pytest.approx(0.5)


# --- SessionLearner ------------------------------------------------------

def test_session_learner_flags_only_after_enough_losing_trades():
    lr = SessionLearner(min_trades=6, avoid_below_net_r=-3.0)
    for _ in range(5):
        lr.observe("s1", "asian", -1.0)
    assert not lr.should_avoid("s1", "asian")   # only 5 trades, not enough yet
    lr.observe("s1", "asian", -1.0)             # 6th -> net -6R
    assert lr.should_avoid("s1", "asian")
    # A different, profitable session is never avoided.
    for _ in range(10):
        lr.observe("s1", "london", +1.5)
    assert not lr.should_avoid("s1", "london")


def test_session_learner_verdicts_sorted_worst_first():
    lr = SessionLearner(min_trades=3, avoid_below_net_r=-2.0)
    for _ in range(4):
        lr.observe("s1", "asian", -1.0)
    for _ in range(4):
        lr.observe("s1", "london", +2.0)
    v = lr.verdicts("s1")
    assert v[0].session == "asian" and v[0].avoid is True
    assert v[-1].session == "london" and v[-1].avoid is False


# --- ledger review roundtrip --------------------------------------------

def test_ledger_review_roundtrip_and_learner_rebuild(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    rv = review_trade(_row(exit_price=1.0900, pnl_usd=-100.0, exit_reason="sl"), risk_usd=100)
    led.log_review(rv)
    led.log_review(review_trade(_row(), risk_usd=100))
    rows = led.all_reviews_for_learner()
    assert len(rows) == 2
    lr = SessionLearner()
    for cfg, sess, r in rows:
        lr.observe(cfg, sess, float(r))
    recent = led.recent_reviews(10)
    assert len(recent) == 2 and "why" in recent[0]
    led.close()


# --- BOS strategy --------------------------------------------------------

def test_last_confirmed_swings_basic():
    highs = [1, 2, 5, 2, 1, 2, 1]
    lows = [1, 0, 1, 1, 0.5, 1, 1]
    sh, sl = last_confirmed_swings(highs, lows, k=2)
    assert sh == 5.0    # pivot high at idx 2
    assert sl == 0.5    # pivot low at idx 4 (min of its window)


def _bos_bars(n=120, session_hour=13):
    """Flat structure, then a clean upside break during the NY-open window."""
    start = pd.Timestamp("2026-07-21 09:00", tz="UTC")
    ts = [start + timedelta(minutes=30 * i) for i in range(n)]
    rows = []
    for i in range(n):
        base = 100.0
        hi, lo, op, cl = base + 1, base - 1, base, base
        # Establish a swing high well before the open window.
        if i == 40:
            hi, cl = base + 3, base + 2
        rows.append({"timestamp": ts[i], "open": op, "high": hi, "low": lo, "close": cl, "volume": 100})
    df = pd.DataFrame(rows)
    # Force a breakout bar during the NY session (13:00-17:00 UTC).
    for i in range(len(df)):
        h = df.loc[i, "timestamp"].hour
        if h == 13 and i > 60:
            df.loc[i, ["high", "close"]] = [base_break(df), base_break(df)]
            break
    return df


def base_break(df):
    return float(df["high"].max()) + 2.0


def test_bos_fires_long_on_structure_break_in_session():
    cfg = {"symbol": "US30", "capital": {"risk_usd_per_trade": 100},
           "strategy": {"bos": {"session_open_hour_utc": 13, "trade_window_hours": 4,
                                "swing_lookback": 40, "swing_strength": 2},
                        "exit": {"tp_r_multiple": 2.0}}}
    strat = MarketOpenBOSStrategy(cfg)
    bars = _bos_bars()
    fired = []
    for i in range(len(bars)):
        for t in strat.on_bar(i, bars):
            fired.append((i, t))
    assert fired, "BOS should fire on a clean structure break during the session"
    idx, spec = fired[0]
    assert spec.direction == Direction.LONG
    assert spec.stop_loss < spec.entry_price < spec.take_profit
    # Only one trade per day.
    assert len(fired) == 1


def test_bos_ignores_breaks_outside_session_window():
    cfg = {"symbol": "US30", "capital": {"risk_usd_per_trade": 100},
           "strategy": {"bos": {"session_open_hour_utc": 3, "trade_window_hours": 1,
                                "swing_lookback": 40, "swing_strength": 2},
                        "exit": {"tp_r_multiple": 2.0}}}
    strat = MarketOpenBOSStrategy(cfg)
    bars = _bos_bars()  # break happens at 13:00, window is 03:00-04:00 -> no trade
    fired = [t for i in range(len(bars)) for t in strat.on_bar(i, bars)]
    assert not fired
