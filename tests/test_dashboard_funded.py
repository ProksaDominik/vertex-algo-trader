"""The Funded page's payout road and account curve follow run_lucid_futures_funded.py's rules."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dashboard  # noqa: E402


def test_fresh_account_needs_three_winning_races_for_the_first_payout():
    # $1,500 a day (owner 2026-10-21): after two $1,500 days the 40% rule asks for $3,750 in the cycle, so the
    # third race takes $750 -- and the first payout is $1,650 instead of the $500 of the old $1,040 cap
    p = dashboard._funded_path(50_000.0, 50_000.0, 0.0, 0.0, 0, 0)
    assert [s["take"] for s in p["steps"]] == [1_500.0, 1_500.0, 750.0]
    assert [s["balance"] for s in p["steps"]] == [51_500.0, 53_000.0, 53_750.0]
    assert p["steps"][-1]["line"] == 50_100.0                      # locked once a close reaches $52,100
    assert p["amount"] == 1_650.0 and p["payout_no"] == 1
    assert p["chance"] == pytest.approx(2000 / 3500 * 2000 / 3500 * 2900 / 3650)
    assert 0.25 < p["chance"] < 0.27            # no market direction; the replay with the trend side: ~30%


def test_later_payouts_are_2500_with_the_locked_line():
    p = dashboard._funded_path(52_100.0, 52_100.0, 0.0, 0.0, 0, 1)
    assert sum(s["race"] for s in p["steps"]) == 3
    assert p["amount"] == 2_500.0 and p["payout_no"] == 2
    assert all(s["line"] == 50_100.0 for s in p["steps"])
    assert p["chance"] == pytest.approx(2000 / 3500 * 3500 / 5000 * 5000 / 5750)    # 1,500 + 1,500 + 750
    assert 0.33 < p["chance"] < 0.36


def test_money_reached_early_waits_for_the_third_day_without_racing():
    # one day in the cycle, already $2,600 up with the best day at 40%
    p = dashboard._funded_path(52_600.0, 52_600.0, 2_600.0, 1_040.0, 1, 0)
    assert [s["race"] for s in p["steps"]] == [False, False]
    assert p["chance"] == 1.0 and p["amount"] == 500.0
    assert dashboard._funded_path(52_600.0, 52_600.0, 2_600.0, 1_040.0, 3, 0)["steps"] == []


def test_need_includes_the_40_percent_rule():
    # $1,200 best day needs $3,000 of cycle profit even though the money is already there
    assert dashboard._funded_need(52_700.0, 2_700.0, 1_200.0, 0) == pytest.approx(300.0)


def test_curve_tracks_balance_loss_line_and_payouts():
    trades = [{"closed_ts": "2026-10-05T01:27:00+00:00", "pnl_usd": 1_040.0},
              {"closed_ts": "2026-10-06T00:32:00+00:00", "pnl_usd": 1_040.0},
              {"closed_ts": "2026-10-07T01:00:00+00:00", "pnl_usd": 520.0}]
    pays = [{"ts": "2026-10-07T20:00:00+00:00", "amount_usd": 500.0}]
    cv = dashboard._funded_curve(trades, pays, "2026-10-04T00:00:00+00:00")
    assert list(cv["what"]) == ["start", "trade", "trade", "trade", "payout"]
    assert list(cv["balance"]) == [50_000.0, 51_040.0, 52_080.0, 52_600.0, 52_100.0]
    assert list(cv["line"]) == [48_000.0, 49_040.0, 50_080.0, 50_100.0, 50_100.0]


def test_curve_of_an_account_without_trades_is_just_its_start():
    cv = dashboard._funded_curve([], [], "2026-10-08T07:53:29+00:00")
    assert len(cv) == 1 and cv["balance"].iloc[0] == 50_000.0
    assert dashboard._funded_curve([], []).empty


# --- the trial page's trade stories ---------------------------------------------------------------
from zoneinfo import ZoneInfo  # noqa: E402

PRAGUE = ZoneInfo("Europe/Prague")


def _story(t, prev=None, rec=None, **kw):
    return dict(dashboard._race_story(t, prev, rec, PRAGUE, **kw))


def test_a_restart_says_which_evaluation_ended_and_how():
    prev = {"attempt": 19, "pnl_usd": -2055.0, "closed_ts": "2026-10-08T10:34:03+00:00"}
    t = {"attempt": 20, "direction": "long", "lots": 2.0, "entry_price": 31123.125, "exit_price": 31198.75,
         "tp": 31198.75, "line": 31073.12, "pnl_usd": 3018.0, "reason": "pass_while_away",
         "opened_ts": "2026-10-08T10:34:19+00:00", "closed_ts": "2026-10-08T11:40:00+00:00"}
    s = _story(t, prev)
    assert "Evaluation #19 had just ended (failed" in s["Why it started"]
    assert "above its 200-day average" in s["Why this direction"]
    assert "laptop was asleep" in s["How it ended"] and "passed" in s["How it ended"]
    assert "75.62 points" in s["What it aimed for"]


def test_the_planned_19_00_race_and_a_loss_line_end():
    t = {"attempt": 12, "direction": "long", "lots": 1.0, "entry_price": 31491.75, "exit_price": 31394.17,
         "tp": 31644.75, "line": 31394.17, "pnl_usd": -1955.1, "reason": "line",
         "opened_ts": "2026-10-06T23:00:10+00:00", "closed_ts": "2026-10-07T09:03:00+00:00"}
    s = _story(t, None, {"best_pts": 25.5, "worst_pts": 97.58})
    assert s["Why it started"].startswith("The day's planned race")
    assert "fell to the loss line" in s["How it ended"] and "failed" in s["How it ended"]
    assert "at best +25.50 points" in s["How it ended"]


def test_before_the_trend_rule_races_always_bought_and_short_races_say_why():
    old = {"direction": "long", "lots": 1.0, "entry_price": 30000.0, "exit_price": 30100.0, "pnl_usd": 2000.0,
           "reason": "tp", "tp": 30100.0, "opened_ts": "2026-10-01T23:00:00+00:00",
           "closed_ts": "2026-10-02T02:00:00+00:00"}
    assert "before 7 Oct" in _story(old)["Why this direction"]
    live = {"direction": "short", "lots": 2.0, "entry_price": 30000.0, "tp": 29924.5, "line": 30050.0,
            "side_reason": "SHORT: last close 27,000.00 below the 200-day average 28,000.00",
            "opened_ts": "2026-10-08T23:00:00+00:00"}
    s = _story(live, live=True)
    assert s["Why this direction"].startswith("Sold short because last close 27,000.00 below")
    assert "How it can end" in s and "How it ended" not in s
