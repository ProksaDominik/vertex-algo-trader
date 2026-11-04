"""The fleet simulator (lucid_fleet.run) follows the bots' rules on scripted NQ minute prices."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import lucid_fleet as lf  # noqa: E402

ET = "America/New_York"
P0 = 30_000.0


def bars_for(days: list[tuple[str, float]], start: float = P0, last_end: str = "16:59") -> pd.DataFrame:
    """One Lucid session per (trading day, move): flat from 18:00 ET the evening before, the move made
    between 19:00 and 21:00 ET, then flat to 16:59 ET (the last session to `last_end`)."""
    frames, px = [], start
    for i, (day, move) in enumerate(days):
        d = pd.Timestamp(day)
        prev = (d - pd.Timedelta(days=3 if d.dayofweek == 0 else 1)).strftime("%Y-%m-%d")
        end = last_end if i == len(days) - 1 else "16:59"
        idx = pd.date_range(pd.Timestamp(f"{prev} 18:00", tz=ET), pd.Timestamp(f"{day} {end}", tz=ET), freq="1min")
        xs = [pd.Timestamp(f"{prev} {t}", tz=ET).value for t in ("18:00", "19:00", "21:00")] + [idx[-1].value]
        ys = [px, px, px + move, px + move]
        p = np.interp(idx.as_unit("ns").asi8, xs, ys)
        frames.append(pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0},
                                   index=idx.tz_convert("UTC")))
        px += move
    return pd.concat(frames)


def run(days, slots=1, stagger=0, **kw):
    cfg = lf.FleetConfig(slots=slots, start=days[0][0], stagger=stagger, **{"eval_fee": 185.0, "reset_fee": 120.0, **kw})
    return lf.run(bars_for(days), cfg)


def test_evaluation_passes_and_becomes_a_funded_account():
    r = run([("2026-10-06", +80)])
    s = r["slots"][0]
    assert s["stage"] == "funded" and s["passes"] == 1 and s["fees"] == 185.0
    assert any("PASSED" in e["what"] for e in r["events"])


def test_failed_evaluation_pays_the_reset():
    r = run([("2026-10-06", -60)])
    s = r["slots"][0]
    assert s["stage"] == "eval" and s["resets"] == 1 and s["fees"] == 185.0 + 120.0 and s["bal"] == 0.0


def test_three_funded_wins_take_the_first_payout_like_the_funded_bot():
    r = run([("2026-10-06", +80), ("2026-10-07", +80), ("2026-10-08", +80), ("2026-10-09", +80)])
    s = r["slots"][0]
    # funded days at most $1,500 (2026-10-21): +$1,506.50, +$1,506.50, then the 40% rule's +$761.50 -> $3,774.50;
    # payout = everything above $52,100
    assert lf.PLANS["50K"].take_max == 1_500.0
    assert s["payouts"] == 1 and s["paid"] == pytest.approx(1_674.5)
    assert s["bal"] == pytest.approx(2_100.0) and s["floor"] == pytest.approx(100.0)
    assert s["net"] == pytest.approx(0.9 * 1_674.5 - 185.0)


def test_blown_funded_account_buys_a_new_evaluation():
    r = run([("2026-10-06", +80), ("2026-10-07", -120)])
    s = r["slots"][0]
    assert s["blown"] == 1 and s["stage"] == "eval" and s["evals"] == 2 and s["fees"] == 370.0


def test_slots_start_on_consecutive_days_and_race_together():
    r = run([("2026-10-06", -60), ("2026-10-07", -60), ("2026-10-08", -60)], slots=3, stagger=1)
    bought = [(e["day"], e["slot"]) for e in r["events"] if e["what"] == "evaluation bought"]
    assert bought == [("2026-10-06", 1), ("2026-10-07", 2), ("2026-10-08", 3)]
    assert [s["resets"] for s in r["slots"]] == [3, 2, 1]


def test_a_race_still_running_is_shown_open():
    b = bars_for([("2026-10-06", +10)], last_end="10:00")          # the session is not over yet
    r = lf.run(b, lf.FleetConfig(slots=1, start="2026-10-06"))
    s = r["slots"][0]
    assert s["open"] and s["open"]["pnl"] > 0 and r["daily"]["racing"].iloc[-1]
    assert s["passes"] == 0 and s["resets"] == 0


def test_trend_side_follows_the_200_day_average():
    idx = pd.date_range("2025-01-01 23:30", periods=260, freq="B", tz="UTC")       # 18:30-19:30 ET
    falling = pd.DataFrame({"open": 0.0, "high": 0.0, "low": 0.0, "close": np.linspace(40_000, 30_000, 260),
                            "volume": 1.0}, index=idx)
    sides = lf.trend_sides(falling)
    last = sorted(sides)[-1]
    assert sides[last] == -1 and sides[sorted(sides)[10]] == 1        # too little history -> long


def test_step_up_adds_slots_only_from_profit():
    # 2 slots to start; both pass and take their first payout on Fri 9 Oct -> slot 3 opens at the next session
    win = [("2026-10-06", +80), ("2026-10-07", +80), ("2026-10-08", +80), ("2026-10-09", +80), ("2026-10-13", 0)]
    r = run(win, slots=3, stagger=0, start_slots=2, step_usd=100.0)
    bought = [(e["day"], e["slot"]) for e in r["events"] if e["what"].startswith("evaluation bought")]
    assert bought == [("2026-10-06", 1), ("2026-10-06", 2), ("2026-10-13", 3)]
    r = run(win, slots=3, stagger=0, start_slots=2, step_usd=1e9)
    assert r["slots"][2]["stage"] == "waiting" and r["slots"][2]["fees"] == 0.0


LOSING_WEEK = [("2026-10-06", -60), ("2026-10-07", -60), ("2026-10-08", -60), ("2026-10-09", -60), ("2026-10-12", -60)]


def test_stop_rule_buys_nothing_more_once_the_money_made_is_down_the_limit():
    # owner 2026-10-21: "if the total ever falls to -$10,000, stop buying evaluations" -- here a $500 limit:
    # $185 + three $120 resets = -$545, so the 4th failed evaluation buys no reset and the 5th day has no race
    r = run(LOSING_WEEK, stop_usd=500.0)
    s = r["slots"][0]
    assert s["stage"] == "stopped" and s["resets"] == 3 and s["fees"] == pytest.approx(545.0)
    stop = [e for e in r["events"] if e["what"].startswith("STOP RULE")]
    assert len(stop) == 1 and stop[0]["day"] == "2026-10-09" and stop[0]["slot"] == 0
    assert not [e for e in r["events"] if e["day"] == "2026-10-12"]           # stopped: no race, nothing bought
    assert s["net"] == pytest.approx(-545.0)


def test_without_the_stop_rule_the_resets_go_on():
    r = run(LOSING_WEEK, stop_usd=None)
    assert r["slots"][0]["stage"] == "eval" and r["slots"][0]["resets"] == 5


def test_the_owners_stop_rule_is_the_default():
    assert lf.FleetConfig().stop_usd == 10_000.0
