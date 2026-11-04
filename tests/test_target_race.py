"""Target race timing and target maths (metatraderbot/target_race.py)."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metatraderbot import futures, target_race  # noqa: E402

ET = ZoneInfo("America/New_York")


def et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET).astimezone(timezone.utc)


def test_lucid_minute_counts_from_the_18_00_open():
    assert target_race.lucid_minute(et(2026, 9, 30, 18, 0)) == 0
    assert target_race.lucid_minute(et(2026, 9, 30, 19, 0)) == 60
    assert target_race.lucid_minute(et(2026, 10, 1, 0, 0)) == 360
    assert target_race.lucid_minute(et(2026, 10, 1, 16, 0)) == 22 * 60


def test_entry_window_runs_through_midnight():
    cfg = target_race.RaceConfig()                       # 19:00 -> 12:00 ET
    assert not cfg.entry_window(et(2026, 9, 30, 18, 30))
    assert cfg.entry_window(et(2026, 9, 30, 19, 0))
    assert cfg.entry_window(et(2026, 10, 1, 2, 0))
    assert cfg.entry_window(et(2026, 10, 1, 8, 0))       # 14:00 in Prague
    assert cfg.entry_window(et(2026, 10, 1, 11, 59))
    assert not cfg.entry_window(et(2026, 10, 1, 12, 0))
    assert not cfg.entry_window(et(2026, 10, 1, 15, 0))


def test_entry_window_across_the_dst_change():
    cfg = target_race.RaceConfig()
    # US DST ends Sunday 2026-11-01; wall-clock ET decides, not UTC.
    assert cfg.entry_window(et(2026, 11, 1, 19, 30))
    assert not cfg.entry_window(et(2026, 11, 1, 17, 30))


def test_exit_due_from_16_00_until_the_reopen():
    cfg = target_race.RaceConfig()
    assert not cfg.exit_due(et(2026, 10, 1, 15, 59))
    assert cfg.exit_due(et(2026, 10, 1, 16, 0))
    assert cfg.exit_due(et(2026, 10, 1, 16, 44))
    assert cfg.exit_due(et(2026, 10, 1, 17, 30))
    assert not cfg.exit_due(et(2026, 10, 1, 18, 0))
    assert not cfg.exit_due(et(2026, 10, 1, 19, 0))


def test_later_entry_configurable():
    cfg = target_race.RaceConfig(entry="08:00")
    assert not cfg.entry_window(et(2026, 10, 1, 7, 59))
    assert cfg.entry_window(et(2026, 10, 1, 8, 0))


def test_target_points_covers_need_and_commission():
    s = futures.spec("NQ")
    pts = target_race.target_points(3000.0, 4, s["point_value"], s["tick_size"], s["commission"])
    # $3,000 + 8 x $1.75 commission over 4 x $20/pt = 37.675 -> 37.75 on the tick grid, + 1 tick
    assert pts == pytest.approx(38.0)
    pnl = pts * 4 * s["point_value"] - 2 * s["commission"] * 4
    assert pnl >= 3000.0


def test_target_points_after_a_losing_day_and_floor():
    s = futures.spec("NQ")
    far = target_race.target_points(4200.0, 4, s["point_value"], s["tick_size"], s["commission"])
    near = target_race.target_points(3000.0, 4, s["point_value"], s["tick_size"], s["commission"])
    assert far > near
    # nothing left to make: still one tick beyond the fill plus the margin tick
    assert target_race.target_points(-50.0, 4, 20.0, 0.25, 0.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        target_race.target_points(1000.0, 0, 20.0, 0.25, 1.75)


def test_next_entry_now_tonight_or_after_the_weekend():
    cfg = target_race.RaceConfig()
    now = et(2026, 9, 30, 9, 13)                          # Wed, window open, no race yet
    assert target_race.next_entry(now, cfg, None) == now
    # today's race done -> tonight 19:00 ET (the Thursday trading day)
    assert target_race.next_entry(now, cfg, "2026-09-30") == et(2026, 9, 30, 19, 0)
    # Friday after the close -> Sunday 19:00 ET
    assert target_race.next_entry(et(2026, 10, 2, 17, 30), cfg, "2026-10-02") == et(2026, 10, 4, 19, 0)
    # Saturday -> Sunday 19:00 ET
    assert target_race.next_entry(et(2026, 10, 3, 12, 0), cfg, None) == et(2026, 10, 4, 19, 0)
    # the evening before Thanksgiving's short session is skipped
    assert target_race.next_entry(et(2026, 11, 25, 17, 30), cfg, "2026-11-25") == et(2026, 11, 29, 19, 0)


def test_holiday_sessions_are_skipped():
    assert target_race.skip_day("2026-11-26")            # Thanksgiving early close
    assert target_race.skip_day("2026-12-25")            # closed
    assert not target_race.skip_day("2026-10-01")
