"""ORBTrendStrategy: the intraday leg of the Lucid futures stack.

Pins the rules the research tested: a 45-minute opening range from 09:30 ET,
the FIRST breakout of the day decides (with the daily trend -> trade, against
it -> no trade today), entries only before 12:00 ET, stop at the far side of
the range, holiday sessions skipped, time exit from 15:55 ET.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import lucid_rules  # noqa: E402
from metatraderbot.lab_registry import instantiate  # noqa: E402

ET = ZoneInfo("America/New_York")


def sessions(n_days: int, drift: float, last_day_path: list[float]) -> pd.DataFrame:
    """RTH 15m bars (09:30-15:45 ET open times) for n_days whose closes trend
    by `drift` per day THROUGH OVERNIGHT GAPS -- flat inside each day, so no
    history day can break out of its own opening range -- then one final day
    whose bar closes follow `last_day_path`."""
    rows, px = [], 20_000.0
    day = pd.Timestamp("2026-08-03")
    made = 0
    while made < n_days + 1:
        if day.weekday() < 5:
            final = made == n_days
            for k in range(26):
                t = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET) + pd.Timedelta(minutes=15 * k)
                if final:
                    if k >= len(last_day_path):
                        break
                    c = last_day_path[k]
                    o = rows[-1]["close"] if k else c
                else:
                    o = c = px
                rows.append({"t": pd.Timestamp(t).tz_convert("UTC"), "open": o, "close": c,
                             "high": max(o, c) + 1.0, "low": min(o, c) - 1.0, "volume": 100.0})
            px += drift
            made += 1
        day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows).set_index("t")
    df.index.name = "timestamp"
    return df


def strat():
    return instantiate("orb_trend", {"orb_trend": {"or_minutes": 45, "last_entry": "12:00"}}, 2.0, "MNQ")


def run(bars: pd.DataFrame):
    s = strat()
    out = []
    for i in range(len(bars)):
        for spec in s.on_bar(i, bars) or []:
            out.append((bars.index[i].tz_convert(ET).time(), spec))
    return out


OR = [20_560.0, 20_565.0, 20_562.0]            # the 09:30, 09:45, 10:00 bars


def test_with_trend_breakout_goes_long_with_stop_at_range_low():
    bars = sessions(22, drift=+26.0, last_day_path=OR + [20_563.0, 20_580.0, 20_585.0])
    sig = run(bars)
    assert len(sig) == 1
    t, spec = sig[0]
    assert t.hour == 10 and t.minute == 30           # the first bar closing beyond the range
    assert spec.direction.value == "long"
    low = bars["low"].iloc[-6:-3].min()
    assert spec.stop_loss == pytest.approx(low)
    assert spec.take_profit == pytest.approx(spec.entry_price + 2 * (spec.entry_price - spec.stop_loss))


def test_first_breakout_against_the_trend_means_no_trade_today():
    # uptrend, but the first break is DOWN; a later up-break must not trade
    bars = sessions(22, drift=+26.0, last_day_path=OR + [20_540.0, 20_600.0, 20_610.0])
    assert run(bars) == []


def test_no_trade_without_enough_history_for_the_trend():
    bars = sessions(5, drift=+26.0, last_day_path=OR + [20_563.0, 20_580.0])
    assert run(bars) == []


def test_holiday_sessions_are_skipped(monkeypatch):
    bars = sessions(22, drift=+26.0, last_day_path=OR + [20_563.0, 20_580.0])
    last_day = bars.index[-1].tz_convert(ET).date()
    monkeypatch.setattr(lucid_rules, "HOLIDAY_SESSIONS", frozenset({last_day}))
    assert run(bars) == []


def test_time_exit_from_1555_new_york():
    s = strat()
    assert not s.exit_due(datetime(2026, 9, 30, 15, 54, tzinfo=ET))
    assert s.exit_due(datetime(2026, 9, 30, 15, 55, tzinfo=ET))
    assert s.exit_due(datetime(2026, 9, 30, 19, 55, tzinfo=ZoneInfo("UTC")))   # 15:55 EDT
