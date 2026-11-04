"""Market breadth for the race (owner, 2026-10-21) and the second-strategy plumbing: the S&P 500 up/down count,
the side it gives a trading day, the 200-day fallback, and the Real test of a second pair of bots."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import breadth, lucid_fleet as lf, target_race  # noqa: E402

ET = ZoneInfo("America/New_York")


def _closes(n_stocks: int, ups: list[int], start="2026-10-12") -> dict[str, pd.Series]:
    """n_stocks daily closes over len(ups)+1 sessions; on session k, ups[k] of them rise and the rest fall."""
    days = pd.bdate_range(start, periods=len(ups) + 1)
    out = {}
    for i in range(n_stocks):
        px = [100.0]
        for k, u in enumerate(ups):
            px.append(px[-1] * (1.01 if i < u else 0.99))
        out[f"S{i}"] = pd.Series(px, index=pd.DatetimeIndex(days))
    return out


def test_count_and_the_side_for_a_trading_day(tmp_path):
    counts = breadth.count(_closes(400, [300, 100, 200]))
    assert counts["2026-10-13"] == {"adv": 300, "dec": 100, "n": 400}
    assert counts["2026-10-14"]["adv"] == 100
    assert "2026-10-12" not in counts                      # the first day has no change
    # the race of trading day Thu 15 Oct (19:00 New York on Wed 14) uses Wed's session
    side, why = breadth.side_for("2026-10-15", tmp_path, counts)
    assert side == -1 and "SHORT" in why and "100 up / 300 down" in why
    assert breadth.side_for("2026-10-14", tmp_path, counts)[0] == 1
    assert breadth.side_for("2026-10-16", tmp_path, counts)[0] == 0         # 200 up / 200 down: no side
    assert breadth.side_for("2026-11-02", tmp_path, counts)[0] == 0         # too old
    assert breadth.count(_closes(50, [40])) == {}                           # too few stocks: not counted


def test_the_latest_session_is_counted_after_16_30_new_york():
    assert breadth.last_session(datetime(2026, 10, 21, 16, 29, tzinfo=ET)) == "2026-10-20"
    assert breadth.last_session(datetime(2026, 10, 21, 16, 31, tzinfo=ET)) == "2026-10-21"
    assert breadth.last_session(datetime(2026, 10, 24, 12, 0, tzinfo=ET)) == "2026-10-23"   # Saturday -> Friday
    assert breadth.last_session(datetime(2026, 10, 26, 9, 0, tzinfo=ET)) == "2026-10-23"    # Monday morning


def test_ensure_counts_missing_sessions_once_and_keeps_them(tmp_path):
    syms = [f"S{i}" for i in range(400)]
    (tmp_path / "breadth").mkdir()
    (tmp_path / "breadth" / "sp500.json").write_text(json.dumps(syms))
    data = _closes(400, [250, 150], start="2026-10-19")
    calls = []

    def fetch(sym, interval, rng):
        calls.append(sym)
        s = data[sym]
        return pd.DataFrame({"close": s.to_numpy()}, index=s.index.tz_localize("UTC") + pd.Timedelta(hours=13, minutes=30))

    now = datetime(2026, 10, 21, 17, 0, tzinfo=ET)
    got = breadth.ensure(tmp_path, fetch=fetch, now=now, workers=4)
    assert got["2026-10-20"]["adv"] == 250 and got["2026-10-21"]["adv"] == 150
    assert len(calls) == 400
    breadth.ensure(tmp_path, fetch=fetch, now=now)                          # already there: no download
    assert len(calls) == 400
    assert json.loads((tmp_path / "breadth" / "sp500_ad.json").read_text())["2026-10-21"]["dec"] == 250


def test_the_wikipedia_list_is_read_in_yahoo_spelling():
    html = ('<table id="constituents"><tr id="a"><td id="b"><a href="x">MMM</a></td></tr>'
            '<tr id="c"><td id="d"><a href="y">BRK.B</a></td></tr></table>')
    assert breadth.parse_symbols(html) == ["BRK-B", "MMM"]


def test_breadth_side_falls_back_to_the_200_day_rule_without_a_count(monkeypatch, tmp_path):
    monkeypatch.setattr(breadth, "ensure", lambda data: {})
    monkeypatch.setattr(breadth, "side_for", lambda day, data: (0, "no breadth count yet"))

    class Eng:
        def get_bars(self, sym, tf, n):
            return pd.DataFrame({"close": np.linspace(100, 200, 260)})        # rising: above the 200-day average

    side, why = target_race.race_side(Eng(), "NQ", "breadth", now=datetime(2026, 10, 21, 19, 0, tzinfo=ET))
    assert side == 1 and "200-day" in why and "no breadth count yet" in why
    monkeypatch.setattr(breadth, "side_for", lambda day, data: (-1, f"SHORT: day {day}"))
    side, why = target_race.race_side(Eng(), "NQ", "breadth", now=datetime(2026, 10, 21, 19, 0, tzinfo=ET))
    assert side == -1 and why == "breadth SHORT: day 2026-10-22"        # 19:00 New York belongs to the next day


def test_the_real_test_of_a_second_pair_of_bots_reads_its_own_files(tmp_path):
    (tmp_path / "lucid_futures_trades_ad.jsonl").write_text(
        json.dumps({"opened_ts": "2026-10-21T23:00:00+00:00", "direction": "short"}) + "\n")
    (tmp_path / "lucid_futures_trades.jsonl").write_text(
        json.dumps({"opened_ts": "2026-10-21T23:00:00+00:00", "direction": "long"}) + "\n")
    assert lf.bot_races(tmp_path)["2026-10-22"]["eval"][1] == 1
    assert lf.bot_races(tmp_path, trial_tag="_ad")["2026-10-22"]["eval"][1] == -1
    assert "funded" not in lf.bot_races(tmp_path, trial_tag="_ad", funded_tag="_Sim102_AD")["2026-10-22"]


def test_the_replay_uses_given_sides_and_replays_days_before_the_bots_started():
    idx = pd.date_range("2026-10-18 22:00", periods=3 * 24 * 60, freq="1min", tz="UTC")
    px = np.full(len(idx), 20_000.0)
    et = idx.tz_convert(lf.ET)
    px[et >= pd.Timestamp("2026-10-19 20:00", tz=lf.ET)] = 19_900.0      # Mon evening on: down 100 points
    bars = pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)
    cfg = dict(slots=1, start="2026-10-20", stagger=0, start_slots=None)
    long_ = lf.run(bars, lf.FleetConfig(**cfg))["events"]                   # the 200-day default: long here
    assert any(e["what"].startswith("evaluation failed") for e in long_)
    short = lf.run(bars, lf.FleetConfig(**cfg, sides={"2026-10-20": -1}))["events"]
    assert any(e["what"].startswith("evaluation PASSED") and "more stocks fell" in e["why"] for e in short)
    # bots that started later: before their start the account replays at 19:00, after it only their races count
    none = lf.run(bars, lf.FleetConfig(**cfg, sides={"2026-10-20": -1}, bot_races={}, replay_before="2026-10-21"))
    assert any(e["what"].startswith("evaluation PASSED") for e in none["events"])
    later = lf.run(bars, lf.FleetConfig(**cfg, bot_races={}, replay_before="2026-10-20"))
    assert [e["what"] for e in later["events"]] == ["evaluation bought"]
