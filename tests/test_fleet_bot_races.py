"""The Fleet page's Real test races when and the way the paper bots raced (owner, 2026-10-19: "we passed trial and
we have profit on trial but it doesn't show on the fleet page" -- the bots had raced at 05:23 New York, the Real
test at 19:00 from another price)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import lucid_fleet as lf  # noqa: E402


def _bars(days: int = 3) -> pd.DataFrame:
    """Flat 1-minute NQ bars at 20,000 for a few sessions, Sun 18:00 ET on; it dips 60 points after Monday 19:30 ET
    and rises 100 points from Tuesday 06:00 ET."""
    idx = pd.date_range("2026-10-18 22:00", periods=days * 24 * 60, freq="1min", tz="UTC")
    px = np.full(len(idx), 20_000.0)
    et = idx.tz_convert(lf.ET)
    dip = (et >= pd.Timestamp("2026-10-19 19:30", tz=lf.ET)) & (et < pd.Timestamp("2026-10-20 05:00", tz=lf.ET))
    px[dip] = 19_940.0                                    # a 19:00 buy hits the loss line (-50 points)
    up = et >= pd.Timestamp("2026-10-20 06:00", tz=lf.ET)
    px[up] = 20_100.0                                     # a 05:30 buy (at 20,000) reaches the target (+75.5)
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=idx)


def _write(d: Path, trades: list[dict], name: str = "lucid_futures_trades.jsonl") -> None:
    (d / name).write_text("\n".join(json.dumps(t) for t in trades) + "\n", encoding="utf-8")


def test_bot_races_takes_each_days_first_race_of_each_bot_and_the_open_one(tmp_path):
    _write(tmp_path, [{"opened_ts": "2026-10-19T23:00:00+00:00", "direction": "long"},
                      {"opened_ts": "2026-10-20T01:00:00+00:00", "direction": "short"},      # same day, later
                      {"opened_ts": "2026-09-29T23:00:00+00:00", "direction": "long"}])     # before `since`
    (tmp_path / "state_lucid_futures_funded.json").write_text(json.dumps(
        {"run": {"tickets": {"7": {"opened_ts": "2026-10-20T09:30:00+00:00", "direction": "short"}}}}))
    br = lf.bot_races(tmp_path, since="2026-10-01")
    assert list(br) == ["2026-10-20"]                     # 19:00 ET on the 19th belongs to the 20th
    assert br["2026-10-20"]["eval"] == (pd.Timestamp("2026-10-19T23:00:00+00:00"), 1)
    assert br["2026-10-20"]["funded"] == (pd.Timestamp("2026-10-20T09:30:00+00:00"), -1)   # still open


def test_the_real_test_races_when_the_bot_raced_not_at_19_00(tmp_path):
    bars = _bars()
    cfg = dict(slots=1, start="2026-10-20", stagger=0, start_slots=None)
    at_19 = lf.run(bars, lf.FleetConfig(**cfg))["events"]
    assert any(e["what"].startswith("evaluation failed") for e in at_19)        # the plain replay: a 19:00 buy
    late = {"2026-10-20": {"eval": (pd.Timestamp("2026-10-20 05:30", tz=lf.ET).tz_convert("UTC"), 1)}}
    ev = lf.run(bars, lf.FleetConfig(**cfg, bot_races=late))["events"]
    passed = [e for e in ev if e["what"].startswith("evaluation PASSED")]
    assert passed and "Tue 05:30 New York (when and the way the bot raced)" in passed[0]["why"]
    none = lf.run(bars, lf.FleetConfig(**cfg, bot_races={}))["events"]       # the bots didn't race: no race
    assert [e["what"] for e in none] == ["evaluation bought"]
