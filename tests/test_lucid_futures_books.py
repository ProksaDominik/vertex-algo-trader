"""The funded bots' account rules and the futures lab's Lucid scoring.

Funded race: the MLL floor trails the best end-of-day balance and locks at
$50,100 once one reaches $52,100, and a big day makes the payout wait for
Lucid's 40% rule (the $1,500 daily cap is above 40% of the first payout's
$2,600 since 2026-10-21). The superseded ORB roster bot: per-trade risk tapers
from $250 toward a quarter of that as equity nears the floor. Lab: no new
signal in the last minutes before Lucid's cutoff, and a shadow trade still
open at the flatten window is scored where the bot would have closed it --
not left open overnight.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ET = ZoneInfo("America/New_York")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def funded():
    return _load("funded_bot", "scripts/run_lucid_futures_funded.py")


@pytest.fixture(scope="module")
def roster():
    return _load("funded_roster_bot", "scripts/run_lucid_futures_funded_roster.py")


@pytest.fixture(scope="module")
def lab():
    return _load("lab_bot", "scripts/run_lab.py")


def test_mll_trails_the_best_eod_balance_then_locks(funded):
    run = funded.new_run()
    assert funded.mll_floor(run) == 48_000.0
    run["peak_eod"] = 51_500.0
    assert funded.mll_floor(run) == 49_500.0
    run["mll_locked"] = True                       # set once an EOD balance reached 52,100
    run["peak_eod"] = 55_000.0
    assert funded.mll_floor(run) == 50_100.0       # locked: never trails again


def test_a_big_day_makes_the_payout_wait_for_the_40_percent_rule(funded):
    first = funded.KEEP_USD + funded.PAYOUT_WANT[0] - funded.ANCHOR      # $2,600 to the first payout
    assert first == 2_600.0 and funded.DAY_TAKE_MAX == 1_500.0
    # one $1,500 day: the money needs $1,100 more, but the 40% rule needs the cycle at $3,750 -> $2,250 more
    run = funded.new_run()
    run.update(realized_usd=1_500.0, day_start_realized=1_500.0, cycle_pnl=1_500.0, cycle_best_day=1_500.0)
    assert funded.payout_need(run) == pytest.approx(1_500.0 / funded.CONSISTENCY - 1_500.0)
    assert funded.day_take(funded.payout_need(run)) == 1_500.0


def test_roster_risk_tapers_toward_the_floor_but_never_below_a_quarter(roster):
    run = roster.new_run()                         # floor 48,000
    assert roster.risk_budget(50_000.0, run) == pytest.approx(250.0)
    assert roster.risk_budget(49_000.0, run) == pytest.approx(125.0)
    assert roster.risk_budget(48_200.0, run) == pytest.approx(62.5)
    assert roster.risk_budget(60_000.0, run) == pytest.approx(250.0)


def test_roster_payout_level_sits_above_lucids_buffer(roster):
    assert roster.ANCHOR + roster.PAYOUT_TARGET_USD > roster.MLL_LOCK_TRIGGER


def test_lab_blocks_entries_near_the_cutoff(lab):
    bar = lambda h, m: pd.Timestamp(datetime(2026, 9, 29, h, m, tzinfo=ET)).tz_convert("UTC")
    assert lab.lucid_entry_ok(bar(10, 0), "15m")           # closes 10:15 ET
    assert not lab.lucid_entry_ok(bar(16, 15), "15m")      # closes 16:30 ET -- no new entries
    assert not lab.lucid_flat_due(bar(16, 0), "15m")       # closes 16:15 ET
    assert lab.lucid_flat_due(bar(16, 30), "15m")          # closes 16:45 ET -- flat


def test_lab_scores_an_open_trade_at_the_flatten_bar(lab):
    idx = pd.DatetimeIndex([pd.Timestamp(datetime(2026, 9, 29, h, m, tzinfo=ET)).tz_convert("UTC")
                            for h, m in [(16, 0), (16, 15), (16, 30)]])
    bars = pd.DataFrame({"open": [100.0, 101.0, 102.0], "high": [101.5, 102.5, 103.0],
                         "low": [99.5, 100.5, 101.5], "close": [101.0, 102.0, 102.5]}, index=idx)
    sig = {"ts": str(idx[0] - pd.Timedelta(minutes=15)), "entry": 100.0, "sl": 95.0, "tp": 110.0,
           "dir": "long"}
    r = lab.score_open(sig, bars, flat_due=lambda ts: lab.lucid_flat_due(ts, "15m"))
    assert r == pytest.approx((102.5 - 100.0) / 5.0)         # closed at the 16:30 bar's close
    assert lab.score_open(sig, bars) is None                 # MT5 mode: still open
