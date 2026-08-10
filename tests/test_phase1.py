"""Phase 1 regression tests — pure-Python, no MT5 terminal required.

Run:  pytest tests/ -q
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.instruments import get_specs  # noqa: E402
from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.mt5_engine import normalize_volume  # noqa: E402
from metatraderbot.portfolio_runner import (  # noqa: E402
    _currencies,
    validate_portfolio_cfg,
)


# --- C1: timestamp-keyed dedup + migration -------------------------------

def test_dedup_keys_on_bar_ts_not_index(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    led.log_signal(
        config_name="slot_a", symbol="EURUSD", timeframe="30m", direction="long",
        entry_price=1.1, sl=1.09, tp=1.13, lots=0.1, bar_index=299,
        bar_ts="2026-07-21T10:00:00+00:00",
    )
    # Same bar -> duplicate.
    assert led.already_have_signal("slot_a", "2026-07-21T10:00:00+00:00")
    # NEXT bar, same window position (the killer case before the fix).
    assert not led.already_have_signal("slot_a", "2026-07-21T10:30:00+00:00")
    # Other slot untouched.
    assert not led.already_have_signal("slot_b", "2026-07-21T10:00:00+00:00")
    led.close()


def test_migration_adds_bar_ts_and_revives_old_rows(tmp_path):
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " config_name TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,"
        " direction TEXT NOT NULL, entry_price REAL NOT NULL, stop_loss REAL NOT NULL,"
        " take_profit REAL NOT NULL, lots REAL NOT NULL, bar_index INTEGER)"
    )
    conn.execute(
        "INSERT INTO signals(ts, config_name, symbol, timeframe, direction,"
        " entry_price, stop_loss, take_profit, lots, bar_index)"
        " VALUES ('t','slot_a','EURUSD','30m','long',1.1,1.09,1.13,0.1,299)"
    )
    conn.commit()
    conn.close()

    led = Ledger(db)  # migrates
    cols = {r[1] for r in led.conn.execute("PRAGMA table_info(signals)")}
    assert "bar_ts" in cols
    # Old NULL-bar_ts rows must NOT block future bars (slot revives).
    assert not led.already_have_signal("slot_a", "2026-07-21T10:00:00+00:00")
    led.close()


def test_recent_closed_pnls_skips_unreconciled(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    for pnl in (10.0, -100.0, None, 25.0):
        led.log_closed(
            opened_ts="t0", ticket=1, config_name="slot_a", symbol="EURUSD",
            direction="long", entry_price=1.1, exit_price=1.2 if pnl is not None else None,
            sl=1.0, tp=1.3, lots=0.1, pnl_usd=pnl,
            exit_reason="tp" if pnl is not None else "unreconciled",
        )
    assert led.recent_closed_pnls("slot_a", 10) == [25.0, -100.0, 10.0]
    led.close()


# --- H7: volume normalization --------------------------------------------

@pytest.mark.parametrize(
    "lots,vmin,vmax,step,expected",
    [
        (0.117, 0.01, 100.0, 0.01, 0.11),   # floors to step, never up
        (0.0086, 0.01, 100.0, 0.01, None),  # below min -> None (policy decides)
        (0.01, 0.01, 100.0, 0.01, 0.01),
        (250.0, 0.01, 100.0, 0.01, 100.0),  # clamps to max
        (0.34, 0.1, 100.0, 0.1, 0.3),       # index-style 0.1 step
        (2.5, 1.0, 50.0, 1.0, 2.0),         # 1-lot step
        (0.0, 0.01, 100.0, 0.01, None),
        (0.05, 0.01, 100.0, 0.0, None),     # broken step -> refuse
        (0.30000000000000004, 0.01, 100.0, 0.01, 0.3),  # float-noise safe
    ],
)
def test_normalize_volume(lots, vmin, vmax, step, expected):
    assert normalize_volume(lots, vmin, vmax, step) == expected


# --- C5: unknown symbol must raise, not silently size 1000x --------------

def test_get_specs_raises_on_unknown_symbol():
    with pytest.raises(ValueError, match="No contract specs"):
        get_specs("EURNOK")
    assert get_specs("EURUSD")["contract_mult"] == 100_000.0


# --- M6: config validation ------------------------------------------------

def _valid_cfg():
    return {
        "capital": {
            "risk_usd_per_trade": 100,
            "daily_stop_loss_usd": -400,
            "max_total_drawdown_usd": -800,
            "max_positions_per_config": 2,
            "trail_start_r": 1.0,
        },
        "configs": [
            {
                "name": "a", "symbol": "EURUSD", "strategy": "DonchianBreakout",
                "timeframe": "30m", "magic_number": 2001,
                "strategy_params": {"exit": {"tp_r_multiple": 3.0}},
            },
            {
                "name": "b", "symbol": "USDJPY", "strategy": "MACDTrend",
                "timeframe": "30m", "magic_number": 2002,
                "strategy_params": {"exit": {"tp_r_multiple": 3.0}},
            },
        ],
    }


def test_valid_config_passes():
    assert validate_portfolio_cfg(_valid_cfg()) == []


def test_duplicate_magic_rejected():
    cfg = _valid_cfg()
    cfg["configs"][1]["magic_number"] = 2001
    assert any("magic_number 2001" in e for e in validate_portfolio_cfg(cfg))


def test_positive_daily_stop_rejected():
    cfg = _valid_cfg()
    cfg["capital"]["daily_stop_loss_usd"] = 400
    assert any("daily_stop_loss_usd" in e for e in validate_portfolio_cfg(cfg))


def test_unknown_symbol_rejected():
    cfg = _valid_cfg()
    cfg["configs"][0]["symbol"] = "EURNOK"
    assert any("PAIR_SPECS" in e for e in validate_portfolio_cfg(cfg))


def test_tp_below_trail_rejected():
    cfg = _valid_cfg()
    cfg["configs"][0]["strategy_params"]["exit"]["tp_r_multiple"] = 0.333
    assert any("trail_start_r" in e for e in validate_portfolio_cfg(cfg))


def test_unknown_strategy_rejected():
    cfg = _valid_cfg()
    cfg["configs"][0]["strategy"] = "QuantumHyperdrive"
    assert any("unknown strategy" in e for e in validate_portfolio_cfg(cfg))


# --- currency split used by exposure caps ---------------------------------

def test_currency_split():
    assert _currencies("EURUSD") == ("EUR", "USD")
    assert _currencies("XAUUSD") == ("XAU", "USD")
    assert _currencies("US30") == ("US30",)


# --- live YAMLs must pass their own validation ----------------------------

@pytest.mark.parametrize("name", ["portfolio_ftmo.yaml", "portfolio_yolo.yaml"])
def test_shipped_configs_are_valid(name):
    yaml = pytest.importorskip("yaml")
    with open(ROOT / "config" / name) as f:
        cfg = yaml.safe_load(f)
    assert validate_portfolio_cfg(cfg) == []
