"""Integration: auto_retune wired into PortfolioRunner's quarantine trip,
including restart persistence. Pure Python — no MT5 terminal required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.mt5_engine import AccountInfo  # noqa: E402
from metatraderbot.portfolio_runner import PortfolioRunner, WARMUP_BARS  # noqa: E402
from metatraderbot.state import BotState  # noqa: E402


def _make_bars(n=WARMUP_BARS + 1, base=1.10):
    ts = pd.date_range("2026-07-20 00:00", periods=n, freq="30min", tz="UTC")
    rows = [{"open": base, "high": base + 0.0005, "low": base - 0.0005,
              "close": base, "volume": 100} for _ in range(n)]
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "timestamp"
    return df


class FakeEngine:
    """Just enough for initialize()'s warmup — no order placement needed
    here since the test drives _slot_quarantined()/_apply_retune() directly,
    not a live signal-to-order cycle."""

    def __init__(self):
        self.bars = _make_bars()

    def connect(self):
        return AccountInfo(1, "srv", 10_000.0, 10_000.0, 0, 10_000.0, "USD", True, 2)

    def disconnect(self):
        pass

    def get_bars(self, symbol, timeframe, count):
        return self.bars.tail(count)


def _cfg(**overrides):
    capital = {
        "risk_usd_per_trade": 100, "daily_stop_loss_usd": -1_000_000,
        "max_total_drawdown_usd": -1_000_000, "max_positions_per_config": 2,
        "max_concurrent_positions_total": 6, "trail_start_r": 0.0,
        "slot_health": {"enabled": True, "window_trades": 3, "quarantine_below_net_r": -2.0,
                        "cooldown_hours": 24},
        "auto_retune": {"enabled": True, "step_pct": 0.20, "floor_frac": 0.30, "max_retunes": 5},
    }
    capital.update(overrides)
    return {
        "capital": capital,
        "configs": [{
            "name": "t_eurusd_donchian", "symbol": "EURUSD",
            "strategy": "DonchianBreakout", "timeframe": "30m", "magic_number": 9001,
            "strategy_params": {"donchian": {"period": 10}, "exit": {"tp_r_multiple": 3.0}},
        }],
        "watchdog": {"enabled": True, "stale_tick_s": 300},
    }


def _mk_runner(tmp_path, fake, ledger, cfg=None, name="tbook"):
    st = BotState(tmp_path / f"state_{name}.json")
    return PortfolioRunner(cfg or _cfg(), fake, ledger, control_path=None, state=st, book_name=name)


def test_quarantine_trip_also_retunes_tp_r_multiple(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, name="a")
    r.initialize()
    slot = r.slots[0]
    assert slot.strategy.tp_r_multiple == 3.0

    # Feed a losing rolling window (3 trades, net -3R) to trip the -2R/3-trade
    # quarantine threshold from _cfg() above.
    for pnl in (-100.0, -100.0, -100.0):
        r._health[slot.name].recent_r.append(pnl / r.risk_usd)

    assert r._slot_quarantined(slot.name) is True
    assert slot.strategy.tp_r_multiple == pytest.approx(3.0 * 0.80)

    saved = r.state.get("auto_retune")
    assert saved[slot.name]["retune_count"] == 1
    assert saved[slot.name]["original_tp_r"] == 3.0


def test_retuned_value_survives_a_restart(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r1 = _mk_runner(tmp_path, fake, led, name="a")
    r1.initialize()
    slot = r1.slots[0]
    for pnl in (-100.0, -100.0, -100.0):
        r1._health[slot.name].recent_r.append(pnl / r1.risk_usd)
    r1._slot_quarantined(slot.name)
    assert slot.strategy.tp_r_multiple == pytest.approx(3.0 * 0.80)

    # --- simulated crash: brand-new runner, same state path ---
    r2 = _mk_runner(tmp_path, fake, led, name="a")
    r2.initialize()
    slot2 = r2.slots[0]
    assert slot2.strategy.tp_r_multiple == pytest.approx(3.0 * 0.80), (
        "restart must reapply the persisted retune, not reset to the yaml's original value"
    )
    assert r2._retune[slot.name].retune_count == 1


def test_auto_retune_disabled_leaves_tp_r_multiple_untouched(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, cfg=_cfg(auto_retune={"enabled": False}), name="a")
    r.initialize()
    slot = r.slots[0]
    for pnl in (-100.0, -100.0, -100.0):
        r._health[slot.name].recent_r.append(pnl / r.risk_usd)

    assert r._slot_quarantined(slot.name) is True  # quarantine still fires
    assert slot.strategy.tp_r_multiple == 3.0  # but the parameter never moves
