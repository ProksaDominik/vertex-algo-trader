"""Contract-multiplier drift safety gate — the XAUUSD incident regression
test: a broker misreporting usd_per_point() must never silently size a real
position off the disputed number. Pure Python — no MT5 terminal required.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.mt5_engine import AccountInfo, OrderResult  # noqa: E402
from metatraderbot.portfolio_runner import PortfolioRunner, WARMUP_BARS  # noqa: E402
from metatraderbot.state import BotState  # noqa: E402


def _make_breakout_bars(n=WARMUP_BARS + 1, base=4100.0):
    ts = pd.date_range("2026-07-20 00:00", periods=n, freq="30min", tz="UTC")
    rows = []
    for i in range(n):
        o = c = base
        h, lo = base + 0.05, base - 0.05
        if i == n - 2:  # breakout on the last CLOSED bar
            c = base + 50.0
            h = c + 0.2
        rows.append({"open": o, "high": h, "low": lo, "close": c, "volume": 100})
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "timestamp"
    return df


class FakeEngine:
    def __init__(self, usd_per_point_value):
        self.bars = _make_breakout_bars()
        self._upp = usd_per_point_value
        self.orders_placed = 0
        self.tradeable = True      # mirrors MT5Engine.is_tradeable

    def is_tradeable(self, symbol):
        return self.tradeable

    def connect(self):
        return self.account_info()

    def disconnect(self):
        pass

    def account_info(self):
        return AccountInfo(1, "srv", 10_000.0, 10_000.0, 0, 10_000.0, "USD", True, 2)

    def terminal_health(self, sample_symbol=None):
        return {"connected": True, "trade_allowed": True, "tick_age_s": 1.0}

    def get_bars(self, symbol, timeframe, count):
        return self.bars.tail(count)

    def get_positions(self, magic):
        return []

    def usd_per_point(self, symbol):
        return self._upp

    def volume_spec(self, symbol):
        return (0.01, 100.0, 0.01)

    def stops_min_distance(self, symbol):
        return 0.0

    def place_market_order(self, symbol, direction, lots, sl, tp, magic,
                           comment="", signal_entry_price=None, **kw):
        self.orders_placed += 1
        return OrderResult(101, lots, signal_entry_price or 4150.0, sl, tp, 10009, "done")


def _cfg(**capital_extra):
    capital = {
        "risk_usd_per_trade": 100, "daily_stop_loss_usd": -1_000_000,
        "max_total_drawdown_usd": -1_000_000, "max_positions_per_config": 2,
        "max_concurrent_positions_total": 6, "trail_start_r": 0.0,
        "slot_health": {"enabled": False},
    }
    capital.update(capital_extra)
    return {
        "capital": capital,
        "configs": [{
            "name": "t_xauusd_donchian", "symbol": "XAUUSD",
            "strategy": "DonchianBreakout", "timeframe": "30m", "magic_number": 9001,
            "strategy_params": {"donchian": {"period": 10}, "exit": {"tp_r_multiple": 3.0}},
        }],
        "watchdog": {"enabled": True, "stale_tick_s": 300},
    }


def _mk_runner(tmp_path, fake, ledger, cfg=None, name="tbook"):
    st = BotState(tmp_path / f"state_{name}.json")
    return PortfolioRunner(cfg or _cfg(), fake, ledger, control_path=None, state=st, book_name=name)


def test_large_drift_refuses_the_trade_instead_of_sizing_off_it(tmp_path):
    """instruments.py says XAUUSD is 100.0 $/point/lot. A 'live' value of
    10.0 (the real IC-Markets-demo incident: misreported trade_tick_value)
    is 90% drift — must skip the signal, not size a position 10x too big."""
    fake = FakeEngine(usd_per_point_value=10.0)
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")  # force processing
    r._guard_tick()
    r._process_slot(slot)

    assert fake.orders_placed == 0, "a 90% multiplier drift must refuse the trade, not size off it"
    assert len(slot.tracked) == 0


def test_small_drift_still_trades_using_live(tmp_path):
    """A modest, plausible live update (e.g. broker revised a spec by 10%)
    must still go through — only large drift is refused."""
    fake = FakeEngine(usd_per_point_value=110.0)  # 10% off instruments.py's 100.0
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)

    assert fake.orders_placed == 1
    assert len(slot.tracked) == 1


def test_drift_threshold_is_configurable(tmp_path):
    fake = FakeEngine(usd_per_point_value=10.0)  # 90% drift
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, cfg=_cfg(max_contract_mult_drift=0.95))
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)

    assert fake.orders_placed == 1, "raising the threshold above the observed drift should allow the trade"
