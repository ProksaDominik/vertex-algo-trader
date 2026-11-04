"""Phases 2-4 regression tests: state, FTMO guard, sizing, and the
restart-mid-position / guard-flatten integration scenarios via a FakeEngine.
Pure Python — no MT5 terminal required.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.ftmo_guard import FTMOGuard  # noqa: E402
from metatraderbot.ledger import Ledger  # noqa: E402
from metatraderbot.mt5_engine import AccountInfo, OrderResult  # noqa: E402
from metatraderbot.portfolio_runner import PortfolioRunner  # noqa: E402
from metatraderbot.state import BotState  # noqa: E402


# --- BotState ------------------------------------------------------------

def test_state_roundtrip_and_corrupt_recovery(tmp_path):
    p = tmp_path / "state.json"
    s = BotState(p)
    s.set("initial_balance_anchor", 10_000.0)
    s.set_halt(reason="test", scope="daily", day="2026-07-21")
    s2 = BotState(p)
    assert s2.get("initial_balance_anchor") == 10_000.0
    assert s2.halted
    s2.clear_halt()
    assert not BotState(p).halted

    p.write_text("{ not json !!!", encoding="utf-8")
    s3 = BotState(p)  # must not raise
    assert not s3.halted
    assert p.with_suffix(".json.corrupt").exists()


# --- FTMO guard ----------------------------------------------------------

def _guard(tmp_path, **kw):
    st = BotState(tmp_path / "s.json")
    defaults = dict(account_size=10_000.0, state=st, scope="book",
                    daily_trigger_pct=3.5, overall_trigger_pct=8.0,
                    weekend_flat=False)
    defaults.update(kw)
    return FTMOGuard(**defaults), st


def test_guard_daily_trip_includes_floating(tmp_path):
    g, _ = _guard(tmp_path)
    wed = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)
    g.roll_daily_anchor(realized_alltime=0.0, equity_now=None, now=wed)
    # -200 realized + -160 floating = -360 > -350 trigger -> trip
    d = g.check(realized_alltime=-200.0, floating=-160.0, now=wed)
    assert d.action == "flatten_halt" and d.scope == "daily"
    # Floating alone must be able to trip it (AUDIT C3).
    g2, _ = _guard(tmp_path / "b")
    g2.roll_daily_anchor(0.0, None, now=wed)
    d2 = g2.check(realized_alltime=0.0, floating=-400.0, now=wed)
    assert d2.action == "flatten_halt" and d2.scope == "daily"


def test_guard_daily_resets_at_prague_midnight(tmp_path):
    g, st = _guard(tmp_path)
    # 21:50 UTC Tuesday = 23:50 Prague (CEST). -300 on the day: no trip yet.
    t1 = datetime(2026, 7, 21, 21, 50, tzinfo=timezone.utc)
    g.roll_daily_anchor(0.0, None, now=t1)
    assert g.check(realized_alltime=-300.0, floating=0.0, now=t1).action == "none"
    # 22:10 UTC = 00:10 Prague Wednesday -> new day, anchor rolls to -300.
    t2 = datetime(2026, 7, 21, 22, 10, tzinfo=timezone.utc)
    assert g.check(realized_alltime=-300.0, floating=0.0, now=t2).action == "none"
    assert (st.get("daily") or {}).get("realized_at_start") == -300.0
    # Another -300 within the NEW day trips it.
    d = g.check(realized_alltime=-600.0, floating=-60.0, now=t2 + timedelta(hours=2))
    assert d.action == "flatten_halt" and d.scope == "daily"


def test_guard_daily_halt_autoclears_next_day(tmp_path):
    g, st = _guard(tmp_path)
    t1 = datetime(2026, 7, 21, 10, 0, tzinfo=timezone.utc)
    g.roll_daily_anchor(0.0, None, now=t1)
    st.set_halt(reason="daily", scope="daily", day=g.prague_day(t1))
    assert g.check(realized_alltime=-360.0, floating=0.0, now=t1).action == "block"
    t2 = t1 + timedelta(days=1)
    assert g.check(realized_alltime=-360.0, floating=0.0, now=t2).action == "none"
    assert not st.halted


def test_guard_total_floor_and_manual_halt(tmp_path):
    g, st = _guard(tmp_path)
    now = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)
    d = g.check(realized_alltime=-700.0, floating=-150.0, now=now)  # -850 <= -800
    assert d.action == "flatten_halt" and d.scope == "total"
    st.set_halt(reason=d.reason, scope="total")
    nxt = now + timedelta(days=3)
    assert g.check(realized_alltime=-850.0, floating=0.0, now=nxt).action == "block"


def test_guard_weekend_and_news_windows(tmp_path):
    g, _ = _guard(tmp_path, weekend_flat=True,
                  news_blackout=[{"weekday": 4, "start": "12:00", "end": "14:00"}])
    fri_news = datetime(2026, 7, 24, 12, 30, tzinfo=timezone.utc)
    assert g.in_news_blackout(fri_news)
    fri_late = datetime(2026, 7, 24, 21, 0, tzinfo=timezone.utc)
    assert g.in_weekend_flat(fri_late)
    sat = datetime(2026, 7, 25, 9, 0, tzinfo=timezone.utc)
    assert g.in_weekend_flat(sat)
    mon = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
    assert not g.in_weekend_flat(mon)


# --- FakeEngine + runner integration -------------------------------------

def make_bars(n=301, base=1.10, breakout_at_last_closed=True):
    ts = pd.date_range("2026-07-20 00:00", periods=n, freq="30min", tz="UTC")
    rows = []
    for i in range(n):
        o = c = base
        h, lo = base + 0.0005, base - 0.0005
        if breakout_at_last_closed and i == n - 2:  # last CLOSED bar
            c = base + 0.0050
            h = c + 0.0002
        rows.append({"open": o, "high": h, "low": lo, "close": c, "volume": 100})
    df = pd.DataFrame(rows, index=ts)
    df.index.name = "timestamp"
    return df


class FakeEngine:
    def __init__(self):
        self.positions: dict[int, SimpleNamespace] = {}
        self.closed_deals: dict[int, dict] = {}
        self.closed_calls: list[int] = []
        self._next = 101
        self.bars = make_bars()
        # Mirrors MT5Engine.is_tradeable. Flip to False to simulate the
        # rollover window, where daily-bar signals are born and where
        # order_send returns 10018 MARKET_CLOSED.
        self.tradeable = True

    def is_tradeable(self, symbol):
        return self.tradeable

    def connect(self):
        return self.account_info()

    def disconnect(self):
        pass

    def account_info(self):
        return AccountInfo(1, "srv", 10_000.0, 10_000.0, 0, 10_000.0, "USD", True, 2)

    def get_bars(self, symbol, timeframe, count):
        return self.bars.tail(count)

    def get_positions(self, magic):
        return [p for p in self.positions.values() if p.magic == magic]

    def get_all_open_tickets(self):
        return set(self.positions.keys())

    def place_market_order(self, symbol, direction, lots, sl, tp, magic,
                           comment="", signal_entry_price=None, **kw):
        t = self._next
        self._next += 1
        price = signal_entry_price or 1.105
        self.positions[t] = SimpleNamespace(
            ticket=t, magic=magic, symbol=symbol, volume=lots, price_open=price,
            price_current=price, sl=sl, tp=tp, type=0 if direction == "long" else 1,
            time=int(datetime.now(timezone.utc).timestamp()), profit=0.0, swap=0.0)
        return OrderResult(t, lots, price, sl, tp, 10009, "done")

    def close_position(self, ticket, **kw):
        self.closed_calls.append(ticket)
        return self.close_ticket(ticket, profit=self.positions[ticket].profit) \
            if ticket in self.positions else False

    def close_ticket(self, ticket, profit):
        p = self.positions.pop(ticket, None)
        if p is None:
            return False
        self.closed_deals[ticket] = {
            "exit_price": p.price_open, "pnl_usd": profit,
            "exit_reason": "manual", "closed_ts": datetime.now(timezone.utc)}
        return True

    def get_closed_deal(self, ticket, offset_fallback=0):
        return self.closed_deals.get(ticket)

    def modify_sltp(self, ticket, sl, tp):
        p = self.positions.get(ticket)
        if p is None:
            return False
        p.sl, p.tp = sl, tp
        return True

    def usd_per_point(self, symbol):
        return 100_000.0

    def volume_spec(self, symbol):
        return (0.01, 100.0, 0.01)

    def stops_min_distance(self, symbol):
        return 0.0

    def terminal_health(self, sample_symbol=None):
        return {"connected": True, "trade_allowed": True, "tick_age_s": 1.0}

    def broker_offset_seconds(self, symbol, fallback=0):
        return 0

    def get_spread(self, symbol):
        return getattr(self, "_spread", 0.0001)


def base_cfg(**capital_extra):
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
            "name": "t_eurusd_donchian", "symbol": "EURUSD",
            "strategy": "DonchianBreakout", "timeframe": "30m", "magic_number": 9001,
            "strategy_params": {"donchian": {"period": 10}, "exit": {"tp_r_multiple": 3.0}},
        }],
        "watchdog": {"enabled": True, "stale_tick_s": 300},
    }


def _mk_runner(tmp_path, fake, ledger, cfg=None, name="tbook"):
    st = BotState(tmp_path / f"state_{name}.json")
    return PortfolioRunner(cfg or base_cfg(), fake, ledger,
                           control_path=None, state=st, book_name=name)


def test_restart_mid_position_reconciles(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")

    r1 = _mk_runner(tmp_path, fake, led, name="a")
    r1.initialize()
    slot = r1.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")  # force processing
    r1._guard_tick()
    r1._process_slot(slot)
    assert len(fake.positions) == 1, "breakout should have opened a position"
    ticket = next(iter(fake.positions))
    assert ticket in slot.tracked

    # --- simulated crash: brand-new runner, same ledger + state path ---
    r2 = _mk_runner(tmp_path, fake, led, name="a")
    r2.initialize()
    slot2 = r2.slots[0]
    assert slot2.tracked == {}
    r2._manage_positions(slot2, None)
    assert ticket in slot2.tracked, "restart must adopt the live position"

    # --- position closes while bot runs; reconcile writes the ledger ---
    fake.close_ticket(ticket, profit=57.0)
    r2._manage_positions(slot2, None)
    assert led.total_realized() == pytest.approx(57.0)
    assert ticket not in slot2.tracked


def test_reconcile_requeues_until_deal_available(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)
    ticket = next(iter(fake.positions))
    # Vanish WITHOUT deal history (lagging broker).
    fake.positions.pop(ticket)
    r._manage_positions(slot, None)
    assert ticket in slot.tracked, "must re-queue, not drop (AUDIT H4)"
    # Deal shows up later -> reconciled.
    fake.closed_deals[ticket] = {"exit_price": 1.1, "pnl_usd": -100.0,
                                 "exit_reason": "sl",
                                 "closed_ts": datetime.now(timezone.utc)}
    r._manage_positions(slot, None)
    assert ticket not in slot.tracked
    assert led.total_realized() == pytest.approx(-100.0)


def test_delayed_reconciliation_records_the_real_close_time_not_now(tmp_path):
    """Regression for a live incident: the bot's machine slept for ~4.5h
    (laptop lid closed), so _manage_positions didn't run and notice the
    position had closed until it woke back up. MT5's own broker-side SL
    had actually fired within ~20 minutes, correctly -- but the ledger
    stamped closed_ts as the moment reconciliation finally happened (hours
    late), not the real exit time from the deal record. That stretched the
    dashboard's replay window to match, making a fast, correct SL fill
    look like it "wandered" for hours before quitting. closed_ts must come
    from the deal's own real close time, not wall-clock now()."""
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)
    ticket = next(iter(fake.positions))

    # The position closes (broker-side SL) at a real time in the past...
    real_close_time = datetime.now(timezone.utc) - timedelta(hours=4, minutes=33)
    fake.positions.pop(ticket)
    fake.closed_deals[ticket] = {
        "exit_price": 1.09, "pnl_usd": -86.14, "exit_reason": "sl",
        "closed_ts": real_close_time,
    }
    # ...but the bot only gets around to noticing/reconciling it "now"
    # (simulating the machine having been asleep in between).
    r._manage_positions(slot, None)

    row = led.conn.execute(
        "SELECT closed_ts FROM closed_positions WHERE ticket = ?", (ticket,)).fetchone()
    assert row is not None
    recorded = datetime.fromisoformat(row[0])
    assert recorded == real_close_time, (
        f"closed_ts must be the real MT5 deal time ({real_close_time.isoformat()}), "
        f"not the reconciliation wall-clock time ({row[0]})"
    )


def test_orphaned_ticket_reconciled_even_without_a_tracking_slot(tmp_path):
    """Regression: a live incident where 2 profitable positions the user
    manually closed never showed up on the dashboard, because their
    strategy leg (Donchian) had been removed from the config in a roster
    swap -- no slot existed anymore to notice the ticket had closed.
    _reconcile_orphaned_tickets must catch these from the ledger's own
    order history instead of relying on a live slot's `tracked` dict."""
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, name="orphan")
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)
    ticket = next(iter(fake.positions))
    assert ticket in slot.tracked

    # Simulate: on the next restart the owning strategy leg is gone from
    # config, so this slot (and its `tracked` entry) no longer exists --
    # only the original order row in the ledger remains.
    slot.tracked.pop(ticket)

    # The user closes the now-orphaned position.
    fake.close_ticket(ticket, profit=42.0)

    r._reconcile_orphaned_tickets()

    assert led.total_realized() == pytest.approx(42.0)
    row = led.conn.execute(
        "SELECT exit_reason FROM closed_positions WHERE ticket = ?", (ticket,)).fetchone()
    assert row is not None and row[0] == "manual"

    # Idempotent: running it again must not double-count the same close.
    r._reconcile_orphaned_tickets()
    assert led.total_realized() == pytest.approx(42.0)


def test_orphaned_ticket_still_open_is_not_falsely_closed(tmp_path):
    """The still-open guard matters: get_closed_deal on a position with no
    exit deal yet would misread its ENTRY deal as the exit (0 P&L, wrong
    price) -- must check get_all_open_tickets first and skip anything
    still genuinely open."""
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, name="orphan2")
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)
    ticket = next(iter(fake.positions))
    slot.tracked.pop(ticket)  # orphaned, but still genuinely open

    r._reconcile_orphaned_tickets()

    assert led.total_realized() == 0.0
    row = led.conn.execute(
        "SELECT 1 FROM closed_positions WHERE ticket = ?", (ticket,)).fetchone()
    assert row is None


def test_guard_flattens_and_halts_on_floating_loss(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["ftmo"] = {"enabled": True, "scope": "book", "account_size_usd": 10_000,
                   "daily_trigger_pct": 3.5, "overall_trigger_pct": 8.0,
                   "weekend_flat": False}
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")
    r._guard_tick()
    r._process_slot(slot)
    ticket = next(iter(fake.positions))
    fake.positions[ticket].profit = -400.0  # floating breach of -3.5%/-$350
    r._guard_tick()
    assert ticket in fake.closed_calls, "guard must FLATTEN on trip (AUDIT C2)"
    assert r.state.halted
    assert r._entries_blocked
    # And the halt survives a restart.
    r2 = _mk_runner(tmp_path, fake, led, cfg=cfg)
    assert r2.state.halted


def test_missing_sl_is_repaired(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    t = 501
    fake.positions[t] = SimpleNamespace(
        ticket=t, magic=slot.magic, symbol="EURUSD", volume=0.5, price_open=1.10,
        price_current=1.10, sl=0.0, tp=0.0, type=0,
        time=int(datetime.now(timezone.utc).timestamp()), profit=0.0, swap=0.0)
    r._manage_positions(slot, None)
    assert fake.positions[t].sl > 0, "adopted position must get a stop (AUDIT H5)"
    # Reconstructed 1R stop: risk 100 / (0.5 lots * 100k) = 0.002 below entry.
    assert fake.positions[t].sl == pytest.approx(1.10 - 0.002, abs=1e-6)


def test_percent_equity_sizing_compounds(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg(risk_mode="percent_equity", risk_pct=4.0, account_size_usd=200)
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._realized_alltime = 0.0
    assert r._current_risk_usd() == pytest.approx(8.0)      # 4% of 200
    r._realized_alltime = 300.0
    assert r._current_risk_usd() == pytest.approx(20.0)     # 4% of 500 — compounds
    r._realized_alltime = -150.0
    assert r._current_risk_usd() == pytest.approx(2.0)      # 4% of 50 — shrinks
    r._realized_alltime = -250.0
    assert r._current_risk_usd() == 0.0                     # book gone -> no risk


def test_adaptive_risk_taper_shrinks_near_either_barrier(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg(risk_mode="percent_equity", risk_pct=10.0, account_size_usd=200,
                   adaptive_risk={"enabled": True, "mode": "symmetric", "band_pct": 10.0,
                                  "floor_frac": 0.15, "target_pct": 10.0, "loss_pct": 10.0})
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()

    r._realized_alltime = 0.0
    assert r._current_risk_usd() == pytest.approx(20.0)   # far from both barriers -> full 10% of 200

    r._realized_alltime = 15.0     # book_equity 215 (base 21.5, compounds); eq_frac 1.075 is
    assert r._current_risk_usd() == pytest.approx(5.375)  # 0.025 from +10% target (band 0.10) -> 25% of 21.5

    r._realized_alltime = -15.0    # book_equity 185 (base 18.5); eq_frac 0.925 is 0.025 from the
    assert r._current_risk_usd() == pytest.approx(4.625)  # -10% loss limit -> same 25% (symmetric)

    r._realized_alltime = -19.0    # book_equity 181 (base 18.1); eq_frac 0.905 is 0.005 from the
    assert r._current_risk_usd() == pytest.approx(2.715)  # loss limit -> floor engages at 15%


def test_adaptive_risk_taper_disabled_by_default(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg(risk_mode="percent_equity", risk_pct=10.0, account_size_usd=200)
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._realized_alltime = 15.0
    assert r._current_risk_usd() == pytest.approx(21.5), "no taper unless adaptive_risk.enabled"


def test_circuit_breaker_halts_from_high_water(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg(circuit_breaker={"enabled": True, "halt_drawdown_pct": 35},
                   account_size_usd=200)
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r.ledger.log_closed(opened_ts="t", ticket=1, config_name="x", symbol="EURUSD",
                        direction="long", entry_price=1, exit_price=1, sl=0, tp=0,
                        lots=0.1, pnl_usd=200.0, exit_reason="tp")
    r._guard_tick()   # book equity 400 -> HWM 400
    assert r.state.get("equity_high_water") == pytest.approx(400.0)
    assert not r.state.halted
    r.ledger.log_closed(opened_ts="t", ticket=2, config_name="x", symbol="EURUSD",
                        direction="long", entry_price=1, exit_price=1, sl=0, tp=0,
                        lots=0.1, pnl_usd=-150.0, exit_reason="sl")
    r._guard_tick()   # book equity 250: 37.5% below HWM 400 -> halt
    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "breaker"


def test_watchdog_blocks_entries_when_degraded(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    fake.terminal_health = lambda sample_symbol=None: {
        "connected": True, "trade_allowed": False, "tick_age_s": 1.0}
    r._guard_tick()
    assert r._entries_blocked and "AlgoTrading" in r._entries_blocked
    fake.terminal_health = lambda sample_symbol=None: {
        "connected": True, "trade_allowed": True, "tick_age_s": 1.0}
    r._guard_tick()
    assert r._entries_blocked is None


def test_disabled_slots_are_not_built(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["configs"].append({
        "name": "t_disabled", "symbol": "GBPUSD", "strategy": "DonchianBreakout",
        "timeframe": "30m", "magic_number": 9009, "enabled": False,
        "strategy_params": {"donchian": {"period": 10}, "exit": {"tp_r_multiple": 3.0}},
    })
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    assert [s.name for s in r.slots] == ["t_eurusd_donchian"]


def test_pending_signals_persist_across_restart(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["configs"] = [{
        "name": "t_nowick", "symbol": "XAUUSD", "strategy": "NoWick",
        "timeframe": "30m", "magic_number": 9002,
        "strategy_params": {"exit": {"tp_r_multiple": 3.0}},
    }]
    r = _mk_runner(tmp_path, fake, led, cfg=cfg, name="nw")
    r.initialize()
    from metatraderbot.strategy import Direction, PendingSignal
    r.slots[0].strategy.pending.append(PendingSignal(
        direction=Direction.LONG, entry_level=4000.0, signal_bar_index=250,
        signal_bar_high=4010.0, signal_bar_low=4000.0, bars_since_signal=2))
    r._persist_runtime_state()
    r2 = _mk_runner(tmp_path, fake, led, cfg=cfg, name="nw")
    r2.initialize()
    pend = r2.slots[0].strategy.pending
    assert len(pend) == 1 and pend[0].entry_level == 4000.0 \
        and pend[0].bars_since_signal == 2


def test_new_strategy_pending_signals_persist_across_restart(tmp_path):
    """Regression for a live crash: EMACrossoverStrategy.pending was a bare
    `PendingCross | None` (not a list like every other stateful strategy),
    so _persist_runtime_state's `for s in sigs` blew up with
    "'PendingCross' object is not iterable" inside _heartbeat and took the
    bot down. Also covers the adjacent bug where restart-restore hardcoded
    NoWick's PendingSignal shape, silently dropping BOS/SRRetest pending
    setups instead of restoring them. All 3 new strategies must round-trip
    a pending setup through persist -> restart -> restore intact."""
    from metatraderbot.strategies_bos_pullback import PendingBOS
    from metatraderbot.strategies_ema_cross import PendingCross
    from metatraderbot.strategies_sr_retest import PendingRetest
    from metatraderbot.strategy import Direction

    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["configs"] = [
        {"name": "t_bos", "symbol": "EURUSD", "strategy": "TrendPullbackBOS",
         "timeframe": "30m", "magic_number": 9101,
         "strategy_params": {"exit": {"tp_r_multiple": 2.0}}},
        {"name": "t_sr", "symbol": "EURUSD", "strategy": "SRRetest",
         "timeframe": "30m", "magic_number": 9102,
         "strategy_params": {"exit": {"tp_r_multiple": 2.0}}},
        {"name": "t_ema", "symbol": "EURUSD", "strategy": "EMACross",
         "timeframe": "30m", "magic_number": 9103,
         "strategy_params": {"exit": {"tp_r_multiple": 2.0}}},
    ]
    r = _mk_runner(tmp_path, fake, led, cfg=cfg, name="newstrats")
    r.initialize()

    bos_slot = next(s for s in r.slots if s.strategy_name == "TrendPullbackBOS")
    sr_slot = next(s for s in r.slots if s.strategy_name == "SRRetest")
    ema_slot = next(s for s in r.slots if s.strategy_name == "EMACross")

    bos_slot.strategy.pending.append(PendingBOS(
        direction=Direction.LONG, bos_level=1.1050, swing_extreme=1.1000, bars_since_bos=3))
    sr_slot.strategy.pending.append(PendingRetest(
        direction=Direction.SHORT, level=1.1080, bars_since_break=5))
    ema_slot.strategy.pending.append(PendingCross(
        direction=Direction.LONG, bars_since_cross=2))

    # This exact call is what crashed live before EMACross.pending became a list.
    r._persist_runtime_state()

    r2 = _mk_runner(tmp_path, fake, led, cfg=cfg, name="newstrats")
    r2.initialize()
    bos2 = next(s for s in r2.slots if s.strategy_name == "TrendPullbackBOS")
    sr2 = next(s for s in r2.slots if s.strategy_name == "SRRetest")
    ema2 = next(s for s in r2.slots if s.strategy_name == "EMACross")

    assert len(bos2.strategy.pending) == 1
    assert bos2.strategy.pending[0].bos_level == 1.1050
    assert bos2.strategy.pending[0].bars_since_bos == 3

    assert len(sr2.strategy.pending) == 1
    assert sr2.strategy.pending[0].level == 1.1080
    assert sr2.strategy.pending[0].direction == Direction.SHORT

    assert len(ema2.strategy.pending) == 1
    assert ema2.strategy.pending[0].bars_since_cross == 2


def test_k11_daily_trade_cap_blocks_further_entries(tmp_path):
    """CLAUDE.md kill switch K11: trade_count_today >= N -> HALT new
    entries until rollover, incremented at order-send. Must block the
    (N+1)th entry attempt the same day, but not touch position management
    for trades already open."""
    from metatraderbot.strategy import Direction, TradeSpec

    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["max_trades_per_day"] = 2
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    def mk_trade(price):
        return TradeSpec(direction=Direction.LONG, entry_price=price,
                          stop_loss=price - 0.002, take_profit=price + 0.004,
                          lots=0.1, signal_bar_index=300)

    r._execute_trade(slot, mk_trade(1.1000), bar_index=300, bar_ts="t1")
    r._execute_trade(slot, mk_trade(1.1010), bar_index=301, bar_ts="t2")
    assert len(fake.positions) == 2
    assert r._trades_today() == 2

    r._execute_trade(slot, mk_trade(1.1020), bar_index=302, bar_ts="t3")
    assert len(fake.positions) == 2, "3rd entry the same day must be blocked by K11"
    assert r._trades_today() == 2


def test_closed_market_holds_the_signal_instead_of_burning_it(tmp_path):
    """A signal hitting a closed market must NOT be consumed.

    Regression: daily-bar signals are born at the bar close (21:00-22:00 UTC),
    which is the broker rollover window. order_send returned 10018
    MARKET_CLOSED, `if result.ok:` did nothing, but the signal had ALREADY been
    logged -- so already_have_signal() suppressed that bar forever and the
    trade was lost. Live, the YOLO book placed 2 orders in 4 days and both
    died this way, which is why it looked like it never traded.
    """
    from metatraderbot.strategy import Direction, TradeSpec

    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, cfg=base_cfg())
    r.initialize()
    slot = r.slots[0]
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.1000,
                      stop_loss=1.0980, take_profit=1.1060, lots=0.1,
                      signal_bar_index=300)

    # market shut -> no order, and crucially no signal recorded. Raises
    # _MarketClosedRetry so the caller (_process_slot) knows to hold the
    # bar instead of marking it seen -- see test below for the loop-level
    # regression this guards (last_bar_ts advancing regardless of outcome).
    from metatraderbot.portfolio_runner import _MarketClosedRetry

    fake.tradeable = False
    with pytest.raises(_MarketClosedRetry):
        r._execute_trade(slot, trade, bar_index=300, bar_ts="bar-1")
    assert len(fake.positions) == 0, "must not trade into a closed market"
    assert not led.already_have_signal(slot.name, "bar-1"), (
        "signal must NOT be consumed — otherwise the bar can never be retried")

    # market reopens, same bar is retried and now fills
    fake.tradeable = True
    r._execute_trade(slot, trade, bar_index=300, bar_ts="bar-1")
    assert len(fake.positions) == 1, "must trade once the market reopens"
    assert led.already_have_signal(slot.name, "bar-1")


def test_process_slot_retries_closed_market_across_polls(tmp_path):
    """Loop-level regression for the same bug as the test above, but
    exercised through _process_slot (the actual poll loop) instead of
    calling _execute_trade directly.

    _execute_trade alone correctly avoids logging the signal when the
    market is shut, but _process_slot used to advance slot.last_bar_ts to
    the new bar's timestamp UNCONDITIONALLY, the moment the bar was first
    seen -- regardless of whether _execute_trade actually resolved it. That
    made `latest_ts <= slot.last_bar_ts` true on the very next poll, so the
    method returned before ever reaching the retry logic again: the "hold
    and retry" from the test above was never actually exercised live. This
    is the gap that let YOLO's signals die silently for a week even after
    the _execute_trade-level fix landed.
    """
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, cfg=base_cfg())
    r.initialize()
    slot = r.slots[0]
    slot.last_bar_ts = pd.Timestamp.min.tz_localize("UTC")  # force processing
    breakout_bar_ts = fake.bars.index[-2]  # last CLOSED bar, per make_bars()

    # market shut on the poll where the breakout bar first appears
    fake.tradeable = False
    r._process_slot(slot)
    assert len(fake.positions) == 0, "must not trade into a closed market"
    assert slot.last_bar_ts < breakout_bar_ts, (
        "last_bar_ts must NOT advance past an unresolved bar — otherwise "
        "the next poll's `latest_ts <= last_bar_ts` check skips retrying it")

    # still shut a poll later -> still held, not silently dropped
    r._process_slot(slot)
    assert len(fake.positions) == 0
    assert slot.last_bar_ts < breakout_bar_ts

    # market reopens -> the SAME bar is retried on the next poll and fills
    fake.tradeable = True
    r._process_slot(slot)
    assert len(fake.positions) == 1, "must trade once the market reopens"
    assert slot.last_bar_ts == breakout_bar_ts


def test_max_hold_days_closes_stale_positions(tmp_path):
    """A leg with exit.max_hold_days must close positions older than that.

    This is the change that makes a 10-30 day FTMO challenge winnable:
    scripts/timestop_race_test.py measured the 30-day pass rate rising from
    31.9% to 43.1% purely because 1-2 day holds free position slots ~10x
    faster than the 18-day median the untimed roster produces.
    """
    from datetime import datetime, timedelta, timezone

    from metatraderbot.strategy import Direction, TradeSpec

    fake = FakeEngine()
    cfg = base_cfg()
    cfg["configs"][0].setdefault("strategy_params", {})
    cfg["configs"][0]["strategy_params"].setdefault("exit", {})["max_hold_days"] = 2
    r = _mk_runner(tmp_path, fake, Ledger(tmp_path / "led.sqlite"), cfg=cfg)
    r.initialize()
    slot = r.slots[0]
    assert slot.max_hold_days == 2, "max_hold_days must reach the slot"

    r._execute_trade(slot, TradeSpec(
        direction=Direction.LONG, entry_price=1.1000, stop_loss=1.0980,
        take_profit=1.1060, lots=0.1, signal_bar_index=300),
        bar_index=300, bar_ts="t1")
    assert len(fake.positions) == 1
    ticket = next(iter(slot.tracked))

    # Fresh position: survives.
    r._manage_positions(slot, None)
    assert len(fake.positions) == 1, "a young position must not be time-stopped"

    # Backdate it past the cap: must be closed.
    old = datetime.now(timezone.utc) - timedelta(days=3)
    slot.tracked[ticket].opened_ts = old.isoformat()
    r._manage_positions(slot, None)
    assert len(fake.positions) == 0, "position older than max_hold_days must close"


def test_max_hold_days_absent_leaves_positions_alone(tmp_path):
    """Without max_hold_days nothing changes — the default is unchanged behaviour."""
    from datetime import datetime, timedelta, timezone

    from metatraderbot.strategy import Direction, TradeSpec

    fake = FakeEngine()
    r = _mk_runner(tmp_path, fake, Ledger(tmp_path / "led2.sqlite"), cfg=base_cfg())
    r.initialize()
    slot = r.slots[0]
    assert slot.max_hold_days is None

    r._execute_trade(slot, TradeSpec(
        direction=Direction.LONG, entry_price=1.1000, stop_loss=1.0980,
        take_profit=1.1060, lots=0.1, signal_bar_index=300),
        bar_index=300, bar_ts="t1")
    ticket = next(iter(slot.tracked))
    slot.tracked[ticket].opened_ts = (
        datetime.now(timezone.utc) - timedelta(days=99)).isoformat()
    r._manage_positions(slot, None)
    assert len(fake.positions) == 1, "no time stop configured -> never closes on age"


def test_daily_stop_action_block_keeps_positions_open(tmp_path):
    """daily_stop_action 'block' must stop NEW entries WITHOUT flattening.

    scripts/day_filter_test.py validated dropping subsequent ENTRIES on a
    losing day (OOS: 800 trades @ +0.255R -> 469 @ +0.495R). It never closed
    live positions early. On a 4-5R book the winners need days to reach
    target, so the legacy 'flatten' behaviour would cut off the very trades
    the edge depends on.
    """
    from metatraderbot.strategy import Direction, TradeSpec

    def mk_trade(price):
        return TradeSpec(direction=Direction.LONG, entry_price=price,
                         stop_loss=price - 0.002, take_profit=price + 0.010,
                         lots=0.1, signal_bar_index=300)

    for action, expect_open in (("block", 1), ("flatten", 0)):
        fake = FakeEngine()
        cfg = base_cfg()
        cfg["capital"]["daily_stop_loss_usd"] = -100.0
        cfg["capital"]["daily_stop_action"] = action
        cfg["capital"]["max_trades_per_day"] = 0        # don't let K11 interfere
        led = Ledger(tmp_path / f"led_{action}.sqlite")
        r = _mk_runner(tmp_path, fake, led, cfg=cfg)
        r.initialize()

        # One open position, and a realised loss that trips the daily stop.
        r._execute_trade(r.slots[0], mk_trade(1.1000), bar_index=300, bar_ts="t1")
        assert len(fake.positions) == 1
        led.log_closed(
            opened_ts=datetime.now(timezone.utc).isoformat(), ticket=99,
            config_name=r.slots[0].name, symbol="EURUSD", direction="long",
            entry_price=1.0, exit_price=0.99, sl=0.98, tp=1.05, lots=0.1,
            pnl_usd=-250.0, exit_reason="sl")

        r._guard_tick()

        assert r.state.halted, f"{action}: daily stop should halt"
        assert len(fake.positions) == expect_open, (
            f"{action}: expected {expect_open} open position(s), "
            f"got {len(fake.positions)}")

        # Either way, no NEW entry may open.
        r._execute_trade(r.slots[0], mk_trade(1.1100), bar_index=301, bar_ts="t2")
        assert len(fake.positions) == expect_open, f"{action}: new entry leaked through"


def test_skip_weekdays_blocks_entries_on_listed_days(tmp_path, monkeypatch):
    """skip_weekdays blocks NEW entries on the listed weekdays (Mon=0..Sun=6).

    Motivation: a 15m->1d resample gives Sunday a ~12-bar stub instead of 96,
    so RSI/swing-stop inputs are computed from ~3 hours of thin trade. Sunday
    entries measured negative in BOTH halves of a 10.5-year sample.
    """
    from datetime import datetime, timezone

    from metatraderbot import portfolio_runner as pr
    from metatraderbot.strategy import Direction, TradeSpec

    def mk_trade(price):
        return TradeSpec(direction=Direction.LONG, entry_price=price,
                         stop_loss=price - 0.002, take_profit=price + 0.004,
                         lots=0.1, signal_bar_index=300)

    class FrozenDatetime(datetime):
        _now = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)  # a Sunday

        @classmethod
        def now(cls, tz=None):
            return cls._now

    monkeypatch.setattr(pr, "datetime", FrozenDatetime)

    # Sunday, and Sunday is in skip_weekdays -> blocked.
    fake = FakeEngine()
    cfg = base_cfg()
    cfg["capital"]["skip_weekdays"] = [6]
    r = _mk_runner(tmp_path, fake, Ledger(tmp_path / "a.sqlite"), cfg=cfg)
    r.initialize()
    assert FrozenDatetime._now.weekday() == 6
    r._execute_trade(r.slots[0], mk_trade(1.1000), bar_index=300, bar_ts="t1")
    assert len(fake.positions) == 0, "Sunday entry must be blocked"

    # Same instant, but Sunday is NOT listed -> allowed.
    fake2 = FakeEngine()
    cfg2 = base_cfg()
    cfg2["capital"]["skip_weekdays"] = [0]  # only Mondays blocked
    r2 = _mk_runner(tmp_path, fake2, Ledger(tmp_path / "b.sqlite"), cfg=cfg2)
    r2.initialize()
    r2._execute_trade(r2.slots[0], mk_trade(1.1000), bar_index=300, bar_ts="t1")
    assert len(fake2.positions) == 1, "unlisted weekday must trade normally"


def test_skip_weekdays_defaults_to_empty(tmp_path):
    """Absent from config, the gate must not change existing behaviour."""
    fake = FakeEngine()
    r = _mk_runner(tmp_path, fake, Ledger(tmp_path / "led.sqlite"), cfg=base_cfg())
    assert r.skip_weekdays == set()


def test_k11_disabled_by_default(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)  # base_cfg doesn't set max_trades_per_day
    r.initialize()
    assert r.max_trades_per_day == 0
    slot = r.slots[0]
    from metatraderbot.strategy import Direction, TradeSpec
    for i, price in enumerate([1.10, 1.11, 1.12, 1.13]):
        trade = TradeSpec(direction=Direction.LONG, entry_price=price,
                           stop_loss=price - 0.002, take_profit=price + 0.004,
                           lots=0.1, signal_bar_index=300 + i)
        r._execute_trade(slot, trade, bar_index=300 + i, bar_ts=f"t{i}")
    assert len(fake.positions) == 4, "max_trades_per_day=0 must mean unlimited"


def test_k2_weekly_stop_blocks_new_entries(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["weekly_stop_pct"] = 0.04
    cfg["capital"]["account_size_usd"] = 10_000
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._guard_tick()
    assert not r._entries_blocked

    led.log_closed(opened_ts="2026-01-01T00:00:00+00:00", ticket=1, config_name="x",
                    symbol="EURUSD", direction="long", entry_price=1.1, exit_price=1.09,
                    sl=1.09, tp=1.12, lots=1.0, pnl_usd=-500.0, exit_reason="sl")
    r._guard_tick()
    assert r._entries_blocked
    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "k2_weekly"


def test_k2_weekly_halt_autoclears_next_week(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["weekly_stop_pct"] = 0.04
    cfg["capital"]["account_size_usd"] = 10_000
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r.state.set("weekly_pnl_anchor", {"week": "2020-W01", "realized_at_start": 0.0})
    r.state.set_halt(reason="old", scope="k2_weekly", day="2020-W01")
    r._guard_tick()
    assert not r.state.halted


def test_k4_consecutive_losses_halve_risk_until_recovery(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k4_consecutive_loss_halving_enabled"] = True
    cfg["capital"]["k4_consecutive_loss_trigger"] = 2
    cfg["capital"]["k4_recovery_wins_needed"] = 2
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    base_risk = r._current_risk_usd()

    r._k4_record_trade_result(-10.0)
    assert r._current_risk_usd() == pytest.approx(base_risk)
    r._k4_record_trade_result(-10.0)
    assert r._current_risk_usd() == pytest.approx(base_risk * 0.5)

    r._k4_record_trade_result(10.0)
    assert r._current_risk_usd() == pytest.approx(base_risk * 0.5)
    r._k4_record_trade_result(10.0)
    assert r._current_risk_usd() == pytest.approx(base_risk)


def test_k4_loss_during_recovery_resets_the_win_streak(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k4_consecutive_loss_halving_enabled"] = True
    cfg["capital"]["k4_consecutive_loss_trigger"] = 2
    cfg["capital"]["k4_recovery_wins_needed"] = 2
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    base_risk = r._current_risk_usd()
    r._k4_record_trade_result(-10.0)
    r._k4_record_trade_result(-10.0)  # now in recovery, halved
    r._k4_record_trade_result(10.0)   # 1 win
    r._k4_record_trade_result(-10.0)  # a loss resets the win streak, stays halved
    assert r._current_risk_usd() == pytest.approx(base_risk * 0.5)
    r._k4_record_trade_result(10.0)
    r._k4_record_trade_result(10.0)   # 2 consecutive wins now -> restored
    assert r._current_risk_usd() == pytest.approx(base_risk)


def test_k6_spread_anomaly_blocks_that_symbol_only(tmp_path):
    fake = FakeEngine()
    fake._spread = 0.0001
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k6_spread_anomaly_enabled"] = True
    cfg["capital"]["k6_spread_multiple"] = 5.0
    cfg["capital"]["k6_persist_s"] = 0.0
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    for i in range(5):
        trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                           take_profit=1.104, lots=0.1, signal_bar_index=300 + i)
        r._execute_trade(slot, trade, bar_index=300 + i, bar_ts=f"t{i}")
    n_before = len(fake.positions)
    assert n_before == 5

    fake._spread = 0.01  # ~100x the baseline -- a real spread blowout
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=310)
    r._execute_trade(slot, trade, bar_index=310, bar_ts="t5")
    assert len(fake.positions) == n_before, "signal during a spread anomaly must be skipped"


def test_k7_closes_and_halts_when_sl_repair_fails(tmp_path):
    fake = FakeEngine()
    orig_place = fake.place_market_order

    def place_no_sl(*a, **kw):
        res = orig_place(*a, **kw)
        fake.positions[res.ticket].sl = 0  # simulate the broker dropping the SL on fill
        return res
    fake.place_market_order = place_no_sl
    fake.modify_sltp = lambda ticket, sl, tp: False  # repair always fails

    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")

    assert len(fake.positions) == 0, "an unrepairable, unprotected position must be closed"
    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "k7_unprotected"


def test_k7_does_not_close_when_repair_succeeds(tmp_path):
    fake = FakeEngine()
    orig_place = fake.place_market_order

    def place_no_sl(*a, **kw):
        res = orig_place(*a, **kw)
        fake.positions[res.ticket].sl = 0
        return res
    fake.place_market_order = place_no_sl
    # modify_sltp keeps its default (succeeds) behavior.

    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")

    assert len(fake.positions) == 1, "a successfully repaired position must stay open"
    assert not r.state.halted


def test_k8_slippage_recalibration_halts_after_window(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k8_slippage_recalibration_enabled"] = True
    cfg["capital"]["k8_window_trades"] = 3
    cfg["capital"]["k8_slippage_multiple"] = 2.0
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    orig_place = fake.place_market_order

    def place_slippy(symbol, direction, lots, sl, tp, magic, comment="", signal_entry_price=None, **kw):
        res = orig_place(symbol, direction, lots, sl, tp, magic, comment, signal_entry_price, **kw)
        fake.positions[res.ticket].price_open = (signal_entry_price or res.price) + 0.01
        return OrderResult(res.ticket, res.volume, fake.positions[res.ticket].price_open,
                           res.sl, res.tp, res.retcode, res.comment)
    fake.place_market_order = place_slippy

    from metatraderbot.strategy import Direction, TradeSpec
    for i in range(3):
        trade = TradeSpec(direction=Direction.LONG, entry_price=1.10 + i * 0.001,
                           stop_loss=1.098 + i * 0.001, take_profit=1.104 + i * 0.001,
                           lots=0.1, signal_bar_index=300 + i)
        r._execute_trade(slot, trade, bar_index=300 + i, bar_ts=f"t{i}")

    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "k8_slippage"


def test_ntp_zero_transmit_ts_fails_open(monkeypatch):
    """A zero-filled NTP reply must return None, not a 127-year 'drift'.

    Regression: the live FTMO book sat halted on
    'K9 clock drift +3994752758.8s from NTP (limit 5s)' and never placed a
    trade. That number is exactly local_time - (0 - _NTP_EPOCH_OFFSET), i.e.
    a reply whose transmit timestamp was never filled in. K9 read it as real
    drift and halted the book.
    """
    import socket

    from metatraderbot import ntp_check

    class FakeSock:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, t):
            pass

        def sendto(self, *a):
            pass

        def recvfrom(self, n):
            return self.payload, ("127.0.0.1", 123)

    # 48 zero bytes = the malformed reply seen in production.
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSock(b"\0" * 48))
    assert ntp_check.ntp_offset_seconds() is None

    # A well-formed reply still produces a real offset.
    import struct
    import time
    words = [0] * 12
    words[10] = int(time.time()) + ntp_check._NTP_EPOCH_OFFSET
    good = struct.pack("!12I", *words)
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSock(good))
    off = ntp_check.ntp_offset_seconds()
    assert off is not None and abs(off) < 5.0

    # An absurd-but-nonzero timestamp is garbage too, not a year of drift.
    words[10] = ntp_check._NTP_EPOCH_OFFSET + 1  # claims it is 1970
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: FakeSock(struct.pack("!12I", *words)))
    assert ntp_check.ntp_offset_seconds() is None


def test_k9_ntp_drift_halts(tmp_path, monkeypatch):
    import metatraderbot.portfolio_runner as pr
    monkeypatch.setattr(pr, "ntp_offset_seconds", lambda *a, **kw: 10.0)
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k9_ntp_drift_enabled"] = True
    cfg["capital"]["k9_max_drift_s"] = 5.0
    cfg["capital"]["k9_check_interval_s"] = 0.0
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._guard_tick()
    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "k9_clock"


def test_k9_small_drift_does_not_halt(tmp_path, monkeypatch):
    import metatraderbot.portfolio_runner as pr
    monkeypatch.setattr(pr, "ntp_offset_seconds", lambda *a, **kw: 1.0)
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k9_ntp_drift_enabled"] = True
    cfg["capital"]["k9_max_drift_s"] = 5.0
    cfg["capital"]["k9_check_interval_s"] = 0.0
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._guard_tick()
    assert not r.state.halted


def test_k10_lot_anomaly_rejects_and_halts(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["k10_lot_anomaly_enabled"] = True
    cfg["capital"]["k10_window_trades"] = 5
    cfg["capital"]["k10_lot_multiple"] = 3.0
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    for i in range(5):
        trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                           take_profit=1.104, lots=0.1, signal_bar_index=300 + i)
        r._execute_trade(slot, trade, bar_index=300 + i, bar_ts=f"t{i}")
    assert len(fake.positions) == 5, "sanity: the normal-sized trades went through"

    big_trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.09999,
                          take_profit=1.104, lots=0.1, signal_bar_index=310)
    r._execute_trade(slot, big_trade, bar_index=310, bar_ts="t5")

    assert len(fake.positions) == 5, "the oversized order must be rejected, not filled"
    assert r.state.halted
    assert (r.state.get("halt") or {}).get("scope") == "k10_lot"
    row = led.conn.execute(
        "SELECT gate FROM rejections WHERE gate = 'k10_lot_anomaly'").fetchone()
    assert row is not None


def test_cost_gate_rejects_a_stop_too_tight_to_trade_profitably(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["cost_gate_enabled"] = True
    cfg["capital"]["cost_gate_r"] = 0.08
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.09999,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")

    assert len(fake.positions) == 0
    row = led.conn.execute("SELECT reason FROM rejections WHERE gate = 'cost_gate'").fetchone()
    assert row is not None


def test_cost_gate_allows_a_normal_stop(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["cost_gate_enabled"] = True
    cfg["capital"]["cost_gate_r"] = 0.08
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    slot = r.slots[0]

    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")
    assert len(fake.positions) == 1


def test_risk_assertion_does_not_fire_on_normal_sizing(tmp_path):
    """Regression guard: the new `assert actual_risk <= risk_usd*1.02` must
    never trip on ordinary, already-exact sizing math."""
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]
    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")  # must not raise
    assert len(fake.positions) == 1


def test_flatten_positions_below_r_is_selective(tmp_path):
    from metatraderbot.portfolio_runner import TrackedPosition

    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led)
    r.initialize()
    slot = r.slots[0]

    # risk_distance = |1.10 - 1.098| = 0.002; contract_mult(EURUSD) = 100_000
    # -> risk_dollars = 0.002 * 1.0 lots * 100_000 = 200.
    fake.positions[1] = SimpleNamespace(
        ticket=1, magic=slot.magic, symbol=slot.symbol, volume=1.0,
        price_open=1.10, price_current=1.0995, sl=1.098, tp=1.12,
        type=0, time=0, profit=-10.0, swap=0.0)   # -0.05R -> below +1R
    fake.positions[2] = SimpleNamespace(
        ticket=2, magic=slot.magic, symbol=slot.symbol, volume=1.0,
        price_open=1.10, price_current=1.103, sl=1.098, tp=1.12,
        type=0, time=0, profit=250.0, swap=0.0)   # +1.25R -> at/above +1R
    slot.tracked[1] = TrackedPosition(entry=1.10, orig_sl=1.098, orig_tp=1.12, lots=1.0,
                                      is_long=True, opened_ts="2026-01-01T00:00:00+00:00")
    slot.tracked[2] = TrackedPosition(entry=1.10, orig_sl=1.098, orig_tp=1.12, lots=1.0,
                                      is_long=True, opened_ts="2026-01-01T00:00:00+00:00")

    r._flatten_positions_below_r(1.0, "test")
    assert 1 not in fake.positions, "the losing (<1R) position must be closed"
    assert 2 in fake.positions, "the winning (>=1R) position must be left open"


def test_rejection_gates_write_to_the_rejections_table(tmp_path):
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    cfg = base_cfg()
    cfg["capital"]["max_concurrent_positions_total"] = 0  # not the gate under test
    r = _mk_runner(tmp_path, fake, led, cfg=cfg)
    r.initialize()
    r._entries_blocked = "test halt"
    slot = r.slots[0]
    from metatraderbot.strategy import Direction, TradeSpec
    trade = TradeSpec(direction=Direction.LONG, entry_price=1.10, stop_loss=1.098,
                       take_profit=1.104, lots=0.1, signal_bar_index=300)
    r._execute_trade(slot, trade, bar_index=300, bar_ts="t1")

    row = led.conn.execute(
        "SELECT config_name, gate, reason FROM rejections WHERE gate = 'entries_blocked'"
    ).fetchone()
    assert row is not None
    assert row[0] == slot.name


def test_new_strategy_params_reach_the_strategy_instance(tmp_path):
    """The 3 new strategies' YAML strategy_params sub-blocks (bos_pullback /
    sr_retest / ema_cross) must actually reach the strategy constructor --
    _build_strategy_config used to whitelist only the pre-existing strategy
    keys, so these were silently dropped and every instance ran on its
    hardcoded class defaults no matter what the YAML said."""
    fake = FakeEngine()
    led = Ledger(tmp_path / "led.sqlite")
    r = _mk_runner(tmp_path, fake, led, name="cfgpass")

    cfg = r._build_strategy_config("EURUSD", {
        "exit": {"tp_r_multiple": 2.0},
        "bos_pullback": {"htf_multiplier": 6},
        "sr_retest": {"level_lookback": 42},
        "ema_cross": {"fast_period": 8},
    })
    assert cfg["strategy"]["bos_pullback"]["htf_multiplier"] == 6
    assert cfg["strategy"]["sr_retest"]["level_lookback"] == 42
    assert cfg["strategy"]["ema_cross"]["fast_period"] == 8
