"""Lucid futures layer: contract table, roll dates, Lucid's session rules, the
contract cap, and NinjaTraderEngine against a fake bridge (no NinjaTrader).

The engine tests pin the safety rules that matter with real money: sizes are
floored to whole contracts, the Lucid cap is enforced inside the engine, stops
on the wrong side never reach the market, a retried entry is recovered rather
than duplicated, and closed-trade P&L includes Lucid's commission.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import futures, lucid_rules  # noqa: E402
from metatraderbot import nt_engine as nt  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS, contract_mult, pip_size  # noqa: E402
from metatraderbot.lucid_rules import Session, session_state, trading_day  # noqa: E402

ET = ZoneInfo("America/New_York")


def et(*args) -> datetime:
    return datetime(*args, tzinfo=ET)


# --- contract table ------------------------------------------------------------

def test_lucid_list_is_the_34_approved_contracts():
    assert len(futures.LUCID_APPROVED) == 34
    for excluded in ("MBT", "BTC", "M6E", "M6B", "ZN", "VX"):
        assert excluded not in futures.LUCID_APPROVED


def test_trial_basket_maps_to_four_lucid_contracts():
    basket = ["GBPCHF", "USDCAD", "GBPUSD", "GBPJPY", "USDCHF", "XAUUSD"]
    mapped = [futures.futures_for_cfd(s) for s in basket if s not in futures.NOT_ON_LUCID]
    assert mapped == ["6C", "6B", "6S", "MGC"]


def test_crypto_and_crosses_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="crypto"):
        futures.futures_for_cfd("BTCUSD")
    with pytest.raises(ValueError, match="cross"):
        futures.futures_for_cfd("GBPJPY")
    with pytest.raises(ValueError, match="6B"):
        futures.spec("GBPUSD")      # a CFD name points at its futures equivalent


def test_cap_units_micro_vs_mini():
    assert futures.units("MGC") == 1
    assert futures.units("MGC 12-26") == 1
    assert futures.units("6B") == 10
    assert futures.units("QM") == 10      # E-mini crude is not a micro
    with pytest.raises(ValueError):
        futures.units("XYZ")             # unknown must not silently count as micro


def test_contracts_for_risk_floors():
    assert futures.contracts_for_risk(1000, 2.5, 100.0) == 4
    assert futures.contracts_for_risk(999, 2.5, 100.0) == 3
    assert futures.contracts_for_risk(100, 2.5, 100.0) == 0
    assert futures.contracts_for_risk(1000, 0.0, 100.0) == 0


def test_futures_are_registered_for_strategies_and_backtests():
    assert PAIR_SPECS["6B"]["contract_mult"] == 62_500.0
    assert PAIR_SPECS["MGC"]["commission_per_lot"] == 0.80     # per side, like the CFD entries
    assert contract_mult("NQ") == 20.0
    assert pip_size("6C") == 0.00005
    assert pip_size("MGC 12-26") == 0.10
    assert pip_size("EURUSD") == 0.0001                       # CFDs unchanged
    assert pip_size("USDJPY") == 0.01


def test_london_breakout_builds_on_a_futures_symbol():
    from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy
    s = LondonBreakoutStrategy({"symbol": "6B", "capital": {"risk_usd_per_trade": 500.0},
                                "strategy": {"exit": {"tp_r_multiple": 2.0}}})
    assert s.contract_mult == 62_500.0
    assert s.pip == 0.0001


# --- front month ----------------------------------------------------------------

@pytest.mark.parametrize("name,today,expected", [
    ("ES", date(2026, 9, 29), "ES 12-26"),
    ("ES", date(2026, 12, 9), "ES 12-26"),
    ("ES", date(2026, 12, 10), "ES 03-27"),       # 2nd Thursday: 8 days before the 3rd Friday
    ("MNQ", date(2026, 12, 10), "MNQ 03-27"),
    ("6B", date(2026, 12, 6), "6B 12-26"),
    ("6B", date(2026, 12, 7), "6B 03-27"),        # a week before the Mon 14 Dec last trade
    ("GC", date(2026, 9, 24), "GC 10-26"),
    ("GC", date(2026, 9, 25), "GC 12-26"),        # 5 days before first notice (30 Sep)
    ("MGC", date(2026, 9, 29), "MGC 12-26"),
    ("MCL", date(2026, 9, 29), "MCL 11-26"),      # Oct contract expired 22 Sep
    ("CL", date(2026, 10, 13), "CL 12-26"),       # Nov contract's last trade is 20 Oct
    ("NG", date(2026, 9, 29), "NG 11-26"),
])
def test_front_contract(name, today, expected):
    assert futures.front_contract(name, today) == expected


def test_agricultural_contracts_need_an_explicit_month():
    with pytest.raises(ValueError, match="explicitly"):
        futures.front_contract("ZC", date(2026, 9, 29))


# --- Lucid session rules (4:45 PM ET flat, 6 PM reopen, DST, holidays) -----------

@pytest.mark.parametrize("when,expected", [
    (et(2026, 9, 29, 14, 0), Session.OPEN),
    (et(2026, 9, 29, 16, 29), Session.OPEN),
    (et(2026, 9, 29, 16, 31), Session.NO_NEW_ENTRIES),
    (et(2026, 9, 29, 16, 41), Session.FLATTEN),
    (et(2026, 9, 29, 16, 46), Session.CLOSED),
    (et(2026, 9, 29, 17, 30), Session.CLOSED),
    (et(2026, 9, 29, 18, 5), Session.OPEN),
    (et(2026, 10, 2, 17, 30), Session.CLOSED),    # Friday evening
    (et(2026, 10, 3, 12, 0), Session.CLOSED),     # Saturday
    (et(2026, 10, 4, 17, 59), Session.CLOSED),    # Sunday before the open
    (et(2026, 10, 4, 18, 0), Session.OPEN),
    (et(2026, 12, 1, 16, 41), Session.FLATTEN),   # winter time (EST) too
    (et(2026, 11, 27, 12, 31), Session.NO_NEW_ENTRIES),   # early close 12:45
    (et(2026, 11, 27, 12, 41), Session.FLATTEN),
    (et(2026, 11, 27, 13, 0), Session.CLOSED),
    (et(2026, 12, 24, 19, 0), Session.CLOSED),    # no evening session before Christmas
    (et(2026, 12, 25, 10, 0), Session.CLOSED),
])
def test_session_state(when, expected):
    assert session_state(when) == expected


def test_session_state_accepts_utc():
    # 20:41 UTC on 29 Sep is 16:41 EDT.
    assert session_state(datetime(2026, 9, 29, 20, 41, tzinfo=ZoneInfo("UTC"))) == Session.FLATTEN


def test_session_state_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        session_state(datetime(2026, 9, 29, 12, 0))


@pytest.mark.parametrize("when,expected", [
    (et(2026, 9, 28, 16, 0), "2026-09-28"),
    (et(2026, 9, 28, 17, 30), "2026-09-29"),
    (et(2026, 10, 2, 18, 0), "2026-10-05"),       # Friday evening belongs to Monday
    (et(2026, 10, 4, 18, 30), "2026-10-05"),      # Sunday's open is Monday's session
])
def test_trading_day(when, expected):
    assert trading_day(when) == expected


def test_contract_cap_accounting():
    used = lucid_rules.open_units({"MGC 12-26": 5, "6B 12-26": -1})
    assert used == 15
    assert lucid_rules.max_addable("MGC", used) == 25
    assert lucid_rules.max_addable("6B", used) == 2
    assert lucid_rules.max_addable("6B", 35) == 0
    assert lucid_rules.LUCIDPRO_50K.max_units == 40


# --- NinjaTraderEngine against a fake bridge ---------------------------------------

class FakeBridge:
    """Answers the bridge commands the engine uses, in memory."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.positions: list[dict] = []
        self.point_values = {"MGC": 10.0, "6B": 62_500.0}
        self.tickets: dict[int, dict] = {}
        self.fill_price = 2400.0
        self.enter_raises = False

    def __call__(self, cmd: str, **args) -> dict:
        self.calls.append((cmd, args))
        if cmd == "instrument":
            root = args["name"].split(" ")[0]
            return {"name": args["name"], "point_value": self.point_values.get(root, 1.0)}
        if cmd == "positions":
            return {"positions": self.positions}
        if cmd == "quote":
            return {"bid": 2399.9, "ask": 2400.0, "last": 2400.0, "age_s": 1.0}
        if cmd == "enter":
            t = {"ticket": args["ticket"], "magic": args["magic"], "instrument": args["instrument"],
                 "side": args["side"], "qty": args["qty"], "entry_state": "Filled",
                 "entry_filled": args["qty"], "entry_avg": self.fill_price,
                 "open_qty": args["qty"], "closed": False, "bracket": "working",
                 "stop": self.fill_price - args["sl_dist"], "target": self.fill_price + args["tp_dist"],
                 "point_value": self.point_values.get(args["instrument"].split(" ")[0], 1.0),
                 "error": ""}
            self.tickets[int(args["ticket"])] = t
            if self.enter_raises:
                raise nt.BridgeError("socket dropped after the order was sent")
            return t
        if cmd == "tickets":
            return {"tickets": list(self.tickets.values())}
        raise AssertionError(f"unexpected bridge command {cmd}")

    def sent(self, cmd: str) -> list[dict]:
        return [a for c, a in self.calls if c == cmd]


@pytest.fixture
def bridge(monkeypatch):
    b = FakeBridge()
    engine = nt.NinjaTraderEngine(account="Sim101", max_contract_units=40, fill_timeout_s=1.0)
    monkeypatch.setattr(engine, "_call", b)
    b.engine = engine
    return b


def test_order_is_floored_to_whole_contracts(bridge):
    r = bridge.engine.place_market_order("MGC", "long", 3.97, sl=2390.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert r.ok and r.volume == 3.0
    (sent,) = bridge.sent("enter")
    assert sent["qty"] == 3
    assert sent["sl_dist"] == pytest.approx(10.0) and sent["tp_dist"] == pytest.approx(20.0)
    assert sent["instrument"].startswith("MGC ")          # a specific contract month, never the bare root


def test_under_one_contract_never_reaches_the_market(bridge):
    r = bridge.engine.place_market_order("MGC", "long", 0.8, sl=2390.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert not r.ok
    assert bridge.sent("enter") == []


def test_stops_on_the_wrong_side_are_refused(bridge):
    r = bridge.engine.place_market_order("MGC", "long", 2, sl=2410.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert not r.ok and "wrong side" in r.comment
    assert bridge.sent("enter") == []


def test_contract_cap_clamps_inside_the_engine(bridge):
    bridge.positions = [{"instrument": "6B 12-26", "side": "long", "qty": 3}]      # 30 of 40 units
    r = bridge.engine.place_market_order("MGC", "long", 20, sl=2390.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert r.ok and r.volume == 10.0
    assert bridge.sent("enter")[0]["qty"] == 10


def test_contract_cap_counts_manual_positions_and_refuses_when_full(bridge):
    bridge.positions = [{"instrument": "6B 12-26", "side": "short", "qty": 4}]     # 40 of 40
    r = bridge.engine.place_market_order("MGC", "long", 1, sl=2390.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert not r.ok and "cap" in r.comment
    assert bridge.sent("enter") == []


def test_entry_that_reached_ninjatrader_is_recovered_not_lost(bridge):
    """The socket failed after NinjaTrader accepted the order: the engine must
    find the ticket by id and report the real position."""
    bridge.enter_raises = True
    r = bridge.engine.place_market_order("MGC", "long", 2, sl=2390.0, tp=2420.0, magic=9123,
                                         signal_entry_price=2400.0)
    assert r.ok and r.volume == 2.0
    assert len(bridge.sent("enter")) == 1          # looked up, never re-sent


def test_point_value_drift_resolves_to_the_smaller_position(bridge):
    bridge.point_values["MGC"] = 1.0               # NinjaTrader says 1, the contract is 10
    assert bridge.engine.usd_per_point("MGC") == pytest.approx(10.0)


def test_point_value_passes_through_when_sources_agree(bridge):
    assert bridge.engine.usd_per_point("6B") == pytest.approx(62_500.0)


def test_closed_deal_pnl_includes_lucid_commission(bridge):
    bridge.tickets[42] = {"ticket": 42, "magic": 9123, "instrument": "MGC 12-26", "side": "long",
                          "entry_filled": 2, "entry_avg": 2400.0, "open_qty": 0, "closed": True,
                          "closed_qty": 2, "exit_avg": 2410.0, "close_reason": "tp",
                          "closed_utc": 1_790_000_000, "point_value": 10.0}
    deal = bridge.engine.get_closed_deal(42)
    # 10 points x $10 x 2 contracts, minus $0.80 per side x 2 sides x 2 contracts
    assert deal["pnl_usd"] == pytest.approx(200.0 - 3.2)
    assert deal["exit_reason"] == "tp"
    assert deal["closed_ts"].tzinfo is not None


def test_closed_deal_never_books_more_contracts_than_were_opened(bridge):
    """The live Sim101 test of 2026-09-29: the bridge counted one exit fill
    twice (closed_qty=2 on a 1-lot) and the P&L came out doubled."""
    bridge.tickets[44] = {"ticket": 44, "magic": 9199, "instrument": "MGC DEC26", "side": "long",
                          "entry_filled": 1, "entry_avg": 4200.0, "open_qty": 0, "closed": True,
                          "closed_qty": 2, "exit_avg": 4199.3, "close_reason": "manual",
                          "closed_utc": 1_790_690_961, "point_value": 10.0}
    deal = bridge.engine.get_closed_deal(44)
    assert deal["pnl_usd"] == pytest.approx(-7.0 - 1.6)       # 1 contract, not 2


def test_open_ticket_has_no_closed_deal(bridge):
    bridge.tickets[43] = {"ticket": 43, "magic": 9123, "instrument": "MGC 12-26", "side": "short",
                          "entry_filled": 1, "entry_avg": 2400.0, "open_qty": 1, "closed": False}
    assert bridge.engine.get_closed_deal(43) is None
    assert bridge.engine.get_closed_deal(0) is None


def test_bars_get_a_forming_stand_in_stamped_from_the_last_closed_bar(bridge, monkeypatch):
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    opens = [now - pd.Timedelta(minutes=15 * k) for k in (3, 2, 1)]    # last one closed at `now`
    rows = [[o.timestamp(), 1.0, 2.0, 0.5, 1.5, 10] for o in opens]
    monkeypatch.setattr(bridge.engine, "_call", lambda cmd, **a: (
        {"bars": rows} if cmd == "bars" else {"name": a.get("name", "MGC 12-26"), "point_value": 10.0}))
    df = bridge.engine.get_bars("MGC", "15m", 3)
    assert len(df) == 4
    assert df.index[-2] == opens[-1]
    assert df.index[-1] == opens[-1] + pd.Timedelta(minutes=15)       # never the wall clock
    assert df["close"].iloc[-1] == 1.5
    assert str(df.index.tz) == "UTC"


def test_bars_from_the_future_mean_a_time_zone_mismatch(bridge, monkeypatch):
    future_open = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
    monkeypatch.setattr(bridge.engine, "_call", lambda cmd, **a: (
        {"bars": [[future_open.timestamp(), 1, 1, 1, 1, 1]]} if cmd == "bars"
        else {"name": a.get("name", "MGC 12-26"), "point_value": 10.0}))
    with pytest.raises(nt.BridgeError, match="time zone"):
        bridge.engine.get_bars("MGC", "15m", 1)


def test_symbols_resolve_to_the_front_month_and_explicit_contracts_pass_through(bridge):
    name = bridge.engine.resolve_symbol("MGC")
    assert name.startswith("MGC ") and name == futures.front_contract("MGC", datetime.fromisoformat(
        trading_day(datetime.now(ZoneInfo("UTC")))).date())
    assert bridge.engine.resolve_symbol("MGC 02-27") == "MGC 02-27"
    with pytest.raises(ValueError):
        bridge.engine.resolve_symbol("GBPUSD")          # CFD names are not futures


def _one_shot_server(reply: dict):
    """A socket server that answers exactly one request the way VertexBridge does."""
    import json
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    seen: dict = {}

    def serve():
        conn, _ = srv.accept()
        with conn:
            data = b""
            while not data.endswith(b"\n"):
                data += conn.recv(4096)
            seen["line"] = data.decode("utf-8")
            conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], seen


def test_wire_protocol_round_trip():
    port, seen = _one_shot_server({"ok": True, "data": {"pong": 1}})
    engine = nt.NinjaTraderEngine(port=port, timeout_s=5)
    assert engine._call("enter", instrument="MGC 12-26", qty=2, sl_dist=0.5, skip=None) == {"pong": 1}
    assert seen["line"] == "enter\tinstrument=MGC 12-26\tqty=2\tsl_dist=0.5\n"


def test_bridge_refusals_surface_with_their_reason():
    port, _ = _one_shot_server({"ok": False, "error": "account 'X' not found"})
    engine = nt.NinjaTraderEngine(port=port, timeout_s=5)
    with pytest.raises(nt.BridgeError, match="account 'X' not found"):
        engine._call("account", account="X")


def test_unreachable_bridge_says_what_to_check():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                   # nothing listens there now
    engine = nt.NinjaTraderEngine(port=port, timeout_s=2)
    with pytest.raises(nt.BridgeError, match="VertexBridge"):
        engine._call("ping")
    assert engine.is_tradeable("MGC") is False  # fails closed, no exception


def test_arguments_cannot_smuggle_extra_fields():
    engine = nt.NinjaTraderEngine(port=1)
    with pytest.raises(ValueError):
        engine._call("close", ticket="1\tqty=99")


def test_order_result_ok_requires_a_fill():
    assert not nt.OrderResult(0, 0.0, 0.0, 0.0, 0.0, nt.RETCODE_DONE, "").ok
    assert not nt.OrderResult(5, 0.0, 0.0, 0.0, 0.0, nt.RETCODE_DONE, "").ok
    assert nt.OrderResult(5, 1.0, 1.0, 0.0, 0.0, nt.RETCODE_DONE, "").ok
