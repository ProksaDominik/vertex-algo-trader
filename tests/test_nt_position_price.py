"""An open position must never look flat at its entry because the bridge has no price yet.

2026-10-07: after the bridge was recompiled, its ticket replies carried no "last" price until
something asked for a quote, and the engine valued the trial's open NQ at its entry price --
open P&L $0 while NQ was +85 points, so the loss line could not have been seen.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine  # noqa: E402

TICKET = {"ticket": 7, "magic": 9125, "instrument": "NQ DEC26", "side": "long", "entry_avg": 31179.0,
          "open_qty": 1, "point_value": 20.0, "stop": 30779.0, "target": 31329.5, "opened_utc": 0}


def _engine(monkeypatch, replies):
    eng = NinjaTraderEngine(account="Sim101")
    calls = []

    def fake_call(cmd, **args):
        calls.append(cmd)
        r = replies[cmd]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(eng, "_call", fake_call)
    return eng, calls


def test_price_from_the_ticket_when_it_has_one(monkeypatch):
    eng, calls = _engine(monkeypatch, {})
    p = eng._to_position({**TICKET, "last": 31200.0})
    assert p.price_current == 31200.0 and p.profit == 420.0 and calls == []


def test_missing_price_asks_for_a_quote(monkeypatch):
    eng, calls = _engine(monkeypatch, {"quote": {"bid": 31263.25, "ask": 31263.75, "last": 31263.5}})
    p = eng._to_position({**TICKET, "last": None})
    assert calls == ["quote"]
    assert p.price_current == 31263.5 and p.profit == (31263.5 - 31179.0) * 20


def test_quote_still_empty_uses_the_last_minute_bar(monkeypatch):
    eng, calls = _engine(monkeypatch, {"quote": {"bid": None, "ask": None, "last": None},
                                       "bars": {"bars": [[0, 1, 1, 1, 31100.0, 5], [60, 1, 1, 1, 31120.0, 5]]}})
    p = eng._to_position({**TICKET, "last": None})
    assert calls == ["quote", "bars"] and p.price_current == 31120.0 and p.profit == (31120.0 - 31179.0) * 20


def test_no_price_anywhere_falls_back_to_entry(monkeypatch):
    eng, _ = _engine(monkeypatch, {"quote": BridgeError("down"), "bars": BridgeError("down")})
    p = eng._to_position({**TICKET, "last": None})
    assert p.price_current == 31179.0 and p.profit == 0.0


def test_closed_ticket_needs_no_price(monkeypatch):
    eng, calls = _engine(monkeypatch, {})
    p = eng._to_position({**TICKET, "last": None, "open_qty": 0})
    assert calls == [] and p.price_current == 31179.0
