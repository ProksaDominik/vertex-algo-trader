"""NinjaTraderEngine.get_flow: the order-flow reply of VertexBridge 1.3 -> a per-minute frame."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine  # noqa: E402


def _engine(monkeypatch, version, reply=None):
    eng = NinjaTraderEngine(account="Sim101")
    calls = []

    def fake_call(cmd, **args):
        calls.append((cmd, args))
        if cmd == "ping":
            return {"version": version}
        return reply

    monkeypatch.setattr(eng, "_call", fake_call)
    monkeypatch.setattr(eng, "resolve_symbol", lambda s: "NQ DEC26")
    return eng, calls


def test_flow_parses_minutes(monkeypatch):
    t0 = int(datetime(2026, 10, 6, 13, 30, tzinfo=timezone.utc).timestamp())
    reply = {"ticks": 1234, "with_quote": 1200,
             "minutes": [[t0, 500, 300, 2, 90, 40, 10], [t0 + 60, 100, 400, 0, 50, 0, 60]]}
    eng, calls = _engine(monkeypatch, "1.3", reply)
    start = datetime(2026, 10, 5, 22, 0, tzinfo=timezone.utc)
    end = datetime(2026, 10, 6, 21, 0, tzinfo=timezone.utc)
    df, info = eng.get_flow("NQ", start, end, big=10)
    cmd, args = calls[-1]
    assert cmd == "flow" and args["instrument"] == "NQ DEC26"
    assert args["from"] == int(start.timestamp()) and args["to"] == int(end.timestamp()) and args["big"] == 10
    assert list(df.columns) == ["buy", "sell", "other", "trades", "big_buy", "big_sell"]
    assert str(df.index[0]) == "2026-10-06 13:30:00+00:00"
    assert df["buy"].tolist() == [500, 100] and df["big_sell"].tolist() == [10, 60]
    assert info == {"ticks": 1234, "with_quote": 1200}


def test_flow_needs_bridge_1_3(monkeypatch):
    eng, _ = _engine(monkeypatch, "1.2")
    with pytest.raises(BridgeError, match="F5"):
        eng.get_flow("NQ", datetime(2026, 10, 5, tzinfo=timezone.utc), datetime(2026, 10, 6, tzinfo=timezone.utc))
