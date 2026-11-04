"""Paper trading without NinjaTrader's market data (2026-10-15): prices from the minute bars, fills simulated
the way the replays fill them, every trade remembered on disk."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot.paper_engine import PaperEngine  # noqa: E402

T0 = datetime(2026, 10, 15, 13, 0, tzinfo=timezone.utc)


class Feed:
    """NinjaTrader's minute bars: rows (open, high, low, close); the last row is the forming minute."""

    def __init__(self, rows, start=T0):
        self.start, self.rows = start, list(rows)

    def get_bars(self, sym, tf, n):
        idx = pd.date_range(self.start, periods=len(self.rows), freq="1min", tz="UTC")
        df = pd.DataFrame(self.rows, columns=["open", "high", "low", "close"], index=idx)
        df["volume"] = 100.0
        return df.tail(n)

    def resolve_symbol(self, s):
        return "NQ DEC26"

    def usd_per_point(self, s):
        return 20.0


class Offline:
    DELAY_S = 600

    @staticmethod
    def bars(tf, n):
        raise OSError("no internet in this test")


def make(tmp_path, rows, minutes_after=None, internet=Offline):
    feed = Feed(rows)
    clock = {"now": T0 + timedelta(minutes=minutes_after if minutes_after is not None else len(rows) - 1, seconds=30)}
    eng = PaperEngine(account="T", feed=feed, path=tmp_path / "paper_T.json", max_contract_units=4,
                      now=lambda: clock["now"])
    eng.internet = internet
    return eng, feed, clock


def advance(eng, feed, clock, rows):
    feed.rows += rows
    clock["now"] += timedelta(minutes=len(rows))
    eng._bars = None                                   # a new minute: fresh bars


def test_a_buy_fills_at_the_close_plus_a_tick_and_reaches_its_target(tmp_path):
    eng, feed, clock = make(tmp_path, [(100, 101, 99, 100)] * 3)
    r = eng.place_market_order("NQ", "long", 2, sl=50, tp=175.5, magic=9, signal_entry_price=100)
    assert r.ok and r.price == 100.25 and r.tp == 175.75 and r.sl == 50.25
    advance(eng, feed, clock, [(100, 140, 95, 130), (130, 176, 129, 170), (170, 171, 169, 170)])
    assert eng.get_position(r.ticket) is None
    d = eng.get_closed_deal(r.ticket)
    assert d["exit_reason"] == "tp" and d["exit_price"] == 175.75
    assert round(d["pnl_usd"], 2) == round((175.75 - 100.25) * 2 * 20 - 4 * 1.75, 2)


def test_a_minute_touching_both_counts_as_the_stop(tmp_path):
    eng, feed, clock = make(tmp_path, [(100, 101, 99, 100)] * 3)
    r = eng.place_market_order("NQ", "long", 2, sl=50, tp=175.5, magic=9, signal_entry_price=100)
    advance(eng, feed, clock, [(100, 180, 40, 100), (100, 100, 100, 100)])
    assert eng.get_closed_deal(r.ticket)["exit_reason"] == "sl"


def test_trades_survive_a_restart_and_a_close_uses_the_live_price(tmp_path):
    eng, feed, clock = make(tmp_path, [(100, 101, 99, 100)] * 3)
    r = eng.place_market_order("NQ", "short", 1, sl=150, tp=50, magic=9)
    again = PaperEngine(account="T", feed=feed, path=tmp_path / "paper_T.json", now=lambda: clock["now"])
    again.internet = Offline
    assert again.open_contracts() == {"NQ": -1} and again.get_position(r.ticket).price_open == 100
    advance(again, feed, clock, [(100, 101, 90, 92), (92, 93, 91, 92)])
    assert again.close_position(r.ticket)
    d = again.get_closed_deal(r.ticket)
    assert d["exit_reason"] == "manual" and d["exit_price"] == 92.25


def test_no_live_bars_no_trade(tmp_path):
    eng, feed, clock = make(tmp_path, [(100, 101, 99, 100)] * 3, minutes_after=30)   # bars 28 min old, no internet
    try:
        eng.is_tradeable("NQ")
    except OSError:
        pass
    eng._bars = Feed([(100, 101, 99, 100)] * 3).get_bars("NQ", "1m", 3)               # only the old bars
    eng._bars_at, eng.source = 1e18, "NinjaTrader"
    assert not eng.is_tradeable("NQ")
    assert not eng.place_market_order("NQ", "long", 1, sl=50, tp=150, magic=9).ok


def test_the_contract_cap_holds(tmp_path):
    eng, feed, clock = make(tmp_path, [(100, 101, 99, 100)] * 3)
    assert eng.place_market_order("NQ", "long", 3, sl=50, tp=150, magic=9).ok
    assert not eng.place_market_order("NQ", "long", 2, sl=50, tp=150, magic=9).ok


def test_without_ninjatrader_the_internet_prices_are_used(tmp_path):
    """NinjaTrader closed: Yahoo's bars (10 minutes delayed) price and fill the paper trades."""
    class Down(Feed):
        def get_bars(self, sym, tf, n):
            raise ConnectionRefusedError("NinjaTrader closed")

    internet_rows = Feed([(200, 201, 199, 200)] * 5)

    class Internet:
        DELAY_S = 600

        @staticmethod
        def bars(tf, n):
            return internet_rows.get_bars("NQ", tf, n)

    clock = {"now": T0 + timedelta(minutes=14, seconds=30)}          # newest internet minute ended 10.5 min ago
    eng = PaperEngine(account="T", feed=Down([]), path=tmp_path / "p.json", now=lambda: clock["now"])
    eng.internet = Internet
    assert eng.is_tradeable("NQ") and eng.source == "internet" and eng.quote("NQ")["age_s"] < 60
    r = eng.place_market_order("NQ", "long", 1, sl=150, tp=250, magic=9, signal_entry_price=200)
    assert r.ok and r.price == 200.25
    assert eng.get_bars("NQ", "1m", 3)["close"].iloc[-1] == 200
