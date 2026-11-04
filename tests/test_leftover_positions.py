"""A position NinjaTrader would not close (no market connection) must not be forgotten (2026-10-08).

That morning the laptop woke after the funded race's loss line had been touched; the bot booked it from the
price history, but NinjaTrader refused the close and the position stayed open with nothing managing it --
the next race would have stacked on top of it. Both bots now close such leftovers before racing again.
"""
from __future__ import annotations

import logging
import sys

from test_funded_race import run_funded
from test_trial_race_loop import BASE, ROOT, Path_, et, run_bot

sys.path.insert(0, str(ROOT / "src"))
from metatraderbot import target_race  # noqa: E402

LOG = logging.getLogger("test")


class TinyEngine:
    def __init__(self, open_tickets, refuse=()):
        self.open = set(open_tickets)
        self.refuse = set(refuse)
        self.closed = []

    def get_all_open_tickets(self):
        return set(self.open)

    def get_position(self, t):
        return object() if t in self.open else None

    def close_position(self, t, max_retries=3):
        if t in self.refuse:
            return False
        self.open.discard(t)
        self.closed.append(t)
        return True


def test_unmanaged_positions_are_closed_and_managed_ones_left_alone():
    eng = TinyEngine({7, 8})
    assert target_race.close_leftovers(eng, {8: {}}, [], LOG) == []
    assert eng.closed == [7] and eng.open == {8}


def test_a_refused_close_is_kept_and_tried_again_later():
    eng = TinyEngine({7}, refuse={7})
    assert target_race.close_leftovers(eng, {}, [], LOG) == [7]
    eng.refuse.clear()                                     # NinjaTrader connected again
    assert target_race.close_leftovers(eng, {}, [7], LOG) == []
    assert eng.closed == [7]


def test_a_leftover_that_closed_by_itself_is_dropped():
    eng = TinyEngine(set())
    assert target_race.close_leftovers(eng, {}, [7], LOG) == []


def _plant_leftover(eng, refuse=False):
    eng.t[99] = {"entry": BASE, "qty": 1, "sl": BASE - 400, "tp": BASE + 400, "open": True,
                 "opened": et(2026, 10, 1, 1, 0), "sign": 1}
    eng.n = 99
    if refuse:
        real = eng.close_position
        eng.close_position = lambda t, max_retries=3: False if t == 99 else real(t, max_retries)


FLAT = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 2, 23, 0), BASE)])


def test_funded_bot_closes_a_leftover_and_then_races(monkeypatch, tmp_path):
    st, eng, _ = run_funded(monkeypatch, tmp_path, FLAT, et(2026, 10, 1, 18, 30), et(2026, 10, 1, 20, 0),
                            setup=_plant_leftover)
    assert not eng.t[99]["open"] and eng.t[99]["closed"] < et(2026, 10, 1, 19, 0)    # closed at startup
    assert len(eng.orders) == 1 and eng.orders[0]["t"] < et(2026, 10, 1, 19, 10)
    assert st["leftovers"] == []


def test_funded_bot_does_not_race_on_top_of_a_leftover_it_cannot_close(monkeypatch, tmp_path):
    st, eng, _ = run_funded(monkeypatch, tmp_path, FLAT, et(2026, 10, 1, 18, 30), et(2026, 10, 1, 20, 0),
                            setup=lambda e: _plant_leftover(e, refuse=True))
    assert eng.orders == [] and eng.t[99]["open"]
    assert st["leftovers"] == [99]


def test_trial_bot_does_not_race_on_top_of_a_leftover_it_cannot_close(monkeypatch, tmp_path):
    st, eng, _ = run_bot(monkeypatch, tmp_path, FLAT, et(2026, 10, 1, 18, 30), et(2026, 10, 1, 20, 0),
                         setup=lambda e: _plant_leftover(e, refuse=True))
    assert eng.orders == [] and st["leftovers"] == [99]


def test_trial_bot_closes_a_leftover_and_then_races(monkeypatch, tmp_path):
    st, eng, _ = run_bot(monkeypatch, tmp_path, FLAT, et(2026, 10, 1, 18, 30), et(2026, 10, 1, 20, 0),
                         setup=_plant_leftover)
    assert not eng.t[99]["open"]
    assert len(eng.orders) == 1 and st["leftovers"] == []


# --- a position NO ticket explains (2026-10-14: NinjaTrader restarted mid-race, the bridge forgot the
# ticket, the race was booked from the price history, the 2 NQ stayed and blocked every later race) ------

class AccountEngine:
    def __init__(self, held, tickets=(), live=True):
        self.held, self.tickets, self.live, self.flattened = dict(held), set(tickets), live, []

    def open_contracts(self):
        return dict(self.held)

    def get_all_open_tickets(self):
        return set(self.tickets)

    def is_tradeable(self, symbol):
        return self.live

    def flatten(self, symbol):
        self.flattened.append(symbol)
        return self.held.pop(symbol, 0)


def test_an_untracked_position_is_closed_on_a_live_price():
    eng, said = AccountEngine({"NQ DEC26": 2}), []
    assert target_race.close_untracked(eng, LOG, said.append) == {}
    assert eng.flattened == ["NQ DEC26"] and "leftover position +2 NQ DEC26" in said[0]


def test_an_untracked_position_waits_for_a_live_price():
    eng = AccountEngine({"NQ DEC26": 2}, live=False)          # frozen feed: a close would fill at a fake price
    assert target_race.close_untracked(eng, LOG) == {"NQ DEC26": 2} and eng.flattened == []


def test_positions_with_bot_tickets_are_left_to_close_leftovers():
    eng = AccountEngine({"NQ DEC26": 2}, tickets={7})
    assert target_race.close_untracked(eng, LOG) == {} and eng.flattened == []


def test_a_bad_price_is_told_apart_from_a_closed_market():
    import pandas as pd
    from datetime import datetime, timezone

    now = datetime(2026, 10, 15, 8, 4, tzinfo=timezone.utc)

    class Feed:
        def __init__(self, age, last, traded_at, close=29_799.0):
            self.age, self.last, self.at, self.close = age, last, traded_at, close

        def quote(self, sym):
            return {"age_s": self.age, "last": self.last}

        def get_bars(self, sym, tf, n):
            idx = pd.date_range(end=self.at, periods=3, freq="1min")
            return pd.DataFrame({"volume": [300.0, 250.0, 280.0], "close": [self.close] * 3}, index=idx)

    live = pd.Timestamp("2026-10-15 08:03", tz="UTC")
    assert "has not moved for 585 min" in target_race.price_problem(Feed(35_100, 30_753.75, live), "NQ", now)
    # bridge 1.4 re-subscribed: the old price comes back looking fresh -- caught against the minute bars
    assert "955 points away" in target_race.price_problem(Feed(0, 30_753.75, live), "NQ", now)
    assert target_race.price_problem(Feed(2, 29_801.25, live), "NQ", now) is None                 # all good
    closed = pd.Timestamp("2026-10-15 03:00", tz="UTC")                                           # no trading
    assert target_race.price_problem(Feed(35_100, 30_753.75, closed), "NQ", now) is None
