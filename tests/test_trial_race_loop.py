"""End-to-end: run_lucidpro_trial.main() in race mode against a fake NinjaTrader.

A scripted NQ price and a fast clock drive the REAL main loop, so the race's
day-to-day behaviour is checked without a market: entry window, take-profit
pass, one race per day, the 16:00 ET time exit, the growing target after a
losing day, closed-balance vs open-equity MLL, a missed flat window, and the
live-account rule "stop only after a PASS; after a FAIL wait for the reset".
"""
from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")
PV, COMM = 20.0, 1.75          # NQ


def et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET).astimezone(timezone.utc)


class Path_:
    """Piecewise-linear price over time."""

    def __init__(self, points):
        self.pts = sorted(points)

    def __call__(self, t):
        pts = self.pts
        if t <= pts[0][0]:
            return pts[0][1]
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                return round((p0 + f * (p1 - p0)) * 4) / 4
        return pts[-1][1]


class FakeEngine:
    def __init__(self, clock, price, demo=True, login="Sim101"):
        self.clock, self.price, self.demo, self.login = clock, price, demo, login
        self.balance = 50_000.0
        self.t = {}
        self.orders = []
        self.n = 0
        self.daily = None          # completed daily closes for the trend rule; None = no daily history

    def _p(self):
        return self.price(self.clock["now"])

    def _update(self):
        p = self._p()
        for k, t in self.t.items():
            if t["open"]:
                sg = t["sign"]
                if sg * (p - t["tp"]) >= 0:
                    self._close(k, t["tp"], "tp")
                elif sg * (p - t["sl"]) <= 0:
                    self._close(k, t["sl"], "sl")

    def _close(self, k, px, why):
        t = self.t[k]
        t.update(open=False, exit=px, why=why, closed=self.clock["now"])
        pnl = (px - t["entry"]) * t["sign"] * t["qty"] * PV - 2 * COMM * t["qty"]
        t["pnl"] = pnl
        self.balance += pnl

    # --- the NinjaTraderEngine surface the runner uses ---
    def connect(self):
        return SimpleNamespace(login=self.login, is_demo=self.demo, balance=self.balance, server="fake")

    def account_info(self):
        return SimpleNamespace(login=self.login, is_demo=self.demo, balance=self.balance, server="fake")

    def is_tradeable(self, sym):
        return True

    def quote(self, sym):
        p = self._p()
        return {"bid": p - 0.25, "ask": p, "age_s": 0}

    def usd_per_point(self, sym):
        return PV

    def open_contracts(self):
        return {"NQ DEC26": sum(t["qty"] for t in self.t.values() if t["open"])} if any(
            t["open"] for t in self.t.values()) else {}

    def place_market_order(self, symbol, direction, lots, sl, tp, magic, comment="", signal_entry_price=None, **_):
        sign = 1 if direction == "long" else -1
        assert sign * (tp - sl) > 0, (direction, sl, tp)       # the bracket is on the right sides
        ref = signal_entry_price or self._p()
        fill = self._p() if sign > 0 else self._p() - 0.25       # longs pay the ask, shorts sell the bid
        self.n += 1
        k = self.n
        self.t[k] = {"entry": fill, "qty": int(lots), "sl": fill - (ref - sl), "tp": fill + (tp - ref), "open": True,
                     "opened": self.clock["now"], "sign": sign}
        self.orders.append({"t": self.clock["now"], "entry": fill, "tp_dist": sign * (tp - ref), "qty": int(lots),
                            "direction": direction})
        return SimpleNamespace(ok=True, ticket=k, price=fill, sl=self.t[k]["sl"], tp=self.t[k]["tp"],
                               volume=float(lots), retcode=10009, comment="")

    def get_bars(self, symbol, timeframe, count):
        """Completed 1-minute bars from the scripted path + a forming stand-in,
        like NinjaTraderEngine.get_bars (index = bar open time, UTC)."""
        import numpy as np
        import pandas as pd
        if timeframe == "1d":
            if self.daily is None:
                raise RuntimeError("no daily bars")
            c = list(self.daily) + [self._p()]                 # + the forming stand-in
            idx = pd.date_range(end=pd.Timestamp(self.clock["now"]).normalize(), periods=len(c), freq="D")
            return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx).tail(count + 1)
        assert timeframe == "1m"
        now = pd.Timestamp(self.clock["now"]).floor("min")
        xs = np.array([t.timestamp() for t, _ in self.price.pts])
        ys = np.array([p for _, p in self.price.pts])
        idx = pd.DatetimeIndex([now - pd.Timedelta(minutes=i) for i in range(count, -1, -1)], name="timestamp")
        t0 = np.array([t.timestamp() for t in idx])
        o = np.round(np.interp(t0, xs, ys) * 4) / 4
        c = np.round(np.interp(t0 + 60, xs, ys) * 4) / 4
        c[-1] = o[-1]                                   # the forming stand-in: one price
        return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                             "volume": 1.0}, index=idx)

    def get_all_open_tickets(self):
        return {k for k, t in self.t.items() if t["open"]}

    def get_position(self, ticket):
        self._update()
        t = self.t.get(ticket)
        if t is None or not t["open"]:
            return None
        return SimpleNamespace(price_current=self._p())

    def get_closed_deal(self, ticket, offset_fallback=0):
        t = self.t.get(ticket)
        if t is None or t["open"]:
            return None
        return {"pnl_usd": t["pnl"], "exit_price": t["exit"], "closed_ts": t["closed"], "exit_reason": t["why"]}

    def close_position(self, ticket, max_retries=3):
        self._update()
        t = self.t.get(ticket)
        if t is None or not t["open"]:
            return False
        self._close(ticket, self._p() - 0.25 if t["sign"] > 0 else self._p(), "manual")
        return True


def run_bot(monkeypatch, tmp_path, price, start, end, *, reading="lucid", contracts=2, live=False, jumps=(),
            on_poll=None, state=None, daily=None, side_rule="trend200", setup=None):
    monkeypatch.setenv("VERTEX_BROKER", "ninjatrader")
    monkeypatch.setenv("VERTEX_PAPER", "0")              # the test engine, not paper mode from .env
    monkeypatch.setenv("LUCID_TRIAL_STRATEGY", "race")
    monkeypatch.setenv("LUCID_RULE_READING", reading)
    monkeypatch.setenv("LUCID_RACE_CONTRACTS", str(contracts))
    monkeypatch.setenv("LUCID_RACE_ENTRY", "19:00")
    monkeypatch.setenv("LUCID_RACE_SIDE", side_rule)
    login = "LUCIDEVAL1" if live else "Sim101"
    if live:
        monkeypatch.setenv("LUCID_LIVE_CONFIRM", login)
    else:
        monkeypatch.delenv("LUCID_LIVE_CONFIRM", raising=False)
    spec = importlib.util.spec_from_file_location("trial_under_test", ROOT / "scripts" / "run_lucidpro_trial.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for h in list(logging.getLogger().handlers):          # never write the live bot's log file
        if isinstance(h, logging.FileHandler):
            logging.getLogger().removeHandler(h)
            h.close()
    clock = {"now": start}
    eng = FakeEngine(clock, price, demo=not live, login=login)
    if setup is not None:
        setup(eng)
    eng.daily = daily

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"] if tz is None else clock["now"].astimezone(tz)

    alerts = []
    pending = list(jumps)

    def fake_sleep(s):
        clock["now"] += timedelta(seconds=s)
        while pending and clock["now"] >= pending[0][0]:
            clock["now"] = pending.pop(0)[1]
        if on_poll:
            on_poll(clock["now"], eng, m)
        if clock["now"] >= end:
            raise KeyboardInterrupt

    monkeypatch.setattr(m, "datetime", FakeDT)
    monkeypatch.setattr(m, "NinjaTraderEngine", lambda **kw: eng)
    monkeypatch.setattr(m, "alert", lambda msg, **kw: alerts.append(msg))
    monkeypatch.setattr(m, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(m, "TRADES_LOG_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(m, "POLL_INTERVAL_S", 300)
    monkeypatch.setattr(m.threading, "Thread", lambda *a, **k: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(m.time, "sleep", fake_sleep)
    if state is not None:
        (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    assert m.main() == 0
    st = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    return st, eng, alerts


BASE = 30_000.0
DOWN_TREND = [32_000.0 - 10 * i for i in range(250)]      # last close far below its 200-day average
UP_TREND = [28_000.0 + 10 * i for i in range(250)]


def test_race_passes_then_waits_for_the_next_day(monkeypatch, tmp_path):
    # Thu 1 Oct 18:30 ET; NQ climbs 90 pts overnight -> the +75.5 target (2 NQ) fills.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 90), (et(2026, 10, 2, 16, 0), BASE + 90),
                   (et(2026, 10, 2, 18, 0), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0))
    o = eng.orders[0]
    assert o["t"] >= et(2026, 10, 1, 19, 0) and o["t"] < et(2026, 10, 1, 19, 10)
    # ($3,000 + 4 x $1.75) / (2 x $20) = 75.175 -> 75.25 on the tick grid, + 1 tick
    assert o["qty"] == 2 and o["tp_dist"] == pytest.approx(75.5)
    assert st["passes"] == 1 and st["fails"] == 0
    assert st["race_day"] == "2026-10-02"
    # one race a day (2026-10-15): the next attempt waits for the next 19:00 race
    assert len(eng.orders) == 1


def test_old_rule_restarts_at_once_after_a_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCID_TRIAL_ONE_RACE_A_DAY", "0")       # the simulator's earlier immediate restarts
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 90), (et(2026, 10, 2, 16, 0), BASE + 90),
                   (et(2026, 10, 2, 18, 0), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0))
    assert st["passes"] == 1 and len(eng.orders) == 2
    assert eng.orders[1]["t"] < et(2026, 10, 2, 2, 30)


def test_losing_day_exits_at_16_and_next_target_grows(monkeypatch, tmp_path):
    # Race day Fri 2 Oct drifts down 20 pts; Monday's race must chase the loss too.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 59), BASE - 20), (et(2026, 10, 4, 18, 0), BASE - 20),
                   (et(2026, 10, 5, 23, 0), BASE - 20)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 4, 19, 30))
    assert len(eng.orders) == 2
    first = eng.t[1]
    assert first["why"] == "manual" and first["closed"] >= et(2026, 10, 2, 16, 0)
    assert first["closed"] < et(2026, 10, 2, 16, 10)             # the 16:00 ET time exit
    loss = -first["pnl"]
    assert 750 < loss < 900
    need = 3_000 + loss
    expected = (int((need + 7) / 40 / 0.25 - 1e-9) + 1 + 1) * 0.25
    assert eng.orders[1]["tp_dist"] == pytest.approx(expected)
    assert eng.orders[1]["t"] >= et(2026, 10, 4, 19, 0)          # Sunday-evening session = Monday's race
    assert st["fails"] == 0 and st["attempts"] == 0


def test_equity_breach_fails_and_sim_starts_fresh(monkeypatch, tmp_path):
    # Confirmed rule: the open position counts -- -50 pts on 2 NQ = -$2,000 ends it.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 59), BASE - 60), (et(2026, 10, 5, 23, 0), BASE - 60)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 4, 19, 30))
    assert st["fails"] == 1 and st["passes"] == 0
    assert eng.t[1]["closed"] < et(2026, 10, 2, 15, 0)           # intraday, not at the close
    assert st["attempt"]["realized_usd"] == 0.0                  # a fresh $50,000 attempt
    assert len(eng.orders) == 2                                  # and it races again next session


def _dip_path(depth: float, top: float):
    # Like 30 Sep: a deep open dip at 09:25 ET, then a rally past the target by 09:40.
    return Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                  (et(2026, 10, 2, 9, 0), BASE - 5), (et(2026, 10, 2, 9, 25), BASE - depth),
                  (et(2026, 10, 2, 9, 40), BASE + top), (et(2026, 10, 5, 23, 0), BASE + top)])


def test_open_dip_past_the_limit_fails_and_the_next_race_waits_for_19(monkeypatch, tmp_path):
    # -55 pts on 2 NQ = -$2,200 of open equity: breached, although price recovers. One race a day
    # (2026-10-15): the next evaluation does not race again until the next 19:00 New York.
    st, eng, _ = run_bot(monkeypatch, tmp_path, _dip_path(55, 80), et(2026, 10, 1, 18, 30), et(2026, 10, 2, 12, 0))
    assert st["fails"] == 1 and st["passes"] == 0 and len(eng.orders) == 1


def test_open_dip_past_the_limit_fails_and_the_next_attempt_starts_at_once(monkeypatch, tmp_path):
    # The earlier rule (LUCID_TRIAL_ONE_RACE_A_DAY=0): when one fails, start another one until it passes --
    # the next attempt races immediately (09:25 ET is inside the entry window) and catches the rally to +80.
    monkeypatch.setenv("LUCID_TRIAL_ONE_RACE_A_DAY", "0")
    st, eng, _ = run_bot(monkeypatch, tmp_path, _dip_path(55, 80), et(2026, 10, 1, 18, 30), et(2026, 10, 2, 12, 0))
    assert eng.t[1]["why"] == "manual"
    assert st["fails"] == 1
    assert len(eng.orders) >= 2 and eng.orders[1]["t"] < et(2026, 10, 2, 9, 40)   # (the simulator then keeps cycling)
    assert st["passes"] == 1


def test_lenient_reading_still_available_for_research(monkeypatch, tmp_path):
    # The original premise (closing balance only), 4 NQ: the -$2,800 dip is survived.
    st, eng, _ = run_bot(monkeypatch, tmp_path, _dip_path(35, 40), et(2026, 10, 1, 18, 30), et(2026, 10, 2, 12, 0),
                         reading="lenient", contracts=4)
    assert st["passes"] == 1 and st["fails"] == 0
    assert eng.t[1]["why"] == "tp"


def test_missed_flat_window_closes_the_stale_position(monkeypatch, tmp_path):
    # Laptop off from 15:00 ET Friday to Monday 08:00 ET: the Friday race must be
    # closed on the first poll back, then Monday's race goes in (late window).
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 0), BASE - 10), (et(2026, 10, 5, 8, 0), BASE - 12),
                   (et(2026, 10, 5, 23, 0), BASE - 12)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 5, 8, 30),
                         jumps=[(et(2026, 10, 2, 15, 0), et(2026, 10, 5, 8, 0))])
    assert eng.t[1]["why"] == "manual" and eng.t[1]["closed"] >= et(2026, 10, 5, 8, 0)
    assert len(eng.orders) == 2 and eng.orders[1]["t"] >= et(2026, 10, 5, 8, 0)


def test_live_account_stops_only_after_a_pass(monkeypatch, tmp_path):
    # FAIL on a live account: no trading until the balance is back at $50,000.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 59), BASE - 60), (et(2026, 10, 4, 18, 0), BASE - 60),
                   (et(2026, 10, 5, 2, 0), BASE - 40), (et(2026, 10, 6, 23, 0), BASE - 40)])

    def reset_on_monday(now, eng, m):
        if now >= et(2026, 10, 5, 1, 0) and eng.balance < 50_000:
            eng.balance = 50_000.0                               # the owner resets it in Lucid

    st, eng, alerts = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 5, 3, 0),
                              live=True, on_poll=reset_on_monday)
    assert st["fails"] == 1 and st["passes"] == 0
    assert not st.get("halted") and not st.get("awaiting_reset")
    assert len(eng.orders) == 2                                  # raced again after the reset ...
    assert eng.orders[1]["t"] >= et(2026, 10, 5, 1, 0)           # ... and not before it
    assert any("FAILED" in a for a in alerts) and any("reset detected" in a for a in alerts)


def test_live_account_halts_after_a_pass(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 90), (et(2026, 10, 7, 23, 0), BASE + 90)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 6, 20, 0), live=True)
    assert st["passes"] == 1 and st.get("halted") == "pass"
    assert len(eng.orders) == 1                                  # nothing after the pass


# --- the laptop sleeps: what Lucid's server saw must still be booked ----------

def _trades(tmp_path):
    p = tmp_path / "trades.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def _asleep(path_points, wake, end, monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE)] + path_points)
    return run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), end,
                   jumps=[(et(2026, 10, 2, 9, 0), wake)])


def test_asleep_through_the_loss_limit_fails_like_lucid(monkeypatch, tmp_path):
    # 30 Sep/1 Oct case: asleep 09:00-11:00, NQ fell 55 pts (-$2,200 open) and came back.
    st, eng, _ = _asleep([(et(2026, 10, 2, 9, 0), BASE), (et(2026, 10, 2, 10, 0), BASE - 55),
                          (et(2026, 10, 2, 11, 0), BASE + 10), (et(2026, 10, 5, 23, 0), BASE + 10)],
                         et(2026, 10, 2, 11, 0), et(2026, 10, 2, 11, 30), monkeypatch, tmp_path)
    assert st["fails"] == 1 and st["passes"] == 0
    t = _trades(tmp_path)[-1]
    assert t["exit_price"] == pytest.approx(BASE - 50.0)          # booked at the MLL line, not today's price
    assert t["pnl_usd"] == pytest.approx(-2_007.0, abs=1.0)
    assert t["closed_ts"].startswith("2026-10-02T13:")             # ~09:50 ET, while asleep


def test_asleep_through_the_target_passes_like_lucid(monkeypatch, tmp_path):
    st, eng, _ = _asleep([(et(2026, 10, 2, 9, 0), BASE), (et(2026, 10, 2, 10, 0), BASE + 80),
                          (et(2026, 10, 2, 11, 0), BASE + 10), (et(2026, 10, 5, 23, 0), BASE + 10)],
                         et(2026, 10, 2, 11, 0), et(2026, 10, 2, 11, 30), monkeypatch, tmp_path)
    assert st["passes"] == 1 and st["fails"] == 0
    assert _trades(tmp_path)[-1]["exit_price"] == pytest.approx(BASE + 75.5)


def test_take_profit_filled_on_waking_after_the_line_is_a_fail(monkeypatch, tmp_path):
    # Asleep 09:00-11:00: NQ fell through the line at 10:00, then rallied past the
    # target. On waking the simulator fills the take-profit -- Lucid had already
    # ended the attempt at 10:00, so the bot re-scores it as a fail.
    st, eng, _ = _asleep([(et(2026, 10, 2, 9, 0), BASE), (et(2026, 10, 2, 10, 0), BASE - 55),
                          (et(2026, 10, 2, 10, 45), BASE + 80), (et(2026, 10, 5, 23, 0), BASE + 80)],
                         et(2026, 10, 2, 11, 0), et(2026, 10, 2, 11, 10), monkeypatch, tmp_path)
    assert eng.t[1]["why"] == "tp"                                 # the simulator's fill ...
    assert st["fails"] == 1 and st["passes"] == 0                  # ... does not count
    t = _trades(tmp_path)[0]
    assert t["reason"] == "fail_rescored" and t["exit_price"] == pytest.approx(BASE - 50.0)


def test_take_profit_filled_after_the_time_exit_books_the_1600_price(monkeypatch, tmp_path):
    # Asleep 15:00-16:30; NQ rallied past the target only AFTER 16:00. The fill
    # on waking does not count: the race was out at 16:00 at the 15:59 price.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 14, 0), BASE - 10), (et(2026, 10, 2, 16, 0), BASE - 10),
                   (et(2026, 10, 2, 16, 20), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 16, 40),
                         jumps=[(et(2026, 10, 2, 15, 0), et(2026, 10, 2, 16, 30))])
    assert eng.t[1]["why"] == "tp"
    t = _trades(tmp_path)[0]
    assert t["reason"] == "exit_while_away" and t["exit_price"] == pytest.approx(BASE - 10.0)
    assert st["passes"] == 0 and st["fails"] == 0


def test_asleep_through_the_time_exit_books_the_1600_price(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 14, 0), BASE - 10), (et(2026, 10, 2, 16, 0), BASE - 10),
                   (et(2026, 10, 2, 16, 20), BASE + 30), (et(2026, 10, 5, 23, 0), BASE + 30)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 16, 40),
                         jumps=[(et(2026, 10, 2, 15, 0), et(2026, 10, 2, 16, 30))])
    t = _trades(tmp_path)[-1]
    assert t["exit_price"] == pytest.approx(BASE - 10.0)           # the 15:59 close, not the 16:30 rally
    assert st["fails"] == 0 and st["passes"] == 0


def test_mll_floor_locks_at_50100(monkeypatch, tmp_path):
    monkeypatch.setenv("VERTEX_BROKER", "ninjatrader")
    monkeypatch.setenv("LUCID_TRIAL_STRATEGY", "race")
    spec = importlib.util.spec_from_file_location("trial_floor", ROOT / "scripts" / "run_lucidpro_trial.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.mll_floor(50_000) == 48_000
    assert m.mll_floor(51_000) == 49_000
    assert m.mll_floor(52_100) == 50_100
    assert m.mll_floor(53_500) == 50_100                              # locked, no longer trailing


def test_one_race_per_attempt_per_day(monkeypatch, tmp_path):
    # A losing day closed at 16:00 does NOT start a second race for the same
    # attempt -- only a NEW attempt may race again the same day.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 9, 0), BASE - 20), (et(2026, 10, 5, 23, 0), BASE - 20)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 16, 30))
    assert len(eng.orders) == 1 and st["attempts"] == 0


# --- the trend rule (owner, 2026-10-07): long above the 200-day average, short below ------

def test_trend_side_rule():
    from metatraderbot import target_race
    assert target_race.trend_side(UP_TREND) == 1
    assert target_race.trend_side(DOWN_TREND) == -1
    assert target_race.trend_side(DOWN_TREND[:100]) == 1           # too little history: long


def test_short_race_below_the_average_passes_on_a_fall(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 2, 2, 0), BASE - 90), (et(2026, 10, 5, 23, 0), BASE - 90)])
    st, eng, alerts = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                              daily=DOWN_TREND)
    o = eng.orders[0]
    assert o["direction"] == "short" and o["qty"] == 2 and o["tp_dist"] == pytest.approx(75.5)
    assert eng.t[1]["why"] == "tp" and eng.t[1]["pnl"] == pytest.approx(3_013.0)
    assert st["passes"] == 1 and st["fails"] == 0
    assert any("short" in a for a in alerts)


def test_short_race_fails_when_price_rises_through_the_line(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 59), BASE + 60), (et(2026, 10, 5, 23, 0), BASE + 60)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                         daily=DOWN_TREND)
    assert eng.orders[0]["direction"] == "short"
    assert st["fails"] == 1 and st["passes"] == 0
    assert eng.t[1]["closed"] < et(2026, 10, 2, 15, 0)


def test_up_trend_and_missing_daily_bars_race_long(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])
    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                         daily=UP_TREND)
    assert eng.orders[0]["direction"] == "long" and st["passes"] == 1


def test_race_lost_by_ninjatrader_goes_on_and_passes_on_the_price_history(monkeypatch, tmp_path):
    # 2026-10-09: NinjaTrader restarted 6 minutes into evaluation #30's race and forgot the position;
    # the bot dropped the race without a result although the price reached its target two hours later.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 1, 21, 0), BASE + 10), (et(2026, 10, 1, 23, 0), BASE + 90),
                   (et(2026, 10, 2, 16, 0), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])

    def forget(now, eng, m):
        if now >= et(2026, 10, 1, 19, 20):
            for t in eng.t.values():
                if t["open"] and t["opened"] < et(2026, 10, 1, 19, 10):
                    eng.t.clear()                      # NinjaTrader restarted: no position, no closed deal
                    break

    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 1, 0),
                         on_poll=forget)
    assert st["passes"] == 1 and st["fails"] == 0
    assert _trades(tmp_path)[0]["reason"] == "pass_while_away"


def test_real_account_reset_the_same_evening_waits_for_the_next_19_00_race(monkeypatch, tmp_path):
    # Fails at 21:00 New York; the owner resets it at 23:00. A real account races once per trading day.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 1, 21, 0), BASE - 60), (et(2026, 10, 2, 18, 0), BASE - 60),
                   (et(2026, 10, 2, 19, 30), BASE - 60), (et(2026, 10, 5, 23, 0), BASE - 60)])

    def reset_same_evening(now, eng, m):
        if now >= et(2026, 10, 1, 23, 0) and eng.balance < 50_000:
            eng.balance = 50_000.0

    st, eng, _ = run_bot(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 4, 20, 0),
                         live=True, on_poll=reset_same_evening)
    assert st["fails"] == 1
    assert len(eng.orders) == 2                                  # (2 Oct is a Friday: no evening session)
    assert eng.orders[1]["t"] >= et(2026, 10, 4, 19, 0)          # not again on Thu/Fri's trading day
