"""End-to-end: run_lucid_futures_funded.main() -- the FUNDED RACE -- against a fake NinjaTrader.

A scripted NQ price and a fast clock drive the REAL main loop: the daily
take-profit is what the next payout still needs (at most $1,500), payouts
follow Lucid's rules (3 days, $500 minimum, best day <= 40%, $52,100 kept),
the MLL counts open equity and locks at $50,100, a sleeping laptop is judged
from 1-minute history, and a blown simulator account is replaced at the next
session while a real one halts.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from test_trial_race_loop import BASE, DOWN_TREND, ROOT, FakeEngine, Path_, et

LINE = BASE - (2_000 - 3.5) / 20          # 1 NQ from BASE: equity at the $48,000 floor, commission included


def load_module(name: str = "funded_under_test"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "run_lucid_futures_funded.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for h in list(logging.getLogger().handlers):          # never write the live bot's log file
        if isinstance(h, logging.FileHandler):
            logging.getLogger().removeHandler(h)
            h.close()
    return m


def run_funded(monkeypatch, tmp_path, price, start, end, *, live=False, jumps=(), state=None, daily=None,
               side_rule="trend200", setup=None):
    monkeypatch.setenv("LUCID_FUNDED_CONTRACTS", "1")
    monkeypatch.setenv("VERTEX_PAPER", "0")              # the test engine, not paper mode from .env
    monkeypatch.setenv("LUCID_RACE_SIDE", side_rule)
    monkeypatch.setenv("LUCID_FUNDED_ENTRY", "19:00")
    login = "LUCIDFUNDED1" if live else "Sim102"
    if live:
        monkeypatch.setenv("LUCID_FUNDED_LIVE_CONFIRM", login)
    else:
        monkeypatch.delenv("LUCID_FUNDED_LIVE_CONFIRM", raising=False)
    m = load_module()
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


def _trades(tmp_path):
    p = tmp_path / "trades.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_three_winning_days_then_the_first_payout(monkeypatch, tmp_path):
    # Thu-evening, Sunday-evening and Monday-evening races all reach their take.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 85), (et(2026, 10, 4, 19, 0), BASE + 85),
                   (et(2026, 10, 5, 2, 0), BASE + 170), (et(2026, 10, 5, 19, 0), BASE + 170),
                   (et(2026, 10, 6, 2, 0), BASE + 215), (et(2026, 10, 7, 23, 0), BASE + 215)])
    st, eng, alerts = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 6, 17, 30))
    # day 1: the payout needs $2,600 -> capped at $1,500: ($1,500 + $3.50) / $20 -> 75.25 + 1 tick
    # day 2: the 40% rule needs $2,259.75 -> $1,500 again; day 3: the 40% rule needs $753.25 -> 38.25 points
    assert [o["tp_dist"] for o in eng.orders] == [75.5, 75.5, 38.25]
    assert eng.orders[0]["t"] < et(2026, 10, 1, 19, 10) and eng.orders[1]["t"] >= et(2026, 10, 4, 19, 0)
    run = st["run"]
    assert run["realized_usd"] == pytest.approx(1506.5 + 1506.5 + 761.5)
    # paid at Tuesday's close: 3 days, $3,774.50 cycle profit, best day $1,506.50 <= 40% -- everything above
    # $52,100: $1,674.50 (the old $1,040 cap paid $524.50 after the same three wins)
    assert [p["amount_usd"] for p in run["payouts"]] == [pytest.approx(1674.5)]
    assert run["payouts"][0]["to_you_usd"] == pytest.approx(1507.05)
    assert 50_000 + run["realized_usd"] - run["paid_usd"] == pytest.approx(52_100.0)   # $52,100 stays
    assert run["mll_locked"] and run["cycle_days"] == 0
    assert st["runs_paid"] == 1 and st["payouts_total"] == 1 and st["runs_blown"] == 0
    assert any("PAYOUT #1" in a and "virtual" in a for a in alerts)


def test_losing_time_exit_and_the_next_take_is_capped(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 15, 59), BASE - 20), (et(2026, 10, 5, 23, 0), BASE - 20)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 4, 19, 30))
    assert eng.t[1]["why"] == "manual"
    assert et(2026, 10, 2, 16, 0) <= eng.t[1]["closed"] < et(2026, 10, 2, 16, 10)    # out at 16:00 ET
    assert st["run"]["realized_usd"] == pytest.approx(-408.5)
    assert len(eng.orders) == 2 and eng.orders[1]["t"] >= et(2026, 10, 4, 19, 0)
    assert eng.orders[1]["tp_dist"] == 75.5                       # needs $3,008.50 -> still $1,500
    assert st["runs_blown"] == 0


def test_equity_breach_blows_the_account_and_the_next_starts_next_session(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 10, 0), BASE - 120), (et(2026, 10, 5, 23, 0), BASE - 120)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 4, 19, 30))
    assert st["runs_blown"] == 1 and st["runs_total"] == 2
    assert st["history"][0]["outcome"] == "blown" and st["history"][0]["payouts"] == 0
    assert _trades(tmp_path)[0]["pnl_usd"] == pytest.approx(-2_000.0, abs=15.0)
    assert eng.t[1]["closed"] < et(2026, 10, 2, 8, 0)              # intraday, as soon as it happened
    # the replacement account races at the NEXT session, not into the same fall
    assert len(eng.orders) == 2 and eng.orders[1]["t"] >= et(2026, 10, 4, 19, 0)
    assert st["run"]["realized_usd"] == 0.0


def _paused_state(**over):
    run = {"start": "2026-09-28T23:00:00+00:00", "realized_usd": 2_700.0, "paid_usd": 0.0, "payouts": [],
           "peak_eod": 52_700.0, "mll_locked": True, "day": "2026-10-02", "day_start_realized": 2_700.0,
           "cycle_days": 1, "cycle_pnl": 2_700.0, "cycle_best_day": 1_000.0, "days": 3}
    run.update(over)
    return {"run": run, "runs_total": 1, "runs_paid": 0, "runs_blown": 0, "runs_done": 0,
            "payouts_total": 0, "paid_total_usd": 0.0, "history": []}


def test_pauses_once_the_payout_money_is_met_until_the_third_day(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 6, 23, 0), BASE)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 5, 19, 30),
                            state=_paused_state())
    # no race on Thursday evening or Sunday evening -- only after Monday's close pays
    assert len(eng.orders) == 1 and eng.orders[0]["t"] >= et(2026, 10, 5, 19, 0)
    run = st["run"]
    assert [p["amount_usd"] for p in run["payouts"]] == [600.0]   # $52,700 - $52,100
    assert run["payouts"][0]["day"] == "2026-10-05"
    assert eng.orders[0]["tp_dist"] == 75.5                       # payout 2 needs $2,500 -> $1,500 today


def test_live_account_is_told_to_request_the_payout(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 6, 23, 0), BASE)])
    st, eng, alerts = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30),
                                 et(2026, 10, 5, 17, 30), live=True, state=_paused_state())
    assert any("Request <b>$600.00</b>" in a for a in alerts)
    assert not st["halted"]


def test_live_account_halts_when_blown_and_records_it_once(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 10, 0), BASE - 120), (et(2026, 10, 7, 23, 0), BASE - 120)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 6, 12, 0),
                            live=True)
    assert st["halted"] and st["runs_blown"] == 1 and len(st["history"]) == 1
    assert len(eng.orders) == 1                                   # nothing after the blow-up


def _asleep(points, monkeypatch, tmp_path, end):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 9, 0), BASE)] + points)
    return run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), end,
                      jumps=[(et(2026, 10, 2, 9, 0), et(2026, 10, 2, 11, 0))])


def test_asleep_through_the_loss_limit_is_booked_at_the_line(monkeypatch, tmp_path):
    st, eng, _ = _asleep([(et(2026, 10, 2, 10, 0), BASE - 110), (et(2026, 10, 2, 11, 0), BASE + 10),
                          (et(2026, 10, 5, 23, 0), BASE + 10)], monkeypatch, tmp_path, et(2026, 10, 2, 11, 30))
    assert st["runs_blown"] == 1
    t = _trades(tmp_path)[0]
    assert t["reason"] == "mll_while_away" and t["exit_price"] == pytest.approx(LINE, abs=0.01)
    assert t["closed_ts"].startswith("2026-10-02T13:")             # ~09:5x ET, while asleep


def test_take_profit_filled_on_waking_after_the_line_is_rescored(monkeypatch, tmp_path):
    st, eng, _ = _asleep([(et(2026, 10, 2, 10, 0), BASE - 110), (et(2026, 10, 2, 10, 45), BASE + 80),
                          (et(2026, 10, 5, 23, 0), BASE + 80)], monkeypatch, tmp_path, et(2026, 10, 2, 11, 10))
    assert eng.t[1]["why"] == "tp"                                 # the simulator filled the take ...
    assert st["runs_blown"] == 1 and not st["history"][0]["payouts"]   # ... after Lucid had closed it
    t = _trades(tmp_path)[0]
    assert t["reason"] == "mll_rescored" and t["exit_price"] == pytest.approx(LINE, abs=0.01)


def test_take_profit_filled_after_the_time_exit_books_the_1600_price(monkeypatch, tmp_path):
    # The DEMO account's take-profit can fill on NinjaTrader's servers while the
    # laptop sleeps -- here only after 16:00, when the race was already out.
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 14, 0), BASE - 10), (et(2026, 10, 2, 16, 0), BASE - 10),
                   (et(2026, 10, 2, 16, 20), BASE + 90), (et(2026, 10, 5, 23, 0), BASE + 90)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 16, 40),
                            jumps=[(et(2026, 10, 2, 15, 0), et(2026, 10, 2, 16, 30))])
    assert eng.t[1]["why"] == "tp"
    t = _trades(tmp_path)[0]
    assert t["reason"] == "exit_while_away" and t["exit_price"] == pytest.approx(BASE - 10.0)
    assert st["run"]["realized_usd"] == pytest.approx(-203.5) and st["runs_blown"] == 0


# --- the rules, directly ---------------------------------------------------------------

def test_rule_helpers(monkeypatch):
    m = load_module("funded_rules")
    run = m.new_run()
    assert m.mll_floor(run) == 48_000
    for peak, floor in ((51_000, 49_000), (52_100, 50_100), (53_500, 50_100)):
        run["peak_eod"] = peak
        assert m.mll_floor(run) == floor                          # locked, no longer trailing
    assert m.payout_need(m.new_run()) == 2_600                    # $52,100 + the $500 first payout
    assert (m.day_take(3_000), m.day_take(20), m.day_take(523.25)) == (1_500, 150, 523.25)
    assert m.trading_days_between("2026-10-02", "2026-10-05") == 1      # Friday only
    assert m.trading_days_between("2026-10-05", "2026-10-09") == 4


def test_fifth_payout_ends_the_simulated_account(monkeypatch):
    m = load_module("funded_fifth")
    monkeypatch.setattr(m, "alert", lambda msg, **kw: None)
    run = m.new_run()
    run.update(realized_usd=14_200.0, paid_usd=9_500.0, peak_eod=54_700.0, mll_locked=True, day="2026-10-02",
               day_start_realized=14_200.0, cycle_days=2, cycle_pnl=2_600.0, cycle_best_day=1_000.0,
               payouts=[{"n": i + 1} for i in range(4)])
    state = {"run": run, "runs_total": 1, "runs_paid": 1, "runs_blown": 0, "runs_done": 0,
             "payouts_total": 4, "paid_total_usd": 9_500.0, "history": [], "halted": False, "halt_reason": ""}
    assert m.end_of_day(state, "2026-10-05", live=False)          # 3 days, $2,600 -> payout 5 of $2,500
    assert run["payouts"][-1]["amount_usd"] == 2_500.0 and state["paid_total_usd"] == 12_000.0
    m.end_run(state, "done", "five payouts taken", live=False)
    assert state["runs_done"] == 1 and state["runs_total"] == 2 and state["run"]["payouts"] == []


# --- the trend rule: short races below the 200-day average ------------------------------

def test_short_race_below_the_200_day_average_takes_the_day(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 2, 0), BASE - 85), (et(2026, 10, 5, 23, 0), BASE - 85)])
    st, eng, alerts = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                                 daily=DOWN_TREND)
    assert eng.orders[0]["direction"] == "short" and eng.orders[0]["tp_dist"] == 75.5
    assert eng.t[1]["why"] == "tp"
    assert st["run"]["realized_usd"] == pytest.approx(1506.5)
    t = _trades(tmp_path)[0]
    assert t["direction"] == "short" and t["pnl_usd"] == pytest.approx(1506.5)
    assert any("short" in a and "above" in a for a in alerts)      # the account ends ABOVE the line


def test_short_race_blows_when_price_rises_to_the_line(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 10, 0), BASE + 120), (et(2026, 10, 5, 23, 0), BASE + 120)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                            daily=DOWN_TREND)
    assert eng.orders[0]["direction"] == "short"
    assert st["runs_blown"] == 1
    assert _trades(tmp_path)[0]["pnl_usd"] == pytest.approx(-2_000.0, abs=15.0)
    assert eng.t[1]["closed"] < et(2026, 10, 2, 8, 0)


def test_always_long_setting_ignores_the_trend(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 0), BASE),
                   (et(2026, 10, 2, 2, 0), BASE + 85), (et(2026, 10, 5, 23, 0), BASE + 85)])
    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 17, 0),
                            daily=DOWN_TREND, side_rule="long")
    assert eng.orders[0]["direction"] == "long" and st["run"]["realized_usd"] == pytest.approx(1506.5)


def test_funded_race_lost_by_ninjatrader_goes_on_on_the_price_history(monkeypatch, tmp_path):
    price = Path_([(et(2026, 10, 1, 18, 0), BASE), (et(2026, 10, 1, 19, 30), BASE),
                   (et(2026, 10, 1, 21, 0), BASE + 10), (et(2026, 10, 1, 23, 0), BASE + 95),
                   (et(2026, 10, 2, 16, 0), BASE + 95), (et(2026, 10, 5, 23, 0), BASE + 95)])

    def forget(e):
        real = e._update

        def upd():
            if e.clock["now"] >= et(2026, 10, 1, 19, 20) and e.t and all(t["open"] for t in e.t.values()):
                e.t.clear()                            # NinjaTrader restarted: no position, no closed deal
            real()
        e._update = upd

    st, eng, _ = run_funded(monkeypatch, tmp_path, price, et(2026, 10, 1, 18, 30), et(2026, 10, 2, 1, 0),
                            setup=forget)
    trades = _trades(tmp_path)
    assert len(trades) == 1 and trades[0]["reason"] == "take_while_away"
    assert st["run"]["realized_usd"] > 1000
