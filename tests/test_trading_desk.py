"""The trading desk (2026-10-21): the Scout's alarm, the Guard's rules, the budget, the debate's order of speakers,
and the whole loop on a paper book -- with a fake Claude, never the real one."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot import trading_desk as td  # noqa: E402

T0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)          # Tue 20 Oct, 08:00 New York


def minute_rows(n_flat: int = 120, jump: float = 160.0, base: float = 30_000.0) -> list[tuple]:
    """Flat, then a jump over 10 minutes; the last row is the forming minute."""
    rows = [(base, base + 2, base - 2, base)] * n_flat
    for i in range(1, 11):
        p = base + jump * i / 10
        rows.append((p - jump / 10, p + 1, p - jump / 10 - 1, p))
    rows.append(rows[-1])
    return rows


class Feed:
    def __init__(self, rows, start=T0):
        self.start, self.rows = start, list(rows)

    def get_bars(self, sym, tf, n):
        if tf == "1m":
            idx = pd.date_range(self.start, periods=len(self.rows), freq="1min", tz="UTC")
            df = pd.DataFrame(self.rows, columns=["open", "high", "low", "close"], index=idx)
        elif tf == "5m":
            one = self.get_bars(sym, "1m", 10_000)
            df = one.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        else:                                                         # 1d: 260 rising days
            idx = pd.date_range(T0 - timedelta(days=260), periods=260, freq="D", tz="UTC")
            df = pd.DataFrame({"open": 25_000.0, "high": 25_100.0, "low": 24_900.0,
                               "close": [25_000.0 + 10 * i for i in range(260)]}, index=idx)
        df["volume"] = 100.0
        return df.tail(n)

    def resolve_symbol(self, s):
        return "NQ DEC26"


class Offline:
    DELAY_S = 600

    @staticmethod
    def bars(tf, n):
        raise OSError("no internet in this test")


def usage(i=1000, o=200):
    return SimpleNamespace(input_tokens=i, output_tokens=o, cache_creation_input_tokens=0, cache_read_input_tokens=0)


class FakeClaude:
    """Answers by speaker: which system prompt and which answer format was asked for."""

    def __init__(self, chart=None, decision=None):
        self.chart = chart or {"direction": "long", "stop": 30_100.0, "take_profit": 30_260.0, "confidence": "medium",
                               "reason": "breakout above the morning range"}
        self.decision = decision or {"decision": "buy", "stop": 30_100.0, "take_profit": 30_260.0,
                                     "reason": "the news explains the move and the stop sits under the breakout"}
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        schema = ((kw.get("output_config") or {}).get("format") or {}).get("schema")
        system, last = kw["system"], kw["messages"][-1]["content"]
        if system == td.NEWS_SYS:
            who, out = ("news_reply", "Yes: the CPI report came in cool.") if "THE SKEPTIC ASKS" in last else \
                ("news", "Cool inflation data lifted tech; medium confidence the move continues.")
        elif system == td.CHART_SYS:
            who, out = ("chart_reply", json.dumps({**self.chart, "reply": "The stop sits under the breakout."})) \
                if schema is td.CHART_REPLY_SCHEMA else ("chart", json.dumps(self.chart))
        elif system == td.SKEPTIC_SYS:
            who, out = ("challenge", json.dumps({"lean": "take", "to_news": "Is this news new?",
                                                 "to_chart": "Isn't the stop too tight?"})) \
                if schema is td.CHALLENGE_SCHEMA else ("decision", json.dumps(self.decision))
        else:
            who, out = "review", json.dumps({"lessons": ["Skip moves that already ran 1% before the alert."],
                                             "summary": "One trade, one win."})
        with self.lock:
            self.calls.append(who)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=out)], stop_reason="end_turn",
                               usage=usage(), model=kw["model"])


def bars_at(rows):
    idx = pd.date_range(T0, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


# --- the Scout ---------------------------------------------------------------------------------------------------

def test_the_scout_wakes_the_desk_only_on_a_big_move():
    a = td.scout(bars_at(minute_rows(jump=160.0)))                 # +0.53% in 15 minutes
    assert a and a["direction"] == "up" and a["pct15"] >= td.MOVE_15
    assert td.scout(bars_at(minute_rows(jump=60.0))) is None        # +0.2%: nothing
    assert td.scout(bars_at(minute_rows(jump=-200.0)))["direction"] == "down"
    assert td.scout(bars_at(minute_rows(n_flat=30))) is None        # under an hour of bars: no judgement


def test_new_trades_only_in_the_window_and_flat_by_1555():
    ny = lambda d, h, m: datetime(2026, 10, d, h, m, tzinfo=td.ET)  # noqa: E731  (Mon 19 ... Sun 25 Oct)
    assert td.may_open(ny(20, 10, 0)) and not td.may_open(ny(20, 15, 30)) and td.may_open(ny(20, 18, 10))
    assert not td.may_open(ny(24, 12, 0)) and td.may_open(ny(25, 18, 10)) and not td.may_open(ny(23, 18, 10))
    assert td.must_be_flat(ny(20, 15, 56)) and not td.must_be_flat(ny(20, 18, 1)) and td.must_be_flat(ny(24, 9, 0))
    assert td.trading_day(ny(20, 18, 30)) == "2026-10-21" and td.trading_day(ny(20, 17, 59)) == "2026-10-20"


# --- the Guard ---------------------------------------------------------------------------------------------------

def test_the_guard_sizes_in_micros_to_at_most_500_dollars_of_risk():
    acct = td.new_account(1)
    g = td.plan_order("buy", 30_100, 30_260, 30_160, 50_000, acct, 0.0)
    assert g["ok"] and g["direction"] == "long" and g["qty"] == 4               # 60 points: $124 a micro
    assert g["risk_usd"] == pytest.approx(4 * (60 * 2 + 3.5 + 0.5))
    assert td.plan_order("sell", 30_180, 30_100, 30_160, 50_000, acct, 0.0)["qty"] == 11   # 20 points: $44
    assert td.plan_order("buy", 30_200, 30_260, 30_160, 50_000, acct, 0.0)["ok"] is False   # stop above a buy
    assert td.plan_order("buy", 29_900, 30_260, 30_160, 50_000, acct, 0.0)["ok"] is False   # 260 points: too far
    assert td.plan_order("buy", 30_100, 30_162, 30_160, 50_000, acct, 0.0)["ok"] is False   # target 2 points away
    assert td.plan_order("skip", 0, 0, 30_160, 50_000, acct, 0.0)["ok"] is False


def test_the_guard_keeps_a_trade_off_the_loss_line_and_the_daily_limit():
    acct = td.new_account(1)                                                    # loss line $48,000
    near = td.plan_order("buy", 30_100, 30_260, 30_160, 48_300, acct, 0.0)      # $200 of room left after $100
    assert near["ok"] and near["risk_usd"] <= 200
    assert td.plan_order("buy", 30_100, 30_260, 30_160, 48_150, acct, 0.0)["ok"] is False
    assert td.plan_order("buy", 30_100, 30_260, 30_160, 50_000, acct, -1_100.0)["ok"] is False   # $50 of the day left


# --- the budget --------------------------------------------------------------------------------------------------

def test_the_budget_counts_every_call_and_stops_at_the_cap():
    assert td.call_cost("claude-sonnet-5-5", usage(1_000_000, 100_000)) == pytest.approx(2.0 + 1.0)
    assert td.call_cost("claude-haiku-5-5", usage(1_000_000, 0)) == pytest.approx(0.10)
    state = {}
    ai = td.DeskAI(FakeClaude(), state, cap=1.0)
    text, _ = ai.call(ai.small, td.NEWS_SYS, [{"role": "user", "content": "x"}], "low")
    assert text and state["budget"]["calls"] == 1 and state["budget"]["spent"] == pytest.approx(0.0002)
    with pytest.raises(td.OutOfBudget):                         # under 5 cents left: no call starts
        td.DeskAI(FakeClaude(), {}, cap=0.04).call(ai.big, td.CHART_SYS, [{"role": "user", "content": "x"}], "low")


def test_a_new_month_starts_a_new_budget():
    state = {"budget": {"month": "2026-09", "spent": 9.99}}
    assert td.budget_left(state, 10.0, T0) == pytest.approx(10.0) and state["budget"]["month"] == "2026-10"


# --- the debate --------------------------------------------------------------------------------------------------

def test_the_skeptic_challenges_hears_the_replies_then_decides():
    fake = FakeClaude()
    ai = td.DeskAI(fake, {})
    out = td.debate(ai, td.scout(bars_at(minute_rows())), "MARKET", "10:01 [CNBC] CPI cools", "ACCOUNT", "")
    assert out["decision"]["decision"] == "buy" and out["decision"]["stop"] == 30_100.0
    assert fake.calls[:3] == ["news", "chart", "challenge"] and fake.calls[-1] == "decision"
    assert sorted(fake.calls[3:5]) == ["chart_reply", "news_reply"] and out["challenge"]["to_chart"]
    assert out["cost"] > 0


def test_no_chart_trade_means_skip_without_a_debate():
    fake = FakeClaude(chart={"direction": "none", "stop": 0, "take_profit": 0, "confidence": "low",
                             "reason": "the move already ran"})
    out = td.debate(td.DeskAI(fake, {}), td.scout(bars_at(minute_rows())), "M", "H", "A", "")
    assert out["decision"]["decision"] == "skip" and fake.calls == ["news", "chart"]


# --- the news ------------------------------------------------------------------------------------------------------

RSS = b"""<?xml version="1.0"?><rss><channel><title>T</title>
<item><title>Nasdaq jumps as CPI cools</title><pubDate>Tue, 20 Oct 2026 13:58:00 GMT</pubDate></item>
<item><title>Nasdaq jumps as CPI cools!</title><pubDate>Tue, 20 Oct 2026 13:59:00 GMT</pubDate></item>
<item><title>Old story</title><pubDate>Mon, 19 Oct 2026 01:00:00 GMT</pubDate></item>
<item><title>Investing.com style &lt;b&gt;date&lt;/b&gt;</title><pubDate>2026-10-20 13:30:00</pubDate></item>
</channel></rss>"""


def test_headlines_are_read_deduplicated_and_kept_fresh():
    items = td.headlines(fetch=lambda url: RSS, now=T0 + timedelta(minutes=5))
    titles = [x["title"] for x in items]
    assert titles[0].startswith("Nasdaq jumps") and len([t for t in titles if "Nasdaq" in t]) == 1
    assert "Old story" not in titles and any("Investing.com style" in t and "<" not in t for t in titles)
    assert "[CNBC]" in td.headlines_text(items) or "[" in td.headlines_text(items)


# --- the whole loop on a paper book --------------------------------------------------------------------------------

def load_runner():
    spec = importlib.util.spec_from_file_location("desk_runner", ROOT / "scripts" / "run_trading_desk.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_desk(tmp_path, client):
    m = load_runner()
    feed = Feed(minute_rows())
    clock = {"now": T0 + timedelta(minutes=len(feed.rows) - 1, seconds=30)}
    eng = m.DeskPaper(account="Desk", feed=feed, path=tmp_path / "paper.json", max_contract_units=td.MAX_MICROS,
                      now=lambda: clock["now"])
    eng.internet = Offline
    sent = []
    m.alert = lambda msg, **kw: sent.append(msg)
    desk = m.Desk(engine=eng, client=client, now=lambda: clock["now"], brain=tmp_path / "brain",
                  state_path=tmp_path / "state.json", journal_path=tmp_path / "journal.json",
                  heads=lambda: [{"t": T0.isoformat(), "src": "CNBC", "title": "CPI cools"}])
    return m, desk, eng, feed, clock, sent


def test_a_big_move_becomes_a_guarded_trade_and_its_result_is_remembered(tmp_path):
    m, desk, eng, feed, clock, sent = make_desk(tmp_path, FakeClaude())
    desk.step()
    e = desk.journal[0]
    assert e["order"]["qty"] == 4 and e["order"]["direction"] == "long" and desk.today()["trades"] == 1
    assert any("BUY 4 MNQ" in s for s in sent)
    desk.step()                                                # an open trade: no second alert
    assert len(desk.journal) == 1
    feed.rows += [(30_160, 30_270, 30_150, 30_265)] * 2         # the next minute reaches the target
    clock["now"] += timedelta(minutes=2)
    eng._bars = None
    desk.step()
    assert e["outcome"]["why"] == "take-profit" and e["outcome"]["pnl"] == pytest.approx(800 - 14)
    assert desk.balance() == pytest.approx(50_786) and desk.today()["pnl"] == pytest.approx(786)
    notes = {p.name for p in (tmp_path / "brain").iterdir()}
    assert notes == {"Trading desk.md", "Desk journal.md", "Desk lessons.md", "Desk results.md"}
    journal = (tmp_path / "brain" / "Desk journal.md").read_text(encoding="utf-8")
    assert "BUY 4 MNQ" in journal and "Skeptic asked the Chart analyst" in journal and "take-profit" in journal
    assert json.loads((tmp_path / "state.json").read_text())["balance"] == pytest.approx(50_786)


def test_without_a_key_the_scout_still_writes_the_alert_down(tmp_path):
    m, desk, *_ = make_desk(tmp_path, None)
    desk.step()
    assert desk.journal[0]["decision"]["decision"] == "skip" and "No Claude key" in desk.journal[0]["decision"]["reason"]
    assert "order" not in desk.journal[0]


def test_an_evaluation_passes_at_plus_3000_and_a_new_one_starts(tmp_path):
    m, desk, eng, *_ , sent = make_desk(tmp_path, None)
    eng.book["balance"] = 53_050.0
    desk.settle()
    assert desk.state["evals"]["passed"] == 1 and desk.state["account"]["n"] == 2 and desk.balance() == 50_000
    eng.book["balance"] = 47_990.0
    desk.settle()
    assert desk.state["evals"]["failed"] == 1 and desk.state["account"]["n"] == 3
    assert any("PASSED" in s for s in sent) and any("failed" in s for s in sent)


def test_the_day_review_writes_lessons_the_agents_read(tmp_path):
    m, desk, eng, feed, clock, sent = make_desk(tmp_path, FakeClaude())
    desk.step()
    clock["now"] = datetime(2026, 10, 20, 16, 30, tzinfo=td.ET).astimezone(timezone.utc)
    desk.journal[0]["day"] = td.trading_day(clock["now"])
    desk.end_of_day()
    assert desk.state["lessons"][0]["text"].startswith("Skip moves")
    assert "Skip moves" in td.lessons_text(desk.state)
    assert "Skip moves" in (tmp_path / "brain" / "Desk lessons.md").read_text(encoding="utf-8")
