"""The trading desk's loop (src/metatraderbot/trading_desk.py) -- PAPER ONLY, its own account "Desk".

Every minute: let the paper book close what hit its stop or take-profit, close everything at 15:55 New York, settle
the evaluation (+$3,000 passes, the loss line fails -- then a new one starts), let the Scout look for a big move and,
when it finds one the rules allow, run the debate (News, Chart, Skeptic) and place the Skeptic's trade through the
Guard. After the close the Skeptic reviews the day and writes the lessons. Its brain: notes/Trading desk/.

Started by Vertex.ps1 (the watchdog) on the 24/7 laptop. Needs DESK_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) in
.env; without one the Scout still watches and every alert is written down as skipped.

Usage:  .venv/Scripts/python.exe scripts/run_trading_desk.py          (the desk, forever)
        .venv/Scripts/python.exe scripts/run_trading_desk.py --once   (one debate on the market now, no order --
                                                                         a test of the agents; costs a few cents)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from metatraderbot import target_race, trading_desk as td  # noqa: E402
from metatraderbot.alerts import alert  # noqa: E402
from metatraderbot.nt_engine import RETCODE_DONE  # noqa: E402
from metatraderbot.paper_engine import PaperEngine  # noqa: E402

VAULT = ROOT.parents[1]
BRAIN = VAULT / "notes" / "Trading desk"
STATE = ROOT / "data" / "desk_state.json"
JOURNAL = ROOT / "data" / "desk_journal.json"
BOOK = "Desk"

log = logging.getLogger("desk")


def file_log() -> None:
    """logs/trading_desk.log -- only when the desk really runs (an import, e.g. by the tests, writes nothing)."""
    if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        (ROOT / "logs").mkdir(exist_ok=True)
        h = RotatingFileHandler(ROOT / "logs" / "trading_desk.log", maxBytes=2_000_000, backupCount=2,
                                encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)


class DeskPaper(PaperEngine):
    """The bots' paper engine, in micro contracts ($2 a point)."""

    def usd_per_point(self, symbol: str):
        return td.PV if str(symbol).upper().startswith("MNQ") else 20.0


def _read(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for i in range(5):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.3 * (i + 1))


class Desk:
    def __init__(self, engine=None, client=None, now=None, brain: Path = BRAIN, state_path: Path = STATE,
                 journal_path: Path = JOURNAL, heads=None):
        self.engine = engine or DeskPaper(account=BOOK, max_contract_units=td.MAX_MICROS)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.brain, self.state_path, self.journal_path = brain, state_path, journal_path
        self.state = _read(state_path, {})
        self.state.setdefault("account", td.new_account(1))
        self.state.setdefault("evals", {"passed": 0, "failed": 0, "fees": td.EVAL_FEE})
        self.state.setdefault("days", {})
        self.state.setdefault("lessons", [])
        self.state.setdefault("reviews", {})
        self.journal: list[dict] = _read(journal_path, [])
        big, small, cap = td.env_models()
        self.ai = td.DeskAI(client, self.state, cap, big, small) if client is not None else None
        self.heads = heads or td.headlines

    # -- memory --
    def save(self) -> None:
        self.state["heartbeat"] = self.now().isoformat()
        self.state["balance"] = self.balance()
        _write(self.state_path, self.state)
        _write(self.journal_path, self.journal[:400])

    def brain_write(self) -> None:
        td.write_brain(self.brain, self.state, self.journal, self.balance())

    def balance(self) -> float:
        return float(self.engine.book.get("balance", td.START))

    def today(self) -> dict:
        return self.state["days"].setdefault(td.trading_day(self.now()), {"alerts": 0, "trades": 0, "pnl": 0.0})

    # -- the account --
    def settle(self) -> None:
        """Closed trades -> the journal; then the evaluation: passed, failed, or on."""
        open_now = self.engine.get_all_open_tickets()       # also closes what hit its stop or target
        closed = False
        for e in self.journal:
            o = e.get("order") or {}
            if o and not e.get("outcome") and int(o.get("ticket", 0)) not in open_now:
                deal = self.engine.get_closed_deal(o["ticket"])
                if not deal:
                    continue
                why = {"tp": "take-profit", "sl": "stop-loss"}.get(deal["close_reason"], "closed at 15:55")
                e["outcome"] = {"exit": deal["exit_price"], "pnl": deal["pnl_usd"], "why": why,
                                "closed": deal["closed_ts"].isoformat()}
                day = self.state["days"].setdefault(e.get("day") or td.trading_day(self.now()),
                                                    {"alerts": 0, "trades": 0, "pnl": 0.0})
                day["pnl"] = round(float(day.get("pnl", 0.0)) + float(deal["pnl_usd"]), 2)
                alert(f"Desk trade closed: {why}, {td._usd(deal['pnl_usd'])}. Balance ${self.balance():,.2f}.",
                      book="Desk")
                log.info("closed %s: %s %s", o["ticket"], why, deal["pnl_usd"])
                closed = True
        if closed:
            self.brain_write()
        if open_now:
            return
        acct, bal = self.state["account"], self.balance()
        if bal >= td.START + td.TARGET:
            self._next_eval(True, bal)
        elif bal <= td.floor_of(acct):
            self._next_eval(False, bal)

    def _next_eval(self, passed: bool, bal: float) -> None:
        acct, ev = self.state["account"], self.state["evals"]
        ev["passed" if passed else "failed"] = int(ev.get("passed" if passed else "failed", 0)) + 1
        ev["fees"] = float(ev.get("fees", td.EVAL_FEE)) + (td.EVAL_FEE if passed else td.RESET_FEE)
        ev.setdefault("history", []).append({"n": acct["n"], "outcome": "passed" if passed else "failed",
                                             "balance": bal, "ended": self.now().isoformat()})
        msg = (f"Desk evaluation #{acct['n']} {'PASSED' if passed else 'failed'} at ${bal:,.2f}. "
               f"A new one starts (${td.EVAL_FEE if passed else td.RESET_FEE:,.0f}).")
        alert(msg, book="Desk")
        log.warning(msg)
        self.engine.book["balance"] = td.START
        self.engine._save()
        self.state["account"] = td.new_account(int(acct["n"]) + 1)

    def end_of_day(self) -> None:
        """After 16:00 New York: the loss line follows the closing balance; the Skeptic reviews the day."""
        day = td.trading_day(self.now())
        if self.state.get("closed_day") == day:
            return
        self.state["closed_day"] = day
        acct = self.state["account"]
        acct["peak_eod"] = max(float(acct.get("peak_eod", td.START)), self.balance())
        todays = [e for e in self.journal if e.get("day") == day]
        if todays and self.ai is not None:
            got = td.review(self.ai, todays, self.state["lessons"])
            if got:
                self.state["lessons"] = ([{"day": day, "text": t} for t in got["lessons"]] + self.state["lessons"])[:40]
                self.state["reviews"][day] = got["summary"]
                log.info("review %s: %s", day, got["summary"])
        self.brain_write()

    # -- one minute --
    def step(self) -> None:
        now = self.now()
        self.settle()
        if td.must_be_flat(now) and self.engine.get_all_open_tickets():
            n = self.engine.close_all_bot_positions()
            log.info("15:55 New York: flat (%s left)", n)
            self.settle()
        if td.after_close(now):
            self.end_of_day()
        self.maybe_trade(now)
        self.save()

    def maybe_trade(self, now: datetime) -> None:
        day = self.today()
        if (not td.may_open(now) or target_race.skip_day(td.trading_day(now)) or self.engine.get_all_open_tickets()
                or day["trades"] >= td.MAX_TRADES_DAY or day["alerts"] >= td.MAX_ALERTS_DAY
                or float(day.get("pnl", 0.0)) <= -td.DLL + 50):
            return
        last = self.state.get("last_alert")
        if last and (now - datetime.fromisoformat(last)).total_seconds() < td.COOL_DOWN_MIN * 60:
            return
        if not self.engine.is_tradeable("NQ"):
            return                                          # no fresh prices (closed, internet down)
        bars = self.engine.minute_bars()
        found = td.scout(bars)
        if not found:
            return
        self.state["last_alert"] = now.isoformat()
        day["alerts"] += 1
        self.run_alert(found, bars, place=True)

    def run_alert(self, found: dict, bars, place: bool) -> dict:
        now = self.now()
        entry = {"id": uuid.uuid4().hex[:10], "at": now.isoformat(), "day": td.trading_day(now), "alert": found}
        log.info("Scout: %s", td.alert_words(found))
        if self.ai is None:
            entry["decision"] = {"decision": "skip", "stop": 0, "take_profit": 0,
                                 "reason": "No Claude key on this computer: the agents could not be asked."}
        else:
            try:
                brief = td.market_brief(bars, self.engine.get_bars("NQ", "5m", 400),
                                        self.engine.get_bars("NQ", "1d", 260))
                heads = td.headlines_text(self.heads())
                acct_txt = td.account_text(self.balance(), self.state["account"], self.today())
                entry.update(td.debate(self.ai, found, brief, heads, acct_txt, td.lessons_text(self.state)))
            except Exception as e:  # noqa: BLE001 -- an agent failing must never stop the desk
                log.exception("debate failed")
                entry["decision"] = {"decision": "skip", "stop": 0, "take_profit": 0,
                                     "reason": f"The debate failed ({type(e).__name__}); nothing was traded."}
        d = entry["decision"]
        if d.get("decision") in ("buy", "sell"):
            price = float(self.engine.quote("NQ")["last"])
            g = td.plan_order(d["decision"], d["stop"], d["take_profit"], price, self.balance(),
                              self.state["account"], float(self.today().get("pnl", 0.0)))
            entry["guard"] = g
            if g["ok"] and place:
                r = self.engine.place_market_order(td.SYMBOL, g["direction"], g["qty"], g["stop"], g["tp"], td.MAGIC,
                                                   comment="desk", signal_entry_price=price)
                if r.retcode == RETCODE_DONE:
                    entry["order"] = {"ticket": r.ticket, "direction": g["direction"], "qty": g["qty"],
                                      "entry": r.price, "stop": r.sl, "tp": r.tp, "risk_usd": g["risk_usd"]}
                    self.today()["trades"] += 1
                    alert(f"Desk: {d['decision'].upper()} {g['qty']} MNQ at {r.price:,.2f}, stop {r.sl:,.2f}, "
                          f"target {r.tp:,.2f} (${g['risk_usd']:,.0f} at risk). Skeptic: {d.get('reason', '')[:300]}",
                          book="Desk")
                else:
                    entry["guard"] = {**g, "ok": False, "why": f"the paper engine refused: {r.comment}"}
        log.info("decision: %s -- %s", d.get("decision"), str(d.get("reason", ""))[:300])
        b = self.state.get("budget") or {}
        if self.ai is not None and td.budget_left(self.state, self.ai.cap, now) < 0.05 and not b.get("warned"):
            b["warned"] = True
            alert(f"Desk: the ${self.ai.cap:g} AI budget for {b.get('month')} is used up -- the Scout keeps "
                  f"watching, the desk trades again next month.", book="Desk")
        if place:
            self.journal.insert(0, entry)
            self.brain_write()
        return entry


def client_from_env():
    key = (os.environ.get("DESK_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def main() -> int:
    if os.environ.get("VERTEX_PAPER", "").strip() != "1":
        log.error("the trading desk runs on paper only (VERTEX_PAPER=1) -- not starting")
        return 1
    if "--once" in sys.argv:                                # one debate on the market now, nothing placed or kept:
        import tempfile                                     # this laptop's data folder is the 24/7 laptop's copy
        tmp = Path(tempfile.mkdtemp(prefix="desk_once_"))
        desk = Desk(engine=DeskPaper(account=BOOK, path=tmp / "paper.json", max_contract_units=td.MAX_MICROS),
                    client=client_from_env(), brain=tmp / "brain", state_path=tmp / "state.json",
                    journal_path=tmp / "journal.json")
        bars = desk.engine.minute_bars()
        last = float(bars["close"].iloc[-2])
        found = td.scout(bars) or {"direction": "up", "pct15": 0.0, "pct60": 0.0, "price": last,
                                   "at": str(bars.index[-2]), "test": True}
        e = desk.run_alert(found, bars, place=False)
        print(json.dumps({k: e.get(k) for k in ("news", "chart", "challenge", "news_reply", "chart_reply",
                                                 "decision", "guard", "cost")}, indent=1, default=str))
        return 0
    file_log()
    desk = Desk(client=client_from_env())
    log.warning("trading desk started: paper account %s, balance $%s, AI %s", BOOK, f"{desk.balance():,.2f}",
                "on" if desk.ai else "OFF (no Claude key in .env)")
    desk.brain_write()
    while True:
        t0 = time.monotonic()
        try:
            desk.step()
        except Exception:  # noqa: BLE001 -- one bad minute must not end the desk
            log.exception("step failed")
        time.sleep(max(5.0, 60.0 - (time.monotonic() - t0)))


if __name__ == "__main__":
    sys.exit(main())
