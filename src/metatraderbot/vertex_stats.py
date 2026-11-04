"""Everything Vertex (the owner's voice butler, 2026-10-09) can tell: one JSON summary built from the bots'
own records -- the trial (evaluation) bot, every funded account, the prediction lab and the research
expectations -- by period (today, yesterday, this week, this month, all time), plus the live races, the next
race and an event list for spoken announcements.

Every figure has a stable path (e.g. "trial.periods.today.evaluations_passed") so the voice app can show the
number itself as proof, looked up here rather than taken from the model's words. Read-only.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import lucid_rules, target_race

ANCHOR = 50_000.0
SPLIT = 0.90
# What the research expects (notes/Research log.md) -- said alongside the live numbers.
EXPECTATIONS = {
    "evaluation_race_pass_rate": "about 41% of evaluation races reach +$3,000 before the $2,000 loss line",
    "funded_payout_rate": "about 1 in 3 funded accounts reaches its first payout, median 4 trading days",
    "funded_race_win_rate": "a funded race reaches its take-profit about 2 days in 3",
    "money_per_account_month": "long-run replay 2019-2026: about $1,445 a month per account at $115 evaluations",
    "plan": "start with 1 account (decided 14 Oct), buy the next each time the money made reaches another +$2,500, up to 5",
    "plan_odds": "replay 2024-2026: in profit after 3 / 6 / 12 months 75% / 83% / 100%; typical first year about +$92,000;"
                 " worst dip about $5,300; all 5 accounts running after about 4 months",
}


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _day(ts: str | None) -> str | None:
    try:
        return lucid_rules.trading_day(datetime.fromisoformat(ts))
    except (TypeError, ValueError):
        return None


def _periods(today: str) -> dict[str, tuple[str, str]]:
    """Trading-day ranges (inclusive): today, yesterday (previous weekday), this week, this month, all time."""
    d = date.fromisoformat(today)
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return {"today": (today, today), "yesterday": (prev.isoformat(), prev.isoformat()),
            "this_week": ((d - timedelta(days=d.weekday())).isoformat(), today),
            "this_month": (d.replace(day=1).isoformat(), today), "all_time": ("0000-01-01", today)}


def _result(t: dict) -> str:
    r = str(t.get("reason") or "")
    if r in ("tp", "pass_while_away", "take_while_away"):
        return "target"
    if r.startswith(("line", "mll", "fail")) or float(t.get("pnl_usd") or 0) <= -1_800 or             (t.get("line") and abs(float(t["exit_price"]) - float(t["line"])) <= 0.5):
        return "loss line"          # (a race the bot closed at the line is booked a little past it)
    if r in ("time_exit", "exit_while_away"):
        return "16:00 close"
    return "closed"


def _trade_row(t: dict, bot: str) -> dict:
    o = t.get("opened_ts")
    return {"bot": bot, "account": t.get("account"), "number": t.get("attempt") or t.get("run"),
            "trading_day": _day(t.get("closed_ts")), "opened": o, "closed": t.get("closed_ts"),
            "direction": t.get("direction", "long"), "contracts": t.get("lots"),
            "entry": t.get("entry_price"), "exit": t.get("exit_price"), "result": _result(t),
            "pnl_usd": round(float(t.get("pnl_usd") or 0.0), 2)}


def _trade_stats(rows: list[dict], lo: str, hi: str) -> dict:
    sel = [r for r in rows if r["trading_day"] and lo <= r["trading_day"] <= hi]
    return {"races": len(sel), "reached_target": sum(r["result"] == "target" for r in sel),
            "hit_loss_line": sum(r["result"] == "loss line" for r in sel),
            "closed_at_16": sum(r["result"] == "16:00 close" for r in sel),
            "pnl_usd": round(sum(r["pnl_usd"] for r in sel), 2)}


def trial_summary(data: Path, periods: dict) -> dict:
    st = _read_json(data / "state_lucid_futures.json")
    rows = [_trade_row(t, "trial") for t in _read_jsonl(data / "lucid_futures_trades.jsonl")]
    hist = st.get("history") or []
    out_periods = {}
    for name, (lo, hi) in periods.items():
        ev = [h for h in hist if (_day(h.get("resolved_ts")) or "") and lo <= _day(h.get("resolved_ts")) <= hi]
        passed = sum(h.get("outcome") == "pass" for h in ev)
        failed = sum(h.get("outcome") == "fail" for h in ev)
        s = _trade_stats(rows, lo, hi)
        s.update(evaluations_passed=passed, evaluations_failed=failed,
                 pass_rate=round(passed / (passed + failed), 3) if passed + failed else None)
        out_periods[name] = s
    if not hist:                                                # older records: the counters only
        out_periods["all_time"].update(evaluations_passed=int(st.get("passes") or 0),
                                       evaluations_failed=int(st.get("fails") or 0))
    att = st.get("attempt") or {}
    open_race = None
    for k, v in (att.get("tickets") or {}).items():
        open_race = {"direction": v.get("direction"), "contracts": v.get("lots"), "entry": v.get("entry_price"),
                     "take_profit": v.get("tp"), "loss_line": v.get("line"), "opened": v.get("opened_ts"),
                     "why_this_direction": v.get("side_reason"), "open_pnl_usd": att.get("floating_usd")}
    return {"what": "evaluation (trial) bot on the simulator: races 2 NQ for +$3,000 before the $2,000 loss line; "
                    "a new evaluation starts as soon as one ends",
            "evaluations_total": int(st.get("attempts") or 0), "passed_total": int(st.get("passes") or 0),
            "failed_total": int(st.get("fails") or 0),
            "current_evaluation": {"number": int(st.get("attempts") or 0) + 1,
                                   "made_so_far_usd": round(float(att.get("realized_usd") or 0.0), 2),
                                   "still_needed_usd": round(3_000.0 - float(att.get("realized_usd") or 0.0), 2)},
            "open_race": open_race, "periods": out_periods, "recent_trades": rows[-10:][::-1]}


def _funded_files(data: Path) -> list[tuple[Path, Path]]:
    out = []
    for p in [data / "state_lucid_futures_funded.json"] + sorted(data.glob("state_lucid_futures_funded_*.json")):
        tag = p.stem[len("state_lucid_futures_funded"):]
        if tag == "_roster" or not p.exists():
            continue
        out.append((p, data / f"lucid_futures_funded_trades{tag}.jsonl"))
    return out


def funded_summary(data: Path, periods: dict) -> dict:
    accounts, rows, payouts = [], [], []
    blown_days = []
    for sp, tp in _funded_files(data):
        st = _read_json(sp)
        name = str(st.get("account") or sp.stem)
        run = st.get("run") or {}
        for t in _read_jsonl(tp):
            rows.append({**_trade_row(t, "funded"), "account": name})
        for h in st.get("history") or []:
            for p in h.get("payout_list") or []:
                payouts.append({"account": name, "funded_account": h.get("run"), **p})
            if h.get("outcome") == "blown":
                blown_days.append(_day(h.get("end")))
        for p in run.get("payouts") or []:
            payouts.append({"account": name, "funded_account": st.get("runs_total"), **p})
        realized, paid = float(run.get("realized_usd") or 0.0), float(run.get("paid_usd") or 0.0)
        today_pnl = realized - float(run.get("day_start_realized", realized) or 0.0)
        open_race = None
        for v in (run.get("tickets") or {}).values():
            open_race = {"direction": v.get("direction"), "entry": v.get("entry_price"), "take_profit": v.get("tp"),
                         "loss_line": v.get("line"), "opened": v.get("opened_ts"),
                         "why_this_direction": v.get("side_reason"), "open_pnl_usd": st.get("floating_usd")}
        accounts.append({"account": name, "funded_account_number": int(st.get("runs_total") or 1),
                         "balance_usd": round(ANCHOR + realized - paid, 2),
                         "next_payout_needs_usd": st.get("need_usd"), "today_pnl_usd": round(today_pnl, 2),
                         "payouts_from_this_account": len(run.get("payouts") or []),
                         "funded_accounts_blown_so_far": int(st.get("runs_blown") or 0),
                         "open_race": open_race, "stopped": bool(st.get("halted"))})
    out_periods = {}
    for name, (lo, hi) in periods.items():
        s = _trade_stats(rows, lo, hi)
        pp = [p for p in payouts if (_day(p.get("ts")) or "") and lo <= _day(p.get("ts")) <= hi]
        s.update(payouts=len(pp), paid_usd=round(sum(float(p.get("amount_usd") or 0) for p in pp), 2),
                 to_you_usd=round(SPLIT * sum(float(p.get("amount_usd") or 0) for p in pp), 2),
                 accounts_blown=sum(1 for d in blown_days if d and lo <= d <= hi))
        out_periods[name] = s
    return {"what": "funded bot(s) on the simulator: 1 NQ race a day at 19:00 New York, take-profit = what the next "
                    "payout still needs (at most $1,500), payouts $500 first then $2,500, you keep 90%",
            "accounts": accounts, "periods": out_periods, "recent_trades": sorted(rows, key=lambda r: r["closed"] or "")[-10:][::-1],
            "payouts": sorted(payouts, key=lambda p: p.get("ts") or "")[::-1][:10]}


def desk_summary(data: Path, now: datetime | None = None) -> dict | None:
    """The trading desk (scripts/run_trading_desk.py, 2026-10-21) in short; None before it first ran."""
    from . import trading_desk as td
    st = _read_json(data / "desk_state.json")
    if not st:
        return None
    try:
        journal = json.loads((data / "desk_journal.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        journal = []
    acct, ev, bud = st.get("account") or {}, st.get("evals") or {}, st.get("budget") or {}
    today = (st.get("days") or {}).get(td.trading_day(now)) or {}
    return {"what": "the trading desk: four AI agents (Scout finds big moves, News reads headlines, Chart plans the "
                    "trade, Skeptic argues with both and decides) on their own PAPER account in micro NQ contracts; "
                    "trades on big moves, at most 2 a day, $500 at most at risk a trade, AI fees capped at $10 a month",
            "evaluation_number": acct.get("n", 1), "balance_usd": st.get("balance"),
            "loss_line_usd": td.floor_of(acct), "target_usd": td.START + td.TARGET,
            "evaluations_passed": ev.get("passed", 0), "evaluations_failed": ev.get("failed", 0),
            "today": {"alerts": today.get("alerts", 0), "trades": today.get("trades", 0),
                      "pnl_usd": today.get("pnl", 0.0)},
            "ai_fees_this_month_usd": bud.get("spent", 0.0), "last_seen_utc": st.get("heartbeat"),
            "recent": [{"at": e.get("at"), "decision": (e.get("decision") or {}).get("decision"),
                        "reason": str((e.get("decision") or {}).get("reason", ""))[:300],
                        "result_usd": (e.get("outcome") or {}).get("pnl")} for e in journal[:5]],
            "lessons": [x.get("text") for x in (st.get("lessons") or [])[:5]]}


def summary(data: Path, now: datetime | None = None) -> dict:
    """The whole picture, as plain JSON."""
    now = now or datetime.now(timezone.utc)
    today = lucid_rules.trading_day(now)
    periods = _periods(today)
    trial = trial_summary(data, periods)
    funded = funded_summary(data, periods)
    tr_state = _read_json(data / "state_lucid_futures.json")
    one_a_day = os.environ.get("LUCID_TRIAL_ONE_RACE_A_DAY", "1").strip() != "0"      # since 2026-10-15
    done = tr_state.get("race_day") if one_a_day or tr_state.get("race_attempt") == int(tr_state.get("attempts") or 0) + 1         else None
    try:
        nxt = target_race.next_entry(now, target_race.RaceConfig(), done)
        next_race = {"new_york": nxt.astimezone(lucid_rules.ET).strftime("%a %d %b %H:%M"),
                     "utc": nxt.isoformat()}
    except Exception:  # noqa: BLE001
        next_race = None
    lab = _read_json(data / "prediction_lab.json")
    preds = {k: {kk: v.get(kk) for kk in ("raced", "resolved", "wins", "win_rate", "deployed_same_days", "status")}
             for k, v in (lab.get("summary") or {}).items()}
    return {"as_of_utc": now.isoformat(), "now_new_york": now.astimezone(lucid_rules.ET).strftime("%a %d %b %H:%M"),
            "trading_day": today, "periods": {k: {"from": a, "to": b} for k, (a, b) in periods.items()},
            "market_session": lucid_rules.session_state(now).value, "next_race": next_race,
            "trial": trial, "funded": funded,
            "fleet_real_test": real_test(data),
            "prediction_lab": {"what": "ideas scored on new days without trading; one is worth switching to only after"
                                       " 60+ races clearly above the deployed race", "days_scored": lab.get("days"),
                               "predictors": preds},
            "expectations": EXPECTATIONS, "desk": desk_summary(data, now)}


def events(data: Path, since: datetime) -> list[dict]:
    """Things worth saying out loud, newer than `since`: races opened / finished, evaluations passed or failed,
    payouts, funded accounts blown. Each has a short sentence."""
    out = []
    since_iso = since.isoformat()
    st = _read_json(data / "state_lucid_futures.json")
    for v in ((st.get("attempt") or {}).get("tickets") or {}).values():
        if (v.get("opened_ts") or "") > since_iso:
            out.append({"ts": v["opened_ts"], "bot": "trial", "kind": "race_opened",
                        "text": f"Evaluation race started: {'sold short' if v.get('direction') == 'short' else 'bought'}"
                                f" {v.get('lots', 2):g} NQ at {float(v.get('entry_price') or 0):,.2f}."})
    for t in _read_jsonl(data / "lucid_futures_trades.jsonl"):
        if (t.get("closed_ts") or "") > since_iso:
            r = _result(t)
            word = {"target": "reached its target", "loss line": "hit the loss line",
                    "16:00 close": "closed at 16:00"}.get(r, "closed")
            out.append({"ts": t["closed_ts"], "bot": "trial", "kind": "race_closed",
                        "text": f"Evaluation {t.get('attempt')} race {word}: "
                                f"{'+' if float(t.get('pnl_usd') or 0) >= 0 else 'minus '}${abs(float(t.get('pnl_usd') or 0)):,.0f}."})
    for h in st.get("history") or []:
        if (h.get("resolved_ts") or "") > since_iso:
            out.append({"ts": h["resolved_ts"], "bot": "trial", "kind": f"evaluation_{h.get('outcome')}",
                        "text": "An evaluation PASSED." if h.get("outcome") == "pass" else "An evaluation failed."})
    for sp, tp in _funded_files(data):
        fs = _read_json(sp)
        name = str(fs.get("account") or sp.stem)
        for v in ((fs.get("run") or {}).get("tickets") or {}).values():
            if (v.get("opened_ts") or "") > since_iso:
                out.append({"ts": v["opened_ts"], "bot": "funded", "kind": "race_opened",
                            "text": f"Funded race started on {name}: aiming for +${float(v.get('take_usd') or 0):,.0f}."})
        for t in _read_jsonl(tp):
            if (t.get("closed_ts") or "") > since_iso:
                pnl = float(t.get("pnl_usd") or 0)
                out.append({"ts": t["closed_ts"], "bot": "funded", "kind": "race_closed",
                            "text": f"Funded race on {name} {'won' if pnl > 0 else 'closed'}: "
                                    f"{'+' if pnl >= 0 else 'minus '}${abs(pnl):,.0f}."})
        pays = [p for h in fs.get("history") or [] for p in h.get("payout_list") or []] + \
            list((fs.get("run") or {}).get("payouts") or [])
        for p in pays:
            if (p.get("ts") or "") > since_iso:
                out.append({"ts": p["ts"], "bot": "funded", "kind": "payout",
                            "text": f"Payout taken on {name}: ${float(p.get('amount_usd') or 0):,.0f}, "
                                    f"${float(p.get('to_you_usd') or 0):,.0f} to you."})
        for h in fs.get("history") or []:
            if (h.get("end") or "") > since_iso and h.get("outcome") == "blown":
                out.append({"ts": h["end"], "bot": "funded", "kind": "account_blown",
                            "text": f"Funded account {h.get('run')} on {name} was blown."})
    return sorted(out, key=lambda e: e["ts"])


def lookup(stats: dict, path: str):
    """The value at a dotted path ("trial.periods.today.races"); list items by index. None if absent."""
    cur = stats
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


# --- what Vertex says and hears, for the dashboard (owner 2026-10-09: "keep it all in that dashboard") ----------
#
# The laptops' Vertex reports each answer, announcement and state change here (POST /vertex/event); the dashboard
# shows them (the Vertex page + the card on every page) and follows the voice commands meant for its laptop. The
# dashboard also leaves typed questions and switches for a laptop's Vertex (GET /vertex/inbox).
#   data/vertex_log.jsonl    exchanges, announcements, typed questions, switches -- newest last
#   data/vertex_state.json   per laptop: listening / thinking / speaking / idle, voice and announcements on/off

VERTEX_LOG = "vertex_log.jsonl"
VERTEX_STATE = "vertex_state.json"
ONLINE_S = 90                                       # a laptop's Vertex checks in every 30 s
_LOG_KINDS = ("exchange", "announce", "ask", "listen", "set")
_vx_lock = threading.Lock()


def viewer_name(v) -> str:
    """A laptop's name as used in the dashboard link (?viewer=) and in Vertex's settings: letters, digits, - and _."""
    return re.sub(r"[^A-Za-z0-9_-]+", "", str(v or ""))[:40]


def _clean(v, n: int = 2000) -> str:
    return str(v if v is not None else "")[:n]


def record_vertex_event(data: Path, ev: dict) -> dict | None:
    """Store one report from a laptop's Vertex or one request from the dashboard. Returns what was stored."""
    if not isinstance(ev, dict):
        raise ValueError("an event must be a JSON object")
    kind = str(ev.get("kind", ""))
    now = datetime.now(timezone.utc).isoformat()
    viewer = viewer_name(ev.get("viewer"))
    if kind == "state":
        with _vx_lock:
            st = _read_json(data / VERTEX_STATE)
            st[viewer or "?"] = {"state": _clean(ev.get("state"), 20), "ts": now,
                                 "voice": bool(ev.get("voice", True)), "announce": bool(ev.get("announce", True)),
                                 "detail": _clean(ev.get("detail"), 200)}
            tmp = data / (VERTEX_STATE + ".tmp")
            tmp.write_text(json.dumps(st), encoding="utf-8")
            os.replace(tmp, data / VERTEX_STATE)
        return st[viewer or "?"]
    if kind not in _LOG_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    out = {"id": uuid.uuid4().hex[:12], "ts": now, "kind": kind, "viewer": viewer}
    if kind == "exchange":
        out.update(question=_clean(ev.get("question"), 500), speech=_clean(ev.get("speech")),
                   source=_clean(ev.get("source"), 10), reply_to=_clean(ev.get("reply_to"), 20),
                   proof=[{"label": _clean(p.get("label"), 80), "path": _clean(p.get("path"), 120),
                           "value": _clean(p.get("value"), 80), "found": bool(p.get("found"))}
                          for p in (ev.get("proof") or [])[:12] if isinstance(p, dict)],
                   show=[_clean(s, 40) for s in (ev.get("show") or [])[:4]],
                   ui={k: v for k, v in (ev.get("ui") or {}).items() if isinstance(v, (str, int))},
                   did=[_clean(d, 120) for d in (ev.get("did") or [])[:8]])   # what Vertex did on the PC
    elif kind == "announce":
        out.update(text=_clean(ev.get("text"), 500), key=_clean(ev.get("key"), 200))
    elif kind == "ask":
        out.update(text=_clean(ev.get("text"), 500))
    elif kind == "set":
        out.update({k: bool(ev[k]) for k in ("voice", "announce") if k in ev})
    p = data / VERTEX_LOG
    with _vx_lock:
        if kind == "announce" and out["key"]:       # both laptops announce the same event: keep it once
            if any(e.get("key") == out["key"] for e in _read_jsonl(p)[-300:]):
                return None
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        try:
            if p.stat().st_size > 1_000_000:        # keep the file small: the newest 1,500 lines
                lines = p.read_text(encoding="utf-8").splitlines()[-1500:]
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass
    return out


def vertex_feed(data: Path, limit: int = 200) -> list[dict]:
    """The newest `limit` log entries, oldest first."""
    return _read_jsonl(data / VERTEX_LOG)[-limit:]


def vertex_live(data: Path, now: datetime | None = None) -> dict[str, dict]:
    """Each laptop's Vertex: its last state, and whether it checked in within ONLINE_S seconds."""
    now = now or datetime.now(timezone.utc)
    out = {}
    for viewer, v in _read_json(data / VERTEX_STATE).items():
        try:
            age = (now - datetime.fromisoformat(v.get("ts", ""))).total_seconds()
        except (TypeError, ValueError):
            age = 1e9
        out[viewer] = dict(v, online=age < ONLINE_S and v.get("state") != "offline", age_s=round(age))
    return out


def vertex_inbox(data: Path, viewer: str, after: str = "") -> list[dict]:
    """Typed questions and switches the dashboard left for this laptop's Vertex, newer than `after` (a ts)."""
    viewer = viewer_name(viewer)
    after = (after or "").strip().replace(" ", "+")
    return [e for e in _read_jsonl(data / VERTEX_LOG)[-300:]
            if e.get("kind") in ("ask", "listen", "set") and e.get("viewer") == viewer and e.get("ts", "") > after]


VERTEX_SCREEN = "vertex_screen.json"


def record_screen(data: Path, viewer: str, page: str) -> None:
    """What the dashboard on a laptop shows (Vertex tells Claude, so "open the payouts tab" finds the page)."""
    with _vx_lock:
        st = _read_json(data / VERTEX_SCREEN)
        st[viewer_name(viewer)] = {"page": str(page).lower()[:20], "ts": datetime.now(timezone.utc).isoformat()}
        tmp = data / (VERTEX_SCREEN + ".tmp")
        tmp.write_text(json.dumps(st), encoding="utf-8")
        os.replace(tmp, data / VERTEX_SCREEN)


def vertex_screen(data: Path, viewer: str) -> dict:
    """The page this laptop's dashboard shows -- or, with no ?viewer= link, the last one any dashboard showed."""
    st = _read_json(data / VERTEX_SCREEN)
    return st.get(viewer_name(viewer)) or st.get("") or {}


# --- the Fleet page's "Real test": your plan simulated on real prices since 1 Oct ---------------------------------

REAL_TEST_START = "2026-10-01"
REAL_TEST_TTL_S = 300               # worked out again every 5 minutes, in the background (~3 s of work)
_real_test = {"at": -1e18, "data": None, "busy": False}
_WORDS = {"evaluation bought": "bought an evaluation",
          "evaluation failed -- reset": "evaluation race lost, reset bought",
          "evaluation PASSED -- funded account": "evaluation passed, the funded account starts",
          "evaluation race closed at 16:00": "evaluation race closed at 16:00",
          "funded race won": "funded race won",
          "funded account blown -- new evaluation": "funded account blown, new evaluation bought",
          "no race -- payout money reached": "no race, payout money reached"}


def _compute_real_test(data: Path) -> dict:
    from . import free_prices, lucid_fleet as lf
    bars = lf.load_minutes(data / "futures", free_prices.FreeFeed() if free_prices.in_use() else None,
                           since=REAL_TEST_START)
    res = lf.run(bars, lf.FleetConfig(slots=5, start=REAL_TEST_START, stagger=0, eval_fee=115.0, reset_fee=120.0,
                                      start_slots=1, step_usd=2500.0, step_on="money",
                                      bot_races=lf.bot_races(data, since=REAL_TEST_START)))   # the bots' own races
    plan = lf.PLANS["50K"]
    sl, ev = res["slots"], res["events"]
    accounts = []
    for s in sl:
        a = {"account": s["n"], "now": {"waiting": "not bought yet", "eval": "in an evaluation",
                                        "funded": "funded account"}.get(s["stage"], s["stage"])}
        if s["stage"] == "eval":
            a.update(needs_to_pass_usd=round(max(0.0, plan.target - s["bal"]), 2),
                     can_lose_usd=round(s["bal"] - s["floor"], 2))
        elif s["stage"] == "funded":
            want = plan.want[min(s["n_paid"], len(plan.want) - 1)]
            a.update(balance_usd=round(plan.size + s["bal"], 2), next_payout_usd=want,
                     needs_for_next_payout_usd=round(max(0.0, plan.keep + want - s["bal"]), 2),
                     cycle_days=s["cycle_days"])
        elif s["stage"] == "waiting":
            a["bought_when_money_made_reaches_usd"] = 2500.0 * (s["n"] - 1)
        if s.get("open"):
            a["race_open_now_usd"] = round(s["open"]["pnl"], 2)
        accounts.append(a)
    fees = sum(s["fees"] for s in sl)
    paid = sum(s["paid"] for s in sl)
    events = []
    for e in reversed(ev[-20:]):
        w = e["what"]
        events.append({"trading_day": str(e["day"]), "account": e["slot"],
                       "what": _WORDS.get(w, (f"payout {w.split('#')[-1]} taken" if w.startswith("PAYOUT") else
                                              w.replace(" -- ", ", "))),
                       "race_usd": round(e["race"], 2) if e.get("race") is not None else None,
                       "fee_or_payout_usd": round(e["usd"], 2) if e["usd"] else None})
    return {"what": "the Fleet page's Real test tab: your plan (start with 1 account, one more each time the money "
                    "made reaches another +$2,500, up to 5; $115 evaluation, $120 reset) replayed on the real NQ "
                    "prices since 1 Oct at the moments the paper bots really raced and the way they raced (an "
                    "evaluation like the trial bot, a funded race like the funded bot) -- no real money",
            "since": REAL_TEST_START, "trading_days": int(len(res["daily"])),
            "spent_usd": round(fees, 2), "paid_to_you_usd": round(0.9 * paid, 2),
            "result_usd": round(0.9 * paid - fees, 2),
            "evaluations_bought": sum(s["evals"] for s in sl), "resets": sum(s["resets"] for s in sl),
            "evaluation_races": sum(s["passes"] + s["resets"] for s in sl) + sum(s["stage"] == "eval" for s in sl),
            "evaluations_passed": sum(s["passes"] for s in sl), "funded_accounts": sum(s["passes"] for s in sl),
            "funded_blown": sum(s["blown"] for s in sl), "payouts": sum(s["payouts"] for s in sl),
            "accounts": accounts, "recent_events": events,
            "history_comparison": "102 start dates 2024-2026: the first payout typically after 10 trading days and "
                                  "5 evaluation attempts, 83% within a month, all within 3 months"}


def real_test(data: Path, wait: bool = False) -> dict | None:
    """The Real test, worked out again in a background thread every 5 minutes; None until the first is ready."""
    import time as _clock
    stale = _clock.monotonic() - _real_test["at"] > REAL_TEST_TTL_S
    if stale and not _real_test["busy"]:
        _real_test["busy"] = True

        def work() -> None:
            try:
                _real_test["data"] = _compute_real_test(data)
                _real_test["at"] = _clock.monotonic()
            except Exception as e:  # noqa: BLE001 -- keep the last good one
                if _real_test["data"] is None:
                    _real_test["data"] = {"error": f"not available right now ({e!r})"}
            finally:
                _real_test["busy"] = False

        t = threading.Thread(target=work, daemon=True)
        t.start()
        if wait:
            t.join()
    return _real_test["data"]
