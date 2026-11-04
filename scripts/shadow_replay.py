"""'As if always on': replay both bots on real minute prices for the time the laptop was off (2026-10-07).

Owner, 2026-10-07: "my laptop was off -- show me on the dashboard the results as
if it was on: the trades that would have fired and their outcomes, as if real."

From SHADOW_START (the first trading day the bots missed), every Lucid trading
day is replayed with each bot's own rules on NinjaTrader's real 1-minute NQ bars:
  * trial -- the target race: entry at 19:00 ET (first minute bar), take-profit at
    what the evaluation still needs, fail line where equity meets the MLL floor
    (open equity counts, lock at $50,100), out 16:00 ET; a failed or passed
    attempt is followed at once by a new one, which may race again the same
    day until 12:00 ET (the owner's rule) -- exactly run_lucidpro_trial.py;
  * funded -- the funded race through run_lucid_futures_funded.py's OWN
    functions (payout need, day take, fail line, end of day, payouts, blown ->
    a fresh account at the next session).
Fills are the bar opens / the exact target and line prices (the bots' history
audit uses the same rule: adverse move first inside a bar, the target needs a
one-tick trade-through). A race still running is shown open at the last price.

The result is written in the bots' own formats to data/shadow/ -- the
dashboard's "As if always on" view reads it. The live record is never touched.

Usage:  .venv/Scripts/python.exe scripts/shadow_replay.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import trade_brain  # noqa: E402  (minute-bar cache)
from metatraderbot import futures, lucid_rules, target_race  # noqa: E402

SHADOW_START = "2026-10-05"          # first trading day the bots missed (laptop off from Sun 4 Oct evening)
OUT = ROOT / "data" / "shadow"
ET = lucid_rules.ET
PV, TICK, COMM = futures.point_value("NQ"), futures.spec("NQ")["tick_size"], futures.commission_per_side("NQ")
ANCHOR, TARGET, MLL = 50_000.0, 3_000.0, 2_000.0
log = logging.getLogger("shadow_replay")


# --- time helpers ------------------------------------------------------------------

def et(d: date, h: int, m: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, m), tz=ET).tz_convert("UTC")


def race_times(day: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(first entry 19:00 ET the evening before, last entry 12:00 ET, time exit 16:00 ET) of a Lucid day."""
    d = date.fromisoformat(day)
    return et(d - timedelta(days=1), 19), et(d, 12), et(d, 16)


def trading_days(start: str, now: pd.Timestamp) -> list[str]:
    last = lucid_rules.trading_day(now.to_pydatetime())
    out, d = [], date.fromisoformat(start)
    while str(d) <= last:
        if d.weekday() < 5 and not target_race.skip_day(d):
            out.append(str(d))
        d += timedelta(days=1)
    return out


# --- one race on minute bars -------------------------------------------------------

def race(m1: pd.DataFrame, t_from: pd.Timestamp, last_entry: pd.Timestamp, exit_at: pd.Timestamp,
         now: pd.Timestamp, take_usd: float, lots: int, line_for) -> dict | None:
    """Enter at the open of the first minute bar at/after t_from (before last_entry and now);
    return the race with its first event (line / target / exit) or open."""
    later = m1[(m1.index >= t_from) & (m1.index < min(last_entry, now))]
    if later.empty:
        return None
    opened = later.index[0]
    entry = float(later["open"].iloc[0])
    tp = entry + target_race.target_points(take_usd, lots, PV, TICK, COMM)
    line = line_for(entry)
    done = m1[(m1.index >= opened) & (m1.index < now.floor("min"))]          # completed minutes only
    ev = target_race.first_event(done, opened.to_pydatetime(), line=line, tp=tp, tick=TICK,
                                 exit_utc=exit_at.to_pydatetime(), long=True)
    out = {"opened": opened, "entry": entry, "tp": tp, "line": line, "lots": lots, "event": ev}
    if ev is None:
        out["last"] = float(done["close"].iloc[-1]) if len(done) else entry
    return out


def pnl_at(px: float, entry: float, lots: int) -> float:
    return (px - entry) * lots * PV - 2 * COMM * lots


# --- the trial bot -----------------------------------------------------------------

def replay_trial(m1: pd.DataFrame, live: dict, now: pd.Timestamp, lots: int) -> tuple[dict, list[dict]]:
    st = {k: copy.deepcopy(live.get(k)) for k in ("attempts", "passes", "fails", "total_trades", "total_wins",
                                                    "total_losses", "history")}
    st["history"] = [h for h in (st["history"] or []) if h.get("resolved_ts", "") < SHADOW_START]
    trades: list[dict] = []

    def fresh(t):
        return {"opened_ts": t.isoformat(), "realized_usd": 0.0, "peak_equity": ANCHOR, "current_day": None,
                "entries_today": 0, "total_entries": 0, "wins": 0, "losses": 0, "dll_blocked_today": False,
                "day_start_realized": 0.0, "tickets": {}, "raced_day": None}

    att = fresh(pd.Timestamp(f"{SHADOW_START}T00:00:00", tz="UTC") - pd.Timedelta(days=1))

    def resolve(outcome, when, final):
        st["attempts"] += 1
        st["passes" if outcome == "pass" else "fails"] += 1
        st["total_trades"] = (st.get("total_trades") or 0) + att["total_entries"]
        st["history"].append({"outcome": outcome, "opened_ts": att["opened_ts"], "resolved_ts": when.isoformat(),
                              "days": round((when - pd.Timestamp(att["opened_ts"])).total_seconds() / 86400, 2),
                              "final_pct": round((final - ANCHOR) / ANCHOR * 100, 2),
                              "final_usd": round(final - ANCHOR, 2), "trades": att["total_entries"],
                              "wins": att["wins"], "losses": att["losses"], "replayed": True})

    for day in trading_days(SHADOW_START, now):
        t0, last_entry, exit_at = race_times(day)
        t = t0
        while t < now:
            if att["raced_day"] == day:
                break
            floor = min(att["peak_equity"] - MLL, ANCHOR + 100.0)
            room = ANCHOR + att["realized_usd"] - floor
            r = race(m1, t, last_entry, exit_at, now, TARGET - att["realized_usd"], lots,
                     lambda e: round(e - room / (lots * PV), 2))
            if r is None:
                break
            att["raced_day"] = day
            att["total_entries"] += 1
            rec = {"attempt": st["attempts"] + 1, "sym": "NQ", "direction": "long", "entry_price": r["entry"],
                   "lots": float(lots), "risk_usd": 400.0 * lots * PV, "opened_ts": r["opened"].isoformat(),
                   "tp": round(r["tp"], 2), "line": r["line"]}
            if r["event"] is None:                                           # still running now
                att["tickets"] = {"replay": {**rec, "sl": r["entry"] - 400.0, "current_price": r["last"],
                                             "floating_usd": round(pnl_at(r["last"], r["entry"], lots), 2)}}
                break
            what, px, when = r["event"]
            when = pd.Timestamp(when)
            pnl = pnl_at(px, r["entry"], lots)
            att["realized_usd"] += pnl
            att["wins" if pnl > 0 else "losses"] += 1
            trades.append({**rec, "exit_price": round(px, 2), "pnl_usd": round(pnl, 2),
                           "r_multiple": round(pnl / rec["risk_usd"], 4), "closed_ts": when.isoformat(),
                           "reason": {"line": "replay: loss limit", "target": "replay: target",
                                      "exit": "replay: 16:00 close"}[what]})
            if what in ("line", "target"):
                resolve("pass" if what == "target" else "fail", when, ANCHOR + att["realized_usd"])
                att = fresh(when)
                t = when + pd.Timedelta(minutes=1)                             # a new attempt, same day
                continue
            break
        if now >= et(date.fromisoformat(day), 17) and not att["tickets"]:      # end of day: the MLL trails
            att["peak_equity"] = max(att["peak_equity"], ANCHOR + att["realized_usd"])
            att["day_start_realized"] = att["realized_usd"]
    st["attempt"] = att
    st["race_day"] = att["raced_day"]
    st["race_attempt"] = st["attempts"] + 1
    st["current_attempt_start"] = att["opened_ts"]
    return st, trades


# --- the funded bot (its own functions) ----------------------------------------------

def funded_module(clock: dict, trades_path: Path):
    spec = importlib.util.spec_from_file_location("funded_for_replay", ROOT / "scripts" / "run_lucid_futures_funded.py")
    m = importlib.util.module_from_spec(spec)
    root_handlers = list(logging.getLogger().handlers)
    spec.loader.exec_module(m)
    for h in list(logging.getLogger().handlers):          # never write the live bot's log
        if h not in root_handlers:
            logging.getLogger().removeHandler(h)
            h.close()

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            t = clock["t"].to_pydatetime()
            return t if tz is None else t.astimezone(tz)

    m.datetime = Clock
    m.alert = lambda *a, **k: None
    m.log = log
    m.TRADES_LOG_PATH = trades_path
    return m


def _funded_account() -> str:
    """The live funded bot's NinjaTrader account (the dashboard reads prices through it)."""
    try:
        return json.loads((ROOT / "data" / "state_lucid_futures_funded.json").read_text(encoding="utf-8"))["account"]
    except (OSError, ValueError, KeyError):
        return "DEMO9384168"


def replay_funded(m1: pd.DataFrame, now: pd.Timestamp, trades_path: Path) -> dict:
    clock = {"t": pd.Timestamp(f"{SHADOW_START}T00:00:00", tz="UTC") - pd.Timedelta(days=1)}
    if trades_path.exists():
        trades_path.unlink()
    m = funded_module(clock, trades_path)
    state = {"run": m.new_run(), "runs_total": 1, "runs_paid": 0, "runs_blown": 0, "runs_done": 0,
             "payouts_total": 0, "paid_total_usd": 0.0, "history": [], "halted": False, "halt_reason": "",
             "account": _funded_account(), "live": False, "replayed": True}
    days = trading_days(SHADOW_START, now)
    for i, day in enumerate(days):
        run = state["run"]
        t0, last_entry, exit_at = race_times(day)
        clock["t"] = t0
        if run["day"] is None:
            run["day"], run["day_start_realized"] = day, run["realized_usd"]
        need = m.payout_need(run)
        if need > 0 and run["race_day"] != day:
            take = m.day_take(need)
            r = race(m1, t0, last_entry, exit_at, now, take, 1, lambda e: m.fail_line(run, e, 1, PV, "NQ"))
            if r is not None:
                run["race_day"] = day
                info = {"sym": "NQ", "direction": "long", "entry_price": r["entry"], "sl": r["entry"] - 400.0,
                        "tp": round(r["tp"], 2), "line": r["line"], "lots": 1.0, "take_usd": round(take, 2),
                        "need_usd": round(need, 2), "opened_ts": r["opened"].isoformat()}
                run["tickets"]["replay"] = info
                if r["event"] is None:
                    info["current_price"] = r["last"]
                    info["floating_usd"] = round(pnl_at(r["last"], r["entry"], 1), 2)
                    break
                what, px, when = r["event"]
                clock["t"] = pd.Timestamp(when)
                m.book(state, "replay", pnl_at(px, r["entry"], 1), px, pd.Timestamp(when).isoformat(),
                       {"line": "replay: loss limit", "target": "replay: take-profit",
                        "exit": "replay: 16:00 close"}[what])
                if what == "line":
                    m.end_run(state, "blown", f"equity reached the MLL at {pd.Timestamp(when):%H:%M} UTC", False)
                    continue                                              # the new account races next session
        eod = et(date.fromisoformat(day), 17)
        if now >= eod and not state["run"]["tickets"]:
            clock["t"] = eod
            nxt = days[i + 1] if i + 1 < len(days) else lucid_rules.trading_day((eod + pd.Timedelta(hours=1)).to_pydatetime())
            if m.end_of_day(state, nxt, False):
                m.end_run(state, "done", "five payouts taken", False)
    state["floating_usd"] = sum(t.get("floating_usd", 0.0) for t in state["run"]["tickets"].values())
    return state


def update() -> dict:
    now = pd.Timestamp(datetime.now(timezone.utc))
    m1 = trade_brain._bars_1m()
    if m1 is None or m1.index[-1] < pd.Timestamp(f"{SHADOW_START}T00:00:00", tz="UTC"):
        raise RuntimeError("no minute bars for the replay period -- is NinjaTrader running?")
    now = min(now, m1.index[-1] + pd.Timedelta(minutes=1))             # as far as the prices go
    live = json.loads((ROOT / "data" / "state_lucid_futures.json").read_text(encoding="utf-8"))
    lots = 1                                                            # as the live trial: bridge 1.1 -> 1 NQ
    try:
        from metatraderbot.nt_engine import NinjaTraderEngine
        if NinjaTraderEngine(account="Sim101", timeout_s=10.0).bridge_version() >= (1, 2):
            lots = 2
    except Exception:  # noqa: BLE001
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    tst, ttr = replay_trial(m1, live, now, lots)
    fst = replay_funded(m1, now, OUT / "funded_trades.jsonl")
    meta = {"start": SHADOW_START, "until": now.isoformat(), "updated": datetime.now(timezone.utc).isoformat(),
            "trial_contracts": lots}
    tst["replay"] = meta
    fst["replay"] = meta
    (OUT / "state_trial.json").write_text(json.dumps(tst, indent=1, default=str), encoding="utf-8")
    (OUT / "trial_trades.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ttr), encoding="utf-8")
    (OUT / "state_funded.json").write_text(json.dumps(fst, indent=1, default=str), encoding="utf-8")
    return {"trial": tst, "trial_trades": ttr, "funded": fst, "meta": meta}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = update()
    print(f"replayed {r['meta']['start']} .. {r['meta']['until'][:16]} UTC (trial {r['meta']['trial_contracts']} NQ)")
    for t in r["trial_trades"]:
        print(f"  TRIAL #{t['attempt']} {t['opened_ts'][:16]} -> {t['closed_ts'][:16]}  {t['entry_price']:,.2f} -> "
              f"{t['exit_price']:,.2f}  {t['pnl_usd']:+,.2f}  {t['reason']}")
    for k, v in (r["trial"]["attempt"].get("tickets") or {}).items():
        print(f"  TRIAL open: {v['opened_ts'][:16]} @ {v['entry_price']:,.2f}, take {v['tp']:,.2f}, line {v['line']:,.2f}, "
              f"now {v['current_price']:,.2f} ({v['floating_usd']:+,.2f})")
    print(f"  trial record: {r['trial']['attempts']} attempts, {r['trial']['passes']} passed, {r['trial']['fails']} failed")
    p = OUT / "funded_trades.jsonl"
    for line in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        t = json.loads(line)
        print(f"  FUNDED acct #{t['run']} {t['opened_ts'][:16]} -> {t['closed_ts'][:16]}  {t['entry_price']:,.2f} -> "
              f"{t['exit_price']:,.2f}  {t['pnl_usd']:+,.2f}  {t['reason']} (take ${t['take_usd']:,.0f})")
    fr = r["funded"]["run"]
    for k, v in fr["tickets"].items():
        print(f"  FUNDED open: {v['opened_ts'][:16]} @ {v['entry_price']:,.2f}, take {v['tp']:,.2f}, line {v['line']:,.2f}, "
              f"now {v.get('current_price', 0):,.2f} ({v.get('floating_usd', 0):+,.2f})")
    print(f"  funded: account #{r['funded']['runs_total']}, balance ${50000 + fr['realized_usd'] - fr['paid_usd']:,.2f}, "
          f"{len(fr['payouts'])} payouts, blown {r['funded']['runs_blown']}, cycle {fr['cycle_days']} days ${fr['cycle_pnl']:+,.2f}")
