"""Lucid FUNDED-stage bot on CME futures, through NinjaTrader 8 -- the FUNDED RACE.

What to trade AFTER passing the evaluation. Owner, 2026-10-01: "at least a
$500 withdrawal in a small amount of time -- one account first, then more".

STRATEGY (docs/LUCIDPRO_PASS_RESEARCH.md section 7b, scripts/lucid_funded_balance.py)
-------------------------------------------------------------------------------------
Once per Lucid trading day, from 19:00 ET (entries until 12:00 ET): buy 1 NQ
with a take-profit limit at what the NEXT PAYOUT still needs -- at least
$150, at most $1,500 -- and only a far safety stop; out at 16:00 ET. What
the next payout still needs is the largest of
  * the money:    the balance must reach $52,100 + the payout ($500 for the
                  first one, $2,500 for each later one),
  * the 40% rule: the cycle's best day at most 40% of the cycle's profit,
  * the minimum:  $500 of profit per payout cycle.
Once all three are met the bot PAUSES (no race) until the cycle has 3
trading days; the payout is then taken at that day's close. The 40% rule is
not a daily cap: a $1,500 day only makes the payout wait for $3,750 of cycle
profit, and that payout is bigger ($1,650 instead of $500 after three wins).
Until 2026-10-21 the cap was $1,040 (40% of the $2,600 the first payout
needs); the payout research of that day (notes/Research log) replayed the
owner's plan 12 months from every start date: $1,240-$1,750 all beat $1,040 on
real NQ 2021-25 and the index 2017-25, also with coin-flip sides and worse
fills -- $1,500, mid-plateau: typical year +$116.8k vs +$67.2k (real NQ).

Replayed from every start day 2019-26 under Lucid's confirmed rules (adverse
move first inside a bar): 36% of funded accounts take a $500+ payout within
10 trading days (median 3), 63% are blown first; ~$818 is paid per funded
account, ~$280/week per account slot after evaluation fees. A coin flip with
capped downside (the evaluation fee), not a market edge.

RULES MODELLED (LucidPro 50K funded; Lucid support 2026-09-30)
--------------------------------------------------------------
* Max Loss Limit: EQUITY, the open position included, $2,000 below the best
  end-of-day balance -- locked at $50,100 once an end-of-day balance reaches
  $52,100. Checked every poll AND against 1-minute price history, so a
  sleeping laptop cannot hide a breach Lucid's server would have seen.
* Daily Loss Limit: $1,200 of REALIZED loss -> done for the day.
* Flat by 4:45 PM ET (the race is out at 16:00 ET); no holiday sessions.
* Payouts: >= 3 trading days per cycle, >= $500 cycle profit, best day <= 40%
  of it, $52,100 stays in the account; payout 1 at most $2,000, payouts 2-5
  at most $2,500 (90% to the trader); after the fifth, Lucid's live review.

SIMULATOR vs REAL ACCOUNT
-------------------------
On a NinjaTrader SIMULATION account (NT_FUNDED_ACCOUNT, default Sim102)
payouts are VIRTUAL -- booked in the bot's state, since a simulator cannot
pay out -- and a blown account is followed at once by a fresh $50,000 one:
a forward test of the replay's numbers. On a REAL Lucid funded account (set
LUCID_FUNDED_LIVE_CONFIRM=<account>) the bot alerts when a payout is due --
request it in Lucid's dashboard -- and books it the same way; a blown account
or the fifth payout HALTS it.

The previous funded strategy (ORB + trend roster: ~24% of funded accounts
reached a payout, median ~70 days) is kept as run_lucid_futures_funded_roster.py.

Usage:
    .venv/Scripts/python.exe scripts/run_lucid_futures_funded.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from metatraderbot import futures, lucid_rules, target_race  # noqa: E402
from metatraderbot.alerts import alert  # noqa: E402
from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine  # noqa: E402

# --- which funded account this process trades ----------------------------------------
# One process per funded account (owner, 2026-10-09: "we will be trading multiple accounts").
# The first account (NT_FUNDED_ACCOUNT) keeps the original file names; every other one is started
# with --account <NinjaTrader account> and gets its own state, trade log and log file.
PRIMARY_ACCOUNT = os.environ.get("NT_FUNDED_ACCOUNT", "Sim102").strip()


def account_tag(account: str) -> str:
    """'' for the first funded account, '_<name>' for every other one (used in its file names)."""
    import re
    return "" if account == PRIMARY_ACCOUNT else "_" + re.sub(r"[^A-Za-z0-9]+", "_", account).strip("_")


ACCOUNT = (sys.argv[sys.argv.index("--account") + 1].strip()
           if "--account" in sys.argv[:-1] else PRIMARY_ACCOUNT)
TAG = account_tag(ACCOUNT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"lucid_futures_funded{TAG}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("lucid_futures_funded")

# --- account rules (LucidPro 50K funded) --------------------------------------
PLAN = lucid_rules.LUCIDPRO_50K
ANCHOR = PLAN.start_balance                    # 50,000
MLL_USD = PLAN.max_loss_limit                  # 2,000
MLL_LOCK_TRIGGER = ANCHOR + MLL_USD + 100.0    # 52,100 -- Initial Trail Balance
MLL_LOCKED_FLOOR = ANCHOR + 100.0              # 50,100
DLL_USD = PLAN.daily_loss_limit                # 1,200 of realized loss
KEEP_USD = MLL_LOCK_TRIGGER                    # what must stay in the account after a payout
PAYOUT_CAPS = (2_000.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)   # payout 1, then payouts 2-5
PAYOUT_WANT = (500.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)     # taken once this much is above the keep
MIN_CYCLE_PROFIT = 500.0
CONSISTENCY = 0.40                             # best day <= 40% of the cycle's profit
MIN_CYCLE_DAYS = 3
SPLIT = 0.90                                   # the trader's share of a payout

# --- the funded race -------------------------------------------------------------
RACE = target_race.RaceConfig(
    symbol="NQ",
    contracts=int(os.environ.get("LUCID_FUNDED_CONTRACTS", "1")),
    entry=os.environ.get("LUCID_FUNDED_ENTRY", "19:00").strip(),
)
DAY_TAKE_MAX = 1_500.0                         # owner 2026-10-21 (was $1,040): bigger payouts, same pace
DAY_TAKE_MIN = 150.0
# Which way to race: "trend200" = long above the 200-day average, short below; "long" = always long.
# --side breadth: a PAPER-ONLY funded account racing the A-D breadth side (owner 2026-10-21: compare the new
# strategy with the current one side by side -- started with its own --account, e.g. Sim102-AD).
SIDE_ARG = sys.argv[sys.argv.index("--side") + 1].strip().lower() if "--side" in sys.argv[:-1] else ""
RACE_SIDE_RULE = SIDE_ARG or os.environ.get("LUCID_RACE_SIDE", "trend200").strip().lower()
MAGIC = 9242                                   # pinned; 9230-9241 were the roster's micros

# --- plumbing ------------------------------------------------------------------
STATE_PATH = ROOT / "data" / f"state_lucid_futures_funded{TAG}.json"
TRADES_LOG_PATH = ROOT / "data" / f"lucid_futures_funded_trades{TAG}.jsonl"
BOOK = "Lucid-Funded" + (f" {ACCOUNT}" if TAG else "") + (" A-D" if SIDE_ARG == "breadth" else "")
POLL_INTERVAL_S = 15
HISTORY_AUDIT_S = 60             # how often the open race is checked against 1-minute history
MISSING_TICKET_POLLS = 20        # ~5 min before a vanished ticket is written off
BRACKET_RETRIES_PER_DAY = 3

_HEARTBEAT_TIMEOUT_S = 180
_last_heartbeat = time.monotonic()
_heartbeat_lock = threading.Lock()


def _touch_heartbeat() -> None:
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.monotonic()


def _self_watchdog() -> None:
    """Force-exit if the loop stops making progress (a hung bridge call), so
    the launcher's watchdog restarts a clean process -- as in the trial bot."""
    last_wake = time.monotonic()
    while True:
        time.sleep(30)
        now = time.monotonic()
        if now - last_wake > 90:
            # This thread itself was frozen far past its 30 s sleep: the PC
            # was asleep (Windows' monotonic clock keeps counting through
            # sleep), not the loop hung. Resetting avoids killing a healthy
            # bot on every wake-up (2026-10-01: both bots died this way).
            log.warning(f"Resumed after ~{now - last_wake:.0f}s asleep -- watchdog reset.")
            _touch_heartbeat()
        last_wake = now
        with _heartbeat_lock:
            stale_s = time.monotonic() - _last_heartbeat
        if stale_s > _HEARTBEAT_TIMEOUT_S:
            log.error(f"Self-watchdog: no loop progress for {stale_s:.0f}s -- force-exiting.")
            os._exit(1)


# --- state: one "run" = one funded account, from $50,000 until it is blown -------
# or has taken its five payouts

def new_run() -> dict:
    return {"start": datetime.now(timezone.utc).isoformat(), "realized_usd": 0.0, "paid_usd": 0.0,
            "payouts": [], "peak_eod": ANCHOR, "mll_locked": False, "breach": None,
            "day": None, "day_start_realized": 0.0, "dll_hit": False, "days": 0,
            "cycle_days": 0, "cycle_pnl": 0.0, "cycle_best_day": 0.0,
            "race_day": None, "bracket_retries": {}, "trades": 0, "wins": 0, "losses": 0, "tickets": {}}


def load_state() -> dict:
    state = {"run": new_run(), "runs_total": 1, "runs_paid": 0, "runs_blown": 0, "runs_done": 0,
             "payouts_total": 0, "paid_total_usd": 0.0, "history": [], "halted": False, "halt_reason": ""}
    if STATE_PATH.exists():
        try:
            state.update(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.error("State file unreadable -- starting a fresh funded account")
    state["run"] = {**new_run(), **(state.get("run") or {})}
    return state


def save_state(state: dict) -> None:
    # OneDrive briefly locks files it is syncing; retry instead of losing a poll.
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, STATE_PATH)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


# --- the account's numbers -------------------------------------------------------

def balance(run: dict) -> float:
    """Closed balance: the start, plus closed trades, minus payouts taken."""
    return ANCHOR + run["realized_usd"] - run["paid_usd"]


def mll_floor(run: dict) -> float:
    """$2,000 under the best end-of-day balance; locked at $50,100 once that best reached $52,100."""
    return min(run["peak_eod"] - MLL_USD, MLL_LOCKED_FLOOR)


def payout_need(run: dict) -> float:
    """Profit still needed before the next payout can be taken: the largest of
    the money above the $52,100 keep, the 40% rule and the $500 cycle minimum.
    <= 0 means met -- only the 3-day minimum can still be missing."""
    today = run["realized_usd"] - run["day_start_realized"]
    cycle, best = run["cycle_pnl"] + today, max(run["cycle_best_day"], today)
    k = min(len(run["payouts"]), len(PAYOUT_WANT) - 1)
    return max(KEEP_USD + PAYOUT_WANT[k] - balance(run),
               best / CONSISTENCY - cycle if best > 0 else 0.0,
               MIN_CYCLE_PROFIT - cycle)


def day_take(need: float) -> float:
    """The day's take-profit in $: what the payout needs, between $150 and $1,500."""
    return min(max(need, DAY_TAKE_MIN), DAY_TAKE_MAX)


def fail_line(run: dict, entry: float, lots: float, pv: float, sym: str, long: bool = True) -> float:
    """The price at which equity -- this position and its round-trip commission
    included -- reaches the MLL floor."""
    room = balance(run) - mll_floor(run) - 2 * futures.commission_per_side(sym) * lots
    return round(entry - (1.0 if long else -1.0) * room / (lots * pv), 2)


def position_pnl(info: dict, price: float, pv: float) -> float:
    sign = 1.0 if info["direction"] == "long" else -1.0
    return ((price - info["entry_price"]) * sign * info["lots"] * pv
            - 2 * futures.commission_per_side(info["sym"]) * info["lots"])


def floating_usd(engine, run: dict) -> float:
    total = 0.0
    for key, info in run["tickets"].items():
        pos = engine.get_position(int(key))
        if pos is not None:
            total += position_pnl(info, float(pos.price_current), _pv(engine, info["sym"]))
    return total


def _pv(engine, sym: str) -> float:
    return engine.usd_per_point(sym) or futures.point_value(sym)


def trading_days_between(a: str, b: str) -> int:
    """Weekdays from trade date a (included) to b (excluded) -- at least 1."""
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    return max(1, sum(1 for i in range((d1 - d0).days) if (d0 + timedelta(days=i)).weekday() < 5))


# --- booking -----------------------------------------------------------------------

def log_trade(state: dict, info: dict, exit_price: float, pnl: float, closed_ts: str, reason: str) -> None:
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"run": state["runs_total"], "sym": info["sym"], "direction": info["direction"],
              "entry_price": info["entry_price"], "exit_price": round(float(exit_price), 2),
              "lots": info["lots"], "pnl_usd": round(pnl, 2), "opened_ts": info["opened_ts"],
              "closed_ts": closed_ts, "tp": info.get("tp"), "line": info.get("line"),
              "take_usd": info.get("take_usd"), "reason": reason}
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def book(state: dict, key: str, pnl: float, exit_price: float, closed_ts: str, reason: str) -> None:
    """Credit a closed race to the funded account and log it."""
    run = state["run"]
    info = run["tickets"].pop(key)
    run["realized_usd"] += pnl
    run["trades"] += 1
    run["wins" if pnl > 0 else "losses"] += 1
    log_trade(state, info, exit_price, pnl, closed_ts, reason)
    log.info(f"[{info['sym']}] race closed ({reason}) @ {exit_price:,.2f}: ${pnl:+,.2f} -- balance "
             f"${balance(run):,.2f}, payout still needs ${max(0.0, payout_need(run)):,.0f}")
    alert(f"{'🟢' if pnl > 0 else '🔴'} <b>FUNDED race closed</b> ({reason}) {info['sym']} @ {exit_price:,.2f}\n"
          f"Profit <b>${pnl:+,.2f}</b> -- balance ${balance(run):,.0f}, "
          f"MLL ${mll_floor(run):,.0f}", book=BOOK, html=True)


def close_and_book(engine, state: dict, key: str, reason: str) -> None:
    """Close one race at market and book it from NinjaTrader's fill -- or at
    the last price if the fill is not reported yet. A position that is
    already gone (its bracket filled) or will not close is left in the
    tickets for reconcile() / the next poll."""
    info = state["run"]["tickets"][key]
    _touch_heartbeat()
    pos = engine.get_position(int(key))
    if pos is None:
        return
    last = float(pos.price_current)
    engine.close_position(int(key))
    _touch_heartbeat()
    if engine.get_position(int(key)) is not None:
        log.warning(f"[{info['sym']}] ticket {key} did not close -- trying again next poll")
        return
    deal = engine.get_closed_deal(int(key))
    if deal is not None:
        book(state, key, float(deal["pnl_usd"]), float(deal["exit_price"]), deal["closed_ts"].isoformat(), reason)
    else:
        book(state, key, position_pnl(info, last, _pv(engine, info["sym"])), last,
             datetime.now(timezone.utc).isoformat(), reason)


def close_all(engine, state: dict, code: str, why: str) -> None:
    """Close every open race; `code` is the trade log's reason (mll / dll / time_exit)."""
    run = state["run"]
    if run["tickets"]:
        log.warning(f"closing {len(run['tickets'])} position(s): {why}")
    for key in list(run["tickets"]):
        close_and_book(engine, state, key, code)


# --- what Lucid's server saw ---------------------------------------------------------

def race_event(engine, run: dict, key: str, until_utc: datetime | None = None):
    """First line / target / time-exit event in the 1-minute history since the
    race opened (target_race.first_event), optionally only up to until_utc.
    None if nothing happened, or no bars."""
    import pandas as pd
    info = run["tickets"][key]
    sym, long_ = info["sym"], info["direction"] == "long"
    opened = datetime.fromisoformat(info["opened_ts"])
    now = datetime.now(timezone.utc)
    minutes = int((now - opened).total_seconds() // 60) + 3
    bars = engine.get_bars(sym, "1m", max(5, min(minutes, 3000))).iloc[:-1]     # completed bars only
    if until_utc is not None:
        bars = bars[bars.index < pd.Timestamp(until_utc).floor("min")]
    line = fail_line(run, info["entry_price"], info["lots"], _pv(engine, sym), sym, long_)
    return target_race.first_event(bars, opened, line=line, tp=float(info["tp"]),
                                   tick=futures.spec(sym)["tick_size"],
                                   exit_utc=target_race.exit_time(opened, RACE), long=long_)


def settle_event(engine, state: dict, key: str, event) -> None:
    """Book an event the history found at the price Lucid would have used; close
    the simulator's position at today's price (the funded account is what counts)."""
    what, px, when = event
    run = state["run"]
    info = run["tickets"][key]
    engine.close_position(int(key))
    pnl = position_pnl(info, px, _pv(engine, info["sym"]))
    reason = {"line": "mll_while_away", "target": "take_while_away", "exit": "exit_while_away"}[what]
    log.warning(f"[{info['sym']}] price history: {what} at {when:%Y-%m-%d %H:%M} UTC while the bot was not "
                f"watching -- booked at {px:,.2f}, ${pnl:+,.2f}")
    book(state, key, pnl, px, when.isoformat(), reason)


def reconcile(engine, state: dict) -> str | None:
    """Credit every race NinjaTrader closed itself (take-profit, safety stop,
    outside close). A take-profit fill is checked against the 1-minute history
    first: on waking from sleep the simulator can fill the limit although the
    fail line was touched earlier -- Lucid would have closed the account then.
    Returns "line" if such a rescore ended the account."""
    run = state["run"]
    for key, info in list(run["tickets"].items()):
        ticket = int(key)
        if engine.get_position(ticket) is not None:
            info["missing"] = 0
            continue
        deal = engine.get_closed_deal(ticket)
        if deal is None:
            # NinjaTrader no longer knows the position (restarted while the PC
            # was off): judge it from the price history, as Lucid would have.
            history_ok = True
            try:
                ev = race_event(engine, run, key)
            except Exception as e:  # noqa: BLE001 -- no history yet: count it as missing
                log.debug(f"history settlement of a vanished ticket skipped: {e!r}")
                ev, history_ok = None, False
            if ev is not None:
                settle_event(engine, state, key, ev)
                if ev[0] == "line":
                    return "line"
                continue
            if history_ok:
                # Still running on the price history: NinjaTrader lost the position (2026-10-09 the trial
                # dropped such a race and missed its pass). Keep it; the history audit settles it.
                if not info.get("on_history"):
                    info["on_history"] = True
                    log.warning(f"[{info['sym']}] ticket {ticket}: NinjaTrader no longer shows the position -- "
                                f"the race goes on, judged on the price history")
                info["missing"] = 0
                continue
            info["missing"] = int(info.get("missing", 0)) + 1
            if info["missing"] >= MISSING_TICKET_POLLS:
                log.warning(f"[{info['sym']}] ticket {ticket} vanished without a closed-deal record "
                            f"-- written off without P&L (NinjaTrader restarted?)")
                del run["tickets"][key]
            continue
        if deal.get("exit_reason") in ("tp", "sl"):
            # A bracket fill can come while this bot is not watching -- on waking,
            # or at any hour on an account whose orders NinjaTrader holds on its
            # servers (the DEMO account). Lucid judges what happened FIRST.
            try:
                ev = race_event(engine, run, key, until_utc=deal["closed_ts"])
            except Exception as e:  # noqa: BLE001 -- no history: trust the fill
                log.debug(f"bracket-fill history check skipped: {e!r}")
                ev = None
            if ev is not None and ev[0] in ("line", "exit"):
                info = run["tickets"][key]
                log.warning(f"[{info['sym']}] {deal['exit_reason']} filled at {deal['exit_price']:,.2f}, but "
                            f"{'the fail line' if ev[0] == 'line' else 'the 16:00 ET exit'} came first at "
                            f"{ev[2]:%H:%M} UTC -- booked at {ev[1]:,.2f}")
                book(state, key, position_pnl(info, ev[1], _pv(engine, info["sym"])), ev[1],
                     ev[2].isoformat(), "mll_rescored" if ev[0] == "line" else "exit_while_away")
                if ev[0] == "line":
                    return "line"
                continue
        if deal.get("close_reason") in ("protection_lost", "gap"):
            # The bridge closed the race at once: its stop/target could not be
            # placed -- an order-handling failure, not a race result. Race again.
            day = lucid_rules.trading_day(datetime.now(timezone.utc))
            tries = run["bracket_retries"] if run["bracket_retries"].get("day") == day else {"day": day, "n": 0}
            if tries["n"] < BRACKET_RETRIES_PER_DAY:
                tries["n"] += 1
                run["race_day"] = None
                log.warning(f"race closed at entry ({deal.get('close_reason')}) -- racing again "
                            f"(retry {tries['n']}/{BRACKET_RETRIES_PER_DAY} today)")
            run["bracket_retries"] = tries
        book(state, key, float(deal["pnl_usd"]), float(deal["exit_price"]), deal["closed_ts"].isoformat(),
             deal.get("exit_reason") or "closed")
    return None


# --- the end of a trading day, payouts, the end of a funded account ----------------------

def take_payout_if_due(state: dict, live: bool) -> bool:
    """Lucid's payout rules at the day's close. Returns True after the fifth payout."""
    run = state["run"]
    k = min(len(run["payouts"]), len(PAYOUT_CAPS) - 1)
    avail = balance(run) - KEEP_USD
    if not (run["cycle_days"] >= MIN_CYCLE_DAYS and run["cycle_pnl"] >= MIN_CYCLE_PROFIT
            and run["cycle_best_day"] <= CONSISTENCY * run["cycle_pnl"] and avail >= PAYOUT_WANT[k]):
        return False
    amount = round(min(PAYOUT_CAPS[k], avail), 2)
    n = len(run["payouts"]) + 1
    run["payouts"].append({"n": n, "amount_usd": amount, "to_you_usd": round(SPLIT * amount, 2),
                           "day": run["day"], "ts": datetime.now(timezone.utc).isoformat(),
                           "cycle_days": run["cycle_days"], "cycle_pnl": round(run["cycle_pnl"], 2),
                           "best_day": round(run["cycle_best_day"], 2)})
    run["paid_usd"] += amount
    run.update(cycle_days=0, cycle_pnl=0.0, cycle_best_day=0.0)
    if n == 1:
        state["runs_paid"] += 1
    state["payouts_total"] += 1
    state["paid_total_usd"] = round(state["paid_total_usd"] + amount, 2)
    log.warning(f"PAYOUT #{n} of funded account #{state['runs_total']}: ${amount:,.2f} "
                f"(${SPLIT * amount:,.2f} to you) -- balance now ${balance(run):,.2f}")
    how = (f"Request <b>${amount:,.2f}</b> in Lucid's dashboard now." if live
           else "Simulator: booked as a virtual payout.")
    alert(f"💰 <b>PAYOUT #{n}</b> -- ${amount:,.2f} (${SPLIT * amount:,.2f} to you)\n{how}\n"
          f"Balance after it ${balance(run):,.0f}", book=BOOK, html=True)
    return n >= len(PAYOUT_CAPS)


def end_of_day(state: dict, new_day: str, live: bool) -> bool:
    """Close the books on run["day"] (the bot is flat): the end-of-day balance
    moves the MLL, the day joins the payout cycle, a due payout is taken.
    Returns True after the fifth payout."""
    run = state["run"]
    bal = balance(run)
    day_pnl = run["realized_usd"] - run["day_start_realized"]
    run["peak_eod"] = max(run["peak_eod"], bal)
    if not run["mll_locked"] and bal >= MLL_LOCK_TRIGGER:
        run["mll_locked"] = True
        log.warning(f"MLL LOCKED at ${MLL_LOCKED_FLOOR:,.0f} (end-of-day balance ${bal:,.2f})")
    n = trading_days_between(run["day"], new_day)
    run["days"] += n
    run["cycle_days"] += n
    run["cycle_pnl"] += day_pnl
    run["cycle_best_day"] = max(run["cycle_best_day"], day_pnl)
    log.info(f"end of {run['day']}: day ${day_pnl:+,.2f}, balance ${bal:,.2f}, cycle {run['cycle_days']} days "
             f"${run['cycle_pnl']:+,.2f} (best day ${run['cycle_best_day']:,.2f}), MLL ${mll_floor(run):,.2f}")
    done = take_payout_if_due(state, live)
    run.update(day=new_day, day_start_realized=run["realized_usd"], dll_hit=False)
    return done


def end_run(state: dict, outcome: str, reason: str, live: bool) -> None:
    """outcome: 'blown' (the MLL) or 'done' (five payouts -> Lucid's live review).
    The simulator starts a fresh $50,000 funded account; a real one halts."""
    run = state["run"]
    state["history"].append({"run": state["runs_total"], "outcome": outcome, "reason": reason,
                             "start": run["start"], "end": datetime.now(timezone.utc).isoformat(),
                             "days": run["days"] + 1, "payouts": len(run["payouts"]),
                             "paid_usd": round(run["paid_usd"], 2), "final_balance": round(balance(run), 2),
                             "trades": run["trades"], "payout_list": run["payouts"]})
    state["runs_blown" if outcome == "blown" else "runs_done"] += 1
    paid = run["paid_usd"]
    log.warning(f"FUNDED ACCOUNT #{state['runs_total']} {outcome.upper()} ({reason}): {len(run['payouts'])} "
                f"payout(s), ${paid:,.2f} paid. Overall {state['runs_paid']}/{state['runs_total']} accounts paid, "
                f"${state['paid_total_usd']:,.2f} in payouts.")
    alert(f"{'💥' if outcome == 'blown' else '🏁'} <b>FUNDED ACCOUNT #{state['runs_total']} {outcome.upper()}</b> "
          f"-- {reason}\n{len(run['payouts'])} payout(s), ${paid:,.0f} paid\n"
          f"Overall: {state['runs_paid']}/{state['runs_total']} accounts paid, ${state['paid_total_usd']:,.0f}",
          book=BOOK, html=True)
    if live:
        state.update(halted=True, halt_reason=f"funded account {outcome}: {reason}")
        alert("⏹️ <b>Funded bot halted</b> -- " + ("the account is blown." if outcome == "blown" else
              "five payouts taken; Lucid's live review is next."), book=BOOK, html=True)
    else:
        # The next simulated account starts with the NEXT session's race, as a
        # newly bought one would -- not the same day, into the move that just
        # ended the last one.
        state["run"] = new_run()
        state["run"]["race_day"] = lucid_rules.trading_day(datetime.now(timezone.utc))
        state["runs_total"] += 1


# --- the race -----------------------------------------------------------------------

def open_race(engine, state: dict, day: str, need: float) -> bool:
    """The day's race: RACE.contracts NQ long or short (the trend rule), take-profit at the $ the next
    payout still needs (day_take), far safety stop. Returns True once open."""
    run = state["run"]
    sym = RACE.symbol
    if not engine.is_tradeable(sym):
        target_race.note_frozen_price(engine, sym, log, lambda m: alert(m, book=BOOK, html=True))
        return False
    try:                        # never on a price the market's own minute bars contradict (2026-10-15)
        problem = target_race.price_problem(engine, sym)
    except Exception:  # noqa: BLE001 -- no bars to compare with: the quote's own freshness check stands
        problem = None
    if problem:
        target_race.note_frozen_price(engine, sym, log, lambda m: alert(m, book=BOOK, html=True), problem)
        return False
    q = engine.quote(sym) or {}
    side, why_side = target_race.race_side(engine, sym, RACE_SIDE_RULE)
    direction = "long" if side > 0 else "short"
    ask = float(q.get("ask" if side > 0 else "bid") or 0.0)            # the price this side pays
    pv = engine.usd_per_point(sym)
    if ask <= 0 or not pv:
        log.warning(f"[{sym}] race entry skipped -- no live quote or point value")
        return False
    contracts = RACE.contracts
    if contracts > 1 and hasattr(engine, "bridge_version") and engine.bridge_version() < (1, 2):
        # VertexBridge 1.1 could lose the take-profit when a multi-contract
        # entry filled in parts; 1 contract cannot fill in parts.
        log.warning(f"[{sym}] VertexBridge older than 1.2 -- racing 1 contract instead of {contracts}")
        contracts = 1
    take = day_take(need)
    tp_pts = target_race.target_points(take, contracts, pv, futures.spec(sym)["tick_size"],
                                       futures.commission_per_side(sym))
    result = engine.place_market_order(symbol=sym, direction=direction, lots=contracts, sl=ask - side * RACE.stop_points,
                                       tp=ask + side * tp_pts, magic=MAGIC, comment="lucid_funded_race",
                                       signal_entry_price=ask)
    if not result.ok:
        log.warning(f"[{sym}] race order failed retcode={result.retcode} {result.comment}")
        return False
    line = fail_line(run, result.price, result.volume, pv, sym, long=side > 0)
    run["tickets"][str(result.ticket)] = {
        "sym": sym, "direction": direction, "side_reason": why_side, "entry_price": result.price, "sl": result.sl, "tp": result.tp,
        "line": line, "lots": result.volume, "take_usd": round(take, 2), "need_usd": round(need, 2),
        "opened_ts": datetime.now(timezone.utc).isoformat()}
    run["race_day"] = day
    log.info(f"[{sym}] FUNDED RACE opened {direction.upper()} ticket={result.ticket} {result.volume:g} @ {result.price:,.2f} "
             f"take {result.tp:,.2f} ({tp_pts:g} pts = ${take:,.0f}; payout needs ${need:,.0f}) "
             f"fail line {line:,.2f}, safety stop {result.sl:,.2f} -- {why_side}")
    alert(f"🏁 <b>FUNDED race</b> {sym} {direction} x{result.volume:g} @ {result.price:,.2f}\n"
          f"Take {result.tp:,.2f} (+${take:,.0f}; the payout still needs ${need:,.0f})\n"
          f"Account ends {'below' if side > 0 else 'above'} {line:,.2f}; out {RACE.exit} ET", book=BOOK, html=True)
    return True


# --- main loop -----------------------------------------------------------------

def live_confirmed(login: str) -> bool:
    """A REAL account trades only when its name is in LUCID_FUNDED_LIVE_CONFIRM -- one name, or several
    comma-separated (owner, 2026-10-09: up to 5 accounts at a time)."""
    return login in {x.strip() for x in os.environ.get("LUCID_FUNDED_LIVE_CONFIRM", "").split(",") if x.strip()}


def main() -> int:
    account = ACCOUNT
    if SIDE_ARG and os.environ.get("VERTEX_PAPER", "").strip() != "1":
        log.error(f"--side {SIDE_ARG} runs on paper only (VERTEX_PAPER=1) -- not starting")
        return 2
    if os.environ.get("VERTEX_PAPER", "").strip() == "1":
        # paper trading on NinjaTrader's free minute bars, fills simulated (owner 2026-10-15) -- paper_engine.py
        from metatraderbot.paper_engine import PaperEngine
        engine = PaperEngine(account=account, max_contract_units=PLAN.max_units)
    else:
        engine = NinjaTraderEngine(account=account, max_contract_units=PLAN.max_units)
    while True:
        try:
            acc = engine.connect()
            break
        except BridgeError as e:
            # Most often: the account does not exist yet (create Sim102 in
            # NinjaTrader) or NinjaTrader is not running. Wait, don't crash-loop.
            log.error(f"cannot use NinjaTrader account {account!r}: {e} -- retrying in 60s")
            time.sleep(60)
    live = not acc.is_demo
    if live and not live_confirmed(acc.login):
        raise RuntimeError(f"{acc.login} is not a simulation account. To trade it, add it to "
                           f"LUCID_FUNDED_LIVE_CONFIRM in .env (several accounts: comma-separated).")
    state = load_state()
    state.update(account=acc.login, live=live, strategy={
        "name": "funded race", "symbol": RACE.symbol, "contracts": RACE.contracts, "entry": RACE.entry,
        "last_entry": RACE.last_entry, "exit": RACE.exit, "stop_points": RACE.stop_points,
        "day_take_min": DAY_TAKE_MIN, "day_take_max": DAY_TAKE_MAX, "payout_want": list(PAYOUT_WANT)})
    state.pop("roster", None)
    save_state(state)
    run = state["run"]
    log.warning(f"Lucid funded bot on {'LIVE' if live else 'SIM'} account {acc.login}: FUNDED RACE -- "
                f"{RACE.contracts} {RACE.symbol} once a day ({'long above / short below the 200-day average' if RACE_SIDE_RULE == 'trend200' else 'always long'}), entries {RACE.entry}-{RACE.last_entry} ET, "
                f"out {RACE.exit} ET, take = what the next payout needs (${DAY_TAKE_MIN:,.0f}-${DAY_TAKE_MAX:,.0f}). "
                f"Funded account #{state['runs_total']}: balance ${balance(run):,.2f}, "
                f"{len(run['payouts'])} payout(s), {len(run['tickets'])} open.")
    if state["halted"]:
        log.warning(f"HALTED: {state['halt_reason']} -- open positions are still managed.")

    retry_at = 0.0                   # monotonic time before which a failed entry is not retried
    next_untracked = 0.0             # next check for a position no race holds
    next_audit = datetime.min.replace(tzinfo=timezone.utc)
    paused_noted = None
    try:                             # positions left open by an earlier run whose race is already booked
        state["leftovers"] = target_race.close_leftovers(engine, run["tickets"], state.get("leftovers") or [], log)
    except Exception as e:  # noqa: BLE001 -- NinjaTrader not ready: checked again before the next race
        log.warning(f"leftover check skipped: {e!r}")
    _touch_heartbeat()
    threading.Thread(target=_self_watchdog, daemon=True, name="self-watchdog").start()

    while True:
        try:
            _touch_heartbeat()
            if reconcile(engine, state) == "line":
                end_run(state, "blown", "the fail line was touched before the take-profit filled", live)
                save_state(state)
                time.sleep(POLL_INTERVAL_S)
                continue
            run = state["run"]
            now = datetime.now(timezone.utc)
            day = lucid_rules.trading_day(now)
            if run["day"] is None:
                run["day"] = day
                run["day_start_realized"] = run["realized_usd"]

            # ---- what Lucid's server saw while this bot was not looking ----
            if run["tickets"] and now >= next_audit:
                next_audit = now + timedelta(seconds=HISTORY_AUDIT_S)
                for key in list(run["tickets"]):
                    try:
                        event = race_event(engine, run, key)
                    except Exception as e:  # noqa: BLE001 -- no bars yet / bridge hiccup: next time
                        log.debug(f"history audit skipped: {e!r}")
                        event = None
                    if event is not None:
                        settle_event(engine, state, key, event)
                        if event[0] == "line":
                            end_run(state, "blown", f"equity reached the MLL at {event[2]:%H:%M} UTC", live)
                            break
                if state["run"] is not run or not run["tickets"]:
                    save_state(state)
                    time.sleep(POLL_INTERVAL_S)
                    continue

            # ---- still open from an EARLIER trading day: the flat window was missed ----
            for key, info in list(run["tickets"].items()):
                if lucid_rules.trading_day(datetime.fromisoformat(info["opened_ts"])) != day:
                    close_and_book(engine, state, key, "missed_flat_window")

            # ---- end of a trading day, once flat (the day's P&L is complete) ----
            if day != run["day"] and not run["tickets"]:
                if end_of_day(state, day, live):
                    end_run(state, "done", "five payouts taken -- Lucid's live review", live)
                    save_state(state)
                    time.sleep(POLL_INTERVAL_S)
                    continue

            floating = floating_usd(engine, run)
            equity = balance(run) + floating
            floor = mll_floor(run)
            state["floating_usd"] = round(floating, 2)

            # ---- the Max Loss Limit: equity, open position included ----
            # A breach is final the moment it happens (Lucid support) -- even
            # if closing the position takes another poll and the price recovers.
            if not state["halted"] and (run["breach"] or equity <= floor):
                if not run["breach"]:
                    run["breach"] = f"equity ${equity:,.0f} reached the MLL ${floor:,.0f}"
                close_all(engine, state, "mll", f"MLL: {run['breach']}")
                if not run["tickets"]:
                    end_run(state, "blown", run["breach"], live)
                save_state(state)
                time.sleep(POLL_INTERVAL_S)
                continue

            # ---- the Daily Loss Limit: realized loss today ----
            day_realized = run["realized_usd"] - run["day_start_realized"]
            if not run["dll_hit"] and day_realized <= -DLL_USD:
                run["dll_hit"] = True
                close_all(engine, state, "dll", "daily loss limit")
                alert(f"🛑 <b>Daily loss limit</b> -- ${day_realized:,.0f} realized today; done for the day.",
                      book=BOOK, html=True)

            # ---- out at 16:00 ET; Lucid's own flat rule at 16:45 ----
            session = lucid_rules.session_state(now)
            if run["tickets"] and (RACE.exit_due(now) or session == lucid_rules.Session.FLATTEN):
                close_all(engine, state, "time_exit", f"{RACE.exit} ET time exit")

            if not run["tickets"] and time.monotonic() >= next_untracked:
                # a position no race holds is closed at once, not only before the next race (2026-10-14)
                next_untracked = time.monotonic() + target_race.LEFTOVER_RETRY_S
                try:
                    target_race.close_untracked(engine, log, lambda m: alert(m, book=BOOK, html=True))
                except Exception as e:  # noqa: BLE001 -- NinjaTrader busy: next time
                    log.warning(f"untracked-position check skipped: {e!r}")

            # ---- the day's race ----
            need = payout_need(run)
            state["need_usd"] = round(need, 2)
            if (session == lucid_rules.Session.OPEN and not state["halted"] and not run["dll_hit"]
                    and not run["tickets"] and run["race_day"] != day and RACE.entry_window(now)
                    and not target_race.skip_day(day) and time.monotonic() >= retry_at):
                if need <= 0:
                    if paused_noted != (state["runs_total"], day):
                        paused_noted = (state["runs_total"], day)
                        log.info(f"payout money reached -- no race today; cycle day {run['cycle_days'] + 1} "
                                 f"of {MIN_CYCLE_DAYS}, the payout is taken at a close once the cycle has "
                                 f"{MIN_CYCLE_DAYS} days")
                else:
                    # an old position still open in NinjaTrader would stack under the new race: close it first
                    state["leftovers"] = target_race.close_leftovers(engine, run["tickets"],
                                                                     state.get("leftovers") or [], log)
                    held = {} if state["leftovers"] else target_race.close_untracked(
                        engine, log, lambda m: alert(m, book=BOOK, html=True))
                    if state["leftovers"] or held:
                        retry_at = time.monotonic() + target_race.LEFTOVER_RETRY_S
                    elif not open_race(engine, state, day, need):
                        retry_at = time.monotonic() + 120
                _touch_heartbeat()

            save_state(state)
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.warning("Stopped by user.")
            save_state(state)
            return 0
        except Exception as e:  # noqa: BLE001 -- the loop must survive a bad poll
            log.error(f"Loop error (continuing): {e!r}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
