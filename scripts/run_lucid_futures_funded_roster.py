"""SUPERSEDED 2026-10-01 by the funded race (scripts/run_lucid_futures_funded.py):
this ORB + trend roster reached a payout in ~24% of funded accounts with a
median of ~70 days (docs/LUCIDPRO_PASS_RESEARCH.md section 7). Kept runnable,
with its own state, trade log and log file, in case it is wanted again.

Lucid FUNDED-stage bot on CME futures, through NinjaTrader 8.

The futures counterpart of run_lucidpro_funded.py (MT5). Same purpose -- what
to trade AFTER passing the evaluation: steady, survivable income toward
payouts, not the trial bot's variance race -- but a new roster, because the
MT5 one cannot run on Lucid: its legs hold 1d/4h trades for days and trade
Bitcoin, and Lucid requires every position flat by 4:45 PM ET and does not
list Bitcoin.

ROSTER
------
Read from data/futures/funded_roster.json, written by
scripts/futures_roster_search.py: every lab strategy config screened on
~15 months of real futures history under Lucid's rules (flat by the cutoff,
Lucid's commissions, one tick of slippage), keeping only configs positive in
both chronological halves AND with costs doubled. Re-run the search to
refresh it; the bot re-reads the file at start-up. One leg per contract.

RULES MODELLED (LucidPro 50K funded)
------------------------------------
* Max Loss Limit: $2,000 below the highest END-OF-DAY balance, checked every
  poll (stricter than end-of-day marking, like the trial bot since
  2026-08-29). It stops trailing and LOCKS at $50,100 once an end-of-day
  balance reaches $52,100 -- the Initial Trail Balance. That lock comes from
  two independent secondary sources (see run_lucidpro_funded.py's history),
  not from Lucid's own terms: verify before trusting it with a real account.
* Daily Loss Limit: $1,200, soft -- new entries stop at 75% of it, and at the
  full limit every position is closed and the day is over. It does not end
  the account.
* Flat by 4:45 PM ET (earlier on holidays): no entries in the last 15 minutes,
  everything closed 5 minutes before (lucid_rules).
* 4 minis / 40 micros in total, enforced here and inside the engine.

RISK
----
A roster leg with "contracts" trades that FIXED size (the 2026-09-30 roster:
2 MNQ + 1 MGC -- payout 40% vs blow 43% over 2016-26, 53% vs 45% over 2021-26;
4+2 blew 55%) -- scripts/lucid_policy_mc.py and lucid_riskbased_mc.py found
fixed contracts beat both a taper and fixed-$-risk sizing for this stack --
shrunk only when a full stop-out would not fit in 90% of the room above the
MLL floor. A leg without "contracts" falls back to BASE_RISK_PCT of the
$50,000 anchor (a full stop-out, commission included), tapered down as equity
nears the MLL floor -- the MT5 funded bot's "loss_only" taper:

    frac = clamp((equity - mll_floor) / MLL_USD, TAPER_FLOOR, 1.0)

RUNS
----
On a NinjaTrader SIMULATION account the bot measures survival the way the MT5
funded bot does: a run ends as "payout" once its closed balance reaches the
payout level, or "blown" at the MLL, and a fresh $50,000 run starts. On a REAL
Lucid funded account a blown MLL is a real loss of the account, so the bot
closes everything and HALTS; reaching the payout level only alerts (request
the payout yourself -- trading continues).

Usage:
    .venv/Scripts/python.exe scripts/run_lucid_futures_funded_roster.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from metatraderbot import futures, lucid_rules  # noqa: E402
from metatraderbot.alerts import alert  # noqa: E402
from metatraderbot.lab_registry import instantiate  # noqa: E402
from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "lucid_futures_funded_roster.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("lucid_futures_funded_roster")

# --- account rules (LucidPro 50K funded) --------------------------------------
PLAN = lucid_rules.LUCIDPRO_50K
ANCHOR = PLAN.start_balance                    # 50,000
MLL_USD = PLAN.max_loss_limit                  # 2,000
MLL_LOCK_TRIGGER = ANCHOR + MLL_USD + 100.0    # 52,100 -- Initial Trail Balance
MLL_LOCKED_FLOOR = ANCHOR + 100.0              # 50,100
DLL_USD = PLAN.daily_loss_limit                # 1,200, soft
DLL_ENTRY_BLOCK_FRAC = 0.75                    # stop opening at 75% of the DLL
# Lucid pays out only what sits above the $52,100 buffer; $2,600 above the
# anchor leaves a $500 withdrawal -- the level a run counts as "paid".
PAYOUT_TARGET_USD = 2_600.0

# --- risk ----------------------------------------------------------------------
BASE_RISK_PCT = 0.005                          # $250 per full stop-out
TAPER_FLOOR = 0.25

# --- plumbing ------------------------------------------------------------------
ROSTER_PATH = ROOT / "data" / "futures" / "funded_roster.json"
STATE_PATH = ROOT / "data" / "state_lucid_futures_funded_roster.json"
TRADES_LOG_PATH = ROOT / "data" / "lucid_futures_funded_roster_trades.jsonl"
BOOK = "Lucid-Funded"
POLL_INTERVAL_S = 15
BAR_COUNT = 300
MISSING_TICKET_POLLS = 20        # ~5 min before a vanished ticket is written off
# Pinned per contract (one leg per contract), never derived from roster order:
# a roster change must not hand one contract's history to another.
MAGICS = {"MES": 9230, "MNQ": 9231, "MYM": 9232, "M2K": 9233, "MGC": 9234, "MCL": 9235,
          "6E": 9236, "6B": 9237, "6J": 9238, "6A": 9239, "6C": 9240, "6S": 9241}

_HEARTBEAT_TIMEOUT_S = 180
_last_heartbeat = time.monotonic()
_heartbeat_lock = threading.Lock()


def _touch_heartbeat() -> None:
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.monotonic()


def _self_watchdog() -> None:
    """Force-exit if the loop stops making progress (a hung bridge call), so
    the launcher's watchdog restarts a clean process -- as in the MT5 bots."""
    while True:
        time.sleep(30)
        with _heartbeat_lock:
            stale_s = time.monotonic() - _last_heartbeat
        if stale_s > _HEARTBEAT_TIMEOUT_S:
            log.error(f"Self-watchdog: no loop progress for {stale_s:.0f}s -- force-exiting.")
            os._exit(1)


# --- state ---------------------------------------------------------------------

def new_run() -> dict:
    return {"run_start": datetime.now(timezone.utc).isoformat(), "realized_usd": 0.0,
            "peak_eod": ANCHOR, "mll_locked": False, "day": None, "day_start_equity": ANCHOR,
            "dll_hit": False, "entries_today": 0, "trades": 0, "wins": 0, "losses": 0,
            "payout_alerted": False, "tickets": {}}


def load_state() -> dict:
    state = {"run": new_run(), "runs_total": 1, "runs_paid": 0, "runs_blown": 0,
             "runs": [], "halted": False, "halt_reason": ""}
    if STATE_PATH.exists():
        try:
            state.update(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.error("State file unreadable -- starting a fresh run")
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


def log_trade(record: dict) -> None:
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def mll_floor(run: dict) -> float:
    return MLL_LOCKED_FLOOR if run["mll_locked"] else run["peak_eod"] - MLL_USD


def risk_budget(equity: float, run: dict) -> float:
    frac = (equity - mll_floor(run)) / MLL_USD
    return BASE_RISK_PCT * ANCHOR * max(TAPER_FLOOR, min(1.0, frac))


# --- roster --------------------------------------------------------------------

def load_roster() -> list[dict]:
    if not ROSTER_PATH.exists():
        log.error(f"No roster at {ROSTER_PATH} -- run scripts/futures_roster_search.py first.")
        return []
    legs = json.loads(ROSTER_PATH.read_text(encoding="utf-8")).get("legs") or []
    out = []
    for leg in legs:
        sym = leg["symbol"]
        if sym not in MAGICS:
            log.error(f"roster leg on {sym} has no pinned magic -- skipped")
            continue
        leg = dict(leg)
        leg["label"] = f"{leg['family']}_{sym}_{leg['tf']}_{leg['tp_r']:g}R"
        leg["magic"] = MAGICS[sym]
        leg["strategy"] = instantiate(leg["family"], {leg["family"]: leg["params"]}, leg["tp_r"],
                                      sym, risk_usd=BASE_RISK_PCT * ANCHOR)
        out.append(leg)
    return out


# --- trading helpers -----------------------------------------------------------

def floating_usd(engine: NinjaTraderEngine, run: dict) -> float:
    total = 0.0
    for t in run["tickets"]:
        pos = engine.get_position(int(t))
        if pos is not None:
            total += pos.profit - 2 * futures.commission_per_side(pos.symbol) * pos.volume
    return total


def reconcile(engine: NinjaTraderEngine, state: dict) -> None:
    """Credit every ticket that has closed (stop, target, flatten, manual)."""
    run = state["run"]
    for key, info in list(run["tickets"].items()):
        ticket = int(key)
        if engine.get_position(ticket) is not None:
            info["missing"] = 0
            continue
        deal = engine.get_closed_deal(ticket)
        if deal is None:
            info["missing"] = int(info.get("missing", 0)) + 1
            if info["missing"] >= MISSING_TICKET_POLLS:
                log.warning(f"[{info['sym']}] ticket {ticket} vanished without a closed-deal record "
                            f"-- written off without P&L (NinjaTrader restarted?)")
                del run["tickets"][key]
            continue
        del run["tickets"][key]
        pnl = float(deal["pnl_usd"])
        run["realized_usd"] += pnl
        run["trades"] += 1
        run["wins" if pnl > 0 else "losses"] += 1
        r_mult = pnl / info["risk_usd"] if info.get("risk_usd") else 0.0
        log_trade({"run": state["runs_total"], "label": info["label"], "sym": info["sym"],
                   "direction": info["direction"], "entry_price": info["entry_price"],
                   "exit_price": deal["exit_price"], "lots": info["lots"], "risk_usd": info["risk_usd"],
                   "r_multiple": round(r_mult, 4), "pnl_usd": round(pnl, 2),
                   "exit_reason": deal["exit_reason"], "opened_ts": info["opened_ts"],
                   "closed_ts": deal["closed_ts"].isoformat()})
        eq = ANCHOR + run["realized_usd"]
        alert(f"{'🟢' if pnl > 0 else '🔴'} <b>CLOSED</b> {deal['exit_reason'].upper()} {info['sym']} "
              f"{info['direction']}\nProfit: <b>${pnl:+,.2f}</b> [{r_mult:+.2f}R]\n"
              f"Balance ${eq:,.0f} ({(eq - ANCHOR) / ANCHOR:+.2%}), MLL floor ${mll_floor(run):,.0f}",
              book=BOOK, html=True)


def close_all(engine: NinjaTraderEngine, run: dict, why: str) -> None:
    if run["tickets"]:
        log.warning(f"closing {len(run['tickets'])} position(s): {why}")
    for key in list(run["tickets"]):
        engine.close_position(int(key))
        _touch_heartbeat()


def end_run(state: dict, outcome: str, reason: str, equity: float) -> None:
    run = state["run"]
    days = (datetime.now(timezone.utc) - datetime.fromisoformat(run["run_start"])).total_seconds() / 86400
    state["runs"].append({"outcome": outcome, "reason": reason, "day": run.get("day"),
                          "days": round(days, 2), "final_equity": round(equity, 2),
                          "final_pct": round((equity - ANCHOR) / ANCHOR * 100, 2),
                          "trades": run["trades"], "resolved_ts": datetime.now(timezone.utc).isoformat()})
    state["runs_paid" if outcome == "payout" else "runs_blown"] += 1
    paid, total = state["runs_paid"], state["runs_total"]
    log.warning(f"RUN #{total} {outcome.upper()} ({reason}) at ${equity:,.0f} after {days:.1f} days. "
                f"{paid}/{total} runs reached payout.")
    alert(f"{'💰' if outcome == 'payout' else '💥'} <b>RUN #{total} {outcome.upper()}</b> -- {reason}\n"
          f"Equity ${equity:,.0f} after {days:.1f} days, {run['trades']} trades\n"
          f"🏆 {paid}/{total} runs reached payout", book=BOOK, html=True)


def open_trade(engine: NinjaTraderEngine, state: dict, leg: dict, spec, risk_usd: float) -> None:
    run = state["run"]
    sym = leg["symbol"]
    pv = engine.usd_per_point(sym)
    if not pv:
        log.warning(f"[{sym}] no point value from NinjaTrader -- skipped")
        return
    sl_dist = abs(spec.entry_price - spec.stop_loss)
    per_contract = sl_dist * pv + 2 * futures.commission_per_side(sym)
    if leg.get("contracts"):
        # Fixed contracts (roster): the Monte Carlo found they beat $-risk
        # sizing for this stack. Still never more than a full stop-out that
        # fits in 90% of the room above the MLL floor -- a bigger stop could
        # only end the account.
        run = state["run"]
        room = ANCHOR + run["realized_usd"] + state.get("floating_usd", 0.0) - mll_floor(run)
        contracts = min(int(leg["contracts"]),
                        futures.contracts_for_risk(0.9 * max(room, 0.0), per_contract, 1.0))
    else:
        contracts = futures.contracts_for_risk(risk_usd, per_contract, 1.0)
    used = lucid_rules.open_units(engine.open_contracts())
    contracts = min(contracts, lucid_rules.max_addable(sym, used, PLAN))
    if contracts < 1:
        log.info(f"[{sym}] {leg['label']} signal skipped -- one contract risks ${per_contract:,.0f}, "
                 f"budget ${risk_usd:,.0f} (or the contract cap is full)")
        return
    result = engine.place_market_order(symbol=sym, direction=spec.direction.value, lots=contracts,
                                       sl=spec.stop_loss, tp=spec.take_profit, magic=leg["magic"],
                                       comment=leg["label"], signal_entry_price=spec.entry_price)
    if not result.ok:
        log.warning(f"[{sym}] order failed: {result.comment}")
        return
    real_risk = result.volume * (abs(result.price - result.sl) * pv + 2 * futures.commission_per_side(sym))
    run["tickets"][str(result.ticket)] = {
        "sym": sym, "label": leg["label"], "direction": spec.direction.value,
        "entry_price": result.price, "sl": result.sl, "tp": result.tp, "lots": result.volume,
        "risk_usd": round(real_risk, 2), "opened_ts": datetime.now(timezone.utc).isoformat()}
    run["entries_today"] += 1
    log.info(f"[{sym}] opened {spec.direction.value} {result.volume:.0f} @ {result.price} "
             f"stop {result.sl} target {result.tp} risk ${real_risk:,.0f} ({leg['label']})")
    alert(f"{'🟢📈' if spec.direction.value == 'long' else '🔴📉'} <b>OPENED</b> {sym} "
          f"{spec.direction.value} x{result.volume:.0f} @ {result.price}\n"
          f"{leg['label']} -- risk ${real_risk:,.0f}", book=BOOK, html=True)


# --- main loop -----------------------------------------------------------------

def main() -> int:
    account = os.environ.get("NT_FUNDED_ACCOUNT", "Sim102").strip()
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
    if live and os.environ.get("LUCID_FUNDED_LIVE_CONFIRM", "").strip() != acc.login:
        raise RuntimeError(f"{acc.login} is not a simulation account. To trade it, set "
                           f"LUCID_FUNDED_LIVE_CONFIRM={acc.login} in .env.")
    legs = load_roster()
    state = load_state()
    state["roster"] = [{k: v for k, v in leg.items() if k != "strategy"} for leg in legs]
    state["account"] = acc.login
    state["live"] = live
    save_state(state)
    log.warning(f"Lucid funded bot on {'LIVE' if live else 'SIM'} account {acc.login}: "
                f"{len(legs)} legs -- " + ", ".join(leg["label"] for leg in legs))
    if not legs:
        log.error("empty roster -- nothing to trade (the bot keeps running to manage state)")

    last_bar: dict[str, object] = {}
    _touch_heartbeat()
    threading.Thread(target=_self_watchdog, daemon=True, name="self-watchdog").start()

    while True:
        try:
            _touch_heartbeat()
            run = state["run"]
            reconcile(engine, state)
            now = datetime.now(timezone.utc)
            floating = floating_usd(engine, run)
            equity = ANCHOR + run["realized_usd"] + floating
            state["floating_usd"] = round(floating, 2)

            # ---- trade-date rollover: mark the end-of-day balance ----
            day = lucid_rules.trading_day(now)
            if run["day"] is None:
                run["day"] = day
                run["day_start_equity"] = equity
            elif day != run["day"]:
                balance = ANCHOR + run["realized_usd"]      # flat by now (Lucid rule)
                run["peak_eod"] = max(run["peak_eod"], balance)
                if not run["mll_locked"] and balance >= MLL_LOCK_TRIGGER:
                    run["mll_locked"] = True
                    log.warning(f"MLL LOCKED at ${MLL_LOCKED_FLOOR:,.0f} (EOD balance ${balance:,.0f})")
                run.update(day=day, day_start_equity=equity, dll_hit=False, entries_today=0)

            # ---- account-ending: the Max Loss Limit ----
            floor = mll_floor(run)
            if equity <= floor:
                close_all(engine, run, f"MLL breach: equity ${equity:,.0f} <= floor ${floor:,.0f}")
                end_run(state, "blown", f"MLL: equity ${equity:,.0f} <= ${floor:,.0f}", equity)
                if live:
                    state.update(halted=True, halt_reason=f"MLL breached at ${equity:,.0f}")
                else:
                    state["run"] = new_run()
                    state["runs_total"] += 1
                save_state(state)
                time.sleep(POLL_INTERVAL_S)
                continue

            # ---- payout level (closed balance, so floating profit cannot count) ----
            balance = ANCHOR + run["realized_usd"]
            if balance >= ANCHOR + PAYOUT_TARGET_USD and not run["tickets"]:
                if live:
                    if not run["payout_alerted"]:
                        run["payout_alerted"] = True
                        alert(f"💰 <b>{acc.login}: payout level reached</b> -- balance ${balance:,.0f}. "
                              f"Request the payout in Lucid's dashboard.", book=BOOK, html=True)
                else:
                    end_run(state, "payout", f"balance ${balance:,.0f}", balance)
                    state["run"] = new_run()
                    state["runs_total"] += 1
                    save_state(state)
                    time.sleep(POLL_INTERVAL_S)
                    continue

            # ---- daily loss limit (soft) ----
            day_pnl = equity - run["day_start_equity"]
            if day_pnl <= -DLL_USD and not run["dll_hit"]:
                run["dll_hit"] = True
                close_all(engine, run, f"daily loss limit: ${day_pnl:,.0f} today")
                alert(f"🛑 <b>Daily loss limit</b> -- ${day_pnl:,.0f} today; flat until the next session.",
                      book=BOOK, html=True)

            # ---- Lucid's flat rule ----
            session = lucid_rules.session_state(now)
            if session == lucid_rules.Session.FLATTEN and run["tickets"]:
                close_all(engine, run, "Lucid 4:45 PM ET flat rule")

            state["risk_today_usd"] = round(risk_budget(equity, run), 2)
            entries_ok = (session == lucid_rules.Session.OPEN and not state["halted"]
                          and not run["dll_hit"] and day_pnl > -DLL_ENTRY_BLOCK_FRAC * DLL_USD)
            # ---- per-strategy time exit (e.g. ORB closes at 15:55 ET, as tested) ----
            by_label = {leg["label"]: leg for leg in legs}
            for key, info in list(run["tickets"].items()):
                leg = by_label.get(info["label"])
                if leg is not None and hasattr(leg["strategy"], "exit_due") and leg["strategy"].exit_due(now):
                    log.info(f"[{info['sym']}] time exit for {info['label']}")
                    engine.close_position(int(key))
                    _touch_heartbeat()

            if entries_ok:
                for leg in legs:
                    sym = leg["symbol"]
                    if any(t["sym"] == sym for t in run["tickets"].values()):
                        continue                      # one position per contract
                    if not engine.is_tradeable(sym):
                        continue
                    count = max(BAR_COUNT, getattr(leg["strategy"], "REQUIRED_BARS", 0))
                    bars = engine.get_bars(sym, leg["tf"], count)
                    key = f"{sym}|{leg['tf']}"
                    if last_bar.get(key) == bars.index[-1]:
                        continue                      # no new closed bar
                    last_bar[key] = bars.index[-1]
                    closed = bars.iloc[:-1]
                    specs = leg["strategy"].on_bar(len(closed) - 1, closed) or []
                    if specs:
                        open_trade(engine, state, leg, specs[0], risk_budget(equity, run))
                    _touch_heartbeat()

            save_state(state)
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.warning("Stopped by user.")
            return 0
        except Exception as e:  # noqa: BLE001 -- the loop must survive a bad poll
            log.error(f"Loop error (continuing): {e!r}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
