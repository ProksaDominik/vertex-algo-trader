"""LucidPro evaluation-passing trial runner.

Repeatedly attempts a fresh $50,000 virtual LucidPro evaluation, live on the
MT5 demo account, using the exact config validated in tonight's backtest
research (see scratch/lucidpro_sameday_search.py and scratch/lucidpro_*.py):
15m London Breakout on the six symbols this account can actually fill (see
SYMBOLS below -- the original US500/NAS100 basket could not), 90% risk/trade
(raised from 50% on
2026-08-14, see scratch/lucidpro_higherisk_sweep.py), 10 concurrent
positions, 10 entries/day cap.

LucidPro rules modeled (all confirmed via web research this session, not
guessed):
    +6% target   -- checked continuously (can pass intraday, same-day).
    -4% Max Loss Limit -- trailing off the peak of each day's CLOSE.
        CHANGED 2026-08-29: now checked EVERY POLL, not only at end of day.
        LucidPro's documented rule marks this at the daily close, and that is
        what the ~70% backtest assumed -- but pairing an EOD-only FAIL check
        with a continuous PASS check made the bot structurally incapable of
        failing. Combined with an attempt that was not persisted across
        restarts, it recorded 8 attempts / 8 passes / 0 fails, which measured
        state loss rather than performance. Live pass rates from here are
        STRICTER than that backtest figure and not directly comparable to it.
    -2.4% Daily Loss Limit -- SOFT: blocks new entries for the rest of the
        calendar day once floating equity dips this far; does not fail the
        account and does not close existing positions.

Every attempt is virtual bookkeeping laid on top of real (demo) MT5 trades:
risk_usd is a FIXED $45,000 (90% of the $50,000 virtual anchor) per trade
for the whole attempt, not recalculated from a live account balance -- this
matches the backtest's convention exactly, so the live pass rate is
comparable to the backtested ~70.0%/1-day, ~72.6%/5-day figures (up from
64.2%/67.5% at the old 50% risk -- see scratch/lucidpro_higherisk_sweep.py).

On PASS or FAIL: every open position opened by this trial is closed at
market, the outcome + duration is appended to the stats file, and a brand
new $50,000 attempt starts on the very next bar. Stats accumulate in
data/state_lucidpro_trial.json (attempts, passes, fails, pass_rate, and a
per-attempt history) so the dashboard / a quick script can report "how many
tries, how many passed" at any time.

EOD_HOUR_UTC is a documented ASSUMPTION (22:00 UTC, approx. CME futures
daily settlement) -- Lucid's exact EOD boundary was not independently
confirmed this session. Flag and correct if real fills disagree.

Paper/demo only (MetaQuotes-Demo). This is NOT connected to a real Lucid
account -- CFD proxy instruments, for pass-rate validation only, per the
user's explicit "test the pass rate" request.

FUTURES MODE (VERTEX_BROKER=ninjatrader in .env), added 2026-09-29:
the same runner on real CME futures through NinjaTrader 8 -- the only way to
reach a Lucid account, since Lucid is futures-only and has no MT5 connection.
What changes: the basket maps to Lucid-approved contracts (FUTURES_SYMBOLS),
sizes are whole contracts under Lucid's 4-mini / 40-micro cap, entries stop
15 minutes before and positions are flattened 5 minutes before Lucid's
4:45 PM ET flat rule, the trading day rolls at 17:00 ET, and results go to
separate state/trade files. On NinjaTrader's Sim101 it keeps cycling virtual
attempts exactly like the MT5 version (that is the futures pass-rate test);
on a real Lucid account it STOPS trading once the evaluation PASSES. After a
fail it waits until the account is reset to $50,000, then keeps racing.

Since 2026-09-30 futures mode runs the TARGET RACE by default (target_race.py,
docs/LUCIDPRO_PASS_RESEARCH.md): once a day long 2 NQ, limit at the exact
target, far safety stop, out 16:00 ET. LUCID_RULE_READING picks how the
simulator attempts judge the limits: "lucid" (default) = the rules Lucid
support confirmed -- MLL on equity including the open position, DLL on
realized losses then liquidation; "lenient" = closed balance only (research).
LUCID_TRIAL_STRATEGY=orb_trend switches back to the ORB + trend stack.

Usage:
    .venv/Scripts/python.exe scripts/run_lucidpro_trial.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # picks up TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID for metatraderbot.alerts

from metatraderbot import futures, lucid_rules, target_race  # noqa: E402
from metatraderbot.alerts import alert  # noqa: E402
from metatraderbot.instruments import contract_mult  # noqa: E402
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy  # noqa: E402

# Which broker layer to run on -- see FUTURES MODE in the module docstring.
FUTURES = os.environ.get("VERTEX_BROKER", "mt5").strip().lower() == "ninjatrader"
if FUTURES:
    from metatraderbot.nt_engine import NinjaTraderEngine  # noqa: E402
else:
    from metatraderbot.mt5_engine import MT5Engine  # noqa: E402

# A second, PAPER-ONLY trial bot that picks the side another way, next to the deployed one (owner, 2026-10-21:
# "one tab the current strategy and one tab this new strategy and we will compare them"): --strategy ad = A-D
# breadth (src/metatraderbot/breadth.py) -- its own state, trade log, log file and paper book, never a real account.
STRATEGY = (sys.argv[sys.argv.index("--strategy") + 1].strip().lower() if "--strategy" in sys.argv[:-1] else "")
STRAT_TAG = f"_{STRATEGY}" if STRATEGY else ""
LOG_NAME = ("lucid_futures" if FUTURES else "lucidpro_trial") + STRAT_TAG
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"{LOG_NAME}.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(LOG_NAME)

# --- Config: exactly the backtested "passing" setup -------------------------
# --- REWORKED 2026-08-25 -----------------------------------------------------
#
# Was ["US500", "NAS100", "XAUUSD"]. Two of those three CANNOT FILL on this
# account: both report trade_mode=0 (DISABLED) while still streaming quotes, so
# they look tradeable and silently reject. A scan of every deal since 2025-01
# (400+ entries) contains no index trade at all, and magics 9100 (US500) and
# 9101 (NAS100) have zero fills in the account's entire history -- only 9102
# (XAUUSD) ever filled. The bot has therefore been running a ONE-symbol basket
# while its published pass rate was measured on three.
#
# Measured cost of that, at 2x cost stress (scratch/_trial_sweep.py):
#     believed (3 symbols incl. the dead ones) : 66.1% (1D) / 71.2% (5D)
#     actually running (XAUUSD alone)          : 48.5% (1D) / 60.7% (5D)
#
# Replacement basket chosen by greedy search over the 18 symbols with confirmed
# fills in this account's real deal history, then validated out-of-sample --
# selected on the first 70% of days, frozen, scored once on the last 30%
# (scratch/_trial_oos.py):
#     train 75.6%   TEST 71.4% (1D) / 77.3% (5D)   n_test=774
#     incumbent on those same test days: 47.3% (1D) / 59.5% (5D)
#     -> +24.1 pp out-of-sample, z=9.7, p=5e-22, Cohen's h=0.50
#
# The train-selected basket is the SAME six symbols as the full-history one,
# which is the real evidence it is not the search fitting noise: 6-of-18 picked
# by score would not reproduce itself on a different sample if it were.
# GBPCHF, not EURCHF: the first search ran without GBPCHF because that cell had
# died with MemoryError during the cache build, so its absence was an artifact
# of the build, not a result. With the cache complete it is selected in every
# configuration -- 50%/70%/90% risk, train-only and full-history alike. The
# measured gain is small and within noise on its own (+1.1 pp out-of-sample,
# n=783, five of six symbols shared with the alternative); it is taken because
# it is consistently chosen and costs nothing, not because 1.1 pp is proven.
SYMBOLS = ["GBPCHF", "USDCAD", "GBPUSD", "GBPJPY", "USDCHF", "XAUUSD"]
TIMEFRAME = "15m"
VIRTUAL_ACCOUNT = 50_000.0
TARGET_PCT = 0.06        # +6%, checked continuously
MLL_PCT = 0.04           # -4%, trailing off peak EOD close, checked EOD-only
DLL_PCT = 0.024          # -2.4%, soft, blocks new entries only
RISK_PCT = 0.90          # 90% of the $50,000 virtual anchor, FIXED per attempt.
# Was 0.50 until 2026-08-14. scratch/lucidpro_higherisk_sweep.py extended the
# extreme-risk sweep (previously only tested 10% and 50%) across 40-99.9% at
# conc=10/day=10 on the same 15m LondonBreakout cache: pass rate climbs
# monotonically with risk and plateaus around 90-95% (1D 69.8-70.2%, 5D
# 72.3-72.7%) vs 50%'s 64.2%/67.5%. 90% captures nearly all of that gain
# (within 0.4pp of the 99.9% ceiling) while leaving some margin before the
# broker's real max-lot clamp (see the CLAMPED-order handling below) bites
# even harder than it already does at 50%.
RISK_USD = RISK_PCT * VIRTUAL_ACCOUNT   # $45,000/trade -- matches backtest convention
MAX_CONCURRENT = 10
MAX_PER_DAY = 10
EOD_HOUR_UTC = 22        # ASSUMPTION -- see module docstring
MAGIC_BASE = 9100        # first unused block (FTMO=6100s, YOLO=7100/7300/7900s, FTMO_4DAY=8100s)

# PINNED PER SYMBOL, never derived from the SYMBOLS index.
#
# This was `MAGIC_BASE + i for i, sym in enumerate(SYMBOLS)` until 2026-08-25.
# Reworking the basket that day would have silently renumbered everything:
# XAUUSD moves from index 2 to index 3, so magic 9102 -- which owns the only
# two trades this bot has ever filled -- would have been handed to GBPJPY, and
# XAUUSD's own history would have become unattributable. That is exactly the
# bug that orphaned a live position in run_lucidpro_funded.py (ticket
# 10124062219) when the BTCUSD legs were disabled there.
#
# New symbols get NEW numbers (9110+); 9100-9102 stay retired so the existing
# ledger rows and deal history keep resolving to the roster that opened them.
SYMBOL_MAGICS = {
    # v2 basket, active 2026-08-25
    "USDCAD": 9110,
    "GBPUSD": 9111,
    "GBPJPY": 9112,
    "XAUUSD": 9113,
    "USDCHF": 9114,
    "EURCHF": 9115,   # selected in the first pass, replaced by GBPCHF; kept pinned
    "GBPCHF": 9116,
    # retired -- kept so historical fills still resolve to a name
    "US500":  9100,   # never filled: index CFDs are trade-disabled on this account
    "NAS100": 9101,   # never filled: same
    # 9102 was XAUUSD under the original basket (its only 2 historical fills)
}
POLL_INTERVAL_S = 15
STATE_PATH = ROOT / "data" / "state_lucidpro_trial.json"
TRADES_LOG_PATH = ROOT / "data" / "lucidpro_trades.jsonl"
BAR_COUNT = 300

# --- Futures mode (VERTEX_BROKER=ninjatrader) --------------------------------
#
# REWORKED 2026-09-30. London Breakout on the mapped CFD basket (6C/6B/6S/MGC)
# has NO edge on futures: -0.08/-0.03/+0.05/+0.04R per trade over ~15 months of
# real futures bars (scripts/futures_roster_search.py). Futures mode therefore
# trades the one intraday idea that survived the futures research instead --
# opening-range breakout with the daily-trend filter on MNQ and MGC
# (strategies_orb_trend.py; docs/FUTURES_RESEARCH_REPORT.md) -- at a FIXED
# evaluation size chosen by scripts/futures_stack_mc.py's LucidPro Monte
# Carlo: 6 MNQ + 3 MGC -> ~40% pass, ~1.52 failed evaluations per pass,
# median ~16 trading days over 2016-26 (44% / 1.29 / 10 days over 2021-26),
# versus 31% / 2.24 for a zero-edge strategy of the same size. Same odds as
# 4+2 at about twice the speed. The owner's budget is 1-2 failed evaluations
# per pass. Fixed contracts beat fixed-$-risk sizing over 2016-26
# (lucid_riskbased_mc.py): the edge lives on wide-range days, which
# risk-based sizing shrinks. (Figures after the 2026-09-30 correction that
# stopped the Monte Carlo capping days at -$1,300; the capped run had shown
# an inflated ~49% / 1.04.) CLAMP_RISK_TO_MLL_ROOM still trims contracts on
# very wide-range days so no single stop can end the attempt.
# Not the CFD experiment's 90%-risk race, which that budget rules out.
FUTURES_STACK = {"MNQ": 6, "MGC": 3}          # contracts per leg
FUTURES_STRATEGY = ("orb_trend", {"or_minutes": 45, "last_entry": "12:00", "exit_time": "15:55"}, 2.0)

# --- REWORKED AGAIN 2026-09-30: the TARGET RACE is the default ----------------
# docs/LUCIDPRO_PASS_RESEARCH.md: no strategy edge gets a LucidPro evaluation
# near the owner's 60-70% (best stacks 30-45%). Lucid support confirmed that the
# MLL counts the open position's equity, so the race's ~83% (closed-balance
# premise) is off the table; under the confirmed rules it is still the best
# option found: ~41% on 50K at 2 NQ (real MNQ bars 2025-26: 44%), 1.45 failed
# evaluations per pass -- the arithmetic ceiling for a $3,000 target against a
# $2,000 limit is ~40%. LUCID_TRIAL_STRATEGY=orb_trend brings back ORB + trend.
TRIAL_STRATEGY = os.environ.get("LUCID_TRIAL_STRATEGY", "race").strip().lower()
if TRIAL_STRATEGY not in ("race", "orb_trend"):
    raise ValueError(f"LUCID_TRIAL_STRATEGY must be 'race' or 'orb_trend', not {TRIAL_STRATEGY!r}")
RACE = target_race.RaceConfig(
    symbol=os.environ.get("LUCID_RACE_SYMBOL", "NQ").strip().upper(),
    contracts=int(os.environ.get("LUCID_RACE_CONTRACTS", "2")),
    entry=os.environ.get("LUCID_RACE_ENTRY", "19:00").strip(),
)
RACE_MODE = FUTURES and TRIAL_STRATEGY == "race"
# Which way to race (owner, 2026-10-07: "we can't work only with trades that predict up"):
# "trend200" = long above the 200-day average, short below (target_race.race_side); "long" = always long.
RACE_SIDE_RULE = "breadth" if STRATEGY == "ad" else os.environ.get("LUCID_RACE_SIDE", "trend200").strip().lower()
# How the Sim101 attempts judge Lucid's limits (a real Lucid account enforces
# its own). "lucid" = the rules Lucid support CONFIRMED on 2026-09-30: the MLL
# is checked against equity INCLUDING the open position, in real time; the DLL
# counts realized losses only and then sets Done-for-Day and liquidates what is
# open. "lenient" (MLL and DLL on the closed balance) was the race's original
# premise and is kept for research only. "strict" is an alias of "lucid".
RULE_READING = os.environ.get("LUCID_RULE_READING", "lucid").strip().lower()
RULE_READING = "lucid" if RULE_READING == "strict" else RULE_READING
if RULE_READING not in ("lucid", "lenient"):
    raise ValueError(f"LUCID_RULE_READING must be 'lucid' or 'lenient', not {RULE_READING!r}")

FUTURES_SYMBOLS = [RACE.symbol] if RACE_MODE else list(FUTURES_STACK)
FUTURES_SYMBOL_MAGICS = {   # pinned, same rule as SYMBOL_MAGICS
    "6C": 9120,             # retired London Breakout legs -- kept so history resolves
    "6B": 9121,
    "6S": 9122,
    "MGC": 9123,
    "MNQ": 9124,
    "NQ": 9125,             # target race
}
LUCID_PLAN = lucid_rules.LUCIDPRO_50K
# LucidPro 50K is the account modelled above; the percentages must agree with it.
assert abs(TARGET_PCT * VIRTUAL_ACCOUNT - LUCID_PLAN.profit_target) < 1e-6
assert abs(MLL_PCT * VIRTUAL_ACCOUNT - LUCID_PLAN.max_loss_limit) < 1e-6
assert abs(DLL_PCT * VIRTUAL_ACCOUNT - LUCID_PLAN.daily_loss_limit) < 1e-6

# A trade whose stop-loss risk exceeds the distance left to the Max Loss Limit
# can never reach its stop: Lucid fails the account first, so its "stop" is a
# fiction and one loss ends a paid evaluation. With this on, a futures trade is
# sized so its full stop-out (plus commission) fits inside MLL_ROOM_FRACTION of
# the remaining room. This is a deliberate departure from the CFD experiment's
# fixed 90% risk -- set it to False only to reproduce that experiment on Sim101.
CLAMP_RISK_TO_MLL_ROOM = True
MLL_ROOM_FRACTION = 0.9

if FUTURES:
    SYMBOLS = FUTURES_SYMBOLS
    SYMBOL_MAGICS = FUTURES_SYMBOL_MAGICS
    STATE_PATH = ROOT / "data" / f"state_lucid_futures{STRAT_TAG}.json"
    TRADES_LOG_PATH = ROOT / "data" / f"lucid_futures_trades{STRAT_TAG}.jsonl"
BOOK = ("Lucid-Futures" if FUTURES else "LucidPro-Eval") + (" A-D" if STRATEGY == "ad" else "")
# The paper trial races like a real account: once per trading day at 19:00 New York, also after a fail or a
# pass (owner 2026-10-15: "tell me if the rule when one fails start another one is good" -- restarts win ~39%
# vs ~42-44% for the 19:00 race, and the money plan assumes one race a day). 0 = the old immediate restarts.
ONE_RACE_A_DAY = os.environ.get("LUCID_TRIAL_ONE_RACE_A_DAY", "1").strip() != "0"

# Self-watchdog: 2026-08-17 -- the funded bot was found silently hung for
# ~22h with zero log output (still "running" per Windows -- likely an MT5
# IPC call blocked forever after a connectivity blip instead of raising).
# The external 15-min watchdog.ps1 only checks whether the process EXISTS,
# so a hung-but-alive process passes that check forever. Same fix applied
# here: force-kill this process if the main loop stops making progress, so
# the external watchdog gets a dead process to restart within minutes
# instead of a hung one that looks fine. See run_lucidpro_funded.py for
# the funded-side twin of this.
_HEARTBEAT_TIMEOUT_S = 180  # 12x POLL_INTERVAL_S -- generous vs. normal cadence, far below "hours"
_last_heartbeat = time.monotonic()
_heartbeat_lock = threading.Lock()


def _touch_heartbeat() -> None:
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.monotonic()


def _self_watchdog() -> None:
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
            log.error(f"Self-watchdog: no loop progress for {stale_s:.0f}s (limit "
                      f"{_HEARTBEAT_TIMEOUT_S}s) -- likely stuck in a blocking MT5 call. "
                      f"Force-exiting so the external watchdog restarts a clean process.")
            os._exit(1)  # nuclear on purpose: a truly hung C-extension call can't be interrupted any other way


def log_trade(attempt_number: int, sym: str, direction: str, entry_price: float,
              exit_price: float, lots: float, risk_usd: float, r_multiple: float,
              opened_ts: str, closed_ts: str, extra: dict | None = None) -> None:
    """Append-only trade log, one JSON object per line -- independent of
    the aggregate state file so it can grow indefinitely without needing
    the whole state re-written on every trade. `extra` adds the race's
    target / MLL line / exit reason, which the dashboard's trade charts draw."""
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt": attempt_number, "sym": sym, "direction": direction,
        "entry_price": entry_price, "exit_price": exit_price, "lots": lots,
        "risk_usd": risk_usd, "r_multiple": round(r_multiple, 4),
        "pnl_usd": round(r_multiple * risk_usd, 2),
        "opened_ts": opened_ts, "closed_ts": closed_ts,
    }
    record.update({k: v for k, v in (extra or {}).items() if v is not None})
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_state() -> dict:
    defaults = {
        "attempts": 0, "passes": 0, "fails": 0,
        "total_trades": 0, "total_wins": 0, "total_losses": 0,
        "current_attempt_start": None, "current_attempt_opened": 0,
        "history": [],   # [{outcome, days, opened_ts, resolved_ts, final_pct, trades, wins, losses}]
    }
    if STATE_PATH.exists():
        try:
            on_disk = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            defaults.update(on_disk)
        except (OSError, ValueError):
            log.error("State file unreadable, starting fresh")
    return defaults


def save_state(state: dict) -> None:
    # 2026-08-18 -- this whole project lives under OneDrive, which syncs the
    # data/ dir constantly given how often this writes; its sync client
    # transiently locks files mid-sync, causing os.replace() to raise
    # PermissionError (WinError 5) with no fault of ours. Retry briefly
    # instead of letting a normal, expected OneDrive lock blip cost a whole
    # loop iteration. See run_lucidpro_funded.py's twin of this fix.
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


class Attempt:
    """One $50,000 virtual evaluation attempt. Tracks realized + floating R
    (in USD, off the fixed RISK_USD), the trailing EOD peak, and per-day
    entry counts -- exactly the mechanics validated in
    scratch/lucidpro_sameday_search.py's simulate_day, translated to a live
    polling loop instead of a historical bar-path replay."""

    def __init__(self):
        self.opened_ts = datetime.now(timezone.utc)
        self.realized_usd = 0.0
        self.peak_equity = VIRTUAL_ACCOUNT
        self.current_day: str | None = None
        self.entries_today = 0
        self.total_entries = 0   # cumulative trades this attempt, for stats -- entries_today resets daily
        self.wins = 0
        self.losses = 0
        self.dll_blocked_today = False
        self.day_start_realized = 0.0        # realized_usd when the trading day began (Lucid's DLL base)
        self.tickets: dict[int, dict] = {}   # ticket -> {sym, entry_price, sl, tp, direction, lots, risk_usd}

    def floating_usd(self, engine: MT5Engine) -> float:
        # Uses each ticket's OWN risk_usd, not the global RISK_USD constant --
        # a clamped (broker-max-limited) order has genuinely less real dollar
        # risk than the nominal $25,000, and crediting it the full amount
        # would overstate that position's real contribution to equity.
        total = 0.0
        for ticket, info in self.tickets.items():
            positions = engine_positions_by_ticket(engine, ticket)
            if positions is None:
                continue
            price = positions.price_current
            entry = info["entry_price"]
            sl_dist = abs(entry - info["sl"])
            if sl_dist <= 0:
                continue
            sign = 1.0 if info["direction"] == "long" else -1.0
            r_multiple = (price - entry) * sign / sl_dist
            total += r_multiple * info["risk_usd"]
        return total

    def equity(self, engine: MT5Engine) -> float:
        return VIRTUAL_ACCOUNT + self.realized_usd + self.floating_usd(engine)

    def day_key(self, now: datetime) -> str:
        if FUTURES:
            # CME trade date, rolling at 17:00 ET (daylight saving included) --
            # after Lucid's 4:45 PM flat rule, so the mark is a flat balance.
            return lucid_rules.trading_day(now)
        # A "trading day" rolls at EOD_HOUR_UTC, not midnight -- matches the
        # backtest's EOD-close marking convention.
        shifted = now
        if now.hour < EOD_HOUR_UTC:
            shifted = now.replace(hour=0) - __import__("datetime").timedelta(days=1)
        return shifted.strftime("%Y-%m-%d")

    # --- persistence ------------------------------------------------------
    #
    # Added 2026-08-29. Until now an Attempt lived only in memory, so every
    # restart silently reset realized_usd to 0, peak_equity to the anchor and
    # tickets to {} -- ERASING the attempt's accumulated loss. The self-watchdog
    # restarts this process regularly (it fired at 12:28 that day), so a losing
    # attempt was reliably wiped before it could ever breach the MLL, while the
    # pass check -- which runs every poll -- still fired on the first winner.
    #
    # That asymmetry, not skill, produced the "8 attempts, 8 passed, 0 failed"
    # record: at 90% risk one winning trade (+0.08R x $45,000 = +$3,600) clears
    # the +$3,000 target, and any loss was forgotten on the next restart.
    # Those 8 passes are an artifact and should not be read as a pass rate.
    def to_dict(self) -> dict:
        return {
            "opened_ts": self.opened_ts.isoformat(),
            "realized_usd": self.realized_usd,
            "peak_equity": self.peak_equity,
            "current_day": self.current_day,
            "entries_today": self.entries_today,
            "total_entries": self.total_entries,
            "wins": self.wins,
            "losses": self.losses,
            "dll_blocked_today": self.dll_blocked_today,
            "day_start_realized": self.day_start_realized,
            "tickets": {str(k): v for k, v in self.tickets.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Attempt":
        a = cls()
        try:
            a.opened_ts = datetime.fromisoformat(d["opened_ts"])
        except Exception:
            pass
        a.realized_usd = float(d.get("realized_usd", 0.0))
        a.peak_equity = float(d.get("peak_equity", VIRTUAL_ACCOUNT))
        a.current_day = d.get("current_day")
        a.entries_today = int(d.get("entries_today", 0))
        a.total_entries = int(d.get("total_entries", 0))
        a.wins = int(d.get("wins", 0))
        a.losses = int(d.get("losses", 0))
        a.dll_blocked_today = bool(d.get("dll_blocked_today", False))
        a.day_start_realized = float(d.get("day_start_realized", a.realized_usd))
        a.tickets = {int(k): v for k, v in (d.get("tickets") or {}).items()}
        return a


def engine_positions_by_ticket(engine: MT5Engine, ticket: int):
    # Through the engine rather than the MetaTrader5 package directly, so the
    # same runner works on NinjaTrader.
    return engine.get_position(ticket)


def reconcile_closed_tickets(engine: MT5Engine, attempt: "Attempt", attempt_number: int,
                              state: dict) -> None:
    """A ticket that hit its SL/TP naturally (not via close_all()) used to
    just vanish from attempt.tickets with no accounting: floating_usd()
    silently skips any ticket engine_positions_by_ticket can't find (see its
    docstring), so a mid-attempt SL/TP close previously contributed NOTHING
    to attempt.realized_usd -- a real win or loss disappeared from equity
    until the whole attempt resolved, which could delay-or-miss the target/
    MLL checks against the account's true state. FIXED 2026-08-16: detect
    the close every poll via MT5Engine.get_closed_deal() (already built for
    exactly this), credit its REAL broker-reported P&L (commission+swap
    included, not the R-multiple approximation), and alert on it."""
    for ticket, info in list(attempt.tickets.items()):
        if engine_positions_by_ticket(engine, ticket) is not None:
            continue  # still open
        deal = engine.get_closed_deal(ticket)
        if deal is not None and RACE_MODE and deal.get("exit_reason") in ("tp", "sl"):
            # On waking from sleep the simulator can fill the take-profit (or
            # stop) although the fail line or the 16:00 ET exit came first --
            # Lucid judges what happened first. Check the 1-minute history up
            # to the fill.
            try:
                ev = audit_history(engine, attempt, until_utc=deal["closed_ts"])
            except Exception as e:  # noqa: BLE001 -- no history: trust the fill
                log.debug(f"bracket-fill history check skipped: {e!r}")
                ev = None
            if ev is not None and ev[0] in ("fail", "exit"):
                what, px, when, _, _ = ev
                pv = engine.usd_per_point(info["sym"]) or futures.point_value(info["sym"])
                sign = 1.0 if info["direction"] == "long" else -1.0
                lots = float(info["lots"])
                pnl = (px - info["entry_price"]) * sign * lots * pv - 2 * futures.commission_per_side(info["sym"]) * lots
                log.warning(f"[{info['sym']}] {deal['exit_reason']} filled at {deal['exit_price']:.2f}, but "
                            f"{'the fail line' if what == 'fail' else 'the 16:00 ET exit'} came first at "
                            f"{when:%H:%M} UTC -- booked at {px:.2f}")
                deal = {**deal, "pnl_usd": pnl, "exit_price": px, "closed_ts": when,
                        "exit_reason": "fail_rescored" if what == "fail" else "exit_while_away"}
        if deal is None and RACE_MODE:
            # NinjaTrader no longer knows the position (restarted while the PC
            # was off): judge it from the price history, as Lucid would have.
            history_ok = True
            try:
                ev = audit_history(engine, attempt)
            except Exception as e:  # noqa: BLE001 -- no history: fall through to the write-off
                log.debug(f"history settlement of a vanished ticket skipped: {e!r}")
                ev, history_ok = None, False
            if ev is not None:
                settle_from_history(engine, attempt, state, ev)
                continue
            if history_ok:
                # The history says the race is still running: NinjaTrader lost the position (2026-10-09:
                # it restarted at 01:53 Prague, evaluation #30's race was dropped here, and the history
                # shows its target was hit at 03:46). Keep it -- the history audit settles it at the
                # target, the fail line or the 16:00 ET exit, exactly as Lucid's server would.
                if not info.get("on_history"):
                    info["on_history"] = True
                    log.warning(f"[{info['sym']}] ticket {ticket}: NinjaTrader no longer shows the position -- "
                                f"the race goes on, judged on the price history")
                info["missing"] = 0
                continue
            # No history yet (NinjaTrader still connecting after a restart):
            # keep the ticket and try again -- write it off only after ~5 min.
            info["missing"] = int(info.get("missing", 0)) + 1
            if info["missing"] < 20:
                continue
        del attempt.tickets[ticket]
        if deal is None:
            log.warning(f"[{info['sym']}] ticket {ticket} no longer open but no closed-deal "
                        f"record found -- dropping without crediting P&L (broker history gap)")
            continue
        pnl_usd = deal["pnl_usd"]
        if info.get("virtual_entry") and not str(deal.get("exit_reason", "")).endswith("rescored"):
            # The race's own entry price, not NinjaTrader's fill: a race carried on from the
            # replayed record (2026-10-07) keeps its replayed entry, as its floating P&L does.
            pv_ = engine.usd_per_point(info["sym"]) or futures.point_value(info["sym"])
            lots_ = float(info["lots"])
            pnl_usd = ((float(deal["exit_price"]) - info["entry_price"]) * (1.0 if info["direction"] == "long" else -1.0)
                       * lots_ * pv_ - 2 * futures.commission_per_side(info["sym"]) * lots_)
            deal = {**deal, "pnl_usd": pnl_usd}
        r_multiple = pnl_usd / info["risk_usd"] if info["risk_usd"] else 0.0
        if RACE_MODE and deal.get("close_reason") in ("protection_lost", "gap"):
            # The bridge closed the race at once because its stop/target could
            # not be placed -- an order-handling failure, not a race result.
            # Let the same attempt race again (at most 3 times a day).
            day = lucid_rules.trading_day(datetime.now(timezone.utc))
            tries = state.get("race_bracket_retries", {})
            if tries.get("day") != day:
                tries = {"day": day, "n": 0}
            if tries["n"] < 3:
                tries["n"] += 1
                state["race_attempt"] = None
                log.warning(f"[{info['sym']}] race closed at entry -- its stop/target could not be placed "
                            f"({deal.get('close_reason')}); racing again (retry {tries['n']}/3 today)")
            state["race_bracket_retries"] = tries
        attempt.realized_usd += pnl_usd
        if pnl_usd > 0:
            attempt.wins += 1
        else:
            attempt.losses += 1
        log_trade(attempt_number, info["sym"], info["direction"], info["entry_price"],
                  deal["exit_price"], info["lots"], info["risk_usd"], r_multiple,
                  info.get("opened_ts", ""), deal["closed_ts"].isoformat(),
                  {"tp": info.get("tp"), "line": info.get("line"), "reason": deal.get("exit_reason")})
        attempts_done = state.get("attempts", 0)
        passes = state.get("passes", 0)
        pass_rate = passes / attempts_done if attempts_done else 0.0
        pnl_pct = pnl_usd / VIRTUAL_ACCOUNT
        result_emoji = "🟢" if pnl_usd > 0 else "🔴"
        exit_emoji = {"tp": "🎯", "sl": "🛑"}.get(deal["exit_reason"], "✋")
        attempt_eq = attempt.equity(engine)
        alert(
            f"{result_emoji} <b>CLOSED</b> {exit_emoji} <b>{deal['exit_reason'].upper()}</b> "
            f"{info['sym']} {info['direction']}\n"
            f"Profit: <b>${pnl_usd:+,.2f}</b> ({pnl_pct:+.2%} of account)  [{r_multiple:+.2f}R]\n"
            f"📊 Attempt equity: ${attempt_eq:,.0f} "
            f"({(attempt_eq - VIRTUAL_ACCOUNT) / VIRTUAL_ACCOUNT:+.2%})  "
            f"day {attempt.current_day}, {attempt.entries_today}/{MAX_PER_DAY} entries today\n"
            f"🏆 Lifetime: {passes}/{attempts_done} attempts passed ({pass_rate:.1%})",
            book=BOOK,
            html=True,
        )


def close_all(engine: MT5Engine, attempt: Attempt, attempt_number: int) -> float:
    """Close every position this attempt opened; returns total realized
    dollar P&L from those closes (for the final stats record). Uses each
    ticket's own risk_usd (see floating_usd's docstring for why). Appends
    every closed trade to TRADES_LOG_PATH for the dashboard's trade table
    and equity curve."""
    realized_from_closes = 0.0
    for ticket, info in list(attempt.tickets.items()):
        pos = engine_positions_by_ticket(engine, ticket)
        if pos is not None:
            engine.close_position(ticket)
            # Re-fetch to compute realized P&L in R terms after the close.
            entry = info["entry_price"]
            sl_dist = abs(entry - info["sl"])
            if sl_dist > 0:
                sign = 1.0 if info["direction"] == "long" else -1.0
                exit_price = pos.price_current
                r = (exit_price - entry) * sign / sl_dist
                realized_from_closes += r * info["risk_usd"]
                if r > 0:
                    attempt.wins += 1
                else:
                    attempt.losses += 1
                log_trade(attempt_number, info["sym"], info["direction"], entry, exit_price,
                          info["lots"], info["risk_usd"], r, info.get("opened_ts", ""),
                          datetime.now(timezone.utc).isoformat(),
                          {"tp": info.get("tp"), "line": info.get("line"), "reason": "closed_by_bot"})
        del attempt.tickets[ticket]
    return realized_from_closes


def resolve_attempt(state: dict, attempt: Attempt, outcome: str, final_equity: float) -> None:
    days = (datetime.now(timezone.utc) - attempt.opened_ts).total_seconds() / 86400.0
    state["attempts"] += 1
    state["total_trades"] = state.get("total_trades", 0) + attempt.total_entries
    state["total_wins"] = state.get("total_wins", 0) + attempt.wins
    state["total_losses"] = state.get("total_losses", 0) + attempt.losses
    if outcome == "pass":
        state["passes"] += 1
    else:
        state["fails"] += 1
    final_pct = round((final_equity - VIRTUAL_ACCOUNT) / VIRTUAL_ACCOUNT * 100, 2)
    state["history"].append({
        "outcome": outcome,
        "days": round(days, 2),
        "opened_ts": attempt.opened_ts.isoformat(),
        "resolved_ts": datetime.now(timezone.utc).isoformat(),
        "final_pct": final_pct,
        "final_usd": round(final_equity - VIRTUAL_ACCOUNT, 2),
        "trades": attempt.total_entries,
        "wins": attempt.wins,
        "losses": attempt.losses,
    })
    pass_rate = state["passes"] / state["attempts"] if state["attempts"] else 0.0
    log.warning(
        f"ATTEMPT #{state['attempts']} RESOLVED: {outcome.upper()} after {days:.1f} days "
        f"({final_pct:+.2f}%). Running: {state['passes']}/{state['attempts']} pass ({pass_rate:.1%})"
    )
    save_state(state)
    trade_win_rate = (state["total_wins"] / state["total_trades"]) if state["total_trades"] else 0.0
    outcome_emoji = "🎉✅" if outcome == "pass" else "💔❌"
    alert(
        f"{outcome_emoji} <b>ATTEMPT #{state['attempts']} {outcome.upper()}</b> "
        f"after {days:.1f} days ({final_pct:+.2f}%)\n"
        f"This attempt: {attempt.total_entries} trades ({attempt.wins}🟢/{attempt.losses}🔴)\n"
        f"🏆 Lifetime: {state['passes']}/{state['attempts']} passed ({pass_rate:.1%})  "
        f"{state['total_trades']} trades, {trade_win_rate:.1%} win rate\n"
        f"🔄 Starting a fresh $50,000 attempt.",
        book=BOOK,
        html=True,
    )



def mll_floor(peak_equity: float) -> float:
    """LucidPro's Max Loss Limit line: $2,000 below the highest END-OF-DAY
    balance, locked at $50,100 once that high reaches $52,100 (the initial trail
    balance) -- support.lucidtrading.com "LucidPro Drawdown". MT5 mode keeps
    the plain trailing line it was validated with."""
    floor = peak_equity - MLL_PCT * VIRTUAL_ACCOUNT
    return min(floor, VIRTUAL_ACCOUNT + 100.0) if FUTURES else floor

def size_futures_order(engine, attempt: Attempt, sym: str, spec, eq: float) -> int:
    """Whole contracts for a futures signal: the strategy's risk budget, cut to
    fit the room left above the Max Loss Limit (CLAMP_RISK_TO_MLL_ROOM) and
    Lucid's contract cap. 0 means skip the trade."""
    pv = engine.usd_per_point(sym)
    if not pv:
        log.warning(f"[{sym}] no point value from NinjaTrader -- skipping")
        return 0
    sl_dist = abs(spec.entry_price - spec.stop_loss)
    contracts = FUTURES_STACK.get(sym) or futures.contracts_for_risk(RISK_USD, sl_dist, pv)
    if CLAMP_RISK_TO_MLL_ROOM:
        room = eq - (mll_floor(attempt.peak_equity))
        per_contract = sl_dist * pv + 2 * futures.commission_per_side(sym)
        by_room = futures.contracts_for_risk(room * MLL_ROOM_FRACTION, per_contract, 1.0)
        if by_room < contracts:
            log.info(f"[{sym}] {contracts} contracts would risk ${contracts * per_contract:,.0f} with "
                     f"${room:,.0f} left above the MLL -- sizing {by_room}")
            contracts = by_room
    used = lucid_rules.open_units(engine.open_contracts())
    allowed = lucid_rules.max_addable(sym, used, LUCID_PLAN)
    if contracts > allowed:
        log.info(f"[{sym}] Lucid cap: {used}/{LUCID_PLAN.max_units} units in use -- "
                 f"{allowed} more {sym} allowed")
        contracts = allowed
    if contracts < 1:
        log.info(f"[{sym}] signal skipped -- under one contract after the risk and cap limits "
                 f"(stop {sl_dist:g} x ${pv:g}/point)")
    return contracts


def open_race(engine, attempt: Attempt, state: dict, day: str, magic: int) -> bool:
    """The day's single target-race entry: RACE.contracts long or short (the trend rule) with the
    take-profit exactly where the CLOSED balance reaches the evaluation target,
    and only the far safety stop. Sized on purpose past the MLL room (see
    target_race.py) -- so it bypasses size_futures_order's MLL clamp; the Lucid
    contract cap still applies inside the engine. Returns True once opened."""
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
    need = VIRTUAL_ACCOUNT * TARGET_PCT - attempt.realized_usd
    contracts = RACE.contracts
    if contracts > 1 and hasattr(engine, "bridge_version") and engine.bridge_version() < (1, 2):
        # VertexBridge 1.1 could lose the take-profit when a multi-contract entry
        # filled in parts (2026-10-01, ticket 1790856546515); 1 contract cannot
        # fill in parts. Same pass rate per try (~41%), just slower. Compile
        # bridge 1.2 in NinjaTrader (NinjaScript Editor, F5) to race full size.
        log.warning(f"[{sym}] VertexBridge older than 1.2 -- racing 1 contract instead of {contracts}")
        contracts = 1
    tp_pts = target_race.target_points(need, contracts, pv, futures.spec(sym)["tick_size"],
                                       futures.commission_per_side(sym))
    result = engine.place_market_order(
        symbol=sym, direction=direction, lots=contracts, sl=ask - side * RACE.stop_points, tp=ask + side * tp_pts,
        magic=magic, comment="lucid_race", signal_entry_price=ask,
    )
    if not result.ok:
        log.warning(f"[{sym}] race order failed retcode={result.retcode} {result.comment}")
        return False
    risk_usd = result.volume * abs(result.price - result.sl) * pv
    room = VIRTUAL_ACCOUNT + attempt.realized_usd - mll_floor(attempt.peak_equity)
    line = round(result.price - side * room / (result.volume * pv), 2)   # equity hits the MLL here
    attempt.tickets[result.ticket] = {
        "sym": sym, "entry_price": result.price, "sl": result.sl, "tp": result.tp, "line": line,
        "direction": direction, "side_reason": why_side,
        "lots": result.volume, "risk_usd": risk_usd, "opened_ts": datetime.now(timezone.utc).isoformat(),
    }
    attempt.entries_today += 1
    attempt.total_entries += 1
    state["race_day"] = day
    state["race_attempt"] = state.get("attempts", 0) + 1     # which attempt raced on race_day
    log.info(f"[{sym}] RACE opened {direction.upper()} ticket={result.ticket} {result.volume:g} @ {result.price:.2f} "
             f"target {result.tp:.2f} ({tp_pts:g} pts for ${need:,.0f}) safety stop {result.sl:.2f} -- {why_side}")
    alert(
        f"🏁 <b>RACE</b> {sym} {direction} x{result.volume:g} @ {result.price:.2f}\n"
        f"Target {result.tp:.2f} ({tp_pts:g} pts = the ${need:,.0f} still needed)  "
        f"safety stop {result.sl:.2f}  out {RACE.exit} ET\n"
        f"Rule reading: {RULE_READING}",
        book=BOOK,
        html=True,
    )
    return True


HISTORY_AUDIT_S = 60     # how often the open race is checked against 1-minute history


def audit_history(engine, attempt: Attempt, until_utc: datetime | None = None):
    """What a REAL Lucid account would have done while this bot was not looking.

    The bot sees prices only at its polls -- and not at all while the laptop
    sleeps. On 2026-10-01 a 47-minute sleep hid a drop that breached the MLL
    (open equity counts -- confirmed by Lucid support); the simulator kept the
    attempt alive. Lucid's server watches every tick, so replay the completed
    1-minute bars since the race opened (target_race.first_event) and return
    the first event:
      ("fail", line, when)  the low crossed the MLL line,
      ("pass", target, when) the high traded through the take-profit,
      ("exit", close, when)  16:00 ET came first -- the race's time exit.
    A bar that touches both the line and the target counts as the fail.
    until_utc: only the bars before that minute (checking a take-profit fill).
    Returns (outcome, price, when, ticket, info) or None."""
    import pandas as pd
    if not RACE_MODE or len(attempt.tickets) != 1:
        return None
    ticket, info = next(iter(attempt.tickets.items()))
    sym, long_ = info["sym"], info["direction"] == "long"
    pv = engine.usd_per_point(sym)
    tp = info.get("tp")
    if not tp:
        pos = engine.get_position(ticket)
        tp = float(pos.tp) if pos is not None and pos.tp else None
    if not pv or not tp:
        return None
    opened = datetime.fromisoformat(info["opened_ts"])
    now = datetime.now(timezone.utc)
    minutes = int((now - opened).total_seconds() // 60) + 3
    bars = engine.get_bars(sym, "1m", max(5, min(minutes, 3000))).iloc[:-1]     # completed bars only
    if until_utc is not None:
        bars = bars[bars.index < pd.Timestamp(until_utc).floor("min")]
    sign = 1.0 if long_ else -1.0
    room = VIRTUAL_ACCOUNT + attempt.realized_usd - mll_floor(attempt.peak_equity)
    line = info["entry_price"] - sign * room / (info["lots"] * pv)
    ev = target_race.first_event(bars, opened, line=line if RULE_READING == "lucid" else None, tp=float(tp),
                                 tick=futures.spec(sym)["tick_size"],
                                 exit_utc=target_race.exit_time(opened, RACE), long=long_)
    if ev is None:
        return None
    return ({"line": "fail", "target": "pass", "exit": "exit"}[ev[0]], ev[1], ev[2], ticket, info)


def settle_from_history(engine, attempt: Attempt, state: dict, event) -> None:
    """Book an event audit_history() found at the price Lucid would have used,
    close the simulator's position (at today's price -- the virtual attempt is
    what counts), and log the trade with the real exit price and time."""
    outcome, px, when, ticket, info = event
    sym = info["sym"]
    pv = engine.usd_per_point(sym) or futures.point_value(sym)
    engine.close_position(ticket)
    attempt.tickets.pop(ticket, None)
    sign = 1.0 if info["direction"] == "long" else -1.0
    lots = float(info["lots"])
    pnl = (px - info["entry_price"]) * sign * lots * pv - 2 * futures.commission_per_side(sym) * lots
    attempt.realized_usd += pnl
    if pnl > 0:
        attempt.wins += 1
    else:
        attempt.losses += 1
    risk = float(info.get("risk_usd") or 0.0)
    log_trade(state["attempts"] + 1, sym, info["direction"], info["entry_price"], px, lots, risk,
              pnl / risk if risk else 0.0, info.get("opened_ts", ""), when.isoformat(),
              {"tp": info.get("tp"), "line": info.get("line"), "reason": f"{outcome}_while_away"})
    what = {"fail": "the MLL line", "pass": "the target", "exit": "the 16:00 ET time exit"}[outcome]
    log.warning(f"[{sym}] price history: {what} was reached at {when:%Y-%m-%d %H:%M} UTC while the bot was not "
                f"watching (laptop asleep?) -- booked at {px:.2f}, ${pnl:+,.2f}")
    alert(f"🕒 <b>Missed while away</b>: {sym} reached {what} at {when:%H:%M} UTC -- booked at {px:.2f} "
          f"(${pnl:+,.0f}), as Lucid would have.", book=BOOK, html=True)


def main() -> int:
    # NAS100's real broker ticker is USTEC (confirmed live: "NAS100" isn't a
    # recognized symbol on this account) -- same mapping portfolio_yolo.yaml
    # already uses. Without this, is_tradeable() fails OPEN on the unknown
    # symbol (by design, so a probe outage never silently halts trading) and
    # every NAS100 order would then hit a real "unknown symbol" broker error.
    if STRATEGY and (not FUTURES or os.environ.get("VERTEX_PAPER", "").strip() != "1"):
        log.error(f"--strategy {STRATEGY} runs on paper only (VERTEX_PAPER=1, futures mode) -- not starting")
        return 2
    if FUTURES:
        # The contract cap is also enforced inside the engine, so no order
        # path can exceed it.
        if os.environ.get("VERTEX_PAPER", "").strip() == "1":
            # paper trading on NinjaTrader's free minute bars, fills simulated (owner 2026-10-15: NinjaTrader's
            # real-time data stopped being free) -- src/metatraderbot/paper_engine.py
            from metatraderbot.paper_engine import PaperEngine
            engine = PaperEngine(account=(os.environ.get("NT_ACCOUNT") or "Sim101").strip() + STRAT_TAG.upper(),
                                 max_contract_units=LUCID_PLAN.max_units)
        else:
            engine = NinjaTraderEngine(max_contract_units=LUCID_PLAN.max_units)
    else:
        engine = MT5Engine(symbol_map={"NAS100": "USTEC"})
    # Futures: NinjaTrader may not be open yet (the desktop icon can be clicked
    # first). Wait for it instead of exiting -- an exited bot stays dead until
    # the next launch. Same 60 s retry as the funded bot.
    while True:
        try:
            acc = engine.connect()
            break
        except Exception as e:  # noqa: BLE001 -- bridge down / outdated: keep waiting, say why
            if not FUTURES:
                raise
            log.warning(f"NinjaTrader not reachable yet ({e}) -- start NinjaTrader and connect it; "
                        f"retrying in 60s")
            time.sleep(60)
    log.warning(f"Connected: {'DEMO' if acc.is_demo else 'REAL'} account {acc.login}")
    live_lucid = False
    if not acc.is_demo:
        if not FUTURES:
            raise RuntimeError("REAL account detected -- refusing to run the trial bot on it.")
        # To NinjaTrader a Lucid evaluation is a real broker account. Trading
        # one takes an explicit opt-in naming that exact account, so a wrong
        # NT_ACCOUNT can never start trading an account by accident.
        if os.environ.get("LUCID_LIVE_CONFIRM", "").strip() != acc.login:
            raise RuntimeError(
                f"{acc.login} is not a NinjaTrader simulation account. To trade it, set "
                f"LUCID_LIVE_CONFIRM={acc.login} in .env -- only after the Sim101 run has been validated.")
        live_lucid = True
        log.warning(f"LIVE Lucid account {acc.login}: trading stops for good once this attempt passes or fails.")

    if RACE_MODE:
        strategies = {RACE.symbol: RACE}          # only its exit_due() is used by the loop
        log.warning(f"Strategy: TARGET RACE -- {RACE.contracts} {RACE.symbol} once a day ({'long above / short below the 200-day average' if RACE_SIDE_RULE == 'trend200' else 'always long'}), entries "
                    f"{RACE.entry}-{RACE.last_entry} ET, out {RACE.exit} ET, safety stop "
                    f"{RACE.stop_points:g} pts. Sim attempts judged with the {RULE_READING.upper()} "
                    f"rule reading.")
    elif FUTURES:
        from metatraderbot.lab_registry import instantiate
        fam, params, tp_r = FUTURES_STRATEGY
        strategies = {sym: instantiate(fam, {fam: params}, tp_r, sym) for sym in SYMBOLS}
    else:
        strategies = {sym: LondonBreakoutStrategy({
            "symbol": sym,
            "capital": {"risk_usd_per_trade": RISK_USD},
            "strategy": {"exit": {"tp_r_multiple": 2.0}},
        }) for sym in SYMBOLS}
    magics = {sym: SYMBOL_MAGICS[sym] for sym in SYMBOLS}

    state = load_state()
    # RESUME the in-flight attempt rather than starting a fresh one. Creating a
    # new Attempt() here was the actual bug behind "never fails": it wiped
    # realized_usd, the trailing peak and the open-ticket map on every restart,
    # so a losing attempt's drawdown was forgotten and the MLL could never be
    # reached. The watchdog restarts this process often enough that a losing
    # attempt essentially never survived to be judged.
    if state.get("attempt"):
        attempt = Attempt.from_dict(state["attempt"])
        resumed = True
    else:
        attempt = Attempt()
        resumed = False
    if state["current_attempt_start"] is None:
        state["current_attempt_start"] = attempt.opened_ts.isoformat()
    state["attempt"] = attempt.to_dict()
    save_state(state)
    log.warning(
        f"LucidPro trial runner started. {state['attempts']} prior attempts "
        f"({state['passes']} passed, {state['fails']} failed). "
        + (f"RESUMED attempt: realised ${attempt.realized_usd:+,.2f}, "
           f"peak ${attempt.peak_equity:,.2f}, {len(attempt.tickets)} open, "
           f"{attempt.total_entries} entries."
           if resumed else "Starting a fresh $50,000 attempt.")
    )
    if live_lucid and state.get("halted"):
        log.warning(f"Trading is HALTED on {acc.login} (attempt {state['halted']}). Positions are "
                    f"still managed; remove \"halted\" from {STATE_PATH.name} to trade again.")

    def halt_if_live(outcome: str) -> None:
        # A real evaluation: a PASS moves the account on -- stop for good. A
        # FAIL does not end the run (owner, 2026-09-30: "we stop only if we
        # pass"): Lucid breaches the account, so wait until it is reset to the
        # starting balance, then keep racing.
        if not live_lucid:
            return
        if outcome == "pass":
            state["halted"] = outcome
            save_state(state)
            alert(f"⏹️ <b>{acc.login}: evaluation PASSED</b> -- the bot has stopped "
                  f"opening trades on this account.", book=BOOK, html=True)
        else:
            state["awaiting_reset"] = True
            save_state(state)
            alert(f"🔁 <b>{acc.login}: attempt FAILED</b> -- reset the account in Lucid's dashboard "
                  f"(or buy a new evaluation and point NT_ACCOUNT / LUCID_LIVE_CONFIRM at it). "
                  f"The bot keeps running and races again once the balance is back at "
                  f"${VIRTUAL_ACCOUNT:,.0f}.", book=BOOK, html=True)

    last_bar_ts: dict[str, object] = {sym: None for sym in SYMBOLS}
    race_retry_at = 0.0          # monotonic time before which a failed race entry is not retried
    next_untracked = 0.0         # next check for a position no race holds
    next_audit_utc = datetime.min.replace(tzinfo=timezone.utc)   # next 1-minute history check of the open race

    _touch_heartbeat()
    threading.Thread(target=_self_watchdog, daemon=True, name="self-watchdog").start()

    while True:
        try:
            _touch_heartbeat()
            reconcile_closed_tickets(engine, attempt, state["attempts"] + 1, state)
            # Snapshot the attempt EVERY poll. A crash between polls must cost at
            # most one poll of progress, not the attempt's whole accumulated
            # drawdown -- losing that is what made this bot unable to fail.
            snap = attempt.to_dict()
            # Display-only: lets the dashboard show true attempt equity (and so
            # the real distance to the reset threshold) without opening its own
            # MT5 connection. Not restored by from_dict -- floating is derived
            # from live positions, never carried forward.
            try:
                snap["floating_usd"] = attempt.floating_usd(engine)
            except Exception:
                snap["floating_usd"] = 0.0
            state["attempt"] = snap
            save_state(state)

            now = datetime.now(timezone.utc)
            day = attempt.day_key(now)
            if attempt.current_day is None:
                attempt.current_day = day
            elif day != attempt.current_day:
                # New trading day: mark yesterday's EOD close against the
                # trailing peak, then reset the daily counters.
                eq_close = attempt.equity(engine)
                if eq_close <= mll_floor(attempt.peak_equity):
                    close_all(engine, attempt, state["attempts"] + 1)
                    resolve_attempt(state, attempt, "fail", eq_close)
                    attempt = Attempt()
                    state["current_attempt_start"] = attempt.opened_ts.isoformat()
                    state["attempt"] = attempt.to_dict()
                    save_state(state)
                    halt_if_live("fail")
                else:
                    attempt.peak_equity = max(attempt.peak_equity, eq_close)
                attempt.current_day = day
                attempt.entries_today = 0
                attempt.dll_blocked_today = False
                attempt.day_start_realized = attempt.realized_usd

            eq = attempt.equity(engine)
            balance = VIRTUAL_ACCOUNT + attempt.realized_usd
            # Futures, lenient reading: only the CLOSED balance can breach the
            # MLL (open positions count once they close, and at the day's end).
            on_balance = FUTURES and RULE_READING == "lenient"
            mll_basis = balance if on_balance else eq

            # --- what Lucid would have seen while we were not looking -----------
            if RACE_MODE and attempt.tickets and now >= next_audit_utc:
                next_audit_utc = now + timedelta(seconds=HISTORY_AUDIT_S)
                try:
                    event = audit_history(engine, attempt)
                except Exception as e:  # noqa: BLE001 -- no bars yet / bridge hiccup: try next time
                    log.debug(f"history audit skipped: {e!r}")
                    event = None
                if event is not None:
                    settle_from_history(engine, attempt, state, event)
                    outcome = event[0]
                    if outcome in ("pass", "fail"):
                        resolve_attempt(state, attempt, outcome, VIRTUAL_ACCOUNT + attempt.realized_usd)
                        attempt = Attempt()
                        state["current_attempt_start"] = attempt.opened_ts.isoformat()
                        halt_if_live(outcome)
                    state["attempt"] = attempt.to_dict()
                    save_state(state)
                    time.sleep(POLL_INTERVAL_S)
                    continue

            # --- MLL, checked EVERY POLL --------------------------------------
            #
            # This used to be checked ONLY on a day rollover, because LucidPro's
            # documented rule marks the max-loss limit at the daily close. The
            # PASS check, however, ran every poll. That asymmetry meant a winning
            # attempt resolved within seconds while a losing one had to survive
            # until the next EOD boundary -- and with the watchdog restarting
            # this process regularly, and the attempt not persisted, it almost
            # never got there. Result: 8 attempts, 8 passes, 0 fails.
            #
            # Checking continuously is STRICTER than the modelled product rule,
            # so live pass rates from here are not directly comparable to the
            # ~70% backtest figure, which assumed EOD-only marking. That is the
            # intended trade: an attempt that is down 4% and keeps trading is not
            # a realistic evaluation of anything.
            if mll_basis <= mll_floor(attempt.peak_equity):
                log.warning("MLL BREACH: %s $%.2f is $%.2f below the trailing "
                            "peak $%.2f (limit %.1f%% = $%.0f) -- failing this "
                            "attempt and starting a fresh $%.0f account.",
                            "balance" if on_balance else "equity",
                            mll_basis, attempt.peak_equity - mll_basis, attempt.peak_equity,
                            MLL_PCT * 100, MLL_PCT * VIRTUAL_ACCOUNT, VIRTUAL_ACCOUNT)
                close_all(engine, attempt, state["attempts"] + 1)
                resolve_attempt(state, attempt, "fail", eq)
                attempt = Attempt()
                state["current_attempt_start"] = attempt.opened_ts.isoformat()
                state["attempt"] = attempt.to_dict()
                save_state(state)
                halt_if_live("fail")
                time.sleep(POLL_INTERVAL_S)
                continue

            if not FUTURES:
                if eq <= attempt.peak_equity - DLL_PCT * VIRTUAL_ACCOUNT:
                    attempt.dll_blocked_today = True
            elif not attempt.dll_blocked_today:
                # Lucid's DLL: REALIZED losses since the day began (confirmed
                # by Lucid support 2026-09-30) ...
                day_pnl = attempt.realized_usd - attempt.day_start_realized
                if day_pnl <= -DLL_PCT * VIRTUAL_ACCOUNT:
                    attempt.dll_blocked_today = True
                    log.warning(f"DLL reached: realized day P&L ${day_pnl:+,.2f} -- Done for Day until the next "
                                f"session ({RULE_READING} reading)")
                    if RULE_READING == "lucid":          # ... then Lucid liquidates what is open
                        for ticket in list(attempt.tickets):
                            engine.close_position(ticket)
                            _touch_heartbeat()
            if eq >= VIRTUAL_ACCOUNT * (1 + TARGET_PCT):
                close_all(engine, attempt, state["attempts"] + 1)
                resolve_attempt(state, attempt, "pass", eq)
                attempt = Attempt()
                state["current_attempt_start"] = attempt.opened_ts.isoformat()
                state["attempt"] = attempt.to_dict()
                save_state(state)
                halt_if_live("pass")
                time.sleep(POLL_INTERVAL_S)
                continue

            entries_allowed = True
            if FUTURES:
                # Lucid closes anything still open at 4:45 PM ET (earlier on
                # holidays). Flatten on our own terms a few minutes before, and
                # open nothing new in the last 15 minutes of the day.
                session = lucid_rules.session_state(now)
                if session == lucid_rules.Session.FLATTEN and attempt.tickets:
                    log.warning(f"Lucid flat rule: closing {len(attempt.tickets)} position(s) "
                                f"before the {lucid_rules.flat_by(now.astimezone(lucid_rules.ET).date())} ET cutoff")
                    for ticket in list(attempt.tickets):
                        engine.close_position(ticket)
                        _touch_heartbeat()
                    # P&L is credited by reconcile_closed_tickets() next poll.
                # Still open from an EARLIER trading day: the bot was down through
                # the 4:45 PM ET flat window (the laptop was off). Lucid flattens
                # a real account itself; on the simulator, close it now.
                for ticket, info in list(attempt.tickets.items()):
                    try:
                        opened_day = lucid_rules.trading_day(datetime.fromisoformat(info["opened_ts"]))
                    except (KeyError, ValueError):
                        continue
                    if opened_day != day:
                        log.warning(f"[{info['sym']}] ticket {ticket} is from trading day {opened_day} -- "
                                    f"the bot missed the 4:45 PM ET flat; closing it now")
                        engine.close_position(ticket)
                        _touch_heartbeat()
                if live_lucid and state.get("awaiting_reset"):
                    # After a failed attempt: trade again only once Lucid has
                    # reset the account to its starting balance.
                    try:
                        live_balance = float(engine.account_info().balance)
                    except Exception as e:  # noqa: BLE001 -- bridge hiccup: just keep waiting
                        live_balance = 0.0
                        log.debug(f"account_info while awaiting reset: {e!r}")
                    if live_balance >= VIRTUAL_ACCOUNT - 1.0:
                        state.pop("awaiting_reset", None)
                        save_state(state)
                        log.warning(f"{acc.login} reset detected (balance ${live_balance:,.2f}) -- racing again")
                        alert(f"▶️ <b>{acc.login}</b> reset detected -- the race resumes.", book=BOOK, html=True)
                entries_allowed = (session == lucid_rules.Session.OPEN
                                   and not (live_lucid and (state.get("halted") or state.get("awaiting_reset"))))
                # The strategy's own time exit (ORB: 15:55 ET, as tested).
                for ticket, info in list(attempt.tickets.items()):
                    strat = strategies.get(info["sym"])
                    if strat is not None and hasattr(strat, "exit_due") and strat.exit_due(now):
                        engine.close_position(ticket)
                        _touch_heartbeat()

            if RACE_MODE and not attempt.tickets and time.monotonic() >= next_untracked:
                # a position no race holds is closed at once, not only before the next race (2026-10-14)
                next_untracked = time.monotonic() + target_race.LEFTOVER_RETRY_S
                try:
                    target_race.close_untracked(engine, log, lambda m: alert(m, book=BOOK, html=True))
                except Exception as e:  # noqa: BLE001 -- NinjaTrader busy: next time
                    log.warning(f"untracked-position check skipped: {e!r}")

            if RACE_MODE:
                # One race per ATTEMPT per trading day. A NEW attempt -- after a
                # fail (owner, 2026-10-01: "when one fails start another one
                # until it passes"), after a pass on the simulator, or after a
                # real account's reset -- races again the same day, inside the
                # usual entry hours.
                raced_today = (state.get("race_day") == day
                               and state.get("race_attempt", state["attempts"] + 1) == state["attempts"] + 1)
                if (live_lucid or ONE_RACE_A_DAY) and state.get("race_day") == day:
                    # A REAL account races once per trading day, at 19:00 ET -- also after a reset: restarts
                    # later in the day win less (~39% vs ~42-44%, research 2026-10-08), and the money plan
                    # (the Fleet page) assumes one race a day.
                    raced_today = True
                if (entries_allowed and not attempt.dll_blocked_today and not attempt.tickets
                        and not raced_today and RACE.entry_window(now)
                        and not target_race.skip_day(day) and time.monotonic() >= race_retry_at):
                    # an old position still open in NinjaTrader would stack under the new race: close it first
                    state["leftovers"] = target_race.close_leftovers(engine, attempt.tickets,
                                                                     state.get("leftovers") or [], log)
                    held = {} if state["leftovers"] else target_race.close_untracked(
                        engine, log, lambda m: alert(m, book=BOOK, html=True))
                    if state["leftovers"] or held:
                        race_retry_at = time.monotonic() + target_race.LEFTOVER_RETRY_S
                    elif not open_race(engine, attempt, state, day, magics[RACE.symbol]):
                        race_retry_at = time.monotonic() + 120
                    _touch_heartbeat()
            elif entries_allowed and not attempt.dll_blocked_today and attempt.entries_today < MAX_PER_DAY:
                for sym in SYMBOLS:
                    if len(attempt.tickets) >= MAX_CONCURRENT or attempt.entries_today >= MAX_PER_DAY:
                        break
                    if not engine.is_tradeable(sym):
                        continue
                    bars = engine.get_bars(sym, TIMEFRAME, max(BAR_COUNT, getattr(strategies[sym], "REQUIRED_BARS", 0)))
                    if last_bar_ts[sym] is not None and bars.index[-1] == last_bar_ts[sym]:
                        continue  # no new closed bar yet
                    last_bar_ts[sym] = bars.index[-1]
                    closed_bars = bars.iloc[:-1]  # drop the still-forming bar
                    if len(closed_bars) < 20:
                        continue
                    signals = strategies[sym].on_bar(len(closed_bars) - 1, closed_bars)
                    for spec in signals:
                        if len(attempt.tickets) >= MAX_CONCURRENT or attempt.entries_today >= MAX_PER_DAY:
                            break
                        # 50% risk on a $25,000 fixed base can compute a lot
                        # size past the broker's real max (confirmed live:
                        # US500 caps at 250 lots, and a typical stop distance
                        # there implies well beyond that). Clamp and log --
                        # a smaller real position means the live outcome for
                        # that trade is LESS aggressive than backtested, a
                        # known, expected source of divergence, not a bug to
                        # silently hide by letting the order just fail.
                        if FUTURES:
                            # Futures net per contract: one open ticket each.
                            if any(t["sym"] == sym for t in attempt.tickets.values()):
                                log.info(f"[{sym}] signal skipped -- a position is already open on it")
                                break
                            wanted = int(spec.lots + 1e-9)
                            lots = size_futures_order(engine, attempt, sym, spec, eq)
                            if lots < 1:
                                continue
                        else:
                            vol_min, vol_max, vol_step = engine.volume_spec(sym)
                            lots = round(spec.lots, 2)
                            wanted = spec.lots
                            if lots > vol_max:
                                log.warning(f"[{sym}] signal wants {lots} lots, broker max is {vol_max} "
                                            f"-- clamping (this trade will be under-risked vs backtest)")
                                lots = vol_max
                            if lots < vol_min:
                                log.info(f"[{sym}] signal wants {lots} lots, below broker min {vol_min} -- skipping")
                                continue
                        result = engine.place_market_order(
                            symbol=sym, direction=spec.direction.value, lots=lots,
                            sl=spec.stop_loss, tp=spec.take_profit, magic=magics[sym],
                            comment="lucidpro_trial", signal_entry_price=spec.entry_price,
                        )
                        if result.ok:
                            real_sl_dist = abs(result.price - result.sl)
                            mult = (engine.usd_per_point(sym) or futures.point_value(sym)) if FUTURES else contract_mult(sym)
                            real_risk_usd = result.volume * real_sl_dist * mult
                            attempt.tickets[result.ticket] = {
                                "sym": sym, "entry_price": result.price, "sl": result.sl,
                                "direction": spec.direction.value, "lots": result.volume,
                                "risk_usd": real_risk_usd,
                                "opened_ts": datetime.now(timezone.utc).isoformat(),
                            }
                            attempt.entries_today += 1
                            attempt.total_entries += 1
                            tag = " (CLAMPED)" if result.volume < wanted else ""
                            log.info(f"[{sym}] opened {spec.direction.value} ticket={result.ticket} "
                                     f"lots={result.volume} risk=${real_risk_usd:,.0f}{tag} "
                                     f"({len(attempt.tickets)}/{MAX_CONCURRENT} concurrent)")
                            dir_emoji = "🟢📈" if spec.direction.value == "long" else "🔴📉"
                            alert(
                                f"{dir_emoji} <b>OPENED</b> {sym} {spec.direction.value}{tag}\n"
                                f"lots={result.volume} @ {result.price:.5f}  risk=${real_risk_usd:,.0f}\n"
                                f"{len(attempt.tickets)}/{MAX_CONCURRENT} concurrent, "
                                f"{attempt.entries_today}/{MAX_PER_DAY} entries today",
                                book=BOOK,
                                html=True,
                            )
                        else:
                            log.warning(f"[{sym}] order failed retcode={result.retcode} {result.comment}")

            state["current_attempt_opened"] = attempt.entries_today
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.warning("Stopped by user.")
            return 0
        except Exception as e:  # noqa: BLE001 -- must not crash the loop
            log.error(f"Loop error (continuing): {e!r}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
