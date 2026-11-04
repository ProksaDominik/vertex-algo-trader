"""Portfolio runner — orchestrates multiple strategies concurrently on MT5.

Main loop:
1. For each strategy slot, poll MT5 for the latest closed bar.
2. If a new bar closed since last check, refresh the rolling window and call
   strategy.on_bar(). Strategies return TradeSpecs.
3. For each TradeSpec, place a market order via MT5 (magic number ties it to slot).
4. Every 60s, log a heartbeat + check kill switches.

Single-threaded polling loop — MT5 Python API is sync anyway, threads buy nothing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Type

import pandas as pd

from collections import deque
from dataclasses import asdict

from metatraderbot.alerts import alert
from metatraderbot.auto_retune import RetuneState, next_tp_r_multiple
from metatraderbot.ftmo_guard import FTMOGuard
from metatraderbot.instruments import PAIR_SPECS, contract_mult
from metatraderbot.ledger import Ledger
from metatraderbot.mt5_engine import MT5Engine, normalize_volume
from metatraderbot.ntp_check import ntp_offset_seconds
from metatraderbot.state import BotState
from metatraderbot.trade_review import SessionLearner, classify_session, review_trade
from metatraderbot.strategies_confluence import ConfluenceStrategy
from metatraderbot.strategies_bollinger import BollingerBounceStrategy
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy
from metatraderbot.strategies_ma_cross import SMACrossStrategy
from metatraderbot.strategies_macd import MACDTrendStrategy
from metatraderbot.strategies_bos import MarketOpenBOSStrategy
from metatraderbot.strategies_orb import ORBStrategy
from metatraderbot.strategies_rsi import RSIReversionStrategy
from metatraderbot.strategies_sweep_reclaim import SweepReclaimStrategy
from metatraderbot.strategies_tod_reversion import TODReversionStrategy
from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy
from metatraderbot.strategies_stochastic import StochasticStrategy
from metatraderbot.strategies_bos_pullback import TrendPullbackBOSStrategy
from metatraderbot.strategies_sr_retest import SRBreakoutRetestStrategy
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy
from metatraderbot.strategies_asian_fade import AsianFadeStrategy
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy
from metatraderbot.strategies_h4_pullback import H4PullbackStrategy
from metatraderbot.strategies_supertrend import SupertrendStrategy
from metatraderbot.strategies_smc_bos import SMCBOSStrategy
from metatraderbot.challenge_phase import evaluate as evaluate_phase
from metatraderbot.challenge_phase import next_phase
from metatraderbot.risk_ratchet import current_risk_pct as equity_ratchet_pct
from metatraderbot.strategy import Direction, NoWickStrategy, TradeSpec

log = logging.getLogger(__name__)

STRATEGY_CLASSES: dict[str, Type] = {
    # Confluence wraps other entries in this same registry, so it is
    # registered here alongside them and resolves its legs lazily at
    # construction time (see strategies_confluence._build_leg).
    "Confluence": ConfluenceStrategy,
    "NoWick": NoWickStrategy,
    "BollingerBounce": BollingerBounceStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "MACDTrend": MACDTrendStrategy,
    "RSIReversion": RSIReversionStrategy,
    "SMACross": SMACrossStrategy,
    "Stochastic": StochasticStrategy,
    "ORB": ORBStrategy,
    "MarketOpenBOS": MarketOpenBOSStrategy,
    "SweepReclaim": SweepReclaimStrategy,
    "TODReversion": TODReversionStrategy,
    "VWAPReversion": VWAPReversionStrategy,
    "TrendPullbackBOS": TrendPullbackBOSStrategy,
    "SRRetest": SRBreakoutRetestStrategy,
    "EMACross": EMACrossoverStrategy,
    "AsianFade": AsianFadeStrategy,
    "LondonBreakout": LondonBreakoutStrategy,
    "LiquiditySweep": SweepReclaimStrategy,
    "H4Pullback": H4PullbackStrategy,
    "Supertrend": SupertrendStrategy,
    "SMC_BOS": SMCBOSStrategy,
}

WARMUP_BARS = 300


class _MarketClosedRetry(Exception):
    """Raised by _execute_trade when the symbol isn't tradeable right now
    (broker rollover window). Caught by _process_slot so it holds the bar
    instead of advancing slot.last_bar_ts past it — otherwise the bar is
    marked seen regardless of outcome and the signal is lost forever the
    moment the market reopens (confirmed live: YOLO's only 3 signals since
    2026-08-05 all died this way, retcode 10018 "Market closed")."""


def expand_configs(configs: list) -> list:
    """Expand each per-strategy entry into one LEG per symbol.

    A strategy is now written ONCE with a `symbols:` list and a `magic_base:`
    (e.g. Donchian trading a basket) instead of copy-pasting a slot per pair.
    Each leg gets a unique name `Strategy__SYMBOL`, its own magic
    (magic_base + index), and carries `strategy_group` = the strategy's name
    so the dashboard and comparisons aggregate by STRATEGY, not by pair.
    Legacy single-`symbol` entries pass through unchanged."""
    out = []
    for entry in configs:
        syms = entry.get("symbols")
        if syms:
            base = int(entry.get("magic_base", 0))
            for i, sym in enumerate(syms):
                leg = {k: v for k, v in entry.items() if k not in ("symbols", "magic_base")}
                leg["symbol"] = sym
                leg["magic_number"] = base + i
                leg["name"] = f"{entry.get('name', 'strat')}__{sym}"
                leg["strategy_group"] = entry.get("name", "")
                out.append(leg)
        else:
            leg = dict(entry)
            leg.setdefault("strategy_group", entry.get("name", ""))
            out.append(leg)
    return out


def validate_portfolio_cfg(cfg: dict) -> list[str]:
    """Hard config errors that must refuse startup (AUDIT M6). Returns a
    list of problems; empty list = safe to start."""
    errors: list[str] = []
    cap = cfg.get("capital") or {}
    raw = cfg.get("configs") or []
    for e in raw:
        if e.get("symbols") and e.get("magic_base") is None:
            errors.append(f"{e.get('name','?')}: has symbols[] but no magic_base")
    configs = expand_configs(raw)

    if not configs:
        errors.append("configs: empty — nothing to trade")
    if float(cap.get("risk_usd_per_trade", 0)) <= 0:
        errors.append("capital.risk_usd_per_trade must be > 0")
    if float(cap.get("daily_stop_loss_usd", -1)) >= 0:
        errors.append("capital.daily_stop_loss_usd must be negative (it's a loss level)")
    if float(cap.get("max_total_drawdown_usd", -1)) >= 0:
        errors.append("capital.max_total_drawdown_usd must be negative")
    if int(cap.get("max_positions_per_config", 1)) <= 0:
        errors.append("capital.max_positions_per_config must be >= 1")

    seen_magic: dict[int, str] = {}
    seen_name: set[str] = set()
    for entry in configs:
        name = entry.get("name", "<unnamed>")
        if name in seen_name:
            errors.append(f"duplicate config name: {name}")
        seen_name.add(name)
        magic = entry.get("magic_number")
        if magic is None:
            errors.append(f"{name}: missing magic_number")
        elif magic in seen_magic:
            errors.append(f"{name}: magic_number {magic} already used by {seen_magic[magic]}")
        else:
            seen_magic[int(magic)] = name
        strat = entry.get("strategy")
        if strat not in STRATEGY_CLASSES:
            errors.append(f"{name}: unknown strategy {strat!r}")
        sym = entry.get("symbol", "")
        if sym not in PAIR_SPECS:
            errors.append(
                f"{name}: symbol {sym!r} missing from instruments.PAIR_SPECS "
                f"(silent fallback removed — AUDIT C5)"
            )
        params = entry.get("strategy_params") or {}
        ex = params.get("exit") or {}
        tp = float(ex.get("tp_r_multiple", 1.0))
        if tp <= 0:
            errors.append(f"{name}: exit.tp_r_multiple must be > 0")
        # Effective trail = this leg's override if set, else the book default.
        # Reversion legs deliberately set trail_start_r: 0 (tight TP, no trail).
        trail_start = float(ex["trail_start_r"]) if "trail_start_r" in ex \
            else float(cap.get("trail_start_r", 0.0))
        if trail_start > 0 and tp <= trail_start:
            errors.append(
                f"{name}: tp_r_multiple ({tp}) <= effective trail_start_r ({trail_start}) — "
                f"TP would always fire before the trail can engage; fix one of them"
            )
    return errors


def _currencies(symbol: str) -> tuple[str, ...]:
    """Best-effort currency codes touched by a symbol, for cross-slot
    correlated-exposure tracking. A 6-letter alphabetic symbol (EURUSD,
    GBPJPY, XAUUSD — gold genuinely shares USD exposure with other USD
    pairs the same way two JPY pairs share JPY exposure) splits into
    base+quote; anything else (US30, DE40 — index symbols with digits in
    the name) is its own standalone bucket since it doesn't share currency
    exposure with FX pairs."""
    if len(symbol) == 6 and symbol.isalpha():
        return (symbol[:3], symbol[3:])
    return (symbol,)


@dataclass
class TrackedPosition:
    entry: float
    orig_sl: float
    orig_tp: float
    lots: float
    is_long: bool
    opened_ts: str
    # How many ticks we've tried (and failed) to find this position's exit
    # deal in history after it disappeared from open positions. Deal history
    # often lags a close by a moment — the old code popped the position on
    # the FIRST miss and the trade was never ledgered (AUDIT H4).
    reconcile_attempts: int = 0
    # Phase 4 exit-management state.
    partial_done: bool = False
    adds: int = 0


@dataclass
class SlotHealth:
    """Self-improving layer: rolling live performance per slot.

    Watches realized R-multiples over a sliding window. When a slot's
    recent net result degrades past the configured threshold, the slot is
    QUARANTINED — no new entries — for a cooldown period, then allowed to
    try again. This adapts the portfolio toward what's currently working
    WITHOUT mutating strategy parameters live (that would be curve-fitting
    the account in real time). Rebuilt from the ledger on every start."""
    recent_r: deque = None
    quarantined_until: datetime | None = None

    def __post_init__(self):
        if self.recent_r is None:
            self.recent_r = deque(maxlen=50)


@dataclass
class StrategySlot:
    name: str
    symbol: str
    strategy_name: str
    timeframe: str
    magic: int
    strategy: object
    last_bar_ts: pd.Timestamp
    tracked: dict[int, TrackedPosition] = None
    # Which STRATEGY this leg belongs to (for grouping N per-symbol legs of
    # one strategy in the dashboard / comparisons).
    strategy_group: str = ""
    # Per-strategy exit overrides (None -> fall back to the book-level value).
    # This is how each strategy gets the exit that best fits its idea:
    # trend strategies run a wide TP + 1R trail; reversion strategies take a
    # tight TP with no trail. Set via strategy_params.exit in the config.
    trail_start_r: float | None = None
    trail_lock_fraction: float | None = None
    skyrocket_multiple: float | None = None
    # Hard time stop: close the position after this many days regardless of
    # where price is. scripts/timebox_strategy_test.py found this keeps RSI
    # positive in BOTH halves while cutting median hold from 18 days to 1-2,
    # which is what makes a 10-30 day challenge winnable at all -- with
    # 18-day holds the book only cycles 4-6 trades/month and cannot compound
    # to +10%. None = no time stop (the original behaviour).
    max_hold_days: float | None = None

    def __post_init__(self):
        if self.tracked is None:
            self.tracked = {}


class PortfolioRunner:
    def __init__(self, portfolio_cfg: dict, engine: MT5Engine, ledger: Ledger,
                 control_path: Path | None = None, state: BotState | None = None,
                 book_name: str = "bot"):
        self.cfg = portfolio_cfg
        self.engine = engine
        self.ledger = ledger
        self.book_name = book_name
        self.state = state if state is not None else BotState(Path("data") / f"state_{book_name}.json")
        # Shared start/stop control file the dashboard writes to. When
        # trading_enabled is false, new signals are still detected (keeps
        # strategy state — e.g. NoWick's pending setups — correct) but no
        # new orders are placed; existing open positions keep being managed
        # normally (reconciliation, trailing stop, skyrocket exit all still
        # run). Missing/unreadable file defaults to enabled.
        self.control_path = control_path

        cap = portfolio_cfg["capital"]
        self.risk_usd = float(cap["risk_usd_per_trade"])
        self.daily_stop_usd = float(cap["daily_stop_loss_usd"])
        # What the daily stop DOES when it trips:
        #   "flatten" (default, legacy) -- close every open position, then halt
        #   "block"                     -- stop NEW entries, let open trades run
        # "block" is the behaviour validated in scripts/day_filter_test.py:
        # that test dropped subsequent ENTRIES on a losing day, it never closed
        # live positions early. On a 4-5R book whose winners need days to
        # reach target, flattening at the moment a day turns red would cut off
        # precisely the trades the edge depends on.
        self.daily_stop_action = str(cap.get("daily_stop_action", "flatten")).lower()
        # Which FTMO evaluation phase this book is in, and how hard to cut
        # risk once the phase target is banked but the 4-trading-day
        # requirement is not yet met (see challenge_phase.PhaseStatus.banked).
        fg_cfg = portfolio_cfg.get("ftmo") or {}
        self.challenge_phase = str(fg_cfg.get("phase", "challenge")).lower()
        self.post_target_risk_frac = float(fg_cfg.get("post_target_risk_frac", 0.1))
        self.max_dd_usd = float(cap.get("max_total_drawdown_usd", -1000))
        self.max_positions = int(cap["max_positions_per_config"])
        # Ratchet trailing stop: once profit reaches trail_start_r x the
        # original risk distance (R), lock in trail_lock_fraction of whatever
        # profit has been reached so far by moving SL forward — re-evaluated
        # every tick, so it keeps ratcheting as price advances further, and
        # never loosens. Lets a trade close (with a smaller but real profit)
        # on a pullback instead of needing to round-trip all the way back to
        # the original SL, or sit open for a long time waiting for a distant
        # fixed TP to be hit exactly. 0 (default) disables it.
        self.trail_start_r = float(cap.get("trail_start_r", 0.0))
        self.trail_lock_fraction = float(cap.get("trail_lock_fraction", 0.5))
        # Close early (at market) when the current candle's range is this many
        # times the recent average candle range AND it's moving favorably AND
        # the trade is in profit — captures an outsized spike instead of
        # waiting for a (possibly much smaller) fixed TP or letting it
        # round-trip back down. 0 (default) disables it.
        self.skyrocket_multiple = float(cap.get("skyrocket_multiple", 0.0))
        # Cap on how much risk_usd can be exposed to any single currency at
        # once, summed across EVERY slot's open positions — not just the
        # current one. max_positions_per_config only limits one slot at a
        # time; it does nothing to stop five different slots (different
        # strategies, different timeframes) from all independently holding
        # JPY or USD exposure simultaneously, which is exactly what a single
        # broad currency move (seen live more than once) hits all at once.
        # In R-multiples of risk_usd_per_trade; 0 (default) disables it.
        self.max_currency_exposure_r = float(cap.get("max_currency_exposure_r", 0.0))
        # Portfolio-wide caps (AUDIT H2). Approximates each open position as
        # one R of risk. 0 disables. Without these, 19 slots x 5 positions
        # x risk_usd could sit open at once — $9,500 of theoretical open
        # risk on the FTMO plan.
        self.max_total_open_risk_r = float(cap.get("max_total_open_risk_r", 0.0))
        self.max_concurrent_positions_total = int(cap.get("max_concurrent_positions_total", 0))
        # K11 (CLAUDE.md kill switches §11.1): "trade_count_today >= N -> HALT
        # new entries until rollover". Incremented at order-SEND, not close
        # (a rejected order doesn't count). 0 disables. This is what makes a
        # high-frequency config arithmetically safe: N x risk_pct must stay
        # under the daily loss limit even if every trade that day loses.
        self.max_trades_per_day = int(cap.get("max_trades_per_day", 0))
        # Weekdays on which no NEW entry may open (Mon=0 .. Sun=6). Existing
        # positions are still managed normally. Default [] = trade every day,
        # so the behaviour is unchanged unless a config opts in.
        self.skip_weekdays = {int(d) for d in (cap.get("skip_weekdays") or [])}
        # Replay-gap staleness gate (AUDIT H6): a signal from a bar older
        # than this many closed bars is state-only — strategies still see the
        # bar, but no order fires. A breakout from hours ago has no edge left.
        self.max_signal_age_bars = int(cap.get("max_signal_age_bars", 2))
        # When ideal lots < broker volume_min: "skip" keeps risk honest
        # (default), "floor" trades volume_min anyway and accepts the
        # inflated risk (AUDIT H7 — the old code always floored, silently).
        self.min_lot_policy = str(cap.get("min_lot_policy", "skip")).lower()
        # Above this, a live-vs-static contract-multiplier disagreement is
        # treated as a broker data bug, not a legitimate live update — the
        # signal is skipped instead of sized off the disputed number. See
        # the drift check in _execute_trade for the XAUUSD incident that
        # motivated this (10x oversized real risk from a misreported
        # trade_tick_value, silently accepted under the old "always use live"
        # policy). 0.25-0.50 drift still just warns and proceeds.
        self.max_contract_mult_drift = float(cap.get("max_contract_mult_drift", 0.50))

        # --- CLAUDE.md kill switches (spec §11.1), the ones without an
        # existing equivalent mechanism in this codebase. 0/False disables
        # each; account_size_usd (below) is what percentage-based ones are
        # measured against. ---
        # K2: weekly_pnl <= -weekly_stop_pct -> HALT new entries until Monday
        # (Prague-day week boundary, matching the existing daily anchor's
        # timezone). Unlike K1 (the existing daily guard), this does NOT
        # flatten open positions — the spec says "HALT", not "flatten_halt".
        self.weekly_stop_pct = float(cap.get("weekly_stop_pct", 0.0))
        # K4: 4 consecutive losses -> risk x0.5 until 2 wins. Tracked book-
        # wide (across all slots), not per-slot — a string of losses on
        # DIFFERENT strategies is still a sign something's off with current
        # conditions, not just one bad slot (that's what slot_health/
        # quarantine already handles separately).
        self.k4_consecutive_loss_n = int(cap.get("k4_consecutive_loss_trigger", 4))
        self.k4_recovery_wins_needed = int(cap.get("k4_recovery_wins_needed", 2))
        self.k4_enabled = bool(cap.get("k4_consecutive_loss_halving_enabled", False))
        # K6: spread > k6_spread_multiple x its recent rolling median for
        # >= k6_persist_s seconds -> HALT NEW ENTRIES for that symbol only.
        self.k6_enabled = bool(cap.get("k6_spread_anomaly_enabled", False))
        self.k6_spread_multiple = float(cap.get("k6_spread_multiple", 5.0))
        self.k6_persist_s = float(cap.get("k6_persist_s", 30.0))
        self._spread_history: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(ts, spread)]
        self._k6_bad_since: dict[str, float] = {}  # symbol -> monotonic ts anomaly started
        # K8: slippage > k8_slippage_multiple x the modeled/assumed slippage
        # (instruments.PAIR_SPECS spread_price, the same static estimate the
        # backtester uses), sustained over the last k8_window trades -> HALT,
        # recalibrate (manual — this never auto-resumes, matching the spec's
        # "recalibrate" framing: the assumption itself needs revisiting).
        self.k8_enabled = bool(cap.get("k8_slippage_recalibration_enabled", False))
        self.k8_window = int(cap.get("k8_window_trades", 20))
        self.k8_multiple = float(cap.get("k8_slippage_multiple", 2.0))
        self._slippage_history: list[float] = []  # price units, most recent last
        # K9: local machine clock drift from NTP truth > k9_max_drift_s ->
        # HALT. Checked at most once per k9_check_interval_s (a real NTP
        # query every tick would be wasteful and could itself be blocked by
        # a flaky network — fails OPEN on an unreachable NTP server, since a
        # network hiccup reaching pool.ntp.org says nothing about whether
        # THIS machine's clock is actually wrong).
        self.k9_enabled = bool(cap.get("k9_ntp_drift_enabled", False))
        self.k9_max_drift_s = float(cap.get("k9_max_drift_s", 5.0))
        self.k9_check_interval_s = float(cap.get("k9_check_interval_s", 300.0))
        self._k9_last_check = 0.0
        # K10: a new order's lot size > k10_multiple x the rolling median of
        # the last k10_window FILLED orders (book-wide) -> reject the order,
        # HALT, alert.
        self.k10_enabled = bool(cap.get("k10_lot_anomaly_enabled", False))
        self.k10_window = int(cap.get("k10_window_trades", 20))
        self.k10_multiple = float(cap.get("k10_lot_multiple", 3.0))
        # Cost gate (spec §1.1): reject a signal where
        # (spread + commission + slippage) / stop_distance > cost_gate_r.
        # Uses the SAME static per-symbol spread_price/commission_per_lot
        # instruments.py already ships (documented there as "close enough
        # for strategy selection" -- this project has no live-tick-based
        # spread feed wired into the signal path yet).
        self.cost_gate_enabled = bool(cap.get("cost_gate_enabled", False))
        self.cost_gate_r = float(cap.get("cost_gate_r", 0.08))
        # Self-improving slot health / quarantine.
        sh = cap.get("slot_health") or {}
        self.slot_health_enabled = bool(sh.get("enabled", True))
        self.slot_health_window = int(sh.get("window_trades", 8))
        self.slot_health_min_net_r = float(sh.get("quarantine_below_net_r", -4.0))
        self.slot_health_cooldown_h = float(sh.get("cooldown_hours", 24.0))
        self._health: dict[str, SlotHealth] = {}

        # --- Phase 4: sizing modes + circuit breaker + exit extensions ---
        # risk_mode "fixed_usd" (default) or "percent_equity": risk_pct of
        # (account_size_usd + realized book P&L) — fixed-fraction sizing that
        # compounds with the book, not the shared broker account (AUDIT C4).
        self.risk_mode = str(cap.get("risk_mode", "fixed_usd")).lower()
        self.risk_pct = float(cap.get("risk_pct", 3.0))
        self.account_size = float(cap.get("account_size_usd", 0.0))
        cb = cap.get("circuit_breaker") or {}
        self.breaker_enabled = bool(cb.get("enabled", False))
        self.breaker_dd_pct = float(cb.get("halt_drawdown_pct", 35.0))
        pt = cap.get("partial_take") or {}
        self.partial_enabled = bool(pt.get("enabled", False))
        self.partial_at_r = float(pt.get("at_r", 1.5))
        self.partial_fraction = float(pt.get("fraction", 0.5))
        py = cap.get("pyramid") or {}
        self.pyramid_enabled = bool(py.get("enabled", False))
        self.pyramid_trigger_r = float(py.get("trigger_r", 1.0))
        self.pyramid_max_adds = int(py.get("max_adds", 1))
        self.pyramid_add_fraction = float(py.get("add_fraction", 0.5))

        # --- Phase 3: FTMO guard ---
        fg = portfolio_cfg.get("ftmo") or {}
        self.guard: FTMOGuard | None = None
        if bool(fg.get("enabled", False)):
            self.guard = FTMOGuard(
                account_size=float(fg.get("account_size_usd", 10_000)),
                state=self.state,
                scope=str(fg.get("scope", "book")),
                daily_trigger_pct=float(fg.get("daily_trigger_pct", 3.5)),
                overall_trigger_pct=float(fg.get("overall_trigger_pct", 8.0)),
                target_pct=float(fg.get("target_pct", 10.0)),
                target_action=str(fg.get("target_action", "alert")),
                news_blackout=fg.get("news_blackout") or [],
                weekend_flat=bool(fg.get("weekend_flat", True)),
                weekend_close_utc=str(fg.get("weekend_close_utc", "20:30")),
            )

        # --- Self-improving: bounded learning from every closed trade ---
        # Each trade gets a stored post-mortem (why it won/lost); aggregated
        # per (slot, session) it can stand a slot down during a session where
        # it has repeatedly lost. Never mutates SL/TP/size (that's curve
        # fitting). Rebuilt from the ledger on start.
        lr = cap.get("learning") or {}
        self.learning_enabled = bool(lr.get("enabled", True))
        self.learn_min_trades = int(lr.get("session_min_trades", 6))
        self.learn_avoid_below_net_r = float(lr.get("session_avoid_below_net_r", -3.0))
        self.learner = SessionLearner(self.learn_min_trades, self.learn_avoid_below_net_r)

        # --- Live strategy auto-retuning (see auto_retune.py) ---
        # UNLIKE everything above, this one DOES mutate a strategy's own exit
        # parameter (tp_r_multiple) from recent live results, with no human
        # review — explicitly the "curve-fitting the account live" risk
        # avoided everywhere else in this file. Off by default; only on
        # because it was explicitly requested. Bounded (floor + max count).
        ar = cap.get("auto_retune") or {}
        self.auto_retune_enabled = bool(ar.get("enabled", False))
        self.auto_retune_step_pct = float(ar.get("step_pct", 0.20))
        self.auto_retune_floor_frac = float(ar.get("floor_frac", 0.30))
        self.auto_retune_max_retunes = int(ar.get("max_retunes", 5))
        self._retune: dict[str, RetuneState] = {}

        # --- Adaptive (path-dependent) risk taper ---
        # Validated in scratch/ftmo_4day_adaptive_risk.py via real historical
        # replay: sizing by distance to the NEAREST barrier (the +target% or
        # the -loss% limit), not a flat risk%, measurably improves both pass
        # rate AND fail rate simultaneously versus flat sizing at the same
        # base risk — full size while there's room on both sides, tapering
        # toward floor_frac of base as equity nears either one. Multiplies
        # whatever _current_risk_usd() already computed from risk_mode; off
        # by default so existing books (flat sizing) are unaffected.
        adr = cap.get("adaptive_risk") or {}
        self.adaptive_risk_enabled = bool(adr.get("enabled", False))
        self.adaptive_risk_mode = str(adr.get("mode", "symmetric")).lower()
        self.adaptive_risk_band_pct = float(adr.get("band_pct", 10.0))
        self.adaptive_risk_floor = float(adr.get("floor_frac", 0.15))
        self.adaptive_risk_target_pct = float(adr.get("target_pct", 10.0))
        self.adaptive_risk_loss_pct = float(adr.get("loss_pct", 10.0))

        # --- Phase 2: watchdog + fail-closed account probe ---
        wd = portfolio_cfg.get("watchdog") or {}
        self.watchdog_enabled = bool(wd.get("enabled", True))
        self.stale_tick_s = float(wd.get("stale_tick_s", 300.0))
        self._entries_blocked: str | None = None
        self._watchdog_degraded: str | None = None
        self._acct_fail_count = 0
        self._realized_alltime = 0.0
        self._floating = 0.0

        self.poll_interval_s = int(
            portfolio_cfg.get("mt5", {}).get("poll_interval_s", 5)
        )

        self.slots: list[StrategySlot] = []
        self.starting_equity: float = 0.0
        self._stopped = False

    def initialize(self) -> None:
        # Refuse to start on an unsafe/contradictory config (AUDIT M6).
        problems = validate_portfolio_cfg(self.cfg)
        if problems:
            for p in problems:
                log.error(f"CONFIG ERROR: {p}")
            raise ValueError(
                f"Refusing to start: {len(problems)} config problem(s) — see log above."
            )

        acc = self.engine.connect()
        self.starting_equity = acc.equity
        mode = "DEMO" if acc.is_demo else "REAL"
        log.info(
            f"MT5 connected — account {acc.login} ({acc.server}) "
            f"[{mode}] equity={acc.equity:.2f} {acc.currency}"
        )
        if not acc.is_demo:
            log.warning(
                "*** REAL account detected. This bot will place real trades. "
                "Set is_demo check before continuing if unintended. ***"
            )
        # Netting accounts merge same-symbol positions into one ticket with a
        # shared SL/TP — with multiple slots (and two bots) on the same
        # symbols, per-magic tracking and independent exits fall apart
        # (AUDIT M1). .env.example even documents this account as "Netting",
        # while config comments claim hedging — so verify at runtime.
        if not acc.is_hedging and not bool(self.cfg.get("mt5", {}).get("allow_netting", False)):
            raise RuntimeError(
                f"Account margin_mode={acc.margin_mode} is NOT retail-hedging. "
                f"This portfolio runs multiple slots per symbol and requires "
                f"hedging mode. Set mt5.allow_netting: true to override (unsafe)."
            )

        # Persist the overall-drawdown anchor ONCE, on first ever run —
        # restarts must never re-anchor it (AUDIT C2).
        if self.state.get("initial_balance_anchor") is None:
            self.state.set("initial_balance_anchor", acc.balance)
            self.state.set("anchor_captured_ts", datetime.now(timezone.utc).isoformat())
            self.state.save()
            log.info(f"Initial balance anchor captured: {acc.balance:.2f} (persisted)")

        # Slots can be disabled in the YAML (enabled: false) without losing
        # their config — Phase 5 walk-forward benched the slots that lose
        # out-of-sample; re-enable only with new evidence.
        expanded_configs = expand_configs(self.cfg["configs"])
        active_configs = [e for e in expanded_configs if e.get("enabled", True)]
        skipped = len(expanded_configs) - len(active_configs)
        if skipped:
            log.info(f"{skipped} slot(s) disabled in config — {len(active_configs)} active")

        # Seed the self-improving slot-health monitor from the ledger so
        # quarantine decisions survive restarts, and restore any persisted
        # quarantine benches.
        if self.slot_health_enabled:
            saved_q = self.state.get("quarantine") or {}
            for entry in active_configs:
                h = SlotHealth()
                pnls = self.ledger.recent_closed_pnls(entry["name"], self.slot_health_window)
                for pnl in reversed(pnls):  # oldest first into the deque
                    h.recent_r.append(pnl / self.risk_usd)
                until = saved_q.get(entry["name"])
                if until:
                    try:
                        h.quarantined_until = datetime.fromisoformat(until)
                    except ValueError:
                        pass
                self._health[entry["name"]] = h

        # Rebuild the learned session memory from stored trade reviews.
        if self.learning_enabled:
            for cfg_name, session, r in self.ledger.all_reviews_for_learner():
                self.learner.observe(cfg_name, session, float(r))

        # Restore any persisted auto-retunes so a restart doesn't silently
        # revert a slot back to its original (pre-retune) exit parameter.
        saved_rt = self.state.get("auto_retune") or {}
        if self.auto_retune_enabled:
            for entry in active_configs:
                rt = saved_rt.get(entry["name"])
                if rt:
                    self._retune[entry["name"]] = RetuneState(
                        original_tp_r=float(rt["original_tp_r"]),
                        current_tp_r=float(rt["current_tp_r"]),
                        retune_count=int(rt["retune_count"]),
                    )

        for entry in active_configs:
            slot = self._build_slot(entry)
            rt = self._retune.get(entry["name"])
            if rt is not None and hasattr(slot.strategy, "tp_r_multiple"):
                slot.strategy.tp_r_multiple = rt.current_tp_r
                log.info(
                    f"  [{entry['name']}] restored auto-retuned tp_r_multiple="
                    f"{rt.current_tp_r:.3f} (retune #{rt.retune_count}, original {rt.original_tp_r:.3f})"
                )
            self.slots.append(slot)
            log.info(
                f"  slot ready: {entry['name']} = {entry['symbol']} "
                f"{entry['strategy']} {entry['timeframe']} magic={entry['magic_number']}"
            )

        # Restore persisted pending setups (AUDIT M8) — stateful strategies
        # otherwise forget half-formed setups across a restart. Generic over
        # whichever dataclass a strategy's `pending` list holds (declared via
        # PENDING_CLS on the strategy class) rather than hardcoding one
        # strategy's shape — `_persist_runtime_state` serializes with a
        # plain `asdict(s)`, so reconstruction is just the inverse.
        saved_pending = self.state.get("pending_signals") or {}
        for slot in self.slots:
            if not hasattr(slot.strategy, "pending"):
                continue
            pending_cls = getattr(type(slot.strategy), "PENDING_CLS", None)
            if pending_cls is None:
                continue
            for raw in saved_pending.get(slot.name, []):
                try:
                    kwargs = dict(raw)
                    kwargs["direction"] = Direction(kwargs["direction"])
                    slot.strategy.pending.append(pending_cls(**kwargs))
                except (KeyError, ValueError, TypeError) as e:
                    log.warning(f"[{slot.name}] could not restore pending signal: {e}")
            if slot.strategy.pending:
                log.info(f"[{slot.name}] restored {len(slot.strategy.pending)} pending setup(s)")

        if self.state.halted:
            halt = self.state.get("halt") or {}
            log.error(
                f"STARTING IN HALTED STATE — scope={halt.get('scope')} reason={halt.get('reason')}. "
                f"Daily halts clear at the next CE(S)T day; total/breaker halts need a manual "
                f"state reset (delete the 'halt' block in the state file)."
            )

    def _build_slot(self, entry: dict) -> StrategySlot:
        strategy_cls = STRATEGY_CLASSES[entry["strategy"]]
        params = entry.get("strategy_params", {}) or {}
        strat_cfg = self._build_strategy_config(entry["symbol"], params)
        strategy = strategy_cls(strat_cfg)

        # Warm up with recent bars (drop the last, which may be forming)
        raw = self.engine.get_bars(entry["symbol"], entry["timeframe"], WARMUP_BARS + 1)
        closed = raw.iloc[:-1]
        last_ts = closed.index[-1] if len(closed) else pd.Timestamp.min.tz_localize("UTC")

        ex = params.get("exit", {}) or {}
        return StrategySlot(
            name=entry["name"],
            symbol=entry["symbol"],
            strategy_name=entry["strategy"],
            timeframe=entry["timeframe"],
            magic=int(entry["magic_number"]),
            strategy=strategy,
            last_bar_ts=last_ts,
            strategy_group=entry.get("strategy_group", entry["name"]),
            trail_start_r=(float(ex["trail_start_r"]) if "trail_start_r" in ex else None),
            trail_lock_fraction=(float(ex["trail_lock_fraction"]) if "trail_lock_fraction" in ex else None),
            max_hold_days=(float(ex["max_hold_days"]) if "max_hold_days" in ex else None),
            skyrocket_multiple=(float(ex["skyrocket_multiple"]) if "skyrocket_multiple" in ex else None),
        )

    def _build_strategy_config(self, symbol: str, params: dict) -> dict:
        """Construct a config dict shaped for the strategy classes' expectations.
        Fills in defaults for any keys the user didn't specify in portfolio.yaml.
        """
        cfg = {
            "symbol": symbol,
            "capital": {"risk_usd_per_trade": self.risk_usd},
            "strategy": {
                "trend": params.get("trend", {}) or {},
                "signal": params.get("signal", {}) or {},
                "entry": params.get("entry", {}) or {},
                "exit": params.get("exit", {}) or {},
                "session": params.get("session"),
                "bollinger": params.get("bollinger", {}),
                "donchian": params.get("donchian", {}),
                "orb": params.get("orb", {}),
                "bos": params.get("bos", {}),
                "macd": params.get("macd", {}),
                "sweep": params.get("sweep", {}),
                "tod": params.get("tod", {}),
                "vwap": params.get("vwap", {}),
                "bos_pullback": params.get("bos_pullback", {}),
                "sr_retest": params.get("sr_retest", {}),
                "ema_cross": params.get("ema_cross", {}),
                "asian_fade": params.get("asian_fade", {}),
                "london_breakout": params.get("london_breakout", {}),
                "h4_pullback": params.get("h4_pullback", {}),
            },
        }
        # PASS THROUGH anything not in the whitelist above.
        #
        # The whitelist was silently dropping every key it did not name --
        # including `rsi`, so a leg configured as RSI(7) 25/75 was constructed
        # with RSIReversionStrategy's DEFAULTS (14, 30/70) and became an exact
        # duplicate of the RSI(14) leg. Two "different" legs were firing the
        # same signal, which invalidated the variant-diversification result
        # that was measured offline (where the params are passed directly and
        # do take effect). Same for `supertrend`, `stochastic`, `ma_cross`,
        # `smc_bos` and `confluence`.
        #
        # Defaults still apply for the keys named above; this only adds the
        # ones the whitelist forgot, so existing behaviour is unchanged for
        # any strategy whose params were already listed.
        for k, v in params.items():
            if k not in cfg["strategy"]:
                cfg["strategy"][k] = v

        s = cfg["strategy"]
        s["trend"].setdefault("ema_period", 50)
        s["signal"].setdefault("wick_max_pct", 0.05)
        s["signal"].setdefault("body_min_pct", 0.60)
        s["entry"].setdefault("retrace_bars", 5)
        s["exit"].setdefault("sl_swing_lookback", 10)
        s["exit"].setdefault("tp_r_multiple", 1.0)
        return cfg

    # --- main loop ---

    def run(self) -> None:
        log.info("Portfolio runner started. Ctrl+C to stop.")
        last_heartbeat = 0.0
        reconnect_backoff_s = 5  # grows on repeated failures, resets on success
        try:
            while not self._stopped:
                # Guards + watchdog run every tick and may FLATTEN + HALT,
                # but never kill the loop: a halted bot keeps reconciling,
                # keeps its ledger honest, and auto-resumes when allowed
                # (AUDIT C2 — the old version exited the process, and the
                # next start_all.ps1 run resurrected it with fresh anchors).
                self._guard_tick()

                ipc_failed_this_tick = False
                for slot in self.slots:
                    try:
                        bars = self._process_slot(slot)
                        self._manage_positions(slot, bars)
                    except RuntimeError as e:
                        if "IPC" in str(e):
                            ipc_failed_this_tick = True
                            log.warning(f"[{slot.name}] IPC error, will attempt reconnect")
                            break  # stop processing other slots; try reconnect
                        else:
                            log.exception(f"[{slot.name}] slot error: {e}")
                    except Exception as e:
                        log.exception(f"[{slot.name}] slot error: {e}")

                if ipc_failed_this_tick:
                    log.info(f"Attempting MT5 reconnect (backoff={reconnect_backoff_s}s)")
                    time.sleep(reconnect_backoff_s)
                    if self.engine.reconnect():
                        log.info("Reconnected to MT5 successfully")
                        reconnect_backoff_s = 5
                    else:
                        reconnect_backoff_s = min(reconnect_backoff_s * 2, 300)  # cap at 5 min
                        log.warning(f"Reconnect failed; next attempt in {reconnect_backoff_s}s")
                    continue  # skip heartbeat this tick

                if time.time() - last_heartbeat > 60:
                    self._heartbeat()
                    last_heartbeat = time.time()

                time.sleep(self.poll_interval_s)

        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            try:
                self._persist_runtime_state()
            except Exception as e:  # noqa: BLE001 — shutdown must complete
                log.warning(f"final state persist failed: {e}")
            self.engine.disconnect()
            self.ledger.close()
            log.info("Shutdown complete.")

    def stop(self):
        self._stopped = True

    # ------------------------------------------------------------- guards

    def flatten_book(self, reason: str) -> bool:
        """Close every position this book owns (all slots' magics). The
        hard kill-switch action — retried per position by the engine."""
        alert(f"FLATTEN: {reason}", book=self.book_name)
        all_ok = True
        for slot in self.slots:
            for p in self.engine.get_positions(slot.magic):
                if self.engine.close_position(p.ticket):
                    log.warning(f"[{slot.name}] flattened ticket={p.ticket} ({reason})")
                else:
                    all_ok = False
                    log.error(f"[{slot.name}] FAILED to flatten ticket={p.ticket} — will retry next tick")
        return all_ok

    def _flatten_positions_below_r(self, min_r: float, reason: str) -> None:
        """Close every open position (any slot) whose CURRENT floating
        R-multiple is below min_r. Used for the news-blackout rule (CLAUDE.md
        FORBIDDEN: holding through high-impact news below +1R) -- unlike
        flatten_book, this is selective, not "close everything"."""
        alert(f"PARTIAL FLATTEN ({reason})", book=self.book_name)
        for slot in self.slots:
            live = {p.ticket: p for p in self.engine.get_positions(slot.magic)}
            for ticket, tracked in list(slot.tracked.items()):
                p = live.get(ticket)
                if p is None:
                    continue
                risk_distance = abs(tracked.entry - tracked.orig_sl)
                cmult = contract_mult(slot.symbol)
                risk_dollars = risk_distance * tracked.lots * cmult
                if risk_dollars <= 0:
                    continue
                r_multiple = p.profit / risk_dollars
                if r_multiple < min_r:
                    if self.engine.close_position(ticket):
                        log.warning(
                            f"[{slot.name}] {reason}: closed ticket={ticket} at {r_multiple:+.2f}R"
                        )
                    else:
                        log.error(f"[{slot.name}] {reason}: FAILED to close ticket={ticket}")

    def _book_floating(self) -> float:
        total = 0.0
        for slot in self.slots:
            for p in self.engine.get_positions(slot.magic):
                total += p.profit + getattr(p, "swap", 0.0)
        return total

    def _guard_tick(self) -> None:
        """Evaluate every protection layer; set/clear the entry block."""
        self._realized_alltime = self.ledger.total_realized()
        self._floating = self._book_floating()

        equity_now = None
        try:
            acc = self.engine.account_info()
            equity_now = acc.equity
            self._acct_fail_count = 0
        except Exception as e:
            self._acct_fail_count += 1
            log.warning(f"account_info() failed ({self._acct_fail_count}x): {e}")

        # Fail CLOSED, not open (AUDIT M3): if the account is unreadable
        # repeatedly, stop opening new risk until it recovers.
        if self._acct_fail_count >= 3:
            self._entries_blocked = "account info unreadable"
            return

        # Watchdog (AUDIT M3): terminal gone / AlgoTrading off / stale feed
        # -> pause entries, keep managing, auto-resume on recovery.
        if self.watchdog_enabled and self.slots:
            health = self.engine.terminal_health(sample_symbol=self.slots[0].symbol)
            degraded = None
            if not health["connected"]:
                degraded = "terminal disconnected"
            elif not health["trade_allowed"]:
                degraded = "AlgoTrading disabled in terminal"
            elif (health["tick_age_s"] is not None
                  and health["tick_age_s"] > self.stale_tick_s
                  and datetime.now(timezone.utc).weekday() < 5):
                degraded = f"stale ticks ({health['tick_age_s']:.0f}s)"
            if degraded != self._watchdog_degraded:
                if degraded:
                    alert(f"WATCHDOG: {degraded} — pausing new entries", book=self.book_name)
                else:
                    alert("WATCHDOG: healthy again — resuming", book=self.book_name)
                self._watchdog_degraded = degraded
            if degraded:
                self._entries_blocked = degraded
                return

        # K9: local clock drift from NTP truth. Throttled — a real UDP round
        # trip every ~5s tick would be wasteful, and an unreachable NTP
        # server is not itself grounds to halt (fails open, see ntp_check.py).
        if self.k9_enabled:
            now_mono = time.monotonic()
            if now_mono - self._k9_last_check >= self.k9_check_interval_s:
                self._k9_last_check = now_mono
                drift = ntp_offset_seconds()
                if drift is not None and abs(drift) > self.k9_max_drift_s:
                    reason = f"K9 clock drift {drift:+.1f}s from NTP (limit {self.k9_max_drift_s:.0f}s)"
                    self.state.set_halt(reason=reason, scope="k9_clock")
                    alert(f"K9 HALT: {reason}", book=self.book_name)
                    self._entries_blocked = "K9 clock drift"
                    return

        # K2: weekly_pnl <= -weekly_stop_pct -> HALT new entries until Monday.
        # Does NOT flatten (spec says HALT, not flatten_halt, unlike K1/the
        # existing daily guard).
        if self.weekly_stop_pct > 0 and self.account_size > 0:
            week_key = self._prague_week_key()
            wk = self.state.get("weekly_pnl_anchor") or {}
            if wk.get("week") != week_key:
                wk = {"week": week_key, "realized_at_start": self._realized_alltime}
                self.state.set("weekly_pnl_anchor", wk)
                self.state.save()
                halt = self.state.get("halt") or {}
                if halt.get("active") and halt.get("scope") == "k2_weekly":
                    log.warning("New week — clearing K2 weekly halt")
                    self.state.clear_halt()
            week_pnl = (self._realized_alltime - float(wk.get("realized_at_start", 0.0))) + self._floating
            if week_pnl <= -self.weekly_stop_pct * self.account_size:
                reason = (f"K2 weekly stop: week P&L {week_pnl:+.2f} <= "
                          f"-{self.weekly_stop_pct * 100:.1f}% of {self.account_size:.0f}")
                self.state.set_halt(reason=reason, scope="k2_weekly", day=week_key)
                alert(f"K2 WEEKLY HALT: {reason}", book=self.book_name)
                self._entries_blocked = "K2 weekly stop"
                return

        # FTMO guard (Phase 3).
        if self.guard is not None:
            d = self.guard.check(
                realized_alltime=self._realized_alltime,
                floating=self._floating,
                equity_now=equity_now,
            )
            if d.action == "flatten_halt":
                self.flatten_book(d.reason)
                self.state.set_halt(reason=d.reason, scope=d.scope,
                                    day=self.guard.prague_day())
                alert(f"GUARD HALT ({d.scope}): {d.reason}", book=self.book_name)
                self._entries_blocked = f"guard halt: {d.scope}"
                return
            if d.action == "block":
                self._entries_blocked = f"{d.scope}: {d.reason}"
                # Weekend-flat also means FLAT — close anything still open.
                if d.scope == "weekend" and self._total_open_positions() > 0:
                    self.flatten_book("weekend-flat window")
                # CLAUDE.md FORBIDDEN list: "holding through high-impact news
                # on the traded currency below +1R". Unlike weekend, this
                # does NOT flatten everything -- a position already at/above
                # +1R is fine to hold through the window; only the ones
                # still under +1R (where an adverse spike could do real
                # damage) get closed.
                if d.scope == "news" and self._total_open_positions() > 0:
                    self._flatten_positions_below_r(1.0, "news blackout: below +1R")
                return
            if d.scope == "target":
                alert(f"TARGET REACHED: {d.reason}", book=self.book_name)

        # YOLO circuit breaker (Phase 4): book equity vs its high-water mark.
        if self.breaker_enabled and self.account_size > 0:
            book_equity = self.account_size + self._realized_alltime + self._floating
            hwm = self.state.get("equity_high_water")
            if hwm is None or book_equity > hwm:
                hwm = book_equity
                self.state.set("equity_high_water", hwm)
                self.state.save()
            dd_pct = (hwm - book_equity) / hwm * 100.0 if hwm > 0 else 0.0
            if dd_pct >= self.breaker_dd_pct:
                reason = (f"circuit breaker: book equity {book_equity:.2f} is "
                          f"{dd_pct:.1f}% below high-water {hwm:.2f}")
                self.flatten_book(reason)
                self.state.set_halt(reason=reason, scope="breaker")
                alert(f"BREAKER HALT: {reason}", book=self.book_name)
                self._entries_blocked = "circuit breaker"
                return

        self._check_challenge_phase()

        # Legacy USD kill switches (kept as a secondary belt; guard/breaker
        # are the primary protections now). Flatten + halt, never process-exit.
        today = datetime.now(timezone.utc).date().isoformat()
        # Non-guard books: clear yesterday's daily halt (guard books do
        # this inside guard.check at the CE(S)T rollover).
        if self.guard is None and self.state.halted:
            halt = self.state.get("halt") or {}
            if halt.get("scope") == "daily" and halt.get("day") != today:
                log.warning("New UTC day — clearing legacy daily halt")
                self.state.clear_halt()
        daily_legacy = self.ledger.daily_pnl(today)
        if daily_legacy <= self.daily_stop_usd:
            reason = f"legacy daily stop: ${daily_legacy:.2f} <= ${self.daily_stop_usd}"
            if self.daily_stop_action == "block":
                # Entries only: open positions keep being managed to their own
                # SL/TP. See the daily_stop_action note in __init__.
                if not self.state.halted:
                    log.warning(f"{reason} — blocking NEW entries, "
                                f"open positions continue to be managed")
            else:
                self.flatten_book(reason)
            self.state.set_halt(reason=reason, scope="daily", day=today)
            self._entries_blocked = "legacy daily stop"
            return
        if equity_now is not None and self.starting_equity - equity_now >= abs(self.max_dd_usd):
            reason = (f"legacy total DD: equity {equity_now:.2f} vs "
                      f"session start {self.starting_equity:.2f}")
            self.flatten_book(reason)
            self.state.set_halt(reason=reason, scope="total")
            self._entries_blocked = "legacy total DD"
            return

        if self.state.halted:
            halt = self.state.get("halt") or {}
            self._entries_blocked = f"halted ({halt.get('scope')})"
            return

        self._entries_blocked = None

    def _trading_enabled(self) -> bool:
        """Reads the dashboard-controlled start/stop flag. Missing file,
        missing key, or unreadable JSON all default to enabled — a bot
        should never go silently inert just because this file hasn't been
        created yet or got corrupted."""
        if self.control_path is None:
            return True
        try:
            data = json.loads(self.control_path.read_text())
            return bool(data.get("trading_enabled", True))
        except (OSError, ValueError):
            return True

    def _process_slot(self, slot: StrategySlot) -> pd.DataFrame | None:
        """Returns `raw` (still indexed by timestamp, still-forming bar
        included) so callers needing the *live* candle — the skyrocket check
        in _manage_positions — can reuse this fetch instead of hitting MT5
        again for the same symbol/timeframe this same tick. Internally this
        method still works off closed_bars (forming bar dropped) for signal
        detection, since a still-forming bar can't be treated as a finished
        signal bar."""
        raw = self.engine.get_bars(slot.symbol, slot.timeframe, WARMUP_BARS + 1)
        closed_bars = raw.iloc[:-1]  # drop still-forming bar
        if len(closed_bars) == 0:
            return None
        latest_ts = closed_bars.index[-1]
        bars = closed_bars.reset_index()

        if latest_ts <= slot.last_bar_ts:
            return raw  # no new closed bar, but caller may still want it (skyrocket check)

        # Replay every bar closed since the last check, not just the newest
        # one. A polling gap (bot downtime, laptop sleep) would otherwise
        # silently skip whatever happened mid-gap: NoWick tracks pending
        # setups bar-by-bar and would never see ones that fired and expired
        # inside the gap, and even a stateless strategy (Donchian, MACD) can
        # miss a breakout that already reversed by the time polling resumes,
        # since only the latest bar's relationship to price gets checked.
        new_bar_indices = bars.index[bars["timestamp"] > slot.last_bar_ts].tolist()
        latest_index = len(bars) - 1
        # last_bar_ts advances PER BAR, only once that bar is actually
        # resolved (signal recorded, or a legitimate one-shot skip) — NOT
        # unconditionally up front. Advancing it eagerly (the old behavior)
        # marked every bar "seen" the instant it was first looked at, so a
        # market-closed retry (see _MarketClosedRetry below) never actually
        # got a second poll: the very next tick's `latest_ts <= slot.last_bar_ts`
        # check above would already be true and skip straight past this loop.
        # Confirmed live: this is why YOLO's 3 signals since 2026-08-05 all
        # died on the first "Market closed" attempt and were never retried.

        for bar_index in new_bar_indices:
            # Idempotency key = the bar's TIMESTAMP. The old key (bar_index)
            # was the bar's position in the rolling window — always 299 for
            # the newest bar — which permanently killed a slot after its
            # first signal (AUDIT C1, confirmed in the live ledgers).
            bar_ts_val = bars.loc[bar_index, "timestamp"]
            bar_ts = bar_ts_val.isoformat()
            if self.ledger.already_have_signal(slot.name, bar_ts):
                slot.last_bar_ts = bar_ts_val
                continue

            # on_bar always runs for unseen bars (keeps stateful strategies
            # like NoWick correct); the gates below only stop ORDERS.
            try:
                trades = slot.strategy.on_bar(bar_index, bars)
            except Exception as e:
                log.exception(f"[{slot.name}] strategy.on_bar error: {e}")
                slot.last_bar_ts = bar_ts_val
                continue
            if not trades:
                slot.last_bar_ts = bar_ts_val
                continue

            if not self._trading_enabled():
                slot.last_bar_ts = bar_ts_val
                continue  # scanning paused — strategy state stays current, but no new orders

            # Replay-staleness gate (AUDIT H6): after downtime, old bars are
            # replayed for state, but executing an hours-old breakout at
            # TODAY's market price is not the trade the backtest measured.
            age = latest_index - bar_index
            if age > self.max_signal_age_bars:
                log.info(
                    f"[{slot.name}] {len(trades)} signal(s) from {bar_ts} skipped — "
                    f"{age} bars stale (max_signal_age_bars={self.max_signal_age_bars})"
                )
                slot.last_bar_ts = bar_ts_val
                continue

            if len(self.engine.get_positions(slot.magic)) >= self.max_positions:
                log.info(f"[{slot.name}] signal skipped — slot at max_positions ({self.max_positions})")
                slot.last_bar_ts = bar_ts_val
                continue

            try:
                for trade in trades:
                    self._execute_trade(slot, trade, bar_index, bar_ts)
            except _MarketClosedRetry:
                # Do NOT advance last_bar_ts: leave this bar "new" so the
                # next poll re-enters this loop and tries again. Also stop
                # processing any newer bars this tick — the retry needs to
                # resolve (or age out via max_signal_age_bars) in order.
                break

            slot.last_bar_ts = bar_ts_val

        return raw

    def _currency_exposure_r(self, currency: str) -> float:
        """Risk currently committed to `currency`, in R-multiples of
        risk_usd_per_trade, summed across EVERY slot's open positions —
        not just one. Approximates each open position's risk as one R
        (risk_usd_per_trade) regardless of exactly how far price has moved
        since entry; precise enough for a portfolio-level exposure cap."""
        total_r = 0.0
        for slot in self.slots:
            if currency not in _currencies(slot.symbol):
                continue
            total_r += len(slot.tracked)
        return total_r

    def _slot_quarantined(self, name: str) -> bool:
        """Self-improving gate: is this slot currently benched for degraded
        live performance? Also handles quarantine expiry + (re)entry."""
        if not self.slot_health_enabled:
            return False
        h = self._health.get(name)
        if h is None:
            return False
        now = datetime.now(timezone.utc)
        if h.quarantined_until is not None:
            if now < h.quarantined_until:
                return True
            log.warning(f"[{name}] quarantine EXPIRED — slot re-enabled on probation")
            h.quarantined_until = None
            h.recent_r.clear()  # fresh window; don't instantly re-trip on old losses
            q = self.state.get("quarantine") or {}
            if q.pop(name, None) is not None:
                self.state.set("quarantine", q)
                self.state.save()
            return False
        window = list(h.recent_r)[-self.slot_health_window:]
        if len(window) >= self.slot_health_window and sum(window) <= self.slot_health_min_net_r:
            h.quarantined_until = now + timedelta(hours=self.slot_health_cooldown_h)
            q = self.state.get("quarantine") or {}
            q[name] = h.quarantined_until.isoformat()
            self.state.set("quarantine", q)
            self.state.save()
            alert(
                f"SLOT QUARANTINED: {name} net {sum(window):+.1f}R over last "
                f"{len(window)} trades — benched until "
                f"{h.quarantined_until.isoformat(timespec='minutes')}",
                book=self.book_name,
            )
            if self.auto_retune_enabled:
                self._apply_retune(name)
            return True
        return False

    def _apply_retune(self, name: str) -> None:
        """Live parameter mutation — see auto_retune.py for the bounds. Runs
        every time a slot's rolling window trips the quarantine threshold, in
        ADDITION to quarantining it (not instead of): the parameter change
        persists past the cooldown, so the slot comes back on probation
        already retuned, not with its original (just-proven-degraded) exit."""
        slot = next((s for s in self.slots if s.name == name), None)
        if slot is None or not hasattr(slot.strategy, "tp_r_multiple"):
            return
        rt = self._retune.get(name)
        if rt is None:
            rt = RetuneState(
                original_tp_r=float(slot.strategy.tp_r_multiple),
                current_tp_r=float(slot.strategy.tp_r_multiple),
            )
            self._retune[name] = rt
        new_tp, applied = next_tp_r_multiple(
            rt, step_pct=self.auto_retune_step_pct,
            floor_frac=self.auto_retune_floor_frac,
            max_retunes=self.auto_retune_max_retunes,
        )
        if not applied:
            log.info(f"[{name}] auto-retune skipped — already at floor or max_retunes reached")
            return
        old_tp = rt.current_tp_r
        rt.current_tp_r = new_tp
        rt.retune_count += 1
        slot.strategy.tp_r_multiple = new_tp
        saved = self.state.get("auto_retune") or {}
        saved[name] = {
            "original_tp_r": rt.original_tp_r,
            "current_tp_r": rt.current_tp_r,
            "retune_count": rt.retune_count,
        }
        self.state.set("auto_retune", saved)
        self.state.save()
        alert(
            f"AUTO-RETUNE: {name} tp_r_multiple {old_tp:.3f} -> {new_tp:.3f} "
            f"(retune #{rt.retune_count}/{self.auto_retune_max_retunes}) after degraded live performance",
            book=self.book_name,
        )
        log.warning(f"[{name}] AUTO-RETUNE tp_r_multiple {old_tp:.3f} -> {new_tp:.3f}")

    def _total_open_positions(self) -> int:
        return sum(len(s.tracked) for s in self.slots)

    def _prague_week_key(self) -> str:
        """ISO (year, week) on the same Prague-day boundary the daily anchor
        uses, e.g. '2026-W05'. K2's week rollover point."""
        d = date.fromisoformat(FTMOGuard.prague_day())
        iso_year, iso_week, _ = d.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"

    def _trades_today(self) -> int:
        """K11's counter, day-scoped on the same Prague-day boundary the
        FTMO guard uses elsewhere in this file (one definition of "today"
        across the whole system, not two)."""
        rec = self.state.get("trade_count_day") or {}
        if rec.get("day") != FTMOGuard.prague_day():
            return 0
        return int(rec.get("count", 0))

    def _increment_trade_count(self) -> None:
        day = FTMOGuard.prague_day()
        rec = self.state.get("trade_count_day") or {}
        if rec.get("day") != day:
            rec = {"day": day, "count": 0}
        rec["count"] = int(rec.get("count", 0)) + 1
        self.state.set("trade_count_day", rec)
        self.state.save()

    def _current_risk_usd(self) -> float:
        """Risk per trade in USD. percent_equity mode compounds with the
        BOOK's realized P&L (account_size + ledger), not the shared broker
        account — floating P&L deliberately excluded so an open winner
        doesn't inflate the next trade's risk before it's banked.

        equity_ratchet mode is CLAUDE.md's non-negotiable risk ratchet
        (spec §3.3, risk_ratchet.py): risk shrinks in drawdown AND shrinks
        again near the profit target (protect gains), and NEVER increases
        to chase a deadline — same book-equity basis as percent_equity.

        K4 (consecutive-loss halving) is applied AFTER the mode-specific
        base, as a multiplier -- it composes with whichever mode is active
        rather than replacing it."""
        if self.risk_mode == "equity_ratchet" and self.account_size > 0:
            book_equity = max(self.account_size + self._realized_alltime, 0.0)
            pct = equity_ratchet_pct(book_equity, self.account_size)
            base = max(book_equity * pct, 0.0)
        elif self.risk_mode == "percent_equity" and self.account_size > 0:
            book_equity = max(self.account_size + self._realized_alltime, 0.0)
            base = max(book_equity * self.risk_pct / 100.0, 0.0)
        else:
            base = self.risk_usd

        # Adaptive taper: shrink base risk as book equity nears EITHER the
        # +target% or the -loss% barrier, full size while far from both.
        # Uses the same realized-only book_equity as percent_equity/
        # equity_ratchet above (a floating winner does not inflate sizing) —
        # the backtest that validated this (scratch/ftmo_4day_adaptive_risk.py)
        # marks equity to market including floating P&L, so live results will
        # track the backtest directionally but not exactly; this is the same
        # deliberate conservatism already applied to the other sizing modes.
        if self.adaptive_risk_enabled and self.account_size > 0:
            book_equity = max(self.account_size + self._realized_alltime, 0.0)
            eq_frac = book_equity / self.account_size
            dist_to_target = (1.0 + self.adaptive_risk_target_pct / 100.0) - eq_frac
            dist_to_loss = eq_frac - (1.0 - self.adaptive_risk_loss_pct / 100.0)
            if self.adaptive_risk_mode == "loss_only":
                nearest = dist_to_loss
            elif self.adaptive_risk_mode == "target_only":
                nearest = dist_to_target
            else:
                nearest = min(dist_to_target, dist_to_loss)
            band = self.adaptive_risk_band_pct / 100.0
            taper_frac = max(self.adaptive_risk_floor, min(1.0, nearest / band)) if band > 0 else 1.0
            base *= taper_frac

        if self.k4_enabled and (self.state.get("k4_state") or {}).get("recovery_mode"):
            base *= 0.5

        # Target banked, day count not yet met: there is nothing left to win
        # from the target and a full-size loss can still take the pass away.
        # Trade small enough to keep logging trading days without risking it.
        if (self.state.get("phase_status") or {}).get("banked"):
            base *= self.post_target_risk_frac

        return base

    def _check_challenge_phase(self) -> None:
        """Report where this book stands in the FTMO evaluation, and alert
        loudly the moment a phase is actually PASSED (target AND day count),
        not merely when the target is first touched."""
        if self.account_size <= 0:
            return
        status = evaluate_phase(
            self.challenge_phase,
            profit_usd=self._realized_alltime,
            account_size=self.account_size,
            trading_days=self.ledger.distinct_trading_days(),
        )
        prev = self.state.get("phase_status") or {}
        self.state.set("phase_status", {
            "phase": status.phase, "passed": status.passed,
            "target_reached": status.target_reached, "banked": status.banked,
            "trading_days": status.trading_days,
            "profit_usd": round(status.profit_usd, 2),
            "profit_pct": round(status.profit_pct, 3),
            "headline": status.headline,
        })

        # Only speak up when something actually changed, so the log stays
        # readable across thousands of ticks.
        if status.passed and not prev.get("passed"):
            nxt = next_phase(status.phase)
            banner = "=" * 60
            msg = (f"\n{banner}\n{status.headline}\n{status.detail}\n{banner}")
            log.warning(msg)
            alert(f"{status.headline} — {status.detail}", book=self.book_name)
            if nxt == "funded":
                alert("NEXT TRADES ARE ON A FUNDED ACCOUNT — REAL MONEY. "
                      "Re-size risk before trading (2% is a challenge setting; "
                      "0.5-1% is the funded setting).", book=self.book_name)
            self.state.save()
            return

        if status.banked and not prev.get("banked"):
            log.warning(f"{status.headline} — {status.detail}")
            alert(f"{status.headline}. Risk cut to "
                  f"{self.post_target_risk_frac:.0%} of normal until the day "
                  f"count is met, so the pass cannot be given back.",
                  book=self.book_name)
            self.state.save()

    def _k4_record_trade_result(self, pnl_usd: float | None) -> None:
        """K4: 4 consecutive losses -> risk x0.5 until 2 (consecutive) wins.
        pnl_usd=None (unreconciled trades) doesn't count either way -- the
        outcome genuinely isn't known."""
        if not self.k4_enabled or pnl_usd is None:
            return
        st = self.state.get("k4_state") or {
            "consecutive_losses": 0, "recovery_mode": False, "wins_since_recovery": 0,
        }
        is_win = pnl_usd > 0
        if st.get("recovery_mode"):
            if is_win:
                st["wins_since_recovery"] = int(st.get("wins_since_recovery", 0)) + 1
                if st["wins_since_recovery"] >= self.k4_recovery_wins_needed:
                    st = {"consecutive_losses": 0, "recovery_mode": False, "wins_since_recovery": 0}
                    alert(
                        f"K4: risk restored to full after {self.k4_recovery_wins_needed} "
                        f"consecutive wins", book=self.book_name,
                    )
            else:
                st["wins_since_recovery"] = 0  # a loss during recovery resets the win streak
        elif is_win:
            st["consecutive_losses"] = 0
        else:
            st["consecutive_losses"] = int(st.get("consecutive_losses", 0)) + 1
            if st["consecutive_losses"] >= self.k4_consecutive_loss_n:
                st["recovery_mode"] = True
                st["wins_since_recovery"] = 0
                alert(
                    f"K4: {st['consecutive_losses']} consecutive losses — risk halved "
                    f"until {self.k4_recovery_wins_needed} consecutive wins",
                    book=self.book_name,
                )
        self.state.set("k4_state", st)
        self.state.save()

    def _execute_trade(self, slot: StrategySlot, trade: TradeSpec, bar_index: int, bar_ts: str) -> None:
        # --- central entry block: guard halts, watchdog, blackout, weekend ---
        if self._entries_blocked:
            log.info(f"[{slot.name}] signal skipped — entries blocked ({self._entries_blocked})")
            self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                      gate="entries_blocked", reason=str(self._entries_blocked))
            return

        # --- partial-bar guard: skip entries on configured weekdays ---
        # A 15m->1d resample (and the broker's own D1) gives Sunday a stub bar
        # of ~12 bars vs 96 for a full session, so RSI and the swing stop are
        # computed from ~3 hours of thin trade. Measured over 10.5y
        # (scripts/loss_attribution.py): Sunday entries ran -0.426R in the
        # first half and -0.188R in the second, while Mon-Fri ran -0.066R then
        # +0.071R -- negative in BOTH halves, and only ~10% of trades.
        if self.skip_weekdays:
            dow = datetime.now(timezone.utc).weekday()  # Mon=0 .. Sun=6
            if dow in self.skip_weekdays:
                log.info(f"[{slot.name}] signal skipped — weekday {dow} in skip_weekdays "
                         f"(partial/illiquid bar)")
                self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                          gate="skip_weekday", reason=f"dow={dow}")
                return

        # --- self-improving gate: quarantine (whole slot) + learned session filter ---
        if self._slot_quarantined(slot.name):
            log.info(f"[{slot.name}] signal skipped — slot quarantined (degraded live edge)")
            self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                      gate="quarantine", reason="degraded live edge")
            return
        if self.learning_enabled:
            session = classify_session(datetime.now(timezone.utc).hour)
            if self.learner.should_avoid(slot.name, session):
                log.info(
                    f"[{slot.name}] signal skipped — learned to avoid the {session} session "
                    f"(repeated losses there); existing positions still managed"
                )
                self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                          gate="learned_avoid", reason=f"session={session}")
                return

        # --- K11: daily trade-count cap (CLAUDE.md kill switches §11.1) ---
        if self.max_trades_per_day > 0:
            today_count = self._trades_today()
            if today_count >= self.max_trades_per_day:
                log.info(
                    f"[{slot.name}] signal skipped — K11 daily trade cap reached "
                    f"({today_count}/{self.max_trades_per_day})"
                )
                self.ledger.log_rejection(
                    config_name=slot.name, symbol=slot.symbol, gate="k11_daily_cap",
                    reason=f"{today_count}/{self.max_trades_per_day}")
                return

        # --- K6: spread anomaly (symbol-specific, checked before any sizing
        # work) -- spread > k6_spread_multiple x its recent rolling median,
        # sustained for k6_persist_s, pauses new entries for THIS symbol
        # only (other symbols/slots are unaffected). ---
        if self.k6_enabled:
            spread = self.engine.get_spread(slot.symbol)
            if spread is not None:
                hist = self._spread_history.setdefault(slot.symbol, [])
                now_mono = time.monotonic()
                hist.append((now_mono, spread))
                cutoff = now_mono - 600  # keep 10 min of samples for the median
                del hist[:len(hist) - 200]  # hard cap regardless of time
                hist[:] = [(t, s) for t, s in hist if t >= cutoff]
                if len(hist) >= 5:
                    samples = sorted(s for _, s in hist)
                    median = samples[len(samples) // 2]
                    if median > 0 and spread > self.k6_spread_multiple * median:
                        since = self._k6_bad_since.setdefault(slot.symbol, now_mono)
                        if now_mono - since >= self.k6_persist_s:
                            log.info(
                                f"[{slot.name}] signal skipped — K6 spread anomaly on "
                                f"{slot.symbol}: {spread:.5f} > {self.k6_spread_multiple}x "
                                f"median {median:.5f}, sustained {now_mono - since:.0f}s"
                            )
                            self.ledger.log_rejection(
                                config_name=slot.name, symbol=slot.symbol, gate="k6_spread_anomaly",
                                reason=f"spread={spread:.5f} median={median:.5f}")
                            return
                    else:
                        self._k6_bad_since.pop(slot.symbol, None)

        # --- portfolio-wide risk caps (AUDIT H2) ---
        total_open = self._total_open_positions()
        if self.max_concurrent_positions_total > 0 and total_open + 1 > self.max_concurrent_positions_total:
            log.info(
                f"[{slot.name}] signal skipped — portfolio at max concurrent "
                f"positions ({total_open}/{self.max_concurrent_positions_total})"
            )
            self.ledger.log_rejection(
                config_name=slot.name, symbol=slot.symbol, gate="max_concurrent",
                reason=f"{total_open}/{self.max_concurrent_positions_total}")
            return
        if self.max_total_open_risk_r > 0 and total_open + 1 > self.max_total_open_risk_r:
            log.info(
                f"[{slot.name}] signal skipped — total open risk would exceed "
                f"{self.max_total_open_risk_r}R (~{total_open + 1}R requested)"
            )
            self.ledger.log_rejection(
                config_name=slot.name, symbol=slot.symbol, gate="max_total_open_risk_r",
                reason=f"~{total_open + 1}R requested > {self.max_total_open_risk_r}R")
            return
        if self.max_currency_exposure_r > 0:
            for currency in _currencies(slot.symbol):
                if self._currency_exposure_r(currency) + 1 > self.max_currency_exposure_r:
                    log.info(
                        f"[{slot.name}] signal skipped — {currency} exposure already at/above "
                        f"{self.max_currency_exposure_r}R across the portfolio"
                    )
                    self.ledger.log_rejection(
                        config_name=slot.name, symbol=slot.symbol, gate="max_currency_exposure_r",
                        reason=f"{currency} >= {self.max_currency_exposure_r}R")
                    return

        # --- live sizing (AUDIT C5) + broker volume normalization (H7) ---
        direction = trade.direction.value
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        if sl_distance <= 0:
            log.warning(f"[{slot.name}] signal dropped — zero SL distance")
            self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                      gate="zero_sl_distance", reason="strategy bug indicator")
            return

        # Stops-level clamp (AUDIT M5): a strategy SL tighter than the
        # broker's minimum distance would be rejected as Invalid stops.
        # Widen the distance BEFORE sizing so risk-per-trade stays exact.
        min_dist = self.engine.stops_min_distance(slot.symbol)
        if min_dist > 0 and sl_distance < min_dist * 1.2:
            new_dist = min_dist * 1.2
            log.info(
                f"[{slot.name}] SL distance {sl_distance:.5f} below broker stops "
                f"level — widened to {new_dist:.5f} (lots resized to keep risk)"
            )
            if trade.direction == Direction.LONG:
                trade.stop_loss = trade.entry_price - new_dist
                trade.take_profit = trade.entry_price + new_dist * (
                    abs(trade.take_profit - trade.entry_price) / sl_distance)
            else:
                trade.stop_loss = trade.entry_price + new_dist
                trade.take_profit = trade.entry_price - new_dist * (
                    abs(trade.take_profit - trade.entry_price) / sl_distance)
            sl_distance = new_dist

        # --- cost gate (CLAUDE.md spec §1.1): reject if (spread + commission
        # + slippage) / stop_distance > cost_gate_r. Static per-symbol specs
        # (instruments.PAIR_SPECS) -- no live-tick-based spread feed wired
        # into the signal path; commission converted from $/lot to PRICE
        # units via the same contract_mult used for sizing everywhere else.
        # Slippage uses the spread itself as a same-order-of-magnitude proxy
        # (no live slippage model exists yet either). ---
        if self.cost_gate_enabled:
            specs = PAIR_SPECS.get(slot.symbol)
            if specs is not None:
                cmult = contract_mult(slot.symbol)
                commission_price = (specs["commission_per_lot"] / cmult) if cmult else 0.0
                spread_price = specs["spread_price"]
                slippage_price = spread_price  # proxy, see docstring above
                cost_r = (spread_price + commission_price + slippage_price) / sl_distance
                if cost_r > self.cost_gate_r:
                    log.info(
                        f"[{slot.name}] signal skipped — cost gate: cost {cost_r:.3f}R "
                        f"> {self.cost_gate_r}R (spread={spread_price:.5f} "
                        f"commission={commission_price:.5f} slippage={slippage_price:.5f} "
                        f"stop_distance={sl_distance:.5f})"
                    )
                    self.ledger.log_rejection(
                        config_name=slot.name, symbol=slot.symbol, gate="cost_gate",
                        reason=f"cost {cost_r:.3f}R > {self.cost_gate_r}R")
                    return

        risk_usd = self._current_risk_usd()
        live_mult = self.engine.usd_per_point(slot.symbol)
        if live_mult is not None:
            static_mult = contract_mult(slot.symbol)
            drift = abs(live_mult - static_mult) / static_mult if static_mult else 0.0
            if drift > self.max_contract_mult_drift:
                # Confirmed live on XAUUSD: the broker misreported
                # trade_tick_value by 10x (drift=90%), and "using LIVE
                # anyway" (the old behavior below) silently sized real
                # positions ~10x over the intended risk_usd_per_trade —
                # a $100-risk trade became a ~$1000-risk trade. A drift
                # this large is far more likely a broker data bug than a
                # genuine live update; refuse rather than guess which of
                # the two numbers to trust with real position size.
                alert(
                    f"SIZING REFUSED: {slot.name} contract multiplier drift {drift:.0%} "
                    f"(live={live_mult:.2f} vs instruments.py={static_mult:.2f}) — signal "
                    f"skipped. Check MT5 symbol_info for {slot.symbol} before it trades again.",
                    book=self.book_name,
                )
                log.error(
                    f"[{slot.name}] signal skipped — contract multiplier drift {drift:.0%} "
                    f"exceeds the {self.max_contract_mult_drift:.0%} safety limit "
                    f"(live={live_mult:.2f} vs instruments.py={static_mult:.2f})"
                )
                self.ledger.log_rejection(
                    config_name=slot.name, symbol=slot.symbol, gate="contract_mult_drift",
                    reason=f"drift {drift:.0%} > {self.max_contract_mult_drift:.0%}")
                return
            if drift > 0.25:
                log.warning(
                    f"[{slot.name}] contract multiplier drift {drift:.0%}: "
                    f"live={live_mult:.1f} vs instruments.py={static_mult:.1f} — "
                    f"using LIVE; update PAIR_SPECS so backtests stay honest"
                )
            ideal_lots = risk_usd / (sl_distance * live_mult)
        else:
            ideal_lots = trade.lots * (risk_usd / self.risk_usd if self.risk_usd else 1.0)
            log.warning(f"[{slot.name}] live symbol specs unavailable — falling back to static sizing")

        vol_min, vol_max, vol_step = self.engine.volume_spec(slot.symbol)
        lots = normalize_volume(ideal_lots, vol_min, vol_max, vol_step)
        floored = False
        if lots is None:
            if self.min_lot_policy == "floor":
                lots = vol_min
                floored = True
                actual_risk = lots * sl_distance * (live_mult or contract_mult(slot.symbol))
                log.warning(
                    f"[{slot.name}] ideal lots {ideal_lots:.4f} < volume_min {vol_min} — "
                    f"FLOORED to {vol_min} (actual risk ~${actual_risk:.2f} vs target ${risk_usd:.2f})"
                )
            else:
                log.info(
                    f"[{slot.name}] signal skipped — ideal lots {ideal_lots:.4f} below "
                    f"volume_min {vol_min} (min_lot_policy=skip keeps risk honest)"
                )
                self.ledger.log_rejection(
                    config_name=slot.name, symbol=slot.symbol, gate="min_lot_skip",
                    reason=f"ideal_lots {ideal_lots:.4f} < volume_min {vol_min}")
                return

        # Explicit risk assertion (CLAUDE.md implementation rules): the whole
        # point of computing lots FROM risk_usd is that actual risk should
        # never exceed the target by more than rounding. Skipped when the
        # lot was FLOORED to the broker minimum -- that path already logs
        # and deliberately accepts oversized risk (min_lot_policy=floor),
        # so asserting here would just crash on an already-handled case.
        if not floored:
            actual_risk = lots * sl_distance * (live_mult or contract_mult(slot.symbol))
            assert actual_risk <= risk_usd * 1.02 + 1e-6, (
                f"{slot.name}: actual_risk ${actual_risk:.2f} exceeds target ${risk_usd:.2f} "
                f"by more than 2% (lots={lots}, sl_distance={sl_distance})"
            )

        # --- K10: lot-size anomaly -- reject + halt if this order's size is
        # way out of line with recent history (book-wide, from the ledger's
        # own filled-order record so it survives restarts). ---
        if self.k10_enabled:
            recent_lots = [
                r[0] for r in self.ledger.conn.execute(
                    "SELECT filled_lots FROM orders WHERE retcode = 10009 "
                    "ORDER BY id DESC LIMIT ?", (self.k10_window,)
                ).fetchall()
                if r[0] is not None and r[0] > 0
            ]
            if len(recent_lots) >= 5:
                srt = sorted(recent_lots)
                med = srt[len(srt) // 2]
                if med > 0 and lots > self.k10_multiple * med:
                    reason = (f"K10 lot anomaly: {slot.name} order {lots} lots > "
                              f"{self.k10_multiple}x recent median {med}")
                    self.state.set_halt(reason=reason, scope="k10_lot")
                    alert(f"K10 HALT: {reason}", book=self.book_name)
                    log.error(f"[{slot.name}] signal REJECTED — {reason}")
                    self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                              gate="k10_lot_anomaly", reason=reason)
                    return

        # --- market-closed gate -------------------------------------------
        # Daily-bar signals are born at the bar close (21:00-22:00 UTC), which
        # is exactly the broker's rollover window: many symbols are shut and
        # order_send returns 10018 MARKET_CLOSED. That code is not in
        # TRANSIENT_RETCODES (correctly -- retrying 0.4s later cannot help,
        # the market is closed for hours), so the order simply failed and
        # `if result.ok:` did nothing. The signal was already logged, so
        # ledger.already_have_signal() suppressed the same bar forever and the
        # trade was lost permanently.
        #
        # Observed live: the YOLO book placed 2 orders in 4 days and BOTH
        # failed this way ("Market closed for this symbol right now"), which
        # is why it appeared never to trade.
        #
        # Checking BEFORE log_signal means the signal is never recorded, so
        # the next tick re-evaluates the same bar and retries once the market
        # opens -- within the max_signal_age_bars window. This ONLY works if
        # the caller (_process_slot) also holds off marking the bar seen —
        # raise so it can tell "retry me" apart from every other gate above,
        # which are legitimate one-shot skips.
        if not self.engine.is_tradeable(slot.symbol):
            log.info(f"[{slot.name}] market closed for {slot.symbol} — "
                     f"holding signal, will retry while the bar is fresh")
            self.ledger.log_rejection(config_name=slot.name, symbol=slot.symbol,
                                      gate="market_closed",
                                      reason="symbol not tradeable right now")
            raise _MarketClosedRetry()

        signal_id = self.ledger.log_signal(
            config_name=slot.name,
            symbol=slot.symbol,
            timeframe=slot.timeframe,
            direction=direction,
            entry_price=trade.entry_price,
            sl=trade.stop_loss,
            tp=trade.take_profit,
            lots=lots,
            bar_index=bar_index,
            bar_ts=bar_ts,
        )

        # Show the MONEY on every entry, not just the lot size. "0.1 lots"
        # tells you nothing about exposure -- 0.1 lots of US30 and 0.1 lots of
        # EURUSD differ by orders of magnitude, and the minimum lot on some
        # instruments already exceeds what a small book should risk.
        book_equity = self.account_size + self._realized_alltime
        mult = self.engine.usd_per_point(slot.symbol) or \
            PAIR_SPECS.get(slot.symbol, {}).get("contract_mult", 0.0)
        stop_dist = abs(trade.entry_price - trade.stop_loss)
        risk_usd = stop_dist * mult * lots
        notional = trade.entry_price * mult * lots
        reward_usd = abs(trade.take_profit - trade.entry_price) * mult * lots
        pct = (risk_usd / book_equity * 100.0) if book_equity > 0 else float("nan")

        log.info(
            f"[{slot.name}] SIGNAL {direction.upper()} {lots} lots @ mkt "
            f"SL={trade.stop_loss:.5f} TP={trade.take_profit:.5f}"
        )
        log.info(
            f"[{slot.name}]   BOOK ${book_equity:,.2f}  ->  RISKING "
            f"${risk_usd:,.2f} ({pct:.2f}% of book)  |  target ${reward_usd:,.2f}"
            f"  |  position notional ${notional:,.2f}"
        )
        if pct > 25.0:
            log.warning(
                f"[{slot.name}]   OVERSIZED: {pct:.1f}% of the book on one "
                f"trade. Broker minimum lot ({lots}) is too large for a "
                f"${book_equity:,.0f} book on {slot.symbol}."
            )

        result = self.engine.place_market_order(
            symbol=slot.symbol,
            direction=direction,
            lots=lots,
            sl=trade.stop_loss,
            tp=trade.take_profit,
            magic=slot.magic,
            comment=slot.name,
            signal_entry_price=trade.entry_price,
        )

        self.ledger.log_order(
            signal_id=signal_id,
            config_name=slot.name,
            symbol=slot.symbol,
            direction=direction,
            requested_price=trade.entry_price,
            filled_price=result.price,
            requested_lots=lots,
            filled_lots=result.volume,
            sl=result.sl,
            tp=result.tp,
            ticket=result.ticket,
            retcode=result.retcode,
            comment=result.comment,
        )

        tag = "OK" if result.ok else f"FAIL(retcode={result.retcode})"
        if result.partial:
            tag = f"PARTIAL({result.volume} of {lots})"
        log.info(
            f"[{slot.name}] ORDER {tag} ticket={result.ticket} "
            f"filled_price={result.price} {result.comment}"
        )

        if result.ok:
            self._increment_trade_count()
            slot.tracked[result.ticket] = TrackedPosition(
                entry=result.price, orig_sl=result.sl, orig_tp=result.tp,
                lots=result.volume, is_long=(direction == "long"),
                opened_ts=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            )

            # K8: slippage recalibration -- fill price vs requested price,
            # rolling window, checked against the same static spread_price
            # used as the modeled/assumed cost elsewhere (no live slippage
            # model exists). NEVER auto-resumes ("recalibrate" means the
            # assumption itself needs a human look, not a timer).
            if self.k8_enabled:
                slip = abs(result.price - trade.entry_price)
                self._slippage_history.append(slip)
                del self._slippage_history[:-self.k8_window]
                if len(self._slippage_history) >= self.k8_window:
                    specs = PAIR_SPECS.get(slot.symbol)
                    modeled = specs["spread_price"] if specs else None
                    if modeled and modeled > 0:
                        avg_slip = sum(self._slippage_history) / len(self._slippage_history)
                        if avg_slip > self.k8_multiple * modeled:
                            reason = (
                                f"K8 slippage recalibration: avg slippage {avg_slip:.5f} > "
                                f"{self.k8_multiple}x modeled {modeled:.5f} over last "
                                f"{self.k8_window} trades"
                            )
                            self.state.set_halt(reason=reason, scope="k8_slippage")
                            alert(f"K8 HALT: {reason}", book=self.book_name)
                            log.error(reason)

            # Order-integrity check (AUDIT H5): verify the fill actually
            # carries its SL/TP; repair immediately if the broker dropped
            # them. A position without a stop is never acceptable.
            live = {p.ticket: p for p in self.engine.get_positions(slot.magic)}
            p = live.get(result.ticket)
            if p is not None and (p.sl == 0 or p.tp == 0):
                repaired = self.engine.modify_sltp(result.ticket, sl=result.sl, tp=result.tp)
                log.warning(
                    f"[{slot.name}] ticket={result.ticket} filled WITHOUT "
                    f"{'SL' if p.sl == 0 else 'TP'} — repair "
                    f"{'succeeded' if repaired else 'FAILED (will retry each tick)'}"
                )
                # K7: a position without a stop is never acceptable -- if the
                # repair itself failed, don't leave it open unprotected and
                # hope the next tick's retry works; close it immediately and
                # halt (never auto-resumes -- an SL/TP repeatedly dropping on
                # fill means something's actually wrong with order handling).
                if not repaired:
                    closed = self.engine.close_position(result.ticket)
                    reason = (f"K7: ticket={result.ticket} ({slot.name}) filled without "
                              f"{'SL' if p.sl == 0 else 'TP'}, repair failed, "
                              f"{'closed' if closed else 'CLOSE ALSO FAILED'}")
                    slot.tracked.pop(result.ticket, None)
                    self.state.set_halt(reason=reason, scope="k7_unprotected")
                    alert(f"K7 HALT: {reason}", book=self.book_name)
                    log.error(reason)

    def _manage_positions(self, slot: StrategySlot, bars: pd.DataFrame | None) -> None:
        """Runs every poll tick (not just on new bars): reconciles positions
        that closed since the last check into the ledger, and ratchets SL
        forward once a position has earned enough profit to start trailing.

        `bars` is whatever _process_slot already fetched this tick for this
        slot's symbol/timeframe — reused for the skyrocket check below
        instead of fetching the same candles from MT5 a second time.

        Without the reconciliation half of this, closed_positions never gets
        written to — which silently breaks both the dashboard's trade history
        AND the daily kill switch (it sums today's P&L from that table)."""
        live = {p.ticket: p for p in self.engine.get_positions(slot.magic)}

        # Positions we were tracking that are no longer open -> closed since
        # last tick. Reconcile from MT5's deal history into the ledger.
        # Deal history can lag the close by a moment, so a miss RE-QUEUES
        # for the next tick instead of dropping the trade (AUDIT H4 — the
        # old pop-first version silently lost trades from the ledger, which
        # also blinded the daily loss guard to those losses).
        for ticket in list(slot.tracked):
            if ticket in live:
                continue
            tracked = slot.tracked[ticket]
            deal = self.engine.get_closed_deal(
                ticket, offset_fallback=int(self.state.get("broker_offset_s") or 0))
            if deal is None:
                tracked.reconcile_attempts += 1
                if tracked.reconcile_attempts < 24:  # ~2 min of retries at 5s polls
                    continue
                log.error(
                    f"[{slot.name}] ticket={ticket} closed but NO deal history after "
                    f"{tracked.reconcile_attempts} attempts — ledgering as unreconciled"
                )
                slot.tracked.pop(ticket)
                self.ledger.log_closed(
                    opened_ts=tracked.opened_ts, ticket=ticket, config_name=slot.name,
                    symbol=slot.symbol,
                    direction="long" if tracked.is_long else "short",
                    entry_price=tracked.entry, exit_price=None,
                    sl=tracked.orig_sl, tp=tracked.orig_tp, lots=tracked.lots,
                    pnl_usd=None, exit_reason="unreconciled",
                )
                continue
            slot.tracked.pop(ticket)
            self.ledger.log_closed(
                opened_ts=tracked.opened_ts,
                ticket=ticket,
                config_name=slot.name,
                symbol=slot.symbol,
                direction="long" if tracked.is_long else "short",
                entry_price=tracked.entry,
                exit_price=deal["exit_price"],
                sl=tracked.orig_sl,
                tp=tracked.orig_tp,
                lots=tracked.lots,
                pnl_usd=deal["pnl_usd"],
                exit_reason=deal["exit_reason"],
                # The REAL close time from MT5's own deal record, not "now"
                # (the default) -- without this, any reconciliation delay
                # (a laptop sleeping, a network blip, poll backlog) stamps
                # closed_ts hours late, which stretches the dashboard replay
                # window to match and makes a prompt, correct SL/TP fill
                # look like it "quit late" after wandering for hours.
                closed_ts=deal["closed_ts"].isoformat(),
            )
            # Feed the self-improving slot-health window.
            if self.slot_health_enabled and self.risk_usd > 0:
                h = self._health.setdefault(slot.name, SlotHealth())
                h.recent_r.append(deal["pnl_usd"] / self.risk_usd)

            # K4: book-wide consecutive-loss tracking (feeds _current_risk_usd).
            self._k4_record_trade_result(deal["pnl_usd"])

            # Bounded learning: post-mortem this trade, remember it, and let
            # it inform the per-session filter. This is the "learns after
            # every win/loss / answers why it happened" piece — done honestly.
            if self.learning_enabled:
                try:
                    rv = review_trade({
                        "config_name": slot.name, "symbol": slot.symbol,
                        "direction": "long" if tracked.is_long else "short",
                        "entry_price": tracked.entry, "exit_price": deal["exit_price"],
                        "sl": tracked.orig_sl, "pnl_usd": deal["pnl_usd"],
                        "exit_reason": deal["exit_reason"], "opened_ts": tracked.opened_ts,
                        "closed_ts": deal["closed_ts"],
                    }, self.risk_usd)
                    self.ledger.log_review(rv)
                    self.learner.observe(slot.name, rv.session, rv.r_multiple)
                    log.info(f"[{slot.name}] REVIEW: {rv.why}")
                except Exception as e:  # noqa: BLE001 — a review must never break trading
                    log.warning(f"[{slot.name}] trade review failed: {e}")

            log.info(
                f"[{slot.name}] CLOSED ticket={ticket} reason={deal['exit_reason']} "
                f"pnl={deal['pnl_usd']:+.2f}"
            )

        # Positions open on MT5 that we don't have local state for (bot
        # restarted mid-trade, or a stray manual position) — start tracking
        # them from their current SL so the trailing-stop logic still applies.
        for ticket, p in live.items():
            if ticket not in slot.tracked:
                # p.time is MT5's own (broker-clock) timestamp, not true UTC —
                # treating it as UTC directly used to write an opened_ts that
                # could land HOURS after the position's own (correctly true-
                # UTC) closed_ts once it later closed, showing as a negative/
                # "0m" duration and a reversed entry/exit order in the replay.
                # Stale-tick weekends fall back to the last persisted offset
                # (AUDIT M4).
                offset = self.engine.broker_offset_seconds(
                    p.symbol, fallback=int(self.state.get("broker_offset_s") or 0))
                opened_ts = datetime.fromtimestamp(p.time, tz=timezone.utc) - timedelta(seconds=offset)
                slot.tracked[ticket] = TrackedPosition(
                    entry=p.price_open, orig_sl=p.sl, orig_tp=p.tp, lots=p.volume,
                    is_long=(p.type == 0),
                    opened_ts=opened_ts.isoformat(timespec="microseconds"),
                )

        # Stop-integrity sweep (AUDIT H5): NO live position may sit without
        # a stop loss — adopted, stray, or broker-stripped. Repair from the
        # original tracked SL when we have it, else reconstruct a 1R stop
        # from the configured risk and live contract specs.
        for ticket, p in live.items():
            if p.sl != 0:
                continue
            tracked = slot.tracked.get(ticket)
            repair_sl = tracked.orig_sl if (tracked and tracked.orig_sl) else 0.0
            if repair_sl == 0.0:
                mult = self.engine.usd_per_point(slot.symbol)
                if mult and p.volume > 0:
                    dist = self.risk_usd / (p.volume * mult)
                    is_long = (p.type == 0)
                    repair_sl = p.price_open - dist if is_long else p.price_open + dist
            if repair_sl == 0.0:
                log.error(f"[{slot.name}] ticket={ticket} has NO SL and no way to derive one — check manually!")
                continue
            if self.engine.modify_sltp(ticket, sl=repair_sl, tp=p.tp):
                log.warning(f"[{slot.name}] ticket={ticket} was missing SL — repaired to {repair_sl:.5f}")
                if tracked and not tracked.orig_sl:
                    tracked.orig_sl = repair_sl
            else:
                log.warning(f"[{slot.name}] ticket={ticket} missing SL, repair FAILED — retrying next tick")

        # Hard time stop. Close anything older than the leg's max_hold_days,
        # wherever price happens to be. This is a DELIBERATE cap on hold time,
        # not a risk control: scripts/timebox_strategy_test.py showed RSI keeps
        # positive expectancy in both halves of history at a 1-2 day cap while
        # median hold falls from 18 days, and scripts/timestop_race_test.py
        # showed that raises the 30-day FTMO pass rate from 31.9% to 43.1%
        # purely by freeing position slots faster.
        if slot.max_hold_days:
            now = datetime.now(timezone.utc)
            for ticket, tracked in list(slot.tracked.items()):
                if ticket not in live:
                    continue
                try:
                    opened = datetime.fromisoformat(tracked.opened_ts)
                except (TypeError, ValueError):
                    continue
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                age_days = (now - opened).total_seconds() / 86_400.0
                if age_days < slot.max_hold_days:
                    continue
                if self.engine.close_position(ticket):
                    log.info(f"[{slot.name}] ticket={ticket} closed by TIME STOP "
                             f"after {age_days:.2f}d (max_hold_days="
                             f"{slot.max_hold_days})")
                else:
                    log.warning(f"[{slot.name}] ticket={ticket} time-stop close "
                                f"FAILED at {age_days:.2f}d — retrying next tick")

        # Ratchet trailing stop: once profit reaches trail_start_r x the
        # original risk distance, lock in trail_lock_fraction of the current
        # gain by moving SL forward. Re-checked every tick (not a one-shot
        # move), so it keeps tightening as price advances further, and only
        # ever ratchets forward — never loosens. This is what lets a trade
        # close via SL on a pullback with a smaller real profit, instead of
        # needing the full (often distant) TP to be hit exactly.
        # Per-strategy exit override: this leg's trail settings if set, else
        # the book-level default.
        trail_start = slot.trail_start_r if slot.trail_start_r is not None else self.trail_start_r
        trail_lock = slot.trail_lock_fraction if slot.trail_lock_fraction is not None else self.trail_lock_fraction
        if trail_start > 0:
            for ticket, tracked in slot.tracked.items():
                p = live.get(ticket)
                if p is None:
                    continue
                r_dist = abs(tracked.entry - tracked.orig_sl)
                if r_dist <= 0:
                    continue
                profit_dist = (p.price_current - tracked.entry) if tracked.is_long \
                    else (tracked.entry - p.price_current)
                profit_r = profit_dist / r_dist
                if profit_r < trail_start:
                    continue
                lock_r = profit_r * trail_lock
                candidate_sl = tracked.entry + lock_r * r_dist if tracked.is_long \
                    else tracked.entry - lock_r * r_dist
                improves = (candidate_sl > p.sl) if tracked.is_long else (candidate_sl < p.sl)
                if not improves:
                    continue
                if self.engine.modify_sltp(ticket, sl=candidate_sl, tp=p.tp):
                    log.info(f"[{slot.name}] ticket={ticket} SL trailed -> {candidate_sl:.5f} (locking {lock_r:.2f}R)")
                else:
                    log.warning(f"[{slot.name}] ticket={ticket} trail SL move failed")

        # Partial take (Phase 4): bank a configured fraction once the trade
        # reaches at_r, let the remainder run with the trail.
        if self.partial_enabled:
            for ticket, tracked in list(slot.tracked.items()):
                p = live.get(ticket)
                if p is None or tracked.partial_done:
                    continue
                r_dist = abs(tracked.entry - tracked.orig_sl)
                if r_dist <= 0:
                    continue
                profit_dist = (p.price_current - tracked.entry) if tracked.is_long \
                    else (tracked.entry - p.price_current)
                if profit_dist / r_dist < self.partial_at_r:
                    continue
                if self.engine.close_partial(ticket, tracked.lots * self.partial_fraction):
                    tracked.partial_done = True
                    log.info(
                        f"[{slot.name}] ticket={ticket} PARTIAL TAKE — closed "
                        f"{self.partial_fraction:.0%} at +{self.partial_at_r}R, rest runs"
                    )
                else:
                    log.warning(f"[{slot.name}] ticket={ticket} partial take failed")

        # Pyramiding (Phase 4, default OFF): one capped add into a proven
        # winner, stop at breakeven so the add can't turn the trade into a
        # net loser. Respects the central entry block and portfolio caps.
        if self.pyramid_enabled and not self._entries_blocked:
            for ticket, tracked in list(slot.tracked.items()):
                p = live.get(ticket)
                if p is None or tracked.adds >= self.pyramid_max_adds:
                    continue
                r_dist = abs(tracked.entry - tracked.orig_sl)
                if r_dist <= 0:
                    continue
                profit_dist = (p.price_current - tracked.entry) if tracked.is_long \
                    else (tracked.entry - p.price_current)
                if profit_dist / r_dist < self.pyramid_trigger_r:
                    continue
                if (self.max_concurrent_positions_total > 0
                        and self._total_open_positions() + 1 > self.max_concurrent_positions_total):
                    continue
                add_lots = tracked.lots * self.pyramid_add_fraction
                vol_min, vol_max, vol_step = self.engine.volume_spec(slot.symbol)
                add_lots = normalize_volume(add_lots, vol_min, vol_max, vol_step)
                if add_lots is None:
                    continue
                direction = "long" if tracked.is_long else "short"
                tracked.adds += 1  # count the attempt — never retry-loop an add
                result = self.engine.place_market_order(
                    symbol=slot.symbol, direction=direction, lots=add_lots,
                    sl=tracked.entry, tp=tracked.orig_tp, magic=slot.magic,
                    comment=f"{slot.name}_pyr",
                )
                self.ledger.log_order(
                    signal_id=None, config_name=slot.name, symbol=slot.symbol,
                    direction=direction, requested_price=p.price_current,
                    filled_price=result.price, requested_lots=add_lots,
                    filled_lots=result.volume, sl=result.sl, tp=result.tp,
                    ticket=result.ticket, retcode=result.retcode,
                    comment=f"pyramid: {result.comment}",
                )
                if result.ok:
                    slot.tracked[result.ticket] = TrackedPosition(
                        entry=result.price, orig_sl=result.sl, orig_tp=result.tp,
                        lots=result.volume, is_long=tracked.is_long,
                        opened_ts=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                        adds=self.pyramid_max_adds,  # an add never pyramids again
                    )
                    log.info(f"[{slot.name}] PYRAMID add {result.volume} lots at +{self.pyramid_trigger_r}R (BE stop)")

        # Skyrocket exit: an outsized favorable candle closes the trade now,
        # rather than waiting for the (often much smaller) fixed TP, which
        # can otherwise cap the win or let price round-trip back down before
        # ever reaching it.
        skyrocket = slot.skyrocket_multiple if slot.skyrocket_multiple is not None else self.skyrocket_multiple
        if skyrocket > 0 and bars is not None and len(bars) >= 21:
            for ticket, tracked in list(slot.tracked.items()):
                p = live.get(ticket)
                if p is None:
                    continue
                if self._is_skyrocket_move(bars, tracked, p, skyrocket):
                    if self.engine.close_position(ticket):
                        log.info(
                            f"[{slot.name}] ticket={ticket} SKYROCKET EXIT — "
                            f"closed early, profit={p.profit:+.2f}"
                        )
                    else:
                        log.warning(f"[{slot.name}] ticket={ticket} skyrocket close failed")

    def _is_skyrocket_move(self, bars: pd.DataFrame, tracked: TrackedPosition, p,
                           skyrocket_multiple: float | None = None) -> bool:
        """True if the current (possibly still-forming) candle is an outsized
        move in this position's favor: range >= skyrocket_multiple x the
        recent average candle range, closing in the favorable direction,
        with the trade currently in profit. `bars` is the slot's already-
        fetched raw fetch (indexed by timestamp, still-forming bar included)
        from _process_slot this same tick — no separate MT5 call needed."""
        closed_bars = bars.iloc[:-1]
        avg_range = float((closed_bars["high"] - closed_bars["low"]).tail(20).mean())
        if avg_range <= 0:
            return False
        current = bars.iloc[-1]
        current_range = float(current["high"] - current["low"])
        mult = skyrocket_multiple if skyrocket_multiple is not None else self.skyrocket_multiple
        if current_range < mult * avg_range:
            return False
        bullish = current["close"] > current["open"]
        favorable = bullish if tracked.is_long else (not bullish)
        return favorable and p.profit > 0

    def _reconcile_orphaned_tickets(self) -> None:
        """Positions whose owning strategy/slot no longer exists in the
        current config (e.g. after a roster swap removed that leg) are
        invisible to _manage_positions -- it only reconciles tickets in a
        live slot's `tracked` dict. If one of those orphaned positions gets
        closed (by the user, or broker-side SL/TP), it would otherwise never
        land in closed_positions, silently blinding the dashboard and the
        daily-loss guard to it. Sweep the ledger's own order history for
        tickets with no closed_positions row that also aren't currently
        tracked by any live slot, and reconcile any MT5 confirms are closed.

        Cheap in the common case: normally 0 rows once caught up, since this
        only ever finds NEW orphans (a slot's own tracked tickets are
        excluded, and anything already reconciled here stays reconciled)."""
        tracked_tickets = {t for slot in self.slots for t in slot.tracked}
        order_tickets = {
            r[0] for r in self.ledger.conn.execute(
                "SELECT DISTINCT ticket FROM orders WHERE ticket != 0")
        }
        closed_tickets = {
            r[0] for r in self.ledger.conn.execute(
                "SELECT DISTINCT ticket FROM closed_positions")
        }
        orphaned = order_tickets - closed_tickets - tracked_tickets
        if not orphaned:
            return

        try:
            still_open = self.engine.get_all_open_tickets()
        except Exception as e:
            log.warning(f"orphan-reconcile: could not fetch open tickets: {e}")
            return

        for ticket in orphaned - still_open:
            try:
                row = self.ledger.conn.execute(
                    "SELECT ts, config_name, symbol, direction, filled_price, sl, tp, filled_lots "
                    "FROM orders WHERE ticket = ? ORDER BY ts LIMIT 1", (ticket,)
                ).fetchone()
                if row is None:
                    continue
                opened_ts, config_name, symbol, direction, entry_price, sl, tp, lots = row
                deal = self.engine.get_closed_deal(
                    ticket, offset_fallback=int(self.state.get("broker_offset_s") or 0))
                if deal is None:
                    continue
                self.ledger.log_closed(
                    opened_ts=opened_ts, ticket=ticket, config_name=config_name, symbol=symbol,
                    direction=direction, entry_price=entry_price, exit_price=deal["exit_price"],
                    sl=sl, tp=tp, lots=lots, pnl_usd=deal["pnl_usd"],
                    exit_reason=deal["exit_reason"], closed_ts=deal["closed_ts"].isoformat(),
                )
                log.info(
                    f"[orphan-reconcile] ticket={ticket} ({config_name}) closed "
                    f"pnl={deal['pnl_usd']:.2f} reason={deal['exit_reason']}"
                )
            except Exception as e:  # noqa: BLE001 — one bad ticket must not skip the rest
                log.warning(f"orphan-reconcile: ticket={ticket} failed: {e}")

    def _persist_runtime_state(self) -> None:
        """Everything worth surviving a crash, refreshed once a minute:
        pending setups (AUDIT M8) and the last known-good broker offset."""
        pending: dict[str, list] = {}
        for slot in self.slots:
            sigs = getattr(slot.strategy, "pending", None)
            if sigs:
                pending[slot.name] = [
                    {**asdict(s), "direction": s.direction.value} for s in sigs
                ]
        self.state.set("pending_signals", pending)
        if self.slots:
            offset = self.engine.broker_offset_seconds(
                self.slots[0].symbol, fallback=int(self.state.get("broker_offset_s") or 0))
            self.state.set("broker_offset_s", offset)
        self.state.save()

    def _heartbeat(self) -> None:
        self._persist_runtime_state()
        self._reconcile_orphaned_tickets()
        try:
            acc = self.engine.account_info()
        except Exception as e:
            log.warning(f"heartbeat account_info error: {e}")
            return

        today = datetime.now(timezone.utc).date().isoformat()
        daily = self.ledger.daily_pnl(today)
        total_open = sum(len(self.engine.get_positions(s.magic)) for s in self.slots)

        self.ledger.log_heartbeat(
            equity=acc.equity,
            balance=acc.balance,
            open_positions=total_open,
            daily_pnl=daily,
        )
        status = f" [BLOCKED: {self._entries_blocked}]" if self._entries_blocked else ""
        log.info(
            f"HEARTBEAT equity={acc.equity:.2f} bal={acc.balance:.2f} "
            f"open={total_open} daily_pnl={daily:+.2f} "
            f"book_pnl={self._realized_alltime + self._floating:+.2f}{status}"
        )
