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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Type

import pandas as pd

from metatraderbot.ledger import Ledger
from metatraderbot.mt5_engine import MT5Engine
from metatraderbot.strategies_bollinger import BollingerBounceStrategy
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy
from metatraderbot.strategies_ma_cross import SMACrossStrategy
from metatraderbot.strategies_macd import MACDTrendStrategy
from metatraderbot.strategies_rsi import RSIReversionStrategy
from metatraderbot.strategies_stochastic import StochasticStrategy
from metatraderbot.strategy import NoWickStrategy, TradeSpec

log = logging.getLogger(__name__)

STRATEGY_CLASSES: dict[str, Type] = {
    "NoWick": NoWickStrategy,
    "BollingerBounce": BollingerBounceStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "MACDTrend": MACDTrendStrategy,
    "RSIReversion": RSIReversionStrategy,
    "SMACross": SMACrossStrategy,
    "Stochastic": StochasticStrategy,
}

WARMUP_BARS = 300


@dataclass
class TrackedPosition:
    entry: float
    orig_sl: float
    orig_tp: float
    lots: float
    is_long: bool
    opened_ts: str


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

    def __post_init__(self):
        if self.tracked is None:
            self.tracked = {}


class PortfolioRunner:
    def __init__(self, portfolio_cfg: dict, engine: MT5Engine, ledger: Ledger, control_path: Path | None = None):
        self.cfg = portfolio_cfg
        self.engine = engine
        self.ledger = ledger
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

        self.poll_interval_s = int(
            portfolio_cfg.get("mt5", {}).get("poll_interval_s", 5)
        )

        self.slots: list[StrategySlot] = []
        self.starting_equity: float = 0.0
        self._stopped = False

    def initialize(self) -> None:
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

        for entry in self.cfg["configs"]:
            self.slots.append(self._build_slot(entry))
            log.info(
                f"  slot ready: {entry['name']} = {entry['symbol']} "
                f"{entry['strategy']} {entry['timeframe']} magic={entry['magic_number']}"
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

        return StrategySlot(
            name=entry["name"],
            symbol=entry["symbol"],
            strategy_name=entry["strategy"],
            timeframe=entry["timeframe"],
            magic=int(entry["magic_number"]),
            strategy=strategy,
            last_bar_ts=last_ts,
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
            },
        }
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
                if not self._kill_switches_ok():
                    log.warning("Kill switch tripped, stopping.")
                    break

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
            self.engine.disconnect()
            self.ledger.close()
            log.info("Shutdown complete.")

    def stop(self):
        self._stopped = True

    def _kill_switches_ok(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        daily = self.ledger.daily_pnl(today)
        if daily <= self.daily_stop_usd:
            log.error(f"Daily stop hit: ${daily:.2f} <= ${self.daily_stop_usd}")
            return False
        try:
            acc = self.engine.account_info()
        except Exception as e:
            log.warning(f"account_info() failed: {e}")
            return True
        if self.starting_equity - acc.equity >= abs(self.max_dd_usd):
            log.error(
                f"Total DD hit: equity={acc.equity:.2f} vs start={self.starting_equity:.2f}"
            )
            return False
        return True

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
        slot.last_bar_ts = latest_ts

        for bar_index in new_bar_indices:
            if self.ledger.already_have_signal(slot.name, bar_index):
                continue

            open_positions = self.engine.get_positions(slot.magic)
            if len(open_positions) >= self.max_positions:
                break  # full up — no point replaying the rest

            try:
                trades = slot.strategy.on_bar(bar_index, bars)
            except Exception as e:
                log.exception(f"[{slot.name}] strategy.on_bar error: {e}")
                continue

            if not self._trading_enabled():
                continue  # scanning paused — strategy state stays current, but no new orders

            for trade in trades:
                self._execute_trade(slot, trade, bar_index)

        return raw

    def _execute_trade(self, slot: StrategySlot, trade: TradeSpec, bar_index: int) -> None:
        lots = round(max(0.01, trade.lots), 2)
        direction = trade.direction.value

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
        )

        log.info(
            f"[{slot.name}] SIGNAL {direction.upper()} {lots} lots "
            f"@ mkt SL={trade.stop_loss:.5f} TP={trade.take_profit:.5f}"
        )

        result = self.engine.place_market_order(
            symbol=slot.symbol,
            direction=direction,
            lots=lots,
            sl=trade.stop_loss,
            tp=trade.take_profit,
            magic=slot.magic,
            comment=slot.name,
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
        log.info(
            f"[{slot.name}] ORDER {tag} ticket={result.ticket} "
            f"filled_price={result.price} {result.comment}"
        )

        if result.ok:
            slot.tracked[result.ticket] = TrackedPosition(
                entry=result.price, orig_sl=result.sl, orig_tp=result.tp,
                lots=result.volume, is_long=(direction == "long"),
                opened_ts=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            )

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
        for ticket in list(slot.tracked):
            if ticket in live:
                continue
            tracked = slot.tracked.pop(ticket)
            deal = self.engine.get_closed_deal(ticket)
            if deal is None:
                log.warning(f"[{slot.name}] ticket={ticket} closed but no deal history found")
                continue
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
            )
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
                offset = self.engine.broker_offset_seconds(p.symbol)
                opened_ts = datetime.fromtimestamp(p.time, tz=timezone.utc) - timedelta(seconds=offset)
                slot.tracked[ticket] = TrackedPosition(
                    entry=p.price_open, orig_sl=p.sl, orig_tp=p.tp, lots=p.volume,
                    is_long=(p.type == 0),
                    opened_ts=opened_ts.isoformat(timespec="microseconds"),
                )

        # Ratchet trailing stop: once profit reaches trail_start_r x the
        # original risk distance, lock in trail_lock_fraction of the current
        # gain by moving SL forward. Re-checked every tick (not a one-shot
        # move), so it keeps tightening as price advances further, and only
        # ever ratchets forward — never loosens. This is what lets a trade
        # close via SL on a pullback with a smaller real profit, instead of
        # needing the full (often distant) TP to be hit exactly.
        if self.trail_start_r > 0:
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
                if profit_r < self.trail_start_r:
                    continue
                lock_r = profit_r * self.trail_lock_fraction
                candidate_sl = tracked.entry + lock_r * r_dist if tracked.is_long \
                    else tracked.entry - lock_r * r_dist
                improves = (candidate_sl > p.sl) if tracked.is_long else (candidate_sl < p.sl)
                if not improves:
                    continue
                if self.engine.modify_sltp(ticket, sl=candidate_sl, tp=p.tp):
                    log.info(f"[{slot.name}] ticket={ticket} SL trailed -> {candidate_sl:.5f} (locking {lock_r:.2f}R)")
                else:
                    log.warning(f"[{slot.name}] ticket={ticket} trail SL move failed")

        # Skyrocket exit: an outsized favorable candle closes the trade now,
        # rather than waiting for the (often much smaller) fixed TP, which
        # can otherwise cap the win or let price round-trip back down before
        # ever reaching it.
        if self.skyrocket_multiple > 0 and bars is not None and len(bars) >= 21:
            for ticket, tracked in list(slot.tracked.items()):
                p = live.get(ticket)
                if p is None:
                    continue
                if self._is_skyrocket_move(bars, tracked, p):
                    if self.engine.close_position(ticket):
                        log.info(
                            f"[{slot.name}] ticket={ticket} SKYROCKET EXIT — "
                            f"closed early, profit={p.profit:+.2f}"
                        )
                    else:
                        log.warning(f"[{slot.name}] ticket={ticket} skyrocket close failed")

    def _is_skyrocket_move(self, bars: pd.DataFrame, tracked: TrackedPosition, p) -> bool:
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
        if current_range < self.skyrocket_multiple * avg_range:
            return False
        bullish = current["close"] > current["open"]
        favorable = bullish if tracked.is_long else (not bullish)
        return favorable and p.profit > 0

    def _heartbeat(self) -> None:
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
        log.info(
            f"HEARTBEAT equity={acc.equity:.2f} bal={acc.balance:.2f} "
            f"open={total_open} daily_pnl={daily:+.2f}"
        )
