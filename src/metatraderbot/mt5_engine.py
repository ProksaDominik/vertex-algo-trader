"""MT5 desktop connection layer.

Wraps the MetaTrader5 Python API for paper trading. Handles:
- Connecting to a running MT5 terminal
- Resolving broker-specific symbol names (XAUUSD vs XAUUSDm vs XAUUSD.a etc.)
- Fetching historical + latest bars for strategy warm-up and live decisions
- Reading account info + open positions
- Placing market orders with SL/TP

Only usable on Windows (where MT5 desktop runs); the import guards make
the module quiet if MetaTrader5 isn't installed.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
    from metatraderbot.mt5_lock import install_mt5_lock
    install_mt5_lock(mt5)  # serialize IPC calls across the Funded/Trial/dashboard processes
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

log = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "1m":  "TIMEFRAME_M1",
    "5m":  "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h":  "TIMEFRAME_H1",
    "4h":  "TIMEFRAME_H4",
    "1d":  "TIMEFRAME_D1",
}


@dataclass
class AccountInfo:
    login: int
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    is_demo: bool
    # ACCOUNT_MARGIN_MODE_*: 0=netting, 1=exchange, 2=retail hedging.
    # This codebase runs 39 concurrent slots, several on the SAME symbol —
    # on a netting account those merge into one position with shared SL/TP
    # and per-magic tracking falls apart (AUDIT M1), so the runner asserts
    # hedging mode at startup.
    margin_mode: int = 2

    @property
    def is_hedging(self) -> bool:
        return self.margin_mode == 2


@dataclass
class OrderResult:
    ticket: int
    volume: float
    price: float
    sl: float
    tp: float
    retcode: int
    comment: str

    @property
    def ok(self) -> bool:
        # 10009 DONE, 10010 DONE_PARTIAL — a partial fill still means a real
        # position exists and MUST be tracked (treating it as failure left
        # live positions invisible to the runner; AUDIT H3).
        # 0 is ALSO success on this broker's bridge (MetaQuotes-Demo) --
        # confirmed empirically 2026-08-20: order_send returns retcode=0,
        # comment="Done" on a genuinely filled order (real deal/order
        # tickets, position actually opens/closes). Not a documented MQL5
        # code; this broker just doesn't follow the standard numbering.
        # Treating 0 as failure (the bug before this fix) meant every real
        # fill got logged as a failure and never tracked by the runner.
        #
        # BUT retcode=0 alone is not enough (fixed 2026-08-21): a REJECTED
        # order can also come back as retcode=0 with an all-zero body, and
        # accepting those made the runner log phantom "opened" trades and
        # fire Telegram alerts for positions that never existed on the
        # broker. A genuine fill always carries a non-zero order ticket, so
        # require one before believing an ambiguous 0.
        if self.retcode in (10009, 10010):
            return True
        return self.retcode == 0 and self.ticket != 0

    @property
    def partial(self) -> bool:
        return self.retcode == 10010


# Retcodes worth retrying: transient market/infrastructure conditions that
# typically clear within seconds. Everything else is a permanent rejection
# for this attempt (bad request, market closed, not enough money, ...).
TRANSIENT_RETCODES = {
    10004,  # REQUOTE
    10020,  # PRICE_CHANGED
    10021,  # PRICE_OFF (off quotes)
    10024,  # TOO_MANY_REQUESTS
    10031,  # CONNECTION (no connection to trade server)
}


def normalize_volume(lots: float, vol_min: float, vol_max: float, vol_step: float) -> Optional[float]:
    """Snap a lot size to the broker's volume constraints.

    Floors to the nearest step (never round UP — rounding up silently
    increases risk), clamps to vol_max, and returns None when the ideal
    size is below vol_min (caller decides whether to skip the trade or
    floor to vol_min — flooring inflates risk and must be a deliberate,
    configured choice; AUDIT H7).
    """
    if lots <= 0 or vol_step <= 0:
        return None
    stepped = int(lots / vol_step + 1e-9) * vol_step
    stepped = round(stepped, 8)
    if stepped > vol_max > 0:
        stepped = vol_max
    if stepped < vol_min:
        return None
    return stepped


class MT5Engine:
    def __init__(self, symbol_map: Optional[dict] = None):
        if not _MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5   (Windows only)"
            )
        self.symbol_map = symbol_map or {}
        self._connected = False
        self._resolved: dict[str, str] = {}
        self._filling_mode: dict[str, int] = {}
        self._broker_offset: int = 0   # last known-good broker-clock offset, see _cached_offset

    # --- connection lifecycle ---

    def connect(self) -> AccountInfo:
        if self._connected:
            return self.account_info()

        login_env = os.environ.get("MT5_LOGIN")
        password = os.environ.get("MT5_PASSWORD")
        server = os.environ.get("MT5_SERVER")
        mt5_path = os.environ.get("MT5_PATH")

        expected_login: Optional[int] = None
        if login_env:
            try:
                expected_login = int(login_env)
            except ValueError:
                raise RuntimeError(f"MT5_LOGIN must be integer, got: {login_env!r}")

        errors: list[str] = []

        # ATTEMPT 1: attach to a running MT5 session (fastest path when MT5 is
        # already logged in — no re-authentication, no IPC timeout).
        log.info("Attempt 1/2: attach to running MT5 session (no creds)")
        if mt5.initialize():
            info = mt5.account_info()
            if info is not None:
                if expected_login is None or info.login == expected_login:
                    log.info(f"Attached OK: login={info.login} server={info.server}")
                    self._connected = True
                    return self._to_acc(info)
                else:
                    log.warning(
                        f"Attached to wrong account {info.login}, expected {expected_login}. "
                        f"Shutting down and trying explicit credentials."
                    )
            else:
                log.warning("Attached but account_info was None; trying explicit creds.")
            mt5.shutdown()
        else:
            errors.append(f"attach: {mt5.last_error()}")
            log.info(f"Attach failed: {mt5.last_error()} — trying explicit credentials")

        # ATTEMPT 2: launch/relaunch MT5 with explicit credentials.
        if not (expected_login and password and server):
            raise RuntimeError(
                "Cannot connect to MT5: no running session AND no MT5_LOGIN/PASSWORD/SERVER "
                "in .env. Fix: either (a) open MT5 desktop and log in manually, then run again, "
                f"or (b) set the three env vars. Prior errors: {errors}"
            )

        log.info(
            f"Attempt 2/2: launch MT5 with explicit credentials login={expected_login} "
            f"server={server}" + (f" path={mt5_path}" if mt5_path else "")
        )
        init_kwargs = dict(login=expected_login, password=password, server=server, timeout=60000)
        if mt5_path:
            ok = mt5.initialize(mt5_path, **init_kwargs)
        else:
            ok = mt5.initialize(**init_kwargs)

        if not ok:
            errors.append(f"launch: {mt5.last_error()}")
            raise RuntimeError(
                f"Both MT5 attach and explicit-creds launch failed. "
                f"Errors: {errors}. "
                f"Try: (a) open MT5 desktop manually, log in to {expected_login} on {server}, "
                f"wait until Market Watch shows live prices, then re-run. "
                f"(b) verify the password in .env is your MT5 MASTER password, not investor."
            )

        self._connected = True
        return self.account_info()

    def _to_acc(self, info) -> AccountInfo:
        return AccountInfo(
            login=info.login,
            server=info.server,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            currency=info.currency,
            is_demo=(info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO),
            margin_mode=getattr(info, "margin_mode", 2),
        )

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    @staticmethod
    def is_ipc_error(err) -> bool:
        """Whether an mt5.last_error() tuple / string looks like a transient IPC failure."""
        if err is None:
            return False
        code = None
        if isinstance(err, tuple) and err:
            code = err[0]
        elif isinstance(err, (int, float)):
            code = int(err)
        elif isinstance(err, str):
            for token in ("-10001", "-10005", "IPC send failed", "IPC timeout"):
                if token in err:
                    return True
        return code in (-10001, -10005)

    def reconnect(self) -> bool:
        """Reset the connection. Returns True if reconnected + account_info readable."""
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False
        try:
            self.connect()
            # Verify by reading account info
            info = mt5.account_info()
            return info is not None
        except Exception as e:
            log.warning(f"Reconnect failed: {e}")
            return False

    def account_info(self) -> AccountInfo:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        return self._to_acc(info)

    # --- symbol resolution ---

    def resolve_symbol(self, canonical: str) -> str:
        """Find the broker's actual symbol name for the canonical name (e.g. XAUUSD)."""
        if canonical in self._resolved:
            return self._resolved[canonical]

        # Explicit config override
        if canonical in self.symbol_map:
            override = self.symbol_map[canonical]
            if mt5.symbol_info(override) is not None:
                self._resolved[canonical] = override
                return override
            raise ValueError(
                f"symbol_map[{canonical}]={override!r} not found on broker."
            )

        # Try canonical + common broker suffixes
        candidates = [
            canonical,
            f"{canonical}.a", f"{canonical}.raw", f"{canonical}_raw",
            f"{canonical}m", f"{canonical}pro", f"{canonical}c",
        ]
        for c in candidates:
            if mt5.symbol_info(c) is not None:
                self._resolved[canonical] = c
                log.info(f"Resolved {canonical} -> {c}")
                return c

        # Fuzzy fallback
        all_syms = mt5.symbols_get() or []
        base = canonical[:3]
        quote = canonical[3:6] if len(canonical) >= 6 else ""
        matches = [s.name for s in all_syms if base in s.name and (not quote or quote in s.name)]
        raise ValueError(
            f"Cannot resolve {canonical}. "
            f"Fuzzy matches: {matches[:10]}"
        )

    # --- bars ---

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        broker_sym = self.resolve_symbol(symbol)
        tf_attr = TIMEFRAME_MAP.get(timeframe)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe {timeframe}")
        tf = getattr(mt5, tf_attr)

        if not mt5.symbol_select(broker_sym, True):
            log.warning(f"symbol_select({broker_sym}) failed")

        # copy_rates_from_pos(sym, tf, 0, N) gives us the last N bars, position 0 = most recent
        # (which may still be forming). Runner drops the last row for signal decisions.
        rates = mt5.copy_rates_from_pos(broker_sym, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"copy_rates_from_pos({broker_sym}, {timeframe}) empty: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "tick_volume"]]
        df = df.rename(columns={"tick_volume": "volume"})
        df = df.set_index("timestamp")

        # Bar timestamps arrive on the BROKER's clock, not UTC (this server runs
        # UTC+3). Correct them here, at the single point every caller goes
        # through, so strategies never have to know about it.
        #
        # This was a live/backtest divergence, not a cosmetic issue. The
        # historical parquet files are true UTC (their weekly gap runs Fri 20:45
        # -> Sun 22:00, matching the real FX week), so every session strategy was
        # BACKTESTED on true UTC but ran LIVE on broker time. LondonBreakout's
        # 07:00-10:00 "London" window was really firing 04:00-07:00 UTC -- the
        # tail of the Asian session, three hours before London opens. It was
        # measuring the wrong range and breaking out of it at the wrong time.
        offset = self._cached_offset(broker_sym)
        if offset:
            df.index = df.index - timedelta(seconds=offset)
        return df

    def _cached_offset(self, broker_sym: str) -> int:
        """Broker-clock offset, remembered across calls.

        broker_offset_seconds() needs a FRESH tick to measure against, and
        returns its fallback when the market is shut. Caching the last
        known-good value means bars stay correctly stamped over a weekend
        instead of silently reverting to an uncorrected clock -- which would
        make the correction come and go with the session.
        """
        measured = self.broker_offset_seconds(broker_sym, fallback=0)
        if measured:
            self._broker_offset = measured
        return getattr(self, "_broker_offset", 0)

    # --- positions + orders ---

    def get_positions(self, magic: int) -> list:
        positions = mt5.positions_get() or []
        return [p for p in positions if p.magic == magic]

    def get_position(self, ticket: int):
        """The open position with this ticket, or None once it is closed."""
        positions = mt5.positions_get(ticket=ticket)
        return positions[0] if positions else None

    def get_all_open_tickets(self) -> set[int]:
        """Every currently-open ticket on the account, regardless of magic --
        used to check whether an orphaned ledger ticket (one whose owning
        strategy/slot no longer exists in the config) is actually closed
        before treating it as such (AUDIT: calling get_closed_deal on a
        still-open ticket misreads its entry deal as the exit)."""
        positions = mt5.positions_get() or []
        return {p.ticket for p in positions}

    def broker_offset_seconds(self, symbol: str, fallback: int = 0) -> int:
        """Broker-server clock minus true UTC, measured via a live tick.
        MT5 position/deal timestamps (`p.time`, `deal.time`) are in the
        broker's clock, not true UTC (historically several hours off) — any
        caller treating one of those as UTC directly needs this correction
        first, or ends up writing a timestamp that's hours late into the
        ledger (which is exactly what happened before this existed: a
        reconciled position's opened_ts landed *after* its own closed_ts).

        A STALE tick (weekend, market closed) would compute a garbage offset
        of hours (AUDIT M4) — detected via tick age; `fallback` (the last
        known-good offset, persisted by the runner) is returned instead."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time:
            return fallback
        broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        offset = int((broker_now - datetime.now(timezone.utc)).total_seconds())
        # Real broker offsets are a few hours at most; a tick from hours ago
        # (weekend) shows up as a huge negative offset. 6h magnitude cutoff.
        if abs(offset) > 6 * 3600:
            return fallback
        return offset

    def stops_min_distance(self, symbol: str) -> float:
        """Broker's minimum SL/TP distance from price, in PRICE units
        (SYMBOL_TRADE_STOPS_LEVEL is quoted in points). Used to clamp entry
        stops and to skip trailing modifies that would be rejected with
        10016 Invalid stops (AUDIT M5)."""
        try:
            broker_sym = self.resolve_symbol(symbol)
            info = mt5.symbol_info(broker_sym)
        except ValueError:
            info = None
        if info is None:
            return 0.0
        level = getattr(info, "trade_stops_level", 0) or 0
        point = getattr(info, "point", 0.0) or 0.0
        return level * point

    def terminal_health(self, sample_symbol: str | None = None) -> dict:
        """Watchdog probe: terminal connectivity, AlgoTrading switch, and
        tick freshness for one sample symbol. Never raises."""
        out = {"connected": False, "trade_allowed": False, "tick_age_s": None}
        try:
            term = mt5.terminal_info()
            if term is not None:
                out["connected"] = bool(getattr(term, "connected", False))
                out["trade_allowed"] = bool(getattr(term, "trade_allowed", False))
            if sample_symbol:
                tick = mt5.symbol_info_tick(self.resolve_symbol(sample_symbol))
                if tick is not None and tick.time:
                    age = datetime.now(timezone.utc) - datetime.fromtimestamp(tick.time, tz=timezone.utc)
                    # Broker clock offset can make this negative; clamp at 0.
                    out["tick_age_s"] = max(0.0, age.total_seconds())
        except Exception as e:  # noqa: BLE001 — health probe must not throw
            log.warning(f"terminal_health probe error: {e}")
        return out

    def get_spread(self, symbol: str) -> Optional[float]:
        """Current bid/ask spread in PRICE units (K6 spread-anomaly probe).
        None when there's no live tick to read."""
        try:
            broker_sym = self.resolve_symbol(symbol)
            tick = mt5.symbol_info_tick(broker_sym)
        except ValueError:
            tick = None
        if tick is None or tick.ask <= 0 or tick.bid <= 0:
            return None
        return float(tick.ask - tick.bid)

    def close_partial(self, ticket: int, volume: float) -> bool:
        """Close part of a position (partial take-profit). Volume is snapped
        to the symbol's step; if the remainder would fall below volume_min,
        the whole position is closed instead (brokers reject dust)."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        p = positions[0]
        vol_min, _vol_max, vol_step = self.volume_spec(p.symbol)
        vol = normalize_volume(volume, vol_min, p.volume, vol_step)
        if vol is None:
            return False
        if p.volume - vol < vol_min:
            return self.close_position(ticket)
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return False
        result = None
        for filling in self._filling_candidates(p.symbol):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": vol,
                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask,
                "deviation": 20,
                "magic": p.magic,
                "comment": "partial_take",
                "type_filling": filling,
            }
            result = mt5.order_send(**request)
            if result is not None and result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                self._filling_mode[p.symbol] = filling
                break
        # Requires a real deal ticket when retcode==0 -- see close_position().
        return result is not None and (
            result.retcode in (10009, 10010)
            or (result.retcode == 0 and (getattr(result, "deal", 0)
                                         or getattr(result, "order", 0))))

    def usd_per_point(self, symbol: str) -> Optional[float]:
        """Account-currency P&L per lot per 1.0 price-unit move, from LIVE
        broker specs: tick_value / tick_size (the two are NOT interchangeable
        — tick_value is money per tick per lot, tick_size is the price
        increment of one tick; dividing converts to money per 1.0 of price).
        Replaces the frozen approximations in instruments.py for live sizing
        (AUDIT C5). Returns None when the info isn't available so the caller
        can fall back explicitly instead of sizing off garbage."""
        try:
            broker_sym = self.resolve_symbol(symbol)
        except ValueError:
            return None
        info = mt5.symbol_info(broker_sym)
        if info is None:
            return None
        tick_value = getattr(info, "trade_tick_value", 0.0) or 0.0
        tick_size = getattr(info, "trade_tick_size", 0.0) or 0.0
        if tick_value <= 0 or tick_size <= 0:
            return None
        live = tick_value / tick_size

        # --- drift gate against the frozen table ----------------------------
        #
        # The arithmetic above is right; the BROKER'S INPUT CAN BE WRONG. This
        # server reports trade_tick_value=0.1 for XAUUSD, giving 10.0 USD per
        # price unit per lot. Live P&L on four open gold positions proves the
        # true figure is 100.0: -1350.30 USD over a 192.90 move on 0.07 lots is
        # exactly 100/point/lot. Sizing off the broker's number therefore built
        # gold positions TEN TIMES too large -- ~4,722 USD of risk each where
        # 500 was intended, against a 2,000 USD daily loss limit.
        #
        # So the two independent sources are compared, and on material
        # disagreement the LARGER multiplier wins, because a larger multiplier
        # yields a SMALLER position. Under-sizing costs a little profit;
        # over-sizing loses the account. Never silently prefer the broker.
        try:
            from metatraderbot.instruments import contract_mult
            frozen = float(contract_mult(symbol) or 0.0)
        except Exception:
            frozen = 0.0
        if frozen > 0 and max(live, frozen) / min(live, frozen) >= 2.0:
            safe = max(live, frozen)
            log.error(
                "CONTRACT MULTIPLIER DRIFT on %s: broker tick_value/tick_size "
                "gives %.4f, instruments.contract_mult gives %.4f (%.1fx apart). "
                "Using %.4f -- the larger value, which sizes SMALLER. Verify "
                "against a real fill before trusting either.",
                symbol, live, frozen, max(live, frozen) / min(live, frozen), safe)
            return safe
        return live

    def volume_spec(self, symbol: str) -> tuple[float, float, float]:
        """(volume_min, volume_max, volume_step) from live symbol info,
        with conservative retail-FX defaults when unavailable."""
        try:
            broker_sym = self.resolve_symbol(symbol)
            info = mt5.symbol_info(broker_sym)
        except ValueError:
            info = None
        if info is None:
            return (0.01, 100.0, 0.01)
        return (
            getattr(info, "volume_min", 0.01) or 0.01,
            getattr(info, "volume_max", 100.0) or 100.0,
            getattr(info, "volume_step", 0.01) or 0.01,
        )

    def _filling_candidates(self, broker_sym: str) -> list[int]:
        """Order-type_filling candidates to try for this symbol, best guess first.

        symbol_info().filling_mode is a bitmask over SYMBOL_FILLING_FOK(1) /
        SYMBOL_FILLING_IOC(2) — NOT the same enum as the order request's
        type_filling field (ORDER_FILLING_FOK=0, IOC=1, RETURN=2), so the bit
        has to be translated rather than reused directly. The MetaTrader5
        python package doesn't expose SYMBOL_FILLING_* as named constants, so
        the bit values are hardcoded per MQL5's ENUM_SYMBOL_FILLING_MODE.
        Hardcoding IOC caused most symbols here to reject every order with
        retcode 10030 ("Unsupported filling mode") since several of them
        (e.g. USDJPY, USDCHF, GBPUSD, AUDUSD — filling_mode=1) only accept
        FOK, while others (XAUUSD, NZDUSD — filling_mode=3) accept both.
        """
        if broker_sym in self._filling_mode:
            return [self._filling_mode[broker_sym]]

        info = mt5.symbol_info(broker_sym)
        mask = getattr(info, "filling_mode", 0) if info else 0
        SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
        ordered: list[int] = []
        if mask & SYMBOL_FILLING_FOK:
            ordered.append(mt5.ORDER_FILLING_FOK)
        if mask & SYMBOL_FILLING_IOC:
            ordered.append(mt5.ORDER_FILLING_IOC)
        ordered.append(mt5.ORDER_FILLING_RETURN)  # always a valid fallback to try
        # De-dupe while preserving order.
        seen = set()
        return [m for m in ordered if not (m in seen or seen.add(m))]

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        lots: float,
        sl: float,
        tp: float,
        magic: int,
        comment: str = "",
        signal_entry_price: float | None = None,
        deviation_points: int = 20,
        max_retries: int = 3,
    ) -> OrderResult:
        broker_sym = self.resolve_symbol(symbol)
        tick = mt5.symbol_info_tick(broker_sym)
        info = mt5.symbol_info(broker_sym)
        if tick is None or info is None:
            raise RuntimeError(f"No tick/info for {broker_sym}")

        # Indices in particular (US30, DE40, ...) go SYMBOL_TRADE_MODE_DISABLED
        # outside their own session even though ticks keep updating, which
        # otherwise shows up as a live signal firing, an order_send round
        # trip, and a "Trade disabled" rejection every time — same lost
        # signal either way, but skipping the doomed order_send call entirely
        # gives a clear, honest reason in the log/ledger instead of a bare
        # broker rejection code.
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return OrderResult(0, lots, 0.0, sl, tp, -1, "Market closed for this symbol right now")

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        digits = info.digits

        last: Optional[OrderResult] = None
        for attempt in range(max_retries + 1):
            # Refresh the tick every attempt — on a requote/price-changed
            # retry, the whole point is to re-quote against the CURRENT price.
            tick = mt5.symbol_info_tick(broker_sym)
            if tick is None:
                raise RuntimeError(f"No tick for {broker_sym}")
            price = tick.ask if direction == "long" else tick.bid

            # sl/tp arrive as absolute prices computed by the strategy against
            # the signal bar's CLOSE, not the live price. Re-anchor the same
            # R-distances to the live price so execution latency can't leave
            # SL/TP on the wrong side of the fill ("Invalid stops").
            a_sl, a_tp = sl, tp
            if signal_entry_price is not None and signal_entry_price > 0:
                sl_distance = abs(signal_entry_price - sl)
                tp_distance = abs(tp - signal_entry_price)
                if direction == "long":
                    a_sl = price - sl_distance
                    a_tp = price + tp_distance
                else:
                    a_sl = price + sl_distance
                    a_tp = price - tp_distance
            a_sl = round(a_sl, digits)
            a_tp = round(a_tp, digits)

            result = None
            for filling in self._filling_candidates(broker_sym):
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": broker_sym,
                    "volume": lots,
                    "type": order_type,
                    "price": price,
                    "sl": a_sl,
                    "tp": a_tp,
                    "deviation": deviation_points,
                    "magic": magic,
                    # 29, not the commonly-cited MQL5 limit of 31 -- empirically
                    # confirmed 2026-08-20 on this broker (MetaQuotes-Demo,
                    # package 5.0.5735): a 30+ char comment makes order_send
                    # silently return None with a misleading client-side error
                    # (blamed "comment", "unnamed arguments", or "invalid
                    # request" depending on which other field it notices
                    # first), regardless of the comment's actual content --
                    # bisected char-by-char, 29 always works, 30 always fails.
                    # This is why every roster leg's descriptive comment
                    # (e.g. "lucidpro_funded_Donchian_XAUUSD_1R") had been
                    # failing every single order since go-live.
                    "comment": comment[:29],
                    "type_filling": filling,
                }
                # **request, NOT order_send(request) or order_send(request=request) --
                # this installed MetaTrader5 package version (5.0.5735) silently
                # rejects both of those with a client-side (-2, "Unnamed arguments
                # not allowed")/"Invalid request" error and returns an empty
                # TradeRequest echo, despite its own docstring saying
                # order_send(request). Confirmed 2026-08-20: this is why every
                # order attempt had been failing since go-live, unrelated to the
                # IPC-concurrency issue mt5_lock.py separately fixes.
                result = mt5.order_send(**request)
                if result is not None and result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                    self._filling_mode[broker_sym] = filling  # cache the one that stuck
                    break

            if result is None:
                # order_send returning None is usually an IPC/connection
                # hiccup — treat as transient.
                last = OrderResult(0, lots, price, a_sl, a_tp, -1,
                                   f"order_send returned None: {mt5.last_error()}")
                transient = True
            else:
                filled_vol = getattr(result, "volume", 0.0) or lots
                filled_price = getattr(result, "price", 0.0) or price
                last = OrderResult(
                    ticket=getattr(result, "order", 0),
                    volume=filled_vol,
                    price=filled_price,
                    sl=a_sl,
                    tp=a_tp,
                    retcode=result.retcode,
                    comment=str(result.comment or ""),
                )
                transient = result.retcode in TRANSIENT_RETCODES

            if last.ok or not transient:
                break
            if attempt < max_retries:
                delay = 0.4 * (2 ** attempt) + random.uniform(0.0, 0.25)
                log.warning(
                    f"order_send {broker_sym} transient retcode={last.retcode} "
                    f"({last.comment}) — retry {attempt + 1}/{max_retries} in {delay:.2f}s"
                )
                time.sleep(delay)

        return last

    def close_position(self, ticket: int, max_retries: int = 3) -> bool:
        for attempt in range(max_retries + 1):
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                # Already gone (closed by SL/TP/another actor) — the caller's
                # goal state is "position not open", so this counts as done.
                return attempt > 0
            p = positions[0]
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                return False
            result = None
            for filling in self._filling_candidates(p.symbol):
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": p.symbol,
                    "volume": p.volume,
                    "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": p.ticket,
                    "price": tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask,
                    "deviation": 20,
                    "magic": p.magic,
                    "type_filling": filling,
                }
                result = mt5.order_send(**request)
                if result is not None and result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                    self._filling_mode[p.symbol] = filling
                    break
            # Same ambiguity as OrderResult.ok: this broker returns retcode=0
            # for success, but ALSO for a no-op rejection -- e.g. any close
            # attempted while the market is shut comes back retcode=0,
            # comment="Done", deal=0, order=0 and closes nothing. Reporting
            # that as success made a "close all" claim 21/21 closed while all
            # 21 were still open. A real fill always carries a deal ticket.
            if result is not None and (
                result.retcode in (10009, 10010)
                or (result.retcode == 0 and (getattr(result, "deal", 0)
                                             or getattr(result, "order", 0)))):
                return True
            transient = result is None or result.retcode in TRANSIENT_RETCODES
            if not transient or attempt >= max_retries:
                if result is not None:
                    log.warning(f"close_position ticket={ticket} failed retcode={result.retcode}")
                return False
            delay = 0.4 * (2 ** attempt) + random.uniform(0.0, 0.25)
            log.warning(f"close_position ticket={ticket} transient — retry in {delay:.2f}s")
            time.sleep(delay)
        return False

    def is_tradeable(self, symbol: str) -> bool:
        """Can this symbol be traded RIGHT NOW?

        Daily-bar strategies signal at the bar close, which coincides with the
        broker's rollover window -- order_send then returns 10018
        MARKET_CLOSED and the signal is wasted. Checking first lets the caller
        hold the signal and retry when the session reopens.

        Fails OPEN (returns True) if MT5 gives no usable answer: a probe that
        cannot see the symbol state must not silently stop the book from
        trading. A genuinely closed market will still be caught by order_send.
        """
        broker_sym = self.resolve_symbol(symbol)
        info = mt5.symbol_info(broker_sym)
        if info is None:
            if not mt5.symbol_select(broker_sym, True):
                return True                      # cannot tell -> let it try
            info = mt5.symbol_info(broker_sym)
            if info is None:
                return True

        # trade_mode: 0 = DISABLED, 4 = FULL. Anything that is not outright
        # disabled is worth attempting -- CLOSEONLY/LONGONLY/SHORTONLY may
        # still accept the specific order the caller wants.
        if getattr(info, "trade_mode", 4) == 0:
            return False

        # A stale or empty tick is the reliable "session is shut" signal:
        # brokers stop publishing quotes outside trading hours.
        tick = mt5.symbol_info_tick(broker_sym)
        if tick is None or not tick.bid or not tick.ask:
            return False
        return True

    def modify_sltp(self, ticket: int, sl: float, tp: float) -> bool:
        """Move an open position's SL/TP without closing it (e.g. breakeven lock).
        No type_filling needed — TRADE_ACTION_SLTP isn't a fill, just a modify.
        Candidates inside the broker's stops-level distance from the current
        price are skipped up front (would be rejected 10016; AUDIT M5)."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        p = positions[0]
        info = mt5.symbol_info(p.symbol)
        if info is None:
            return False
        min_dist = (getattr(info, "trade_stops_level", 0) or 0) * (getattr(info, "point", 0.0) or 0.0)
        if min_dist > 0 and sl:
            ref = p.price_current
            if abs(ref - sl) < min_dist:
                log.debug(f"modify_sltp ticket={ticket} skipped — SL within stops level ({min_dist})")
                return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": p.symbol,
            "sl": round(sl, info.digits),
            "tp": round(tp, info.digits),
        }
        result = mt5.order_send(**request)
        return result is not None and result.retcode in (0, mt5.TRADE_RETCODE_DONE)  # see OrderResult.ok

    def get_closed_deal(self, ticket: int, offset_fallback: int = 0) -> Optional[dict]:
        """Look up the exit deal for a position that's no longer open, so the
        caller can reconcile it into the ledger (exit price, P&L, reason)."""
        # position=0 is not "no filter" -- it matches whatever unrelated deal
        # the terminal happens to return, whose price and timestamps then get
        # written into the ledger as this trade's exit (see the phantom
        # $100,000 record of 2026-08-21). A real position never has ticket 0.
        if not ticket:
            log.warning("get_closed_deal called with ticket=0 -- refusing to look up a "
                        "position that cannot exist.")
            return None
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        exit_deal = deals[-1]  # last deal on this position = the one that closed it
        reason_map = {
            mt5.DEAL_REASON_SL: "sl",
            mt5.DEAL_REASON_TP: "tp",
        }
        offset = self.broker_offset_seconds(exit_deal.symbol, fallback=offset_fallback)
        return {
            "exit_price": exit_deal.price,
            "pnl_usd": sum(d.profit + d.commission + d.swap for d in deals),
            "exit_reason": reason_map.get(exit_deal.reason, "manual"),
            "closed_ts": datetime.fromtimestamp(exit_deal.time, tz=timezone.utc) - timedelta(seconds=offset),
        }
