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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
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
        return self.retcode == 10009  # TRADE_RETCODE_DONE


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
        return AccountInfo(
            login=info.login,
            server=info.server,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            currency=info.currency,
            is_demo=(info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO),
        )

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
        return df

    # --- positions + orders ---

    def get_positions(self, magic: int) -> list:
        positions = mt5.positions_get() or []
        return [p for p in positions if p.magic == magic]

    def broker_offset_seconds(self, symbol: str) -> int:
        """Broker-server clock minus true UTC, measured via a live tick.
        MT5 position/deal timestamps (`p.time`, `deal.time`) are in the
        broker's clock, not true UTC (historically several hours off) — any
        caller treating one of those as UTC directly needs this correction
        first, or ends up writing a timestamp that's hours late into the
        ledger (which is exactly what happened before this existed: a
        reconciled position's opened_ts landed *after* its own closed_ts)."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time:
            return 0
        broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        return int((broker_now - datetime.now(timezone.utc)).total_seconds())

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
        price = tick.ask if direction == "long" else tick.bid

        # sl/tp arrive as absolute prices computed by the strategy against
        # the signal bar's CLOSE, not the live price — normally close enough
        # to not matter, but with a tight tp_r_multiple the TP distance can
        # be small enough (single-digit pips on some pairs) that ordinary
        # execution latency during a fast move eats the whole gap, leaving
        # the live price already past the stale TP by the time the order
        # actually sends: MT5 then rejects the whole order as "Invalid
        # stops" and the signal is lost outright. Re-anchoring to the live
        # price — same R-distance, just measured from where price actually
        # is now instead of where it was a few hundred ms ago — keeps SL/TP
        # correctly positioned relative to `price` regardless of how far
        # price moved between signal and execution.
        if signal_entry_price is not None and signal_entry_price > 0:
            sl_distance = abs(signal_entry_price - sl)
            tp_distance = abs(tp - signal_entry_price)
            if direction == "long":
                sl = price - sl_distance
                tp = price + tp_distance
            else:
                sl = price + sl_distance
                tp = price - tp_distance

        # Round SL/TP to symbol's price precision
        digits = info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        lots = round(lots, 2)

        result = None
        for filling in self._filling_candidates(broker_sym):
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": broker_sym,
                "volume": lots,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": magic,
                "comment": comment[:31],
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                self._filling_mode[broker_sym] = filling  # cache the one that stuck
                break

        if result is None:
            return OrderResult(0, lots, price, sl, tp, -1, f"order_send returned None: {mt5.last_error()}")

        return OrderResult(
            ticket=getattr(result, "order", 0),
            volume=lots,
            price=price,
            sl=sl,
            tp=tp,
            retcode=result.retcode,
            comment=str(result.comment or ""),
        )

    def close_position(self, ticket: int) -> bool:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
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
                "magic": p.magic,
                "type_filling": filling,
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
                self._filling_mode[p.symbol] = filling
                break
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def modify_sltp(self, ticket: int, sl: float, tp: float) -> bool:
        """Move an open position's SL/TP without closing it (e.g. breakeven lock).
        No type_filling needed — TRADE_ACTION_SLTP isn't a fill, just a modify."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        p = positions[0]
        digits = mt5.symbol_info(p.symbol).digits
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": p.symbol,
            "sl": round(sl, digits),
            "tp": round(tp, digits),
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def get_closed_deal(self, ticket: int) -> Optional[dict]:
        """Look up the exit deal for a position that's no longer open, so the
        caller can reconcile it into the ledger (exit price, P&L, reason)."""
        deals = mt5.history_deals_get(position=ticket)
        if not deals:
            return None
        exit_deal = deals[-1]  # last deal on this position = the one that closed it
        reason_map = {
            mt5.DEAL_REASON_SL: "sl",
            mt5.DEAL_REASON_TP: "tp",
        }
        offset = self.broker_offset_seconds(exit_deal.symbol)
        return {
            "exit_price": exit_deal.price,
            "pnl_usd": sum(d.profit + d.commission + d.swap for d in deals),
            "exit_reason": reason_map.get(exit_deal.reason, "manual"),
            "closed_ts": datetime.fromtimestamp(exit_deal.time, tz=timezone.utc) - timedelta(seconds=offset),
        }
