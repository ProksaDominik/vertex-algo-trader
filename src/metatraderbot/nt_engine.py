"""NinjaTrader 8 connection layer -- the futures twin of mt5_engine.py.

Lucid Trading accounts cannot be reached from MetaTrader 5, and Tradovate's
API is closed to prop accounts, so orders go through NinjaTrader 8 (which
connects to Lucid over Rithmic or Tradovate). A small NinjaScript add-on,
ninjatrader/VertexBridge.cs, runs inside NinjaTrader and answers one-line
requests on 127.0.0.1; this module is the Python client for it.

The public methods mirror MT5Engine's, so the runners keep their structure.
Differences a caller has to know about:

  * Symbols are futures roots ("6B", "MGC") or explicit contracts
    ("MGC 12-26"). A root resolves to the front month (futures.front_contract)
    and is re-resolved every trade date, so a roll takes effect on its own.
  * Sizes are whole contracts. `lots` is floored, never rounded up.
  * Futures accounts NET per instrument, so the bridge allows ONE open bot
    ticket per instrument; a second signal on the same contract is refused.
  * Tickets are ids this engine generates; the bridge encodes them (plus the
    magic) in NinjaTrader order names, so an entry retried after a timeout is
    recognised instead of being placed twice.
  * Stop-loss and take-profit are real OCO orders at the exchange, placed by
    the bridge the moment the entry fills and re-anchored to the fill price
    (same R-distances as the signal, like MT5Engine.place_market_order).
  * Bar timestamps are true UTC bar OPEN times, like the corrected MT5 bars.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from metatraderbot import futures, lucid_rules

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 45973   # not 36973: NinjaTrader's own ATI server holds that port
# 1.0 could open a position while "protecting" one that was already closed
# (fixed in 1.1) -- never trade through it.
MIN_BRIDGE_VERSION = (1, 1)

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

# MT5's numbering, so callers that branch on retcodes / position types keep working.
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
RETCODE_REJECTED = -1
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

_TERMINAL_STATES = {"Filled", "Cancelled", "Rejected", "Unknown"}


class BridgeError(RuntimeError):
    """The NinjaTrader bridge is unreachable or refused a request."""


@dataclass
class AccountInfo:
    login: str               # NinjaTrader account name, e.g. "Sim101" or the Lucid account id
    server: str              # connection name as shown in NinjaTrader
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    is_demo: bool            # True only for NinjaTrader's own simulation accounts
    margin_mode: int = 0     # futures net per instrument (MT5's 0 = netting)

    @property
    def is_hedging(self) -> bool:
        return False


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
        # A fill is the only proof a position exists -- same rule MT5Engine
        # learned the hard way (AUDIT H3 / the retcode=0 phantom trades).
        return self.retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL) and self.ticket != 0 and self.volume > 0

    @property
    def partial(self) -> bool:
        return self.retcode == RETCODE_DONE_PARTIAL


@dataclass
class Position:
    """The fields of MT5's TradePosition that the runners read."""
    ticket: int
    magic: int
    symbol: str              # NinjaTrader contract, e.g. "MGC 12-26"
    volume: float
    type: int                # POSITION_TYPE_BUY / POSITION_TYPE_SELL
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    time: int                # entry time, epoch seconds UTC


def _fmt(v) -> str:
    if isinstance(v, float):
        return repr(v)
    return str(v)


class NinjaTraderEngine:
    def __init__(
        self,
        symbol_map: Optional[dict] = None,
        account: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_s: float = 15.0,
        fill_timeout_s: float = 20.0,
        max_contract_units: Optional[int] = None,
    ):
        """max_contract_units: hard cap on TOTAL open size in Lucid micro
        units (micro = 1, mini = 10; LucidPro 50K = 40). Enforced here, at the
        engine layer, so every caller inherits it -- the lesson of the
        usd_per_point gate, which once existed one layer above the code that
        needed it."""
        self.symbol_map = symbol_map or {}
        self.account = account or os.environ.get("NT_ACCOUNT", "Sim101")
        self.host = host or os.environ.get("NT_BRIDGE_HOST", DEFAULT_HOST)
        self.port = int(port or os.environ.get("NT_BRIDGE_PORT", DEFAULT_PORT))
        self.timeout_s = timeout_s
        self.fill_timeout_s = fill_timeout_s
        self.max_contract_units = max_contract_units
        self._connected = False
        self._resolved: dict[str, tuple[str, str]] = {}   # canonical -> (contract, trade date)
        self._instrument_cache: dict[str, dict] = {}
        self._ticket_lock = threading.Lock()
        self._last_ticket = 0

    # --- transport ---------------------------------------------------------

    def _call(self, cmd: str, **args) -> dict:
        parts = [cmd]
        for k, v in args.items():
            if v is None:
                continue
            s = _fmt(v)
            if any(c in s for c in "\t\r\n"):
                raise ValueError(f"{k}={s!r}: tabs/newlines are not allowed in bridge arguments")
            parts.append(f"{k}={s}")
        payload = ("\t".join(parts) + "\n").encode("utf-8")
        buf = b""
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
                sock.sendall(payload)
                while not buf.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except OSError as e:
            raise BridgeError(
                f"NinjaTrader bridge unreachable at {self.host}:{self.port} ({e}). "
                f"Is NinjaTrader 8 running with the VertexBridge add-on compiled?") from e
        if not buf:
            raise BridgeError(f"bridge closed the connection without answering {cmd!r}")
        try:
            reply = json.loads(buf.decode("utf-8"))
        except ValueError as e:
            raise BridgeError(f"unreadable bridge reply to {cmd!r}: {buf[:200]!r}") from e
        if not reply.get("ok"):
            raise BridgeError(f"{cmd} refused by NinjaTrader: {reply.get('error')}")
        return reply.get("data") or {}

    # --- connection lifecycle ---------------------------------------------

    def connect(self) -> AccountInfo:
        version = str(self._call("ping").get("version", "0"))
        try:
            parsed = tuple(int(x) for x in version.split("."))
        except ValueError:
            parsed = (0,)            # unrecognisable -> treat as outdated, never trade through it
        if parsed < MIN_BRIDGE_VERSION:
            raise BridgeError(
                f"VertexBridge {version} in NinjaTrader is outdated (need "
                f"{'.'.join(map(str, MIN_BRIDGE_VERSION))}+). Copy the new add-on with "
                f"ninjatrader\\Install-VertexBridge.ps1, then NinjaScript Editor -> F5.")
        acc = self.account_info()
        if not acc.server or acc.server == "none":
            log.warning("NinjaTrader account %s has no active connection yet -- "
                        "connect Lucid/Sim in NinjaTrader's Control Center.", acc.login)
        self._connected = True
        return acc

    def bridge_version(self) -> tuple:
        """The VertexBridge add-on's version as a tuple, e.g. (1, 2); (0,) if unknown."""
        try:
            return tuple(int(x) for x in str(self._call("ping").get("version", "0")).split("."))
        except (BridgeError, ValueError):
            return (0,)

    def disconnect(self) -> None:
        self._connected = False

    @staticmethod
    def is_ipc_error(err) -> bool:
        return isinstance(err, BridgeError)

    def reconnect(self) -> bool:
        try:
            self._call("ping")
            self._connected = True
            return True
        except BridgeError as e:
            log.warning(f"Reconnect failed: {e}")
            return False

    def account_info(self) -> AccountInfo:
        d = self._call("account", account=self.account)
        cash = float(d.get("cash") or 0.0)
        unreal = float(d.get("unrealized") or 0.0)
        name = str(d.get("name", self.account))
        return AccountInfo(
            login=name,
            server=str(d.get("connection") or "none"),
            balance=cash,
            equity=cash + unreal,
            margin=0.0,
            free_margin=cash,
            currency="USD",
            # NinjaTrader 8.1's own paper account ("DEMO" + digits, on its
            # Simulation connection) is not a Simulator-provider account, so
            # the bridge does not flag it -- but it is paper money all the same.
            is_demo=bool(d.get("is_sim")) or name.upper().startswith("DEMO"),
        )

    # --- symbols -------------------------------------------------------------

    def _instrument(self, contract: str) -> dict:
        info = self._instrument_cache.get(contract)
        if info is None:
            info = self._call("instrument", name=contract)
            self._instrument_cache[contract] = info
        return info

    def resolve_symbol(self, canonical: str) -> str:
        """Futures root or explicit contract -> NinjaTrader contract name."""
        target = self.symbol_map.get(canonical, canonical).strip()
        if " " in target:                       # explicit contract, e.g. "MGC 12-26"
            futures.spec(target)                # must be Lucid-approved
            return self._instrument(target)["name"]
        today = lucid_rules.trading_day(datetime.now(timezone.utc))
        cached = self._resolved.get(canonical)
        if cached and cached[1] == today:
            return cached[0]
        contract = futures.front_contract(target, datetime.fromisoformat(today).date())
        name = self._instrument(contract)["name"]
        if cached and cached[0] != name:
            log.warning(f"ROLL {canonical}: {cached[0]} -> {name} (open tickets stay on their own contract)")
        self._resolved[canonical] = (name, today)
        return name

    # --- market data ---------------------------------------------------------

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        contract = self.resolve_symbol(symbol)
        minutes = TIMEFRAME_MINUTES.get(timeframe)
        if minutes is None:
            raise ValueError(f"Unsupported timeframe {timeframe}")
        d = self._call("bars", instrument=contract, minutes=minutes, count=count)
        rows = d.get("bars") or []
        if not rows:
            raise RuntimeError(f"no {timeframe} bars for {contract} (NinjaTrader returned none)")
        df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.drop(columns="t").set_index("timestamp")

        # The bridge returns COMPLETED bars only. MT5's copy_rates also returns
        # the still-forming bar last, and every runner drops the last row for
        # signal decisions -- so without a stand-in the newest completed bar
        # would be thrown away and every signal would fire one bar late.
        period = pd.Timedelta(minutes=minutes)
        now = pd.Timestamp.now(tz="UTC")
        last_close = df.index[-1] + period
        if minutes < 1440 and last_close > now + pd.Timedelta(minutes=2):
            raise BridgeError(
                f"{contract} bar stamped {df.index[-1]} closes in the future (now {now}) -- "
                f"NinjaTrader's time zone setting and the bridge disagree")
        # The stand-in is stamped from the last COMPLETED bar, never from the
        # wall clock: runners detect a new bar by this index changing, and a
        # clock-derived stamp would keep advancing through the daily 17:00 ET
        # halt and the weekend -- re-running the strategy on the same closed
        # bar and re-emitting its signal as a duplicate entry.
        forming_ts = df.index[-1] + period
        last_close_px = float(df["close"].iloc[-1])
        forming = pd.DataFrame(
            {"open": [last_close_px], "high": [last_close_px], "low": [last_close_px],
             "close": [last_close_px], "volume": [0.0]},
            index=pd.DatetimeIndex([forming_ts], name="timestamp"))
        return pd.concat([df, forming])

    FLOW_COLUMNS = ("buy", "sell", "other", "trades", "big_buy", "big_sell")

    def get_flow(self, symbol: str, start_utc: datetime, end_utc: datetime, big: int = 10) -> tuple[pd.DataFrame, dict]:
        """Order flow for research (VertexBridge 1.3+): every trade in [start_utc, end_utc) --
        at most 3 days -- summed per minute: volume bought (at the ask), sold (at the bid),
        unclassified, number of trades, and the part of each side in trades of >= `big`
        contracts. Returns (frame indexed by minute open UTC, {"ticks", "with_quote"}).
        Slow (NinjaTrader loads every trade); give the engine a long timeout_s."""
        if self.bridge_version() < (1, 3):
            raise BridgeError("order flow needs VertexBridge 1.3 -- in NinjaTrader open New > NinjaScript "
                              "Editor and press F5 once")
        contract = self.resolve_symbol(symbol)
        d = self._call("flow", **{"instrument": contract, "from": int(start_utc.timestamp()),
                                  "to": int(end_utc.timestamp()), "big": int(big)})
        df = pd.DataFrame(d.get("minutes") or [], columns=["t", *self.FLOW_COLUMNS])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), unit="s", utc=True), name="timestamp")
        return df.astype("int64"), {"ticks": int(d.get("ticks") or 0), "with_quote": int(d.get("with_quote") or 0)}

    def _quote(self, symbol: str) -> dict:
        return self._call("quote", instrument=self.resolve_symbol(symbol))

    def quote(self, symbol: str) -> Optional[dict]:
        """{"bid", "ask", "age_s", ...} for the front contract, or None."""
        try:
            return self._quote(symbol)
        except (BridgeError, ValueError) as e:
            log.warning(f"quote({symbol}): {e}")
            return None

    def get_spread(self, symbol: str) -> Optional[float]:
        try:
            q = self._quote(symbol)
        except (BridgeError, ValueError):
            return None
        bid, ask = q.get("bid"), q.get("ask")
        if not bid or not ask or bid <= 0 or ask <= 0:
            return None
        return float(ask - bid)

    def terminal_health(self, sample_symbol: str | None = None) -> dict:
        out = {"connected": False, "trade_allowed": False, "tick_age_s": None}
        try:
            d = self._call("account", account=self.account)
            out["connected"] = bool(d.get("connected"))
            out["trade_allowed"] = bool(d.get("connected"))
            if sample_symbol:
                out["tick_age_s"] = self._quote(sample_symbol).get("age_s")
        except Exception as e:  # noqa: BLE001 -- health probe must not throw
            log.warning(f"terminal_health probe error: {e}")
        return out

    def is_tradeable(self, symbol: str, max_quote_age_s: float = 120.0) -> bool:
        """A live, fresh quote on a connected account. Unlike MT5Engine this
        fails CLOSED: with the bridge down no order can reach the exchange
        anyway, and a closed futures session shows up as a stale quote."""
        try:
            q = self._quote(symbol)
        except (BridgeError, ValueError) as e:
            log.warning(f"is_tradeable({symbol}): {e}")
            return False
        age = q.get("age_s")
        return age is not None and age <= max_quote_age_s and bool(q.get("bid")) and bool(q.get("ask"))

    # --- contract specs ------------------------------------------------------

    def usd_per_point(self, symbol: str) -> Optional[float]:
        """USD per contract per 1.0 price move, from NinjaTrader's instrument
        definition, gated against the static futures table the same way
        MT5Engine gates broker tick values: on a 2x+ disagreement the LARGER
        figure wins, because a larger multiplier sizes a SMALLER position."""
        try:
            info = self._instrument(self.resolve_symbol(symbol))
        except (BridgeError, ValueError):
            return None
        live = float(info.get("point_value") or 0.0)
        if live <= 0:
            return None
        try:
            frozen = futures.point_value(info["name"])
        except ValueError:
            frozen = 0.0
        if frozen > 0 and max(live, frozen) / min(live, frozen) >= 2.0:
            safe = max(live, frozen)
            log.error(
                "CONTRACT MULTIPLIER DRIFT on %s: NinjaTrader says %.4f, futures.py says %.4f. "
                "Using %.4f -- the larger value, which sizes SMALLER.", info["name"], live, frozen, safe)
            return safe
        return live

    def volume_spec(self, symbol: str) -> tuple[float, float, float]:
        """(min, max, step) in contracts. max reflects the engine's contract
        cap for this symbol when one is set."""
        vol_max = 1000.0
        if self.max_contract_units is not None:
            vol_max = float(self.max_contract_units // futures.units(self.resolve_symbol(symbol)))
        return (1.0, vol_max, 1.0)

    def stops_min_distance(self, symbol: str) -> float:
        return 0.0      # exchanges have no MT5-style stops level; the bridge checks the side

    def broker_offset_seconds(self, symbol: str, fallback: int = 0) -> int:
        return 0        # the bridge converts NinjaTrader's clock to UTC itself

    # --- tickets & positions -------------------------------------------------

    def _tickets(self) -> list[dict]:
        return self._call("tickets", account=self.account).get("tickets") or []

    def _ticket(self, ticket: int) -> Optional[dict]:
        for t in self._tickets():
            if int(t["ticket"]) == int(ticket):
                return t
        return None

    def _live_price(self, instrument: str) -> Optional[float]:
        """The price when a ticket reply carries none. After a bridge or NinjaTrader restart the
        bridge has no market-data subscription for a position it recovered until something asks
        for a quote -- the reply's "last" stays empty and the position looked flat at its entry
        (2026-10-07, after the bridge 1.3 compile: the trial's open P&L read $0 although NQ was
        +85 pts, so a loss-line breach could not have been seen). Asking for a quote subscribes
        the feed again; the last 1-minute close covers the first moments."""
        try:
            q = self._call("quote", instrument=instrument)
            for k in ("last", "bid", "ask"):
                v = q.get(k)
                if v is not None and v == v and float(v) > 0:
                    return float(v)
            rows = self._call("bars", instrument=instrument, minutes=1, count=2).get("bars") or []
            if rows:
                return float(rows[-1][4])
        except (BridgeError, ValueError, TypeError, IndexError):
            pass
        return None

    def _to_position(self, t: dict) -> Position:
        entry = float(t.get("entry_avg") or 0.0)
        last = t.get("last")
        qty = int(t.get("open_qty") or 0)
        price = float(last) if last else None
        if price is None and qty > 0:
            price = self._live_price(t["instrument"])
            if price is None:
                log.warning("no live price for %s ticket %s -- valuing it at its entry", t["instrument"], t["ticket"])
        if price is None:
            price = entry
        sign = 1.0 if t["side"] == "long" else -1.0
        pv = float(t.get("point_value") or 0.0)
        return Position(
            ticket=int(t["ticket"]),
            magic=int(t["magic"]),
            symbol=t["instrument"],
            volume=float(qty),
            type=POSITION_TYPE_BUY if t["side"] == "long" else POSITION_TYPE_SELL,
            price_open=entry,
            price_current=price,
            sl=float(t.get("stop") or 0.0),
            tp=float(t.get("target") or 0.0),
            profit=(price - entry) * sign * qty * pv,
            time=int(t.get("opened_utc") or 0),
        )

    def get_positions(self, magic: int) -> list:
        return [self._to_position(t) for t in self._tickets()
                if int(t.get("open_qty") or 0) > 0 and int(t["magic"]) == int(magic)]

    def get_position(self, ticket: int) -> Optional[Position]:
        t = self._ticket(ticket)
        if t is None or int(t.get("open_qty") or 0) <= 0:
            return None
        return self._to_position(t)

    def get_all_open_tickets(self) -> set[int]:
        return {int(t["ticket"]) for t in self._tickets() if int(t.get("open_qty") or 0) > 0}

    def open_contracts(self) -> dict[str, int]:
        """Every open position on the account -- bot or manual -- as
        {contract: signed quantity}. The contract cap counts all of them,
        because Lucid does."""
        rows = self._call("positions", account=self.account).get("positions") or []
        return {r["instrument"]: int(r["qty"]) * (1 if r["side"] == "long" else -1)
                for r in rows if int(r.get("qty") or 0) != 0}

    # --- orders --------------------------------------------------------------

    def _new_ticket(self) -> int:
        with self._ticket_lock:
            t = max(int(time.time() * 1000), self._last_ticket + 1)
            self._last_ticket = t
            return t

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
        deviation_points: int = 0,
        max_retries: int = 0,
    ) -> OrderResult:
        contract = self.resolve_symbol(symbol)
        qty = int(lots + 1e-9)
        if qty < 1:
            return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED,
                               f"size {lots:.3f} contracts is below 1 -- skipped")

        ref = signal_entry_price if signal_entry_price and signal_entry_price > 0 else None
        if ref is None:
            q = self._call("quote", instrument=contract)
            ref = q.get("ask") if direction == "long" else q.get("bid")
            if not ref:
                return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED, "no live quote to anchor stops")
        if direction == "long":
            valid = sl < ref < tp
        elif direction == "short":
            valid = tp < ref < sl
        else:
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if not valid:
            return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED,
                               f"stops on the wrong side: sl={sl} entry={ref} tp={tp}")
        sl_dist, tp_dist = abs(ref - sl), abs(tp - ref)

        if self.max_contract_units is not None:
            used = lucid_rules.open_units(self.open_contracts())
            per = futures.units(contract)
            allowed = max(0, (self.max_contract_units - used) // per)
            if allowed < 1:
                return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED,
                                   f"contract cap reached ({used}/{self.max_contract_units} units in use)")
            if qty > allowed:
                log.warning(f"[{symbol}] {qty} contracts would exceed the cap "
                            f"({used}/{self.max_contract_units} units in use) -- clamping to {allowed}")
                qty = allowed

        ticket = self._new_ticket()
        try:
            t = self._call("enter", account=self.account, instrument=contract, side=direction,
                           qty=qty, sl_dist=float(sl_dist), tp_dist=float(tp_dist),
                           magic=int(magic), ticket=ticket)
        except BridgeError as e:
            # The request may have reached NinjaTrader before the socket
            # failed. The ticket id makes a lookup safe: never assume "not sent".
            t = self._safe_ticket(ticket)
            if t is None:
                return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED, str(e))

        t = self._await_fill(ticket, t)
        filled = int(t.get("entry_filled") or 0) if t else 0
        if filled <= 0:
            state = t.get("entry_state") if t else "unknown"
            err = (t or {}).get("error") or ""
            return OrderResult(0, 0.0, 0.0, sl, tp, RETCODE_REJECTED,
                               f"entry not filled ({state}) {err}".strip())
        if t.get("error"):
            log.error(f"[{symbol}] ticket {ticket}: {t['error']}")
        return OrderResult(
            ticket=ticket,
            volume=float(filled),
            price=float(t.get("entry_avg") or ref),
            sl=float(t.get("stop") or 0.0),
            tp=float(t.get("target") or 0.0),
            retcode=RETCODE_DONE if filled >= qty else RETCODE_DONE_PARTIAL,
            comment=f"{t.get('entry_state')} bracket={t.get('bracket')}",
        )

    def _safe_ticket(self, ticket: int) -> Optional[dict]:
        try:
            return self._ticket(ticket)
        except BridgeError:
            return None

    def _await_fill(self, ticket: int, t: Optional[dict]) -> Optional[dict]:
        """Wait for the entry to finish and its protective bracket to be
        placed. A market entry that neither fills nor fails in time is
        cancelled; a fill that races the cancel is still reported, so the
        caller never loses track of a real position."""
        deadline = time.monotonic() + self.fill_timeout_s
        while True:
            if t is not None and self._settled(t):
                return t
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
            t = self._safe_ticket(ticket) or t
        log.warning(f"ticket {ticket}: entry still {t and t.get('entry_state')} after "
                    f"{self.fill_timeout_s:.0f}s -- cancelling it")
        try:
            self._call("cancel", account=self.account, ticket=ticket)
        except BridgeError as e:
            log.warning(f"cancel ticket {ticket}: {e}")
        for _ in range(15):
            time.sleep(0.2)
            t = self._safe_ticket(ticket) or t
            if t is not None and t.get("entry_state") in _TERMINAL_STATES:
                break
        return t

    @staticmethod
    def _settled(t: dict) -> bool:
        if t.get("entry_state") not in _TERMINAL_STATES:
            return False
        if int(t.get("entry_filled") or 0) <= 0:
            return True
        return t.get("bracket") in ("working", "failed", "none") or bool(t.get("closed"))

    def close_position(self, ticket: int, max_retries: int = 3) -> bool:
        """Close a ticket at market. Returns False if it was not open to begin
        with (MT5Engine semantics), True once it is confirmed flat."""
        for attempt in range(max_retries + 1):
            t = self._safe_ticket(ticket)
            if t is None or int(t.get("open_qty") or 0) <= 0:
                return attempt > 0
            try:
                self._call("close", account=self.account, ticket=ticket, reason="manual")
            except BridgeError as e:
                log.warning(f"close_position ticket={ticket}: {e}")
            if self._await_flat(ticket):
                return True
        return False

    def close_partial(self, ticket: int, volume: float) -> bool:
        t = self._safe_ticket(ticket)
        if t is None:
            return False
        open_qty = int(t.get("open_qty") or 0)
        qty = int(volume + 1e-9)
        if qty < 1 or open_qty <= 0:
            return False
        if qty >= open_qty:
            return self.close_position(ticket)
        try:
            self._call("close", account=self.account, ticket=ticket, qty=qty, reason="partial")
        except BridgeError as e:
            log.warning(f"close_partial ticket={ticket}: {e}")
            return False
        deadline = time.monotonic() + self.fill_timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.2)
            t = self._safe_ticket(ticket)
            if t is not None and int(t.get("open_qty") or 0) <= open_qty - qty:
                return True
        return False

    def _await_flat(self, ticket: int) -> bool:
        deadline = time.monotonic() + self.fill_timeout_s
        while time.monotonic() < deadline:
            t = self._safe_ticket(ticket)
            if t is not None and int(t.get("open_qty") or 0) <= 0:
                return True
            time.sleep(0.2)
        return False

    def modify_sltp(self, ticket: int, sl: float, tp: float) -> bool:
        try:
            d = self._call("modify", account=self.account, ticket=ticket,
                           sl=float(sl or 0.0), tp=float(tp or 0.0))
        except BridgeError as e:
            log.debug(f"modify_sltp ticket={ticket}: {e}")
            return False
        return bool(d.get("changed"))

    def flatten(self, symbol: str) -> int:
        """NinjaTrader's own flatten for one contract: cancels its working
        orders and closes the ENTIRE position, bot or manual. Returns the
        signed position it found. For cleanup and emergencies only."""
        d = self._call("flatten", account=self.account, instrument=self.resolve_symbol(symbol))
        return int(d.get("position_before") or 0)

    def close_all_bot_positions(self) -> int:
        """Flatten every open bot ticket (Lucid's 4:45 PM ET rule). Returns
        how many were still open after trying."""
        still_open = 0
        for ticket in sorted(self.get_all_open_tickets()):
            if not self.close_position(ticket):
                t = self._safe_ticket(ticket)
                if t is not None and int(t.get("open_qty") or 0) > 0:
                    still_open += 1
        return still_open

    def get_closed_deal(self, ticket: int, offset_fallback: int = 0) -> Optional[dict]:
        """Exit price, net P&L and reason for a ticket that is no longer open.

        P&L is computed from the fills and NinjaTrader's point value, minus
        Lucid's published commission for both sides of every contract -- the
        figure Lucid's own balance moves by."""
        if not ticket:
            log.warning("get_closed_deal called with ticket=0 -- refusing to look up a "
                        "position that cannot exist.")
            return None
        t = self._safe_ticket(ticket)
        if t is None or not t.get("closed") or int(t.get("entry_filled") or 0) <= 0:
            return None
        qty = int(t.get("closed_qty") or 0)
        filled = int(t.get("entry_filled") or 0)
        if qty > filled:
            # More closed than was ever opened is an accounting fault in the
            # bridge (it once counted each exit fill twice) -- never let it
            # inflate the P&L the attempt is judged on.
            log.error(f"ticket {ticket}: bridge reports {qty} contracts closed of {filled} opened -- "
                      f"booking {filled}")
            qty = filled
        entry = float(t.get("entry_avg") or 0.0)
        exit_px = t.get("exit_avg")
        if exit_px is None or qty <= 0:
            return None
        sign = 1.0 if t["side"] == "long" else -1.0
        pv = float(t.get("point_value") or 0.0) or futures.point_value(t["instrument"])
        gross = (float(exit_px) - entry) * sign * qty * pv
        commission = 2 * futures.commission_per_side(t["instrument"]) * qty
        reason = {"sl": "sl", "tp": "tp"}.get(t.get("close_reason"), "manual")
        closed_utc = t.get("closed_utc") or time.time()
        return {
            "exit_price": float(exit_px),
            "pnl_usd": gross - commission,
            "exit_reason": reason,
            "close_reason": t.get("close_reason") or "",      # the bridge's own word: protection_lost, gap, ...
            "closed_ts": datetime.fromtimestamp(float(closed_utc), tz=timezone.utc),
        }
