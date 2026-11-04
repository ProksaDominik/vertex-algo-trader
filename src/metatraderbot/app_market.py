"""Market open/closed status for the Vertex app.

Determined from LIVE broker state, not a hardcoded session table. Sessions vary
by broker, instrument and DST, and holidays are not on any calendar this code
would ship with -- but a symbol whose ticks have stopped arriving is closed,
whatever the clock says.

Two independent signals, because neither is sufficient alone:

* ``trade_mode`` -- indices report SYMBOL_TRADE_MODE_DISABLED (0) outside their
  own session. Reliable when present, but several symbols stay at FULL (4) all
  weekend, so a non-disabled mode does not prove the market is open.
* ``tick age`` -- the decisive one. Brokers stop publishing quotes when a market
  shuts, so a tick that is hours old means closed even while trade_mode says
  otherwise. This is exactly the case that made a "close all positions" attempt
  silently do nothing on a Saturday: MT5 accepted every request and returned
  retcode=0/"Done" with no deal, because gold's mode was still 4 while its last
  tick was 14 hours old.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

FRESH_SECONDS = 300      # a live market ticks far more often than this
STALE_SECONDS = 1800     # beyond this it is certainly shut


@dataclass
class SymbolStatus:
    symbol: str
    open: bool
    tick_age_s: float | None
    trade_mode: int
    reason: str


@dataclass
class MarketStatus:
    open: bool
    n_open: int
    n_total: int
    symbols: list[SymbolStatus]
    next_open: datetime | None = None
    label: str = "UNKNOWN"

    @property
    def partial(self) -> bool:
        """Some venues trading, others shut -- normal on weekday evenings."""
        return 0 < self.n_open < self.n_total


def _next_fx_open(now: datetime) -> datetime | None:
    """Next Sunday 22:00 UTC, the conventional FX week open.

    An estimate, deliberately labelled as one in the UI: brokers differ by an
    hour or two and holidays shift it. Good enough to answer "roughly when does
    this come back", not to schedule anything against.
    """
    days = (6 - now.weekday()) % 7          # weekday(): Mon=0 .. Sun=6
    cand = (now + timedelta(days=days)).replace(hour=22, minute=0, second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=7)
    return cand


_OFFSET_CACHE = None                     # module-level: survives Streamlit reruns
_OFFSET_CACHE_PATH = "data/.broker_offset.json"
OFFSET_SNAP_TOLERANCE_S = 240            # how close to a whole hour counts as fresh


def _broker_offset(engine, symbols: list[str]) -> tuple[int, bool]:
    """(offset_seconds, trusted). Broker-server clock minus true UTC.

    Measuring this from a tick is only valid while that tick is FRESH. A broker
    clock sits a whole number of hours from UTC, so for a live tick
    ``tick.time - now`` lands within seconds of a multiple of 3600. Once the
    market shuts the tick stops advancing and the same expression drifts away
    from any whole hour -- which is exactly how this returned 5015s (true value
    10800) after gold's Friday close and made the panel report a shut market as
    trading.

    So the residual to the nearest hour IS the freshness test. Take the largest
    offset across the basket (the freshest symbol), and accept it only when it
    snaps cleanly. A confident reading is cached, in memory and on disk, so a
    weekend restart still knows the offset it learned while the market was open.
    """
    global _OFFSET_CACHE
    import json
    import os
    import time as _time

    best = None
    for sym in symbols:
        try:
            tick = _mt5().symbol_info_tick(engine.resolve_symbol(sym))
        except Exception:
            continue
        if tick is not None and getattr(tick, "time", 0):
            d = tick.time - _time.time()
            best = d if best is None else max(best, d)

    if best is not None:
        snapped = round(best / 3600.0) * 3600
        if abs(best - snapped) <= OFFSET_SNAP_TOLERANCE_S:
            _OFFSET_CACHE = int(snapped)
            try:
                os.makedirs("data", exist_ok=True)
                with open(_OFFSET_CACHE_PATH, "w", encoding="utf-8") as fh:
                    json.dump({"offset_s": int(snapped)}, fh)
            except Exception:
                pass
            return int(snapped), True

    if _OFFSET_CACHE is not None:
        return int(_OFFSET_CACHE), True
    try:
        with open(_OFFSET_CACHE_PATH, encoding="utf-8") as fh:
            cached = int(json.load(fh)["offset_s"])
        _OFFSET_CACHE = cached
        return cached, True
    except Exception:
        return 0, False


def _mt5():
    import MetaTrader5 as mt5
    return mt5


def check(engine, symbols: list[str]) -> MarketStatus:
    """Live status for a list of canonical symbols. Never raises."""
    try:
        import MetaTrader5 as mt5
    except Exception:
        return MarketStatus(False, 0, len(symbols), [], None, "NO CONNECTION")

    now = datetime.now(timezone.utc)

    # Tick timestamps are on the BROKER's clock (this server runs UTC+3), so
    # comparing them straight to UTC made every age read 3 hours too low --
    # negative, in fact, for a fresh tick. A market shut for up to 3 hours
    # therefore still reported as "trading". That is the same class of bug that
    # made a close-all silently do nothing on a Saturday while the engine
    # cheerfully reported success.
    offset, offset_trusted = _broker_offset(engine, symbols)
    if not offset_trusted:
        # Without a trustworthy offset every tick age is meaningless, and a
        # meaningless age must not be rendered as "trading". This panel said
        # OPEN on Fri 2026-08-28 21:36 UTC while the broker rejected every
        # order with retcode 10018 (market closed), because the offset had been
        # measured from a stale tick: 5015s instead of the true 10800, which
        # made ages come out NEGATIVE and every symbol look freshly quoted.
        # Reporting UNKNOWN is the honest answer; claiming OPEN is a lie the
        # user would act on.
        out = [SymbolStatus(s, False, None, -1, "clock offset unknown") for s in symbols]
        return MarketStatus(False, 0, len(out), out, _next_fx_open(now), "UNKNOWN")

    out: list[SymbolStatus] = []
    for sym in symbols:
        try:
            broker = engine.resolve_symbol(sym)
            info = mt5.symbol_info(broker)
            tick = mt5.symbol_info_tick(broker)
        except Exception:
            out.append(SymbolStatus(sym, False, None, -1, "unavailable"))
            continue

        mode = int(getattr(info, "trade_mode", -1)) if info is not None else -1
        age = None
        if tick is not None and getattr(tick, "time", 0):
            tick_utc = datetime.fromtimestamp(tick.time, tz=timezone.utc) - timedelta(seconds=offset)
            age = (now - tick_utc).total_seconds()

        if mode == 0:
            out.append(SymbolStatus(sym, False, age, mode, "session closed"))
        elif age is None:
            out.append(SymbolStatus(sym, False, age, mode, "no quotes"))
        elif age <= FRESH_SECONDS:
            out.append(SymbolStatus(sym, True, age, mode, "trading"))
        elif age >= STALE_SECONDS:
            out.append(SymbolStatus(sym, False, age, mode, "quotes stale"))
        else:
            # Between the two thresholds: thin session or a quiet instrument.
            out.append(SymbolStatus(sym, True, age, mode, "thin/quiet"))

    n_open = sum(1 for s in out if s.open)
    st = MarketStatus(open=n_open > 0, n_open=n_open, n_total=len(out), symbols=out)
    if not out:
        st.label = "UNKNOWN"
    elif n_open == 0:
        st.label = "CLOSED"
        st.next_open = _next_fx_open(now)
    elif n_open < len(out):
        st.label = "PARTIAL"
    else:
        st.label = "OPEN"
    return st


def humanise_age(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def until(target: datetime | None) -> str:
    if target is None:
        return "—"
    d = (target - datetime.now(timezone.utc)).total_seconds()
    if d <= 0:
        return "now"
    if d < 3600:
        return f"{d / 60:.0f}m"
    if d < 86400:
        return f"{d / 3600:.0f}h {(d % 3600) / 60:.0f}m"
    return f"{d / 86400:.0f}d {(d % 86400) / 3600:.0f}h"
