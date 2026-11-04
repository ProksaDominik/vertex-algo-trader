"""Cross-process serialization for calls into the MetaTrader5 module.

The MetaTrader5 Python package's IPC channel to one running terminal is not
safe under concurrent access from independent OS processes: two bot
processes (LucidPro Funded + Trial, plus chart_api/dashboard reading live
positions) all attached to the same terminal and calling into the API
around the same instant produced spurious client-side errors that had
nothing to do with the request itself -- a generic (-2, "Invalid comment
argument") on order_send(), and "IPC send failed" on copy_rates_from_pos()
-- plus calls hanging long enough to trip the self-watchdog. Diagnosed
2026-08-20: both symptoms started exactly when the Funded bot became a
second process against the same terminal the Trial bot was already using.

A cross-process file lock (Windows byte-range lock via msvcrt -- MT5
desktop only runs on Windows, matching mt5_engine.py's own constraint)
makes sure only one process is inside an mt5.* call at a time.

Call install_mt5_lock(mt5) once, right after `import MetaTrader5 as mt5`,
in every process that talks to MT5. Patches the module object in place, so
every other file that does its own `import MetaTrader5 as mt5` gets the
same, already-locked functions for free (same object in sys.modules) --
no need to repeat the call in files that import metatraderbot.mt5_engine.
Idempotent: safe to call more than once or from more than one file.
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows import guard, mirrors mt5_engine.py
    msvcrt = None  # type: ignore

_LOCK_PATH = Path(__file__).resolve().parents[2] / "data" / ".mt5_ipc.lock"

# Only functions that actually cross the IPC boundary to the terminal need
# serializing. Pure local reads (last_error(), the TIMEFRAME_*/ORDER_*/...
# constants) are left alone -- locking those would only add contention for
# no benefit.
_LOCKED_FUNCS = (
    "initialize", "shutdown", "login",
    "account_info", "terminal_info",
    "symbol_info", "symbol_info_tick", "symbol_select", "symbols_get",
    "copy_rates_from_pos", "copy_rates_from", "copy_rates_range",
    "copy_ticks_from", "copy_ticks_range",
    "positions_get", "positions_total",
    "orders_get", "orders_total",
    "history_deals_get", "history_orders_get",
    "order_send", "order_check", "order_calc_margin", "order_calc_profit",
)


@contextmanager
def mt5_ipc_lock():
    """Hold the cross-process MT5 IPC lock for the duration of the `with` block."""
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(_LOCK_PATH, "a+b")
    try:
        fh.seek(0)
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:
                # LK_LOCK already retries internally (~10s) before raising --
                # if still contended after that, just try again instead of
                # surfacing a false failure to a caller that did nothing wrong.
                time.sleep(0.05)
        try:
            yield
        finally:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        fh.close()


def install_mt5_lock(mt5_module) -> None:
    """Wrap the IPC-crossing functions on an imported MetaTrader5 module so
    every call is serialized across processes. Idempotent."""
    if msvcrt is None or mt5_module is None:
        return
    if getattr(mt5_module, "_ipc_lock_installed", False):
        return

    def _wrap(fn):
        @functools.wraps(fn)
        def _locked(*args, **kwargs):
            with mt5_ipc_lock():
                return fn(*args, **kwargs)
        return _locked

    for name in _LOCKED_FUNCS:
        fn = getattr(mt5_module, name, None)
        if fn is not None:
            setattr(mt5_module, name, _wrap(fn))
    mt5_module._ipc_lock_installed = True
