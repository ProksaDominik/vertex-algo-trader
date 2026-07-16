"""
MT5 connectivity check — Phase 0 gate.

Prerequisites:
- MT5 desktop terminal is RUNNING and logged in
- Python venv activated
- `pip install -r requirements.txt` has succeeded

Usage:
    python scripts/test_connection.py

Exits 0 if everything is green, non-zero on any failure.
Prints the exact gold symbol name for your broker — save that value for Phase 1.
"""
from __future__ import annotations

import sys

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[FAIL] MetaTrader5 package not installed.")
    print("       Fix: activate the venv and run `pip install -r requirements.txt`.")
    sys.exit(2)


GOLD_CANDIDATES = [
    "XAUUSD",
    "XAUUSDm",
    "XAUUSD.raw",
    "XAUUSDx",
    "XAUUSD.",
    "GOLD",
    "GOLD.raw",
    "GOLDm",
]


def main() -> int:
    if not mt5.initialize():
        err = mt5.last_error()
        print(f"[FAIL] mt5.initialize() failed: {err}")
        print("       Common causes:")
        print("       - MT5 desktop is not running -> launch it and log in")
        print("       - Multiple MT5 terminals installed -> close all but one")
        print("       - AlgoTrading disabled -> Tools > Options > Expert Advisors > Allow algo")
        return 1

    info = mt5.account_info()
    if info is None:
        print(f"[FAIL] account_info() returned None: {mt5.last_error()}")
        print("       Make sure you are logged into an account in MT5.")
        mt5.shutdown()
        return 1

    mode = "DEMO" if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "REAL"
    print("[OK] Connected to MT5")
    print(f"     Account: {info.login} ({info.server})")
    print(f"     Balance: {info.balance:.2f} {info.currency}")
    print(f"     Type:    {mode}")

    if mode == "REAL":
        print()
        print("[WARN] You are on a REAL account. The bot will refuse to place orders here")
        print("       until you explicitly opt in via config. For Phase 0 this is fine.")

    found = None
    for sym in GOLD_CANDIDATES:
        s = mt5.symbol_info(sym)
        if s is not None:
            found = sym
            break

    if found is None:
        all_syms = mt5.symbols_get() or []
        gold_syms = [s.name for s in all_syms if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
        print()
        print(f"[WARN] None of the common gold symbols were found on this broker.")
        print(f"       Gold-like symbols available: {gold_syms or '(none)'}")
        print(f"       Tell the developer which one to use in config/settings.yaml.")
        mt5.shutdown()
        return 1

    if not mt5.symbol_select(found, True):
        print(f"[WARN] Could not add {found} to Market Watch (symbol_select failed).")

    tick = mt5.symbol_info_tick(found)
    if tick is None or (tick.bid == 0 and tick.ask == 0):
        print(f"[WARN] Symbol {found} exists but has no live tick (market closed?)")
        print(f"       Gold trades ~ Mon 00:00 - Sat 00:00 UTC. Try again when market is open.")
        mt5.shutdown()
        return 0

    spread = tick.ask - tick.bid
    print()
    print(f"[OK] Gold symbol: {found}")
    print(f"     Current bid/ask: {tick.bid} / {tick.ask}")
    print(f"     Spread:          {spread:.5f}")

    mt5.shutdown()
    print()
    print("[PHASE 0 GREEN] Save the symbol name above. Paste this output back to continue to Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
