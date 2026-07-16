"""
Download 15-minute bars for any supported pair from Dukascopy.

Usage:
    python scripts/download_pair.py EURUSD
    python scripts/download_pair.py EURUSD 2024-01-01 2024-12-31

Output: data/<symbol_lowercase>_15m.parquet
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    from dukascopy_python import fetch, INTERVAL_MIN_15, OFFER_SIDE_BID
    from dukascopy_python import instruments as _inst
except ImportError as e:
    print(f"[FAIL] dukascopy-python not installed: {e}")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]

SYMBOL_TO_INSTRUMENT_NAME = {
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "US500":  "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
    "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "USDCHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "NZDUSD": "INSTRUMENT_FX_MAJORS_NZD_USD",
}


def resolve_instrument(symbol: str):
    name = SYMBOL_TO_INSTRUMENT_NAME.get(symbol.upper())
    if name is None:
        raise KeyError(
            f"Unknown symbol {symbol}. Supported: {sorted(SYMBOL_TO_INSTRUMENT_NAME)}"
        )
    if not hasattr(_inst, name):
        raise AttributeError(f"Dukascopy has no instrument constant {name}")
    return getattr(_inst, name), name


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_pair.py SYMBOL [START] [END]")
        print(f"Supported: {sorted(SYMBOL_TO_INSTRUMENT_NAME)}")
        return 1

    symbol = sys.argv[1].upper()
    try:
        inst, inst_name = resolve_instrument(symbol)
    except (KeyError, AttributeError) as e:
        print(f"[FAIL] {e}")
        return 1

    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    bt = cfg["backtest"]

    start_str = sys.argv[2] if len(sys.argv) > 2 else bt["start_date"]
    stop_str = sys.argv[3] if len(sys.argv) > 3 else bt["end_date"]
    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(stop_str).replace(tzinfo=timezone.utc)

    print(f"[INFO] {symbol} ({inst_name}): {start.date()} -> {stop.date()}")
    df = fetch(inst, INTERVAL_MIN_15, OFFER_SIDE_BID, start, stop)
    if df is None or df.empty:
        print("[FAIL] No bars returned.")
        return 1

    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    out = ROOT / "data" / f"{symbol.lower()}_15m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print(f"[OK] {symbol}: {len(df):,} bars saved to {out.name}")
    print(f"     Range: {df.index[0]} -> {df.index[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
