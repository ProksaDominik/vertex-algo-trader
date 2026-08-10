"""Fetch DAILY bars (2015-present) for the 7 majors that have both Dukascopy
support and CFTC COT coverage -- daily resolution is plenty for a weekly-
frequency COT signal and is far faster/lighter than 15m history.

Output: data/<symbol>_1d.parquet

Usage: python scripts/fetch_daily_history.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from dukascopy_python import fetch, INTERVAL_DAY_1, OFFER_SIDE_BID
from dukascopy_python import instruments as _inst

ROOT = Path(__file__).resolve().parents[1]

SYMBOL_TO_INSTRUMENT_NAME = {
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "USDCHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "NZDUSD": "INSTRUMENT_FX_MAJORS_NZD_USD",
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    "US500": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
}


def main() -> int:
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    stop = datetime(2026, 7, 28, tzinfo=timezone.utc)
    for symbol, inst_name in SYMBOL_TO_INSTRUMENT_NAME.items():
        inst = getattr(_inst, inst_name)
        print(f"[fetch] {symbol}: {start.date()} -> {stop.date()}")
        df = fetch(inst, INTERVAL_DAY_1, OFFER_SIDE_BID, start, stop)
        if df is None or df.empty:
            print(f"  [FAIL] no bars")
            continue
        df = df.rename(columns=str.lower)
        df.index.name = "timestamp"
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        out = ROOT / "data" / f"{symbol.lower()}_1d.parquet"
        df.to_parquet(out)
        print(f"  [OK] {len(df):,} bars -> {out.name}  range {df.index[0]} -> {df.index[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
