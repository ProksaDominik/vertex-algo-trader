"""
Download historical XAUUSD 15-minute bars from Dukascopy.

No broker, no MT5, no API key. Dukascopy (Swiss bank) exposes decades of tick
data via public S3 buckets; the `dukascopy-python` package wraps that.

Usage:
    python scripts/download_data.py                 # uses config dates
    python scripts/download_data.py 2024-01-01 2024-12-31
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    from dukascopy_python import fetch, INTERVAL_MIN_15, OFFER_SIDE_BID
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD as INSTRUMENT_XAUUSD
except ImportError as e:
    print(f"[FAIL] dukascopy-python import error: {e}")
    print("       Run: pip install -r requirements.txt")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    bt = cfg["backtest"]
    start_str = sys.argv[1] if len(sys.argv) > 1 else bt["start_date"]
    stop_str = sys.argv[2] if len(sys.argv) > 2 else bt["end_date"]
    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(stop_str).replace(tzinfo=timezone.utc)

    print(f"[INFO] Fetching XAUUSD 15m bars: {start.date()} -> {stop.date()}")
    print("       This can take 1-3 minutes depending on the range.")

    df = fetch(INSTRUMENT_XAUUSD, INTERVAL_MIN_15, OFFER_SIDE_BID, start, stop)

    if df is None or df.empty:
        print("[FAIL] No bars returned. Check dates / network / instrument name.")
        return 1

    df = df.rename(columns=str.lower)
    df.index.name = "timestamp"
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    out = ROOT / cfg["backtest"]["data_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print()
    print(f"[OK] Saved {len(df):,} bars to {out}")
    print(f"     Range:  {df.index[0]} -> {df.index[-1]}")
    print()
    print("Last 3 bars:")
    print(df.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
