"""Pull historical CFTC Commitment-of-Traders (Legacy Futures-Only) data for
the 7 major currency futures via the public Socrata API (free, no key) and
cache it to data/cot_<currency>.parquet.

Net non-commercial (large speculator) position = long - short, as % of open
interest, is the classic COT positioning signal: extremes tend to precede
reversals (contrarian) or confirm trends (momentum), depending on the
strategy. This script only fetches/caches; scripts/cot_signal.py builds and
backtests the actual signal.

Usage: python scripts/fetch_cot.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONTRACTS = {
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "NZD": "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
}

BASE = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"


def fetch(exact_name: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    rows = []
    offset = 0
    limit = 1000
    while True:
        params = {
            "$where": f"market_and_exchange_names = '{exact_name}' AND "
                      f"report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00.000'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": str(limit),
            "$offset": str(offset),
        }
        url = BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows += batch
        offset += limit
        if len(batch) < limit:
            break
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    for code, name in CONTRACTS.items():
        print(f"[fetch] {code}: {name}")
        df = fetch(name)
        if df.empty:
            print(f"  [FAIL] no rows returned")
            continue
        keep = ["report_date_as_yyyy_mm_dd", "open_interest_all",
                "noncomm_positions_long_all", "noncomm_positions_short_all",
                "comm_positions_long_all", "comm_positions_short_all"]
        df = df[keep].copy()
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], utc=True)
        for c in keep[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.rename(columns={"report_date_as_yyyy_mm_dd": "date"}).sort_values("date")
        out = out_dir / f"cot_{code.lower()}.parquet"
        df.to_parquet(out)
        print(f"  [OK] {len(df)} weekly rows, {df['date'].min().date()} -> {df['date'].max().date()} -> {out.name}")


if __name__ == "__main__":
    main()
