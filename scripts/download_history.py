"""Download MULTI-YEAR 15-minute history for the whole test basket from Dukascopy.

The repo shipped with only calendar-2024 on disk (`data/<sym>_15m.parquet`),
which is one year — far too short to separate a real edge from a 2024 regime
artifact. This pulls ~10 years per symbol into `data/<sym>_15m_long.parquet`
so the existing 2024 files stay untouched and every old result stays
reproducible.

Usage:
    python scripts/download_history.py                    # full basket, 2016-01-01 -> today
    python scripts/download_history.py EURUSD XAUUSD      # just these
    python scripts/download_history.py --start 2018-01-01
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from dukascopy_python import fetch, INTERVAL_MIN_15, OFFER_SIDE_BID
    from dukascopy_python import instruments as _inst
except ImportError as e:  # pragma: no cover
    print(f"[FAIL] dukascopy-python not installed: {e}")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"

# The full 21-symbol basket the strategy comparisons use, mapped to Dukascopy
# instrument constants. Indices are resolved dynamically (their constant names
# vary by package version) and simply skipped when unavailable.
SYMBOLS: dict[str, str] = {
    "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD",
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY",
    "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD",
    "USDCHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "NZDUSD": "INSTRUMENT_FX_MAJORS_NZD_USD",
    "EURJPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "GBPJPY": "INSTRUMENT_FX_CROSSES_GBP_JPY",
    "NZDJPY": "INSTRUMENT_FX_CROSSES_NZD_JPY",
    "AUDJPY": "INSTRUMENT_FX_CROSSES_AUD_JPY",
    "CHFJPY": "INSTRUMENT_FX_CROSSES_CHF_JPY",
    "GBPCHF": "INSTRUMENT_FX_CROSSES_GBP_CHF",
    "EURCHF": "INSTRUMENT_FX_CROSSES_EUR_CHF",
    "AUDCHF": "INSTRUMENT_FX_CROSSES_AUD_CHF",
    "EURAUD": "INSTRUMENT_FX_CROSSES_EUR_AUD",
    "AUDCAD": "INSTRUMENT_FX_CROSSES_AUD_CAD",
    "EURGBP": "INSTRUMENT_FX_CROSSES_EUR_GBP",
    "XAUUSD": "INSTRUMENT_FX_METALS_XAU_USD",
    "US500": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",

    # NON-FX, added 2026-08-05. Everything tested until now was FX
    # mean-reversion, which is ONE quadrant: 17 pairs worth only ~4.9
    # independent bets (scripts/correlation_audit.py). That cap on
    # independence is what limits the portfolio's daily mu/sigma to 0.03-0.12
    # and therefore caps the FTMO pass rate -- see
    # scripts/variance_reduction_race.py, where cutting volatility just cut
    # drift with it.
    #
    # Indices, commodities and crypto are structurally uncorrelated with FX
    # majors AND they trend rather than mean-revert, so trend-following (which
    # loses on FX) is the right tool for them. Two uncorrelated streams is the
    # only untested route to a higher mu/sigma.
    "US30":   "INSTRUMENT_IDX_AMERICA_E_D_J_IND",
    "NAS100": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
    "DE40":   "INSTRUMENT_IDX_EUROPE_E_DAAX",
    "JP225":  "INSTRUMENT_IDX_ASIA_E_N225JAP",
    "BRENT":  "INSTRUMENT_CMD_ENERGY_E_BRENT",
    "WTI":    "INSTRUMENT_CMD_ENERGY_E_LIGHT",
    "COPPER": "INSTRUMENT_CMD_METALS_COPPER_CMD_USD",
    "XPTUSD": "INSTRUMENT_CMD_METALS_XPT_CMD_USD",
    "BTCUSD": "INSTRUMENT_VCCY_BTC_USD",
    "ETHUSD": "INSTRUMENT_VCCY_ETH_USD",
}


def resolve(name: str):
    return getattr(_inst, name, None)


def fetch_symbol(symbol: str, const_name: str, start: datetime, stop: datetime) -> int:
    inst = resolve(const_name)
    if inst is None:
        print(f"  [SKIP] {symbol}: no Dukascopy constant {const_name}")
        return 0

    out = OUT_DIR / f"{symbol.lower()}_15m_long.parquet"
    frames: list[pd.DataFrame] = []

    # Year-at-a-time keeps each request small enough that a transient failure
    # only costs one year, not the whole symbol.
    for year in range(start.year, stop.year + 1):
        y0 = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
        y1 = min(stop, datetime(year, 12, 31, 23, 59, tzinfo=timezone.utc))
        if y0 >= y1:
            continue
        for attempt in range(3):
            try:
                df = fetch(inst, INTERVAL_MIN_15, OFFER_SIDE_BID, y0, y1)
                if df is not None and len(df):
                    frames.append(df)
                    print(f"  {symbol} {year}: {len(df):>6} bars")
                else:
                    print(f"  {symbol} {year}: empty")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [WARN] {symbol} {year} failed after 3 tries: {e!r}")
                else:
                    time.sleep(2 * (attempt + 1))

    if not frames:
        print(f"  [FAIL] {symbol}: nothing downloaded")
        return 0

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    full.columns = [c.lower() for c in full.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in full.columns]
    full = full[keep].dropna()
    full.to_parquet(out)
    print(f"  [OK] {symbol}: {len(full)} bars {full.index[0].date()} -> {full.index[-1].date()} -> {out.name}")
    return len(full)


def main() -> int:
    args = [a for a in sys.argv[1:]]
    start_str = "2016-01-01"
    if "--start" in args:
        i = args.index("--start")
        start_str = args[i + 1]
        del args[i:i + 2]

    wanted = [a.upper() for a in args] or list(SYMBOLS)
    start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    stop = datetime.now(timezone.utc)

    print(f"Downloading {len(wanted)} symbols, {start.date()} -> {stop.date()}, 15m bars")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for sym in wanted:
        const = SYMBOLS.get(sym)
        if const is None:
            print(f"[SKIP] {sym}: not in basket map")
            continue
        print(f"\n=== {sym} ===")
        total += fetch_symbol(sym, const, start, stop)

    print(f"\nDONE. {total:,} bars written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
