"""Download NQ order flow from NinjaTrader, one Lucid trading day at a time (2026-10-07).

Needs VertexBridge 1.3 (NinjaTrader: New > NinjaScript Editor, F5). For each trading day it
asks NinjaTrader for every trade from 18:00 ET the evening before to 17:00 ET, summed per
minute into volume bought (at the ask), sold (at the bid) and big trades (>= 10 contracts),
and saves data/orderflow/NQ_<day>.parquet. Goes back day by day until NinjaTrader has no
more trade history (5 weekdays in a row without trades) or --since. Rerunning skips days
already saved, so it can be stopped any time.

NinjaTrader loads every trade of the day (~0.5-1 million), so one day takes a while;
it runs beside the bots without touching them (its own connection, read-only).

Usage:  .venv/Scripts/python.exe scripts/fetch_orderflow.py [--since 2024-01-01] [--days 800]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot import lucid_rules  # noqa: E402
from metatraderbot.nt_engine import BridgeError, NinjaTraderEngine  # noqa: E402

ET = ZoneInfo("America/New_York")
OUT = ROOT / "data" / "orderflow"


def day_window(day: pd.Timestamp) -> tuple[datetime, datetime]:
    """Lucid day `day`: 18:00 ET the evening before (Sunday for a Monday) to 17:00 ET."""
    prev = day - pd.Timedelta(days=3 if day.dayofweek == 0 else 1)
    start = datetime(prev.year, prev.month, prev.day, 18, 0, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, 17, 0, tzinfo=ET)
    return start, end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2016-01-01")
    ap.add_argument("--days", type=int, default=2500, help="at most this many trading days back")
    ap.add_argument("--big", type=int, default=10)
    ap.add_argument("--account", default="Sim101")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    eng = NinjaTraderEngine(account=a.account, timeout_s=200.0)
    eng.connect()
    if eng.bridge_version() < (1, 3):
        print("VertexBridge 1.3 is not compiled yet: NinjaTrader > New > NinjaScript Editor > press F5, "
              "then run this again.")
        return 2
    today = pd.Timestamp(lucid_rules.trading_day(datetime.now(ET)))
    days = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=a.days)[::-1]
    empty_run, done = 0, 0
    for day in days:
        if day < pd.Timestamp(a.since):
            break
        f = OUT / f"NQ_{day:%Y-%m-%d}.parquet"
        if f.exists():
            empty_run = 0
            continue
        start, end = day_window(day)
        t0 = time.monotonic()
        try:
            df, info = eng.get_flow("NQ", start, end, big=a.big)
        except BridgeError as e:
            print(f"{day:%Y-%m-%d}: {e}")
            if "timed out" in str(e):
                continue
            return 1
        if df.empty:
            empty_run += 1
            print(f"{day:%Y-%m-%d}: no trades ({'holiday?' if empty_run < 5 else 'end of NinjaTrader history'})")
            if empty_run >= 5:
                break
            continue
        empty_run = 0
        df.to_parquet(f)
        done += 1
        vol = int(df[["buy", "sell", "other"]].to_numpy().sum())
        q = info["with_quote"] / max(info["ticks"], 1)
        print(f"{day:%Y-%m-%d}: {info['ticks']:,} trades, {vol:,} contracts, bought {df['buy'].sum() / max(vol, 1):.1%}"
              f", with bid/ask {q:.0%} ({time.monotonic() - t0:.0f}s)", flush=True)
    saved = sorted(OUT.glob("NQ_*.parquet"))
    print(f"done: {done} new days; {len(saved)} saved"
          + (f", {saved[0].stem[3:]} .. {saved[-1].stem[3:]}" if saved else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
