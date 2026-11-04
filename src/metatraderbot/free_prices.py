"""Free market prices from the internet -- everything runs without NinjaTrader (owner, 2026-10-15: "I want
everything to run on the other one with delay because NinjaTrader is paid" -- its real-time data costs ~$100 a
month, and its sign-in kept failing for hours).

Yahoo Finance's futures (front contract, rolled by Yahoo), **10 minutes delayed** for CME / CBOT / NYMEX:
NQ, ES, YM, RTY (Russell), GC (gold), CL (oil), the six currency futures (6E 6B 6J 6A 6C 6S), ZN (10-year
Treasuries), and the VIX index. Micro contracts (MNQ, MES, ...) use their full-size twin's prices (the same
market). Not available: NYSE TICK (market breadth) and trade-by-trade order flow.

Bars come with the newest (still forming) one last, indexed by the bar's start time in UTC, as NinjaTrader's do.
Intraday bars above 1 minute are built here from 5-minute bars (one download per market), so a lab polling 12
markets stays within Yahoo's limits.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&{span}"
DELAY_S = 600                     # Yahoo delays CME futures by 10 minutes
YAHOO = {"NQ": "NQ=F", "MNQ": "NQ=F", "ES": "ES=F", "MES": "ES=F", "YM": "YM=F", "MYM": "YM=F",
         "RTY": "RTY=F", "M2K": "RTY=F", "GC": "GC=F", "MGC": "GC=F", "CL": "CL=F", "MCL": "CL=F",
         "6E": "6E=F", "6B": "6B=F", "6J": "6J=F", "6A": "6A=F", "6C": "6C=F", "6S": "6S=F", "ZN": "ZN=F",
         "VIX": "^VIX", "^VIX": "^VIX"}
RESAMPLE = {"15m": "15min", "30m": "30min", "1h": "1h"}
TTL = {"1m": 20, "5m": 90, "1d": 3600}
_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def ticker(symbol: str) -> str:
    """Yahoo's ticker for a symbol or a NinjaTrader contract name ("NQ DEC26" -> "NQ=F")."""
    root = str(symbol).split()[0].upper()
    return YAHOO.get(root, root if root.startswith("^") or root.endswith("=F") else root + "=F")


def _fetch(tk: str, interval: str, rng: str) -> pd.DataFrame:
    """rng: a Yahoo range ("60d", "2y") or "<minutes>m" = the last that many minutes (period1/period2)."""
    if rng.endswith("m") and rng[:-1].isdigit():
        now = int(time.time())
        span = f"period1={now - 60 * int(rng[:-1])}&period2={now}"
    else:
        span = f"range={rng}"
    url = URL.format(ticker=urllib.parse.quote(tk), interval=interval, span=span)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode("utf-8"))
    res = d["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    n = len(res.get("timestamp") or [])
    df = pd.DataFrame({"open": q.get("open") or [None] * n, "high": q.get("high") or [None] * n,
                       "low": q.get("low") or [None] * n, "close": q.get("close") or [None] * n,
                       "volume": [v or 0 for v in (q.get("volume") or [0] * n)]},
                      index=pd.to_datetime(res.get("timestamp") or [], unit="s", utc=True))
    df.index.name = "timestamp"
    return df.dropna(subset=["open", "high", "low", "close"]).astype(float).round(7)   # Yahoo sends float32


def _cached(tk: str, interval: str, rng: str) -> pd.DataFrame:
    key = f"{tk}/{interval}/{rng}"
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < TTL.get(interval, 60):
        return hit[1]
    df = _fetch(tk, interval, rng)
    _cache[key] = (time.monotonic(), df)
    return df


def bars(timeframe: str = "1m", count: int = 600, symbol: str = "NQ") -> pd.DataFrame:
    """Bars for `symbol`: "1m" (up to 5 days back), "5m" / "15m" / "30m" / "1h" (60 days, built from 5-minute
    bars), "1d" (2 years, 10 when more than ~480 are asked for)."""
    tk = ticker(symbol)
    if timeframe == "1m":
        # the last `count` minutes and some more, counted back from now -- Yahoo's "1d" range starts at midnight
        # New York and cut off the 19:00 race (2026-10-16); at most 7 days (Yahoo's limit for minute bars)
        minutes = min(int(count * 1.1) + 120, 7 * 24 * 60)
        df = _cached(tk, "1m", f"{(minutes // 60 + 1) * 60}m")
    elif timeframe == "5m" or timeframe in RESAMPLE:
        df = _cached(tk, "5m", "60d")
        if timeframe in RESAMPLE:
            df = df.resample(RESAMPLE[timeframe], label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(
                subset=["open", "high", "low", "close"])
    elif timeframe == "1d":
        df = _cached(tk, "1d", "10y" if count > 480 else "2y")
    else:
        raise ValueError(f"free prices: unsupported timeframe {timeframe}")
    return df.tail(count)


def age_s(df: pd.DataFrame, now: datetime | None = None, minutes: int = 1) -> float:
    """Seconds since the newest traded bar ended (bars of `minutes` length)."""
    traded = df[df["volume"] > 0] if (df["volume"] > 0).any() else df
    end = traded.index[-1] + pd.Timedelta(minutes=minutes)
    return ((now or datetime.now(timezone.utc)) - end.to_pydatetime()).total_seconds()


class FreeFeed:
    """The NinjaTraderEngine calls that read prices -- for the labs, the trade brain, the dashboard and the Fleet
    simulator. It places no orders (paper trades are filled by paper_engine.py)."""

    def connect(self):
        bars("1m", 5)                               # the internet must answer
        return SimpleNamespace(login="internet prices", server="Yahoo Finance (10 min delayed)", is_demo=True,
                               balance=0.0)

    account_info = connect

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        return bars(timeframe, count, symbol)

    def resolve_symbol(self, symbol: str) -> str:
        return str(symbol).split()[0].upper()

    def quote(self, symbol: str) -> dict:
        b = bars("1m", 30, symbol)
        p = float(b["close"].iloc[-1])
        return {"instrument": self.resolve_symbol(symbol), "bid": p, "ask": p, "last": p,
                "age_s": max(0.0, age_s(b) - DELAY_S)}

    def is_tradeable(self, symbol: str, max_quote_age_s: float = 300.0) -> bool:
        """Fresh for a delayed feed: the newest minute ended at most ~5 minutes after the 10-minute delay."""
        try:
            return self.quote(symbol)["age_s"] <= max(max_quote_age_s, 300.0)
        except Exception:  # noqa: BLE001 -- no internet / market closed
            return False

    def usd_per_point(self, symbol: str):
        return None

    def bridge_version(self) -> tuple:
        return (0, 0)


def in_use() -> bool:
    """True while the bots paper-trade on internet prices: VERTEX_PAPER=1 in .env, and no
    VERTEX_PRICE_SOURCE=ninjatrader (the way back to NinjaTrader's live prices once its data is paid for)."""
    import os
    from pathlib import Path
    paper, src = os.environ.get("VERTEX_PAPER"), os.environ.get("VERTEX_PRICE_SOURCE")
    if paper is None or src is None:
        try:
            from dotenv import dotenv_values
            env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
            paper = paper if paper is not None else env.get("VERTEX_PAPER")
            src = src if src is not None else env.get("VERTEX_PRICE_SOURCE")
        except Exception:  # noqa: BLE001
            pass
    return str(paper or "").strip() == "1" and str(src or "internet").strip().lower() != "ninjatrader"
