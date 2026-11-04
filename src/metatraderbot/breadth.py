"""Market breadth for the race, free (owner, 2026-10-21: "A-D breadth ... I really like these numbers" -- race the
side most stocks moved the session before; it beat the 200-day rule in the 1-account plan on both price histories,
see notes/Research log).

The official NYSE advancers-minus-decliners number is not on a free feed, so it is counted here: of the S&P 500
stocks, how many closed up minus how many closed down in the last US session -- from Yahoo's daily closes, like the
bots' prices (free_prices). The S&P 500 list comes from Wikipedia (kept a month). Counts are kept in
data/breadth/sp500_ad.json, one row per session date: {"adv", "dec", "n"}.

    ensure(data_dir)            count the sessions that are missing (about 30 s for 500 stocks, 8 at a time)
    side_for(day, data_dir)     (+1 buy / -1 sell / 0 unknown, why) for a Lucid trading day's race
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ET = "America/New_York"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MIN_STOCKS = 300                  # a session counts only when this many stocks have both closes
LIST_DAYS = 30                    # how long the S&P 500 list is kept before it is read again
STALE_DAYS = 5                    # a count older than this (calendar days) is not used for a race


def _dir(data_dir: Path) -> Path:
    d = Path(data_dir) / "breadth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_symbols(html: str) -> list[str]:
    """The tickers of Wikipedia's constituents table, in Yahoo's spelling (BRK.B -> BRK-B)."""
    table = html.split('id="constituents"', 1)[1].split("</table>", 1)[0]
    syms = re.findall(r'<tr[^>]*>\s*<td[^>]*><a[^>]*>([A-Z.\-]+)</a>', table)
    return sorted({s.replace(".", "-") for s in syms})


def symbols(data_dir: Path, fetch_html=None) -> list[str]:
    p = _dir(data_dir) / "sp500.json"
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - p.stat().st_mtime < LIST_DAYS * 86400 and len(saved) >= 400:
            return saved
    except (OSError, ValueError):
        saved = []
    try:
        if fetch_html is None:
            req = urllib.request.Request(WIKI, headers={"User-Agent": "Mozilla/5.0 (Vertex breadth)"})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        else:
            html = fetch_html()
        syms = parse_symbols(html)
        if len(syms) >= 400:
            p.write_text(json.dumps(syms), encoding="utf-8")
            return syms
        log.warning("S&P 500 list: only %d symbols read -- keeping the saved list", len(syms))
    except Exception as e:  # noqa: BLE001 -- no internet / page changed: the saved list
        log.warning("S&P 500 list not read (%r) -- keeping the saved list", e)
    return saved


def load(data_dir: Path) -> dict[str, dict]:
    try:
        return json.loads((_dir(data_dir) / "sp500_ad.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def count(closes: dict[str, pd.Series]) -> dict[str, dict]:
    """{session date: {"adv", "dec", "n"}} from each stock's daily closes (indexed by session date)."""
    px = pd.DataFrame(closes).sort_index()
    px = px[~px.index.duplicated()]
    chg = px.pct_change()
    n = chg.notna().sum(axis=1)
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    return {str(pd.Timestamp(d).date()): {"adv": int(adv[d]), "dec": int(dec[d]), "n": int(n[d])}
            for d in px.index if n[d] >= MIN_STOCKS}


def last_session(now: datetime | None = None) -> str:
    """The latest US session whose closes are in (from 16:30 New York), as a date (weekends skipped;
    a holiday simply never gets a count)."""
    from zoneinfo import ZoneInfo
    t = now or datetime.now(ZoneInfo(ET))
    t = (t if t.tzinfo else t.replace(tzinfo=ZoneInfo(ET))).astimezone(ZoneInfo(ET))
    d = t.date() if t.hour * 60 + t.minute >= 16 * 60 + 30 else t.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return str(d)


def _session_dates(index) -> pd.DatetimeIndex:
    ix = pd.DatetimeIndex(index)
    return pd.DatetimeIndex((ix.tz_convert(ET) if ix.tz is not None else ix).date)


def ensure(data_dir: Path, fetch=None, now: datetime | None = None, rng: str = "1mo", workers: int = 8,
           retry_s: float = 1200) -> dict:
    """Count the sessions that are not saved yet (the last month of daily closes for every stock). Tried at most
    every `retry_s` seconds while the latest session is still missing (a holiday, Yahoo not updated yet)."""
    have = load(data_dir)
    want = last_session(now)
    if want in have:
        return have
    mark = _dir(data_dir) / "last_try.txt"
    try:
        if time.time() - mark.stat().st_mtime < retry_s:
            return have
    except OSError:
        pass
    mark.write_text(datetime.now().isoformat(), encoding="utf-8")
    if fetch is None:
        from . import free_prices
        fetch = free_prices._fetch
    syms = symbols(data_dir)
    if not syms:
        return have

    def one(s: str):
        try:
            d = fetch(s, "1d", rng)["close"]
            d.index = _session_dates(d.index)
            return s, d[~d.index.duplicated(keep="last")]
        except Exception:  # noqa: BLE001 -- one stock missing does not matter
            return s, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        closes = {s: d for s, d in ex.map(one, syms) if d is not None and len(d)}
    new = count(closes)
    have.update(new)
    tmp = _dir(data_dir) / "sp500_ad.json.tmp"
    tmp.write_text(json.dumps(dict(sorted(have.items())), indent=0), encoding="utf-8")
    tmp.replace(_dir(data_dir) / "sp500_ad.json")
    log.info("breadth: %d stocks read, sessions counted up to %s", len(closes), max(new) if new else "none")
    return have


def side_for(trading_day: str, data_dir: Path, counts: dict | None = None) -> tuple[int, str]:
    """(+1 buy / -1 sell / 0 unknown, why) for a Lucid trading day: the last session BEFORE that day."""
    counts = load(data_dir) if counts is None else counts
    before = [d for d in counts if d < trading_day]
    if not before:
        return 0, "no breadth count yet"
    d = max(before)
    if (pd.Timestamp(trading_day) - pd.Timestamp(d)).days > STALE_DAYS:
        return 0, f"breadth count too old ({d})"
    c = counts[d]
    diff = c["adv"] - c["dec"]
    side = 1 if diff > 0 else (-1 if diff < 0 else 0)
    word = "long" if side > 0 else ("SHORT" if side < 0 else "no side")
    return side, (f"{word}: {d} S&P 500 stocks {c['adv']} up / {c['dec']} down "
                  f"({'+' if diff >= 0 else ''}{diff})")
