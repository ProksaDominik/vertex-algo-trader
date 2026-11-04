"""Paper trading without NinjaTrader's market-data subscription (owner, 2026-10-15: "we go without
NinjaTrader" -- its real-time data stopped being free and costs ~$100 a month).

The bots keep running unchanged; only their engine is swapped (VERTEX_PAPER=1 in .env):
  * prices: NinjaTrader's free live 1-minute NQ bars when NinjaTrader is open and signed in (the `feed`, a
    NinjaTraderEngine used read-only -- no order ever reaches it); otherwise Yahoo Finance's NQ bars from the
    internet, 10 minutes delayed (free_prices.py) -- so NinjaTrader is not needed at all (2026-10-15: its sign-in
    kept failing for hours and the bots had no prices);
  * fills: here, the way the Fleet page and the 21-month big test fill them (scripts/soak_replay.py) -- a buy
    at the latest minute's close plus one tick, a sell at the close; afterwards each new minute is checked,
    the stop first when a minute touches both, the take-profit when price reaches it; commission $1.75 a
    side per contract;
  * memory: open trades and results are saved in data/paper_<account>.json after every change, so a restart
    forgets nothing (on 14 Oct NinjaTrader's restart made the bridge forget an open trade).
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import free_prices
from .nt_engine import (POSITION_TYPE_BUY, POSITION_TYPE_SELL, RETCODE_DONE, AccountInfo, NinjaTraderEngine,
                        OrderResult, Position)

ROOT = Path(__file__).resolve().parents[2]
TICK = 0.25
COMM = 1.75                     # per contract per side, as in the replays
FRESH_S = 180                   # the newest traded minute must have ended within this many seconds
BARS_CACHE_S = 10


def _utc(ts) -> datetime:
    return pd.Timestamp(ts).tz_convert("UTC").to_pydatetime() if pd.Timestamp(ts).tzinfo else \
        pd.Timestamp(ts).tz_localize("UTC").to_pydatetime()


class PaperEngine:
    """The NinjaTraderEngine surface the race bots use; prices from NinjaTrader's minute bars, fills here."""

    def __init__(self, account: str = "Sim101", feed=None, path: Path | None = None,
                 max_contract_units: float | None = None, now=None, read_only: bool = False, **_):
        self.account = account
        self.read_only = read_only           # the dashboard looks, only the bot writes
        if feed is None and os.environ.get("VERTEX_PRICE_SOURCE", "").strip().lower() == "ninjatrader":
            feed = NinjaTraderEngine(account=account, timeout_s=8)
        self.feed = feed                           # None: internet prices only (owner 2026-10-15)
        self.internet = free_prices                # the fallback price source (tests may replace it)
        self.source = "NinjaTrader"
        self.path = path or ROOT / "data" / f"paper_{re.sub(r'[^A-Za-z0-9_-]+', '_', account)}.json"
        self.cap = max_contract_units
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._bars: pd.DataFrame | None = None
        self._bars_at = 0.0
        try:
            self.book = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.book = {"account": account, "balance": 50_000.0, "tickets": {}}

    # --- memory ------------------------------------------------------------------------------------------
    def _save(self) -> None:
        if self.read_only:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.book, indent=1, default=str), encoding="utf-8")
        for attempt in range(5):                 # OneDrive may hold the file for a moment
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.3 * (attempt + 1))

    # --- the market: NinjaTrader's minute bars --------------------------------------------------------------
    def resolve_symbol(self, symbol: str) -> str:
        return "NQ"                                # paper trades the instrument, not a contract month

    def _traded_age(self, bars: pd.DataFrame) -> float:
        traded = bars[bars["volume"] > 0]
        if not len(traded):
            return float("inf")
        return max(0.0, (self._now() - (_utc(traded.index[-1]) + timedelta(minutes=1))).total_seconds())

    def minute_bars(self) -> pd.DataFrame:
        """Recent NQ minute bars, the newest (still forming) one last; cached a few seconds. NinjaTrader's live
        bars when it has fresh ones, else the internet's (10 minutes delayed)."""
        if self._bars is None or time.monotonic() - self._bars_at > BARS_CACHE_S:
            bars = None
            try:
                if self.feed is None:
                    raise LookupError("internet prices only")
                nt = self.feed.get_bars("NQ", "1m", 1500)
                if self._traded_age(nt) <= FRESH_S:
                    bars, self.source = nt, "NinjaTrader"
            except Exception:  # noqa: BLE001 -- NinjaTrader closed or not signed in
                pass
            if bars is None:
                bars, self.source = self.internet.bars("1m", 1500), "internet"   # ~a whole race (19:00 -> 16:00)
            self._bars, self._bars_at = bars, time.monotonic()
        return self._bars

    def _price_and_age(self) -> tuple[float, float]:
        """The latest price and how stale it is for its source (a delayed feed counts from its own delay)."""
        bars = self.minute_bars()
        age = self._traded_age(bars)
        if self.source == "internet":
            age = max(0.0, age - self.internet.DELAY_S)
        return float(bars["close"].iloc[-1]), age

    def quote(self, symbol: str) -> dict:
        p, age = self._price_and_age()
        return {"instrument": self.resolve_symbol(symbol), "bid": p, "ask": p + TICK, "last": p, "age_s": age}

    def is_tradeable(self, symbol: str, max_quote_age_s: float = FRESH_S) -> bool:
        try:
            _, age = self._price_and_age()
        except Exception:  # noqa: BLE001 -- NinjaTrader closed: no prices, no trading
            return False
        return age <= max(max_quote_age_s, FRESH_S)

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if timeframe == "1m":                     # the same source as the prices the trades fill at
            return self.minute_bars().tail(count)
        try:
            if self.feed is None:
                raise LookupError("internet prices only")
            return self.feed.get_bars(symbol, timeframe, count)
        except Exception:  # noqa: BLE001 -- NinjaTrader closed: daily bars from the internet
            return self.internet.bars(timeframe, count)

    def usd_per_point(self, symbol: str):
        try:
            return self.feed.usd_per_point(symbol) if self.feed is not None else 20.0   # NQ
        except Exception:  # noqa: BLE001
            return 20.0

    def volume_spec(self, symbol: str):
        try:
            return self.feed.volume_spec(symbol) if self.feed is not None else (1.0, 4.0, 1.0)
        except Exception:  # noqa: BLE001
            return (1.0, 4.0, 1.0)

    def bridge_version(self) -> tuple:
        try:
            return self.feed.bridge_version() if self.feed is not None else (1, 4)
        except Exception:  # noqa: BLE001
            return (1, 4)

    # --- the account ----------------------------------------------------------------------------------------
    def account_info(self) -> AccountInfo:
        bal = float(self.book.get("balance", 50_000.0))
        return AccountInfo(login=self.account, server="paper (NinjaTrader minute bars)", balance=bal, equity=bal,
                           margin=0.0, free_margin=bal, currency="USD", is_demo=True)

    def connect(self) -> AccountInfo:
        self.minute_bars()                         # NinjaTrader must be open: it is the price source
        return self.account_info()

    # --- trades -------------------------------------------------------------------------------------------
    def _open(self) -> dict[str, dict]:
        return {k: t for k, t in self.book["tickets"].items() if t["open"]}

    def _update(self) -> None:
        """Walk the minutes since each open trade was last checked: the stop first, then the take-profit."""
        opn = self._open()
        if not opn:
            return
        bars = self.minute_bars()
        done = bars.iloc[:-1]                      # completed minutes only
        changed = False
        for k, t in opn.items():
            after = pd.Timestamp(t["checked_through"])
            rows = done[done.index > after]
            if not len(rows):
                continue
            sg = t["sign"]
            hit = None
            for ts, r in rows.iterrows():
                adverse, favour = (r["low"], r["high"]) if sg > 0 else (r["high"], r["low"])
                if sg * (adverse - t["sl"]) <= 0:
                    hit = ("sl", t["sl"] - sg * TICK, ts)
                    break
                if sg * (favour - t["tp"]) >= 0:
                    hit = ("tp", t["tp"], ts)
                    break
            if hit:
                why, px, ts = hit
                self._close(k, px, why, _utc(ts) + timedelta(minutes=1))
            else:
                t["checked_through"] = str(rows.index[-1])
            changed = True
        if changed:
            self._save()

    def _close(self, k: str, px: float, why: str, when: datetime) -> None:
        t = self.book["tickets"][k]
        pnl = (px - t["entry"]) * t["sign"] * t["qty"] * t["pv"] - 2 * COMM * t["qty"]
        t.update(open=False, exit=px, why=why, closed=when.isoformat(), pnl=round(pnl, 2))
        self.book["balance"] = round(float(self.book.get("balance", 50_000.0)) + pnl, 2)

    def place_market_order(self, symbol, direction, lots, sl, tp, magic, comment="", signal_entry_price=None,
                           **_) -> OrderResult:
        self._update()
        qty = int(lots)
        open_qty = sum(t["qty"] for t in self._open().values())
        if self.cap is not None and open_qty + qty > self.cap:
            return OrderResult(ticket=0, volume=0.0, price=0.0, sl=sl, tp=tp, retcode=10019,
                               comment=f"paper: contract cap {self.cap:g} reached")
        p, age = self._price_and_age()
        if age > FRESH_S:
            return OrderResult(ticket=0, volume=0.0, price=0.0, sl=sl, tp=tp, retcode=10018,
                               comment="paper: no live minute bars")
        sign = 1 if direction == "long" else -1
        fill = p + TICK if sign > 0 else p
        ref = signal_entry_price or fill
        k = str(max(int(time.time() * 1000), int(max(self.book["tickets"], key=int, default="0")) + 1))
        bars = self.minute_bars()
        t = {"symbol": self.resolve_symbol(symbol), "sign": sign, "qty": qty, "entry": fill,
             "sl": fill - (ref - sl), "tp": fill + (tp - ref), "magic": int(magic), "comment": comment,
             "pv": float(self.usd_per_point(symbol) or 20.0), "opened": self._now().isoformat(),
             "checked_through": str(bars.index[-1]), "open": True}
        self.book["tickets"][k] = t
        self._save()
        return OrderResult(ticket=int(k), volume=float(qty), price=fill, sl=t["sl"], tp=t["tp"], retcode=RETCODE_DONE,
                           comment="paper fill")

    def _position(self, k: str, t: dict) -> Position:
        p, _ = self._price_and_age()
        return Position(ticket=int(k), magic=t["magic"], symbol=t["symbol"], volume=float(t["qty"]),
                        type=POSITION_TYPE_BUY if t["sign"] > 0 else POSITION_TYPE_SELL, price_open=t["entry"],
                        price_current=p, sl=t["sl"], tp=t["tp"],
                        profit=(p - t["entry"]) * t["sign"] * t["qty"] * t["pv"],
                        time=int(datetime.fromisoformat(t["opened"]).timestamp()))

    def get_position(self, ticket) -> Position | None:
        self._update()
        t = self.book["tickets"].get(str(ticket))
        return self._position(str(ticket), t) if t and t["open"] else None

    def get_positions(self, magic: int) -> list:
        self._update()
        return [self._position(k, t) for k, t in self._open().items() if t["magic"] == int(magic)]

    def get_all_open_tickets(self) -> set[int]:
        self._update()
        return {int(k) for k in self._open()}

    def open_contracts(self) -> dict[str, int]:
        self._update()
        out: dict[str, int] = {}
        for t in self._open().values():
            out[t["symbol"]] = out.get(t["symbol"], 0) + t["sign"] * t["qty"]
        return {k: v for k, v in out.items() if v}

    def get_closed_deal(self, ticket, offset_fallback: int = 0) -> dict | None:
        self._update()
        t = self.book["tickets"].get(str(ticket))
        if not t or t["open"]:
            return None
        return {"exit_price": float(t["exit"]), "pnl_usd": float(t["pnl"]),
                "exit_reason": {"sl": "sl", "tp": "tp"}.get(t["why"], "manual"), "close_reason": t["why"],
                "closed_ts": datetime.fromisoformat(t["closed"])}

    def close_position(self, ticket, max_retries: int = 3) -> bool:
        self._update()
        t = self.book["tickets"].get(str(ticket))
        if not t or not t["open"]:
            return False
        p, age = self._price_and_age()
        if age > FRESH_S:
            return False                          # no live price: a close would be at a made-up price
        self._close(str(ticket), p - TICK if t["sign"] > 0 else p + TICK, "manual", self._now())
        self._save()
        return True

    def close_all_bot_positions(self) -> int:
        return sum(0 if self.close_position(int(k)) else 1 for k in list(self._open()))

    def flatten(self, symbol: str) -> int:
        before = self.open_contracts().get(self.resolve_symbol(symbol), 0)
        for k in [k for k, t in self._open().items() if t["symbol"] == self.resolve_symbol(symbol)]:
            self.close_position(int(k))
        return before

    def modify_sltp(self, ticket, sl: float, tp: float) -> bool:
        t = self.book["tickets"].get(str(ticket))
        if not t or not t["open"]:
            return False
        t.update(sl=float(sl), tp=float(tp))
        self._save()
        return True
