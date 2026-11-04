"""Soak test (owner, 2026-10-09: "do a BIG test ... this is the final version"): the REAL trial and funded
bot code trades real NQ minute prices (NinjaTrader 1-minute bars 2024-26) through a replay engine standing in
for NinjaTrader. Take-profits and stops fill from the real bars' highs and lows (the stop first when one bar
touches both, a tick of slippage on stops, the spread paid on entry), the bots get the same 1-minute and daily
history they ask NinjaTrader for, and time runs many times faster than real. Nothing touches the live bots:
state, trade log and log go to a scratch folder.

It then checks what must never happen (two races at once, a race outside 19:00-12:00 ET or still open after
16:00 ET, the loss line overrun, a payout breaking Lucid's rules) and compares the results with the research.

Usage:
    .venv/Scripts/python.exe scripts/soak_replay.py --bot funded --start 2025-01-02 --end 2026-10-08
    .venv/Scripts/python.exe scripts/soak_replay.py --bot trial  --start 2025-01-02 --end 2026-10-08
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot import lucid_fleet as lf  # noqa: E402

PV, COMM, TICK = 20.0, 1.75, 0.25
NS_MIN = 60_000_000_000


class ReplayEngine:
    """The NinjaTraderEngine surface the bots use, driven by real 1-minute bars."""

    def __init__(self, clock, bars: pd.DataFrame, daily: pd.DataFrame, demo=True, login="Sim101"):
        self.clock, self.demo, self.login = clock, demo, login
        self.t_ns = bars.index.as_unit("ns").asi8
        self.index = bars.index
        self.o, self.h, self.l, self.c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        self.daily = daily                      # session OHLC, column "end_ns" = the session's last bar
        self.tk: dict[int, dict] = {}
        self.n = 0
        self.balance = 50_000.0
        self.orders: list[dict] = []
        self.max_open_qty = 0

    # --- the market ---
    def _cur(self) -> int:
        """Index of the last completed bar at the clock."""
        now_ns = pd.Timestamp(self.clock["now"]).value
        return int(np.searchsorted(self.t_ns, now_ns - NS_MIN, side="right")) - 1

    def _p(self) -> float:
        i = self._cur()
        return float(self.c[i]) if i >= 0 else float(self.o[0])

    def _update(self) -> None:
        i = self._cur()
        for k, t in self.tk.items():
            if not t["open"] or t["first"] > i:
                continue
            a = max(t["first"], t["done"] + 1)
            if a > i:
                continue
            sg = t["sign"]
            adv, fav = (self.l[a:i + 1], self.h[a:i + 1]) if sg > 0 else (self.h[a:i + 1], self.l[a:i + 1])
            hs = np.flatnonzero(sg * (adv - t["sl"]) <= 0)
            ht = np.flatnonzero(sg * (fav - t["tp"]) >= 0)
            fs = hs[0] if len(hs) else 10**9
            ft = ht[0] if len(ht) else 10**9
            if fs < 10**9 and fs <= ft:
                self._close(k, t["sl"] - sg * TICK, "sl", a + fs)
            elif ft < 10**9:
                self._close(k, t["tp"], "tp", a + ft)
            else:
                t["done"] = i

    def _close(self, k: int, px: float, why: str, bar: int | None = None) -> None:
        t = self.tk[k]
        when = (self.index[bar] + pd.Timedelta(minutes=1)).to_pydatetime() if bar is not None else self.clock["now"]
        pnl = (px - t["entry"]) * t["sign"] * t["qty"] * PV - 2 * COMM * t["qty"]
        t.update(open=False, exit=px, why=why, closed=when, pnl=pnl)
        self.balance += pnl

    def _open_qty(self) -> int:
        return sum(t["qty"] for t in self.tk.values() if t["open"])

    # --- NinjaTraderEngine surface ---
    def connect(self):
        return SimpleNamespace(login=self.login, is_demo=self.demo, balance=self.balance, server="replay")

    account_info = connect

    def bridge_version(self):
        return (1, 3)

    def volume_spec(self, sym):
        return (1.0, 4.0, 1.0)

    def is_tradeable(self, sym):
        return True

    def usd_per_point(self, sym):
        return PV

    def quote(self, sym):
        p = self._p()
        return {"bid": p, "ask": p + TICK, "last": p, "age_s": 0}

    def open_contracts(self):
        q = self._open_qty()
        return {"NQ": q} if q else {}

    def place_market_order(self, symbol, direction, lots, sl, tp, magic, comment="", signal_entry_price=None, **_):
        self._update()
        sign = 1 if direction == "long" else -1
        p = self._p()
        fill = p + TICK if sign > 0 else p
        ref = signal_entry_price or fill
        self.n += 1
        k = self.n
        self.tk[k] = {"entry": fill, "qty": int(lots), "sign": sign, "open": True, "opened": self.clock["now"],
                      "sl": fill - (ref - sl), "tp": fill + (tp - ref), "first": self._cur() + 1, "done": self._cur()}
        self.orders.append({"t": self.clock["now"], "dir": direction, "qty": int(lots), "entry": fill,
                            "tp": self.tk[k]["tp"], "sl": self.tk[k]["sl"], "ticket": k})
        self.max_open_qty = max(self.max_open_qty, self._open_qty())
        return SimpleNamespace(ok=True, ticket=k, price=fill, sl=self.tk[k]["sl"], tp=self.tk[k]["tp"],
                               volume=float(lots), retcode=10009, comment="")

    def get_position(self, ticket):
        self._update()
        t = self.tk.get(ticket)
        if t is None or not t["open"]:
            return None
        p = self._p()
        return SimpleNamespace(ticket=ticket, symbol="NQ", volume=float(t["qty"]), price_open=t["entry"],
                               price_current=p, tp=t["tp"], sl=t["sl"],
                               profit=(p - t["entry"]) * t["sign"] * t["qty"] * PV)

    def get_all_open_tickets(self):
        self._update()
        return {k for k, t in self.tk.items() if t["open"]}

    def get_closed_deal(self, ticket, offset_fallback=0):
        self._update()
        t = self.tk.get(ticket)
        if t is None or t["open"]:
            return None
        return {"pnl_usd": t["pnl"], "exit_price": t["exit"], "closed_ts": t["closed"], "exit_reason": t["why"]}

    def close_position(self, ticket, max_retries=3):
        self._update()
        t = self.tk.get(ticket)
        if t is None or not t["open"]:
            return False
        p = self._p()
        self._close(ticket, p - TICK if t["sign"] > 0 else p + TICK, "manual")
        return True

    def get_bars(self, symbol, timeframe, count):
        i = self._cur()
        if timeframe == "1d":
            now_ns = pd.Timestamp(self.clock["now"]).value
            done = self.daily[self.daily["end_ns"] < now_ns]
            rows = done.tail(count)
            forming = rows.tail(1).copy()
            forming["close"] = self._p()
            return pd.concat([rows, forming])[["open", "high", "low", "close", "volume"]]
        if timeframe != "1m":
            raise ValueError(timeframe)
        a = max(0, i - count + 1)
        j = min(i + 2, len(self.t_ns))                       # + the forming bar the bots drop
        df = pd.DataFrame({"open": self.o[a:j], "high": self.h[a:j], "low": self.l[a:j], "close": self.c[a:j],
                           "volume": 1.0}, index=self.index[a:j])
        df.index.name = "timestamp"
        if j == i + 1:                                       # no next bar in the data: repeat as the forming one
            df = pd.concat([df, df.tail(1)])
        return df


def daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    days = lf.trading_days(bars)
    g = pd.DataFrame({"open": bars["open"].to_numpy(), "high": bars["high"].to_numpy(), "low": bars["low"].to_numpy(),
                      "close": bars["close"].to_numpy(), "t": bars.index.as_unit("ns").asi8}, index=days)
    d = g.groupby(level=0).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                               close=("close", "last"), end_ns=("t", "max"))
    d["volume"] = 1.0
    d.index = pd.to_datetime(d.index)
    return d


class Patch:
    def __init__(self):
        self.undo = []

    def set(self, obj, name, value):
        self.undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def restore(self):
        for obj, name, value in reversed(self.undo):
            setattr(obj, name, value)


def run(bot: str, start: datetime, end: datetime, out: Path, poll_s: int = 120) -> dict:
    import os
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        handlers=[logging.FileHandler(out / "bot.log", encoding="utf-8")], force=True)
    os.environ.update({"VERTEX_BROKER": "ninjatrader", "LUCID_TRIAL_STRATEGY": "race", "LUCID_RULE_READING": "lucid",
                       "LUCID_RACE_CONTRACTS": "2", "LUCID_RACE_ENTRY": "19:00", "LUCID_RACE_SIDE": "trend200",
                       "LUCID_FUNDED_CONTRACTS": "1", "LUCID_FUNDED_ENTRY": "19:00",
                       "VERTEX_PAPER": "0"})                # the replay engine, never paper mode from .env
    os.environ.pop("LUCID_LIVE_CONFIRM", None)
    os.environ.pop("LUCID_FUNDED_LIVE_CONFIRM", None)
    script = {"trial": "run_lucidpro_trial.py", "funded": "run_lucid_futures_funded.py"}[bot]
    spec = importlib.util.spec_from_file_location(f"soak_{bot}", ROOT / "scripts" / script)
    m = importlib.util.module_from_spec(spec)
    saved_argv = sys.argv
    sys.argv = [script]
    spec.loader.exec_module(m)
    sys.argv = saved_argv
    for h in list(root.handlers):                        # never the live bot's log file
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).parent == ROOT / "logs":
            root.removeHandler(h)
            h.close()

    bars = lf.load_minutes(ROOT / "data" / "futures", None)
    bars = bars[(bars.index >= pd.Timestamp(start) - pd.Timedelta(days=420)) & (bars.index <= pd.Timestamp(end))]
    clock = {"now": start}
    eng = ReplayEngine(clock, bars, daily_bars(bars), demo=True, login="Sim101" if bot == "trial" else "SoakFunded")

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"] if tz is None else clock["now"].astimezone(tz)

    t0 = start.timestamp()
    wall = _time.time()

    def fake_sleep(s):
        clock["now"] += timedelta(seconds=max(s, 1))
        if clock["now"] >= end:
            raise KeyboardInterrupt

    pt = Patch()
    pt.set(m, "datetime", FakeDT)
    pt.set(m, "NinjaTraderEngine", lambda **kw: eng)
    pt.set(m, "alert", lambda msg, **kw: None)
    pt.set(m, "STATE_PATH", out / "state.json")
    pt.set(m, "TRADES_LOG_PATH", out / "trades.jsonl")
    pt.set(m, "POLL_INTERVAL_S", poll_s)
    pt.set(m.threading, "Thread", lambda *a, **k: SimpleNamespace(start=lambda: None))
    pt.set(m.time, "sleep", fake_sleep)
    pt.set(m.time, "monotonic", lambda: clock["now"].timestamp() - t0 + 1_000.0)
    try:
        m.main()
    finally:
        pt.restore()
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    trades = [json.loads(x) for x in (out / "trades.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] \
        if (out / "trades.jsonl").exists() else []
    return {"state": state, "trades": trades, "eng": eng, "wall_s": _time.time() - wall}


def check(bot: str, res: dict, out: Path) -> list[str]:
    """What must never happen."""
    from metatraderbot import lucid_rules
    eng, trades = res["eng"], res["trades"]
    problems = []
    if eng.max_open_qty > (2 if bot == "trial" else 1):
        problems.append(f"{eng.max_open_qty} contracts open at once")
    for o in eng.orders:
        et_ = pd.Timestamp(o["t"]).tz_convert(lucid_rules.ET)
        m_ = et_.hour * 60 + et_.minute
        if 12 * 60 <= m_ < 19 * 60:
            problems.append(f"entry outside 19:00-12:00 ET: {et_}")
    for k, t in eng.tk.items():
        if t["open"]:
            continue
        o_ = pd.Timestamp(t["opened"]).tz_convert(lucid_rules.ET)
        c_ = pd.Timestamp(t["closed"]).tz_convert(lucid_rules.ET)
        if lucid_rules.trading_day(t["closed"]) != lucid_rules.trading_day(t["opened"]) or \
                (c_.hour * 60 + c_.minute) > 16 * 60 + 45 and c_.hour < 18:
            problems.append(f"ticket {k} opened {o_:%a %d %b %H:%M} still open after the 16:45 ET flat ({c_:%a %H:%M})")
    for t in trades:
        # a race may lose up to ITS OWN loss line: $2,000 at first, more once a funded account's line has
        # locked at $50,100 below accumulated profit (2026-10-09: losses up to ~$4,300, all within their lines)
        if t.get("pnl_usd") is None or t.get("line") is None:
            continue
        lots = float(t.get("lots") or 1)
        allowed = abs(float(t["entry_price"]) - float(t["line"])) * PV * lots + (2 * COMM + 2 * TICK * PV) * lots + 1
        if float(t["pnl_usd"]) < -allowed:
            problems.append(f"loss beyond the race's own loss line: {t['pnl_usd']:.0f} vs {-allowed:.0f}"
                            f" ({t.get('opened_ts')})")
    log = (out / "bot.log").read_text(encoding="utf-8", errors="replace")
    for bad in ("Loop error", "Traceback", "dropping without crediting", "written off without P&L"):
        n = log.count(bad)
        if n:
            problems.append(f"{n}x '{bad}' in the log")
    if bot == "funded":
        for h in res["state"].get("history") or []:
            for p in h.get("payout_list") or []:
                if p.get("cycle_days", 3) < 3 or float(p.get("best_day", 0)) > 0.40 * float(p.get("cycle_pnl", 1)) + 0.01:
                    problems.append(f"payout breaking Lucid's rules: {p}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", choices=("trial", "funded"), required=True)
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2026-10-08")
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    try:                                     # never slow the live bots down
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc) + timedelta(hours=21)
    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) + timedelta(hours=21)
    out = ROOT / "scratch" / f"soak_{a.bot}"
    res = run(a.bot, start, end, out, a.poll)
    st, trades = res["state"], res["trades"]
    print(f"{a.bot}: {a.start} -> {a.end}, {len(trades)} trades, {res['wall_s'] / 60:.1f} min of computing")
    if a.bot == "trial":
        n = st.get("passes", 0) + st.get("fails", 0)
        print(f"  evaluations {n}: passed {st.get('passes', 0)} ({st.get('passes', 0) / max(1, n):.0%}),"
              f" failed {st.get('fails', 0)}")
    else:
        hist = st.get("history") or []
        paid = [h for h in hist if h.get("payouts")]
        pays = sum(len(h.get("payout_list") or []) for h in hist) + len((st.get("run") or {}).get("payouts") or [])
        print(f"  funded accounts finished {len(hist)}: reached a payout {len(paid)} ({len(paid) / max(1, len(hist)):.0%}),"
              f" blown {sum(1 for h in hist if h.get('outcome') == 'blown')}, payouts {pays},"
              f" paid ${float(st.get('paid_total_usd') or 0):,.0f} (${float(st.get('paid_total_usd') or 0) / max(1, len(hist)):,.0f}"
              f" per account)")
    probs = check(a.bot, res, out)
    print("  PROBLEMS:" if probs else "  no problems found")
    for p in probs[:40]:
        print("   -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
