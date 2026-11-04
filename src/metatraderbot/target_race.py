"""Target race -- the LucidPro evaluation strategy (docs/LUCIDPRO_PASS_RESEARCH.md).

Once per Lucid trading day: buy `contracts` of `symbol` -- or sell them short
when it trades below its 200-day average (race_side, since 2026-10-07) --
take profit with a limit order at the price where the account reaches the evaluation target,
keep only a far safety stop, and get out at `exit` ET. The next day repeats
with the new distance to the target.

Lucid support confirmed (2026-09-30) that the Max Loss Limit counts the OPEN
position's equity in real time and the DLL counts realized losses. So each
day is a race between +$3,000 and the -$2,000 limit: a coin flip passes ~40%
of such races, and NQ's upward drift adds a little. Replayed from every start
day 2019-26 on LucidPro 50K: 2 NQ ~41% (real MNQ bars 2025-26: 44%), 1.45
failed evaluations per pass -- the best option found (ORB + trend 35-39%).
(Had only the closing balance counted, 4 NQ would have passed ~83%; it
does not.)

The funded bot (scripts/run_lucid_futures_funded.py) races too, but for a
smaller daily take-profit: only what the next payout still needs, at most
$1,500 -- see its docstring.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from . import lucid_rules

LUCID_DAY_START = 18 * 60          # the Lucid/CME trading day opens 18:00 ET


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def lucid_minute(now_utc: datetime) -> int:
    """Minutes since the current Lucid trading day opened (18:00 ET = 0)."""
    et = now_utc.astimezone(lucid_rules.ET)
    return (et.hour * 60 + et.minute - LUCID_DAY_START) % 1440


def _day_offset(hhmm: str) -> int:
    return (_minutes(hhmm) - LUCID_DAY_START) % 1440


@dataclass(frozen=True)
class RaceConfig:
    symbol: str = "NQ"
    contracts: int = 2            # best under the confirmed rules (15-20 MNQ equivalent)
    entry: str = "19:00"          # ET: first minute of the day an entry may go out
    last_entry: str = "12:00"     # ET, next morning: later starts leave too little time
    exit: str = "16:00"           # ET time exit (Lucid's own flat rule is 16:45)
    stop_points: float = 400.0    # far safety stop -- 82.7% vs 82.9% with none

    def entry_window(self, now_utc: datetime) -> bool:
        """Inside [entry, last_entry) of the current Lucid day. The window
        runs through midnight, so a bot started late in the night -- or at
        08:00 ET (14:00 in Prague) -- still enters that day."""
        m = lucid_minute(now_utc)
        return _day_offset(self.entry) <= m < _day_offset(self.last_entry)

    def exit_due(self, now_utc: datetime) -> bool:
        """From `exit` until the next 18:00 ET open."""
        return lucid_minute(now_utc) >= _day_offset(self.exit)


def skip_day(trade_date: str | date) -> bool:
    """Early-close and closed sessions: too short for the race."""
    d = date.fromisoformat(trade_date) if isinstance(trade_date, str) else trade_date
    return d in lucid_rules.HOLIDAY_SESSIONS


def can_enter(now_utc: datetime, cfg: RaceConfig, done_day: str | None) -> bool:
    """The runner's own gate, minus the account checks: session open, inside
    the window, not a holiday session, and no race yet on this trading day."""
    day = lucid_rules.trading_day(now_utc)
    return (lucid_rules.session_state(now_utc) == lucid_rules.Session.OPEN and cfg.entry_window(now_utc)
            and day != done_day and not skip_day(day))


def next_entry(now_utc: datetime, cfg: RaceConfig, done_day: str | None) -> datetime:
    """When the next race can go in (UTC): `now_utc` if it can right now,
    otherwise the next day's entry time -- weekends and holiday sessions
    skipped. For display; the runner decides with can_enter()."""
    if can_enter(now_utc, cfg, done_day):
        return now_utc
    h, m = (int(x) for x in cfg.entry.split(":"))
    cand = now_utc.astimezone(lucid_rules.ET).replace(hour=h, minute=m, second=0, microsecond=0)
    if cand.astimezone(timezone.utc) <= now_utc:
        cand = cand + timedelta(days=1)
    for _ in range(14):
        c = cand.astimezone(timezone.utc)
        if can_enter(c, cfg, done_day):
            return c
        cand = cand + timedelta(days=1)
    return cand.astimezone(timezone.utc)


def target_points(need_usd: float, contracts: int, point_value: float, tick: float,
                  commission_per_side: float, extra_ticks: int = 1) -> float:
    """Distance from the fill to the take-profit that lifts the CLOSED balance
    to the target after round-trip commission, plus `extra_ticks` of margin.
    Always at least one tick beyond the fill."""
    if contracts < 1 or point_value <= 0 or tick <= 0:
        raise ValueError("contracts, point_value and tick must be positive")
    cost = 2.0 * commission_per_side * contracts
    pts = (max(need_usd, 0.0) + cost) / (contracts * point_value)
    ticks = max(1, math.ceil(pts / tick - 1e-9)) + extra_ticks
    return ticks * tick


SIDE_RULE_DAYS = 200     # owner, 2026-10-07: "we can't work only with trades that predict up"


def trend_side(closes, days: int = SIDE_RULE_DAYS) -> int:
    """The long-term trend rule: +1 (race long) when the last completed daily close is above the
    average of the last `days` closes, -1 (race short) when below. Too little history -> +1.
    Replay 2019-26: the same pass rate as always-long on average (40.9 / 41.6% vs 40.7 / 40.9%),
    but 41% vs 34% races won in the 2022 bear market (docs: notes/Research log 2026-10-07)."""
    c = [float(x) for x in closes if x == x]
    if len(c) < int(days * 0.8):
        return 1
    window = c[-days:]
    return 1 if c[-1] >= sum(window) / len(window) else -1


def race_side(engine, symbol: str, rule: str = "trend200", now=None) -> tuple[int, str]:
    """(+1 long / -1 short, why) for the race about to open. rule 'long' = always long; 'breadth' = the side most
    S&P 500 stocks moved the session before (breadth.py; owner 2026-10-21), the 200-day rule when that count is
    missing. Uses the engine's completed daily bars (the last row is the engine's forming stand-in)."""
    if rule == "breadth":
        return _breadth_side(engine, symbol, now)
    if rule != "trend200":
        return 1, "always long"
    return _trend_race_side(engine, symbol)


def _breadth_side(engine, symbol: str, now=None) -> tuple[int, str]:
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from . import breadth
    data = Path(__file__).resolve().parents[2] / "data"
    et = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    day = str((et.replace(tzinfo=None) + timedelta(hours=6)).date())       # Lucid's trading day (from 18:00 ET)
    try:
        breadth.ensure(data)
    except Exception as e:  # noqa: BLE001 -- no internet: the saved counts, else the 200-day rule
        logging.getLogger(__name__).warning("breadth count not updated: %r", e)
    side, why = breadth.side_for(day, data)
    if side:
        return side, f"breadth {why}"
    t_side, t_why = _trend_race_side(engine, symbol)
    return t_side, f"{t_why} (breadth: {why} -- the 200-day rule instead)"


def _trend_race_side(engine, symbol: str) -> tuple[int, str]:
    try:
        closes = engine.get_bars(symbol, "1d", SIDE_RULE_DAYS + 60)["close"].iloc[:-1]
    except Exception as e:  # noqa: BLE001 -- no daily history: keep the long race
        return 1, f"long (no daily bars: {e})"
    side = trend_side(closes)
    avg = float(closes.tail(SIDE_RULE_DAYS).mean())
    return side, (f"{'long' if side > 0 else 'SHORT'}: last close {float(closes.iloc[-1]):,.2f} "
                  f"{'above' if side > 0 else 'below'} the {SIDE_RULE_DAYS}-day average {avg:,.2f}")


LEFTOVER_RETRY_S = 300          # how often a leftover position is tried again


def close_leftovers(engine, known, leftovers, log) -> list[int]:
    """Bot positions still open in NinjaTrader although no race is managing them -- e.g. a race booked
    from the price history while NinjaTrader had no market connection and refused the close (2026-10-08:
    the laptop slept, the funded race's loss line was booked, the position stayed open). Every open bot
    ticket on the account that is not in `known` (the races being managed) is tried once; returns those
    still open. While any remain, no new race may start -- it would stack on top of the old position."""
    open_now = {int(t) for t in engine.get_all_open_tickets()}
    still = []
    for t in sorted(({int(x) for x in leftovers} | open_now) - {int(k) for k in known}):
        if engine.get_position(t) is None:
            continue
        engine.close_position(t, max_retries=0)
        if engine.get_position(t) is None:
            log.warning(f"leftover position {t} closed -- its race was already booked")
        else:
            still.append(t)
    if still:
        log.warning(f"leftover position(s) {still} still open (NinjaTrader connected?) -- no new race until "
                    f"they are closed; trying again in {LEFTOVER_RETRY_S // 60} min")
    return still


def close_untracked(engine, log, alert_fn=None) -> dict[str, int]:
    """Positions on the account that no bot ticket explains, closed while the bot has no race of its own
    open (call it only then). 2026-10-14: NinjaTrader restarted while the trial race was open; the bridge
    forgot the ticket, the bot judged the race on the price history and booked it, but the 2 NQ stayed in
    the account -- and NinjaTrader refused every later race ("the account already holds a position") for
    a day. The bots' accounts are theirs alone, so anything open there with no race running is a leftover.
    Closed only on a live price (a frozen one would fill at a made-up price). Returns what is still open."""
    held = engine.open_contracts()
    if not held or engine.get_all_open_tickets():      # nothing, or bot tickets: close_leftovers' job
        return {}
    still = {}
    for contract, qty in held.items():
        if not engine.is_tradeable(contract):
            log.warning(f"untracked position {qty:+d} {contract} in the account, no race holds it -- closing it "
                        f"once NinjaTrader has a live price")
            still[contract] = qty
            continue
        engine.flatten(contract)
        left = engine.open_contracts().get(contract, 0)
        if left:
            still[contract] = left
            log.warning(f"untracked position {qty:+d} {contract}: NinjaTrader did not close it ({left:+d} left)")
        else:
            log.warning(f"untracked position {qty:+d} {contract} closed -- no race held it (a position left over "
                        f"from before a NinjaTrader restart)")
            if alert_fn:
                alert_fn(f"🧹 Closed a leftover position {qty:+d} {contract} that no race held -- races can start again.")
    return still


FROZEN_AFTER_S = 300            # a quote this old while the minute bars still trade = a frozen feed
FROZEN_ALERT_S = 2 * 3600       # Telegram at most this often per bot
_frozen_logged: dict[str, float] = {}
_frozen_alerted: dict[str, float] = {}


def price_problem(engine, sym: str, now_utc: datetime | None = None) -> str | None:
    """What is wrong with NinjaTrader's price for `sym` while its minute bars show the market trading (in
    the last 5 minutes) -- None when the price is fine or the market is not trading. Two ways it goes wrong
    (2026-10-14/15, after NinjaTrader's free live data ran out): the quote stops ticking, or it keeps
    repeating an old price that is far from where the market trades (bridge 1.4 subscribes a quiet feed
    again, which replays the last price as if it were new). A race must never start on such a price."""
    q = engine.quote(sym) or {}
    bars = engine.get_bars(sym, "1m", 6)
    traded = bars[bars["volume"] > 0]
    now_utc = now_utc or datetime.now(timezone.utc)
    if not len(traded) or (now_utc - traded.index[-1].to_pydatetime()).total_seconds() >= FROZEN_AFTER_S:
        return None
    close = float(traded["close"].iloc[-1])
    age, last = q.get("age_s"), q.get("last")
    if age is None or not math.isfinite(float(age)) or float(age) >= FROZEN_AFTER_S:
        return f"has not moved for {float(age) / 60:.0f} min" if age is not None and math.isfinite(float(age))             else "is missing"
    if last is None or not math.isfinite(float(last)) or abs(float(last) - close) > max(25.0, 0.001 * close):
        return f"is {abs(float(last or 0) - close):,.0f} points away from the market ({float(last or 0):,.2f} vs {close:,.2f})"
    return None


def note_frozen_price(engine, sym: str, log, alert_fn=None, problem: str | None = None) -> None:
    """When a race cannot start for want of a true live price, say so if the feed (not the market) is at
    fault -- in the log every 10 minutes, on Telegram every 2 hours (2026-10-14: the NQ price stood still
    for 10 hours after a NinjaTrader restart and nothing said why the bots did not race)."""
    import time as _clock
    if problem is None:
        try:
            problem = price_problem(engine, sym)
        except Exception:  # noqa: BLE001 -- bars unavailable: nothing to say
            return
    if problem is None:
        return
    now = _clock.monotonic()
    if now - _frozen_logged.get(sym, -1e18) < 600:
        return
    _frozen_logged[sym] = now
    log.warning(f"[{sym}] NinjaTrader's price {problem} although the market is trading -- no race until it is live "
                f"(NinjaTrader's real-time market data: subscription / connection)")
    if alert_fn and now - _frozen_alerted.get(sym, -1e18) >= FROZEN_ALERT_S:
        _frozen_alerted[sym] = now
        alert_fn(f"🧊 NinjaTrader's {sym} price {problem} although the market trades -- the race cannot start. "
                 f"Check NinjaTrader's real-time market data (subscription / connection), then restart NinjaTrader.")


def exit_time(opened_utc: datetime, cfg: RaceConfig) -> datetime:
    """The race's time exit (UTC): cfg.exit ET on the trading day the position opened."""
    d = date.fromisoformat(lucid_rules.trading_day(opened_utc))
    h, m = (int(x) for x in cfg.exit.split(":"))
    return datetime.combine(d, time(h, m), tzinfo=lucid_rules.ET).astimezone(timezone.utc)


def first_event(bars, opened_utc: datetime, *, line: float | None, tp: float, tick: float,
                exit_utc: datetime, long: bool = True):
    """What Lucid's server saw: replay completed 1-minute bars (index = bar open
    time, UTC) from the minute after the fill and return the first of
      ("line", line, when)    the fail line was touched -- equity at the MLL,
      ("target", tp, when)    price traded one tick through the take-profit,
      ("exit", close, when)   the time exit came first (booked at the last close before it),
    or None if none of them has happened yet. A bar touching both the line and
    the target counts as the line. line=None skips the line (closed-balance
    research reading)."""
    import pandas as pd
    bars = bars[bars.index >= pd.Timestamp(opened_utc).ceil("min")]
    sign = 1.0 if long else -1.0
    exit_ts = pd.Timestamp(exit_utc)
    for ts, b in bars.iterrows():
        if ts >= exit_ts:
            prev = bars[bars.index < exit_ts]
            px = float(prev["close"].iloc[-1]) if len(prev) else float(b["open"])
            return ("exit", px, exit_ts.to_pydatetime())
        adverse, favour = (b["low"], b["high"]) if long else (b["high"], b["low"])
        if line is not None and sign * (adverse - line) <= 0:
            return ("line", line, ts.to_pydatetime())
        if sign * (favour - tp) >= tick:
            return ("target", tp, ts.to_pydatetime())
    return None
