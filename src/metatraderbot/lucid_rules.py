"""Lucid Trading account rules the bot must enforce itself.

Confirmed from Lucid's help centre on 2026-09-29:

  * "All positions must be closed by 4:45 PM EST, Monday through Friday."
    Lucid auto-closes anything still open then -- not a failed account, but a
    market exit at a time the strategy did not choose. So the bot stops
    opening a few minutes earlier and flattens itself before Lucid does.
  * "Trading resumes at 6:00 PM EST, Sunday through Thursday."
  * On holidays with altered closing times, positions must be closed before
    the market closes, "regardless of the usual 4:45 PM cutoff".
  * LucidPro 50K: $3,000 target, $2,000 max loss limit (EOD trailing),
    $1,200 daily loss limit (soft: blocks trading for the day, does not fail
    the account), 4 minis or 40 micros maximum.

"EST" in Lucid's wording is New York time, so everything here is computed in
America/New_York and follows daylight saving automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from zoneinfo import ZoneInfo

from metatraderbot import futures

ET = ZoneInfo("America/New_York")
FLAT_BY = time(16, 45)
REOPEN = time(18, 0)
# CME trade dates roll at 17:00 ET; the positions are already flat by then,
# so equity marked after 17:00 is the day's end-of-day balance.
TRADE_DATE_ROLL = time(17, 0)

# Holiday early closes: the time positions must be flat by, in ET. These are
# set EARLIER than the exchange's published halt on purpose -- flattening early
# only costs trading time, flattening late costs a forced exit. Confirm each
# against Lucid's holiday notice and extend the table every year.
EARLY_CLOSES: dict[date, time] = {
    date(2026, 11, 26): time(12, 45),   # Thanksgiving (CME halts ~13:00 ET)
    date(2026, 11, 27): time(12, 45),   # day after Thanksgiving (~13:15 ET)
    date(2026, 12, 24): time(12, 45),   # Christmas Eve (~13:15 ET)
    # 2027 US holiday Mondays/Fridays: CME equity/metals halt ~13:00 ET.
    date(2027, 1, 18): time(12, 45),    # Martin Luther King Jr. Day
    date(2027, 2, 15): time(12, 45),    # Presidents' Day
    date(2027, 5, 31): time(12, 45),    # Memorial Day
    date(2027, 6, 18): time(12, 45),    # Juneteenth (observed)
    date(2027, 7, 5): time(12, 45),     # Independence Day (observed)
    date(2027, 9, 6): time(12, 45),     # Labor Day
}
# Weekdays with no trading session at all.
CLOSED_DAYS: frozenset[date] = frozenset({
    date(2026, 12, 25),                 # Christmas Day
    date(2027, 1, 1),                   # New Year's Day
    date(2027, 3, 26),                  # Good Friday
})
# Abbreviated, thin US-holiday sessions. The New York-session strategies skip
# them: the futures research excluded them (no complete 09:30-16:00 session),
# so trading them would be trading something never tested.
HOLIDAY_SESSIONS: frozenset[date] = frozenset(EARLY_CLOSES) | CLOSED_DAYS


class Session(str, Enum):
    OPEN = "open"                        # entries allowed
    NO_NEW_ENTRIES = "no_new_entries"    # close to the cutoff: manage, don't open
    FLATTEN = "flatten"                  # close everything now
    CLOSED = "closed"                    # between cutoff and reopen, or weekend


@dataclass(frozen=True)
class LucidPlan:
    name: str
    start_balance: float
    profit_target: float
    max_loss_limit: float
    daily_loss_limit: float
    max_minis: int

    @property
    def max_units(self) -> int:
        return self.max_minis * futures.UNITS_PER_MINI


LUCIDPRO_50K = LucidPlan("LucidPro 50K", 50_000.0, 3_000.0, 2_000.0, 1_200.0, 4)


def _to_et(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("pass a timezone-aware datetime (UTC)")
    return now.astimezone(ET)


def flat_by(d: date) -> time:
    return EARLY_CLOSES.get(d, FLAT_BY)


def session_state(now: datetime, *, entry_cutoff_min: int = 15,
                  flatten_lead_min: int = 5) -> Session:
    """Where `now` falls in Lucid's trading week.

    OPEN until `entry_cutoff_min` before the flat-by time, NO_NEW_ENTRIES until
    `flatten_lead_min` before it, then FLATTEN until the flat-by time itself,
    CLOSED from there to the 18:00 ET reopen (Friday: until Sunday 18:00).
    """
    et = _to_et(now)
    d, t, wd = et.date(), et.time(), et.weekday()   # Mon=0 .. Sun=6

    if wd == 5:
        return Session.CLOSED
    if wd == 6:
        return Session.OPEN if t >= REOPEN else Session.CLOSED
    if d in CLOSED_DAYS:
        return Session.CLOSED

    cutoff = flat_by(d)
    cutoff_dt = datetime.combine(d, cutoff)
    flatten_at = (cutoff_dt - timedelta(minutes=flatten_lead_min)).time()
    no_entries_at = (cutoff_dt - timedelta(minutes=entry_cutoff_min)).time()

    if t >= cutoff:
        # Friday's session does not reopen; neither does the eve of a closed day.
        if wd == 4 or (d + timedelta(days=1)) in CLOSED_DAYS:
            return Session.CLOSED
        return Session.OPEN if t >= REOPEN else Session.CLOSED
    if t >= flatten_at:
        return Session.FLATTEN
    if t >= no_entries_at:
        return Session.NO_NEW_ENTRIES
    return Session.OPEN


def trading_day(now: datetime) -> str:
    """CME trade date for `now` (YYYY-MM-DD): the session that opens Sunday
    18:00 ET is Monday's, and each weekday's rolls over at 17:00 ET."""
    et = _to_et(now)
    shifted = et + timedelta(hours=24) - timedelta(hours=TRADE_DATE_ROLL.hour,
                                                   minutes=TRADE_DATE_ROLL.minute)
    d = shifted.date()
    while d.weekday() >= 5:          # a Sunday-evening session trades as Monday
        d += timedelta(days=1)
    return d.isoformat()


def open_units(positions: dict[str, int]) -> int:
    """Contract-cap units in use for {instrument_or_root: contracts}."""
    return sum(futures.units(name) * abs(int(q)) for name, q in positions.items())


def max_addable(name: str, used_units: int, plan: LucidPlan = LUCIDPRO_50K) -> int:
    """How many more contracts of `name` fit under the plan's size cap."""
    free = plan.max_units - used_units
    return max(0, free // futures.units(name))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
