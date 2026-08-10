"""FTMO guard module — hard equity protections, testable in isolation.

FTMO's actual rules (challenge + funded):
- Max daily loss: today's closed P&L + current floating P&L must stay above
  -5% of the initial account size. Resets at midnight Prague time (CE(S)T).
- Max overall loss: equity must never drop below -10% from initial balance.
- Target: +10% (challenge phase 1).

This module triggers on deliberate BUFFERS under those limits (-3.5% daily,
-8% overall by default) so slippage on the flatten can't turn a guard stop
into an actual breach.

Two scopes:
- "book"    — P&L measured from this bot's own ledger + its own magic-number
              floating P&L. The only honest option while FTMO and YOLO share
              one account (AUDIT C4). Approximates FTMO's per-account math.
- "account" — real account equity vs the persisted initial balance anchor.
              Use this on a dedicated FTMO challenge account.

Pure logic: the runner supplies numbers and executes decisions (flatten,
halt); this module never touches MT5 directly, which is what makes it
unit-testable to the CE(S)T boundary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    _PRAGUE = ZoneInfo("Europe/Prague")
except Exception:  # pragma: no cover — tzdata missing on some Windows setups
    _PRAGUE = timezone(timedelta(hours=2))
    log.warning("zoneinfo Europe/Prague unavailable — using fixed UTC+2 (install tzdata!)")


@dataclass
class GuardDecision:
    action: str          # "none" | "block" | "flatten_halt"
    scope: str = ""      # "daily" | "total" | "target" | "news" | "weekend"
    reason: str = ""


@dataclass
class FTMOGuard:
    account_size: float
    state: object                       # BotState
    scope: str = "book"                 # "book" | "account"
    daily_trigger_pct: float = 3.5      # buffer under FTMO's 5%
    overall_trigger_pct: float = 8.0    # buffer under FTMO's 10%
    target_pct: float = 10.0
    target_action: str = "alert"        # "alert" | "halt"
    news_blackout: list = field(default_factory=list)   # [{"weekday":4,"start":"12:00","end":"15:00"}] UTC
    weekend_flat: bool = True
    weekend_close_utc: str = "20:30"    # Friday flatten time (before 21:00/22:00 broker close)

    # ------------------------------------------------------------------ time

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def prague_day(now: datetime | None = None) -> str:
        return (now or FTMOGuard.now_utc()).astimezone(_PRAGUE).date().isoformat()

    # ----------------------------------------------------------- daily anchor

    def roll_daily_anchor(self, realized_alltime: float, equity_now: float | None,
                          now: datetime | None = None) -> bool:
        """Snapshot the daily anchor at the first check of each Prague day.
        Returns True when a NEW day started (callers clear daily halts)."""
        day = self.prague_day(now)
        daily = self.state.get("daily") or {}
        if daily.get("day") == day:
            return False
        self.state.set("daily", {
            "day": day,
            "realized_at_start": realized_alltime,
            "equity_at_start": equity_now,
        })
        halt = self.state.get("halt") or {}
        if halt.get("active") and halt.get("scope") == "daily":
            log.warning("New CE(S)T day — clearing yesterday's daily halt")
            self.state.clear_halt()
        self.state.save()
        return True

    def daily_pnl(self, realized_alltime: float, floating: float,
                  equity_now: float | None = None) -> float:
        daily = self.state.get("daily") or {}
        if self.scope == "account" and equity_now is not None and daily.get("equity_at_start") is not None:
            return equity_now - float(daily["equity_at_start"])
        return (realized_alltime - float(daily.get("realized_at_start", 0.0))) + floating

    def total_pnl(self, realized_alltime: float, floating: float,
                  equity_now: float | None = None) -> float:
        if self.scope == "account" and equity_now is not None:
            anchor = self.state.get("initial_balance_anchor")
            if anchor:
                return equity_now - float(anchor)
        return realized_alltime + floating

    # ---------------------------------------------------------------- windows

    @staticmethod
    def _hhmm(s: str) -> tuple[int, int]:
        h, m = s.split(":")
        return int(h), int(m)

    def in_news_blackout(self, now: datetime | None = None) -> bool:
        now = now or self.now_utc()
        for w in self.news_blackout or []:
            try:
                if int(w["weekday"]) != now.weekday():
                    continue
                sh, sm = self._hhmm(w["start"])
                eh, em = self._hhmm(w["end"])
                start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
                end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
                if start <= now < end:
                    return True
            except (KeyError, ValueError, TypeError):
                log.warning(f"Malformed news_blackout window ignored: {w}")
        return False

    def in_weekend_flat(self, now: datetime | None = None) -> bool:
        """True from Friday weekend_close_utc until Sunday 22:00 UTC."""
        if not self.weekend_flat:
            return False
        now = now or self.now_utc()
        wd = now.weekday()
        if wd == 5:
            return True
        if wd == 4:
            h, m = self._hhmm(self.weekend_close_utc)
            return now >= now.replace(hour=h, minute=m, second=0, microsecond=0)
        if wd == 6:
            return now < now.replace(hour=22, minute=0, second=0, microsecond=0)
        return False

    # ------------------------------------------------------------------ check

    def check(self, *, realized_alltime: float, floating: float,
              equity_now: float | None = None,
              now: datetime | None = None) -> GuardDecision:
        """Main guard evaluation — call every loop tick."""
        self.roll_daily_anchor(realized_alltime, equity_now, now)

        halt = self.state.get("halt") or {}
        if halt.get("active"):
            return GuardDecision("block", halt.get("scope", ""), halt.get("reason", "halted"))

        total = self.total_pnl(realized_alltime, floating, equity_now)
        if total <= -self.overall_trigger_pct / 100.0 * self.account_size:
            return GuardDecision(
                "flatten_halt", "total",
                f"total P&L {total:+.2f} <= -{self.overall_trigger_pct}% of {self.account_size:.0f} "
                f"(buffer under FTMO's -10%)",
            )

        day = self.daily_pnl(realized_alltime, floating, equity_now)
        if day <= -self.daily_trigger_pct / 100.0 * self.account_size:
            return GuardDecision(
                "flatten_halt", "daily",
                f"daily P&L {day:+.2f} <= -{self.daily_trigger_pct}% of {self.account_size:.0f} "
                f"(buffer under FTMO's -5%)",
            )

        if total >= self.target_pct / 100.0 * self.account_size:
            if self.target_action == "halt":
                return GuardDecision(
                    "flatten_halt", "target",
                    f"target reached: {total:+.2f} >= +{self.target_pct}% — halting to protect the pass",
                )
            if not self.state.get("target_alerted"):
                self.state.set("target_alerted", True)
                self.state.save()
                return GuardDecision("none", "target", f"target reached: {total:+.2f}")

        if self.in_weekend_flat(now):
            return GuardDecision("block", "weekend", "weekend-flat window")
        if self.in_news_blackout(now):
            return GuardDecision("block", "news", "news blackout window")

        return GuardDecision("none")
