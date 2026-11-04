"""Opening-range breakout with a daily-trend filter, New York session.

The one intraday idea that survived the Lucid futures research
(scripts/futures_hypotheses.py, futures_robustness.py, futures_stack_mc.py):
its whole parameter neighbourhood was positive on NQ and GC over 2016-2026
(24/24 cells each), and on real futures bars its signals matched the 10-year
proxy's direction on every shared day. The settings here are the CENTRE of
that plateau, not its best cell.

Rules (all times New York; bars are 15m, timestamped at their open, UTC):
- Opening range = high/low of the first `or_minutes` of the 09:30 session.
- Trend = sign(yesterday's 16:00 close - 20-day average of 16:00 closes).
- After the range, on each completed bar that opens before `last_entry`:
  a CLOSE beyond the range is a breakout. If the day's first breakout is
  with the trend, trade it; if it is against the trend, no trade today.
- Stop: the far side of the range. Target: `tp_r_multiple` x risk.
- One trade per day. Time exit at `exit_time` (runner closes the position).

Needs ~20 sessions of bars for the trend (REQUIRED_BARS); runners fetch that.
"""
from __future__ import annotations

from datetime import time as dtime

import pandas as pd

from metatraderbot import lucid_rules
from metatraderbot.instruments import contract_mult
from metatraderbot.strategy import Direction, TradeSpec, bar_times

ET = "America/New_York"
SESSION_OPEN = dtime(9, 30)
SESSION_LAST_BAR = dtime(15, 45)          # the bar that closes at 16:00


def _hm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


class ORBTrendStrategy:
    REQUIRED_BARS = 2600                   # ~27 sessions of 15m bars (23h/day)

    def __init__(self, config: dict):
        s = config["strategy"]
        p = (s.get("orb_trend") or {})
        self.or_minutes = int(p.get("or_minutes", 45))
        self.last_entry = _hm(p.get("last_entry", "12:00"))
        self.exit_time = _hm(p.get("exit_time", "15:55"))
        self.trend_days = int(p.get("trend_days", 20))
        self.tp_r_multiple = float(s["exit"]["tp_r_multiple"])
        self.risk_usd = float(config["capital"]["risk_usd_per_trade"])
        self.contract_mult = contract_mult(config)
        self._done_day = None              # a day is "done" once its first breakout was seen

    def _trend(self, et_times: pd.DatetimeIndex, bars: pd.DataFrame, today) -> int:
        closes = bars["close"][(et_times.time == SESSION_LAST_BAR) & (et_times.date < today)]
        if len(closes) < self.trend_days:
            return 0
        # Same as the research: yesterday's close vs the 20-day average of
        # closes up to and INCLUDING yesterday.
        prev_close = float(closes.iloc[-1])
        sma = float(closes.iloc[-self.trend_days:].mean())
        return 1 if prev_close > sma else (-1 if prev_close < sma else 0)

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        recent = bars.iloc[: bar_index + 1]
        et = bar_times(recent).tz_convert(ET)
        now = et[-1]
        today = now.date()
        if self._done_day == today or today in lucid_rules.HOLIDAY_SESSIONS:
            return []
        t = now.time()
        or_end = (pd.Timestamp.combine(today, SESSION_OPEN) + pd.Timedelta(minutes=self.or_minutes)).time()
        if t < or_end or t >= self.last_entry:
            return []
        in_or = (et.date == today) & (et.time >= SESSION_OPEN) & (et.time < or_end)
        n_or = int(in_or.sum())
        if n_or * 15 < self.or_minutes:
            return []                      # the range is incomplete (missing bars): never guess
        orh = float(recent["high"][in_or].max())
        orl = float(recent["low"][in_or].min())
        close = float(recent["close"].iloc[-1])
        side = 1 if close > orh else (-1 if close < orl else 0)
        if side == 0:
            return []
        self._done_day = today             # first breakout decides the day
        trend = self._trend(et, recent, today)
        if side != trend:
            return []
        sl = orl if side > 0 else orh
        dist = abs(close - sl)
        if dist <= 0:
            return []
        tp = close + side * dist * self.tp_r_multiple
        return [TradeSpec(direction=Direction.LONG if side > 0 else Direction.SHORT,
                          entry_price=close, stop_loss=sl, take_profit=tp,
                          lots=self.risk_usd / (dist * self.contract_mult),
                          signal_bar_index=bar_index)]

    def exit_due(self, now_utc) -> bool:
        """True once the session's time exit has passed (runners close then)."""
        et = pd.Timestamp(now_utc).tz_convert(ET)
        return et.time() >= self.exit_time
