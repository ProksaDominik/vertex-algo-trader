"""A fleet of LucidPro 50K account slots, simulated from real NQ minute prices (owner, 2026-10-08:
"lets do more account as simulation at once").

Each slot is what one account slot would run for real money: an evaluation race (2 NQ, +$3,000 target
vs the $2,000 loss line); once passed, a funded account racing for payouts (1 NQ, take = what the next
payout still needs, $150-$1,500); a reset fee after a failed evaluation and a new evaluation after a
blown funded account -- until the money made falls to -$10,000 (the owner's stop rule, 2026-10-21). One race per slot per trading day from 19:00 ET -- the real-money rule (restarts
win less) -- out at 16:00 ET, under the rules the bots use: the loss line on open equity, trailing the
best end-of-day balance and locked at +$100; 1-minute bars; the line first when a bar touches both; the
target only once price trades one tick through it; the side of the 200-day trend. No orders: every
outcome comes from the price history, the way the bots judge a race the laptop slept through.

All slots race the same NQ at the same moment, so they win and lose together -- the fleet shows the
money of N slots, not N independent tries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import futures, target_race

ET = "America/New_York"
SYM = "NQ"
PV = futures.point_value(SYM)
TICK = float(futures.spec(SYM)["tick_size"])
COMM = futures.commission_per_side(SYM)
LOCK = 100.0                                          # the loss line stops at +$100 over the start
TAKE_MIN = 150.0
SPLIT = 0.90


@dataclass(frozen=True)
class Plan:
    """A LucidPro account size (amounts relative to its start balance)."""
    name: str
    target: float            # evaluation profit target
    mll: float               # max loss limit: end-of-day trailing, open trades count, locks at +$100
    keep: float              # profit that must stay in a funded account (the MLL + $100)
    want: tuple              # a payout is taken once this much is above the keep: the first, then later ones
    caps: tuple              # the most each payout may be
    min_cycle: float         # least profit in a payout cycle
    eval_contracts: int      # NQ in the evaluation race
    funded_contracts: int    # NQ in the funded race
    size: float = 50_000.0   # the start balance
    day_max: float = 0.0     # the funded race's biggest take; 0 = 40% of what the first payout needs

    @property
    def take_max(self) -> float:
        """The most a funded race aims for. The 40% rule is not a daily cap -- a bigger day makes the payout
        wait for more profit (and pay more); 50K: $1,500 since 2026-10-21 (was 40% of $2,600 = $1,040)."""
        return self.day_max or 0.40 * (self.keep + self.want[0])


PLANS = {
    "50K": Plan("LucidPro 50K", 3_000.0, 2_000.0, 2_100.0, (500.0,) + (2_500.0,) * 4,
                (2_000.0,) + (2_500.0,) * 4, 500.0, 2, 1, day_max=1_500.0),
    # 100K (owner asked 2026-10-08; rules from public summaries of Lucid's help pages): $6,000 target,
    # $3,000 loss limit, $103,100 kept, payouts $2,500 first then $3,100, $750 least per cycle, 6 minis
    "100K": Plan("LucidPro 100K", 6_000.0, 3_000.0, 3_100.0, (750.0,) + (3_100.0,) * 4,
                 (2_500.0,) + (3_100.0,) * 4, 750.0, 3, 2, 100_000.0),
}
ENTRY_MIN, EXIT_MIN = 19 * 60, 16 * 60                # ET minutes of the day


@dataclass
class FleetConfig:
    slots: int = 5
    start: str = "2026-10-09"       # first trading day raced (its race starts 19:00 ET the evening before)
    stagger: int = 1                # slot k buys its first evaluation k trading days after slot 1
    eval_fee: float = 115.0         # the owner's code price for LucidPro 50K (with the daily loss limit; list $185)
    reset_fee: float = 120.0        # list price -- the code may or may not lower it
    plan: str = "50K"
    eval_contracts: int | None = None    # None: the plan's own race sizes
    funded_contracts: int | None = None
    start_slots: int | None = None  # step-up plan (owner, 2026-10-08: "start with 1-2 and then step up"):
    step_usd: float = 2_500.0       # begin with start_slots; add the next slot each time the money to you so far
                                    # reaches another step_usd (paid for from profit); None = all slots by stagger
    step_on: str = "money"          # "money": the step_usd rule; "payout": one more slot after every payout
                                    # (owner, 2026-10-09: "if one goes to profit then I buy another one");
                                    # "payout_in_profit": after a payout, but only while the money made is > 0
    bot_races: dict | None = None   # bot_races(): race when and the way the paper bots really did (an evaluation
                                    # like the trial bot, a funded race like the funded bot; no race on a day the
                                    # bot didn't) instead of 19:00 ET on the trend side -- the Real test
    sides: dict | None = None       # {trading day: +1/-1}: the side of the 19:00 replay (default: the 200-day rule)
    replay_before: str | None = None  # with bot_races: days before this one with no bot race are replayed at 19:00
                                      # (the new strategy's Real test before its bots started, 2026-10-21)
    stop_usd: float | None = 10_000.0  # owner 2026-10-21: once the money made is down this much, buy no more
                                       # evaluations or resets (running accounts finish) -- then look again


@dataclass
class Slot:
    n: int
    stage: str = "waiting"          # waiting / eval / funded / stopped (the stop rule)
    bal: float = 0.0                # the current account's balance vs $50,000, payouts taken off
    peak: float = 0.0               # best end-of-day balance
    cycle_days: int = 0
    cycle_pnl: float = 0.0
    cycle_best: float = 0.0
    n_paid: int = 0                 # payouts from the current funded account
    evals: int = 0                  # evaluations bought
    resets: int = 0
    passes: int = 0
    blown: int = 0
    payouts: int = 0
    paid: float = 0.0               # gross payouts (you get 90%)
    fees: float = 0.0
    open: dict = field(default_factory=dict)      # today's race while it runs

    def floor(self, plan: Plan) -> float:
        return min(self.peak - plan.mll, LOCK)

    def need(self, plan: Plan) -> float:
        """Profit the next payout still needs: the money above the keep, the 40% rule, the cycle minimum."""
        k = min(self.n_paid, len(plan.want) - 1)
        return max(plan.keep + plan.want[k] - self.bal,
                   self.cycle_best / 0.40 - self.cycle_pnl if self.cycle_best > 0 else 0.0,
                   plan.min_cycle - self.cycle_pnl)

    def net(self) -> float:
        return SPLIT * self.paid - self.fees


def trading_days(bars: pd.DataFrame) -> np.ndarray:
    """Lucid's trading day of every bar: the session from 18:00 ET belongs to the next date."""
    local = (bars.index.tz_convert(ET).tz_localize(None) + pd.Timedelta(hours=6)).to_numpy()
    return np.datetime_as_string(local.astype("datetime64[D]"), unit="D")


def trading_day_of(ts) -> str:
    """Lucid's trading day of a moment (from 18:00 ET it is the next date)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return str((t.tz_convert(ET).tz_localize(None) + pd.Timedelta(hours=6)).date())


def bot_races(data_dir: Path, since: str | None = None, trial_tag: str = "", funded_tag: str = "") -> dict[str, dict]:
    """The races the paper bots really ran, per trading day: {day: {"eval": (UTC time, side), "funded": (...)}} --
    the trial bot's first race of the day (an evaluation) and the funded bot's, closed or still open. Owner,
    2026-10-19: "we passed trial and we have profit on trial but it doesn't show on the fleet page" -- the bots
    had raced at 05:23 New York (the laptop was off at 19:00), the Real test at 19:00 from another price."""
    import json
    found: list[tuple[str, dict]] = []
    for stage, trades, state, key in (("eval", f"lucid_futures_trades{trial_tag}.jsonl",
                                       f"state_lucid_futures{trial_tag}.json", "attempt"),
                                      ("funded", f"lucid_futures_funded_trades{funded_tag}.jsonl",
                                       f"state_lucid_futures_funded{funded_tag}.json", "run")):
        p = Path(data_dir) / trades
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                try:
                    found.append((stage, json.loads(ln)))
                except ValueError:
                    pass
        try:                                                 # the race open right now (not in the ledger yet)
            tickets = (json.loads((Path(data_dir) / state).read_text(encoding="utf-8")).get(key) or {}).get("tickets")
            found += [(stage, t) for t in (tickets or {}).values() if isinstance(t, dict)]
        except (OSError, ValueError, AttributeError):
            pass
    out: dict[str, dict] = {}
    for stage, t in found:
        try:
            ts = pd.Timestamp(t["opened_ts"])
        except (KeyError, ValueError, TypeError):
            continue
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        day = trading_day_of(ts)
        if since and day < since:
            continue
        side = -1 if str(t.get("direction", "long")).lower() in ("short", "sell") else 1
        cur = out.setdefault(day, {}).get(stage)
        if cur is None or ts < cur[0]:
            out[day][stage] = (ts, side)
    return out


def trend_sides(bars: pd.DataFrame, days: np.ndarray | None = None) -> dict[str, int]:
    """Each trading day's side by the 200-day rule: +1 when the previous session closed above the
    average of the 200 sessions before it, -1 below (long while the history is too short)."""
    days = trading_days(bars) if days is None else days
    closes = pd.Series(bars["close"].to_numpy(), index=days).groupby(level=0).last()
    sma = closes.rolling(target_race.SIDE_RULE_DAYS).mean()
    side = np.where(closes >= sma, 1, -1)
    side[sma.isna().to_numpy()] = 1
    return dict(zip(closes.index[1:], side[:-1].tolist()))


def _race(seg_o, seg_h, seg_l, seg_c, side: int, k: int, need: float, room: float, complete: bool):
    """One race on a session's bars from 19:00 ET to 16:00 ET: ("line"|"target"|"exit"|"open", P&L,
    entry, take-profit, fail line, bar index of the end)."""
    entry = float(seg_o[0]) + side * TICK                     # the spread paid on entry
    tp = entry + side * target_race.target_points(need, k, PV, TICK, COMM)
    line = entry - side * (room - 2 * COMM * k) / (k * PV)
    adv, fav = (seg_l, seg_h) if side > 0 else (seg_h, seg_l)
    hit_l = side * (adv - line) <= 0
    hit_t = side * (fav - tp) >= TICK
    il = int(np.argmax(hit_l)) if hit_l.any() else len(seg_o)
    it = int(np.argmax(hit_t)) if hit_t.any() else len(seg_o)
    if il < len(seg_o) and il <= it:
        what, px, end = "line", line - side * TICK, il
    elif it < len(seg_o):
        what, px, end = "target", tp, it
    elif complete:
        what, px, end = "exit", float(seg_c[-1]) - side * TICK, len(seg_o) - 1
    else:
        what, px, end = "open", float(seg_c[-1]), len(seg_o) - 1
    pnl = side * (px - entry) * k * PV - (2 * COMM * k if what != "open" else 0.0)
    return what, pnl, entry, tp, line, end, px


def run(bars: pd.DataFrame, cfg: FleetConfig, now_utc: datetime | None = None) -> dict:
    """Replay the fleet from cfg.start over 1-minute bars (UTC index, open/high/low/close)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    P = PLANS[cfg.plan]
    k_eval, k_funded = cfg.eval_contracts or P.eval_contracts, cfg.funded_contracts or P.funded_contracts
    days = trading_days(bars)
    sides = cfg.sides if cfg.sides is not None else trend_sides(bars, days)
    et = bars.index.tz_convert(ET)
    mins = np.asarray(et.hour * 60 + et.minute)
    o, h, l, c = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    slots = [Slot(i + 1) for i in range(cfg.slots)]
    events, daily = [], []
    stopped = False

    def stop_now(day: str) -> bool:
        """The owner's stop rule (2026-10-21), asked before every evaluation or reset is bought: once the money made
        is down cfg.stop_usd, nothing more is bought -- for good; the accounts already running finish."""
        nonlocal stopped
        made_ = sum(x.net() for x in slots)
        if not stopped and cfg.stop_usd is not None and made_ <= -cfg.stop_usd:
            stopped = True
            events.append({"day": day, "slot": 0, "what": "STOP RULE -- no more evaluations bought", "usd": 0.0,
                           "why": f"The money made is -${-made_:,.0f}, down ${cfg.stop_usd:,.0f} or more: by your"
                                  f" stop rule no more evaluations or resets are bought. The accounts already"
                                  f" running finish; then look again before buying more."})
        return stopped
    cut = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    sessions = [(days[a], a, b) for a, b in zip(cut[:-1], cut[1:]) if days[a] >= cfg.start]
    for di, (day, a, b) in enumerate(sessions):
        day = str(day)
        if target_race.skip_day(day):
            continue
        mm = mins[a:b]
        go = np.flatnonzero((mm >= ENTRY_MIN) | (mm < EXIT_MIN))
        go = go[np.argmax(mm[go] >= ENTRY_MIN):] if (mm[go] >= ENTRY_MIN).any() else go[:0]
        if not len(go):
            continue
        i0 = a + go[0]
        i1 = a + (go[np.flatnonzero(mm[go] < EXIT_MIN)[-1]] if (mm[go] < EXIT_MIN).any() else go[-1]) + 1
        complete = b > i1 or di < len(sessions) - 1       # bars after 16:00 ET, or a later session exists
        side = sides.get(day, 1)
        for s in slots:
            if s.stage != "waiting":
                continue
            first = cfg.start_slots is None or s.n <= cfg.start_slots
            made = sum(x.net() for x in slots)
            if first and di >= (s.n - 1) * cfg.stagger:
                what_, why = "evaluation bought", f"Account {s.n} starts: a {P.name} evaluation for ${cfg.eval_fee:,.0f}."
            elif (not first and all(x.stage != "waiting" for x in slots if x.n < s.n)
                  and cfg.step_on in ("payout", "payout_in_profit")
                  and sum(x.payouts for x in slots) >= s.n - cfg.start_slots
                  and (cfg.step_on == "payout" or made > 0)):
                what_ = "evaluation bought -- slot added after a payout"
                why = (f"{sum(x.payouts for x in slots)} payout(s) so far, so account {s.n} is bought"
                       f" (${cfg.eval_fee:,.0f}).")
            elif (not first and cfg.step_on == "money" and all(x.stage != "waiting" for x in slots if x.n < s.n)
                  and made >= cfg.step_usd * (s.n - cfg.start_slots)):
                what_ = "evaluation bought -- slot added from profit"
                why = (f"Money made so far ${made:,.0f} reached ${cfg.step_usd * (s.n - cfg.start_slots):,.0f}, so"
                       f" account {s.n} is paid for from profit (${cfg.eval_fee:,.0f}).")
            else:
                continue
            if stop_now(day):
                continue
            s.stage, s.evals, s.fees = "eval", s.evals + 1, s.fees + cfg.eval_fee
            events.append({"day": day, "slot": s.n, "what": what_, "usd": -cfg.eval_fee, "why": why})
        racing = False
        for s in slots:
            s.open = {}
            if s.stage in ("waiting", "stopped"):
                continue
            si0, sside, started = i0, side, "19:00 New York"
            why_side = (f"more stocks {'rose' if side > 0 else 'fell'} the session before" if cfg.sides is not None
                        else f"NQ {'above' if side > 0 else 'below'} its 200-day average")
            rec = (cfg.bot_races.get(day) or {}).get(s.stage) if cfg.bot_races is not None else None
            if cfg.bot_races is not None and rec is None and not (cfg.replay_before and day < cfg.replay_before):
                continue                                   # that bot did not race this day: neither does the account
            if rec is not None:                            # when and the way the bot for this stage raced
                j = int(np.searchsorted(bars.index[a:b], rec[0], side="right")) - 1
                si0, sside = a + max(j, 0), rec[1]
                if si0 >= i1:
                    continue
                started = f"{bars.index[si0].tz_convert(ET):%a %H:%M} New York"
                why_side = "when and the way the bot raced"
            day_pnl = 0.0
            if s.stage == "eval":
                k, need = k_eval, P.target - s.bal
            else:
                k, need = k_funded, s.need(P)
            if s.stage == "eval" or need > 0:
                take = need if s.stage == "eval" else min(max(need, TAKE_MIN), P.take_max)
                what, pnl, entry, tp, line, end, px = _race(o[si0:i1], h[si0:i1], l[si0:i1], c[si0:i1], sside, k,
                                                            take, s.bal - s.floor(P), complete)
                if what == "open":
                    s.open = {"side": sside, "entry": entry, "tp": tp, "line": line, "pnl": pnl, "stage": s.stage}
                    racing = True
                    continue
                when = bars.index[si0 + end]
                at = f"{when.tz_convert(ET):%a %H:%M} New York"
                story = (f"{'Bought' if sside > 0 else 'Sold short'} {k} NQ at {entry:,.2f} at {started}"
                         f" ({why_side}), aiming for {tp:,.2f}"
                         f" ({abs(tp - entry):,.2f} points, +${take:,.0f}); loss line {line:,.2f}"
                         f" ({abs(entry - line):,.2f} points away). ")
                if what == "line":
                    story += (f"Price {'fell' if side > 0 else 'rose'} to the loss line at {at}"
                              f" ({'-' if pnl < 0 else '+'}${abs(pnl):,.0f}).")
                elif what == "target":
                    story += f"Price reached the take-profit at {at} (+${pnl:,.0f})."
                else:
                    story += (f"Neither level was reached; closed at {px:,.2f} at 16:00 New York"
                              f" ({'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}).")
                base = {"day": day, "slot": s.n, "race": pnl, "ts": when.isoformat()}
                day_pnl = pnl
                s.bal += pnl
                if what == "line":
                    if s.stage == "eval" and stop_now(day):
                        s.stage = "stopped"
                        events.append(base | {"what": "evaluation failed -- no reset (stop rule)", "usd": 0.0,
                                              "why": story + " The evaluation is lost; by your stop rule no reset"
                                                             " is bought."})
                    elif s.stage == "eval":
                        s.resets, s.fees = s.resets + 1, s.fees + cfg.reset_fee
                        events.append(base | {"what": "evaluation failed -- reset", "usd": -cfg.reset_fee,
                                              "why": story + f" The evaluation is lost; a reset costs"
                                                             f" ${cfg.reset_fee:,.0f} and it races again next session."})
                    elif stop_now(day):
                        s.blown += 1
                        s.stage = "stopped"
                        events.append(base | {"what": "funded account blown -- no new evaluation (stop rule)",
                                              "usd": 0.0,
                                              "why": story + f" The funded account is closed after {s.n_paid}"
                                                             f" payout(s); by your stop rule no new evaluation is"
                                                             f" bought."})
                    else:
                        s.blown += 1
                        s.evals, s.fees = s.evals + 1, s.fees + cfg.eval_fee
                        events.append(base | {"what": "funded account blown -- new evaluation", "usd": -cfg.eval_fee,
                                              "why": story + f" The funded account is closed after {s.n_paid}"
                                                             f" payout(s); a new evaluation is bought"
                                                             f" (${cfg.eval_fee:,.0f})."})
                        s.stage = "eval"
                    s.bal = s.peak = 0.0
                    continue
                if s.stage == "eval" and s.bal >= P.target - 1e-6:
                    s.passes += 1
                    events.append(base | {"what": "evaluation PASSED -- funded account", "usd": 0.0,
                                          "why": story + f" The +${P.target:,.0f} target is reached; the funded"
                                                         " account starts with the next race."})
                    s.stage, s.bal, s.peak = "funded", 0.0, 0.0
                    s.cycle_days, s.cycle_pnl, s.cycle_best, s.n_paid = 0, 0.0, 0.0, 0
                    continue
                if s.stage == "eval":
                    events.append(base | {"what": "evaluation race closed at 16:00", "usd": 0.0,
                                          "why": story + f" The evaluation goes on: balance ${P.size + s.bal:,.0f},"
                                                         f" ${P.target - s.bal:,.0f} still to make."})
                else:
                    events.append(base | {"what": "funded race won" if what == "target" else
                                          "funded race closed at 16:00", "usd": 0.0,
                                          "why": story + f" Funded balance ${P.size + s.bal:,.0f}."})
            elif s.stage == "funded":
                events.append({"day": day, "slot": s.n, "what": "no race -- payout money reached", "usd": 0.0,
                               "why": f"The next payout's money is there (balance ${P.size + s.bal:,.0f}). Lucid"
                                      f" needs 3 trading days in the cycle, so the account waits without trading"
                                      f" (this is day {s.cycle_days + 1})."})
            # the day's close: the loss line follows, a funded day joins its payout cycle
            s.peak = max(s.peak, s.bal)
            if s.stage == "funded":
                s.cycle_days += 1
                s.cycle_pnl += day_pnl
                s.cycle_best = max(s.cycle_best, day_pnl)
                kk = min(s.n_paid, len(P.want) - 1)
                avail = s.bal - P.keep
                if (s.cycle_days >= 3 and s.cycle_pnl >= P.min_cycle and s.cycle_best <= 0.40 * s.cycle_pnl
                        and avail >= P.want[kk]):
                    amt = min(P.caps[kk], avail)
                    why = (f"{s.cycle_days} trading days, ${s.cycle_pnl:,.0f} profit in the cycle, best day"
                           f" ${s.cycle_best:,.0f} ({s.cycle_best / s.cycle_pnl:.0%}, at most 40%). Everything above"
                           f" ${P.size + P.keep:,.0f} up to ${P.caps[kk]:,.0f} is paid: ${amt:,.0f},"
                           f" ${SPLIT * amt:,.0f} to you.")
                    s.bal -= amt
                    s.paid += amt
                    s.payouts += 1
                    s.n_paid += 1
                    s.cycle_days, s.cycle_pnl, s.cycle_best = 0, 0.0, 0.0
                    events.append({"day": day, "slot": s.n, "what": f"PAYOUT #{s.n_paid}", "usd": SPLIT * amt,
                                   "why": why})
                    if s.n_paid >= len(P.want) and stop_now(day):
                        events.append({"day": day, "slot": s.n, "what": "5 payouts -- Lucid's live review; no new "
                                       "evaluation (stop rule)", "usd": 0.0,
                                       "why": "Five payouts is the most a funded Pro account takes; by your stop"
                                              " rule no new evaluation is bought."})
                        s.stage, s.bal, s.peak = "stopped", 0.0, 0.0
                    elif s.n_paid >= len(P.want):
                        events.append({"day": day, "slot": s.n, "what": "5 payouts -- Lucid's live review; "
                                       "new evaluation", "usd": -cfg.eval_fee,
                                       "why": "Five payouts is the most a funded Pro account takes; Lucid reviews it"
                                              " for a live account and the slot starts a new evaluation."})
                        s.stage, s.bal, s.peak = "eval", 0.0, 0.0
                        s.evals, s.fees = s.evals + 1, s.fees + cfg.eval_fee
        daily.append({"day": day, "fees": sum(s.fees for s in slots), "paid": sum(s.paid for s in slots),
                      "net": sum(s.net() for s in slots), "racing": racing,
                      "eval": sum(s.stage == "eval" for s in slots), "funded": sum(s.stage == "funded" for s in slots)})
        if racing:
            break
    return {"slots": [asdict(s) | {"net": s.net(), "floor": s.floor(P)} for s in slots], "events": events,
            "daily": pd.DataFrame(daily, columns=["day", "fees", "paid", "net", "racing", "eval", "funded"]),
            "config": asdict(cfg)}


# --- minute bars: the stored history + what NinjaTrader has since ---------------------------------

def load_minutes(data_dir: Path, engine=None, since: str | None = None) -> pd.DataFrame:
    """NQ 1-minute bars (UTC): the stored NinjaTrader history, the recent-bars file, and -- with an
    engine -- the bars NinjaTrader has after them (saved to the recent file). The forming bar is left
    out. `since` (YYYY-MM-DD) trims older bars, keeping 300 trading days before it for the trend rule."""
    long_p, recent_p = data_dir / "NQ_1m_long_nt.parquet", data_dir / "NQ_1m_recent_nt.parquet"

    def utc(d: pd.DataFrame) -> pd.DataFrame:
        d.index = pd.DatetimeIndex(d.index)
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        return d[["open", "high", "low", "close", "volume"]]

    long_df = utc(pd.read_parquet(long_p)) if long_p.exists() else None
    parts = [x for x in (long_df, utc(pd.read_parquet(recent_p)) if recent_p.exists() else None) if x is not None]
    df = pd.concat(parts) if parts else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if engine is not None:
        gap = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 60 if len(df) else 60_000
        if gap > 2:
            fresh = utc(engine.get_bars(SYM, "1m", int(min(gap + 10, 60_000))).iloc[:-1])
            df = pd.concat([df, fresh])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            (df[df.index > long_df.index[-1]] if long_df is not None and len(long_df) else df).to_parquet(recent_p)
    if since:
        df = df[df.index >= pd.Timestamp(since, tz="UTC") - pd.Timedelta(days=430)]
    return df
