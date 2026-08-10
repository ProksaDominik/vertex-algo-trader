"""Trade post-mortem + bounded learning.

The honest version of "the strategy learns after every win/loss". For each
closed trade this builds a structured review — what happened, and a plain-
language WHY — and stores it as memory. Aggregated across trades, those
reviews drive ONE bounded, data-driven adaptation: a per-slot session
filter that stands a strategy down during the session-of-day where it has
demonstrably and repeatedly lost.

Deliberately NOT here: live mutation of SL/TP/size from recent P&L. That is
curve-fitting the account in real time and is how people blow up. Learning
changes WHERE and WHEN a proven edge is allowed to fire, never the edge's
own parameters.

Everything is pure/deterministic given the inputs, so it unit-tests without
MT5 or a database.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# Session-of-day buckets in UTC. FX liquidity/behaviour differs a lot by
# session, and a breakout edge that works in the London/NY overlap often
# bleeds in the thin Asian session — exactly the kind of pattern worth
# learning per slot.
def classify_session(hour_utc: int) -> str:
    if 0 <= hour_utc < 7:
        return "asian"
    if 7 <= hour_utc < 12:
        return "london"
    if 12 <= hour_utc < 17:
        return "ny_overlap"
    if 17 <= hour_utc < 21:
        return "ny_pm"
    return "late"


@dataclass
class TradeReview:
    config_name: str
    symbol: str
    direction: str
    exit_reason: str
    pnl_usd: float
    r_multiple: float
    entry_hour: int
    session: str
    hold_minutes: float
    was_win: bool
    why: str
    mfe_r: Optional[float] = None   # max favourable excursion, in R
    mae_r: Optional[float] = None   # max adverse excursion, in R


def _parse_ts(ts) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return None


def _r_from_prices(entry, sl, exit_price, direction, pnl_usd, risk_usd) -> float:
    """Realized result in R. Prefer price geometry (|entry-sl| = 1R); fall
    back to pnl/risk_usd when the stop distance isn't usable."""
    try:
        risk_dist = abs(float(entry) - float(sl))
        if risk_dist > 0:
            move = (float(exit_price) - float(entry)) if direction == "long" \
                else (float(entry) - float(exit_price))
            return move / risk_dist
    except (TypeError, ValueError):
        pass
    if risk_usd:
        return float(pnl_usd) / float(risk_usd)
    return 0.0


def review_trade(row: dict, risk_usd: float,
                 mfe_r: Optional[float] = None,
                 mae_r: Optional[float] = None) -> TradeReview:
    """Build a review from a closed_positions row (dict-like) plus the
    slot's risk_usd. `row` needs: config_name, symbol, direction,
    entry_price, exit_price, sl, pnl_usd, exit_reason, opened_ts, closed_ts."""
    direction = str(row.get("direction", "long"))
    pnl = float(row.get("pnl_usd") or 0.0)
    reason = str(row.get("exit_reason", "unknown"))
    r = _r_from_prices(row.get("entry_price"), row.get("sl"),
                       row.get("exit_price"), direction, pnl, risk_usd)

    opened = _parse_ts(row.get("opened_ts"))
    closed = _parse_ts(row.get("closed_ts"))
    entry_hour = opened.astimezone(timezone.utc).hour if opened else 0
    session = classify_session(entry_hour)
    hold_min = ((closed - opened).total_seconds() / 60.0) if (opened and closed) else 0.0
    was_win = pnl > 0

    why = _explain(reason, r, pnl, direction, session, hold_min, mfe_r, mae_r)
    return TradeReview(
        config_name=str(row.get("config_name", "")),
        symbol=str(row.get("symbol", "")),
        direction=direction, exit_reason=reason, pnl_usd=pnl, r_multiple=r,
        entry_hour=entry_hour, session=session, hold_minutes=hold_min,
        was_win=was_win, why=why, mfe_r=mfe_r, mae_r=mae_r,
    )


def _explain(reason, r, pnl, direction, session, hold_min, mfe_r, mae_r) -> str:
    held = f"{hold_min/60:.1f}h" if hold_min >= 60 else f"{hold_min:.0f}m"
    if reason == "tp":
        base = f"Hit take-profit for +{r:.2f}R after a clean {direction} run ({held} in {session})."
    elif reason == "skyrocket":
        base = f"Closed early on an outsized favourable candle (+{r:.2f}R) instead of waiting for TP."
    elif reason == "sl" and pnl > 0:
        base = (f"Trailing stop locked +{r:.2f}R — the {direction} move stalled and gave back "
                f"part of the gain before target ({held}, {session}).")
    elif reason == "sl":
        base = (f"Stopped out for {r:.2f}R — {direction} entry reversed against the position "
                f"in the {session} session ({held}).")
    elif reason == "unreconciled":
        base = "Closed but the broker deal history was unavailable — logged without a fill price."
    else:
        base = f"Closed ({reason}) for {r:.2f}R ({held}, {session})."
    if mfe_r is not None and mae_r is not None:
        base += f" [ran +{mfe_r:.2f}R in favour / -{abs(mae_r):.2f}R against at worst]"
    return base


# --------------------------------------------------------------------- learner

@dataclass
class SessionVerdict:
    session: str
    trades: int
    net_r: float
    avoid: bool


class SessionLearner:
    """Aggregates reviews into a per-(slot, session) verdict. A session is
    flagged 'avoid' for a slot once it has enough trades AND a clearly
    negative net R — the slot then stops taking NEW entries during that
    session (existing positions keep being managed). Rebuilt from the ledger
    on every start, so the memory survives restarts."""

    def __init__(self, min_trades: int = 6, avoid_below_net_r: float = -3.0):
        self.min_trades = min_trades
        self.avoid_below_net_r = avoid_below_net_r
        # {config_name: {session: [count, net_r]}}
        self._agg: dict[str, dict[str, list]] = {}

    def observe(self, config_name: str, session: str, r_multiple: float) -> None:
        slot = self._agg.setdefault(config_name, {})
        cell = slot.setdefault(session, [0, 0.0])
        cell[0] += 1
        cell[1] += r_multiple

    def should_avoid(self, config_name: str, session: str) -> bool:
        cell = self._agg.get(config_name, {}).get(session)
        if not cell:
            return False
        count, net_r = cell
        return count >= self.min_trades and net_r <= self.avoid_below_net_r

    def verdicts(self, config_name: str) -> list[SessionVerdict]:
        out = []
        for session, (count, net_r) in self._agg.get(config_name, {}).items():
            out.append(SessionVerdict(
                session=session, trades=count, net_r=net_r,
                avoid=(count >= self.min_trades and net_r <= self.avoid_below_net_r),
            ))
        return sorted(out, key=lambda v: v.net_r)
