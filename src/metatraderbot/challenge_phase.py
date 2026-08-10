"""Where am I in the FTMO evaluation, and have I actually passed?

The existing FTMO guard fires once profit crosses the target. That is NOT the
same as passing, and treating it as such is how a challenge gets thrown away:

  * FTMO requires at least 4 distinct TRADING DAYS. Hitting +10% on day one
    passes nothing -- three more days with trades are still needed.
  * The target must hold at the END. Touching +10% and then giving it back
    leaves you short, so full-size trading after the target is banked risks
    losing a pass that was already in hand.
  * "Challenge" and "Verification" have DIFFERENT targets (10% then 5%) and
    the bot previously had no idea which one it was in.

This module owns that state so the runner can say exactly one of:
    "target reached, N more trading days needed"
    "PHASE 1 PASSED"
    "PHASE 2 PASSED -- funded from here, this is real money"
and so risk can be cut once the target is safely banked.

Nothing here trades or sizes anything; it reports status. The runner decides
what to do with it.
"""
from __future__ import annotations

from dataclasses import dataclass

# Trading objectives, from ftmo.com/en/trading-objectives (2-Step programme).
PHASES = {
    "challenge":    {"target_pct": 10.0, "min_days": 4, "label": "Phase 1 (Challenge)"},
    "verification": {"target_pct": 5.0,  "min_days": 4, "label": "Phase 2 (Verification)"},
    "funded":       {"target_pct": None, "min_days": 0, "label": "Funded account"},
}


@dataclass
class PhaseStatus:
    phase: str
    label: str
    target_pct: float | None
    profit_usd: float
    profit_pct: float
    target_usd: float | None
    trading_days: int
    min_days: int
    target_reached: bool
    days_met: bool
    passed: bool
    headline: str
    detail: str

    @property
    def banked(self) -> bool:
        """Target is in hand but the day count is not yet satisfied.

        This is the window where risk should be cut hard: there is nothing
        left to gain from the target and everything to lose.
        """
        return self.target_reached and not self.days_met and not self.passed


def evaluate(phase: str, profit_usd: float, account_size: float,
             trading_days: int) -> PhaseStatus:
    key = (phase or "challenge").lower()
    spec = PHASES.get(key, PHASES["challenge"])
    target_pct = spec["target_pct"]
    min_days = int(spec["min_days"])
    profit_pct = (profit_usd / account_size * 100.0) if account_size else 0.0

    if target_pct is None:                      # funded: no target to reach
        return PhaseStatus(
            phase=key, label=spec["label"], target_pct=None,
            profit_usd=profit_usd, profit_pct=profit_pct, target_usd=None,
            trading_days=trading_days, min_days=0,
            target_reached=False, days_met=True, passed=False,
            headline="TRADING A FUNDED ACCOUNT -- this is real money",
            detail=(f"P&L {profit_usd:+,.2f} ({profit_pct:+.2f}%). No profit "
                    f"target; the 5% daily / 10% overall loss rules still "
                    f"apply and a breach ends the account."),
        )

    target_usd = account_size * target_pct / 100.0
    target_reached = profit_usd >= target_usd
    days_met = trading_days >= min_days
    passed = target_reached and days_met

    if passed:
        nxt = "Verification (5% target)" if key == "challenge" else "a FUNDED account"
        headline = f"{spec['label']} PASSED"
        detail = (f"{profit_usd:+,.2f} ({profit_pct:+.2f}%) >= target "
                  f"{target_usd:,.2f} ({target_pct}%), over {trading_days} "
                  f"trading days (needed {min_days}). Next: {nxt}.")
    elif target_reached:
        need = min_days - trading_days
        headline = f"{spec['label']}: TARGET REACHED -- {need} more trading day(s) needed"
        detail = (f"{profit_usd:+,.2f} ({profit_pct:+.2f}%) is past the "
                  f"{target_pct}% target, but FTMO needs {min_days} trading "
                  f"days and only {trading_days} have been logged. The target "
                  f"must still hold at the END, so risk should be cut until "
                  f"the day count is met.")
    else:
        remaining = target_usd - profit_usd
        headline = (f"{spec['label']}: {profit_pct:+.2f}% of {target_pct}% "
                    f"({trading_days}/{min_days} trading days)")
        detail = (f"{remaining:,.2f} more profit needed to reach the "
                  f"{target_pct}% target.")

    return PhaseStatus(
        phase=key, label=spec["label"], target_pct=target_pct,
        profit_usd=profit_usd, profit_pct=profit_pct, target_usd=target_usd,
        trading_days=trading_days, min_days=min_days,
        target_reached=target_reached, days_met=days_met, passed=passed,
        headline=headline, detail=detail,
    )


def next_phase(phase: str) -> str | None:
    """What follows a passed phase."""
    return {"challenge": "verification", "verification": "funded"}.get(
        (phase or "").lower())
