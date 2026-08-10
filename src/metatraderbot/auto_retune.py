"""Live strategy parameter auto-retuning.

DIFFERENT from trade_review.py's bounded learning, which only ever changes
WHERE/WHEN a strategy is allowed to fire. This module actually rewrites a
strategy's own exit parameter (tp_r_multiple) from its recent live results,
with no human review — the exact thing called "curve-fitting the account in
real time" everywhere else in this codebase. Built anyway, on explicit
request, with the tightest bounds that still do something:

- each retune step shrinks tp_r_multiple by step_pct of its CURRENT value
  (compounding down, so the steps shrink as the value approaches the floor
  instead of overshooting it)
- never goes below floor_frac of the slot's ORIGINAL configured value
- capped at max_retunes total per slot — past that, the slot-health
  quarantine still fires on further degradation, but the parameter itself
  stops moving, so a structurally broken strategy can't be retuned into a
  degenerate near-zero target forever

Pure/deterministic given the inputs, so it unit-tests without MT5 or a
database, same as trade_review.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetuneState:
    original_tp_r: float
    current_tp_r: float
    retune_count: int = 0


def next_tp_r_multiple(
    state: RetuneState, *, step_pct: float, floor_frac: float, max_retunes: int,
) -> tuple[float, bool]:
    """One retune step. Returns (new_tp_r_multiple, applied).

    applied=False (value unchanged) once max_retunes is reached, or the
    floor has already been reached on a prior step."""
    if state.retune_count >= max_retunes:
        return state.current_tp_r, False
    floor = state.original_tp_r * floor_frac
    if state.current_tp_r <= floor:
        return state.current_tp_r, False
    candidate = state.current_tp_r * (1.0 - step_pct)
    return max(candidate, floor), True
