"""Equity-milestone risk ratchet -- CLAUDE.md "Risk ratchet" (spec §3.3):
reduce risk in drawdown, protect gains near the target, NEVER increase risk
to chase a deadline. Marked a NON-NEGOTIABLE invariant in that spec; the
thresholds below are copied verbatim, not tuned.
"""
from __future__ import annotations


def current_risk_pct(equity: float, initial: float = 10_000.0) -> float:
    """Risk fraction (e.g. 0.005 = 0.5%) of the CURRENT equity, keyed off
    how far equity has moved from `initial` as a fraction of `initial`."""
    if initial <= 0:
        return 0.0
    p = (equity - initial) / initial
    if p >= 0.08:
        return 0.0025   # near target: protect
    if p >= 0.05:
        return 0.0040
    if p >= 0.00:
        return 0.0050   # base
    if p >= -0.03:
        return 0.0040
    if p >= -0.05:
        return 0.0025
    return 0.0015
