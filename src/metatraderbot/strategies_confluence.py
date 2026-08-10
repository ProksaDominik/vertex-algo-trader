"""Confluence: only trade when two independent signals agree.

Measured (scripts/confluence_strategy.py, 10.5y, 17 FX pairs, both halves
positive): requiring RSI(14) and RSI(7) to fire on the same bar in the same
direction lifts the out-of-sample win rate from 33.7% to 65.6%. Race-tested
against the real FTMO rules (scripts/confluence_race_test.py) that took the
10-day pass rate from 32.3% to 39.1% -- the best result in this project.

Why it works: the two settings overlap on only ~10% of signal days, so
agreement is genuinely selective rather than the same signal counted twice.
The equity path becomes many small wins instead of rare large ones, which is
what a 5%-daily-loss limit rewards.

What it does NOT do: raise expectancy. Confluence at 0.75-1.0R earns
+0.099R to +0.134R, LOWER than a single RSI leg at 4R (+0.174R). It trades
payoff for consistency. That is the right trade against a deadline and the
wrong one for open-ended compounding -- the YOLO book keeps the wide targets.

Config:
    strategy: Confluence
    strategy_params:
      confluence:
        legs:
          - {strategy: RSIReversion, rsi: {period: 14, oversold: 30, overbought: 70}}
          - {strategy: RSIReversion, rsi: {period: 7,  oversold: 25, overbought: 75}}
      exit: {tp_r_multiple: 1.0, sl_swing_lookback: 10}

The FIRST leg owns the trade -- its TradeSpec (entry, stop, target, sizing) is
what gets returned. The others are pure confirmation. That matters because the
legs may disagree on stop placement, and mixing them would produce a trade
neither strategy actually validated.
"""
from __future__ import annotations

import logging

import pandas as pd

from metatraderbot.strategy import TradeSpec

log = logging.getLogger(__name__)


def _build_leg(leg_cfg: dict, base_config: dict):
    """Instantiate one sub-strategy from its own params + the shared exit block."""
    # Imported here rather than at module scope: portfolio_runner imports this
    # module while building STRATEGY_CLASSES, so a top-level import of it
    # would be circular.
    from metatraderbot.portfolio_runner import STRATEGY_CLASSES

    name = leg_cfg.get("strategy")
    if name not in STRATEGY_CLASSES:
        raise ValueError(
            f"Confluence leg '{name}' is not a registered strategy. "
            f"Known: {sorted(STRATEGY_CLASSES)}")
    sub = dict(base_config["strategy"])
    for k, v in leg_cfg.items():
        if k != "strategy":
            sub[k] = v
    cfg = dict(base_config)
    cfg["strategy"] = sub
    return STRATEGY_CLASSES[name](cfg)


class ConfluenceStrategy:
    def __init__(self, config: dict):
        s = config["strategy"]
        conf = (s.get("confluence") or {})
        legs = conf.get("legs") or []
        if len(legs) < 2:
            raise ValueError(
                "Confluence needs at least 2 legs — with one leg it is just "
                "that strategy, and the agreement filter does nothing.")
        self.legs = [_build_leg(lc, config) for lc in legs]
        # Require agreement on DIRECTION too, not merely that both fired.
        # Two mean-reversion signals pointing opposite ways is not confluence,
        # it is a disagreement, and taking either side would be arbitrary.
        self.require_same_direction = bool(conf.get("same_direction", True))

    def on_bar(self, bar_index: int, bars: pd.DataFrame) -> list[TradeSpec]:
        specs = []
        for leg in self.legs:
            try:
                out = leg.on_bar(bar_index, bars) or []
            except Exception as e:  # a broken leg must not take the book down
                log.warning(f"Confluence leg {type(leg).__name__} raised: {e!r}")
                return []
            if not out:
                return []          # any leg silent -> no agreement -> no trade
            specs.append(out[0])

        if self.require_same_direction:
            first = specs[0].direction
            if any(sp.direction != first for sp in specs[1:]):
                return []

        # The first leg owns the trade: its stop/target/sizing were validated
        # together. Blending levels across legs would create a trade none of
        # them actually signalled.
        return [specs[0]]
