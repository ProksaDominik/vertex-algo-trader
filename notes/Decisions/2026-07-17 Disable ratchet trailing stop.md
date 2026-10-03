---
type: decision
date: 2026-07-17
status: active
bots: [FTMO, YOLO]
commit: 42790b0
tags: [decision, exits]
---

# Disable ratchet trailing stop

**Decision:** set `trail_start_r: 0.0` on both FTMO and YOLO, so the ratchet
trailing stop is off and trades ride to their SL or TP.

**Why:**
- `trail_start_r` was tuned to 0.15 when `tp_r_multiple` was 0.333. After TP
  was widened to 2.0R (FTMO) / 0.75R (YOLO) because a wider target improved
  P&L/drawdown, a 0.15R ratchet fired almost immediately and locked in tiny
  profits on the first pullback. In one live case a trade closed a few dollars
  up with its TP still ~8× further away.
- The TP-widening backtest never modelled the ratchet, so shipping both
  together was an untested combination.

**Revisit if:** a backtest models the ratchet *together with* the current TP
multiples and shows it helps.

Source: commit `42790b0`. Related: [[STRATEGIES]].
