---
type: decision
date: 2026-08-10
status: active
bots: [FTMO_4DAY]
commit: c25385f
tags: [decision, risk, new-bot]
---

# Add FTMO_4DAY adaptive-risk bot

**Decision:** run a third MT5 paper book, `config/portfolio_ftmo_4day.yaml`,
using path-dependent adaptive risk sizing (`_current_risk_usd()`): per-trade
risk tapers toward a floor as book equity nears either the +target% or the
−loss% barrier.

**Why:** validated via a historical replay in
`scratch/ftmo_4day_adaptive_risk.py` before going live.

**Also in that commit:** months of uncommitted engine work were brought into
git (FTMOGuard, challenge-phase tracking, session learner, auto-retune, K1–K11
kill switches, confluence strategy) so the bot is reproducible on a fresh
machine; fixed a `start_all.ps1` process-match regex that would have confused
the new bot's process with the original FTMO one.

Source: commit `c25385f`. See [[Bots]].
