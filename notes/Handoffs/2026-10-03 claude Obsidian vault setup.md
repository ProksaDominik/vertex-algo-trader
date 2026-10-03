---
type: handoff
date: 2026-10-03
agent: claude
tags: [handoff, vault]
---

# Obsidian vault setup

## What was done
- Made the repo root an Obsidian vault with a shared knowledge base in
  `notes/` (Home, Bots, Decisions, Experiments, Handoffs, Inbox, Journal,
  Templates).
- Wrote `AGENTS.md` (rules for any AI: read the latest handoff first, write one
  at the end, log decisions/experiments) and `CLAUDE.md` (imports AGENTS.md).
- Seeded [[Bots]] from the files on disk and two decisions from git history:
  [[2026-07-17 Disable ratchet trailing stop]],
  [[2026-08-10 Add FTMO_4DAY adaptive-risk bot]].
- Added `scripts/vault_journal.py` — writes `notes/Journal/YYYY-MM-DD.md`
  from all ledgers (read-only), keeps anything written under the notes marker.
- `scripts/daily_review_task.ps1` now calls it after the review, embedding the
  report. Tested end-to-end on 2026-10-03.

## Files changed
- new: `AGENTS.md`, `CLAUDE.md`, `notes/**`, `.obsidian/{app,daily-notes,templates}.json`, `scripts/vault_journal.py`
- edited: `scripts/daily_review_task.ps1`, `.gitignore`

## Open questions / not finished
- Scheduled task `MetatraderbotDailyReview` is **not registered** on this
  machine (last report 2026-09-10) — the journal only auto-updates once it is
  (`scripts/register_daily_review.ps1`).
- MT5 FX books have no heartbeats since mid-August — intentionally stopped, or
  should they be running?
- Lab (futures) signals are not summarised in the journal yet.

## Next steps
- Register the daily task; open the vault in Obsidian; try Codex/ChatGPT on
  the same folder.
