---
type: home
tags: [index]
---

# Vertex Algo Bot — Home

Start page for the project vault. The whole repo is the vault, so the existing
docs are notes too.

## Project docs
- [[README]] — what the system is, stack, setup
- [[STRATEGIES]] — how every strategy works and how "best" was decided
- [[INSTALL]] — installing on a fresh machine
- [[AUDIT]] · [[CHANGELOG]]
- Futures research: [[FUTURES_RESEARCH_PLAN]] · [[FUTURES_RESEARCH_REPORT]] · [[LUCIDPRO_PASS_RESEARCH]]
- Backtest write-ups: [[strategy_comparison]] · [[walkforward_phase5]]

## Living notes
- [[Bots]] — every bot, its config / ledger / log, and current status
- [[Experiments]] — research and backtest log
- **Journal** — `notes/Journal/`, one auto-generated note per trading day
- **Decisions** — `notes/Decisions/`, why live behaviour is the way it is
- **Handoffs** — `notes/Handoffs/`, end-of-session summaries from each AI
- **Inbox** — `notes/Inbox/`, unsorted ideas and pasted chats

## How the AIs use this vault

```
 ChatGPT / Codex ─┐                         ┌─ Claude Code
                  ├─► reads/writes notes/ ◄─┤
   you (Obsidian) ┘          ▲              └─ (both follow AGENTS.md)
                             │
      scripts/vault_journal.py  ← ledgers + daily review, every evening
```

- **Claude Code** opens this repo, reads `CLAUDE.md` → `AGENTS.md`, and
  follows the rules there (read the latest handoff first, write one at the end).
- **ChatGPT**
  - *Codex* (CLI / IDE / app) pointed at this folder reads `AGENTS.md`
    automatically — same rules as Claude.
  - *ChatGPT web/desktop chat* can't see the folder: paste in the relevant
    notes (e.g. [[Bots]] + latest handoff) as context, and save the useful
    answer into `notes/Inbox/` (or ask Claude to file it properly).
- **The journal** is written by a script, not by an AI, so the numbers are
  always straight from the ledgers.

## Daily routine (suggested)
1. Open today's note in `notes/Journal/` — read P&L and the review.
2. Write a line or two under **Observations**.
3. Ask an AI: *"Read the last 7 journal notes and the latest handoff — anything
   abnormal?"*
