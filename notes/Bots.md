---
type: reference
updated: 2026-10-03
tags: [bots, index]
---

# Bots

Every process that trades (paper) or forward-tests. "Last activity" is a
snapshot from the files on **2026-10-03** — refresh it when a bot changes.

## MT5 FX paper books (`scripts/run_paper.py`, one process per book)

| Book | Config | Ledger | Log | Last activity (2026-10-03) |
|---|---|---|---|---|
| FTMO ($10k prop-style, kill switches) | `config/portfolio_ftmo.yaml` | `data/paper_ledger.sqlite` | `logs/paper_portfolio_ftmo.log` | last heartbeat 2026-08-14, 0 closed trades in ledger |
| YOLO ($200, no kill switch) | `config/portfolio_yolo.yaml` | `data/paper_ledger_yolo.sqlite` | `logs/paper_portfolio_yolo.log` | last heartbeat 2026-08-16, 1 closed trade (2026-08-14) |
| FTMO_4DAY (adaptive risk) | `config/portfolio_ftmo_4day.yaml` | `data/paper_ledger_ftmo4day.sqlite` | `logs/paper_portfolio_ftmo_4day.log` | last heartbeat 2026-08-16, 0 closed trades |

Started together by `scripts/start_all.ps1`; dashboard at http://localhost:8501.
Decisions: [[2026-08-10 Add FTMO_4DAY adaptive-risk bot]], [[2026-07-17 Disable ratchet trailing stop]].

## Futures prop-firm simulations (JSONL trade logs)

| Bot | Script | Trades | Log | Last activity (2026-10-03) |
|---|---|---|---|---|
| Lucid futures | `scripts/run_lucidpro_trial.py` (futures mode) | `data/lucid_futures_trades.jsonl` | `logs/lucid_futures.log` | 6 trades, last closed 2026-10-01 |
| LucidPro trial | `scripts/run_lucidpro_trial.py` | `data/lucidpro_trades.jsonl` | `logs/lucidpro_trial.log` | 13 trades, last closed 2026-08-27 |
| LucidPro funded | `scripts/run_lucidpro_funded.py` | `data/lucidpro_funded_trades.jsonl` | `logs/lucidpro_funded.log` | 0 trades logged |
| Lucid futures funded | `scripts/run_lucid_futures_funded.py` (+ `_roster`) | state `data/state_lucid_futures_funded.json` | `logs/lucid_futures_funded.log` | log written 2026-10-01 |

Research behind these: [[FUTURES_RESEARCH_REPORT]], [[LUCIDPRO_PASS_RESEARCH]].

## Strategy lab (shadow mode — no orders)

| Lab | Script | Output | Log | Last activity (2026-10-03) |
|---|---|---|---|---|
| Lab (FX) | `scripts/run_lab.py` | `data/lab_signals.jsonl`, `data/lab_state.json` | `logs/lab.log` | — |
| Lab (futures) | `scripts/run_lab.py` (futures mode) | `data/lab_futures_signals.jsonl`, `data/lab_futures_state.json` | `logs/lab_futures.log` | running (log updated 2026-10-03) |

## Scheduled jobs
- **MetatraderbotDailyReview** — `scripts/daily_review_task.ps1` → report in
  `logs/daily_review/` **and** today's note in `notes/Journal/`
  (register with `scripts/register_daily_review.ps1`, default 23:30).
  As of 2026-10-03 the task is **not registered** on this machine; the last
  report is from 2026-09-10.
