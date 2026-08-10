# AUDIT.md — Vertex Algo Bot (metatraderbot)

Audit date: 2026-07-21. Scope: FTMO + YOLO portfolios. `cryptotraderbot/` untouched (verified: zero cross-imports in either direction).

## 0. Reality check vs. the brief

The brief assumed MQL5 EAs (`.mq5`/`.mqh`, OnTick, Strategy Tester, `.set` presets). **This codebase is Python** (MetaTrader5 package driving MT5 desktop). All MQL5-specific checks were translated to their Python-API equivalents. There is also **no ICT / Fair Value Gap module anywhere in the repo** — the closest thing is NoWick (retrace-to-level). Deliverables adjust accordingly: `.set` presets become the two portfolio YAMLs; Strategy Tester becomes `backtest.py` + pytest (the `tests/` directory is currently **empty — zero tests exist**).

## 1. Bot ownership map

| Component | Belongs to | Notes |
|---|---|---|
| `src/metatraderbot/*` (all 14 files) | **Shared** FTMO + YOLO | One codebase, two processes |
| `config/portfolio_ftmo.yaml` | FTMO (19 slots, magics 2001–2020) | |
| `config/portfolio_yolo.yaml` | YOLO (21 slots, magics 3001–3021) | |
| `config/portfolio.yaml`, `config/settings.yaml` | Legacy/backtest-only | Not loaded by `run_paper.py` |
| `scripts/run_paper.py`, `dashboard.py`, `chart_api.py`, `start_all.ps1` | Shared infra | Dashboard can manually close positions |
| `cryptotraderbot/` | **Out of scope** | Not touched |

Magic numbers: unique within and across books — no collisions.

### Architecture

`run_paper.py` (one process per YAML) → `PortfolioRunner` 5s poll loop → per-slot `_process_slot` (fetch 301 bars, drop forming bar, replay new closed bars through `strategy.on_bar`) → `_execute_trade` → `MT5Engine.place_market_order` → `Ledger` (SQLite, one file per book). `_manage_positions` each tick: reconcile closed tickets from deal history, adopt untracked positions, ratchet trailing stop, skyrocket exit. Kill switches checked each loop from ledger daily P&L + account equity. Dashboard (Streamlit) + `chart_api.py` read the same SQLite files and the shared MT5 terminal; a JSON control file per book pauses new entries.

**Both processes trade one shared demo account — actual equity ≈ $99,000** (confirmed: identical balance 99,003.39 in both ledgers' heartbeats), not the $10k/$200 the configs assume. This matters for several Critical findings below.

## 2. Findings — Critical

### C1. Every slot permanently stops trading after its first trade (CONFIRMED IN PRODUCTION)
`portfolio_runner.py:325` dedups signals via `ledger.already_have_signal(slot.name, bar_index)`, and `ledger.py:148` keys on `(config_name, bar_index)`. But `bar_index` is the **row position inside the rolling 300-bar window** — the newest closed bar is always index **299**. Once a slot logs one signal, `already_have_signal(name, 299)` returns True forever, so `on_bar` is never called again for that slot (also freezing NoWick's pending-setup state). Restarts don't help; the ledger persists. The only way a slot ever trades again is a polling gap creating replay indices ≠ 299.

Production ledgers confirm it exactly: **all 17 FTMO and 22 YOLO signals sit at bar_index 297–299, exactly one 299-row per slot**, 34 signals on Jul 20 (first day after ledger reset) collapsing to 5 on Jul 21 — those 5 only via replay gaps. Both books are currently ~95% dead and this is why.

Fix: key idempotency on `(config_name, bar_timestamp)`.

### C2. Kill switch doesn't flatten, and its anchors reset on every restart
`_kill_switches_ok` (runner:265–281) on trip just breaks the loop and exits — **open positions are left running** with their original SL/TP, unmanaged. `starting_equity` (the total-DD anchor) is captured fresh at every process start and never persisted; `start_all.ps1` auto-restarts bots at logon. Chain: guard trips → process exits → next logon restarts it → total-DD guard re-anchored at the already-drawn-down equity → the bot can lose another −$800, repeatedly. The daily anchor has the same flaw plus C3. Nothing here would survive an FTMO challenge.

### C3. Daily loss guard is blind to floating losses and uses the wrong day boundary
`ledger.daily_pnl` sums **closed** trades only, bucketed by UTC date of reconciliation time. FTMO's daily limit is equity-based (floating included) and resets at midnight CE(S)T. A basket of open losers can breach −5% while the guard reads $0. Additionally `log_closed` stamps `closed_ts = now()` at reconciliation (ledger:129), discarding the engine's corrected deal close time — after downtime, losses attribute to the wrong day.

### C4. FTMO and YOLO share one account; equity guards cross-contaminate
FTMO's total-DD guard reads shared account equity — YOLO's swings can trip or mask it (and vice versa). The account is ~$99k while configs frame $10k/$200, so every percentage in the config comments is ~10× off reality. For an actual FTMO challenge the FTMO bot must run alone on the challenge account, with guards anchored to that account's initial balance and FTMO's own reset clock. Current architecture cannot express that.

### C5. Live lot sizing uses hardcoded, stale contract multipliers with a silent 1000× trap
`instruments.py` was explicitly meant for backtesting ("Phase 2 will replace these with live MT5 symbol_info calls" — never done), yet live sizing uses it. Three separate problems: (a) **unknown symbol falls back to XAUUSD specs** (`PAIR_SPECS.get(symbol, PAIR_SPECS["XAUUSD"])`, mult=100) — add any new FX pair without a dict entry and lots come out ~**1000× oversized**, silently; (b) FX-conversion multipliers are frozen snapshots — USDJPY hardcodes 700 vs ~617 at today's 162 (confirmed: YOLO USDJPY SL loss $26.59 vs $30 target, −11%), CHF-quoted pairs ~10% off; drift is unbounded; (c) index CFDs hardcode mult=1.0, which is broker-dependent. Fix: size from live `symbol_info` (`trade_tick_value`/`trade_tick_size`/`trade_contract_size`) with tick-value-vs-tick-size handled explicitly.

## 3. Findings — High

### H1. The current exit stack inverts risk/reward (be honest with yourself here)
Every deployed slot sets `tp_r_multiple: 0.333` ("per explicit request") **and** trails from 0.15R locking 50% — on low-win-rate breakout entries. Production shows the resulting shape: profitable exits average **≈ +0.18R** while full stops are **−1R**. FTMO book: 11 wins totaling +$198, 2 losses totaling −$207 → net −$8. Breakeven at that win/loss size needs ~85% win rate before spread and commission — which consume another 10–15% of a 0.333R target at Donchian-period-10 distances. The config's own July-15 sweep concluded these strategies need **3–5R** targets; the 0.333 override contradicts the system's own research, and the sweeps that chose parameters didn't model the trailing/skyrocket exits at all (backtest.py has neither). Also note the uncommitted-config churn: HEAD commit says "Disable ratchet trailing stop", working copy re-enables it. Two days of live data is statistical noise, but the structural math doesn't need a sample: this exit stack loses long-run unless entry win rate is extraordinary. Flagged, not changed — per working rules this is your call in Phase 1.

### H2. Portfolio-level open risk is uncapped
FTMO: 19 slots × `max_positions_per_config: 5` × $100 = **$9,500 theoretical open risk** on a "$10k" plan. The cross-slot currency-exposure cap exists in code (`max_currency_exposure_r`) but is **not set in either YAML → disabled**. Donchian re-fires on consecutive trending bars (channel excludes only the current bar), so one strong trend stacks up to 5 correlated entries per slot — accidental pyramiding. C1 currently masks all of this by killing slots after one trade; **fixing C1 unmasks H2**, so they must ship together with a total-open-risk cap and max-concurrent-positions guard.

### H3. No retry on transient retcodes; partial fills counted as failures
`place_market_order` retries only across filling modes. Requote/off-quotes/price-changed/no-connection → signal silently lost (production already shows 4 lost signals to 10018 market-closed and 2 to trade-mode-disabled). No `deviation` field in requests. `OrderResult.ok` accepts only 10009 — 10008 (placed) and 10010 (partial fill) are treated as total failures, so a real position could exist untracked until adoption, logged as FAIL.

### H4. Closed-trade reconciliation drops trades when deal history lags
runner:445–452 pops the ticket from `tracked` **before** `get_closed_deal`; if history isn't queryable yet (common right after close) it logs a warning and the trade is never ledgered → daily guard undercounts realized losses. Re-queue instead of pop-and-forget.

### H5. Positions without stops are never repaired
Adopted positions (restart/manual) take `orig_sl=p.sl` as-is; `sl=0` (no stop) is accepted silently — trailing math then computes `r_dist = |entry − 0|` and never fires, and nothing ever sets a stop. Violates "a position without a stop is never acceptable." No post-fill SL/TP verification exists either.

### H6. Downtime replay executes stale signals at current market price
After a gap, bars are replayed and old signals execute now, with SL/TP re-anchored to the live price (engine:377–385). A breakout from hours ago has no edge left; re-anchoring makes the order *valid* but not *sensible*. Needs a max-signal-age cutoff.

### H7. Min-lot floor silently inflates risk; no broker volume normalization
`round(max(0.01, lots), 2)` (runner:369): confirmed live — YOLO gold ideal 0.0086 → floored to 0.01 = +17% risk; tighter SLs make it worse. Volume step/min/max are assumed 0.01/0.01/∞ rather than read from `symbol_info` (indices often step 0.1 or 1.0 — those orders can reject outright).

## 4. Findings — Medium

- **M1 Netting/hedging ambiguity.** `.env.example` documents the account as "Netting"; config comments claim "hedging-mode account confirmed". Live data behaves hedging today. Both books trade the same symbols concurrently (GBPJPY appears 4×), which on a netting account merges positions, breaks per-magic tracking, and lets one bot close the other's exposure. Assert `ACCOUNT_MARGIN_MODE` at startup and refuse to run 39 slots on a netting account.
- **M2 SQLite concurrency.** No WAL, no busy_timeout, 3+ processes on the same files → intermittent "database is locked".
- **M3 Fail-open guard + no watchdog.** `_kill_switches_ok` returns True when `account_info()` fails; no stale-tick/terminal-disconnected/AlgoTrading-disabled detection — the bot can "run" while functionally dead (reconnect only triggers on IPC-flavored errors from `get_bars`).
- **M4 `broker_offset_seconds` uses last-tick time** — stale on weekends/closed markets → hours-wrong adopted/reconciled timestamps.
- **M5 No stops-level/freeze-level validation** for entry SL/TP or trailing modifies → 10016 rejections possible; trail failures are log-only and retried forever.
- **M6 No startup config validation**: duplicate magics, symbols missing from PAIR_SPECS (→ C5 trap), positive `daily_stop_loss_usd`, trail-vs-TP contradictions (0.15R trail with 0.333R TP barely differ), YOLO $30 risk = 15%/trade on the framed $200 — nothing is checked or refused.
- **M7 Backtest/live divergence.** NoWick backtests fill at the intrabar retrace level (limit-style); live sends a market order at bar close + poll latency — systematically better fills in backtest, material when TP is 0.333R. Backtest also models neither trailing nor skyrocket exits (H1) nor position stacking (H2).
- **M8 No persistent state.** NoWick pending setups and all runner state are memory-only; restart loses them (position adoption partially compensates, pending signals don't).

## 5. Findings — Low

- L1 `get_closed_deal` assumes `deals[-1]` is the exit deal — wrong exit price/reason on partial closes.
- L2 Unbounded `heartbeats`/ledger growth; no retention.
- L3 Legacy drift: `settings.yaml` duplicates NoWick config; `USD_PER_LOT_PER_DOLLAR_MOVE` dead constant; engine's computed `closed_ts` discarded by `log_closed` (see C3).
- L4 ORB (undeployed): day state assumes intraday timeframes; `range_minutes:30` on a 30m chart gives a 1-bar "range". Harmless while undeployed.
- L5 Trailing never trails stray shorts with `sl=0` (`candidate_sl < 0` never true) — subsumed by H5.
- L6 Skyrocket exit uses the still-forming bar — intentional and disclosed, but it makes live behavior untestable in the bar-close backtest (part of M7).

## 6. FTMO rule gap table

| FTMO rule | Current state | Gap |
|---|---|---|
| Max daily loss 5% (equity-based, resets midnight CE(S)T) | Closed-P&L-only, UTC day, per-book ledger, no flatten on trip (C2/C3) | Rebuild: equity+floating vs CE(S)T daily anchor snapshot, block-and-flatten at −3.5%, persisted anchor |
| Max overall loss 10% from initial balance | Process-start equity anchor, resets every restart, shared account (C2/C4) | Persist initial balance; floor at −8%; per-account not per-process |
| Reach +10% | Exit stack currently structurally negative-EV (H1); % math framed on wrong account size (C4) | Fix H1 + real account anchoring before pacing means anything |
| Consistent sizing, no martingale/grid | No martingale present ✓; but uncapped stacking ≈ accidental grid (H2) | Total-open-risk cap, per-trade % cap |
| News blackout / weekend flat | Absent entirely | Build both, switchable |
| One clean account | Two bots + dashboard on one account (C4) | Hard separation for the challenge |

## 7. Recommended fix order

Phase 1 (correctness): C1+H2 together, C5, H3, H4, H5, H7, M1, M6. Phase 2 (self-healing): C2-flatten, persisted state/anchors, watchdog, M2–M5, M8. Phase 3 (FTMO guard module): the gap table above. Phase 4 (YOLO preset): volatility-scaled %-risk, circuit breaker. Phase 5: pytest suite from zero + walk-forward re-run with H1 resolved.

Three questions before Phase 1 (working rule: no silent strategy changes):

1. **H1 / tp_r_multiple 0.333 + 0.15R trail** — restore the sweep's per-slot 3–5R targets, or keep your override? This dominates every other profitability question.
2. **C4** — is the plan a dedicated FTMO account for the challenge (recommended), or must both books stay on one account for now?
3. **C1 fix unmasks H2** — confirm you want the total-open-risk cap + enabled currency cap shipped in the same change (recommended defaults: FTMO max 3 concurrent positions total-book new-risk 3%, currency cap 2R).
