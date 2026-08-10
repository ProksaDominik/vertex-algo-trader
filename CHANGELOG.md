# CHANGELOG

## Phase 8 — Per-strategy config, 3 new strategies, comparison, per-strategy exits, full neon (2026-07-21)

### Config restructured: one entry per strategy (not per pair)
A strategy is now written ONCE with a `symbols:` basket + `magic_base:`
instead of a copy-pasted slot per pair. `expand_configs()` turns each into
per-symbol "legs" named `Strategy__SYMBOL` (magic = magic_base + index), each
carrying `strategy_group` so the dashboard/comparison aggregate by STRATEGY.
Legacy single-`symbol` entries still work. Validation extended (symbols
require magic_base; expanded-magic collisions caught). Dashboard equity
curves now group by strategy — one climbing line per strategy, "just
Donchian" not one-per-pair.

### Three new strategies (`strategies_sweep_reclaim/tod_reversion/vwap_reversion.py`)
- **SweepReclaim** — fades a liquidity grab: pierce a swing low, close back
  above it, go long (mirror for shorts).
- **TODReversion** — at a set hour, fade an over-extension from the day's
  open back toward it.
- **VWAPReversion** — fade a stretch from the session VWAP when the bar
  rejects back toward it (the only volume-aware strategy).
All registered, param-wired, and unit-tested. Full write-up of every
strategy in `STRATEGIES.md`.

### Which is best — head-to-head comparison (`scripts/compare_strategies.py`)
Every strategy run across a 10-symbol basket, validation window, each with a
style-appropriate exit (trend = 2R + 1R trail; reversion = 1R, no trail),
ranked by out-of-sample expectancy (`backtest/strategy_comparison.md`):

| rank | strategy | avg $/trade | positive symbols |
|---|---|---|---|
| 1 | Donchian | **+4.1** | 5/10 |
| 2 | MACD | −5.8 | 3/10 |
| 3 | BOS | −7.9 | 3/10 |
| 4 | VWAPReversion | −16.9 | 3/10 |
| 5 | SweepReclaim | −25.9 | 0/10 |
| 6 | TODReversion | −29.7 | 2/10 |

**Donchian is the only net-positive strategy.** Deployment follows the
evidence: each strategy runs only on the symbols it's positive on (Donchian
on 5, MACD/BOS on their few, VWAP on EURUSD, TOD on EURUSD/YOLO-only), Sweep
ships disabled (0/10).

### Per-strategy exits (`portfolio_runner.py`)
Exit is no longer one book-wide setting. Each strategy leg can override
`trail_start_r` / `trail_lock_fraction` / `skyrocket_multiple` via its
`strategy_params.exit`, so trend legs run a wide TP + 1R trail and reversion
legs run a tight 1R TP with no trail — the exit that fits each idea. SL and
TP are already per-trade (SL at structure, TP = R × that distance), so every
trade's stop/target adapts to current market structure and volatility.
Validation now checks TP-vs-effective-trail per leg.

### Full neon interface
The neon skin now covers the whole app (per the video): data tables (neon
header + glow), tabs (neon active underline), buttons (neon outline → fill),
sidebar edge, ticker chips, header rule, metric glow. Layout unchanged.

### Tests
10 new (67 total, all passing): config expansion + magic-collision +
reversion-exit validation, shipped per-strategy configs valid, and each new
strategy fires correctly (sweep reclaim long, VWAP stretch-below, TOD at the
target hour, one-per-day).

### Honest notes
- Only Donchian has a broad edge. The new strategies are mostly there to be
  tested — and the test says most don't work; they're deployed narrowly or
  disabled accordingly, not because they're good but because the data earned
  them a small live sample.
- Reversion strategies fire a LOT (Sweep ~200 trades/symbol); the tight-TP
  no-trail exit is their fair shot, and they still mostly lose — that's a
  real result, not a config bug.

---

## Phase 7 — Learning engine, BOS strategy, neon dashboard, YOLO reset (2026-07-21)

Prompted by "make it self-healing and self-improving — analyze each lost
trade, remember what happened, learn from it" plus two reference videos of a
neon self-healing terminal.

### Bounded learning — "learns after every win/loss" (`trade_review.py`)
Every closed trade now gets a stored post-mortem: realized R (from price
geometry), exit reason, session-of-day, hold time, win/loss, and a plain-
language WHY ("stopped out for -1.00R — long entry reversed in the asian
session"). Saved to a new `trade_reviews` ledger table = the memory. A
`SessionLearner` aggregates those into per-(slot, session) verdicts and, once
a slot has ≥6 trades and ≤ -3R in a given session, benches that slot for that
session only (existing positions keep being managed). Rebuilt from the ledger
on every restart. Deliberately does NOT mutate SL/TP/size from recent P&L —
that's live curve-fitting and how accounts blow up; the honest version
changes WHERE/WHEN a proven edge fires, never the edge itself. This sits on
top of the existing slot-health quarantine (whole-slot) from Phase 1+.

The videos' "23,000,000+ trades analyzed by AI" is marketing; this is the
real, bounded mechanism behind that idea — no fake AI label.

### New strategy — Market-Open Break of Structure (`strategies_bos.py`)
At a session open (NY 13:00 UTC default) it reads the last confirmed swing
high/low — real fractal-pivot market structure, confirmed k bars later so
there's no lookahead/repaint — and enters the first clean break, stop at the
opposing structure. One trade per symbol per day. Selected by the SAME out-
of-sample discipline as everything else: a walk-forward showed BOS is clearly
positive on GOLD (+$32/trade validation) and marginal on DE40, but NEGATIVE
on US30/USDJPY — so only XAUUSD + DE40 BOS ship (enabled) in each book; the
losing symbols are present but `enabled: false`. Both books now run 6 active
slots (4 prior + 2 BOS).

### Dashboard — neon skin + improving-curves + self-healing panel (`dashboard.py`)
Layout unchanged (per request); restyled to the video's neon look via a CSS
layer on top of the existing theme (glow, neon-cyan/magenta accents, left-
rule subheaders). Two new panels in place: a multi-line neon "Strategy equity
curves" chart (lightweight-charts, one climbing line per strategy — the "how
each is improving" graph from the video), and a "Self-healing" panel showing
the learned session verdicts and a live feed of each trade's why-it-happened
review.

### YOLO reset to $200 (`scripts/reset_yolo.ps1` / `.bat`)
One-click: stops both bots, archives YOLO's ledger + state (realized P&L
zeroed → book equity back to $200, circuit-breaker halt cleared, high-water
re-captured at 200), leaves FTMO fully intact, relaunches both on the new
code. YOLO's breaker had correctly halted it after a 38% drawdown from peak.

### Tests
20 new (57 total, all passing): session classification, review reasoning for
each exit type, R-from-geometry vs pnl fallback, SessionLearner thresholds +
verdict ordering, ledger review roundtrip + learner rebuild, BOS pivot
detection, BOS fires on in-session structure break (one/day), BOS ignores
out-of-session breaks.

### Honest notes
- BOS is new; even the two shipped slots are ~7-week-validation, not a
  guarantee — slot-health + the session learner are the live backstops.
- The learned session filter needs ≥6 trades per session before it acts, so
  it does nothing on day one — that's intentional (no acting on noise).

---

## Phase 6 — Stop/TP geometry analysis (2026-07-21)

Triggered by the question "is the TP too far and the SL too far?". Analyzed
every archived closed trade, then swept stop-placement × take-profit
out-of-sample on the live symbols (`scripts/geometry_sweep.py`,
`strategies_donchian.py` gained a backward-compatible `sl_mode`:
channel / channel_frac / atr; default unchanged).

Findings:
- **The pre-fix "disprofit" was structural, not bad luck.** Under the old
  0.333R TP, ~85% of trades closed green but wins averaged only +0.18R while
  the two real losses were −1.0R each — net negative after costs. Most
  "sl"-reason exits were actually the trailing stop banking a tiny profit,
  not a loss. (This is the H1 issue, already fixed by restoring wider TPs.)
- **Stop loss is NOT too far.** Tightening it (½ channel / ATR) was clearly
  worse on every symbol tested — expectancy fell and drawdown ~doubled
  (US30 went outright negative). The wide Turtle stop is correct; it keeps
  the system in breakouts that wobble. **No change.**
- **Take profit was mildly too far.** Best TP is symbol-specific (USDJPY
  ~1.5R, US30 2–3R) and the gaps are within 7-week-sample noise, so no
  per-symbol curve-fitting. The robust, non-overfit finding: **2R earns
  ~the same as 3R with roughly one-third the drawdown** — and drawdown, not
  raw profit, is what fails an FTMO account.

Change (user-approved): `tp_r_multiple` 3.0 → **2.0** on all slots, both
books. Stop geometry unchanged. `scripts/geometry_sweep.py` kept as the
tool to re-run this analysis on more data later.

---

## Phases 2–5 — Self-healing, FTMO guard, YOLO profile, proof (2026-07-21)

### Phase 2 — Self-healing (`state.py`, `alerts.py`, `mt5_engine.py`, `portfolio_runner.py`)
- **Persistent state file** (`data/state_<book>.json`, atomic writes, corrupt-file
  recovery): overall-DD anchor captured ONCE and never re-anchored by restarts
  (C2), CE(S)T daily anchor, halt state, quarantine benches, NoWick pending
  setups (M8), book high-water mark, last known broker-clock offset (M4).
- **Kill-switch semantics rebuilt**: guard trips now FLATTEN the whole book
  (per-position retries) and set a persisted halt — the process stays alive,
  keeps reconciling, and auto-resumes when allowed (daily halts clear at the
  next CE(S)T day; total/breaker halts need a manual state reset). The old
  behavior (process exit + fresh anchors on the next start_all.ps1) is gone.
- **Watchdog** each tick: terminal disconnected / AlgoTrading off / stale feed
  → entries paused, positions still managed, auto-resume, alert on change (M3).
  Account-info probe now fails CLOSED after 3 consecutive failures.
- **Stops-level handling** (M5): entry SLs tighter than the broker minimum are
  widened BEFORE sizing (risk preserved); trailing modifies inside the minimum
  distance are skipped instead of spamming 10016 rejections.
- **Alerts** (`alerts.py`): every guard trip, quarantine, watchdog change and
  flatten logs loudly and (optionally) pushes to Telegram — set
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in `.env`. Fire-and-forget, stdlib only.

### Phase 3 — FTMO guard module (`ftmo_guard.py`, ftmo block in the YAML)
Pure-logic, unit-tested to the timezone boundary. Daily P&L = today's realized
(from the CE(S)T-midnight anchor) + current floating — floating losses now
count (C3). Triggers at −3.5% daily / −8% overall of the configured account
size (buffers under FTMO's 5%/10%), flatten + halt on trip, +10% target
alert/halt, weekend-flat (Friday 20:30 UTC), static news-blackout windows
(no live calendar feed — maintained manually, ships empty). Scope `book`
approximates FTMO math while both bots share one demo account (C4); switch
to `account` + run the FTMO bot alone for a real challenge.

### Phase 4 — YOLO profile (capital block in `portfolio_yolo.yaml`)
`percent_equity` sizing: 4% of (200 + realized book P&L) — compounds when
winning, shrinks when losing, zero when the book is gone. Circuit breaker:
flatten + halt 35% below the book's persisted high-water mark ("no kill
switch by design" retired — compounding requires survival). Partial take:
bank 50% at +1.5R, remainder rides the 1R trail. Capped pyramiding
implemented but OFF by default (one add, half size, breakeven stop).

### Phase 5 — Proof
- **Test suite**: 37 tests, 0 → 37 (this repo had none). Covers dedup +
  migration, volume normalization, config validation, state roundtrip +
  corruption, FTMO guard daily/total/CE(S)T-rollover/weekend/news, restart-
  mid-position adoption + reconciliation (FakeEngine), reconcile re-queue,
  guard flatten-on-floating-loss + halt persistence across restart, missing-SL
  repair, percent-equity compounding, circuit-breaker HWM, watchdog blocking,
  pending-signal persistence, disabled slots.
- **Walk-forward** (`scripts/walk_forward.py`, results in
  `backtest/walkforward_phase5.{csv,md}`): last ~5.5 months of 30m bars per
  pair, 70/30 split, validation strictly out-of-sample. Exit-stack comparison
  (validation window):

  | book | stack | trades | net P&L | avg PF | positive slots |
  |---|---|---|---|---|---|
  | FTMO | before (0.333R + 0.15R trail) | 3103 | −27,670 | 0.63 | 1/19 |
  | FTMO | after (3R + 1R trail) | 1000 | −7,120 | 0.91 | 4/19 |
  | YOLO | before | 3857 | −8,741 | 0.68 | 1/21 |
  | YOLO | after | 1742 | −4,467 | 0.86 | 4/21 |

  **Honest verdict:** the exit restoration is unambiguously right (bleed cut
  60–75%, PF up across the board) — but the basket as previously deployed
  LOSES out-of-sample. 32 of 40 slots are negative even with good exits.
  **Approved decision: those 32 slots are now `enabled: false` in the YAMLs.**
  Live basket = the 8 that validate positive: FTMO usdjpy/us30/euraud
  Donchian + de40 MACD; YOLO xauusd/usdjpy(15m)/us30 Donchian + de40 MACD.
  Caveats stated plainly: the validation window is ~7 weeks of one regime;
  fills are bar-close idealized; partial-take unmodeled; the 8 survivors are
  evidence, not a guarantee — slot-health quarantine remains on as the live
  backstop. Also observed: the US30 comparison suggests the skyrocket exit
  costs R on validation (worth a dedicated study before trusting it).
- **Restart simulation**: FakeEngine scenario proves a mid-position restart
  adopts the live position, reconciles its close into the ledger, and that
  guard halts survive restarts.

### Migration notes (fresh start)
`scripts/restart_fresh.ps1` (+ `.bat`) stops both bots + dashboard + chart
API, archives ledgers/WAL sidecars/state files to `data/archive_phase5_<ts>/`,
and relaunches via start_all.ps1 — clean DBs, fresh anchors, 8-slot basket.
`requirements.txt` gains `tzdata` (Windows) for CE(S)T handling.

---

## Phase 1 — Correctness + risk containment (2026-07-21)

All AUDIT IDs reference `AUDIT.md`. No strategy ENTRY logic was touched; exit
parameters were changed only with explicit approval (H1 decision).

### C1 — Signal dedup rebuilt on bar timestamps (`ledger.py`, `portfolio_runner.py`)
The idempotency key was `(config_name, bar_index)` where bar_index is the bar's
position in the rolling 300-bar window — always 299 for the newest bar — so a
slot could never fire twice. Confirmed in production: every slot exactly one
signal, then silence. New key: `(config_name, bar_ts)`. In-place DB migration
adds the `bar_ts` column; old NULL rows never match, so all slots revive.
**Expect trade frequency to jump from ~5/day (replay leaks) to the configured
basket's real rate. The risk caps below exist precisely because of this.**

### H2 — Portfolio-wide risk caps (`portfolio_runner.py`, both YAMLs)
New per-book caps, all YAML-tunable, each open position ≈ 1R:
`max_total_open_risk_r` (FTMO 3 / YOLO 5), `max_concurrent_positions_total`
(6 / 8), `max_currency_exposure_r` enabled at 2 (existed, was off),
`max_positions_per_config` lowered 5 → 2. Uncapped, the books could hold
$9,500 / $3,150 of open risk on $10k / $200 framings.

### C5 + H7 — Live sizing and honest volume handling (`mt5_engine.py`, `portfolio_runner.py`, `instruments.py`)
Lots are now computed from live `symbol_info` tick_value/tick_size at order
time (tick value ÷ tick size = money per 1.0 price move — the two are not
interchangeable). `instruments.py` remains for backtests only and now RAISES
on unknown symbols instead of silently returning XAUUSD specs (which sized
unknown FX pairs ~1000× too large). Volume snaps to broker step/min/max via
`normalize_volume` — flooring DOWN, never up. Below-minimum ideal size now
follows `min_lot_policy`: FTMO `skip` (keeps risk honest), YOLO `floor`
(explicit, accepts inflated risk). A >25% live-vs-static multiplier drift
logs a warning to keep backtests honest.

### H3 — Order path: retries, deviation, partial fills (`mt5_engine.py`)
`place_market_order` and `close_position` retry transient retcodes (requote,
price-changed, off-quotes, too-many-requests, no-connection, and `order_send
→ None`) up to 3× with exponential backoff + jitter, refreshing the tick and
re-anchoring SL/TP each attempt. Requests now carry `deviation` (20 points
default). `OrderResult.ok` accepts DONE_PARTIAL — a partial fill is a real
position and is now tracked with its actual filled volume instead of being
logged as a failure and left invisible.

### H4 — Reconciliation re-queues instead of dropping (`portfolio_runner.py`)
Closed positions whose deal history isn't queryable yet are retried for ~2
minutes; only then ledgered as `unreconciled` (NULL pnl) with a loud error.
Previously the first miss silently erased the trade from the ledger — and
from the daily loss guard's view.

### H5 — Stop-integrity enforcement (`portfolio_runner.py`)
Immediately after every fill, and every tick for every live position: missing
SL → repaired from the intended SL, or reconstructed as a 1R stop from
configured risk + live specs. A position without a stop is never acceptable.

### H6 — Replay staleness gate (`portfolio_runner.py`)
After downtime, replayed bars still feed strategy state, but signals older
than `max_signal_age_bars` (default 2) are logged and NOT executed. An
hours-old breakout at today's price is not the trade the backtest measured.

### M1 — Margin-mode assertion (`mt5_engine.py`, `portfolio_runner.py`)
Startup now verifies the account is retail-hedging (`.env.example` documents
this account as *Netting* while config comments claim hedging — runtime truth
wins). Netting refuses to start unless `mt5.allow_netting: true`.

### M6 — Startup config validation (`portfolio_runner.py`)
`validate_portfolio_cfg`: duplicate magics/names, unknown strategies, symbols
missing from PAIR_SPECS, non-negative loss limits, TP ≤ trail contradictions.
Any problem → refuse to start, all problems listed.

### M2 (pulled forward) — SQLite WAL + busy_timeout (`ledger.py`)
Three processes share each ledger file; WAL removes the intermittent
"database is locked" failures.

### H1 (approved decision) — Exit stack restored (`both YAMLs`)
`tp_r_multiple` 0.333 → **3.0** on all 40 slots (the sweep optimum for every
combo present in `backtest/sweep_ftmo.csv` / `sweep_final_9pairs.csv`; the
expansion pairs' sweep CSV is not in the repo — 3.0 matches their "_3r" slot
names and config comments, unverified beyond that). Trailing re-tuned
0.15R → **1.0R** engage (lock 50%) so it protects a mature winner instead of
strangling every trade at ~0.16R. Live data with the old stack: avg win
+0.18R vs −1R losses → needed ~85% WR to break even.

### Self-improving layer (new) — Slot health + quarantine (`portfolio_runner.py`, `ledger.py`, both YAMLs)
Rolling realized-R window per slot (`slot_health` in YAML). A slot netting
≤ −4R (FTMO) / −5R (YOLO) over its last 8 trades is quarantined — no new
entries — for 24h / 12h, then re-enabled on a fresh window. State rebuilds
from the ledger on every restart. Deliberately does NOT mutate strategy
parameters live: that's real-time curve fitting, the exact failure mode the
working rules ban.

### Honest expectations
- The sweeps behind the restored TPs also show several deployed combos with
  **negative expectancy even at their best TP** (e.g. NZDUSD Donchian 30m,
  US500 everything, EURUSD Donchian at any TP). The slot-health quarantine
  will bench chronic losers live, but Phase 5's walk-forward should decide
  which slots deserve to exist at all.
- The backtests that justify the basket predate these exit changes and never
  modeled trailing/skyrocket exits or position stacking. Treat all prior
  projections as invalid until Phase 5 re-runs them.
- FTMO guard math remains structurally wrong until Phase 3 (equity-based
  CE(S)T daily anchor, flatten-on-trip, persisted anchors) — do not attempt
  a real challenge on Phase 1 alone.

### Still open (Phase 2/3/4)
Watchdog (stale ticks / AlgoTrading off), persistent state file, kill-switch
flatten + persisted daily/total anchors, CE(S)T reset, equity-based daily
guard, news blackout + weekend-flat, Telegram alerts, YOLO circuit breaker,
`log_closed` using the deal's true close time, stops-level validation on
modifies, dashboard surfacing of quarantine/guard state.
