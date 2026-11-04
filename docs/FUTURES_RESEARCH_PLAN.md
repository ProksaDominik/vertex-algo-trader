# Lucid futures research plan

Status: **steps 1-6 done 2026-09-30 in a 4-hour session -- results in
[FUTURES_RESEARCH_REPORT.md](FUTURES_RESEARCH_REPORT.md).** One intraday stack
survived (ORB + trend filter on MNQ/MGC) and is deployed to the simulator bots.
Still open: exits, walk-forward re-selection, more New York-session ideas,
and the multi-day decision (step 7).

## The brief (owner's answers, 2026-09-29)

| Question | Answer |
|---|---|
| Objective when risk, profit and speed conflict | **Best expected $ per week** (after eval fees and blown accounts) |
| Plan traded | **LucidPro 50K**: $3,000 target, $2,000 EOD-trailing MLL, $1,200 soft DLL, 4 minis / 40 micros, flat by 4:45 PM ET |
| Failed evaluations acceptable before a pass | **1-2 (conservative)**: the eval stack must be sized to survive, not to race |
| Research compute | Sessions of **<= 1 hour**; the machine is not left on overnight |
| Holding period | **Intraday first.** If no intraday edge survives, holds of **up to ~3 days** are acceptable to the owner -- but **LucidPro does not allow them**: every position is closed at 4:45 PM ET, and even LucidLive prohibits swing trading. Multi-day legs therefore mean a different firm/account that permits overnight holds, and are researched only as a fallback track (step 7). |

Consequence for the trial bot: its 90%-risk "race" sizing contradicts a 1-2
failed-eval budget. It stays as a simulator experiment only; the eval stack
this plan produces replaces it.

## What is known (session 1)

`scripts/futures_roster_search.py`: the lab's 16 strategy families x 37
parameter sets x {15m, 1h} x {1R, 2R} on 12 Lucid contracts, ~15 months of 15m
and ~20 months of 1h history from NinjaTrader, simulated under Lucid's rules
(entry next bar open, no entries in the last 15 min, flat at the cutoff, one
tick + Lucid commission both ways, stop-first on ambiguous bars).

* **No detectable edge in the existing library.** 1,730 tests with n >= 50:
  16% positive, mean -0.094R/trade, **5 with t > 2 where ~40 are expected by
  chance**, none with t > 3. The "proposed roster" in
  `data/futures/funded_roster.json` is therefore best read as noise and is
  running on the simulator only as a forward test.
* **Mean reversion loses everywhere** (stochastic, S/R retest, time-of-day,
  sweep, VWAP, RSI, Bollinger: mean -0.10 to -0.26R). Trend/breakout families
  sit near zero; London breakout is the least bad family (-0.003R).
* **1h beats 15m** (-0.063 vs -0.123R): costs dominate short holds.
* **Gold (MGC) is the least bad contract**; the Canadian dollar (6C) the worst.
  Gold's many positive trend configs reflect ONE regime (the 2025-26 rally),
  not independent evidence.
* **The trial's London Breakout (15m, 2R) has no futures edge**: 6C -0.083,
  6B -0.031, 6S +0.051, MGC +0.036R.

## Next sessions (in order, each resumable)

1. **Futures-native intraday hypotheses** (new strategy families, not the
   CFD library). Each is published or widely documented, fits the
   flat-by-4:45 rule, and gets ONE textbook specification before any tuning:
   * Intraday momentum: first-30-minute return of the US session predicts the
     last-30-minute return on index futures (Gao, Han, Li & Zhou 2018).
   * Opening-range breakout on index futures with a volatility / trend-day
     filter (ORB failed unfiltered here).
   * Overnight gap continuation vs fade at the 9:30 ET open.
   * Pre-FOMC announcement drift in equity index futures (Lucca & Moench 2015).
   * Volatility-breakout (range expansion) with a daily trend filter.
   * Daily-trend filter layered on the 1h trend families that were near zero
     (trade only in the direction of the 20/50-day trend).
2. **Exits**: time stops, breakeven, trailing (families near zero often die
   on exits, not entries). Tested on the survivors of step 1 only.
3. **Robustness gates** (all must pass): walk-forward (rolling 6-month train /
   2-month test), both halves positive, 2x and 3x costs, parameter
   neighbourhood stable (not a lone peak), Benjamini-Hochberg across every
   test run so far (the count accumulates across sessions).
4. **Stacks**: combine survivors with low daily-P&L correlation; allocate risk
   per leg; respect the 40-micro cap.
5. **Lucid Monte Carlo** (block-bootstrap of trading days): for the EVAL stack,
   P(pass), days to pass and P(fail) under the trailing MLL and DLL for risk
   levels 0.25-1.5%; for the FUNDED stack, P(payout before blow-up), weeks to
   payout and expected $/week. Pick the risk level maximising expected
   $/week subject to <= 1-2 expected failed evals per pass.
6. **Deploy** the eval stack to the trial bot and the funded stack to the
   funded bot (both on simulator accounts first); the lab keeps forward-testing
   everything.
7. **Fallback track -- holds up to 3 days** (only if steps 1-5 find nothing
   intraday, and only for an account that allows overnight positions): the
   1h/4h/daily trend families with a 3-day time stop. The MT5 research found
   its best edges at these horizons (Donchian/MACD on 1d/4h), which is why
   this is the natural fallback -- but it cannot run on LucidPro.

## How to run a session

```powershell
# resumes where the last session stopped; stops handing out work at 50 min
.venv\Scripts\python.exe scripts\futures_roster_search.py --no-download --max-minutes 50
```

Checkpoint: `data/futures/roster_search_cells.jsonl` (append-only, one line
per finished cell). Results: `data/futures/roster_search.csv`.

## Practical constraints to solve before real money

* The bots only trade while this PC and NinjaTrader are on. The London and US
  sessions run 09:00-22:45 Prague time, so a real Lucid account needs an
  always-on machine (a small Windows VPS) -- or the bot misses its trades and
  cannot manage open positions beyond the exchange-held stops.
* The funded bot needs the `Sim102` simulation account created in NinjaTrader.
* The MLL lock at $50,100 is from secondary sources -- confirm against the
  funded account's actual terms.
