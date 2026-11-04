# LucidPro evaluation -- highest pass rate (2026-09-30)

Brief (owner): pass the **LucidPro** evaluation, ideally in one day, with a
**60-70% pass rate**; any number of strategies may be stacked. Funded comes later.

Method: a new bar-by-bar replay of the evaluation (`scripts/lucid_pass_research.py`,
`scripts/lucid_pass_study.py`) on 10.5 years of 15m data laid on Lucid's
trading day (18:00 -> 16:45 ET), started on **every historical day** (real
order, no resampling), with Lucid costs, the worst price of every bar, the
account-level take profit, the trailing MLL and the DLL.

## UPDATE 2026-09-30 -- Lucid support (a human) answered: open equity counts

> "If your equity drops below the $2,000 Max Loss Limit while the position is
> open, the account is breached at that point." -- "If the $1,200 Daily Loss
> Limit is reached, the account will be placed into Done for Day status and
> the system will attempt to liquidate the open position." -- "The Daily Loss
> Limit is based on realized losses, not unrealized P&L."

So the **strict** reading holds for the MLL (the AI assistant's answer was
wrong) and the race's ~83% is void. Replay under exactly these rules
(2019-26, rolling starts; real MNQ bars Jun 2025 - Aug 2026 in brackets):

| Plan | Best setup | Pass | Failed evals / pass | Day-1 pass |
|---|---|---|---|---|
| LucidPro 50K | race, 2 NQ from 19:00 ET | **41%** (44%) | 1.45 | 36% |
| LucidPro 50K | ORB-trend NQ 20 MNQ | 39% | 1.53 | 10% |
| LucidPro 25K | race, 1 NQ from 19:00 ET | **45%** (48%) | 1.21 | 40% |

Exhaustive sweep (852 configs: NQ/ES/YM/GC/CL x long/short/trend x entries 19:00-09:30 x sizes,
two-market stacks, MLL lock at +$100 modelled): best **41.5%** on 50K, **46.6%** on 25K; two-market
stacks are worse (38.9% / 42.9%). `data/futures/pass_sweep_confirmed.csv`.

This is the arithmetic ceiling (a coin flip passes 40% of +$3,000 vs -$2,000
races; 44% on 25K) plus a little NQ drift. Deployed: trial bot at 2 NQ with
`LUCID_RULE_READING=lucid` (the confirmed rules). The sections below keep the
earlier analysis for the record.

## Bottom line (before the confirmation)

1. **No strategy edge gets anywhere near 60-70%.** With a $3,000 target and a
   $2,000 MLL a coin-flip strategy passes at most ~40% (and the end-of-day
   trailing pulls that toward ~22% if the pass takes many days). The best
   stacks found (ORB-trend NQ+GC, turn of month, night drift) reach
   **30-45%** -- their edge adds 5-15 points over their zero-edge twins.
2. **60-70%+ is only possible through one rule detail**: whether Lucid's MLL is
   checked against the **closed balance** (lenient) or against **open-position
   equity** (strict). Lucid's own text says only *"If your account balance
   reached the MLL, your account will be breached"*; several review sites say
   unrealized intraday swings do not breach LucidPro/LucidFlex (unlike
   LucidDaily). Not confirmed by Lucid in writing.
3. **The target race** -- long NQ from 19:00 ET, a limit order at the exact
   target, no stop, flat at 16:00 ET -- is the best choice under **both**
   readings:

| LucidPro 25K, 2 NQ (= 20 MNQ) | lenient reading | strict reading |
|---|---|---|
| pass rate (2019-26 proxy) | **85%** | 44% |
| passed on day 1 | 80% | 43% |
| failed evaluations per pass | **0.18** | 1.28 |
| real MNQ futures Jun 2025 - Aug 2026 | **88%** | 41% |

Under the strict reading nothing else does better (ORB-trend 20 MNQ 43%, old
6+3 trial 36%), and 1.3 failed evaluations per pass is inside the owner's 1-2
budget. Under the lenient reading it is ~5x fewer failed evaluations.

## 1. Edge stacks (strict reading, LucidPro 50K)

| Stack | pass | zero-edge twin | failed evals / pass |
|---|---|---|---|
| ORB NQ+GC 6+3 (old trial) | 30% | 20% | 2.34 |
| ORB + turn of month + night x3 | 31% | 18% | 2.18 |
| NQ ORB-trend 20 MNQ | 38% | 33% | 1.63 |
| NQ RTH race 20 MNQ | 40% | 37% | 1.51 |

Legs tested (per micro, 2016-26): ORB-trend NQ +$9.98/trade (t 1.55), GC
+$7.49 (t 2.08) -- reproduces the earlier research; turn of month held one
Lucid day at a time NQ +$22/day (t 1.3); NQ night 19:00-09:30 +$7.5 (t 1.5,
faded after 2021). Annualised Sharpe per leg 0.2-0.6; a 60-70% pass by edge
alone would need a portfolio Sharpe of ~2+ **and** months per evaluation.
The hour-of-day scan's big 18:00 effects in FX/gold are a bid-price artifact of
the 17:00 rollover spread (opposite signs on inverted pairs) -- not tradeable.

## 2. The three readings

| Reading | MLL checked on | DLL (50K, $1,200) |
|---|---|---|
| strict | open equity, every tick | flattens on open P&L |
| middle | closed balance | flattens on open P&L |
| lenient | closed balance | realised only |

**LucidPro 25K has no DLL** (official DLL page), so only the MLL question
matters there. On 50K the DLL is a second unknown: under the middle reading
the 50K race is capped at ~46%. **25K is the robust choice.**

## 3. The race

Rules: at 19:00 ET (start of the Lucid day's night session) buy 2 NQ (or 20
MNQ); place a sell limit at the price where the account reaches the target
(+$1,250 on 25K, net of costs, plus a tick); no stop; exit at 16:00 ET if not
filled. Next day repeat with the new distance to the target.

* **Why it works (lenient reading):** NQ moves 31 points in our favour at some
  point between 19:00 and 16:00 on ~90% of days (real futures 0.90, proxy
  0.90, same outcome on 99% of 282 days). Losses only count at the close.
  This is a rule asymmetry, not market edge: the short version passes 69%.
* **Year by year** (lenient / strict): 2018 83/61, 2019 75/47, 2020 86/50,
  2021 84/44, 2022 87/38, 2023 86/42, 2024 87/43, 2025 89/43, 2026 95/48.
* **Stress** (25K, lenient / strict): costs x2 + limit fills 4 ticks through
  84/42; entry 18:15-02:00 all 84-85 / 43-45; exit 15:15 85/44; 15 MNQ
  82/43; 10 MNQ 76/45. A plateau, not a peak.
* **50K** (40 MNQ): 83% lenient -- only if the DLL is realised-only too.
* **Weak spots:** a calm market (2016-17 style) lowers it (~55-65%); the
  failed attempts are large closed losses (irrelevant on a simulated
  evaluation, fatal on a funded account -- the race is for the evaluation
  only); Lucid could change the rule or its enforcement.
* **Allowed?** Lucid's published bans: cross-account hedging, HFT,
  microscalping (<= 5 s), exploiting platform errors, news windows on
  LucidDaily. Nothing on trade size inside the limits, stops or one-day
  passes; LucidPro evaluations have no consistency rule and no minimum days.

## 4. LucidPro 50K -- the owner's choice (deployed 2026-09-30)

On 50K the $1,200 DLL is a second unknown. Long NQ, limit at the target, far
safety stop, out 16:00 ET; 2019-26 proxy (real MNQ Jun 2025 - Aug 2026):

| Size, entry (ET) | both limits on closed balance | DLL closes on open P&L | MLL on open equity |
|---|---|---|---|
| **4 NQ, 19:00** | **83%** (87%), 77% day 1, 0.21 failed/pass | 43% | 39% (40%) |
| 3 NQ, 19:00 | 79% (84%) | 45% | 40% (41%) |
| 2 NQ, 19:00 | 73% (78%) | 46% | 42% (45%) |
| 4 NQ, 08:00 (14:00 Prague) | 80% (86%) | 38% | 34% (38%) |
| 4 NQ, 09:30 | 78% (82%) | 40% | 36% (20%) |

A safety stop costs nothing at 400 NQ points (82.7% vs 82.9% with none) but
~16 points at 75 (-$6,000) -- stops are what turn an open loss into a closed
one. **Deployed** (`scripts/run_lucidpro_trial.py`, `metatraderbot/target_race.py`):
4 NQ, entries 19:00 -> 12:00 ET (a late start still enters that day), exit
16:00 ET, safety stop 400 points, one race per trading day; `LUCID_RULE_READING`
(lenient/strict) sets how the Sim101 attempts judge the limits.

## 5. Before buying -- ask Lucid support

> 1. **Max Loss Limit ($2,000):** If an open position is temporarily down more
>    than the MLL during the session (unrealized), but it recovers and I close it
>    above the limit before the end of the day, is my account breached? Is the
>    MLL checked against open-position (unrealized) equity in real time, or only
>    against my closed (realized) balance?
> 2. **Daily Loss Limit ($1,200):** Does the DLL count unrealized P&L of open
>    positions, or only closed (realized) losses? When the DLL is reached, are my
>    open positions automatically closed, or am I only blocked from opening new
>    trades?

**Answer 2026-09-30 (Lucid's AI support agent, not a human):** MLL -- a trade
that goes below the limit while open but is closed above it before the day
ends is not a breach; the closing balance is what counts. DLL -- "restricted
from placing new trades until the next session"; the documentation does not
say whether it uses realized or unrealized P&L. Still open: a human's
confirmation, and whether an unrealized DLL hit closes the position or cancels
its take-profit. 4 NQ stays: it beats 2 NQ unless the chance that both limits
use the closed balance is below ~23%.

Both "closed balance only" -> race at 4 NQ (~83%). DLL on open P&L -> 2 NQ
(~46%). MLL on open equity -> 2 NQ (~42%). The race is the best option found
in every case; 60-70% needs the first answer.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\lucid_pass_study.py
```

Sources: [LucidPro Drawdown](https://support.lucidtrading.com/en/articles/12890136-lucidpro-drawdown),
[LucidPro Daily Loss Limit](https://support.lucidtrading.com/en/articles/12890122-lucidpro-daily-loss-limit),
[LucidDaily Drawdown](https://support.lucidtrading.com/en/articles/15998425-luciddaily-drawdown),
[tradetanto rules summary](https://tradetanto.com/learn/lucid-trading-rules-explained-every-plan-rule-and-limit),
[proptradingvibes DLL article](https://www.proptradingvibes.com/blog/lucidpro-daily-loss-limit),
[proptradingvibes prohibited strategies](https://proptradingvibes.com/blog/lucid-trading-prohibited-trading-strategies).

## 6. Machine learning check (2026-10-01) -- `scripts/lucid_ml_race.py`

Question: can a model pick the winning side (long/short) or the days to skip,
and lift the race above ~41%? Method per the ml-pipeline skill: features known
at 19:00 ET only (prior-day returns/range, trend, volatility, 16:00-19:00 gap,
day of week, month position, ES/GC/CL moves), walk-forward by year with a
7-day embargo, logistic regression and gradient boosting, 2020-26 out of sample.

| | result |
|---|---|
| Predictive power (AUC; 0.50 = coin flip) | 0.48-0.51 |
| Best ML rule, full evaluation replay | 33.7% pass, 1.97 failed evals/pass, 13 days |
| Current bot (always long), same replay | 39.9% pass, 1.50 failed evals/pass, 1 day |
| Perfect foresight of the winning side | 77.5% (the theoretical ceiling) |

No edge: NQ's direction over the race is not predictable from this information,
so ML cannot move the pass rate; 70% would need near-perfect foresight.

## 7. Funded account (2026-10-01) -- `scripts/lucid_funded_study.py`

LucidPro 50K funded rules (FundedRules): equity MLL $2,000 trailing the best
close, locked at $50,100 from a $52,100 close; payout 1 cap $2,000 then $2,500
(5 payouts, then live review); $52,100 must remain; per cycle >= $500, >= 3
days, largest winning day <= 40% of the cycle; 90/10 split. The first $2,000
payout therefore needs the account at $54,100. Rolling starts 2019-26, 120-day
horizon, eval cost per funded account ~$358 at list prices (41% pass, resets).

| Strategy (per funded account) | first $2,000 payout | blown first | median days | avg paid (90%) | $/week per account slot* |
|---|---|---|---|---|---|
| **race 1 NQ, take $1,400/day** | **25%** | 75% | **3** | **$854** | **$293** |
| race 1 NQ, take $1,000/day, any payout >= $500 | 32% (any payout) | 68% | 3 | $779 | $260 |
| race 2 NQ, take $1,400/day | 22% | 78% | 3 | $705 | $259 |
| ORB 2 MNQ + 1 MGC (current funded bot) | 24% | 50% | 70 | $640 | $16 |
| ORB 6 MNQ + 3 MGC | 20% | 78% | 26 | $713 | $44 |

*after evaluation costs, counting the evaluation time and the funded account's life.
The ceiling at zero edge is ~33% ($2,000 of room against $4,100 to make);
the 40% consistency rule forces >= 3 winning days. A low blow rate and a fast
$2,000 payout cannot both be had; ~$1,000/week means ~3-4 accounts running in
parallel (Lucid allows 10), with large swings.

### 7b. The balance the owner asked for: a $500+ withdrawal fast (`scripts/lucid_funded_balance.py`)

376 configurations (race sizes 0.5-2 NQ, entries 19:00/02:00/09:30, daily
take-profit caps $700/$1,040/$1,400, own stops, adaptive take-profit + pause,
payout plans $500/$1,000 then $500/$1,000/$2,500, race + ORB stacks).

| Choice | $500+ payout within 10 days | ever | blown before payout | median days | avg paid (90%) | $/week/account |
|---|---|---|---|---|---|---|
| **race 1 NQ @19:00, take what is still needed (max $1,040/day), pause once met; first $500, later $2,500** | **36%** | 37% | 63% | **3** | $818 | $280 |
| same, max $1,400/day; $1,000 payouts | 30% | 31% | 69% | 3 | $912 | $350 |
| stack race + ORB 2 MNQ + 1 MGC | 36% | 36% | 64% | 3 | $806 | $274 |

The zero-edge ceiling for a first $500 payout is ~43% ($2,000 of room
against the $2,600 needed); the adaptive race reaches 37%. ORB adds nothing.

### 7c. Big search + multi-day swings (2026-10-01) -- `lucid_funded_big.py`, `lucid_funded_swing.py`

* **3,088 race configurations** (NQ/ES/GC x long/trend x 8 entry times x sizes x
  daily caps x intraday profit lock x day filters + NQ+GC two-market races),
  ranked on 2019-22 starts, judged on 2023-26 starts: the in-sample leaders
  (39-40% within 10 days) fall to 30-37% out of sample. **Nothing beats the
  current pick** (race 1 NQ @19:00, take <= $1,040, adaptive + pause): 38% /
  34% within 10 days, best out-of-sample $/week ($290).
* **Multi-day swings made Lucid-legal by daily re-entry** (turn of month,
  RSI(2) pullback, 20-day breakout on NQ/ES/GC, alone and stacked with ORB):
  daily re-entry costs leave Sharpe 0.0-0.6 per leg; best funded stack ~$100/week
  per account out of sample, 30-34% payout, 66% blown -- worse than the race.
* The old MT5 funded bot's figures (2-8% blown over 4-26 weeks, $117-500/week)
  were backtests of 8 CFD legs, half of them Bitcoin/JP225, held for days on
  daily/4h charts -- none of it tradeable on Lucid; its one live demo run
  (Aug 14) blew on Aug 28 with a -$6,785 day and no payout.

### 7d. The whole strategy library under the real rules (2026-10-01) -- `scripts/lucid_library_funded.py`

17 families / 38 parameter sets (Donchian, MACD, RSI, Bollinger, Supertrend,
Stochastic, EMA/SMA cross, ORB, London breakout, VWAP, sweep, time-of-day,
Asian fade, S/R retest, BOS pullback, ORB-trend) x NQ/ES/YM/GC/CL/6E/6B/6J x
15m/1h x 1R/2R x $250/$500/$1,000 risk = **3,600 versions**, every trade closed
by Lucid's daily flat rule, through the funded and the evaluation replays;
ranked on 2019-22, judged on 2023-26.

* **Evaluation:** no version beats the race. In-sample leaders (46-53%) fall to
  7-25% on new data; the best stable one (ORB 1h NQ, 2R) is 41% / 41% -- equal
  to the race.
* **Funded, fast $500 payout:** best library version 23% within 10 days on new
  data (race: 36%); most versions lose money after evaluation costs.
* **Funded, lower blow rate:** ORB 15m NQ (30-min range from 13:00 UTC), 2R,
  $500 risk: 44% get a payout, 56% blown, but only 17% within 10 days and
  ~$190/week vs the race's ~$280.

Bug caught before reporting: the index proxies end at 16:15 ET, so the first
run held trades overnight (Lucid flattens at 16:45); every trade now closes at
the day's last bar.

### 7e. Built into the funded bot (2026-10-01) -- `scripts/run_lucid_futures_funded.py`

The 7b pick replaced the ORB + trend roster (kept as
`scripts/run_lucid_futures_funded_roster.py`). Once per Lucid trading day from
19:00 ET (entries until 12:00 ET): 1 NQ long, take-profit = what the next
payout still needs ($150-$1,040), safety stop 400 pts, out 16:00 ET; no race
once the payout money is met (wait out the 3-day minimum). Payout at the day's
close when Lucid's rules allow: >= 3 trading days, >= $500 cycle profit, best
day <= 40%, $52,100 kept; $500 first, then $2,500 (caps $2,000 / $2,500).
MLL on open equity every poll and against 1-minute history (sleep-proof);
a simulator take-profit fill is re-scored as a breach if the line was touched
first (also added to the trial bot). On the simulator (Sim102) payouts are
virtual and a blown account is replaced at the next session; a real account
halts. Dashboard FUNDED page rebuilt like the TRIAL page.
Tests: `tests/test_funded_race.py` (10, end-to-end on a fake NinjaTrader).
Runs on NinjaTrader's own paper account DEMO9384168 ($50,000; NinjaTrader 8.1 hides
local Sim accounts unless multi-provider mode is on, so Sim102 could not be added
without that setting); nt_engine treats DEMO* accounts as paper. Order path checked
with a 1-MNQ round trip on 2026-10-01 (filled, bracket placed, closed).
