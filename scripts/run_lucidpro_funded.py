"""LucidPro FUNDED-STAGE runner -- what to trade AFTER passing the eval,
live on the MT5 demo account. This is a completely different bot from
run_lucidpro_trial.py: that script deliberately maximizes variance (90%
risk/trade) to pass a free-retry evaluation as often as possible; passing
does not fund a strategy worth running afterward with real/simulated-real
money, which run_lucidpro_trial.py's own module docstring already says
explicitly (see scratch/lucidpro_edge_screen.py's docstring).

ROSTER (v5 BLEND, validated 2026-08-14, see scratch/lucidpro_portfolio_search_v4.py's
methodology plus the ad-hoc "consistency roster" + "blend" tests run directly
after it -- SUPERSEDES v4, which maximised $/week but decayed hard over time).

v4's all-wide-target Donchian/Supertrend roster hit $272-509/week, but its
PROFITABLE-after-N-weeks rate COLLAPSES over time (34% at 4wk -> 12% at 26wk
at 1.0% risk) because LucidPro's MLL is a TRAILING (ratcheting) floor: high
per-trade variance means an eventual bad stretch relative to some past peak
becomes near-certain the longer you run it, even with genuine positive edge
("gambler's ruin against a rising floor"). A separate CONSISTENCY roster
(tight 1R Donchian/Supertrend legs, 51-57% per-TRADE win rate, screened in
scratch/lucidpro_edge_screen_untested.csv) barely decays at all (53.5% at 4wk
-> 50.1% at 26wk) but only earns ~$82/week. This BLEND roster mixes 4 legs
from each family -- it keeps most of the consistency roster's durability
while adding real upside:

    CONSISTENCY legs (stability, higher per-trade win rate):
      Donchian    BTCUSD   1d  TP=1R   e_net +0.141R (n=112, win 56.2%)
      Donchian    NAS100   1d  TP=1R   e_net +0.116R (n=169, win 56.2%)
      Donchian    XAUUSD   1d  TP=1R   e_net +0.115R (n=134, win 56.0%)
      Supertrend  JP225    4h  TP=1R   e_net +0.093R (n=475, win 55.6%)
    AGGRESSIVE legs (upside, low per-trade win rate but big e_net):
      Donchian    XAUUSD   4h  TP=6R   e_net +0.645R (n=114, win 25.4%)
      Donchian    BTCUSD   4h  TP=3R   e_net +0.570R (n=144, win 40.3%)
      MACD        XAUUSD   1d  TP=2R   e_net +0.432R (n=103, win 48.5%)
      Ichimoku    BTCUSD   1d  TP=2R   e_net +0.429R (n=172, win 48.3%)

All 8 legs individually verified e_net > 0 at 5x modelled spread+commission
(this project's own measured live-slippage gap is 2-12x).

Economics at BASE_RISK_PCT=0.7% with the TUNED consistency-leg params above
(pooled 1320 trades / 1008 signal-days, $50k account) -- re-measured
2026-08-14 after the parameter sweep, supersedes the pre-tuning numbers:

    horizon   blown%   losing%   PROFITABLE%
    4wk         2.3%     42.6%       55.1%
    8wk         4.3%     40.7%       55.0%
    13wk        6.4%     38.5%       55.1%
    26wk        8.5%     36.6%       54.9%

  $117/week average, week-level win rate 56.1% (both up from the
  pre-tuning $83/wk and 54.6% -- see scratch/lucidpro_blend_param_sweep.py
  for the sweep and its both-halves-positive robustness checks). Durability
  is now essentially FLAT across every horizon (~55% profitable whether
  measured at 4 or 26 weeks) instead of slowly decaying -- 26wk blow risk
  rose slightly (6.3% -> 8.5%) but that's a small price for a result that
  no longer degrades with time. Compare to pure-aggressive v4 at 1.0% risk
  ($272/wk but only 34.4%/12.0% profitable at 4wk/26wk) -- this trades
  roughly 2.3x the $/week for staying reliably in the game, a better ratio
  than before tuning. 1.0% risk on this tuned blend reaches ~$165/week;
  1.5% reaches ~$239/week. There is still no setting that gets both high
  $/week AND high durability simultaneously -- that would need a
  fundamentally stronger edge than exists in this instrument/strategy
  universe, not a smarter allocation.

MLL LOCK CORRECTION (2026-08-14, scratch/lucidpro_mll_lock_taper.py): two
independent sources (tradetanto.com, damnpropfirms.com -- cross-checked,
same sources this project's earlier LucidPro research already used; Lucid's
own official help center blocks automated fetches with a 403, so this is
NOT primary-source-confirmed) agree the MLL trails upward only until equity
first clears the Initial Trail Balance ($52,100 on $50k = start + $2,000
MLL + $100 buffer), then LOCKS FLAT and never moves again -- it does not
keep ratcheting up forever the way every simulation above assumes. (Do not
confuse this with the separate Daily Loss Limit, which DOES convert to 60%
of peak EOD balance and keeps rising -- that's a soft, resets-daily rule,
not the account-ending one modelled here.)

Re-running the numbers above with the corrected lock mechanic: barely moves
the recommended 0.7% risk point (the lock rarely triggers early enough to
matter at that pace), but meaningfully improves the case for pushing
higher if you want more $/week:

    risk   horizon   OLD (trails forever)      NEW (locks flat)
    0.7%    26wk      6.3% blown / 50.6% profit    5.5% blown / 51.4% profit
    1.0%    26wk     23.8% blown / 43.7% profit   19.3% blown / 48.1% profit
    1.5%    26wk     55.0% blown / 28.0% profit   41.9% blown / 41.2% profit

Real improvement, especially at higher risk, but not transformative --
1.5% risk still blows 42% of the time within a year even under the more
forgiving mechanic. Softens but does not overturn the "patience doesn't
help" conclusion. Re-verify against Lucid's own terms before trusting this
for real capital.

RULES MODELED: same LucidPro MLL as the eval (-4% TRAILING off the peak of
each day's EOD close, checked ONLY at end-of-day) -- ASSUMED to also apply
post-funding; not independently re-confirmed for the funded phase specifically,
flag and correct if real account terms disagree. UNLIKE the eval trial, there
is NO target and NO restart-on-breach: an MLL breach on a funded account is a
real adverse event (the firm would revoke funding), so this script CLOSES ALL
POSITIONS AND HALTS on breach rather than opening a fresh attempt. Restarting
requires a human to look at what happened first.

RULES MODELED: same LucidPro MLL as the eval (-4% TRAILING off the peak of
each day's EOD close, checked ONLY at end-of-day) -- ASSUMED to also apply
post-funding; not independently re-confirmed for the funded phase specifically,
flag and correct if real account terms disagree. UNLIKE the eval trial, there
is NO target and NO restart-on-breach: an MLL breach on a funded account is a
real adverse event (the firm would revoke funding), so this script CLOSES ALL
POSITIONS AND HALTS on breach rather than opening a fresh attempt. Restarting
requires a human to look at what happened first.

ADAPTIVE RISK TAPER (scratch/lucidpro_adaptive_taper.py's "loss_only" mode,
validated parameters BAND=0.04 FLOOR=0.25): today's new-entry risk shrinks as
equity nears the trailing MLL floor (peak - MLL) instead of staying flat --
there is no upper "target" once funded, only the floor to avoid.

    dist_to_mll = eq_prev_close - (peak - MLL)
    frac        = clamp(dist_to_mll / BAND, FLOOR, 1.0)
    today_risk  = BASE_RISK_USD * frac

Paper/demo only (MetaQuotes-Demo). Not connected to a real Lucid account.

Usage:
    .venv/Scripts/python.exe scripts/run_lucidpro_funded.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # picks up TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID for metatraderbot.alerts

from metatraderbot.alerts import alert  # noqa: E402
from metatraderbot.instruments import contract_mult  # noqa: E402
from metatraderbot.mt5_engine import MT5Engine  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "lucidpro_funded.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("lucidpro_funded")

# --- Roster: (label, symbol, timeframe, strategy_cls, params, tp_r_multiple) --
# 4 CONSISTENCY legs (tight 1R, higher per-trade win rate, low variance) +
# 4 AGGRESSIVE legs (wide 3-6R, low per-trade win rate, high e_net) -- see
# the module docstring's "v5 BLEND" section for the full reasoning and the
# durability-over-time data that justifies mixing both variance profiles
# instead of running either alone.
#
# Consistency-leg parameters TUNED 2026-08-14 (scratch/lucidpro_blend_param_sweep.py):
# every non-default value below cleared BOTH-HALVES-POSITIVE + n>=100 --
# library defaults did not always clear that bar (see XAUUSD below).
#   Donchian/BTCUSD:  period 20 (default) -> 25   (e_net +0.141 -> +0.231)
#   Donchian/NAS100:  period 20 (default) -> 40   (e_net +0.116 -> +0.188)
#   Donchian/XAUUSD:  period 20 (default) -> 10   -- NOT an e_net win (0.115 -> 0.112),
#                      a ROBUSTNESS FIX: period=20's first-half e_net is -0.199 (negative!),
#                      masked by a +0.429 second half -- every period from 15-40 shows this
#                      same sample-collapse pattern, only period=10 is genuinely both-halves-positive.
#   Supertrend/JP225: period=10/mult=3.0 (default) -> period=14/mult=3.5 (e_net +0.093 -> +0.135, n=349)
# Aggressive legs kept at library defaults -- the sweep found no robust
# (both-halves-positive, n>=100) improvement over them.
# 2026-08-20 -- the three BTCUSD legs are DISABLED, not deleted.
#
# This broker (MetaQuotes-Demo) carries no crypto CFDs at all: its only symbol
# groups are Forex, Indexes, Metals and Nasdaq. The symbol_map entry
# "BTCUSD" -> "BTC" therefore resolved to *Grayscale Bitcoin Mini Trust*, a
# ~$30/share Nasdaq ETF that trades NYSE hours -- not 24/7 spot Bitcoin. Every
# BTCUSD leg was feeding Bitcoin-derived Donchian/Ichimoku parameters (tuned on
# real BTCUSD in scratch/lucidpro_portfolio_search_v4.py) into an equity ETF
# with a completely different price scale, session, and volatility. Once the
# order_send bugs were fixed on 2026-08-20 these legs started producing real
# broker rejections (retcode=10014 "Invalid volume" -- a 0.7%-risk position
# sized against a $30 instrument asks for ~130 lots).
#
# Re-enable ONLY on a broker that actually lists spot/CFD BTCUSD. Restoring
# them here without that changes nothing about the underlying mismap.
DISABLED_LEGS = [
    ("Donchian_BTCUSD_1R",  "BTCUSD", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 25}}, 1.0),
    ("Donchian_BTCUSD_3R",  "BTCUSD", "4h", DonchianBreakoutStrategy, {"donchian": {}},   3.0),
    ("Ichimoku_BTCUSD_2R",  "BTCUSD", "1d", IchimokuStrategy,        {"ichimoku": {}},   2.0),
]

# --- v6 ONE-PAYOUT roster, selected 2026-08-22 -------------------------------
#
# Objective changed, so the roster changed with it. v5 was tuned to compound a
# single account over 26 weeks; the actual goal is ONE $2,000 payout per account
# across several accounts, under a $2,000 DAILY loss limit. That rewards a high
# per-trade win rate and low variance, not the fat right tail v5 was built for
# (its 6R gold leg won 24% of the time -- fine for compounding, far too swingy
# for reliably banking $2,000 once).
#
# Selected by screening 288 (strategy x symbol x timeframe) combinations over
# ~10 years -- scratch/_leg_screen.py -- keeping win rate >= 48%, positive
# expectancy after costs, n >= 100, then picking the strongest while spreading
# across uncorrelated markets. US500/NAS100/US30 are near-duplicates so only one
# index family is taken at the daily timeframe.
#
# Blended: 4,779 trades, 58.8% win rate, +0.158R expectancy, 8.64 trades/week
# (vs v5's ~39% blended win rate). Simulated at 0.5% risk under a $2,000 daily
# limit: 85.3% of accounts bank one $2,000 payout, median 10 trading days
# (scratch/_oneshot.py, 20k Monte Carlo runs per configuration).
#
# JP225 is deliberately ABSENT: broker minimum lot is 1.0 at $100/point, so its
# smallest possible position risks ~$90,000 on a $50k account. It could never
# fire and every signal was being skipped.
V6_ROSTER = [
    # -- RETIRED 2026-08-25, superseded by v7. Kept as a named list so magics,
    # ledger rows and any position still open under them stay attributable.
    ("Donch55_US500_1d",  "US500",  "1d", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("Donch55_XAUUSD_1d", "XAUUSD", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("Donch55_CHFJPY_1d", "CHFJPY", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("Donch55_USDJPY_1d", "USDJPY", "1d", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("Donch55_NAS100_4h", "NAS100", "4h", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("Donch55_XAUUSD_4h", "XAUUSD", "4h", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 1.0),
    ("MACD_GBPJPY_4h",    "GBPJPY", "4h", MACDTrendStrategy,        {"macd": {}},                1.0),
    ("RSI_GBPUSD_4h",     "GBPUSD", "4h", RSIReversionStrategy,     {"rsi": {}},                 1.0),
]

# --- v7 FAST roster, selected 2026-08-25 -------------------------------------
#
# Objective changed again: hold time is now a hard constraint (3-5 days), not a
# free variable. v6 earns more per week (~$683 vs ~$198 at 1% risk) but its
# median hold is 22 days, so the money sits inside open positions where it
# cannot be withdrawn. v7 reaches the $2,000 payout in a median 16 days against
# v6's 35 precisely because it resolves.
#
# Selection, all at 2x broker-quoted costs over ~11 years:
#   * 2,880-cell screen (scratch/_swing35.py): 15 strategy families x 16 symbols
#     x 4h/1d x 3 targets, hard 120h cap. 59 of 833 valid configs passed every
#     gate -- against 0 of 216 for the same screen run intraday
#     (scratch/_daytrade.py), which is why this roster is 4h/1d and not faster.
#   * parameter-plateau test (scratch/_swing_tune.py): only MACD-trend and
#     Donchian+EMA had 100% of their parameter neighbourhood positive. Legs whose
#     neighbourhood was mostly negative are EXCLUDED as artifacts however good
#     their headline number -- see the RSI_GBPUSD_4h note below.
#   * drawdown cap and weekly-correlation gate (scratch/_final_roster.py):
#     per-leg max drawdown <= 25R, pairwise weekly correlation <= 0.45.
#
# Blended: 4,329 trades, 49.5% win, +0.0506R expectancy, 7.83 trades/week,
# portfolio max drawdown -35.2R. At 1% risk: $198/wk, $855/mo, 77.1% of accounts
# bank one $2,000 payout in a median 16 days, 16.2% breach the daily limit.
#
# NOTE the win rate is 49.5% and that is FINE. A 66.9%-win / 2.17-day-hold
# configuration exists (scripts/pairs_cointegration_test.py) and loses 0.047R
# per trade in both halves: win rate is bought with reward:risk and costs then
# tip it negative. These legs are ranked on expectancy, never on win rate.
#
# RSI_GBPUSD_4h is deliberately ABSENT despite being a v6 member. Its parameter
# surface is an artifact: only 43% of the neighbourhood is positive and its best
# cell (+0.366R) sits at period=25/TP=2.5R on n=150, climbing monotonically into
# the corner of the grid as sample size collapses. It carried -38.9R of drawdown
# for $9/week.
ROSTER = [
    # -- the anchor: strongest single leg found in any search, and the only one
    # confirmed independently twice (scratch/_fast3d.py and scratch/_swing35.py).
    # fast_period=10, NOT the library default of 12 nor the fast=8 retune
    # validated in August -- 30/30 of its neighbourhood is positive and 10 works
    # out best within it (+0.123R vs +0.100R at fast=8, +0.089R at fast=12).
    ("MACD10_GBPJPY_4h",  "GBPJPY", "4h", MACDTrendStrategy,
     {"macd": {"fast_period": 10, "slow_period": 26, "signal_period": 9}}, 1.5),
    ("SMAx_EURUSD_4h",    "EURUSD", "4h", SMACrossStrategy,
     {"ma_cross": {"fast_period": 10, "slow_period": 50}}, 1.5),
    ("Super_USDJPY_4h",   "USDJPY", "4h", SupertrendStrategy,
     {"supertrend": {"period": 10, "multiplier": 3.0}}, 1.5),
    ("MACD12_XAUUSD_4h",  "XAUUSD", "4h", MACDTrendStrategy,
     {"macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}, 2.0),
    ("Donch20e_XAUUSD_4h", "XAUUSD", "4h", DonchianBreakoutStrategy,
     {"donchian": {"period": 20, "trend_ema_period": 200}}, 1.0),
    ("MACD10_EURUSD_4h",  "EURUSD", "4h", MACDTrendStrategy,
     {"macd": {"fast_period": 10, "slow_period": 26, "signal_period": 9}}, 2.0),
    ("Boll_USDCHF_1d",    "USDCHF", "1d", BollingerBounceStrategy,
     {"bollinger": {"period": 20}}, 1.5),
]

VIRTUAL_ACCOUNT = 50_000.0

# Every leg whose positions may still be open, active or retired. Exit
# management (trailing stops, time stops) must iterate THIS, never ROSTER --
# retiring a leg does not close its positions.
MANAGED_LEGS = ROSTER + V6_ROSTER + DISABLED_LEGS

# --- Time stop: the constraint v7 is built around ----------------------------
#
# v7's measured statistics come from a backtest with a hard 120h cap
# (scratch/_swing35.py, _final_roster.py). Several legs report a median hold of
# exactly 5.00 days, meaning the cap -- not the take-profit -- is what closes
# them. Without the same rule live, those legs do not underperform the
# backtest; they become DIFFERENT STRATEGIES, running on into the multi-week
# trend trades v7 exists to avoid, with money locked in open positions that
# cannot be withdrawn.
#
# Applied to MANAGED_LEGS, so v6 positions still open at the changeover are
# also brought to an orderly close rather than being left to run untended.
MAX_HOLD_HOURS = 120.0
TIME_STOP_ENABLED = True

# --- DAILY loss limit, not a trailing lifetime drawdown ----------------------
#
# CORRECTED 2026-08-22. This runner previously enforced MLL_PCT = 0.04 as a
# TRAILING drawdown off the equity peak, checked at the end-of-day close, which
# permanently halted the account when breached. That is the wrong rule: the
# account's actual constraint is a $2,000 loss measured WITHIN A SINGLE DAY,
# which resets at rollover. The two behave nothing alike -- a trailing floor
# makes profit raise the bar you must not fall back through, so a long slow
# grind up can be killed by a drawdown the real rules would never notice.
#
# Breaching the daily limit fails the account, so this halts and alerts rather
# than pausing. Entries stop at DAILY_STOP_FRACTION of the limit so a position
# already open cannot carry the day past the line while the bot sits idle.
DAILY_LOSS_LIMIT = 2_000.0
DAILY_STOP_FRACTION = 0.75      # stop opening at -$1,500, leaving room for open risk
PAYOUT_TARGET = 2_000.0         # alert once the account can withdraw

MLL_PCT = 0.04            # retained only for the legacy peak_equity bookkeeping in state

# 1.0% = $500/trade, set 2026-08-22 as a deliberate speed/certainty trade.
#
# Measured in scratch/_payout_sim.py -- a CALENDAR-TIME simulation for "one
# $2,000 payout under a $2,000 daily limit". It supersedes the earlier
# scratch/_oneshot.py sweep, which was wrong: that model sampled R-multiples by
# EXIT date and so assumed a trade resolves inside the sampled window. This
# roster's median hold is 22 days, so profit that looked weekly was actually
# locked up for months, and its "~10 days to payout" was unreachable.
#
#   risk    P(payout)   median days   E[5 accounts]
#   0.5%      99.7%          70          $9,967
#   1.0%      89.7%          35          $8,967   <- chosen
#   2.0%      51.0%          16          $5,100
#   3.0%      63.8%          11          $6,383
#   5.0%      59.7%           5          $5,967
#
# 1.0% halves time-to-payout for ~10% of the expected take. Below it you pay a
# lot of time for little extra certainty; above 3% real money burns for days.
#
# Note the NON-MONOTONICITY -- 2.0% (51.0%) is worse than 3.0% (63.8%). That is
# the daily cap, not noise: at $1,000/trade exactly TWO losses reach $2,000 and
# end the account, while $1,500 lets one through with room to spare. Sizing has
# to be checked against "how many losses fit under the cap", never chosen by
# feel.
#
# Scaling to more accounts does not change this number: the accounts are
# independent and run in parallel, so per-account odds and timing hold.
# --- RAISED to 1.5% on 2026-08-25 with the v7 roster ------------------------
#
# The sweep above sized against an UNCAPPED book. The live runner caps open risk
# at MAX_PORTFOLIO_RISK_PCT (3% = $1,500), so per-trade risk also sets how many
# positions may be open at once -- $1,500 / $750 = TWO at this level, against
# three at 1.0%. That interaction was missed until v7 went live and the runner
# immediately logged "Risk budget full: $1,919 open vs $1,500 cap".
#
# Measured on the v7 signal stream with the cap enforced exactly as the runner
# enforces it (scratch/_concurrency.py, 5,000-run calendar simulation each):
#
#   risk   slots  skipped   e@2x     $/wk   payout   blow
#   0.50%    6      14%    +0.058     $97    73.0%   0.0%
#   0.75%    4      34%    +0.069    $134    80.5%   0.0%
#   1.00%    3      47%    +0.080    $165    84.3%   1.1%
#   1.50%    2      63%    +0.087    $189    85.1%   1.0%   <- chosen
#   2.00%    1      81%    +0.061     $93    74.7%   5.2%
#   3.00%    1      81%    +0.061    $139    80.4%   4.7%
#
# Two things to read out of that. First, the cap is a FILTER, not a tax:
# expectancy per trade RISES from +0.051 uncapped to +0.087 at two slots,
# because the entries it blocks arrive when the book is already crowded and
# crowded periods are exactly when correlated legs lose together. Blow-up
# probability falls from 15.9% uncapped to 1.0%.
#
# Second, it turns over. At one slot 81% of signals are skipped, the filter
# stops being selective and starts discarding good trades, and expectancy drops
# back to +0.061. The optimum is interior -- two concurrent positions.
BASE_RISK_PCT = 0.015
TAPER_BAND = 0.04
TAPER_FLOOR = 0.25

# Ratchet trailing stop (see apply_trailing_stops). START at +0.5R rather than
# the +1.0R used by portfolio_runner/backtest.py: four of this roster's legs
# target exactly 1R, so a 1.0R trigger could never fire on them -- the take
# profit would always be hit first. 0.5R/0.5 means a trade is protected at
# +0.25R by the time it is halfway to a 1R target, and a 6R runner keeps
# ratcheting up behind price all the way out.
#
# NOTE this is a live change to exit behaviour, not a pure addition: trades
# that would previously have round-tripped from +0.6R back to -1R now exit
# somewhere positive, but trades that dip through the trail before resuming
# are stopped out where they previously survived. Net effect is bounded by
# how often price retraces past half the reached profit.
#
# MEASURED 2026-08-21 on 1870 trades / 10.6y of XAUUSD 4h (the 6R leg, which
# has the most to lose from a badly-set trail). Trailing ALWAYS costs
# expectancy here -- it clips the right tail this roster's edge depends on --
# so the setting is a deliberate trade of profit for fewer full-stop losses:
#
#   setting          win%   full-SL%   e_net     vs no-trail
#   none             24.0%    75.2%    +0.600      baseline
#   0.5R / 0.5       65.9%    33.7%    +0.153      -75%  <- too greedy
#   1.0R / 0.33      51.1%    48.3%    +0.340      -43%
#   2.0R / 0.33      38.4%    60.8%    +0.427      -29%  <- chosen
#
# 2.0R/0.33 keeps 71% of the untrailed edge while cutting full-stop losses
# from 75.2% to 60.8% and lifting win rate 24% -> 38%. Starting at 2R also
# means the four 1R-target legs are untouched (they hit TP first), so this
# only affects the wide-target legs where round-tripping actually hurts.
# Do NOT lower the trigger to "win more often" -- that is the 0.5R row.
#   4.0R / 0.80      28.2%    71.0%    +0.478      -20%  <- best tested
#
# TIERED, measured 2026-08-21 on the same 1870 trades. Locking a big share of
# profit is only affordable once a trade is genuinely large: at +2R an 80%
# lock costs 61% of the edge (+0.233), while at +4R the same 80% lock costs
# just 20% (+0.478) and beats every gentler setting. The reason is that early
# trailing strangles the runners this roster's expectancy depends on, whereas
# a trade already +4R has proven itself and is worth protecting hard.
#
# Tier 1 keeps modest protection in the 2R-4R band; tier 2 implements the
# "up $140, lock $100" rule directly -- at ~$31 of real risk per position,
# +4R is ~$124 and an 80% lock leaves ~$99 banked.
#
# NOTE the two tiers are individually measured; the combination is not (the
# backtest engine's trail takes a single start/fraction). Expect it to land
# between +0.427 and +0.478 -- better protection than 4R/0.8 alone, slightly
# less expectancy. Re-measure before treating the blend as validated.
# DISABLED for the v6 roster (2026-08-22). Every leg now targets exactly 1R, so
# a trail starting at +2R can never fire before the take-profit does -- it would
# be dead code that only added risk of an unmeasured interaction. More
# importantly, the 85.3% payout figure this roster was selected on was simulated
# WITHOUT a trail; leaving one active would mean running a configuration nobody
# measured. Re-enable only alongside a fresh simulation.
TRAIL_TIERS: list[tuple[float, float]] = []
# 2026-08-21 -- concurrency is no longer capped by COUNT. The backtest that
# validated this roster ran with unlimited concurrency and sat at a median of
# 40 open positions (>=6 open 99.5% of the time): a 6R target takes weeks to
# resolve, so entries stack by design. Capping at 2/leg made the live bot
# trade a materially different -- and much thinner -- strategy than the one
# the edge was measured on.
#
# What actually needs bounding is RISK, not trade count. Twenty positions at
# $30 of risk is a 1.2% exposure; two at $500 is 2%. So the cap is now the
# sum of open risk against the MLL budget. New entries stop once open risk
# would exceed MAX_PORTFOLIO_RISK_PCT of the anchor; existing trades are
# never touched by it.
#
# 3.0% sits deliberately below the 4% MLL: if every open position stopped out
# at once the account survives, which a count-based cap could never promise.
MAX_CONCURRENT_PER_LEG = 999      # effectively unlimited; risk cap governs
MAX_PORTFOLIO_RISK_PCT = 0.03
EOD_HOUR_UTC = 22         # matches run_lucidpro_trial.py's assumption
MAGIC_BASE = 9200         # first unused block after eval trial's 9100s

# Magic numbers are PINNED PER LABEL, never derived from the roster's index.
# They were index-derived until 2026-08-20, which meant disabling the BTCUSD
# legs silently renumbered every leg after them -- re-pointing magic 9202 from
# Donchian_XAUUSD_1R to Supertrend_JP225_1R and orphaning the live position
# already open under it (ticket 10124062219). A magic is the only durable link
# between an open broker position and the strategy that owns it, so these
# values must stay stable for as long as any position or ledger row references
# them. Add new legs with new numbers; never reuse or shift an existing one.
LEG_MAGICS = {
    # --- v7 roster (active, 2026-08-25). Again new numbers: XAUUSD positions
    # opened by v6's Donch55_XAUUSD_4h under 9215 are still open at the time of
    # this change, and v7 has its own XAUUSD legs. Reusing a number would make
    # those indistinguishable and let the trail/time-stop manage them with the
    # wrong leg's take-profit geometry.
    "MACD10_GBPJPY_4h":    9220,
    "SMAx_EURUSD_4h":      9221,
    "Super_USDJPY_4h":     9222,
    "MACD12_XAUUSD_4h":    9223,
    "Donch20e_XAUUSD_4h":  9224,
    "MACD10_EURUSD_4h":    9225,
    "Boll_USDCHF_1d":      9226,
    # --- v6 roster (retired 2026-08-25) -- kept so open positions and history
    # opened under these magics still resolve to a name.
    "Donch55_US500_1d":    9210,
    "Donch55_XAUUSD_1d":   9211,
    "Donch55_CHFJPY_1d":   9212,
    "Donch55_USDJPY_1d":   9213,
    "Donch55_NAS100_4h":   9214,
    "Donch55_XAUUSD_4h":   9215,
    "MACD_GBPJPY_4h":      9216,
    "RSI_GBPUSD_4h":       9217,
    # --- v5 roster (retired 2026-08-22) -- kept so open positions and history
    # opened under these magics still resolve to a name.
    "Donchian_BTCUSD_1R":  9200,  # disabled (see DISABLED_LEGS)
    "Donchian_NAS100_1R":  9201,
    "Donchian_XAUUSD_1R":  9202,
    "Supertrend_JP225_1R": 9203,
    "Donchian_XAUUSD_6R":  9204,
    "Donchian_BTCUSD_3R":  9205,  # disabled
    "MACD_XAUUSD_2R":      9206,
    "Ichimoku_BTCUSD_2R":  9207,  # disabled
}
POLL_INTERVAL_S = 30      # 1d/4h bars -- no need to poll as tight as the 15m eval trial
STATE_PATH = ROOT / "data" / "state_lucidpro_funded.json"
TRADES_LOG_PATH = ROOT / "data" / "lucidpro_funded_trades.jsonl"
BAR_COUNT = 300

# Self-watchdog: 2026-08-17 -- found the process silently hung for ~22h with
# zero further log output (still "running" per Windows, MT5 IPC calls like
# copy_rates_from_pos apparently blocked forever after a connectivity blip
# instead of raising). The external 15-min watchdog.ps1 only checks whether
# the process EXISTS, so a hung-but-alive process passes that check forever
# and never gets restarted -- exactly what happened. Fix: a background
# thread that force-kills this process if the main loop stops making
# progress, so the external watchdog (or a human) gets a dead process to
# restart within minutes instead of a hung one that looks fine.
_HEARTBEAT_TIMEOUT_S = 180  # 6x POLL_INTERVAL_S -- generous vs. normal cadence, far below "hours"
_last_heartbeat = time.monotonic()
_heartbeat_lock = threading.Lock()


def _touch_heartbeat() -> None:
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.monotonic()


def _self_watchdog() -> None:
    while True:
        time.sleep(30)
        with _heartbeat_lock:
            stale_s = time.monotonic() - _last_heartbeat
        if stale_s > _HEARTBEAT_TIMEOUT_S:
            log.error(f"Self-watchdog: no loop progress for {stale_s:.0f}s (limit "
                      f"{_HEARTBEAT_TIMEOUT_S}s) -- likely stuck in a blocking MT5 call. "
                      f"Force-exiting so the external watchdog restarts a clean process.")
            os._exit(1)  # nuclear on purpose: a truly hung C-extension call can't be interrupted any other way


def log_trade(sym: str, label: str, direction: str, entry_price: float, exit_price: float,
              lots: float, risk_usd: float, r_multiple: float, opened_ts: str, closed_ts: str) -> None:
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label, "sym": sym, "direction": direction,
        "entry_price": entry_price, "exit_price": exit_price, "lots": lots,
        "risk_usd": risk_usd, "r_multiple": round(r_multiple, 4),
        "pnl_usd": round(r_multiple * risk_usd, 2),
        "opened_ts": opened_ts, "closed_ts": closed_ts,
    }
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_state() -> dict:
    defaults = {
        "peak_equity": VIRTUAL_ACCOUNT,
        "realized_usd": 0.0,
        "current_day": None,
        "halted": False,
        "halt_reason": None,
        "total_trades": 0, "total_wins": 0, "total_losses": 0,
        "started_ts": None,
        # Account-run history. A daily-limit breach no longer halts terminally --
        # it blows THIS virtual account, records the run, and starts a fresh one,
        # so the dashboard can show how often the roster survives to a payout
        # instead of a single halt reason.
        "runs": [], "runs_total": 0, "runs_blown": 0, "runs_paid": 0,
        "run_start": None,
    }
    if STATE_PATH.exists():
        try:
            on_disk = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            defaults.update(on_disk)
        except (OSError, ValueError):
            log.error("State file unreadable, starting fresh")
    return defaults


def save_state(state: dict) -> None:
    # 2026-08-18 -- this whole project lives under OneDrive, which syncs the
    # data/ dir constantly given how often this writes; its sync client
    # transiently locks files mid-sync, causing os.replace() to raise
    # PermissionError (WinError 5) with no fault of ours. Retry briefly
    # instead of letting a normal, expected OneDrive lock blip cost a whole
    # loop iteration (or worse, cascade into the self-watchdog's stale-loop
    # detection during a multi-day unattended run).
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, STATE_PATH)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def day_key(now: datetime) -> str:
    shifted = now
    if now.hour < EOD_HOUR_UTC:
        shifted = now.replace(hour=0) - __import__("datetime").timedelta(days=1)
    return shifted.strftime("%Y-%m-%d")


def engine_positions_by_ticket(engine: MT5Engine, ticket: int):
    import MetaTrader5 as mt5
    positions = mt5.positions_get(ticket=ticket)
    return positions[0] if positions else None


def reconcile_closed_tickets(tickets: dict, engine: MT5Engine, state: dict,
                              current_realized_usd: float) -> float:
    """A ticket that hit its SL/TP naturally used to just vanish from
    `tickets` with no accounting: floating_usd() silently skips any ticket
    engine_positions_by_ticket can't find, so a mid-run SL/TP close
    previously contributed NOTHING to realized_usd -- a real win or loss
    disappeared from equity tracking, `total_wins`/`total_losses` in state
    were never incremented anywhere, and the dashboard's win-rate always
    showed 0/0. FIXED 2026-08-16: detect the close every poll via
    MT5Engine.get_closed_deal() (already built for exactly this), credit
    its REAL broker-reported P&L (commission+swap included, not the
    R-multiple approximation), and alert on it. Returns the realized delta
    to add to the caller's running realized_usd."""
    realized_delta = 0.0
    for ticket, info in list(tickets.items()):
        if engine_positions_by_ticket(engine, ticket) is not None:
            continue  # still open
        deal = engine.get_closed_deal(ticket)
        del tickets[ticket]
        if deal is None:
            log.warning(f"[{info['label']}] ticket {ticket} no longer open but no closed-deal "
                        f"record found -- dropping without crediting P&L (broker history gap)")
            continue
        pnl_usd = deal["pnl_usd"]
        r_multiple = pnl_usd / info["risk_usd"] if info["risk_usd"] else 0.0
        realized_delta += pnl_usd
        state["total_wins"] = state.get("total_wins", 0) + (1 if pnl_usd > 0 else 0)
        state["total_losses"] = state.get("total_losses", 0) + (1 if pnl_usd <= 0 else 0)
        log_trade(info["sym"], info["label"], info["direction"], info["entry_price"],
                  deal["exit_price"], info["lots"], info["risk_usd"], r_multiple,
                  info.get("opened_ts", ""), deal["closed_ts"].isoformat())
        total_trades = state.get("total_trades", 0)
        wins = state.get("total_wins", 0)
        win_rate = wins / total_trades if total_trades else 0.0
        equity = VIRTUAL_ACCOUNT + current_realized_usd + realized_delta
        pnl_pct = pnl_usd / VIRTUAL_ACCOUNT
        result_emoji = "🟢" if pnl_usd > 0 else "🔴"
        exit_emoji = {"tp": "🎯", "sl": "🛑"}.get(deal["exit_reason"], "✋")
        alert(
            f"{result_emoji} <b>CLOSED</b> {exit_emoji} <b>{deal['exit_reason'].upper()}</b> "
            f"{info['label']} {info['sym']} {info['direction']}\n"
            f"Profit: <b>${pnl_usd:+,.2f}</b> ({pnl_pct:+.2%} of account)  [{r_multiple:+.2f}R]\n"
            f"📊 Equity: ${equity:,.0f} ({(equity - VIRTUAL_ACCOUNT) / VIRTUAL_ACCOUNT:+.2%})\n"
            f"🏆 Lifetime: {total_trades} trades, {win_rate:.1%} win rate",
            book="LucidPro-Funded",
            html=True,
        )
    return realized_delta


def floating_usd(engine: MT5Engine, magics: dict, exclude=None) -> float:
    """Open P&L across this book, read from the BROKER by magic.

    REWRITTEN 2026-08-29. This used to walk the in-memory `tickets` dict, which
    starts EMPTY on every restart -- and the self-watchdog recycles this process
    routinely. The consequence was severe and silent: with four v6 gold
    positions open at -$6,529, floating read as $0.00, so equity read as a flat
    $50,000, day P&L read as ~0, and the daily-loss limit could never trip. The
    bot believed it was flat while carrying $19,188 of open risk.
    open_portfolio_risk() had already been fixed this way for exactly the same
    reason; the equity path was simply missed.

    Uses the broker's own `profit` rather than reconstructing R-multiples from
    stored entry/stop data. It is authoritative, it needs no per-ticket
    bookkeeping to survive a restart, and it sidesteps the contract-multiplier
    disagreement that mis-sized gold in the first place. It is directly
    comparable to the virtual anchor because positions are SIZED for that anchor
    (1R = BASE_RISK_PCT x VIRTUAL_ACCOUNT), so broker dollars are anchor dollars.

    Covers MANAGED_LEGS, not just the active roster: a retired leg's open
    position loses real money and must count against the daily limit.

    Positions listed in ``exclude`` are skipped. Those are tickets that were
    still open when a previous virtual account blew and could not be closed
    (a shut market rejects every close with retcode 10018). They belong to the
    run that lost them, not to the fresh account, so charging their P&L to the
    new one would blow it instantly on a loss it did not take -- and again on
    the next poll, forever.
    """
    skip = set(exclude or ())
    total = 0.0
    for magic in set(magics.values()):
        try:
            for pos in engine.get_positions(magic):
                if pos.ticket in skip:
                    continue
                total += float(getattr(pos, "profit", 0.0) or 0.0)
                total += float(getattr(pos, "swap", 0.0) or 0.0)
        except Exception:
            continue
    return total


def open_portfolio_risk(engine: MT5Engine, magics: dict) -> float:
    """Total USD at risk across every open position in this book.

    Read from the broker by magic rather than the in-memory `tickets` dict so
    the figure survives a restart (that dict starts empty, which would report
    zero risk while the book is fully loaded and happily open more).

    Uses the LIVE stop, so a trade whose stop has trailed into profit
    correctly contributes 0 -- it can no longer lose money, so it should not
    consume risk budget.
    """
    total = 0.0
    for magic in magics.values():
        for pos in engine.get_positions(magic):
            if not pos.sl:
                continue
            upp = engine.usd_per_point(pos.symbol) or 0.0
            risk = (pos.price_open - pos.sl) * (1.0 if pos.type == 0 else -1.0) * upp * pos.volume
            total += max(0.0, risk)
    return total


def apply_trailing_stops(engine: MT5Engine, magics: dict) -> int:
    """Ratchet trailing stop: lock in part of an open trade's gain.

    Added 2026-08-20. This bot previously had NO exit management whatsoever --
    it set SL/TP at entry and never touched them again, so a trade that ran to
    +0.5R and reversed handed every cent back to the stop. portfolio_runner.py
    (the FTMO/YOLO engine) has had a ratchet trail for months; the LucidPro
    funded runner was written standalone and never got one.

    Two deliberate design choices:

    1. Positions are read from MT5 BY MAGIC, not from the caller's in-memory
       `tickets` dict. That dict starts empty on every restart, so trailing off
       it would silently skip every position opened before the current process
       -- which right now is all of them. Going to the broker means the trail
       manages the real book regardless of restarts.

    2. The original risk distance is recovered from the TAKE-PROFIT, not from
       the current stop: once the stop has been ratcheted, the entry->SL gap no
       longer equals 1R. TP never moves, and each leg's TP sits at a known
       R-multiple of its own risk, so |TP - entry| / tp_r_multiple recovers the
       original 1R distance exactly, however many times the stop has moved.

    Only ever tightens (never widens, never reverses direction), and skips any
    candidate inside the broker's minimum stop distance -- modify_sltp would
    reject those with 10016 anyway. Returns the number of stops moved.
    """
    # ROSTER + retired legs, not ROSTER alone. Retiring v6 on 2026-08-25 left
    # real positions open under magics 9211 and 9215; building this map from the
    # active roster only would have silently stopped trailing them, leaving
    # already-profitable trades with their original entry stops and no exit
    # management at all -- the exact failure this function was written to fix.
    tp_r_by_magic = {magics[label]: tp for label, _s, _tf, _c, _p, tp in MANAGED_LEGS
                     if label in magics}
    moved = 0
    for magic, tp_r in tp_r_by_magic.items():
        for pos in engine.get_positions(magic):
            if not pos.tp or not pos.sl or tp_r <= 0:
                continue
            is_long = pos.type == 0  # POSITION_TYPE_BUY
            sign = 1.0 if is_long else -1.0
            risk_dist = abs(pos.tp - pos.price_open) / tp_r
            if risk_dist <= 0:
                continue
            profit_r = (pos.price_current - pos.price_open) * sign / risk_dist
            # Highest qualifying tier wins (TRAIL_TIERS is ordered high->low),
            # so a +5R trade locks 80% rather than the gentler 2R-band 33%.
            frac = next((f for thr, f in TRAIL_TIERS if profit_r >= thr), None)
            if frac is None:
                continue
            lock_r = profit_r * frac
            candidate = pos.price_open + sign * lock_r * risk_dist
            # Tighten only: for a long the stop must rise, for a short it must fall.
            if (candidate - pos.sl) * sign <= 0:
                continue
            if engine.modify_sltp(pos.ticket, candidate, pos.tp):
                moved += 1
                log.info(f"[trail] ticket={pos.ticket} {pos.symbol} magic={magic} "
                         f"at {profit_r:+.2f}R -> SL {pos.sl:.5f} to {candidate:.5f} "
                         f"(locks {lock_r:+.2f}R)")
    return moved


def apply_time_stops(engine: MT5Engine, magics: dict, offset_s: int = 0) -> int:
    """Close any position held longer than MAX_HOLD_HOURS. Returns count closed.

    Added 2026-08-25 with the v7 roster, which is validated under a hard 120h
    cap. This is not an optimisation -- it is the rule that makes the live book
    match the backtest, and several v7 legs exit on it more often than on their
    take-profit.

    Two corrections that bite here:

    * ``pos.time`` is on the BROKER's clock (this server runs UTC+3), so
      comparing it straight to UTC makes every position read three hours
      younger than it is. ``offset_s`` is the measured broker-minus-UTC offset;
      the same mistake previously wrote ledger timestamps hours late and made a
      market-closed check report a shut market as trading.
    * Positions are read BY MAGIC from the broker rather than from the caller's
      ``tickets`` dict, which starts empty on every restart -- trailing off that
      dict would skip every position opened by a previous process, which after
      a restart is all of them.

    A failed close is logged and retried on the next poll rather than being
    treated as done: ``close_position`` returning without a deal ticket means
    the position is still open (a market-closed weekend does exactly this).
    """
    if not TIME_STOP_ENABLED or MAX_HOLD_HOURS <= 0:
        return 0
    now = time.time()
    closed = 0
    for label, _s, _tf, _c, _p, _tp in MANAGED_LEGS:
        magic = magics.get(label)
        if magic is None:
            continue
        for pos in engine.get_positions(magic):
            opened = getattr(pos, "time", 0)
            if not opened:
                continue
            age_h = (now - (opened - offset_s)) / 3600.0
            if age_h < MAX_HOLD_HOURS:
                continue
            res = engine.close_position(pos.ticket)
            if getattr(res, "ok", False):
                closed += 1
                log.warning(f"[time-stop] closed ticket={pos.ticket} {pos.symbol} "
                            f"magic={magic} ({label}) after {age_h:.1f}h "
                            f"(cap {MAX_HOLD_HOURS:.0f}h) P/L {pos.profit:+.2f}")
            else:
                log.warning(f"[time-stop] FAILED to close ticket={pos.ticket} "
                            f"{pos.symbol} at {age_h:.1f}h -- still open, will "
                            f"retry next poll. {getattr(res, 'comment', '')}")
    return closed


def close_all(tickets: dict, engine: MT5Engine) -> float:
    realized_from_closes = 0.0
    for ticket, info in list(tickets.items()):
        pos = engine_positions_by_ticket(engine, ticket)
        if pos is not None:
            engine.close_position(ticket)
            entry = info["entry_price"]
            sl_dist = abs(entry - info["sl"])
            if sl_dist > 0:
                sign = 1.0 if info["direction"] == "long" else -1.0
                exit_price = pos.price_current
                r = (exit_price - entry) * sign / sl_dist
                realized_from_closes += r * info["risk_usd"]
                log_trade(info["sym"], info["label"], info["direction"], entry, exit_price,
                          info["lots"], info["risk_usd"], r, info.get("opened_ts", ""),
                          datetime.now(timezone.utc).isoformat())
        del tickets[ticket]
    return realized_from_closes


def main() -> int:
    engine = MT5Engine(symbol_map={"NAS100": "USTEC", "JP225": "JPN225", "BTCUSD": "BTC", "ETHUSD": "ETH"})
    acc = engine.connect()
    log.warning(f"Connected: {'DEMO' if acc.is_demo else 'REAL'} account {acc.login}")
    if not acc.is_demo:
        raise RuntimeError("REAL account detected -- refusing to run the funded bot on it "
                            "without an explicit, separately-reviewed go-ahead.")

    state = load_state()
    if state["started_ts"] is None:
        state["started_ts"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
    if state["halted"]:
        # A halt can only be a LEGACY flag now: since 2026-08-29 a daily-limit
        # breach blows the virtual account and opens a fresh one instead of
        # halting. Clear it and start a new run rather than refusing to trade --
        # refusing was the old design's terminal state and it needed a hand edit
        # of the state file to recover.
        log.warning("Clearing a legacy HALTED flag (%s). Breaches now blow the "
                    "virtual account and start a fresh run instead of halting.",
                    state.get("halt_reason"))
        state.update({"halted": False, "halt_reason": None,
                      "realized_usd": 0.0, "peak_equity": VIRTUAL_ACCOUNT,
                      "day_start_equity": VIRTUAL_ACCOUNT,
                      "run_start": datetime.now(timezone.utc).isoformat()})
        save_state(state)
    if not state.get("run_start"):
        state["run_start"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

    strategies = {}
    magics = {}
    for label, sym, tf, cls, params, tp in ROSTER:
        strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym, "capital": {"risk_usd_per_trade": 1.0}, "strategy": strat}
        strategies[label] = cls(cfg)
        magics[label] = LEG_MAGICS[label]
    # Retired legs get a magic entry but NO strategy: nothing may open new
    # positions under them, but apply_trailing_stops/apply_time_stops look up
    # every MANAGED_LEGS label here, and a missing entry would silently leave
    # v6's still-open positions untended at the changeover.
    for label, *_ in MANAGED_LEGS:
        magics.setdefault(label, LEG_MAGICS[label])
    log.warning(f"LucidPro FUNDED runner started. Roster: {[r[0] for r in ROSTER]}")
    log.warning(f"Managing (no new entries) retired legs: "
                f"{sorted(set(magics) - {r[0] for r in ROSTER})}")

    tickets: dict[int, dict] = {}
    last_bar_ts: dict[str, object] = {label: None for label, *_ in ROSTER}
    current_day = state["current_day"]
    peak_equity = state["peak_equity"]
    realized_usd = state["realized_usd"]
    # Restored from state so a restart MID-DAY keeps measuring today's loss from
    # where the day actually opened. Re-baselining to current equity on every
    # restart would silently hand the account a fresh $2,000 allowance -- and
    # the watchdog restarts this process routinely, so that is not a rare edge
    # case. Falls back to the anchor only on a genuinely fresh state file.
    day_start_equity = state.get("day_start_equity") or (VIRTUAL_ACCOUNT + realized_usd)

    # Persisted so a restart during a weekend (when every tick is stale and the
    # offset cannot be measured) does not fall back to 0 and make every open
    # position read three hours younger than it is against the 120h cap.
    broker_offset = int(state.get("broker_offset_s", 0))

    # Tickets belonging to already-blown runs. Their P&L must never reach the
    # current account, or a weekend-stranded loss would blow every fresh
    # account on its first poll, forever.
    excluded_tickets: set[int] = set(state.get("excluded_tickets") or [])

    _touch_heartbeat()
    threading.Thread(target=_self_watchdog, daemon=True, name="self-watchdog").start()

    while True:
        try:
            _touch_heartbeat()
            realized_usd += reconcile_closed_tickets(tickets, engine, state, realized_usd)
            # Trail BEFORE anything else that can exit or open: a stop moved
            # this poll should be in place before the EOD/MLL checks read
            # equity, and before new entries change the book.
            apply_trailing_stops(engine, magics)
            # Then the time stop, which can close outright. Order matters: a
            # position at its 120h cap should still get the benefit of a stop
            # ratcheted this same poll if the close fails and it survives to
            # the next one.
            try:
                _off = engine.broker_offset_seconds(
                    engine.resolve_symbol(ROSTER[0][1]), fallback=broker_offset)
                if _off:
                    broker_offset = _off
                    state["broker_offset_s"] = broker_offset
            except Exception:
                pass    # stale tick / weekend -- keep the last known-good offset
            apply_time_stops(engine, magics, broker_offset)

            now = datetime.now(timezone.utc)
            day = day_key(now)
            if current_day is None:
                current_day = day
                day_start_equity = VIRTUAL_ACCOUNT + realized_usd + floating_usd(engine, magics, excluded_tickets)
                state["day_start_equity"] = day_start_equity
                save_state(state)
            elif day != current_day:
                # Rollover: the daily loss allowance RESETS. Nothing is halted
                # here -- unlike the trailing MLL this replaced, surviving the
                # day means starting the next one with a full $2,000 of room.
                eq_close = VIRTUAL_ACCOUNT + realized_usd + floating_usd(engine, magics, excluded_tickets)
                peak_equity = max(peak_equity, eq_close)
                current_day = day
                day_start_equity = eq_close
                state.update({"peak_equity": peak_equity, "realized_usd": realized_usd,
                              "current_day": current_day, "day_start_equity": day_start_equity})
                save_state(state)
                log.info(f"New trading day {day}: daily loss allowance reset, "
                         f"start equity ${day_start_equity:,.2f}")

            eq = VIRTUAL_ACCOUNT + realized_usd + floating_usd(engine, magics, excluded_tickets)
            day_pnl = eq - day_start_equity

            # --- daily limit breached: FAIL THIS ACCOUNT AND START A FRESH ONE --
            #
            # CHANGED 2026-08-29. This used to halt terminally and exit(1),
            # requiring a hand edit of the state file to resume -- so a breach
            # stopped the bot dead and nothing was recorded beyond a halt reason.
            # It now behaves like the trial runner: close everything, record the
            # blown account in a run history, and open a fresh $50,000 virtual
            # account. The point of running this is to learn how often the roster
            # survives to a payout, and an account that halts on first breach
            # produces exactly one data point.
            #
            # Real money is unaffected either way -- VIRTUAL_ACCOUNT is a
            # bookkeeping anchor over a demo account, not a broker balance.
            if day_pnl <= -DAILY_LOSS_LIMIT:
                realized_usd += close_all(tickets, engine)

                # Anything close_all could not flatten (a shut market rejects
                # every close with retcode 10018) is CARRIED BY THE BLOWN RUN,
                # not inherited by the fresh account.
                #
                # An earlier version instead refused to complete the reset until
                # the book was flat. That was wrong in practice: over a weekend
                # it left the dashboard showing the dead account's -$6,785 for
                # days, with no reset in sight. Carrying the tickets lets the new
                # account start clean immediately while keeping the accounting
                # honest -- the loss stays attributed to the run that took it,
                # and the positions are still managed (trailing and time stops
                # run off MANAGED_LEGS, which is unaffected by this).
                carried = []
                for _m in set(magics.values()):
                    try:
                        carried.extend(p.ticket for p in engine.get_positions(_m))
                    except Exception:
                        pass
                carried = [t for t in carried if t not in excluded_tickets]
                if carried:
                    log.error("%d position(s) could not be closed (market shut?) -- "
                              "carrying them with the blown run; the fresh account "
                              "will not be charged their P&L.", len(carried))
                excluded_tickets.update(carried)

                final_eq = VIRTUAL_ACCOUNT + realized_usd
                runs = state.setdefault("runs", [])
                runs.append({
                    "outcome": "blown",
                    "reason": f"daily loss {day_pnl:,.2f} <= -{DAILY_LOSS_LIMIT:,.0f}",
                    "opened_ts": state.get("run_start"),
                    "resolved_ts": now.isoformat(),
                    "day": current_day,
                    "final_equity": round(final_eq, 2),
                    "final_pct": round((final_eq - VIRTUAL_ACCOUNT) / VIRTUAL_ACCOUNT * 100, 2),
                    "payout_reached": bool(state.get("payout_ready")),
                    "carried_open": len(carried),
                })
                state["runs_total"] = int(state.get("runs_total", 0)) + 1
                state["runs_blown"] = int(state.get("runs_blown", 0)) + 1

                log.error("DAILY LIMIT BREACH: day P&L %.2f on %s -- closed all "
                          "positions, recorded run #%d as blown, starting a fresh "
                          "$%.0f virtual account.",
                          day_pnl, current_day, state["runs_total"], VIRTUAL_ACCOUNT)
                alert(f"🔴 <b>ACCOUNT BLOWN — DAILY LOSS LIMIT</b>\n"
                      f"Day P&L ${day_pnl:,.0f} breached the ${DAILY_LOSS_LIMIT:,.0f} limit.\n"
                      f"All positions closed. Run #{state['runs_total']} recorded as blown; "
                      f"a fresh ${VIRTUAL_ACCOUNT:,.0f} account has started.",
                      book="LucidPro-Funded", html=True)

                # Reset the virtual account. realized_usd is the anchor-relative
                # ledger, so zeroing it IS the reset; the broker balance is
                # untouched and the ledger keeps every trade for the record.
                realized_usd = 0.0
                peak_equity = VIRTUAL_ACCOUNT
                day_start_equity = VIRTUAL_ACCOUNT
                tickets.clear()
                state.update({"halted": False, "halt_reason": None,
                              "payout_ready": False,
                              "peak_equity": peak_equity,
                              "realized_usd": realized_usd,
                              "day_start_equity": day_start_equity,
                              "current_day": current_day,
                              "run_start": now.isoformat(),
                              # Persisted so a restart does not re-adopt the
                              # blown run's positions into the fresh account.
                              "excluded_tickets": sorted(excluded_tickets)})
                save_state(state)
                time.sleep(POLL_INTERVAL_S)
                continue

            # Payout reached: flag it once, keep trading (the operator decides
            # when to actually withdraw -- the bot never moves money).
            if eq - VIRTUAL_ACCOUNT >= PAYOUT_TARGET and not state.get("payout_ready"):
                state["payout_ready"] = True
                # Count it as a SURVIVED run so the stats show both outcomes.
                # Without this the history would only ever record blow-ups and
                # the survival rate would read 0% no matter how well it went.
                state["runs_paid"] = int(state.get("runs_paid", 0)) + 1
                state.setdefault("runs", []).append({
                    "outcome": "payout",
                    "reason": f"equity reached +${eq - VIRTUAL_ACCOUNT:,.0f}",
                    "opened_ts": state.get("run_start"),
                    "resolved_ts": now.isoformat(),
                    "day": current_day,
                    "final_equity": round(eq, 2),
                    "final_pct": round((eq - VIRTUAL_ACCOUNT) / VIRTUAL_ACCOUNT * 100, 2),
                    "payout_reached": True,
                })
                state["runs_total"] = int(state.get("runs_total", 0)) + 1
                save_state(state)
                log.warning(f"PAYOUT AVAILABLE: equity ${eq:,.2f} is "
                            f"${eq - VIRTUAL_ACCOUNT:,.2f} above the anchor.")
                alert(f"💰 <b>PAYOUT AVAILABLE</b>\n"
                      f"Equity ${eq:,.0f} (+${eq - VIRTUAL_ACCOUNT:,.0f} over anchor).\n"
                      f"Target was ${PAYOUT_TARGET:,.0f}.",
                      book="LucidPro-Funded", html=True)

            # Risk taper now measures room left in TODAY's allowance, not
            # distance to a lifetime floor.
            room = DAILY_LOSS_LIMIT + min(0.0, day_pnl)
            frac = max(TAPER_FLOOR, min(1.0, room / DAILY_LOSS_LIMIT))
            today_risk_usd = BASE_RISK_PCT * VIRTUAL_ACCOUNT * frac

            # Stop opening once the day is deep enough into its allowance that
            # open positions could carry it over the line unattended.
            if day_pnl <= -DAILY_STOP_FRACTION * DAILY_LOSS_LIMIT:
                log.info(f"Day P&L ${day_pnl:,.2f} past {DAILY_STOP_FRACTION:.0%} of the "
                         f"${DAILY_LOSS_LIMIT:,.0f} limit -- no new entries until rollover.")
                time.sleep(POLL_INTERVAL_S)
                continue

            # Risk budget, evaluated once per poll against the real book.
            risk_budget = MAX_PORTFOLIO_RISK_PCT * VIRTUAL_ACCOUNT
            open_risk = open_portfolio_risk(engine, magics)
            if open_risk >= risk_budget:
                log.info(f"Risk budget full: ${open_risk:,.0f} open vs "
                         f"${risk_budget:,.0f} cap -- holding new entries "
                         f"(existing positions unaffected).")

            for label, sym, tf, cls, params, tp in ROSTER:
                if open_risk >= risk_budget:
                    break
                per_leg_open = sum(1 for info in tickets.values() if info["label"] == label)
                if per_leg_open >= MAX_CONCURRENT_PER_LEG:
                    continue
                if not engine.is_tradeable(sym):
                    continue
                bars = engine.get_bars(sym, tf, BAR_COUNT)
                if bars is None or len(bars) < 20:
                    continue
                if last_bar_ts[label] is not None and bars.index[-1] == last_bar_ts[label]:
                    continue  # no new closed bar yet -- do NOT advance last_bar_ts until resolved below
                closed_bars = bars.iloc[:-1]
                if len(closed_bars) < 20:
                    continue
                signals = strategies[label].on_bar(len(closed_bars) - 1, closed_bars)
                last_bar_ts[label] = bars.index[-1]
                for spec in signals:
                    if per_leg_open >= MAX_CONCURRENT_PER_LEG:
                        break
                    sl_dist = abs(spec.entry_price - spec.stop_loss)
                    if sl_dist <= 0:
                        continue
                    # LIVE broker spec first, instruments.py only as fallback.
                    #
                    # PAIR_SPECS carries the textbook contract size (XAUUSD:
                    # "1 standard lot = 100 troy oz"), but this broker prices
                    # gold at 10 oz/lot -- so contract_mult said 100 while the
                    # terminal said 10. Sizing divides by that number, so every
                    # gold position came out 10x too SMALL: the bot logged
                    # "risk=$314" while actually risking ~$31, and with gold at
                    # ~89% of this roster's total R that suppressed almost all
                    # of its return. mt5_engine.usd_per_point() reads
                    # tick_value/tick_size from the terminal and is
                    # authoritative per AUDIT C5; portfolio_runner.py has always
                    # used it this way, this runner was simply never updated.
                    mult = engine.usd_per_point(sym) or contract_mult(sym)
                    ideal_lots = today_risk_usd / (sl_dist * mult)
                    vol_min, vol_max, vol_step = engine.volume_spec(sym)
                    lots = round(ideal_lots, 2)
                    if lots < vol_min:
                        log.info(f"[{label}] signal wants {lots} lots, below broker min {vol_min} "
                                 f"-- skipping (min_lot_policy=skip, keeps risk honest)")
                        continue
                    if lots > vol_max:
                        log.warning(f"[{label}] signal wants {lots} lots, broker max {vol_max} -- clamping")
                        lots = vol_max
                    result = engine.place_market_order(
                        symbol=sym, direction=spec.direction.value, lots=lots,
                        sl=spec.stop_loss, tp=spec.take_profit, magic=magics[label],
                        comment=f"lucidpro_funded_{label}", signal_entry_price=spec.entry_price,
                    )
                    # Belt-and-braces on top of OrderResult.ok: a ticket of 0
                    # is not a position. Tracking one poisons everything
                    # downstream -- history_deals_get(position=0) matches an
                    # UNRELATED historical deal, and its price/timestamps get
                    # written into the ledger as this trade's exit. That is
                    # exactly how a phantom "+318R / $100,000" winner with a
                    # 0.0 exit price and a close date before its open date
                    # reached the dashboard on 2026-08-21.
                    if result.ok and not result.ticket:
                        log.error(f"[{label}] order reported success but returned ticket=0 "
                                  f"(retcode={result.retcode} {result.comment!r}) -- treating "
                                  f"as REJECTED and not tracking it.")
                    elif result.ok:
                        real_risk_usd = result.volume * sl_dist * mult
                        tickets[result.ticket] = {
                            "label": label, "sym": sym, "entry_price": result.price, "sl": result.sl,
                            "direction": spec.direction.value, "lots": result.volume,
                            "risk_usd": real_risk_usd, "opened_ts": datetime.now(timezone.utc).isoformat(),
                        }
                        per_leg_open += 1
                        state["total_trades"] = state.get("total_trades", 0) + 1
                        log.info(f"[{label}] opened {spec.direction.value} ticket={result.ticket} "
                                 f"lots={result.volume} risk=${real_risk_usd:,.0f} "
                                 f"(risk_taper={frac:.0%} of base {BASE_RISK_PCT:.1%})")
                        dir_emoji = "🟢📈" if spec.direction.value == "long" else "🔴📉"
                        alert(
                            f"{dir_emoji} <b>OPENED</b> {label} {sym} {spec.direction.value}\n"
                            f"lots={result.volume} @ {result.price:.5f}  risk=${real_risk_usd:,.0f}\n"
                            f"⚖️ Risk taper: {frac:.0%} of base {BASE_RISK_PCT:.1%}",
                            book="LucidPro-Funded",
                            html=True,
                        )
                    else:
                        log.warning(f"[{label}] order failed retcode={result.retcode} {result.comment}")

            save_state(state)
            time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            log.warning("Stopped by user.")
            return 0
        except Exception as e:  # noqa: BLE001 -- must not crash the loop
            log.error(f"Loop error (continuing): {e!r}")
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    raise SystemExit(main())
