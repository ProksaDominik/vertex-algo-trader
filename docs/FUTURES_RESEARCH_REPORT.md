# Lucid futures research -- report (2026-09-29 / 30)

Brief (owner's answers): **best expected $/week** on **LucidPro 50K**, at most
**1-2 failed evaluations per pass**, **intraday** (holds up to ~3 days only as a
fallback -- LucidPro forbids them). Method: the backtest-expert workflow --
state the hypothesis, codify it, then spend most of the effort trying to break
it (friction, parameter plateaus, halves/years, a second dataset, walk-forward,
Monte Carlo of the prop rules).

## Bottom line

* The bot's **existing strategy library has no edge on Lucid futures.**
* Of seven futures-native intraday ideas, **one survived every test**: an
  **opening-range breakout with a daily-trend filter, on MNQ and MGC**. It is
  real (plateau, walk-forward, real futures bars all agree) but **small**, and
  it earned more after 2021 than before.
* **Evaluation (6 MNQ + 3 MGC):** ~**40% pass, ~1.5 failed evaluations per
  pass, median ~16 trading days** (44% / 1.3 / 10 days if 2021-26 conditions
  persist) -- versus 31% / 2.2 for a zero-edge strategy of the same size.
  Inside the 1-2 failed-evaluation budget, at its upper end.
* **Funded (2 MNQ + 1 MGC):** first payout before the MLL **~40% vs 43%**
  (17% still running after 6 months) over 2016-26; **53% vs 45%** over 2021-26.
  Roughly a coin flip -- the weakest link.
* **Multi-day edges are more robust** (turn of the month: both halves positive
  on four markets) but need a firm/account that allows overnight holds.
* **Recommendation:** forward-test on the simulators for ~6-8 weeks (~40+
  trades per leg) before paying for anything. If live $/trade tracks section
  3, run ONE evaluation at 6 MNQ + 3 MGC. Consider the multi-day track in
  parallel if the forward test disappoints.

## 1. The existing library (futures_roster_search.py)

16 families x 37 parameter sets x {15m, 1h} x {1R, 2R} on 12 contracts, ~15
months (15m) / ~20 months (1h) of real futures bars, Lucid's rules.

| | |
|---|---|
| Tests with n >= 50 | 1,730 |
| Profitable after costs | 16% (mean -0.094R/trade) |
| t > 2 | 5 (chance alone: ~40) |
| Mean reversion (RSI, Bollinger, Stochastic, VWAP, S/R, sweep, time-of-day) | all clearly negative |
| Trial's old London Breakout (15m, 2R) | 6C -0.083, 6B -0.031, 6S +0.051, MGC +0.036 R/trade |

Found and fixed on the way: ORB, VWAP and time-of-day strategies had **never
produced a live signal** (they read a column only backtests have) -- on MT5 too.

## 2. Futures-native intraday hypotheses

10.5 years (2016-2026) of 15m data for S&P, Nasdaq, Dow, gold, crude and FX
(proxies for MES/MNQ/MYM/MGC/MCL/6E/6B/6J/6A) with futures costs: 1 tick on
entries, stops and time exits + Lucid commission both ways; a bar spanning
stop and target counts as the stop.

| Hypothesis | Result |
|---|---|
| H1 intraday momentum (first half hour -> last half hour) | no edge after micro costs |
| H2 opening-range breakout (+ narrow-range / trend filters) | **trend-filtered: plateau on NQ and GC** |
| H3 opening gap, fade or follow | no robust edge |
| H4 volatility breakout (open +/- k x prior range) | NQ k=0.75 a lone peak -- rejected |
| H5 afternoon continuation with the trend | plateau on NQ (12/12, both halves +) -- but fires on the same trend days as ORB: on ORB-free days only 47 NQ / 50 GC trades in 10 years; adding it moved pass rates ~1 point. Not deployed |
| H6 London breakout + trend filter | 6J 16/16, but 6E/6B/6A 0/16 each -- one market, likely the 2021-24 yen trend. Rejected |
| H7 crude after the Wednesday inventory report | no edge |

## 3. The survivor: ORB + daily-trend filter

Plateau (futures_robustness.py): opening range 15/30/45/60 min x last entry
11:00/12:00/13:00 x 2R/end of day = 24 cells per market -- **GC 24/24 and NQ
24/24 positive**; CL 14/24, ES 9/24, YM 0/24.

Deployed = centre of the plateau: **45-minute range, entries until 12:00 ET,
first breakout of the day decides (against the trend -> no trade), stop at the
far side of the range, 2R target, out by 15:55 ET.**

| Leg | Trades (10.5 y) | $/trade, 1 micro | t | 1st half / 2nd half | Years positive |
|---|---|---|---|---|---|
| MNQ | 1,054 | +10.47 | 1.61 | -5.39 / +26.34 | 45% |
| MGC | 943 | +7.69 | 2.12 | -0.03 / +15.39 | 73% |

* **Walk-forward** (futures_walkforward.py) -- pick the best cell each January
  on past years only, trade the next year: MNQ **+$11,313** out of sample
  (788 trades, 5/8 years positive), MGC **+$7,067** (467 trades, 7/8). No
  degradation vs in-sample; re-selecting earned about what the fixed centre
  did. Caveat: 2022 alone is ~60% of MNQ's out-of-sample profit.
* **Real futures bars** (NinjaTrader, Jul 2025 - Sep 2026): same direction as
  the proxy on 100% of shared days; MNQ +$9.43/trade (proxy +$11.47), MGC
  +$55.97 (proxy +$51.96).
* **Live code** (strategies_orb_trend.py) reproduces the research signals on
  those bars: 100/100 MNQ and 115/116 MGC days, same direction. It skips US
  holiday sessions (the research never traded them).
* **Exits** (futures_exits.py): nothing beat 2R / no breakeven / hold to the
  close by more than noise; earlier time exits are clearly worse; re-picking
  the exit each year out of sample earned LESS (the signature of fitting noise).

## 4. LucidPro 50K Monte Carlo

20,000 runs, 5-day blocks of the real daily results (lucid_montecarlo.py).
Evaluation: $53,000 target vs a $2,000 trailing MLL (no lock -- the stricter
reading). Funded: MLL locks at $50,100 once an end-of-day balance reaches
$52,100; "payout" = an end-of-day balance of $52,600. Each day's worst point
= all its losing trades first (pessimistic). **Losses are not capped at the
daily limit** -- see the correction note.

| Stack | History | $/week | Eval pass | Fails/pass | Median days | Funded payout / blow |
|---|---|---|---|---|---|---|
| **6 MNQ + 3 MGC** (trial) | 2016-26 | 159 | **40%** | **1.52** | **16** | 44% / 56% |
| 6 MNQ + 3 MGC | 2021-26 | 328 | 44% | 1.29 | 10 | 48% / 52% |
| 4 MNQ + 2 MGC | 2016-26 | 106 | 39% | 1.53 | 27 | 44% / 55% |
| **2 MNQ + 1 MGC** (funded) | 2016-26 | 53 | 34% (19% open) | 1.37 | 63 | **40% / 43%** (17% open) |
| 2 MNQ + 1 MGC | 2021-26 | 109 | 48% | 1.03 | 46 | **53% / 45%** |
| zero edge, 6+3 | 2016-26 | -- | 31% | 2.24 | -- | 35% / 65% |
| zero edge, 2+1 | 2016-26 | -- | 21% | 2.89 | -- | 27% / 59% |

Sizing policies (lucid_policy_mc.py, lucid_riskbased_mc.py), 2016-26:
**fixed contracts beat fixed-$-risk sizing** (4+2: 1.53 failed evals per
pass vs >= 1.95 for $450-$700 per trade -- the edge lives on wide-range days,
which risk sizing shrinks); half-size-until-cushion and tapers are worse;
bigger fixed size resolves faster at the same odds; smaller size has a better
payout:blow ratio but is slow.

**Correction note (2026-09-30):** the first Monte Carlo runs capped each day's
loss at -$1,300 (the soft daily limit). The soft limit only stops NEW trades;
a stop already in the market can lose more than $2,000 on a wide-range day.
The cap hid exactly the worst days and inflated the results (6+3 had shown
49% pass / 1.04 failed evaluations per pass and $339/week). A "$600 daily
stop" policy that looked spectacular (69% pass) was the same artifact in a
stronger form. All figures above are uncapped.

**Deployed sizes:** trial **6 MNQ + 3 MGC** (same odds as 4+2, twice as fast;
its MLL-room clamp still trims contracts on very wide days); funded **2 MNQ +
1 MGC** (blow risk 43% instead of 55% at 4+2 -- the funded account is the
asset to protect), each shrunk only if a full stop-out would not fit in 90% of
the room above the MLL floor.

## 5. Multi-day fallback (futures_multiday.py) -- NOT possible on LucidPro

$ per trade per 1 micro, 10.5 years, futures costs, overnight gaps included:

| | Turn of month (3 days) | RSI(2) pullback (<= 3 days) | 20-day breakout (3 days) |
|---|---|---|---|
| S&P | +50.9, both halves +, 82% yrs | +84.5 (t 2.36) | +9.5 |
| Nasdaq | +83.1, both halves + | +85.5 | +46.5 |
| Dow | +42.4, both halves +, 73% yrs | +20.7 | -3.1 |
| Gold | **+112.8, t 2.94, both halves +, 82% yrs** | -11.1 | +20.3 |
| Crude | +6.3 | +3.2 | -13.4 |

Turn of the month is the most robust effect found anywhere in this research,
but trades ~once a month per market and holds overnight.

## 6. What runs now (simulator)

| Book | Account | Trades |
|---|---|---|
| Trial | Sim101 | ORB + trend, **6 MNQ + 3 MGC**, $50K evaluation attempts |
| Funded | Sim102 (create it in NinjaTrader) | ORB + trend, **2 MNQ + 1 MGC** fixed |
| Lab | none | 296 configs x 12 contracts, forward test |

## 7. Next steps

1. **Forward test** 6-8 weeks on the simulators; compare live $/trade per leg
   to section 3. Proceed only if it tracks.
2. Decide on the multi-day track (a firm that allows overnight holds, or a
   personal account) if the intraday forward test disappoints.
3. Real money needs an always-on machine (VPS), NinjaTrader connected to Lucid
   through CQG/Tradovate, Lucid's actual evaluation fee in the $/week maths,
   and the funded MLL lock verified against the account's terms.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\futures_roster_search.py --no-download --max-minutes 50
.venv\Scripts\python.exe scripts\futures_hypotheses.py
.venv\Scripts\python.exe scripts\futures_robustness.py
.venv\Scripts\python.exe scripts\futures_stack_mc.py
.venv\Scripts\python.exe scripts\futures_walkforward.py
.venv\Scripts\python.exe scripts\futures_exits.py
.venv\Scripts\python.exe scripts\futures_second_edge.py
.venv\Scripts\python.exe scripts\futures_combo_mc.py
.venv\Scripts\python.exe scripts\lucid_policy_mc.py
.venv\Scripts\python.exe scripts\lucid_riskbased_mc.py
.venv\Scripts\python.exe scripts\futures_multiday.py
```
