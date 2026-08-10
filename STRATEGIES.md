# Strategies — what each one does and how it works

Every strategy here shares the same shape: on each closed bar it decides
whether to enter, and if so it returns an entry price, a stop loss (SL), a
take profit (TP), and a lot size. Two ideas are common to all of them:

- **SL is placed at a real level** (a channel edge, a swing, a sweep
  extreme, the day's open, VWAP…), so every trade's stop distance is
  different — it adapts to the market, not a fixed number of pips.
- **TP is a multiple of that stop distance** (`tp_r_multiple`). "R" = the
  risk of one trade. TP at 2R means the target is twice as far as the stop,
  so one winner pays for two losers. Lots are sized so every trade risks the
  same dollar amount regardless of how wide its stop is.

On top of the entry/exit, the runner adds management: a ratchet trailing
stop, an optional partial take, a skyrocket exit, and (per strategy) whether
to use a trail at all. Trend strategies keep a wide TP + trail (let winners
run); reversion strategies use a tight TP and no trail (grab the snap-back
and leave).

---

## Trend / breakout family

### Donchian Breakout  *(the out-of-sample winner)*
Buys strength, sells weakness. It draws a channel from the highest high and
lowest low of the last N bars (default 10). A close **above** the channel top
is a breakout → go long; a close **below** the bottom → go short. The stop
sits at the opposite channel edge (a deliberately wide "Turtle" stop that
keeps you in breakouts that wobble), and TP is 2R. This is the only strategy
that was net-positive across the test basket — trend-following on a decisive
break is a real, durable edge.

### MACD Trend
Trades momentum shifts rather than raw breakouts. MACD is the gap between a
fast and a slow moving average; its signal line is a moving average of that
gap. When MACD crosses **up** through its signal line while positive → long;
crosses **down** while negative → short. Stop at the recent swing low/high,
TP 2R. Fires more often than Donchian and catches turns earlier, but is
noisier — marginal in the test.

### Market-Open BOS (Break of Structure)
Session-timed structure trading. At a session open (New York, 13:00 UTC) it
finds the most recent **confirmed swing high and low** — actual pivots, each
confirmed a few bars after it forms so there's no looking into the future —
and enters the first clean break of that structure (break the swing high →
long; break the swing low → short). Stop at the opposing structure, one
trade per symbol per day. Works on gold and the German index; loses on the
others in the test.

---

## Mean-reversion family

These fade a stretch and target a snap back to "fair value", so they use a
tight 1R TP and no trailing stop.

### VWAP Reversion  *(best of the new three)*
VWAP is the volume-weighted average price of the day — the price most volume
actually traded at, which institutions treat as fair value. When price runs
a configured distance away from VWAP **and** the current bar rejects back
toward it (an up-bar below VWAP / a down-bar above), it enters a reversion
toward VWAP. It's the only strategy that uses volume. Net-negative overall,
but genuinely strong on EURUSD (+$1,911, profit factor 1.83 in the test) —
which is why EURUSD is the only symbol it's deployed on.

### TOD Reversion (Time-of-Day)
Some hours tend to mean-revert — a market that ran far from where it opened
often drifts back later in the session. At a set hour (20:00 UTC default), if
price is stretched more than a set multiple of ATR from the day's opening
price, it fades the move back toward the open. One trade per day. Weak
overall (only 2 of 10 symbols positive) — deployed only on EURUSD, and only
on the aggressive YOLO book.

### Sweep Reclaim (liquidity grab)
Fades stop-hunts. Stops pool just beyond obvious swing highs/lows; price
often spikes **through** such a level to trigger them, then snaps back. This
waits for a bar that pierces a recent swing low but **closes back above it**
(the "reclaim") and goes long — betting the sweep was just grabbing
liquidity, not real intent. Mirror for shorts. It was negative on every
symbol in the test, so it ships **disabled** — kept in the config only so the
learning layer can re-test it if you turn it on.

---

## How "best strategy" was decided

`scripts/compare_strategies.py` runs every strategy across a 10-symbol basket
on the validation window only (data the tuning never saw), each with the exit
that fits its style, and ranks by out-of-sample expectancy. Result
(`backtest/strategy_comparison.md`):

| rank | strategy | avg $/trade | positive symbols |
|---|---|---|---|
| 1 | Donchian | **+4.1** | 5 / 10 |
| 2 | MACD | −5.8 | 3 / 10 |
| 3 | BOS | −7.9 | 3 / 10 |
| 4 | VWAP Reversion | −16.9 | 3 / 10 |
| 5 | Sweep Reclaim | −25.9 | 0 / 10 |
| 6 | TOD Reversion | −29.7 | 2 / 10 |

Honest takeaway: **Donchian is the only strategy with a real edge across the
board.** The others are deployed only on the specific symbols where they
validate positive, and the two worst are disabled. The live slot-health and
session-learning layers are the ongoing backstop — anything that starts
bleeding gets benched automatically.
