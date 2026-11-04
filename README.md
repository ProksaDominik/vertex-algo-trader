# Vertex: a trading bot for prop-firm futures accounts

Vertex trades Nasdaq-100 futures (NQ) automatically on prop-firm accounts
(LucidPro 50K). A prop firm lends you a trading account if you pass its test;
if you lose too much, the account is closed and you only lose the fee.

The project has three parts: the **research** that picked the strategy, the
**bot** that trades it through NinjaTrader 8, and a **dashboard** to watch it.

Not financial advice; this is a personal project.

## Results in one minute

- **Backtest of the plan:** a typical year makes about **+$117,000** (real NQ
  data), and every 12-month stretch tested was profitable.
- **But the most important test:** when the bot picks buy or sell **at random**
  (a coin flip, so no skill at all), the plan still makes money in **88–96%** of
  years. So most of the profit does **not** come from predicting the market. It
  comes from how prop-firm accounts pay: a failed account only costs its fee
  ($115), while a working account pays out up to $2,500 at a time.
- **One account on its own:** passes the firm's test **41%** of the time, makes
  about **$20,900** in an average year, and loses at most **$3,325** in a bad one.
- **Live so far:** **+$1,159**

## The numbers

### The plan, replayed over every 12 months of history (21 Oct 2026)

The plan: start with one account, add another for every $2,500 earned (up to
five), and take at most $1,500 profit a day. Tested on real NQ futures
(2021–25) and on the Nasdaq-100 index (2017–25, a longer history), also under
three stress tests:

- **Worse fills:** every order gets a worse price ($10 per contract, about 3× the
  normal assumption).
- **Coin flip:** buy or sell is picked at random.
- **Harshest:** both at once.

| 12 months | Years in profit | Worst year | Typical year | Best year |
|---|---:|---:|---:|---:|
| As tested, real NQ | 100% | +$38,135 | +$116,849 | +$210,906 |
| As tested, index | 96% | −$4,676 | +$104,303 | +$201,014 |
| Worse fills, real NQ | 100% | +$38,135 | +$113,017 | +$198,694 |
| Worse fills, index | 96% | −$4,721 | +$96,452 | +$201,014 |
| **Harshest, real NQ** | **96%** | **−$10,771** | **+$72,776** | +$229,578 |
| **Harshest, index** | **88%** | **−$21,037** | **+$47,733** | +$280,649 |

How often each kind of year happened:

| Year | As tested | Harshest |
|---|---|---|
| Above $200,000 | 1–2% | about 1% |
| Above $150,000 | about 1 in 4–5 | about 1 in 12 |
| Above $100,000 | about 1 in 2 | about 1 in 4 |
| Lost money | 0–4% | 4–12% (up to 1 year in 8) |

### One account (LucidPro 50K, 25 Oct 2026)

| Price | Passes the test | Funded accounts that get paid | Paid per funded account | Reaches $2k in 2 months | Reaches $5k in 2 months | Years in profit | Worst year | Average year |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| $115 (reset $120) | 41% | 37% | $823 | 52% | 23% | 74% | −$3,325 | $20,900 |

## How the strategy was chosen

The rule was: write the idea down, code it, then try hard to break it (trading
costs, different time periods, a second dataset, walk-forward tests).
Full report: [`docs/FUTURES_RESEARCH_REPORT.md`](docs/FUTURES_RESEARCH_REPORT.md).

- **The first forex strategies had no edge.** On a 10-symbol basket, five of six
  lost money out-of-sample ([`backtest/strategy_comparison.md`](backtest/strategy_comparison.md)).
- **On futures, 1,730 strategy tests:** only 16% were profitable after costs,
  about what luck alone would give.
- **Seven new ideas on 10.5 years of data:** one survived every test (an
  opening-range breakout with a trend filter). It is real but small: it passes
  the firm's test about 40% of the time, against 31% for random trading.

## How it works

- **The bot** ([`scripts/run_lucid_futures_funded.py`](scripts/run_lucid_futures_funded.py)):
  once a trading day, buys 1 NQ contract with a profit target set to what the
  next payout still needs ($150–$1,500), and closes by 16:00 New York time.
- **Safety:** it follows the prop firm's rules itself (the maximum loss, the
  daily loss limit, flat before the session ends) and checks them every few
  seconds.
- **Execution:** a small NinjaTrader 8 add-on
  ([`ninjatrader/VertexBridge.cs`](ninjatrader/VertexBridge.cs)) receives the
  bot's orders on the same computer and places them.
- **Dashboard:** Streamlit: live accounts, trade history and the simulator
  behind the tables above.
- **Tests:** `pytest` in [`tests/`](tests/).

The 21 Oct stress-test variants (worse fills, coin flip) were run in a research
session; that script is not in this repo yet.

Setup: [`INSTALL.md`](INSTALL.md).
