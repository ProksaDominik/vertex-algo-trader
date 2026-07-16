# Vertex Algo Bot

Multi-strategy algorithmic paper-trading system for MetaTrader 5, running two
parallel portfolios against one broker account:

- **FTMO** — conservative, framed against a $10,000 prop-firm-style account,
  with kill switches tuned under FTMO's daily/total drawdown limits.
- **YOLO** — aggressive, framed against a $200 personal account, no kill
  switch by design.

Both portfolios run several strategies at once across a basket of FX pairs,
gold, and indices, each as an independent signal stream (its own MT5 magic
number), monitored live through a custom web dashboard.

## Status

Live paper trading (no real money) on an MT5 demo account. Not investment
advice — this is a personal project for exploring systematic strategies.

## Stack

- **Platform**: MetaTrader 5 desktop (Windows), driven via the `MetaTrader5`
  Python package
- **Language**: Python 3.11+
- **Dashboard**: Streamlit, with live-updating candlestick charts built on
  TradingView's open-source `lightweight-charts` library (via a small local
  JSON API — Streamlit's own chart widgets can't update without a visible
  redraw)
- **Ledger**: SQLite, one file per portfolio

## Strategies

| Strategy | Idea |
|---|---|
| **Donchian Breakout** | Trades in the direction of a break above/below the recent N-bar high/low channel — classic Turtle-style trend following. |
| **MACD Trend** | Enters on a fast/slow EMA (MACD) crossover, filtered to agree with the zero line. |
| **No-Wick Retrace** | Looks for a one-sided ("wickless") candle against a 50-EMA trend, then enters on a retrace to that level. |

Exit management on top of each strategy's own SL/TP:

- **Ratchet trailing stop** — once a trade reaches a configurable profit
  threshold (in R-multiples of its own risk), the stop is moved forward to
  lock in a fraction of the gain, re-evaluated every tick.
- **Skyrocket exit** — closes a trade early (at market) if the current
  candle is an outsized, favorably-moving spike relative to recent average
  range, instead of waiting for a fixed target.
- **Start/stop control** — a dashboard button pauses new signal execution
  per portfolio without touching already-open positions (still fully
  managed — trailing stop, reconciliation, etc. keep running).

## Repo layout

```
metatraderbot/
├── config/
│   ├── portfolio_ftmo.yaml   ← FTMO portfolio: pairs, strategies, risk, exits
│   └── portfolio_yolo.yaml   ← YOLO portfolio (same shape, more aggressive)
├── scripts/
│   ├── run_paper.py          ← main bot entrypoint (one process per portfolio)
│   ├── dashboard.py          ← Streamlit dashboard
│   ├── chart_api.py          ← local JSON API feeding the live JS charts
│   ├── start_all.ps1         ← idempotent recovery script (MT5 + both bots + dashboard)
│   └── assets/               ← dashboard logo/icon
├── src/metatraderbot/
│   ├── strategy.py            ← No-Wick strategy + shared types
│   ├── strategies_donchian.py
│   ├── strategies_macd.py
│   ├── portfolio_runner.py    ← polling loop, position management, exits
│   ├── mt5_engine.py          ← MT5 connection/order wrapper
│   ├── ledger.py              ← SQLite trade/signal/heartbeat logging
│   ├── backtest.py            ← bar-close backtest engine
│   └── instruments.py         ← per-symbol contract specs
├── data/                      ← price history + ledgers (gitignored)
├── logs/                      ← runtime logs (gitignored)
└── tests/
```

## Setup

### 1. MT5 desktop

Install MetaTrader 5 and log into a **demo** account. Leave the terminal
running and AlgoTrading enabled (green button in the toolbar) whenever the
bots are running.

### 2. Python environment

```powershell
cd metatraderbot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Credentials

```powershell
copy .env.example .env
# then fill in MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
```

`.env` is gitignored — never commit real credentials.

### 4. Run everything

```powershell
scripts\start_all.ps1
```

This is idempotent — safe to re-run any time; it only starts whatever isn't
already running (MT5 terminal, FTMO bot, YOLO bot, chart API, dashboard).

Then open the dashboard at **http://localhost:8501**.

### Optional: auto-start on login

A shortcut to `start_all.ps1` in the Windows Startup folder (or a desktop
shortcut for on-demand runs) brings everything back after a reboot without
any manual steps.

## Risk disclosure

This trades real MT5 order flow against a demo account, but the strategies
and risk parameters here are tuned for an aggressive, experimental style
(the YOLO portfolio explicitly so). Nothing in this repo is financial
advice, and none of it should be pointed at a real-money account without
independently understanding and accepting the risk.
