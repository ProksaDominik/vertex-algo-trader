# Vertex Algo Bot — installation & transfer

Moving this app to another Windows PC. The folder is self-contained: every path
resolves from the folder's own location, so it can live anywhere on the target
machine.

---

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 (the MetaTrader5 Python package is Windows-only) |
| Python | **3.11 or 3.12** — not 3.13+, which has no MetaTrader5 wheel |
| MetaTrader 5 | desktop terminal, installed and logged in |
| Disk | ~2 GB (mostly historical price data) |

---

## Transfer

### 1. Copy the folder

Copy the whole `metatraderbot` directory to the new PC. Anywhere is fine —
`C:\trading\metatraderbot`, `D:\vertex`, it does not matter.

**Leave behind** (they are machine-specific and get rebuilt):

```
.venv\           virtual environment — rebuilt by Setup.ps1
logs\            runtime logs
__pycache__\     compiled bytecode
```

**Bring** `data\` if you want your price history and trade ledgers; the app
recreates the folder empty otherwise and re-downloads history on demand.

### 2. Install MetaTrader 5

Install the terminal, log into the account, and **start it once manually** so it
stores the login. Leave AlgoTrading enabled (the toolbar button is green).

### 3. Run setup

```powershell
cd <folder>
.\Setup.ps1
```

Creates `.venv`, installs pinned dependencies, creates `data\` and `logs\`,
copies `.env.example` to `.env`, and verifies every import. It refuses to
continue on Python 3.13+ rather than failing later with a compiler error.

If PowerShell blocks the script:

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup.ps1
```

### 4. Add credentials

Edit `.env`:

```
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
```

`.env` is gitignored. Never commit it.

### 5. Start

```powershell
.\Vertex.ps1
```

Starts MetaTrader 5 (if not running), both bots, the chart API and the app, then
opens <http://localhost:8501>.

---

## Commands

```powershell
.\Vertex.ps1              # start everything and open the app
.\Vertex.ps1 -Status      # what is running
.\Vertex.ps1 -Stop        # stop everything
.\Vertex.ps1 -Restart     # stop then start
.\Vertex.ps1 -NoBots      # UI only, no live trading
.\Vertex.ps1 -Port 8600   # different port
```

---

## Verify the install

Work through these on the new machine before trusting it with an account.

1. **`.\Vertex.ps1 -Status`** — every component reports running.
2. **App loads** at localhost:8501 and the header shows a status pill.
3. **Account tab shows real figures** — equity matches the MT5 terminal.
4. **Positions tab** lists open trades, or says none.
5. **Broker connection** — the header must not show `NO CONNECTION`.

If the header shows `NO CONNECTION`: MetaTrader 5 is not running, not logged in,
or AlgoTrading is disabled.

---

## Broker-specific behaviour worth knowing

These were found the hard way on the original machine and are **per broker** — a
different broker may behave differently, so re-check them after a move.

| Behaviour | Why it matters |
|---|---|
| `order_send` needs `**request`, not `order_send(request)` | positional form silently fails on package 5.0.5735 |
| Order comments must be **≤ 29 chars** | 30+ makes `order_send` return `None` with a misleading error |
| `retcode=0` means success **only with a non-zero ticket** | a rejection also returns 0; ticket 0 means it did not happen |
| Contract size comes from the **terminal**, not `instruments.py` | this broker prices gold at 10 oz/lot, the textbook value is 100 |
| Closes fail silently when the market is shut | returns `retcode=0 / "Done"` and closes nothing |

The code already handles all five. They are listed so a future failure on a new
broker is recognisable rather than mysterious.

---

## Lucid Trading (futures) through NinjaTrader 8

Lucid is a **futures-only** prop firm: it offers 34 CME contracts, no forex or
CFDs, and has **no MetaTrader 5 connection**. Tradovate's API is closed to prop
accounts, so the bots reach Lucid through NinjaTrader 8, using a small add-on
(`ninjatrader\VertexBridge.cs`) that they talk to on 127.0.0.1. With
`VERTEX_BROKER=ninjatrader` the app runs the same three books as on MT5, all on
NinjaTrader, and the dashboard shows exactly three pages:

| Page | Bot | NinjaTrader account | Files |
|---|---|---|---|
| **Trial** | `run_lucidpro_trial.py` in futures mode: repeated $50K LucidPro evaluation attempts, London Breakout on 6C/6B/6S/MGC | `NT_ACCOUNT` (Sim101) | `state_lucid_futures.json`, `lucid_futures_trades.jsonl` |
| **Funded** | `run_lucid_futures_funded.py`: the funded-stage bot -- intraday roster from `futures_roster_search.py`, 0.5% risk tapered near the MLL, runs to payout or blow-up | `NT_FUNDED_ACCOUNT` (Sim102) | `state_lucid_futures_funded.json`, `lucid_futures_funded_trades.jsonl` |
| **Lab** | `run_lab.py` in futures mode: 296 strategy configs forward-tested on 12 Lucid contracts, scored under Lucid's hours. Places no orders | none | `lab_futures_state.json`, `lab_futures_signals.jsonl` |

### Set up

1. **Install NinjaTrader 8** from ninjatrader.com (free for simulation) and
   start it once. Its built-in **Simulation** connection supplies live futures
   prices and history for the simulator accounts.
2. **If Windows blocks the compile** ("An Application Control policy has
   blocked this file"): Smart App Control stops NinjaTrader loading any
   compiled script. Turn it off in Windows Security → App & browser control →
   Smart App Control (reversible on Windows 11 builds 26100.8116 / 26200.8116+).
3. **Install the add-on:**
   ```powershell
   .\ninjatrader\Install-VertexBridge.ps1
   ```
   Then in NinjaTrader: **New → NinjaScript Editor → F5** (compile). Open
   **New → NinjaScript Output**; it should say
   `VertexBridge 1.1 listening on 127.0.0.1:45973`. The bots refuse to trade
   through an older bridge. (Port 45973, not 36973: NinjaTrader's own ATI
   server already holds 36973.)
4. **Create the funded bot's simulation account**, `Sim102`, in NinjaTrader's
   Control Center (Accounts tab). The funded bot waits until it exists.
5. **Settings** in `.env`:
   ```
   VERTEX_BROKER=ninjatrader
   NT_ACCOUNT=Sim101
   NT_FUNDED_ACCOUNT=Sim102
   ```
6. **Start** with `.\Vertex.ps1` (or the desktop icon). In futures mode it
   starts the three books and the dashboard; no MetaTrader component runs.

### The funded roster

The funded bot trades the legs in `data\futures\funded_roster.json`, written by

```powershell
.venv\Scripts\python.exe scripts\futures_roster_search.py
```

It pulls ~15 months of 15m and ~20 months of 1h history for 12 Lucid
contracts from NinjaTrader and tests every lab strategy config under Lucid's
rules (flat by the cutoff, Lucid's commissions, one tick of slippage). A leg
qualifies only if it is positive in both halves of the data AND with costs
doubled; at most one leg per contract and two equity-index legs. The full
table is in `data\futures\roster_search.csv`. Re-run it to refresh the roster,
then restart the funded bot.

### Connecting a real Lucid account

NinjaTrader reaches Lucid through **CQG with the Tradovate credentials** from
your Lucid dashboard -- Lucid lists NinjaTrader under CQG, not Rithmic, so
choose the non-Rithmic version of a plan. Sign in once at trader.tradovate.com
(sign the data agreement, CME certification as Non-Professional); then in
NinjaTrader tick **Tools → Settings → General → Multi-provider**, restart, and
add a **"NinjaTrader"** connection with those credentials and **Account Type:
Simulation**.

### Before a paid evaluation

The ~70% pass rates in `run_lucidpro_trial.py` were measured on **CFD prices
with unlimited size**. They do not carry over to futures under Lucid's limits.
Run on **Sim101 with Lucid's live data** until the futures pass rate is known;
that run is the validation. Only then:

```
NT_ACCOUNT=<your Lucid account, exactly as NinjaTrader shows it>
LUCID_LIVE_CONFIRM=<the same account name>
```

Without `LUCID_LIVE_CONFIRM` matching, the bot refuses to trade any
non-simulation account. On a real Lucid account it **stops opening trades** as
soon as the attempt passes or fails (a real evaluation cannot be retried the way
the virtual ones are). To resume, remove `"halted"` from the state file.

### What the bot enforces for Lucid

| Rule | How |
|---|---|
| Flat by **4:45 PM ET**, Mon–Fri | no new entries from 4:30, flattens itself at 4:40 (holiday early closes in `lucid_rules.EARLY_CLOSES`) |
| **4 minis / 40 micros** (LucidPro 50K) | contracts sized down to fit; also enforced inside the engine, counting manual positions too |
| $2,000 **Max Loss Limit** | a trade's full stop-out must fit in 90% of the room left above the MLL, so a stop can actually be reached before Lucid fails the account (`CLAMP_RISK_TO_MLL_ROOM`) |
| **Approved products only** | `futures.LUCID_APPROVED`; the basket maps GBPUSD→6B, USDCAD→6C, USDCHF→6S, XAUUSD→MGC (GBPJPY/GBPCHF have no Lucid contract) |
| Contract rolls | front month chosen by date rules in `futures.py`, re-checked every trading day |

### Things to know

- **Extend `lucid_rules.EARLY_CLOSES` and `CLOSED_DAYS` every year.** They hold
  the 2026 holidays, set a little earlier than the exchange halts.
- **One bot position per contract.** Futures accounts net per instrument, so the
  bridge refuses a second ticket on the same contract. Don't trade the bot's
  contracts by hand on the same account.
- **Don't restart NinjaTrader with bot positions open.** Their stops stay at the
  exchange, but after a restart the bridge may not recognise its own orders.
- Stops and targets are real exchange orders (OCO), placed the moment the entry
  fills. If a stop is rejected or disappears, the bridge closes the position at
  market rather than leave it unprotected.
- The MT5 funded bot (`run_lucidpro_funded.py`) is **not** ported: its roster
  holds daily/4h trades for days and trades BTC, and Lucid allows neither. The
  futures funded bot is a new, intraday bot (`run_lucid_futures_funded.py`).
- The funded bot's Max Loss Limit **locks at $50,100** once an end-of-day
  balance reaches $52,100. That rule comes from secondary sources, not Lucid's
  own terms -- check it against your funded account's rules.
- ORB, VWAP-reversion and time-of-day strategies never signalled on live bars
  before 2026-09-29 (they read a column only backtests have). Fixed; the lab's
  earlier samples for those families are empty, not negative.

---

## Layout

```
metatraderbot\
├─ Vertex.ps1              app launcher
├─ Setup.ps1               one-time machine setup
├─ requirements.txt        pinned dependencies
├─ .env                    credentials (not in git)
├─ scripts\
│   ├─ dashboard.py        the app UI
│   ├─ chart_api.py        local price feed for charts
│   ├─ run_lucidpro_funded.py
│   └─ run_lucidpro_trial.py
├─ src\metatraderbot\
│   ├─ app_theme.py        design system
│   ├─ app_stats.py        statistics engine
│   ├─ app_reset.py        account reset / archive / restore
│   ├─ mt5_engine.py       broker layer (MetaTrader 5)
│   ├─ nt_engine.py        broker layer (NinjaTrader 8 -> Lucid futures)
│   ├─ futures.py          CME contract specs, Lucid's product list, rolls
│   ├─ lucid_rules.py      Lucid's session times, contract cap, trade date
│   └─ strategies_*.py
├─ ninjatrader\
│   ├─ VertexBridge.cs     NinjaTrader add-on the bot talks to
│   └─ Install-VertexBridge.ps1
├─ data\                   ledgers, state, price history, archives
└─ logs\                   runtime logs
```

---

## Resetting an account

App → **System** → **Danger zone**. Type `RESET` to unlock the button.

The reset closes open positions, archives the state file and trade ledger into
`data\archive\`, and starts the book from $50,000 with clean history.

Two safeguards worth knowing:

- **Nothing is deleted.** Archives are restore points; the same panel restores
  any of them.
- **It aborts if positions cannot be closed.** Closing does not work while the
  market is shut, and resetting state with positions still open would orphan
  them — the bot would forget it owns trades the broker still holds. Reset while
  the market is open.
