"""
Live web dashboard for the paper trading bot.

Usage:
    python -m streamlit run scripts/dashboard.py

Opens in browser at http://localhost:8501. The live section re-renders on a
timer via st.fragment(run_every=...) — only that fragment reruns, not the
whole script, so there's no full-page flash/scroll-jump on every tick.

Reads directly from the SQLite ledgers written by run_paper.py, so the bot
doesn't need to be running (though liveness will show "stopped" if it isn't).
"""
from __future__ import annotations

import base64
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml


class _DropComponentsDeprecation(logging.Filter):
    """Silence Streamlit's 'replace st.components.v1.html with st.iframe' notice.

    Emitted once per render per chart, so with several live charts refreshing
    every few seconds it buries every real error in the log. The advice does
    not apply here: st.iframe takes a URL or Path, while every call site in
    this file passes a self-contained HTML STRING (the lightweight-charts
    widgets built by build_live_chart_html / build_strategy_curves_html).
    There is no non-hacky string-to-iframe path, so the working calls stay and
    the notice is filtered. Revisit if Streamlit actually removes the API.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "st.components.v1.html" not in record.getMessage()


logging.getLogger("streamlit").addFilter(_DropComponentsDeprecation())
for _n in ("streamlit.deprecation_util", "streamlit.elements.lib.policies",
           "streamlit.runtime.scriptrunner.script_runner"):
    logging.getLogger(_n).addFilter(_DropComponentsDeprecation())

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot.mt5_engine import MT5Engine  # noqa: E402 — reuse filling-mode-aware close logic
from metatraderbot import app_theme as vx  # noqa: E402 — design system
from metatraderbot import app_stats as vxs  # noqa: E402 — statistics engine
from metatraderbot import app_reset as vxr  # noqa: E402 — account reset
from metatraderbot import app_market as vxm  # noqa: E402 — market open/closed
from metatraderbot import app_vertex as vxv  # noqa: E402 — Vertex, the voice butler, in the dashboard
from metatraderbot import vertex_stats as vxst  # noqa: E402 — what Vertex hears and says


@st.cache_data(ttl=30, show_spinner=False)
def broker_realised(magic_lo: int, magic_hi: int, since_iso: str) -> dict:
    """Realised P&L straight from the BROKER for a magic range.

    The trade ledger (data/*_trades.jsonl) only records closes the RUNNER
    performed and reconciled. A position closed by hand -- including from this
    dashboard's own close button -- never reaches it, because close_ticket()
    calls the engine directly. On 2026-08-24 that made $1,928 of realised
    profit invisible: 21 positions were closed manually and the panel kept
    reporting the two the bot had closed itself.

    The broker's deal history is the only source that cannot disagree with the
    account, so headline money is read from here and the JSONL is used only for
    per-trade detail.
    """
    try:
        import MetaTrader5 as _mt5
        if not _mt5.initialize():
            return {"ok": False}
        since = datetime.fromisoformat(since_iso)
        deals = _mt5.history_deals_get(since, datetime.now(timezone.utc)
                                       + timedelta(days=2)) or []
        # entry == DEAL_ENTRY_OUT: the closing side, which carries the P&L.
        closed = [d for d in deals if d.entry == 1 and magic_lo <= d.magic <= magic_hi]
        pnl = sum(d.profit + d.commission + d.swap for d in closed)
        wins = sum(1 for d in closed if (d.profit + d.commission + d.swap) > 0)
        return {"ok": True, "n": len(closed), "pnl": float(pnl), "wins": wins,
                "losses": len(closed) - wins}
    except Exception:
        return {"ok": False}


@st.cache_data(ttl=20, show_spinner=False)
def market_status(symbols: tuple[str, ...]) -> dict:
    """Live market state, cached briefly — the header re-renders every few
    seconds and this costs one MT5 round trip per symbol."""
    try:
        eng = _chart_engine()
        s = vxm.check(eng, list(symbols))
        return {
            "label": s.label, "open": s.open, "n_open": s.n_open, "n_total": s.n_total,
            "next_open": s.next_open.isoformat() if s.next_open else None,
            "symbols": [{"symbol": x.symbol, "open": x.open, "age": x.tick_age_s,
                         "mode": x.trade_mode, "reason": x.reason} for x in s.symbols],
        }
    except Exception:
        return {"label": "UNKNOWN", "open": False, "n_open": 0, "n_total": 0,
                "next_open": None, "symbols": []}


def market_pill(ms: dict) -> str:
    colour = {"OPEN": vx.OK, "PARTIAL": vx.WARN, "CLOSED": vx.TEXT_FAINT,
              "NO CONNECTION": vx.DANGER}.get(ms["label"], vx.TEXT_FAINT)
    txt = f'MARKET {ms["label"]}'
    if ms["label"] == "PARTIAL":
        txt += f' {ms["n_open"]}/{ms["n_total"]}'
    return vx.pill(txt, colour)


def render_market_panel(ms: dict) -> None:
    """Per-symbol session table plus the estimated reopen."""
    if not ms["symbols"]:
        st.markdown(vx.banner("Market state unavailable — MetaTrader 5 unreachable.",
                              "warn"), unsafe_allow_html=True)
        return
    if ms["label"] == "CLOSED" and ms["next_open"]:
        nxt = datetime.fromisoformat(ms["next_open"])
        st.markdown(vx.banner(
            f"<b>Market closed.</b> No orders can be placed or closed until it reopens"
            f" — estimated <b>{nxt:%a %d %b %H:%M} UTC</b>, about"
            f" {vxm.until(nxt)} away. The estimate is the conventional FX week open;"
            f" brokers and holidays shift it.", "info"), unsafe_allow_html=True)
    rows = [{"Symbol": s["symbol"],
             "Session": "open" if s["open"] else "closed",
             "Last tick": vxm.humanise_age(s["age"]),
             "Detail": s["reason"]} for s in ms["symbols"]]
    df = pd.DataFrame(rows)

    def _sess(v):
        return (f"color:{vx.OK};font-weight:600" if v == "open"
                else f"color:{vx.TEXT_FAINT}")

    st.dataframe(df.style.map(_sess, subset=["Session"]),
                 width="stretch", hide_index=True)

# Target account sizes per bot — this is what P&L % is computed against,
# regardless of the actual MetaQuotes-Demo balance ($100k default).
#
# FTMO removed 2026-08-15 -- that paper book was stopped and swapped for
# run_lucidpro_funded.py (see LUCIDPRO_FUNDED below, a different shape of
# bot like LUCIDPRO_TRIAL, not a ledger-based one). YOLO stopped and removed
# from the dashboard 2026-08-16 (consolidated down to only the two LucidPro
# bots). FTMO_4day is untouched in these dicts but was already unreachable
# via the sidebar radio below before this change, and its process is also
# stopped -- both this and YOLO are dead entries kept only in case a
# ledger-based book gets reactivated later.
TARGET_ACCOUNT = {
    "FTMO_4DAY": 10_000.0,
}

CONFIG_FILE = {
    "FTMO_4DAY": "portfolio_ftmo_4day.yaml",
}

# Ledger filename per bot -- kept as an explicit dict (not derived from
# each YAML's own ledger.path) so the sidebar can resolve it without a
# config-file read on every rerun.
LEDGER_FILE = {
    "FTMO_4DAY": "paper_ledger_ftmo4day.sqlite",
}


def get_control_path(bot_choice: str) -> Path:
    """Path to the shared start/stop flag file — same naming convention
    run_paper.py uses (derived from the config filename's stem), so both
    processes agree on the same file without any other coordination."""
    # .get, not []: CONFIG_FILE only ever held the legacy FTMO/YOLO bots, so any
    # LucidPro selection raises here. Unreachable in a real run -- those views
    # end in st.stop() -- but st.stop() does not halt a bare import, so this
    # crashed on import and took the whole pytest collection down with it.
    stem = Path(CONFIG_FILE.get(bot_choice, CONFIG_FILE["FTMO_4DAY"])).stem
    return ROOT / "data" / f"control_{stem}.json"


def read_trading_enabled(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        return bool(json.loads(path.read_text()).get("trading_enabled", True))
    except (OSError, ValueError):
        return True


def write_trading_enabled(path: Path, enabled: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trading_enabled": enabled}))

# All open/closed timestamps are shown in Slovakia local time (CET/CEST,
# handles DST automatically), regardless of what clock they were recorded in.
SLOVAKIA_TZ = ZoneInfo("Europe/Bratislava")


def to_slovakia(dt) -> datetime:
    """Convert a true-UTC datetime/Timestamp (aware or naive-as-UTC) to
    Slovakia local time for display."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(SLOVAKIA_TZ)


UP = "▲"
DOWN = "▼"


def _dot(color: str) -> str:
    """Small colored dot matching the app's blue(up)/red-pink(down) palette —
    used instead of emoji (🟢🔴) which render with their own fixed platform
    colors and can't be restyled to match the theme."""
    return f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;background:{color};margin-right:4px;'></span>"


def dir_badge(is_long: bool) -> str:
    color = "#4c6ef5" if is_long else "#f5426c"
    arrow = UP if is_long else DOWN
    return f"{_dot(color)}{arrow}"

# Plain-English explanation of each strategy family, shown next to every trade
# so it's clear which logic opened it and why, without digging into the code.
STRATEGY_INFO = {
    "DonchianBreakout": {
        "icon": "📐",
        "label": "Donchian Breakout",
        "explain": "Watches the highest high / lowest low of the last 20 candles. "
                   "When price pushes outside that channel, it trades in the breakout "
                   "direction, betting the move keeps running.",
    },
    "MACDTrend": {
        "icon": "〰️",
        "label": "MACD Trend",
        "explain": "Tracks a fast vs. slow moving-average crossover (MACD). When the "
                   "fast line crosses the slow one, that's read as a fresh momentum "
                   "shift and it trades in that direction.",
    },
    "NoWick": {
        "icon": "🕯️",
        "label": "No-Wick Retrace",
        "explain": "Looks for a one-sided candle (almost no wick) while price is "
                   "trending — a sign of strong conviction — then waits for a small "
                   "pullback to that level before entering with the trend.",
    },
}


_LABEL_ICON = {"Donchian": "📐", "MACD": "〰️", "BOS": "🏛️", "VWAPReversion": "📊",
               "SweepReclaim": "🌊", "TODReversion": "🕐", "NoWick": "🕯️",
               "Bollinger": "🎯", "RSI": "〽️", "Stochastic": "📉", "SMACross": "✖️", "ORB": "🚪"}


def strategy_cell_html(strategy_type: str, config_name: str, show_explain: bool = True) -> str:
    """Small HTML block: strategy display name + icon (+ optional one-line
    explanation). Resolves the strategy from the config/leg NAME (via
    strategy_label) so it works for both the new Strategy__SYMBOL legs and
    the old per-pair names — never shows a bare '?'."""
    info = STRATEGY_INFO.get(strategy_type)
    if info:
        icon, label, explain = info["icon"], info["label"], info["explain"]
    else:
        label = strategy_label(config_name)
        icon = _LABEL_ICON.get(label, "📊")
        explain = ""
    html = f"<span style='font-weight:600'>{icon} {label}</span>"
    if show_explain and explain:
        html += f"<br><small style='opacity:0.65'>{explain}</small>"
    html += f"<br><small style='opacity:0.4'>{config_name}</small>"
    return html


def _expand_config_legs(configs: list) -> list:
    """Expand per-strategy entries (symbols:[] + magic_base:) into per-symbol
    legs — same rule the runner uses (name = Strategy__SYMBOL, magic =
    magic_base + index). Legacy single-symbol entries pass through. Kept
    inline so the dashboard doesn't import the MT5-dependent runner."""
    out = []
    for entry in configs or []:
        syms = entry.get("symbols")
        if syms:
            base = int(entry.get("magic_base", 0))
            for i, sym in enumerate(syms):
                leg = {k: v for k, v in entry.items() if k not in ("symbols", "magic_base")}
                leg["symbol"] = sym
                leg["magic_number"] = base + i
                leg["name"] = f"{entry.get('name', 'strat')}__{sym}"
                leg["strategy_group"] = entry.get("name", "")
                out.append(leg)
        else:
            leg = dict(entry)
            leg.setdefault("strategy_group", entry.get("name", ""))
            out.append(leg)
    return out


def load_slot_names(bot_choice: str) -> dict[int, dict]:
    """magic_number -> config info, so live MT5 positions can be labeled."""
    path = ROOT / "config" / CONFIG_FILE[bot_choice]
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        c["magic_number"]: {"name": c["name"], "symbol": c["symbol"],
                            "timeframe": c["timeframe"], "strategy": c["strategy"]}
        for c in _expand_config_legs(cfg.get("configs", []))
    }


def load_strategy_by_config_name(bot_choice: str) -> dict[str, str]:
    """config_name -> strategy type, for ledger rows (closed/signals/orders)
    which only store the config name, not the live magic number."""
    path = ROOT / "config" / CONFIG_FILE[bot_choice]
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {c["name"]: c["strategy"] for c in _expand_config_legs(cfg.get("configs", []))}


def get_live_open_positions(slots: dict[int, dict],
                             block_size: int = 1000) -> pd.DataFrame | None:
    """Live open positions straight from MT5 (floating P&L, current price).

    Returns None (not an empty frame) when MT5 can't be reached, so callers
    can fall back to the heartbeat-based count instead of showing "0 open"
    when the truth is "unknown".

    `block_size` sets the width of the catch-all magic range used for
    positions whose magic is not in `slots`. That fallback exists so a
    position opened under an OLD magic (after a config restructure) still
    shows up in its own book rather than vanishing.

    It MUST match how far apart the books' magic blocks actually are.
    FTMO (2000s) and YOLO (3000s) are 1000 apart, hence the default. The two
    LucidPro books are not: trial is 9100-9102 and funded is 9200-9207, so a
    1000-wide window rounds both down to 9000 and each book then claimed the
    other's positions -- the funded book's open trades showed up on the trial
    dashboard and vice versa. Those callers pass block_size=100.
    """
    if not _MT5_AVAILABLE or not slots:
        return None
    if not mt5.initialize():
        return None
    positions = mt5.positions_get() or []
    book_base = (min(slots) // block_size) * block_size if slots else None
    rows = []
    for p in positions:
        if p.magic in slots:
            strat_name = slots[p.magic]["name"]
            strat_type = slots[p.magic].get("strategy", "")
        elif book_base is not None and book_base <= p.magic < book_base + block_size:
            strat_name = f"(magic {p.magic})"
            strat_type = ""
        else:
            continue
        rows.append({
            "Ticket": p.ticket,
            "Magic": p.magic,
            "Strategy": strat_name,
            "StrategyType": strat_type,
            "Symbol": p.symbol,
            "Dir": UP if p.type == mt5.POSITION_TYPE_BUY else DOWN,
            "Volume": p.volume,
            "Entry": p.price_open,
            "Current": p.price_current,
            "SL": p.sl,
            "TP": p.tp,
            "Floating P&L": p.profit,
            "Opened": datetime.fromtimestamp(p.time, tz=timezone.utc),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "Ticket", "Magic", "Strategy", "StrategyType", "Symbol", "Dir", "Volume", "Entry", "Current",
            "SL", "TP", "Floating P&L", "Opened",
        ])
    return pd.DataFrame(rows).sort_values("Opened", ascending=False)


# 2026-08-20 -- order_send was silently failing on BOTH LucidPro bots from
# go-live until 15:33 UTC today: mt5_engine.py called order_send(request)
# positionally (this package version requires **request) AND truncated the
# order comment to 31 chars (this broker's real limit is 29). Either alone
# makes every order fail with a misleading client-side error. So there is
# essentially no executed history to show for the last 3 weeks. This renders
# what the roster WOULD have traded instead -- live strategy code, exact live
# params, replayed against real MT5 bars via the project's own Backtest engine
# (scratch/weekly_reconstruction.py). Replay, not fills; labeled as such
# everywhere, because a plausible-looking equity number nobody actually earned
# is the single most dangerous thing this dashboard could display.
def sim_baseline(bot_key: str) -> dict | None:
    """The 3-week simulated lifecycle (scratch/simulate_3weeks.py) used as the
    OPENING BALANCE the live counters continue from, so the dashboard reads as
    one unbroken history instead of splitting "what would have happened" from
    "what has happened since execution was fixed".

    Kept in the presentation layer ONLY. It is deliberately NOT written into
    data/state_lucidpro_*.json, because the live runners read those files to
    make real risk decisions -- peak_equity there sets the MLL kill-switch
    floor and drives the funded book's risk taper, so seeding it with
    simulated equity would move a real safety limit against simulated gains
    the account never made.
    """
    path = ROOT / "scratch" / "simulate_3weeks_result.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    d = raw.get(bot_key)
    if not d:
        return None
    # window_start/generated_at live at the top level, but every caller wants
    # them alongside the per-bot figures.
    return {**d, "window_start": raw.get("window_start"),
            "generated_at": raw.get("generated_at")}



def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Put the reconstructed and live trade shapes on one set of column names.

    The two sources disagree: reconstruction rows carry leg/symbol/closed_at,
    the live JSONL carries label/sym/closed_ts. Rather than special-case every
    consumer, coalesce once here so the tables and charts below only ever see
    Result / Closed / Strategy / Symbol / Dir / R / P&L.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    def pick(*names):
        for n in names:
            if n in d.columns:
                return d[n]
        return pd.Series([None] * len(d), index=d.index)

    out = pd.DataFrame(index=d.index)
    out["Closed"] = pd.to_datetime(pick("closed_ts", "closed_at"), errors="coerce", utc=True)
    out["Opened"] = pd.to_datetime(pick("opened_ts", "entered_at"), errors="coerce", utc=True)
    out["Strategy"] = pick("leg", "label", "config_name")
    out["Symbol"] = pick("sym", "symbol")
    out["TF"] = pick("timeframe")
    out["Dir"] = pick("direction").map(lambda v: _dir_icon(v) if pd.notna(v) else "")
    out["Entry"] = pd.to_numeric(pick("entry_price"), errors="coerce")
    out["Exit"] = pd.to_numeric(pick("exit_price"), errors="coerce")
    out["SL"] = pd.to_numeric(pick("stop_loss", "sl"), errors="coerce")
    out["TP"] = pd.to_numeric(pick("take_profit", "tp"), errors="coerce")

    # Prefer the precomputed hold time; fall back to the timestamp difference
    # so live JSONL rows (which carry only opened_ts/closed_ts) also get one.
    hold_h = pd.to_numeric(pick("hold_hours"), errors="coerce")
    delta_h = (out["Closed"] - out["Opened"]).dt.total_seconds() / 3600.0
    out["HoldH"] = hold_h.fillna(delta_h)

    def _fmt_hold(h):
        if pd.isna(h):
            return "—"
        if h < 1:
            return f"{h * 60:.0f}m"
        if h < 48:
            return f"{h:.1f}h"
        return f"{h / 24:.1f}d"

    out["Held"] = out["HoldH"].map(_fmt_hold)
    out["Reason"] = pick("exit_reason")
    out["R"] = pd.to_numeric(pick("r_multiple"), errors="coerce")
    out["P&L"] = pd.to_numeric(pick("pnl_usd"), errors="coerce")
    out["Result"] = out["P&L"].map(
        lambda v: "" if pd.isna(v) else ("✅ WIN" if v > 0 else ("❌ LOSS" if v < 0 else "➖ FLAT"))
    )
    ordered = ["Closed", "Strategy", "Symbol", "Dir", "Entry", "Exit", "Held",
               "Reason", "R", "P&L", "Result", "Opened", "SL", "TP", "TF", "HoldH"]
    out = out[[c for c in ordered if c in out.columns]]
    return out.sort_values("Closed", ascending=False)


def _result_color(val) -> str:
    if not isinstance(val, str):
        return ""
    if val.startswith("✅"):
        return f"color: {_UP}; font-weight: 700"
    if val.startswith("❌"):
        return f"color: {_DOWN}; font-weight: 700"
    return f"color: {_AXIS_LABEL}"


_SYMBOL_MAP = {"NAS100": "USTEC", "JP225": "JPN225", "BTCUSD": "BTC", "ETHUSD": "ETH"}

# Each leg's take-profit in R, mirroring ROSTER in run_lucidpro_funded.py.
# Used to reconstruct a trade's ORIGINAL stop from its (never-moving) TP once
# the live stop has been trailed away from it.
_TP_R_BY_LEG = {
    "Donchian_NAS100_1R": 1.0, "Donchian_XAUUSD_1R": 1.0,
    "Supertrend_JP225_1R": 1.0, "Donchian_XAUUSD_6R": 6.0,
    "MACD_XAUUSD_2R": 2.0,
    "Donchian_BTCUSD_1R": 1.0, "Donchian_BTCUSD_3R": 3.0,
    "Ichimoku_BTCUSD_2R": 2.0,
}


@st.cache_resource(show_spinner=False)
def _chart_engine():
    """One shared MT5 connection for drawing historical trade charts."""
    eng = MT5Engine(symbol_map=_SYMBOL_MAP)
    eng.connect()
    return eng


@st.cache_data(ttl=300, show_spinner=False)
def _bars_for_window(symbol: str, timeframe: str, start_iso: str, end_iso: str,
                     pad: int = 30) -> pd.DataFrame | None:
    """Bars covering a closed trade's lifetime, plus `pad` bars either side.

    get_bars only pulls the most recent N bars, so ask for enough to reach
    back past the entry and then slice. Cached: chart views re-render on every
    fragment tick and this is an MT5 IPC round-trip.
    """
    try:
        eng = _chart_engine()
        start = pd.Timestamp(start_iso)
        end = pd.Timestamp(end_iso)
        per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48,
                   "1h": 24, "4h": 6, "1d": 1}.get(timeframe, 24)
        span_days = max(1.0, (pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400.0)
        need = int(span_days * per_day) + pad * 2 + 50
        bars = eng.get_bars(symbol, timeframe, min(need, 5000))
        if bars is None or bars.empty:
            return None
        step = bars.index.to_series().diff().median()
        lo = start - (step * pad if pd.notna(step) else pd.Timedelta(0))
        hi = end + (step * pad if pd.notna(step) else pd.Timedelta(0))
        return bars.loc[(bars.index >= lo) & (bars.index <= hi)]
    except Exception:
        return None  # charts are a nicety; never take the dashboard down for one


def render_closed_trade_chart(row: pd.Series, key: str = "") -> None:
    """Candlestick of one closed trade with entry/exit/SL/TP marked."""
    symbol = row.get("Symbol")
    tf = row.get("TF") or "1h"
    opened, closed = row.get("Opened"), row.get("Closed")
    if not symbol or pd.isna(opened) or pd.isna(closed):
        st.caption("Not enough information on this trade to draw a chart.")
        return
    bars = _bars_for_window(str(symbol), str(tf), opened.isoformat(), closed.isoformat())
    if bars is None or bars.empty:
        st.caption(f"No {tf} price history available for {symbol} over this window.")
        return

    won = (row.get("P&L") or 0) > 0
    fig = go.Figure(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"],
        low=bars["low"], close=bars["close"],
        increasing=dict(line=dict(color=_UP), fillcolor=_UP),
        decreasing=dict(line=dict(color=_DOWN), fillcolor=_DOWN),
        name=str(symbol),
    ))
    # Original stop vs where it ended up. The stop stored on a trade is the
    # LIVE one, already ratcheted by apply_trailing_stops, so on its own it
    # hides that the trade was ever protected. The original 1R stop is
    # recovered from the take-profit -- TP never moves, and each leg's TP sits
    # at a known R-multiple of its own risk, so |TP - entry| / tp_r gives the
    # original risk distance no matter how often the stop has since moved.
    entry, tp, sl = row.get("Entry"), row.get("TP"), row.get("SL")
    tp_r = _TP_R_BY_LEG.get(str(row.get("Strategy", "")))
    orig_sl = None
    if all(v is not None and pd.notna(v) for v in (entry, tp)) and tp_r:
        sign = 1.0 if str(row.get("Dir")) == UP else -1.0
        orig_sl = float(entry) - sign * (abs(float(tp) - float(entry)) / tp_r)
    moved = (orig_sl is not None and sl is not None and pd.notna(sl)
             and abs(float(sl) - orig_sl) > 1e-9)

    lines = [
        (entry, "Entry", _ENTRY_COLOR, "solid"),
        (row.get("Exit"), "Exit", _UP if won else _DOWN, "solid"),
        (tp, "TP", _TP_COLOR, "dot"),
    ]
    if moved:
        lines.append((orig_sl, "SL original", "#8b3a52", "longdash"))
        lines.append((sl, "SL trailed →", _SL_COLOR, "dot"))
    else:
        lines.append((sl, "SL", _SL_COLOR, "dot"))

    for price, label, color, dash in lines:
        if price is not None and pd.notna(price) and float(price) > 0:
            fig.add_hline(y=float(price), line_color=color, line_dash=dash,
                          line_width=1.5, opacity=0.9,
                          annotation_text=f"{label} {float(price):,.5f}",
                          annotation_position="right",
                          annotation_font=dict(color=color, size=11))
    for ts, label, color in ((opened, "IN", _ENTRY_COLOR),
                             (closed, "OUT", _UP if won else _DOWN)):
        fig.add_vline(x=ts, line_color=color, line_dash="dash", line_width=1.2,
                      opacity=0.7, annotation_text=label,
                      annotation_font=dict(color=color, size=11))
    fig.update_layout(
        height=430, margin=dict(l=10, r=70, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_AXIS_LABEL), showlegend=False,
        xaxis=dict(gridcolor=_GRID, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=_GRID),
    )
    st.plotly_chart(fig, width="stretch", key=f"tc_{key}")


def _add_stop_columns(open_df: pd.DataFrame) -> pd.DataFrame:
    """Add original stop + how much profit the trailed stop has locked in.

    The broker only reports the CURRENT stop, so a trailed position looks
    identical to one that never moved. Recover the original from the TP (which
    never moves) and express the gap in R, so "Locked R" answers the only
    question that matters: if this reverses right now, does it still lose?
    """
    if open_df is None or open_df.empty:
        return open_df
    d = open_df.copy()
    orig, locked = [], []
    for _, r in d.iterrows():
        tp_r = _TP_R_BY_LEG.get(str(r.get("StrategyType") or r.get("Strategy", "")))
        entry, tp, sl = r.get("Entry"), r.get("TP"), r.get("SL")
        if not tp_r or not tp or not entry or pd.isna(tp) or pd.isna(entry) or tp <= 0:
            orig.append(float("nan")); locked.append(float("nan")); continue
        sign = 1.0 if r.get("Dir") == UP else -1.0
        risk = abs(float(tp) - float(entry)) / tp_r
        o = float(entry) - sign * risk
        orig.append(o)
        locked.append(((float(sl) - float(entry)) * sign / risk) if (sl and risk) else float("nan"))
    d["SL orig"] = orig
    d["Locked R"] = locked
    return d


def render_close_controls(open_df: pd.DataFrame, book: str) -> None:
    """Per-position and close-all buttons for an open book.

    Close-all is behind a confirm checkbox: it is one click that realises every
    open position at market, and the surrounding page reruns every few seconds,
    so a mis-click has no undo. Per-position buttons are unguarded -- one
    position is a recoverable mistake.

    Worth remembering before using these routinely: several legs here carry
    positive expectancy only because their occasional large winners are allowed
    to run. Closing winners early while losers still travel the full distance to
    their stop inverts that arithmetic. These are an override for when you have
    a reason, not a substitute for the trailing stop.
    """
    st.markdown("**Manual close**")
    floating = float(open_df["Floating P&L"].sum())
    c1, c2 = st.columns([1, 3])
    with c1:
        confirm = st.checkbox("Enable close-all", key=f"cfm_{book}",
                              help="Guard against a mis-click; ticking this arms the button.")
    with c2:
        if st.button(f"⚠ Close ALL {len(open_df)} positions  (${floating:+,.2f})",
                     key=f"closeall_{book}", disabled=not confirm, width="stretch"):
            # Each close is a separate MT5 round trip with retries, so closing a
            # full book takes 30-60s. Without feedback the whole page appears
            # frozen on the one button where that is most alarming -- so show
            # progress per position and never leave the operator guessing
            # whether it worked.
            tickets = open_df["Ticket"].tolist()
            bar = st.progress(0.0, text=f"Closing 0/{len(tickets)}…")
            log_area = st.empty()
            ok, fail, lines = 0, [], []
            for i, t in enumerate(tickets, start=1):
                good, msg = close_ticket(int(t))
                if good:
                    ok += 1
                    lines.append(f"✓ {int(t)} closed")
                else:
                    fail.append(f"{int(t)}: {msg}")
                    lines.append(f"✗ {int(t)} — {msg}")
                bar.progress(i / len(tickets),
                             text=f"Closing {i}/{len(tickets)} · {ok} done, {len(fail)} failed")
                log_area.code("\n".join(lines[-12:]), language=None)
            bar.empty()
            if fail:
                st.error(f"Closed {ok}/{len(tickets)}. {len(fail)} failed:\n\n"
                         + "\n\n".join(fail)
                         + "\n\nA close returns success only with a real deal ticket, so "
                           "these genuinely did not close — most often because the market "
                           "is shut. Retry when it reopens.")
            else:
                st.success(f"Closed all {ok} positions. Realised P&L moves from Floating "
                           f"into your account balance — check Realised P&L above, not "
                           f"Floating, to see it.")
            st.rerun()

    for _, row in open_df.iterrows():
        b1, b2 = st.columns([3, 1])
        pnl = float(row["Floating P&L"])
        colour = _UP if pnl > 0 else (_DOWN if pnl < 0 else _AXIS_LABEL)
        b1.markdown(
            f"`{int(row['Ticket'])}` &nbsp; **{row['Strategy']}** {row['Symbol']} "
            f"{row['Dir']} &nbsp; vol {row['Volume']:.2f} &nbsp; "
            f"<span style='color:{colour}; font-weight:600'>${pnl:+,.2f}</span>",
            unsafe_allow_html=True)
        if b2.button("Close", key=f"close_{book}_{int(row['Ticket'])}", width="stretch"):
            good, msg = close_ticket(int(row["Ticket"]))
            (st.success if good else st.error)(f"Ticket {int(row['Ticket'])}: {msg}")
            st.rerun()


def render_trade_table(df: pd.DataFrame, key: str = "") -> None:
    """Colour-coded closed-trade table; select a row to chart that trade."""
    t = _normalize_trades(df)
    if t.empty:
        st.caption("No closed trades yet.")
        return
    wins = int((t["P&L"] > 0).sum())
    losses = int((t["P&L"] < 0).sum())
    st.caption(f"{len(t)} closed · {wins} won · {losses} lost · "
               f"net ${t['P&L'].sum():+,.2f} · {t['R'].sum():+.2f}R "
               f"· click a row to chart it")

    # Opened/SL/TP/TF are needed to draw the chart but would clutter the
    # table, so keep them in the frame and hide them from the display.
    hide = [c for c in ("Opened", "SL", "TP", "TF", "HoldH") if c in t.columns]
    view = t.drop(columns=hide)
    styled = (
        view.style
        .map(_result_color, subset=["Result"])
        .map(_color_pnl, subset=["P&L", "R"])
        .map(_color_dir, subset=["Dir"])
        .format({"Entry": "{:,.5f}", "Exit": "{:,.5f}", "R": "{:+.3f}",
                 "P&L": "${:+,.2f}", "Held": "{}",
                 "Closed": lambda v: v.strftime("%Y-%m-%d %H:%M") if pd.notna(v) else ""})
    )
    sel = st.dataframe(styled, width="stretch", hide_index=True,
                       on_select="rerun", selection_mode="single-row",
                       key=f"tt_{key}")
    picked = (sel.selection.rows or []) if hasattr(sel, "selection") else []
    if picked:
        row = t.iloc[picked[0]]
        st.markdown(
            f"#### {row['Result']} &nbsp; {row['Strategy']} — {row['Symbol']} "
            f"{row['Dir']} &nbsp;·&nbsp; {row['R']:+.3f}R &nbsp;·&nbsp; "
            f"${row['P&L']:+,.2f} &nbsp;·&nbsp; held {row.get('Held', '—')}",
            unsafe_allow_html=True)
        render_closed_trade_chart(row, key=f"{key}_{picked[0]}")


def render_trade_charts(df: pd.DataFrame, starting_equity: float, key: str = "") -> None:
    """Equity curve + per-trade R bars + per-strategy P&L, all win/loss coloured."""
    t = _normalize_trades(df)
    if t.empty:
        st.caption("No closed trades yet — charts appear once trades close.")
        return
    asc = t.sort_values("Closed")
    asc = asc[asc["Closed"].notna()]
    if asc.empty:
        st.caption("Closed trades carry no usable timestamps.")
        return

    equity = starting_equity + asc["P&L"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=asc["Closed"], y=equity, mode="lines+markers", name="Equity",
        line=dict(color=_UP, width=2),
        marker=dict(size=6, color=[_UP if v > 0 else _DOWN for v in asc["P&L"]]),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=starting_equity, line_dash="dot", line_color=_AXIS_LABEL, opacity=0.6)
    fig.update_layout(
        title="Equity curve", height=320, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_AXIS_LABEL), showlegend=False,
        xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID, tickprefix="$"),
    )
    st.plotly_chart(fig, width="stretch", key=f"eq_{key}")

    c1, c2 = st.columns(2)
    with c1:
        bar = go.Figure(go.Bar(
            x=list(range(1, len(asc) + 1)), y=asc["R"],
            marker_color=[_UP if v > 0 else _DOWN for v in asc["R"]],
            hovertext=[f"{s} {sym}" for s, sym in zip(asc["Strategy"], asc["Symbol"])],
            hovertemplate="#%{x} %{hovertext}<br>%{y:+.3f}R<extra></extra>",
        ))
        bar.update_layout(
            title="R-multiple per trade", height=290,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_AXIS_LABEL), showlegend=False,
            xaxis=dict(title="trade #", gridcolor=_GRID),
            yaxis=dict(gridcolor=_GRID, zerolinecolor=_AXIS_LABEL),
        )
        st.plotly_chart(bar, width="stretch", key=f"r_{key}")
    with c2:
        per = (asc.groupby("Strategy")["P&L"].sum().sort_values())
        hbar = go.Figure(go.Bar(
            x=per.values, y=per.index, orientation="h",
            marker_color=[_UP if v > 0 else _DOWN for v in per.values],
            hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>",
        ))
        hbar.update_layout(
            title="P&L by strategy", height=290,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_AXIS_LABEL), showlegend=False,
            xaxis=dict(gridcolor=_GRID, tickprefix="$"), yaxis=dict(gridcolor=_GRID),
        )
        st.plotly_chart(hbar, width="stretch", key=f"leg_{key}")


def render_weekly_reconstruction(bot_key: str) -> None:
    path = ROOT / "scratch" / "weekly_reconstruction_result.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    trades = data.get(bot_key, [])

    window_start = str(data.get("window_start", ""))[:10]
    generated = str(data.get("generated_at", ""))[:16]
    st.markdown(f"### Last 3 weeks (from {window_start}) — RECONSTRUCTED, not real fills")
    st.warning(
        "**Nothing in this table executed.** Order placement was broken on both bots "
        "from go-live until 2026-08-20 15:33 UTC (two separate bugs in `mt5_engine.py`: "
        "positional `order_send(request)` + a 31-char comment truncation against this "
        "broker's real 29-char limit). Exactly **one** real fill exists — XAUUSD long, "
        "ticket 10124062219, opened 15:33 today, still open — and it is *not* in this "
        "table; it appears in the **Live open positions** section. Everything below is "
        "the live strategy code replayed against real MT5 price bars. "
        f"Generated {generated}."
    )
    if not trades:
        st.caption("No qualifying signals in this window.")
        return

    df = pd.DataFrame(trades)
    mismapped = df[df.get("symbol_mismapped", False) == True]  # noqa: E712
    if not mismapped.empty:
        bad_symbols = sorted(mismapped["symbol"].unique())
        st.error(
            f"⚠️ **{', '.join(bad_symbols)} are mismapped on this broker** — they resolve to "
            "Nasdaq-listed Bitcoin/Ethereum trust ETFs (\"BTC\" = Grayscale Bitcoin Mini "
            "Trust, ~$30/share, NYSE hours only). This account carries no crypto CFDs at "
            "all (only Forex/Indexes/Metals/Nasdaq), so 3 of the 8 funded legs are trading "
            "the wrong instrument entirely. These rows are excluded from every total below."
        )
    valid = df[df.get("symbol_mismapped", False) != True]  # noqa: E712

    # The trial book risks 90% of a $50k anchor per trade ($45,000). Its dollar
    # column is arithmetic, not a survivable equity path -- one -1.5R loss is
    # -$67,500 on a $50,000 account. Say so before showing a single figure.
    death = data.get("trial_account_death")
    if bot_key == "trial":
        risk = data.get("trial_risk_usd_per_trade", 0.0)
        msg = (f"🚨 **Trial dollar figures are notional — read R-multiples instead.** "
               f"This bot risks ${risk:,.0f}/trade (90% of a $50,000 anchor) by design, to "
               "gamble through free-retry evaluations. A single −1.5R loss is −$67,500 on a "
               "$50,000 account.")
        if death:
            msg += (f" On this replay the account would have **blown on trade #"
                    f"{death['trades_survived'] + 1}** ({str(death['at'])[:16]}, "
                    f"{death['leg']}) — equity ${death['equity_after']:,.2f}. "
                    "Every trial number after that point is arithmetic on a dead account: "
                    "the real bot would have hit its kill switch and auto-reset a fresh "
                    "attempt, which this replay does not model.")
        st.error(msg)

    closed = valid[valid["status"] == "CLOSED"]
    open_now = valid[valid["status"] != "CLOSED"]
    total_r = float(valid["r_multiple"].sum()) if not valid.empty else 0.0
    total_pnl = float(valid["est_dollar_pnl"].sum()) if not valid.empty else 0.0
    wins = int((closed["r_multiple"] > 0).sum()) if not closed.empty else 0
    n_live = int(valid.get("bot_was_live", pd.Series(dtype=bool)).sum()) if not valid.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades (valid symbols)", len(valid),
              help=f"{n_live} occurred while the bot was actually running; the rest "
                   "predate go-live entirely and are pure hypotheticals.")
    c2.metric("Closed / still open", f"{len(closed)} / {len(open_now)}")
    c3.metric("Win rate (closed)", f"{wins}/{len(closed)}"
              + (f" ({wins / len(closed):.0%})" if len(closed) else ""))
    c4.metric("Total R (3wk)", f"{total_r:+.2f}R",
              help="R-multiple is the trustworthy metric here — it is risk-normalized "
                   "and independent of the position sizing this replay cannot model.")
    if bot_key != "trial":
        st.caption(f"Estimated P&L over the window: **${total_pnl:,.2f}** "
                   f"(risk ${data.get('funded_risk_usd_per_trade', 0):,.0f}/trade, 0.7% — "
                   "survivable sizing, so this dollar figure is meaningful).")

    # Per-week breakdown -- the actual "3 weeks of data" ask.
    if "week" in valid.columns and not valid.empty:
        wk = valid.groupby("week").agg(
            trades=("r_multiple", "size"),
            wins=("r_multiple", lambda s: int((s > 0).sum())),
            total_R=("r_multiple", "sum"),
            est_usd=("est_dollar_pnl", "sum"),
            while_bot_live=("bot_was_live", "sum"),
        ).reset_index().sort_values("week")
        wk["total_R"] = wk["total_R"].round(2)
        wk["est_usd"] = wk["est_usd"].round(2)
        if bot_key == "trial":
            wk = wk.drop(columns=["est_usd"])  # notional; do not display
        st.markdown("**Week by week**")
        st.dataframe(wk, width="stretch", hide_index=True)

    show_cols = [c for c in (
        "week", "leg", "symbol", "timeframe", "direction", "entry_price", "entered_at",
        "status", "exit_price", "exited_at", "exit_reason", "r_multiple",
        "est_dollar_pnl", "bot_was_live",
    ) if c in df.columns]
    if bot_key == "trial" and "est_dollar_pnl" in show_cols:
        show_cols.remove("est_dollar_pnl")
    st.markdown("**All reconstructed trades**")
    st.dataframe(df[show_cols].sort_values("entered_at", ascending=False),
                 width="stretch", hide_index=True)


# Blue(up)/red-pink(down) palette — used consistently everywhere (charts,
# P&L text, buttons) instead of the traditional green/red.
_BG = "#06080f"
_GRID = "#141a2a"
_AXIS_LABEL = "#93a1b3"
_UP, _DOWN = "#4c6ef5", "#f5426c"          # candle body/wick colors
_TP_COLOR, _SL_COLOR = "#4c6ef5", "#f5426c"  # TP=blue (target), SL=red-pink (stop)
_ENTRY_COLOR = "#ffffff"  # neutral white so the entry marker doesn't blend into either candle color

CHART_API_URL = "http://127.0.0.1:8765"


def _price_precision(symbol: str) -> tuple[int, float]:
    """Decimal places for the price axis/candles, per symbol convention.
    lightweight-charts defaults to precision=2 — fine for indices, but for
    FX pairs whose whole visible range can round to a single value at that
    precision, it can't generate distinct tick labels and draws none at all.
    """
    s = symbol.upper()
    if s in ("US30", "DE40", "US500", "USTEC", "UK100"):
        return 1, 0.1
    if s == "XAUUSD":
        return 2, 0.01
    if s.endswith("JPY"):
        return 3, 0.001
    return 5, 0.00001


def build_live_chart_html(symbol: str, ticket: int, is_long: bool, default_timeframe: str = "15m", height: int = 420) -> str:
    """Self-contained HTML/JS chart using TradingView's own open-source
    lightweight-charts library (same rendering engine TradingView itself
    uses — just not their hosted market-data widget, which can't draw our
    own SL/TP/entry levels). It updates itself via a JS setInterval hitting
    the local chart_api.py server, completely independent of Streamlit's
    rerun cycle — that's what actually removes the flicker, since Streamlit's
    own chart components visibly redraw on every resend no matter what key
    or chart count is used. Native mouse-wheel zoom and drag-to-pan are
    built into the library for free, and a small toolbar switches between
    15m and 1m without reloading the whole component.
    """
    marker_shape = "arrowUp" if is_long else "arrowDown"
    div_id = f"lw_{ticket}"
    precision, min_move = _price_precision(symbol)

    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  #chart_wrap_{ticket} * {{ font-family: 'Inter', sans-serif; }}
</style>
<div id="chart_wrap_{ticket}" style="background:{_BG}; padding:8px; border-radius:10px; border:1px solid rgba(34,211,238,0.22); box-shadow:0 0 26px rgba(34,211,238,0.08), inset 0 0 40px rgba(34,211,238,0.02);">
  <div style="margin-bottom:8px; display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <div id="pnl_{ticket}" style="font-size:1.7rem; font-weight:700; line-height:1;"></div>
    <div id="stats_{ticket}" style="color:{_AXIS_LABEL}; font-size:0.85rem; opacity:0.85;"></div>
  </div>
  <div style="margin-bottom:6px;">
    <button onclick="window.__switchTf_{ticket}('15m')" style="background:rgba(255,255,255,0.04);color:{_AXIS_LABEL};border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;cursor:pointer;margin-right:6px;">15m</button>
    <button onclick="window.__switchTf_{ticket}('1m')" style="background:rgba(255,255,255,0.04);color:{_AXIS_LABEL};border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;cursor:pointer;">1m</button>
    <span id="tf_label_{ticket}" style="color:{_AXIS_LABEL};margin-left:10px;">{default_timeframe}</span>
  </div>
  <div id="{div_id}" style="width:100%; height:{height}px;"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  const container = document.getElementById("{div_id}");
  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth, height: {height},
    layout: {{ background: {{ color: "{_BG}" }}, textColor: "{_AXIS_LABEL}", fontFamily: "Inter, sans-serif" }},
    grid: {{ vertLines: {{ color: "{_GRID}" }}, horzLines: {{ color: "{_GRID}" }} }},
    rightPriceScale: {{ borderColor: "{_GRID}", visible: true, scaleMargins: {{ top: 0.1, bottom: 0.1 }} }},
    timeScale: {{
      borderColor: "{_GRID}", timeVisible: true, secondsVisible: false,
      tickMarkFormatter: (time) => {{
        const d = new Date(time * 1000);
        const hh = String(d.getUTCHours()).padStart(2, '0');
        const mm = String(d.getUTCMinutes()).padStart(2, '0');
        const dd = String(d.getUTCDate()).padStart(2, '0');
        const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
        return mo + "-" + dd + " " + hh + ":" + mm;
      }},
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const priceFormat = {{ type: "price", precision: {precision}, minMove: {min_move} }};
  const series = chart.addCandlestickSeries({{
    upColor: "{_UP}", downColor: "{_DOWN}", borderVisible: false,
    wickUpColor: "{_UP}", wickDownColor: "{_DOWN}",
    priceFormat: priceFormat,
  }});

  // Invisible series purely to force the price scale to include SL/TP.
  // lightweight-charts autoscales the right axis to the *candle* data only —
  // createPriceLine() draws a line but does NOT expand that range, so a TP
  // sitting outside the visible candles (very common right after entry,
  // since it's a 3R target) gets drawn off-screen with nothing visibly
  // wrong to indicate why. This dummy series has real data points at the
  // SL/TP/entry levels (rendered fully transparent) so autoscale sees them.
  const rangeSeries = chart.addLineSeries({{
    color: "rgba(0,0,0,0)", lineWidth: 1,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat,
  }});

  window.addEventListener("resize", () => chart.applyOptions({{ width: container.clientWidth }}));

  let tf = "{default_timeframe}";
  let slLine = null, tpLine = null, entryLine = null, markersSet = false, hadPosition = false;
  const statsEl = document.getElementById("stats_{ticket}");
  const pnlEl = document.getElementById("pnl_{ticket}");

  function fmtDuration(totalSec) {{
    const m = Math.max(0, Math.floor(totalSec / 60));
    if (m < 60) return m + "m";
    const h = Math.floor(m / 60), rm = m % 60;
    if (h < 24) return h + "h " + rm + "m";
    const d = Math.floor(h / 24), rh = h % 24;
    return d + "d " + rh + "h";
  }}

  async function refresh(fullReload) {{
    try {{
      const res = await fetch(`{CHART_API_URL}/chart?symbol={symbol}&timeframe=${{tf}}&ticket={ticket}&count=500`);
      const data = await res.json();
      if (!data.bars || !data.bars.length) return;
      if (fullReload) {{
        series.setData(data.bars);
        chart.timeScale().fitContent();
        markersSet = false;
      }} else {{
        series.update(data.bars[data.bars.length - 1]);
      }}
      if (data.position) {{
        hadPosition = true;
        const p = data.position;
        if (!slLine) {{
          slLine = series.createPriceLine({{price: p.sl, color: "{_SL_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: "SL"}});
          tpLine = series.createPriceLine({{price: p.tp, color: "{_TP_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: "TP"}});
          entryLine = series.createPriceLine({{price: p.entry, color: "{_ENTRY_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, title: "Entry"}});
        }} else {{
          slLine.applyOptions({{price: p.sl}});
          tpLine.applyOptions({{price: p.tp}});
          entryLine.applyOptions({{price: p.entry}});
        }}
        const firstT = data.bars[0].time;
        const lastT = data.bars[data.bars.length - 1].time;
        const lo = Math.min(p.sl, p.tp, p.entry);
        const hi = Math.max(p.sl, p.tp, p.entry);
        if (firstT !== lastT) {{
          rangeSeries.setData([{{time: firstT, value: lo}}, {{time: lastT, value: hi}}]);
        }}
        if (!markersSet) {{
          // Marker time must match an actual bar's time exactly or
          // lightweight-charts silently drops it — a position almost never
          // opens exactly on a candle boundary, so use the snapped bar time
          // the API computes (falls back to the raw open time if missing).
          series.setMarkers([{{
            time: p.entry_bar_time || p.opened_time, position: p.is_long ? "belowBar" : "aboveBar",
            color: "{_ENTRY_COLOR}", shape: "{marker_shape}", text: "Entry " + p.entry.toFixed({precision}),
          }}]);
          markersSet = true;
        }}
        const pnlColor = p.profit >= 0 ? "{_UP}" : "{_DOWN}";
        if (pnlEl) {{
          pnlEl.style.color = pnlColor;
          pnlEl.innerText = (p.profit >= 0 ? "+$" : "-$") + Math.abs(p.profit).toFixed(2);
        }}
        if (statsEl) {{
          // Real elapsed time (browser clock, true UTC) against the
          // API-corrected true-UTC open time — NOT lastT (the last fetched
          // bar's timestamp), which is up to one whole bar-interval stale
          // and used to show "0m" for any position younger than that,
          // since a fresh position's open time is almost always at or
          // after the most recently closed bar.
          const nowUtc = Date.now() / 1000;
          const openedFor = fmtDuration(nowUtc - (p.opened_time_utc ?? p.opened_time));
          statsEl.innerHTML = "Current <b>" + p.current.toFixed({precision}) + "</b> &nbsp; open " + openedFor;
        }}
      }} else if (hadPosition) {{
        // Position just disappeared from MT5 (hit TP/SL/trailing stop, or
        // was manually closed) — without this, the card just sits there
        // showing the last-known numbers forever until someone clicks
        // "Check for new/closed positions", looking like a stuck/broken
        // chart rather than a closed trade.
        hadPosition = false;
        if (pnlEl) {{
          pnlEl.style.color = "{_AXIS_LABEL}";
          pnlEl.innerText = "Trade closed";
        }}
        if (statsEl) {{
          statsEl.innerHTML = 'This position closed — click "Check for new/closed positions" above to remove this card.';
        }}
      }}
    }} catch (e) {{ /* transient fetch error (e.g. chart_api.py not running yet) — next tick retries */ }}
  }}

  window.__switchTf_{ticket} = function(newTf) {{
    tf = newTf;
    document.getElementById("tf_label_{ticket}").innerText = newTf;
    refresh(true);
  }};

  refresh(true);
  setInterval(() => refresh(false), 2000);
}})();
</script>
"""


def _broker_offset_seconds(symbol: str) -> int:
    """Broker-server clock minus true UTC, measured via a live tick (see the
    'Opened' timestamp fix earlier — MT5's own timestamps are in the broker's
    clock, not true UTC, commonly hours off). Needed to convert the ledger's
    true-UTC opened/closed timestamps into the range copy_rates_range expects.

    Called exactly once per review (when the button is clicked, see
    render_closed_positions), NOT on every render — baking a freshly-measured value
    into the chart HTML on every fragment rerun would make the HTML differ
    by a second or two each cycle (tick.time jitter), which defeats Streamlit's
    identical-content check and remounts the iframe — restarting the
    animation and flashing every few seconds."""
    if not _MT5_AVAILABLE or not mt5.initialize():
        return 0
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not tick.time:
        return 0
    broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    return int((broker_now - datetime.now(timezone.utc)).total_seconds())


def build_closed_trade_chart_html(
    symbol: str, is_long: bool, entry: float, sl: float, tp: float, exit_price: float,
    pnl: float, exit_reason: str, opened_broker_ts: int, closed_broker_ts: int,
    ts_from: int, ts_to: int, height: int = 460,
) -> str:
    """Animated replay of a CLOSED trade: real 1-minute candles reveal one
    (or a small batch) at a time from before entry to after exit, while SL/TP/
    entry/exit are drawn as persistent horizontal lines for the whole
    animation — the entry/exit arrow markers additionally pop in the instant
    the reveal reaches that bar. All timestamps are precomputed once by the
    caller (see render_closed_positions) so this function is a pure formatter with no
    MT5 calls and byte-identical output across Streamlit fragment reruns —
    required so the auto-refresh timer doesn't remount the iframe mid-animation."""
    marker_shape = "arrowUp" if is_long else "arrowDown"
    win = pnl > 0
    exit_color = _UP if win else _DOWN
    div_id = f"lwc_{opened_broker_ts}_{symbol}"
    precision, min_move = _price_precision(symbol)
    price_delta_js = f"(current - {entry})" if is_long else f"({entry} - current)"
    # Deliberately NOT reusing _SL_COLOR/_TP_COLOR (which alias the same
    # blue/pink used for up/down candles and profit/loss everywhere else) —
    # in this chart specifically, SL/TP need to read as "risk levels", not
    # blend in with the candle-color language, so they get their own
    # unambiguous red/gold here.
    replay_sl_color, replay_tp_color, replay_entry_color = "#ff5252", "#ffca28", "#ffffff"

    reason_label = {"tp": "hit take-profit", "sl": "hit stop-loss (or trailing stop)", "manual": "closed manually"}.get(exit_reason, exit_reason)

    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  #chart_wrap_{div_id} * {{ font-family: 'Inter', sans-serif; }}
  #chart_wrap_{div_id} button:hover {{ background:rgba(255,255,255,0.1) !important; }}
  #chart_wrap_{div_id} button:disabled {{ opacity:0.35; cursor:default; }}
</style>
<div id="chart_wrap_{div_id}" style="background:{_BG}; padding:8px; border-radius:10px; border:1px solid rgba(34,211,238,0.22); box-shadow:0 0 26px rgba(34,211,238,0.08), inset 0 0 40px rgba(34,211,238,0.02);">
  <div style="margin-bottom:8px; display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <div id="pnl_{div_id}" style="font-size:1.4rem; font-weight:700; color:{exit_color};">
      {'+' if win else ''}{pnl:.2f} USD
    </div>
    <div style="color:{_AXIS_LABEL}; font-size:0.85rem; opacity:0.85;">
      {symbol} {'long' if is_long else 'short'} &nbsp; Entry {entry:.{precision}f} → Exit {exit_price:.{precision}f} &nbsp; ({reason_label})
    </div>
  </div>
  <div style="margin-bottom:6px; display:flex; align-items:center; gap:6px;">
    <button id="play_{div_id}" style="background:rgba(255,255,255,0.04);color:{_AXIS_LABEL};border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;cursor:pointer;">▶ Play</button>
    <button id="pause_{div_id}" style="background:rgba(255,255,255,0.04);color:{_AXIS_LABEL};border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;cursor:pointer;">⏸ Pause</button>
    <button id="restart_{div_id}" style="background:rgba(255,255,255,0.04);color:{_AXIS_LABEL};border:1px solid rgba(255,255,255,0.1);padding:4px 10px;border-radius:6px;cursor:pointer;">⟲ Restart</button>
    <span style="color:{_AXIS_LABEL};opacity:0.7;margin-left:8px;font-size:0.8rem;">1m candles</span>
    <span id="progress_{div_id}" style="color:{_AXIS_LABEL};opacity:0.85;margin-left:auto;font-size:0.8rem;"></span>
  </div>
  <div id="{div_id}" style="width:100%; height:{height}px;"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  const container = document.getElementById("{div_id}");
  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth, height: {height},
    layout: {{ background: {{ color: "{_BG}" }}, textColor: "{_AXIS_LABEL}", fontFamily: "Inter, sans-serif" }},
    grid: {{ vertLines: {{ color: "{_GRID}" }}, horzLines: {{ color: "{_GRID}" }} }},
    rightPriceScale: {{ borderColor: "{_GRID}", visible: true, scaleMargins: {{ top: 0.1, bottom: 0.1 }} }},
    timeScale: {{
      borderColor: "{_GRID}", timeVisible: true, secondsVisible: false,
      tickMarkFormatter: (time) => {{
        const d = new Date(time * 1000);
        const hh = String(d.getUTCHours()).padStart(2, '0');
        const mm = String(d.getUTCMinutes()).padStart(2, '0');
        const dd = String(d.getUTCDate()).padStart(2, '0');
        const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
        return mo + "-" + dd + " " + hh + ":" + mm;
      }},
    }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const priceFormat = {{ type: "price", precision: {precision}, minMove: {min_move} }};
  const series = chart.addCandlestickSeries({{
    upColor: "{_UP}", downColor: "{_DOWN}", borderVisible: false,
    wickUpColor: "{_UP}", wickDownColor: "{_DOWN}",
    priceFormat: priceFormat,
  }});
  const rangeSeries = chart.addLineSeries({{
    color: "rgba(0,0,0,0)", lineWidth: 1,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat,
  }});
  window.addEventListener("resize", () => chart.applyOptions({{ width: container.clientWidth }}));
  // A ResizeObserver too, not just the window "resize" event: this chart
  // renders inside an st.dialog modal, whose final on-screen size can settle
  // a moment after this script runs (CSS/layout catching up to the dialog
  // opening) — without this, the chart can get stuck at whatever narrow
  // width it happened to measure first, which looks "zoomed in" no matter
  // how much time-range padding is added.
  new ResizeObserver(() => chart.applyOptions({{ width: container.clientWidth }})).observe(container);

  function snapToBar(bars, targetTime) {{
    const onOrBefore = bars.filter(b => b.time <= targetTime);
    return onOrBefore.length ? onOrBefore[onOrBefore.length - 1].time : bars[0].time;
  }}
  function fmtTime(t) {{
    const d = new Date(t * 1000);
    return String(d.getUTCMonth()+1).padStart(2,'0') + "-" + String(d.getUTCDate()).padStart(2,'0') + " " +
           String(d.getUTCHours()).padStart(2,'0') + ":" + String(d.getUTCMinutes()).padStart(2,'0');
  }}

  let allBars = [];
  let idx = 0, barsPerTick = 1, frameMs = 60, timer = null;
  let entryBarTime = null, exitBarTime = null;
  let dollarsPerUnit = 0;

  function setPnlText(el, value) {{
    el.innerText = (value >= 0 ? "+" : "") + value.toFixed(2) + " USD";
    el.style.color = value >= 0 ? "{_UP}" : "{_DOWN}";
  }}

  function updatePnl(visibleBars) {{
    const pnlEl = document.getElementById("pnl_{div_id}");
    if (!pnlEl || !visibleBars.length) return;
    const lastTime = visibleBars[visibleBars.length - 1].time;
    // Only "live" while the position is actually open: 0 before entry,
    // frozen at the real final P&L once exit is reached — not still
    // wobbling with price action that happened after the trade closed.
    if (entryBarTime === null || lastTime < entryBarTime) {{
      setPnlText(pnlEl, 0);
      return;
    }}
    if (exitBarTime !== null && lastTime >= exitBarTime) {{
      setPnlText(pnlEl, {pnl});
      return;
    }}
    if (!dollarsPerUnit) return;
    const current = visibleBars[visibleBars.length - 1].close;
    const priceDelta = {price_delta_js};
    setPnlText(pnlEl, priceDelta * dollarsPerUnit);
  }}

  function updateMarkers(visibleBars) {{
    const markers = [];
    if (entryBarTime !== null && visibleBars.some(b => b.time === entryBarTime)) {{
      markers.push({{time: entryBarTime, position: {'"belowBar"' if is_long else '"aboveBar"'}, color: "{_ENTRY_COLOR}", shape: "{marker_shape}", text: "Entry " + ({entry}).toFixed({precision})}});
    }}
    if (exitBarTime !== null && visibleBars.some(b => b.time === exitBarTime)) {{
      markers.push({{time: exitBarTime, position: {'"aboveBar"' if is_long else '"belowBar"'}, color: "{exit_color}", shape: {'"arrowDown"' if is_long else '"arrowUp"'}, text: "Exit " + ({exit_price}).toFixed({precision})}});
    }}
    series.setMarkers(markers);
  }}

  function render() {{
    const visible = allBars.slice(0, idx);
    series.setData(visible);
    updateMarkers(visible);
    updatePnl(visible);
    const label = document.getElementById("progress_{div_id}");
    if (label) {{
      label.innerText = visible.length
        ? `${{fmtTime(visible[visible.length - 1].time)}}  ·  bar ${{visible.length}}/${{allBars.length}}`
        : "";
    }}
  }}

  function pause() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
  }}
  function play() {{
    if (timer || idx >= allBars.length) return;
    timer = setInterval(() => {{
      idx = Math.min(idx + barsPerTick, allBars.length);
      render();
      if (idx >= allBars.length) pause();
    }}, frameMs);
  }}
  function restart() {{
    pause();
    idx = Math.min(1, allBars.length);
    render();
    play();
  }}

  document.getElementById("play_{div_id}").onclick = play;
  document.getElementById("pause_{div_id}").onclick = pause;
  document.getElementById("restart_{div_id}").onclick = restart;

  async function load() {{
    try {{
      const res = await fetch(`{CHART_API_URL}/history?symbol={symbol}&timeframe=1m&from={ts_from}&to={ts_to}`);
      const data = await res.json();
      if (!data.bars || !data.bars.length) return;
      allBars = data.bars;

      // Fixed viewport for the whole animation — computed from the full
      // fetched range up front so the chart doesn't rescale/jump as more
      // candles get revealed each tick. Includes sl/tp directly: the window
      // itself is now just the trade (a couple minutes either side, not
      // hours of context), so even a wide Donchian-style stop is a modest
      // multiple of that tight range, not the 10-70x blowout it would've
      // been back when this was combined with a large context window too.
      let lo = Math.min({entry}, {exit_price}, {sl}, {tp});
      let hi = Math.max({entry}, {exit_price}, {sl}, {tp});
      for (const b of allBars) {{ lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); }}
      const span = hi - lo;
      lo -= span * 0.08;
      hi += span * 0.08;
      const firstT = allBars[0].time, lastT = allBars[allBars.length - 1].time;
      rangeSeries.setData([{{time: firstT, value: lo}}, {{time: lastT, value: hi}}]);
      chart.timeScale().setVisibleRange({{from: firstT, to: lastT}});

      // Exact conversion factor from price movement to dollars, derived from
      // the one real data point we have (the trade's actual final pnl at
      // exit_price) — profit is linear in price for a fixed lot size, so
      // this is precise, not an approximation.
      const exitDelta = {price_delta_js.replace("current", str(exit_price))};
      dollarsPerUnit = exitDelta !== 0 ? {pnl} / exitDelta : 0;

      // SL/TP as bold, unambiguous reference lines — thicker and in their
      // own red/gold (not the candle-color blue/pink) so they read as risk
      // levels at a glance. Entry gets its own dotted white line too (thinner
      // + a different dash style than SL/TP so the three don't compete when
      // they sit close together) — the arrow marker alone only labels the
      // entry bar itself, not the level across the whole replay window.
      series.createPriceLine({{price: {sl}, color: "{replay_sl_color}", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, title: "SL"}});
      series.createPriceLine({{price: {tp}, color: "{replay_tp_color}", lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed, title: "TP"}});
      series.createPriceLine({{price: {entry}, color: "{replay_entry_color}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, title: "Entry"}});

      entryBarTime = snapToBar(allBars, {opened_broker_ts});
      exitBarTime = snapToBar(allBars, {closed_broker_ts});

      // Fixed reveal speed (ms per bar) so a longer trade plays a longer
      // animation and a short one plays a short one, instead of always
      // squeezing every trade into the same ~9s regardless of how long it
      // actually took. barsPerTick only kicks in past maxTotalMs, as a
      // safety valve for the rare very-long trade rather than the norm.
      const perBarMs = 45, maxTotalMs = 45000;
      frameMs = perBarMs;
      barsPerTick = Math.max(1, Math.ceil((allBars.length * perBarMs) / maxTotalMs));

      idx = Math.min(1, allBars.length);
      render();
      play();
    }} catch (e) {{ /* transient fetch error — chart_api.py may not be running */ }}
  }}

  load();
}})();
</script>
"""


def close_ticket(ticket: int) -> tuple[bool, str]:
    """Manually close an open position from the dashboard. Reuses MT5Engine's
    filling-mode-aware close logic — see mt5_engine.py's _filling_candidates
    for why this can't just hardcode ORDER_FILLING_IOC."""
    if not _MT5_AVAILABLE:
        return False, "MetaTrader5 package not available."
    engine = MT5Engine()
    ok = engine.close_position(ticket)
    return ok, ("Closed" if ok else f"Close failed: {mt5.last_error()}")


def _fmt_duration(delta: pd.Timedelta) -> str:
    total_min = max(0, int(delta.total_seconds() // 60))
    if total_min < 60:
        return f"{total_min}m"
    h, m = divmod(total_min, 60)
    if h < 24:
        return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def _color_pnl(val) -> str:
    if pd.isna(val):
        return ""
    return "color: #4c6ef5; font-weight: 600" if val > 0 else (
        "color: #f5426c; font-weight: 600" if val < 0 else ""
    )


def _dir_icon(d: str) -> str:
    return UP if str(d).lower() in ("long", "buy") else DOWN


def _color_dir(val) -> str:
    return "color: #4c6ef5; font-weight: 600" if val == UP else "color: #f5426c; font-weight: 600"


def load_ledger(path: Path):
    with sqlite3.connect(str(path)) as c:
        signals = pd.read_sql("SELECT * FROM signals ORDER BY id DESC", c)
        orders = pd.read_sql("SELECT * FROM orders ORDER BY id DESC", c)
        closed = pd.read_sql("SELECT * FROM closed_positions ORDER BY id DESC", c)
        heartbeats = pd.read_sql(
            "SELECT * FROM heartbeats ORDER BY rowid DESC LIMIT 5000", c
        )
    return signals, orders, closed, heartbeats


_LOGO_PATH = ROOT / "scripts" / "assets" / "vertex_v_icon.png"
_LOGO_SMALL_PATH = ROOT / "scripts" / "assets" / "vertex_v_icon_small.png"


@st.cache_data
def _logo_data_uri(path: Path) -> str:
    """Base64 data URI so the logo can be embedded directly in the header's
    HTML markdown — Streamlit doesn't serve arbitrary local files via a URL,
    so a plain <img src="..."> pointing at the filesystem path won't load."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


st.set_page_config(page_title="Vertex Algo Bot", layout="wide", page_icon=str(_LOGO_PATH),
                   initial_sidebar_state="expanded")

# --- Clean dark/cyan trading-platform theme injected as raw CSS. Purely
# cosmetic — targets Streamlit's data-testid hooks so it survives reruns
# without touching any widget logic below. The stMetricValue overflow/
# text-overflow overrides below are load-bearing: Streamlit's own default
# metric CSS clips long numbers with an ellipsis once a column gets narrow
# (7-wide KPI rows do this even with no custom styling at all) — without
# this override, numbers get cut off regardless of font choice. ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --up: #4c6ef5;
    --up-soft: rgba(76,110,245,0.12);
    --down: #f5426c;
    --down-soft: rgba(245,66,108,0.12);
    --bg: #06080f;
    --panel: #0d1220;
    --panel-border: rgba(255,255,255,0.06);
    --soft-shadow: 0 4px 16px rgba(0,0,0,0.25);
}

.stApp { background: var(--bg); }

/* Trade-replay dialog — stretched to cover almost the whole viewport (the
   default modal width is only ~800px, way too cramped for a candlestick
   chart with room to read the price action). Targets both the dialog
   element itself and its immediate content wrapper since Streamlit's
   internal DOM nesting isn't a stable thing to depend on precisely. */
div[data-testid="stDialog"] {
    width: 95vw !important;
    max-width: 1600px !important;
}
div[data-testid="stDialog"] > div {
    width: 100% !important;
    max-width: 100% !important;
    height: 92vh !important;
}

* { font-family: 'Inter', sans-serif; }

h1, h2, h3 {
    font-weight: 600 !important;
    color: #eef2f6 !important;
}

/* Metric cards — fixes the number-clipping bug: Streamlit's default metric
   value has overflow:hidden + text-overflow:ellipsis baked in, which cuts
   off anything wider than the column at full font size. */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: var(--soft-shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

div[data-testid="stMetricValue"] {
    overflow: visible !important;
    text-overflow: unset !important;
    white-space: nowrap !important;
    font-size: clamp(1.1rem, 2.3vw, 1.8rem) !important;
    line-height: 1.25 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricLabel"] { overflow: visible !important; white-space: nowrap !important; color: #93a1b3 !important; }
div[data-testid="stMetricDelta"] { overflow: visible !important; white-space: nowrap !important; }

/* Delta arrows/text in the cyan(up)/magenta(down) scheme instead of
   Streamlit's default green/red, to match the reference palette. */
div[data-testid="stMetricDelta"] svg[fill="rgb(9, 171, 59)"],
div[data-testid="stMetricDelta"] svg[fill="#09ab3b"] { fill: var(--up) !important; }
div[data-testid="stMetricDelta"] svg[fill="rgb(255, 43, 43)"],
div[data-testid="stMetricDelta"] svg[fill="#ff2b2b"] { fill: var(--down) !important; }

div[data-testid="stAlert"] { border-radius: 10px; }

section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--panel-border);
}

.stButton > button {
    background: var(--panel);
    border: 1px solid var(--down);
    border-radius: 8px;
    color: #ffd7ea;
    transition: background 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    background: var(--down);
    color: white;
}

div[data-testid="stDataFrame"] { border: 1px solid var(--panel-border); border-radius: 8px; overflow: hidden; }

/* Bordered st.container(border=True) panels — used for the major sections
   below so the whole page reads as a clean grid of cards. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--panel-border) !important;
    border-radius: 12px;
    box-shadow: var(--soft-shadow);
}

hr { border-color: var(--panel-border) !important; }

.app-logo {
    display: inline-block; width: 58px; height: 58px; vertical-align: middle;
    margin-right: 10px;
}

.ticker-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--panel); border: 1px solid var(--panel-border);
    border-radius: 20px; padding: 6px 12px; margin: 0 8px 8px 0;
    font-size: 0.85rem; box-shadow: var(--soft-shadow);
}
.ticker-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# --- Neon "self-healing terminal" skin (Phase 6) — layered ON TOP of the
# base theme above so the LAYOUT is unchanged; only colours, glow and the
# accent treatment change to match the reference look. ---
st.markdown("""
<style>
:root {
    --neon-cyan: #22d3ee; --neon-magenta: #ff4fd8; --neon-lime: #a3e635;
    --neon-blue: #4c6ef5; --neon-purple: #a78bfa;
}
.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(76,110,245,0.10), transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, rgba(255,79,216,0.07), transparent 55%),
        var(--bg);
}
h1, h2, h3 { letter-spacing: 0.2px; text-shadow: 0 0 18px rgba(34,211,238,0.25); }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(34,211,238,0.16) !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.05), 0 8px 28px rgba(0,0,0,0.45),
                inset 0 1px 0 rgba(255,255,255,0.03);
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(34,211,238,0.30) !important;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(76,110,245,0.20);
    box-shadow: 0 0 22px rgba(76,110,245,0.10), var(--soft-shadow);
}
div[data-testid="stMetricValue"] { text-shadow: 0 0 16px rgba(34,211,238,0.30); }
/* subheaders get a neon left-rule */
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
    border-left: 3px solid var(--neon-cyan);
    padding-left: 10px;
    background: linear-gradient(90deg, rgba(34,211,238,0.08), transparent 40%);
}
.neon-panel { background: rgba(10,13,22,0.6); border: 1px solid rgba(34,211,238,0.16);
    border-radius: 12px; padding: 12px 14px; }
.why-row { border-left: 3px solid var(--neon-cyan); padding: 6px 10px; margin: 6px 0;
    background: linear-gradient(90deg, rgba(34,211,238,0.06), transparent 60%);
    border-radius: 0 8px 8px 0; font-size: 0.86rem; line-height: 1.45; }
.why-row.loss { border-left-color: var(--neon-magenta);
    background: linear-gradient(90deg, rgba(255,79,216,0.07), transparent 60%); }
.why-tag { display:inline-block; font-size:0.72rem; font-weight:600; padding:1px 7px;
    border-radius:10px; margin-right:6px; }
.why-tag.win { background: rgba(76,110,245,0.18); color:#bcd0ff; }
.why-tag.loss { background: rgba(255,79,216,0.16); color:#ffc7f0; }
.verdict-pill { display:inline-block; font-size:0.74rem; padding:3px 9px; border-radius:12px;
    margin:3px 6px 3px 0; border:1px solid rgba(255,255,255,0.08); }
.verdict-pill.ok { color:#bcd0ff; background:rgba(76,110,245,0.12); }
.verdict-pill.avoid { color:#ffc7f0; background:rgba(255,79,216,0.14); border-color:rgba(255,79,216,0.3); }

/* ---- full-interface neon pass (whole app, layout unchanged) ---- */
/* Data tables: neon header row + faint grid, dark glass body */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(34,211,238,0.20) !important;
    box-shadow: 0 0 18px rgba(34,211,238,0.06);
}
div[data-testid="stDataFrame"] thead tr th {
    background: rgba(34,211,238,0.10) !important;
    color: #d6f4fb !important;
    border-bottom: 1px solid rgba(34,211,238,0.35) !important;
    text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.4px;
}
div[data-testid="stDataFrame"] tbody tr:hover td { background: rgba(34,211,238,0.05) !important; }

/* Tabs: neon underline on the active tab */
button[data-baseweb="tab"] { color: #8fa3bd !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #eaf6ff !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    text-shadow: 0 0 12px rgba(34,211,238,0.4);
}
div[data-baseweb="tab-highlight"] { background: var(--neon-cyan) !important; box-shadow: 0 0 10px var(--neon-cyan); }

/* Buttons: neon outline that fills on hover */
.stButton > button {
    border: 1px solid rgba(34,211,238,0.5) !important;
    color: #d6f4fb !important;
    box-shadow: 0 0 14px rgba(34,211,238,0.10);
}
.stButton > button:hover {
    background: rgba(34,211,238,0.16) !important;
    border-color: var(--neon-cyan) !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(34,211,238,0.35) !important;
}

/* Sidebar: neon edge + heading glow */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(34,211,238,0.22) !important;
    box-shadow: inset -1px 0 0 rgba(34,211,238,0.06);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
    text-shadow: 0 0 14px rgba(34,211,238,0.3);
}

/* Ticker chips: neon ring */
.ticker-chip {
    border: 1px solid rgba(34,211,238,0.25) !important;
    box-shadow: 0 0 14px rgba(34,211,238,0.08);
}

/* Radio / selectbox accents */
div[role="radiogroup"] label:hover { color: var(--neon-cyan) !important; }

/* Metric delta up/down keep the blue/pink scheme but glow a touch */
div[data-testid="stMetricDelta"] { text-shadow: 0 0 10px rgba(76,110,245,0.25); }

/* App header bar (top) faint neon rule */
header[data-testid="stHeader"] { background: transparent !important; border-bottom: 1px solid rgba(34,211,238,0.12); }
</style>
""", unsafe_allow_html=True)


def _epoch(ts) -> int:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return int(t.timestamp()) if pd.notna(t) else 0


# Bright neon palette for the per-strategy equity curves (matches the video).
# Map ANY config/leg name to a clean strategy label — works for the new
# "Strategy__SYMBOL" legs AND the old per-pair names like
# "euraud_donchian_30m_3r" / "yolo_us30_bos_30m_ny", so the graph never
# shows a pair, only the strategy.
_STRAT_KEYWORDS = [
    ("donchian", "Donchian"), ("macd", "MACD"), ("marketopenbos", "BOS"),
    ("_bos", "BOS"), ("bos_", "BOS"), ("nowick", "NoWick"),
    ("vwap", "VWAPReversion"), ("sweep", "SweepReclaim"), ("tod", "TODReversion"),
    ("bollinger", "Bollinger"), ("stochastic", "Stochastic"), ("rsi", "RSI"),
    ("smacross", "SMACross"), ("ma_cross", "SMACross"), ("orb", "ORB"),
]


def strategy_label(config_name: str) -> str:
    # Match against the leading "Strategy" segment only (the "Strategy__SYMBOL"
    # legs from expand_configs, or the whole legacy "<sym>_<strategy>_..." name
    # when there's no "__"), NOT the full lowercased string -- a free substring
    # search let e.g. "TrendPullbackBOS__XAUUSD" match the "bos_" keyword
    # (from "...BOS__xauusd") and get mislabeled as the old MarketOpenBOS "BOS".
    prefix = str(config_name).split("__")[0]
    n = prefix.lower()
    for kw, label in _STRAT_KEYWORDS:
        if kw in n:
            return label
    return prefix


_NEON_CYCLE = ["#22d3ee", "#ff4fd8", "#a3e635", "#f5d90a", "#a78bfa",
               "#4c6ef5", "#fb7185", "#34d399", "#f97316", "#e879f9"]


def build_strategy_curves_html(series_by_config: dict, height: int = 360) -> str:
    """Multi-line neon equity chart — one climbing line per strategy, showing
    how each is doing over time (the 'improving strategies' graph)."""
    import json as _json
    palette = {name: _NEON_CYCLE[i % len(_NEON_CYCLE)] for i, name in enumerate(series_by_config)}
    series_js = _json.dumps({n: [{"time": t, "value": round(v, 2)} for t, v in pts]
                             for n, pts in series_by_config.items()})
    colors_js = _json.dumps(palette)
    legend = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin:0 10px 6px 0;font-size:0.78rem;color:#c7d2e0;'>"
        f"<span style='width:14px;height:3px;border-radius:2px;background:{palette[n]};"
        f"box-shadow:0 0 8px {palette[n]};'></span>{n}</span>"
        for n in series_by_config
    )
    return f"""
<div style="background:#05070e;border:1px solid rgba(34,211,238,0.28);border-radius:12px;padding:10px;
            box-shadow:0 0 30px rgba(34,211,238,0.10), inset 0 0 60px rgba(34,211,238,0.03);">
  <div style="margin-bottom:6px;">{legend}</div>
  <div id="scurves" style="height:{height}px;
       filter: drop-shadow(0 0 3px rgba(255,255,255,0.55)) drop-shadow(0 0 7px rgba(34,211,238,0.45));"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<script>
  const el = document.getElementById("scurves");
  const chart = LightweightCharts.createChart(el, {{
    width: el.clientWidth, height: {height},
    layout: {{ background: {{ color: "#05070e" }}, textColor: "{_AXIS_LABEL}", fontFamily: "Inter, sans-serif" }},
    grid: {{ vertLines: {{ color: "rgba(34,211,238,0.05)" }}, horzLines: {{ color: "rgba(34,211,238,0.05)" }} }},
    rightPriceScale: {{ borderColor: "rgba(34,211,238,0.15)" }},
    timeScale: {{ borderColor: "rgba(34,211,238,0.15)", timeVisible: true }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const data = {series_js}; const colors = {colors_js};
  for (const name in data) {{
    const s = chart.addLineSeries({{ color: colors[name], lineWidth: 3, priceLineVisible: false, lastValueVisible: false }});
    s.setData(data[name]);
  }}
  chart.timeScale().fitContent();
  new ResizeObserver(() => chart.applyOptions({{ width: el.clientWidth }})).observe(el);
</script>
"""


def load_reviews(path: Path):
    """Recent trade reviews + learned per-(slot,session) verdicts for the
    self-healing panel. Tolerant of an older DB with no trade_reviews table."""
    try:
        with sqlite3.connect(str(path)) as c:
            recent = pd.read_sql(
                "SELECT config_name, symbol, direction, exit_reason, pnl_usd, r_multiple, "
                "session, was_win, why FROM trade_reviews ORDER BY id DESC LIMIT 25", c)
            agg = pd.read_sql(
                "SELECT config_name, session, COUNT(*) n, SUM(r_multiple) net_r "
                "FROM trade_reviews GROUP BY config_name, session", c)
        return recent, agg
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# --- Sidebar (outside the live fragment — changing these does a normal full rerun) ---
# VERTEX_BROKER=ninjatrader in .env means the trial bot trades Lucid futures
# through NinjaTrader (see INSTALL.md); its page then comes first. Read with
# dotenv_values so nothing from .env is exported into this process.
from dotenv import dotenv_values  # noqa: E402

_ENV = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
_FUTURES_MODE = (_ENV.get("VERTEX_BROKER") or "").strip().lower() == "ninjatrader"
# Paper trading (VERTEX_PAPER=1, 2026-10-15): the bots fill their own trades on NinjaTrader's minute bars;
# the dashboard reads their paper book (src/metatraderbot/paper_engine.py) instead of NinjaTrader's positions.
_PAPER = (_ENV.get("VERTEX_PAPER") or "").strip() == "1"
# Futures mode shows exactly three pages -- the Lucid trial, funded and lab
# books on NinjaTrader. The MT5 pages stay in the code for MT5 mode.
_PAGES = (("TRIAL", "FUNDED", "DESK", "FLEET", "BRAIN", "LAB", "VERTEX") if _FUTURES_MODE
          else ("LUCIDPRO_TRIAL", "LUCIDPRO_FUNDED", "STRATEGY_LAB"))
# The lab writes separate files per market: its strategy ids do not name a
# symbol, so the MT5 and futures samples must never be mixed.
_LAB_STATE_FILE = ROOT / "data" / ("lab_futures_state.json" if _FUTURES_MODE else "lab_state.json")
_LAB_SIGNALS_FILE = ROOT / "data" / ("lab_futures_signals.jsonl" if _FUTURES_MODE else "lab_signals.jsonl")
# --- Sidebar: brand, the pages with each bot's live status, a live strip, auto-refresh ----------
_PAGE_META = {   # page -> (label, Material icon)
    "TRIAL": ("Trial", "flag"), "FUNDED": ("Funded", "account_balance_wallet"),
    "FLEET": ("Fleet", "groups"), "DESK": ("Desk", "forum"), "BRAIN": ("Brain", "psychology"),
    "LAB": ("Lab", "science"),
    "VERTEX": ("Vertex", "graphic_eq"),
    "LUCIDPRO_TRIAL": ("Trial", "flag"), "LUCIDPRO_FUNDED": ("Funded", "account_balance_wallet"),
    "STRATEGY_LAB": ("Lab", "science"),
}
_KIND_COLOR = {"live": vx.OK, "wait": vx.INFO, "warn": vx.WARN, "alert": vx.DANGER, "idle": vx.TEXT_FAINT}
_REFRESH = {"5 s": 5, "15 s": 15, "30 s": 30, "1 min": 60}
try:
    _LOCAL_TZ = ZoneInfo((_ENV.get("VERTEX_LOCAL_TZ") or "Europe/Prague").strip())
except Exception:  # noqa: BLE001
    _LOCAL_TZ = ZoneInfo("Europe/Prague")
_LOCAL_NAME = str(_LOCAL_TZ.key).split("/")[-1].replace("_", " ")


def _state_file(name: str) -> tuple[dict, float | None]:
    """(content, seconds since it was written) of data/<name>; ({}, None) when missing."""
    path = ROOT / "data" / name
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
        return json.loads(path.read_text(encoding="utf-8")), max(0.0, age)
    except (OSError, ValueError):
        return {}, None


def _bot_line(age: float | None, who: str, racing: bool) -> tuple[str, str]:
    if age is None:
        return "not started yet", "idle"
    if age > 180:                      # the bots save their state every 15-30 s
        return f"{who} · bot stopped", "alert"
    return (f"{who} · race running", "live") if racing else (f"{who} · waiting", "wait")


# The funded account's rules, as run_lucid_futures_funded.py applies them (LucidPro 50K funded).
_F_ANCHOR, _F_KEEP, _F_LOCK, _F_MLL = 50_000.0, 52_100.0, 50_100.0, 2_000.0
_F_WANT = (500.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)
_F_CAPS = (2_000.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)
_F_TAKE_MIN, _F_TAKE_MAX = 150.0, 1_500.0      # the funded bot's DAY_TAKE_MIN / DAY_TAKE_MAX ($1,040 until 2026-10-21)
_F_SPLIT = 0.90                    # the trader's share of a payout


def _funded_need(balance: float, cycle: float, best: float, n_paid: int) -> float:
    """Profit the next payout still needs: the money above $52,100, the 40% rule, the $500 minimum."""
    k = min(n_paid, len(_F_WANT) - 1)
    return max(_F_KEEP + _F_WANT[k] - balance, best / 0.40 - cycle if best > 0 else 0.0, 500.0 - cycle)


# The deployed strategy and the new one, run side by side on paper (owner, 2026-10-21: "one tab the current strategy
# and one tab this new strategy and we will compare them"). trial_tag: the trial bot's file names
# (state_lucid_futures{tag}.json); funded: which funded accounts belong to it (the A-D one is Sim102-AD).
_STRATEGIES = {
    "Current · 200-day trend": {"trial_tag": "", "funded_ad": False,
                                "what": "Current strategy: buy when NQ is above its 200-day average, sell when below."},
    "New · A−D breadth": {"trial_tag": "_ad", "funded_ad": True,
                          "what": "New strategy (2-month paper test): buy when more S&P 500 stocks rose than fell in "
                                  "the session before, sell when more fell -- the same race and rules otherwise."},
}
_AD_TAGS = ("_ad", "_Sim102_AD")   # the new strategy's file tags: trial bot (--strategy ad), funded account Sim102-AD


def _is_ad_account(inst: dict) -> bool:
    return str(inst.get("tag", "")).upper().endswith("_AD")


def _compare_until():
    """The side-by-side test's last day (config/compare_ad_until.txt; owner 2026-10-21: "add the new strategy along
    all of this and we keep it for testing for 2 months"), or None."""
    try:
        return datetime.strptime((ROOT / "config" / "compare_ad_until.txt").read_text(encoding="utf-8").split()[0],
                                 "%Y-%m-%d").date()
    except (OSError, ValueError, IndexError):
        return None


def _compare_on() -> bool:
    """The strategy switch shows while the second strategy runs (the test's dates, or VERTEX_COMPARE_AD=1) or has run."""
    until = _compare_until()
    return ((_ENV.get("VERTEX_COMPARE_AD") or "").strip() == "1" or (until is not None and datetime.now().date() <= until)
            or (ROOT / "data" / "state_lucid_futures_ad.json").exists())


def _funded_instances(data_dir: Path | None = None, env: dict | None = None) -> list[dict]:
    """Every funded account a funded bot trades (owner, 2026-10-09: "we will be trading multiple
    accounts"): the first one (state_lucid_futures_funded.json) and each one started with --account
    (state_lucid_futures_funded_<name>.json), plus accounts listed in LUCID_FUNDED_EXTRA_ACCOUNTS whose
    bot has not written anything yet."""
    d = data_dir or (ROOT / "data")
    env = _ENV if env is None else env
    primary = (env.get("NT_FUNDED_ACCOUNT") or "Sim102").strip()
    out = []
    for p in [d / "state_lucid_futures_funded.json"] + sorted(d.glob("state_lucid_futures_funded_*.json")):
        tag = p.stem[len("state_lucid_futures_funded"):]
        if tag == "_roster" or (tag == "" and not p.exists()):
            continue
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        out.append({"account": str(state.get("account") or (tag.lstrip("_") if tag else primary)), "tag": tag,
                    "state_path": p, "trades_path": d / f"lucid_futures_funded_trades{tag}.jsonl", "state": state,
                    "age_s": max(0.0, datetime.now().timestamp() - p.stat().st_mtime) if p.exists() else None})
    for a in (x.strip() for x in (env.get("LUCID_FUNDED_EXTRA_ACCOUNTS") or "").split(",")):
        if a and not any(i["account"] == a for i in out):
            import re
            tag = "_" + re.sub(r"[^A-Za-z0-9]+", "_", a).strip("_")
            out.append({"account": a, "tag": tag, "state_path": d / f"state_lucid_futures_funded{tag}.json",
                        "trades_path": d / f"lucid_futures_funded_trades{tag}.jsonl", "state": {}, "age_s": None})
    if not out:
        out.append({"account": primary, "tag": "", "state_path": d / "state_lucid_futures_funded.json",
                    "trades_path": d / "lucid_futures_funded_trades.jsonl", "state": {}, "age_s": None})
    return out


def _funded_overview(inst: dict) -> dict:
    """One funded account in a row of plain numbers, from its bot's saved state (no NinjaTrader call)."""
    st_ = inst["state"]
    run = st_.get("run") or {}
    payouts = run.get("payouts") or []
    realized, paid = float(run.get("realized_usd", 0.0)), float(run.get("paid_usd", 0.0))
    bal = _F_ANCHOR + realized - paid
    today = realized - float(run.get("day_start_realized", realized))
    need = _funded_need(bal, float(run.get("cycle_pnl", 0.0)) + today,
                        max(float(run.get("cycle_best_day", 0.0)), today), len(payouts))
    if not st_:
        status = "not started yet"
    elif st_.get("halted"):
        status = "stopped"
    elif inst.get("age_s") is not None and inst["age_s"] > 600:
        status = "bot not running"
    elif run.get("tickets"):
        status = "race running"
    elif need <= 0:
        status = "payout money reached"
    else:
        status = "waiting for 19:00 New York"
    every = [p for h in (st_.get("history") or []) for p in (h.get("payout_list") or [])] + payouts
    return {"Account": inst["account"], "Status": status,
            "Funded account": f"#{int(st_.get('runs_total') or 1)}" if st_ else "—",
            "Balance": f"${bal:,.0f}" if st_ else "—",
            "Next payout needs": (f"${need:,.0f}" if need > 0 else "ready") if st_ else "—",
            "Open trade": (f"{'+' if float(st_.get('floating_usd') or 0) >= 0 else '−'}"
                           f"${abs(float(st_.get('floating_usd') or 0)):,.0f}") if run.get("tickets") else "",
            "Payouts": len(every), "Paid to you": f"${_F_SPLIT * sum(float(p.get('amount_usd') or 0) for p in every):,.0f}",
            "Blown": int(st_.get("runs_blown") or 0)}


def _nav_status() -> dict[str, tuple[str, str]]:
    """page -> (one status line, kind), from the saved state files: cheap, no NinjaTrader call."""
    if not _FUTURES_MODE:
        return {}
    out = {}
    tr, age = _state_file("state_lucid_futures.json")
    who = f"Evaluation #{int(tr.get('attempts') or 0) + 1}"
    if tr.get("halted") == "pass":
        out["TRIAL"] = ("Evaluation passed · stopped", "live")
    elif tr.get("awaiting_reset"):
        out["TRIAL"] = ("Failed · reset the account", "alert")
    else:
        out["TRIAL"] = _bot_line(age, who, bool((tr.get("attempt") or {}).get("tickets")))
    fu, age = _state_file("state_lucid_futures_funded.json")
    who = f"Account #{int(fu.get('runs_total') or 1)}"
    out["FUNDED"] = ((f"{who} · stopped", "alert") if fu.get("halted") else
                     _bot_line(age, who, bool((fu.get("run") or {}).get("tickets"))))
    insts = _funded_instances()
    if len(insts) > 1:                                   # several funded accounts: one line for all
        racing = sum(1 for i in insts if (i["state"].get("run") or {}).get("tickets"))
        down = sum(1 for i in insts if i["age_s"] is None or i["age_s"] > 600)
        out["FUNDED"] = (f"{len(insts)} accounts · {racing} racing" + (f" · {down} not running" if down else ""),
                         "warn" if down else ("live" if racing else "wait"))
    out["FLEET"] = ("Many accounts · simulated", "idle")
    dk, dk_age = _state_file("desk_state.json")
    if not dk:
        out["DESK"] = ("not started yet", "idle")
    else:
        dday = (dk.get("days") or {}).get(max(dk.get("days") or {"": 0})) or {}
        out["DESK"] = (f"Eval #{(dk.get('account') or {}).get('n', 1)} · {dday.get('trades', 0)} trades today"
                       + (" · not running" if dk_age is None or dk_age > 600 else ""),
                       "warn" if dk_age is None or dk_age > 600 else "live")
    br, _ = _state_file("brain.json")
    pl, _ = _state_file("prediction_lab.json")
    out["BRAIN"] = (f"{len(br.get('trades') or {})} trades · {len(pl.get('summary') or {})} predictors", "idle")
    lab, _ = _state_file("lab_futures_state.json")
    out["LAB"] = (f"{len(lab.get('stats') or {})} strategies forward-tested", "idle")
    try:
        on = sorted(k for k, v in vxst.vertex_live(ROOT / "data").items() if v.get("online"))
    except Exception:  # noqa: BLE001
        on = []
    out["VERTEX"] = ((f"Ready on {on[0]}" if len(on) == 1 else f"Ready on {len(on)} laptops", "live") if on
                     else ("Voice butler · not running", "idle"))
    return out


def _paper_status() -> tuple[str, str]:
    """Paper mode: where the bots' prices come from right now (NinjaTrader is optional)."""
    from metatraderbot import lucid_rules as _lr_pp
    if _lr_pp.session_state(datetime.now(timezone.utc)) == _lr_pp.Session.CLOSED:
        return ("Paper trading · market closed", "wait")
    try:
        _pe = _fresh_prices()[1]
        eng = _pe.PaperEngine(account=(_ENV.get("NT_ACCOUNT") or "Sim101").strip(), read_only=True)
        ok = eng.is_tradeable("NQ")
        src = eng.source
    except Exception:  # noqa: BLE001
        ok, src = False, ""
    if not ok:
        return ("Paper trading paused — no fresh prices (internet down?)", "alert")
    return (("Paper trading · live prices from NinjaTrader", "live") if src == "NinjaTrader" else
            ("Paper trading · internet prices (10 min delay) — NinjaTrader not used", "live"))


def _nt_status() -> tuple[str, str]:
    """(text, kind): is NinjaTrader running, logged in and connected?"""
    if _PAPER:
        return _paper_status()
    from metatraderbot.nt_engine import NinjaTraderEngine
    eng = NinjaTraderEngine(account=(_ENV.get("NT_ACCOUNT") or "Sim101").strip(),
                            port=int(_ENV.get("NT_BRIDGE_PORT") or 45973), timeout_s=1.5)
    try:
        conns = eng._call("ping").get("connections") or []
    except Exception:  # noqa: BLE001 -- closed, or open at its login screen (add-ons not loaded)
        import subprocess
        try:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq NinjaTrader.exe", "/NH"], capture_output=True,
                                 text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except Exception:  # noqa: BLE001
            out = ""
        return ("Open, but not logged in", "warn") if "NinjaTrader.exe" in out else ("Not running", "alert")
    states = [str(c.get("status", "")) for c in conns if isinstance(c, dict)]
    if _nt_login_failures() >= 3:
        return ("Its login keeps failing — close it and sign in again", "alert")
    if "Connected" not in states:
        return ("Logged in, no market connection", "warn")
    if _PAPER:                       # paper mode: the bots price and fill on the minute bars, not on the quote
        from metatraderbot import lucid_rules as _lr_pp
        if _lr_pp.session_state(datetime.now(timezone.utc)) == _lr_pp.Session.CLOSED:
            return ("Connected · paper trading (market closed)", "wait")
        try:
            b = eng.get_bars("NQ", "1m", 5)
            traded = b[b["volume"] > 0]
            age = (pd.Timestamp.now(tz="UTC") - traded.index[-1]).total_seconds() if len(traded) else 1e9
        except Exception:  # noqa: BLE001
            age = 1e9
        return (("Live prices · paper trading", "live") if age < 300 else
                ("Connected, but no fresh minute prices — paper trading paused", "alert"))
    # 2026-10-14: "Connected" for 10 hours while NinjaTrader's NQ price stood still -- the bots could not race
    from metatraderbot import lucid_rules as _lr_nt, target_race as _tr_nt
    if _lr_nt.session_state(datetime.now(timezone.utc)) != _lr_nt.Session.CLOSED:
        try:
            problem = _vx_fresh(_tr_nt).price_problem(eng, "NQ")
        except Exception:  # noqa: BLE001
            problem = None
        if problem:
            return (f"Connected, but its NQ price {problem} — no live market data, the bots cannot trade",
                    "alert")
    return ("Connected", "live")


def _nt_login_failures(minutes: int = 15) -> int:
    """How often NinjaTrader's own log says its sign-in failed in the last `minutes`. On 2026-10-08 its
    stored Google sign-in stopped working; it retried every 30 s all night (~1,600 times) while still
    showing "Connected", and NinjaTrader then locked the login ("too many incorrect login attempts")."""
    for d in (Path.home() / "Documents" / "NinjaTrader 8" / "log",):
        try:
            logs = sorted(d.glob("log.*.en.txt"), key=lambda p: p.stat().st_mtime)
            if not logs:
                continue
            with logs[-1].open("rb") as fh:
                fh.seek(max(0, logs[-1].stat().st_size - 65_536))
                tail = fh.read().decode("utf-8", "replace").splitlines()
            since = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
            return sum(1 for x in tail if x[:19] >= since
                       and ("problem authenticating" in x or "Logon failed" in x))
        except OSError:
            continue
    return 0


def _race_rows(nav: dict) -> list[tuple[str, str, str]]:
    """The race now (which bot, when it auto-closes) and the next one for the bots still waiting."""
    from metatraderbot import target_race as _tr
    now = datetime.now(timezone.utc)
    cfg = _tr.RaceConfig(entry=(_ENV.get("LUCID_RACE_ENTRY") or "19:00").strip())
    tr, _ = _state_file("state_lucid_futures.json")
    fu, _ = _state_file("state_lucid_futures_funded.json")
    bots = (("TRIAL", "Trial", tr.get("race_day")), ("FUNDED", "Funded", (fu.get("run") or {}).get("race_day")))
    live = [name for key, name, _ in bots if nav.get(key, ("", ""))[1] == "live"]
    rows = []
    if live:
        close = _tr.exit_time(now, cfg).astimezone(_LOCAL_TZ)
        rows.append(("Race running", f"{' + '.join(live)} <span>· auto-close {close:%H:%M}</span>", "live"))
    waiting = [day for key, name, day in bots if name not in live]
    if waiting:
        nxt = min(_tr.next_entry(now, cfg, day) for day in waiting)
        mins = int((nxt - now).total_seconds() // 60)
        if mins <= 1:
            rows.append(("Next race", "Any minute now", "live"))
        else:
            left = (f"in {mins} min" if mins < 60 else f"in {mins // 60} h {mins % 60:02d} min" if mins < 24 * 60
                    else f"in {mins // 1440} d {mins % 1440 // 60} h")
            rows.append(("Next race", f"{nxt.astimezone(_LOCAL_TZ):%a %H:%M} <span>· {left}</span>", "wait"))
    return rows


_NAV_CSS = f"""
<style>
/* the button that reopens a closed sidebar lives in the hidden top bar: keep it visible */
[data-testid="stExpandSidebarButton"] {{ visibility:visible !important; background:{vx.SURFACE_ALT};
  border:1px solid {vx.BORDER}; border-radius:8px; }}
.stApp:has(section[data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stMainBlockContainer"] {{
  padding-top:3.4rem !important; }}
/* the page switch at the top of every page */
.st-key-nav_top {{ margin-bottom:2px; }}
.st-key-nav_top button {{ min-height:34px; padding:2px 14px; }}
.st-key-nav_top button p {{ font-size:13px; font-weight:500; }}
/* style-only blocks (this one, vx.CSS, ...) must not take a row: each empty row adds a 1rem gap */
div[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] > div > div[data-testid="stMarkdownContainer"] > style:only-child) {{
  display:none; }}
section[data-testid="stSidebar"] {{ background:{vx.SURFACE}; border-right:1px solid {vx.BORDER}; }}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{ height:2.2rem; min-height:2.2rem;
  padding:0.6rem 1rem 0 !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top:0.2rem; }}
.vx-brand {{ display:flex; align-items:center; gap:11px; padding:0 4px 18px; margin-bottom:6px;
  border-bottom:1px solid {vx.BORDER}; }}
.vx-brand img {{ width:34px; height:34px; border-radius:9px; }}
.vx-brand-name {{ font-size:16px; font-weight:650; color:{vx.TEXT}; letter-spacing:-0.2px; line-height:1.2; }}
.vx-brand-sub {{ font-size:11.5px; color:{vx.TEXT_FAINT}; }}
.vx-side-h {{ font-size:10.5px; font-weight:600; letter-spacing:0.9px; text-transform:uppercase;
  color:{vx.TEXT_FAINT}; margin:20px 0 8px 12px; }}
/* the page list: a radio, drawn as cards */
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{ gap:4px; position:relative; padding-top:28px; }}
section[data-testid="stSidebar"] [data-testid="stRadioGroup"]::before {{ content:"Bots"; position:absolute; top:6px;
  left:12px; font-size:10.5px; font-weight:600; letter-spacing:0.9px; text-transform:uppercase; color:{vx.TEXT_FAINT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ width:100%; margin:0 !important; position:relative;
  padding:9px 12px 10px 14px; border-radius:10px; border:1px solid transparent; cursor:pointer;
  transition:background .15s ease, border-color .15s ease; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(4) {{ margin-top:34px !important; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(4)::before {{ content:"Insights";
  position:absolute; top:-26px; left:12px; pointer-events:none; font-size:10.5px; font-weight:600;
  letter-spacing:0.9px; text-transform:uppercase; color:{vx.TEXT_FAINT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(7) {{ margin-top:34px !important; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(7)::before {{ content:"Assistant";
  position:absolute; top:-26px; left:12px; pointer-events:none; font-size:10.5px; font-weight:600;
  letter-spacing:0.9px; text-transform:uppercase; color:{vx.TEXT_FAINT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background:rgba(255,255,255,0.035); }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background:{vx.SURFACE_ALT};
  border-color:{vx.BORDER}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"]::after {{ content:"";
  position:absolute; left:-1px; top:11px; bottom:11px; width:3px; border-radius:0 3px 3px 0; background:{vx.ACCENT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child > div:not([data-testid]) {{
  display:none !important; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:nth-child(2) {{ padding-left:0 !important;
  margin-left:0 !important; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{
  font-size:14px; font-weight:600; color:{vx.TEXT_MUTED}; margin:0; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [data-testid="stMarkdownContainer"] p {{
  color:{vx.TEXT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] span[role="img"] {{ font-size:18px !important;
  margin-right:8px; color:{vx.TEXT_FAINT}; vertical-align:-4px !important; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] span[role="img"] {{ color:{vx.ACCENT}; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stCaptionContainer"] p {{
  font-size:12px; color:{vx.TEXT_MUTED}; margin:3px 0 0 26px; line-height:1.35; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stCaptionContainer"] p::before {{
  content:""; display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px;
  vertical-align:1px; background:{vx.TEXT_FAINT}; }}
@keyframes vxpulse {{ 0% {{ box-shadow:0 0 0 0 {vx.OK}88; }} 70% {{ box-shadow:0 0 0 6px {vx.OK}00; }}
  100% {{ box-shadow:0 0 0 0 {vx.OK}00; }} }}
/* the live strip */
.vx-live {{ background:{vx.BG}; border:1px solid {vx.BORDER}; border-radius:10px; padding:4px 12px; }}
.vx-live-row {{ padding:8px 0; border-bottom:1px solid {vx.BORDER_SOFT}; }}
.vx-live-row:last-child {{ border-bottom:none; }}
.vx-live-row .k {{ font-size:10.5px; font-weight:600; letter-spacing:0.6px; text-transform:uppercase;
  color:{vx.TEXT_FAINT}; margin-bottom:2px; }}
.vx-live-row .v {{ font-size:13px; font-weight:500; color:{vx.TEXT}; }}
.vx-live-row .v span {{ color:{vx.TEXT_MUTED}; font-weight:400; }}
.vx-live-row .v i {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:7px;
  vertical-align:1px; }}
.vx-live-row .v i.live {{ animation:vxpulse 2s infinite; }}
/* auto-refresh: a full-width segmented control */
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] [role="radiogroup"] {{ display:flex; width:100%; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {{ flex:1 1 0; min-height:30px; padding:2px 4px; }}
section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p {{ font-size:12px; }}
</style>
"""


def _dot_css(nav: dict, pages: tuple) -> str:
    """Colour each page's status dot by its kind (pulsing while a race runs)."""
    rules = []
    for i, page in enumerate(pages, start=1):
        kind = nav.get(page, ("", "idle"))[1]
        sel = (f'section[data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type({i}) '
               f'[data-testid="stCaptionContainer"] p::before')
        rules.append(f"{sel} {{ background:{_KIND_COLOR.get(kind, vx.TEXT_FAINT)};"
                     f"{' animation:vxpulse 2s infinite;' if kind == 'live' else ''} }}")
    return "\n".join(rules)


@st.fragment(run_every=30)
def _sidebar_live() -> None:
    """NinjaTrader, the next race and the time -- refreshed every 30 s on its own. When a bot
    changes phase (a race opens or ends), the whole page is redrawn so the list above updates."""
    nav = _nav_status()
    drawn = float(st.session_state.get("_nav_drawn") or 0.0)
    if nav != st.session_state.get("_nav_sig") and datetime.now().timestamp() - drawn > 5:
        st.rerun()
    rows = []
    if _FUTURES_MODE:
        text, kind = _nt_status()
        rows.append(("Prices" if _PAPER else "NinjaTrader", text, kind))
        rows += _race_rows(nav)
    now = datetime.now(timezone.utc)
    rows.append(("Time", f"{_LOCAL_NAME} {now.astimezone(_LOCAL_TZ):%H:%M} "
                         f"<span>· New York {now.astimezone(ZoneInfo('America/New_York')):%H:%M}</span>", ""))
    html = []
    for k, v, kind in rows:
        dot = (f'<i class="{"live" if kind == "live" else ""}" style="background:{_KIND_COLOR[kind]}"></i>'
               if kind else "")
        html.append(f'<div class="vx-live-row"><div class="k">{k}</div><div class="v">{dot}{v}</div></div>')
    st.markdown('<div class="vx-side-h">Live</div><div class="vx-live">' + "".join(html) + "</div>",
                unsafe_allow_html=True)


def _sidebar_nav(pages: tuple) -> str:
    nav = _nav_status()
    st.session_state["_nav_sig"] = nav
    st.session_state["_nav_drawn"] = datetime.now().timestamp()
    st.sidebar.markdown(_NAV_CSS.replace("</style>", _dot_css(nav, pages) + "\n</style>"), unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="vx-brand"><img src="{_logo_data_uri(_LOGO_PATH)}" alt=""/><div>'
        f'<div class="vx-brand-name">Vertex</div><div class="vx-brand-sub">'
        f'{("LucidPro 50K · paper, internet prices" if _PAPER else "LucidPro 50K · NinjaTrader") if _FUTURES_MODE else "LucidPro · MetaTrader 5"}</div></div></div>',
        unsafe_allow_html=True)
    asked = str(st.query_params.get("page", "")).upper()            # ?page=funded opens that page
    if "nav_page" not in st.session_state or st.session_state["nav_page"] not in pages:
        st.session_state["nav_page"] = asked if asked in pages else pages[0]
    label = lambda k: f":material/{_PAGE_META.get(k, (k, 'circle'))[1]}: {_PAGE_META.get(k, (k, ''))[0]}"  # noqa: E731

    def _from_side() -> None:
        st.session_state["nav_top"] = st.session_state["nav_page"]

    def _from_top() -> None:
        if st.session_state.get("nav_top"):
            st.session_state["nav_page"] = st.session_state["nav_top"]
        else:                           # clicking the selected page again must not leave none selected
            st.session_state["nav_top"] = st.session_state["nav_page"]

    page = st.sidebar.radio("Page", pages, key="nav_page", label_visibility="collapsed", format_func=label,
                            captions=[nav.get(k, ("", ""))[0] for k in pages], on_change=_from_side)
    # The same switch at the top of every page: the bots stay reachable with the sidebar closed
    # (2026-10-07: the owner closed it and could not find the other bots).
    if st.session_state.get("nav_top") != page:
        st.session_state["nav_top"] = page
    st.segmented_control("Page", pages, key="nav_top", format_func=label, label_visibility="collapsed",
                         on_change=_from_top)
    if asked != page:
        st.query_params["page"] = page.lower()
    with st.sidebar:
        _sidebar_live()
    return page


def _refresh_seconds(default: int = 5) -> int:
    """The page's auto-refresh, as a small segmented control at the bottom of the sidebar."""
    st.sidebar.markdown('<div class="vx-side-h">Auto-refresh</div>', unsafe_allow_html=True)
    pick = st.sidebar.segmented_control(
        "Auto-refresh", list(_REFRESH), key=f"refresh_{bot_choice}", label_visibility="collapsed", width="stretch",
        default=next((k for k, v in _REFRESH.items() if v == default), "5 s"))
    return _REFRESH.get(pick, default)


# --- Vertex, the voice butler, inside the dashboard (owner 2026-10-09: "keep it in the same style as the whole
# dashboard ... keep it all in that dashboard"). Each laptop's Vertex app reports what it hears and says through
# scripts/vertex_api.py (data/vertex_log.jsonl); the dashboard shows it -- a card on every page while Vertex
# listens or talks, and the VERTEX page -- and follows the voice commands meant for this laptop: pages, tabs,
# the funded account, the fleet settings, scrolling. A laptop's dashboard link carries ?viewer=<its name> (the
# viewer app and Vertex add it); without it the dashboard follows every laptop's Vertex.
def _fresh_prices():
    """free_prices + paper_engine, reloaded when their files change (2026-10-16: the dashboard kept the first
    free_prices, without 5-minute bars, and every trade chart said "unsupported timeframe 5m")."""
    from metatraderbot import free_prices as _fp, paper_engine as _pe
    return _vx_fresh(_fp), _vx_fresh(_pe)


def _vx_fresh(mod):
    """A module from src/, reloaded when its file changed -- Streamlit reloads only the modules beside this
    script (see _fleet_mod)."""
    import importlib
    mtime = Path(mod.__file__).stat().st_mtime
    if getattr(mod, "_loaded_mtime", None) != mtime:
        mod = importlib.reload(mod)
        mod._loaded_mtime = mtime
    return mod


vxst, vxv = _vx_fresh(vxst), _vx_fresh(vxv)
_VX_DATA = ROOT / "data"
_VIEWER = vxst.viewer_name(st.query_params.get("viewer", ""))
_VX_FLEET_PERIODS = ("Real test · since 1 Oct", "Last 3 months", "Last 12 months")      # the FLEET page's periods


def _vx_mine(viewer: str) -> bool:
    return not _VIEWER or viewer == _VIEWER


def _vx_actions(ui: dict) -> dict:
    """A voice command as clicks on this dashboard (app_vertex.command_script), by the labels on the buttons.
    Clicking -- rather than setting the switches from here -- keeps the page exactly as after a real click
    (2026-10-09: set from here, the page switch at the top kept the old page and jumped back to it later)."""
    act: dict = {}
    page = str(ui.get("page") or "").upper()
    if page in _PAGES:
        act["page"] = _PAGE_META.get(page, (page,))[0]
    acct = str(ui.get("account") or "").strip().lower()
    if acct:
        names = [i["account"] for i in _funded_instances()]
        hit = next((n for n in names if n.lower() == acct), None) or next((n for n in names if acct in n.lower()),
                                                                           None)
        if hit and len(names) > 1:                   # one account has no picker
            act["account"] = hit
            act.setdefault("page", _PAGE_META["FUNDED"][0])
    if ui.get("fleet_period") in _VX_FLEET_PERIODS:
        act["fleet_period"] = ui["fleet_period"]
    for k in ("fleet_start", "fleet_up_to"):
        if isinstance(ui.get(k), int) and 1 <= ui[k] <= 5:
            act[k] = ui[k]
    if any(k in act for k in ("fleet_period", "fleet_start", "fleet_up_to")):
        act.setdefault("page", _PAGE_META["FLEET"][0])
    for k in ("tab", "scroll"):
        if ui.get(k):
            act[k] = str(ui[k])
    return act


bot_choice = _sidebar_nav(_PAGES)


def _vx_who(live: dict) -> str:
    """The laptop whose Vertex this dashboard shows: its own (?viewer=), else the busiest / latest online one."""
    if _VIEWER and _VIEWER in live:
        return _VIEWER
    on = [(v.get("state") in ("listening", "thinking", "speaking"), v.get("ts", ""), k)
          for k, v in live.items() if v.get("online")]
    return max(on)[2] if on else ""


def _vx_click(act: dict, nonce: str, key: str = "vx_js") -> None:
    with st.container(key=key):
        components.html(vxv.command_script(act, nonce), height=0)


@st.fragment(run_every=2)
def _vertex_dock() -> None:
    """Every 2 s: new voice commands for this laptop, and the card while Vertex listens or talks (45 s after)."""
    try:
        feed = vxst.vertex_feed(_VX_DATA, 80)
        live = vxst.vertex_live(_VX_DATA)
    except Exception:  # noqa: BLE001 -- a half-written file: next time
        return
    if "_vx_seen" not in st.session_state:          # opening the dashboard: what was said before is not an order
        st.session_state["_vx_seen"] = feed[-1]["ts"] if feed else ""
    new = [e for e in feed if e.get("ts", "") > st.session_state["_vx_seen"]]
    if new:
        st.session_state["_vx_seen"] = new[-1]["ts"]
        cmd = next((e for e in reversed(new) if e.get("kind") == "exchange" and e.get("ui")
                    and _vx_mine(e.get("viewer", ""))), None)
        act = _vx_actions(cmd["ui"]) if cmd else {}
        if act:
            _vx_click(act, cmd.get("id", ""))
    if bot_choice == "VERTEX":                        # the page itself shows everything
        return
    who = _vx_who(live)
    me = live.get(who, {})
    state = me.get("state", "offline") if me.get("online") else "offline"
    said = [e for e in feed if e.get("kind") in ("exchange", "announce")]
    latest = said[-1] if said else None
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(latest["ts"])).total_seconds() if latest else 1e9
    except (TypeError, ValueError):
        age = 1e9
    busy = state in ("listening", "thinking", "speaking")
    if not busy and (age > 45 or st.session_state.get("_vx_dismissed") == (latest or {}).get("id")):
        return
    entry = latest
    if state in ("listening", "thinking") and latest and latest.get("ts", "") < me.get("ts", ""):
        entry = None                                  # a new question is under way
    with st.container(key="vx_dock"):
        st.markdown(vxv.dock_html(entry, state, who, _LOCAL_TZ, me.get("detail", "") if busy else ""),
                    unsafe_allow_html=True)
        c = st.columns(2)
        if c[0].button("Conversation", key="vx_dock_open", icon=":material/forum:", width="stretch"):
            _vx_click({"page": _PAGE_META["VERTEX"][0]}, datetime.now().isoformat(), "vx_js_open")
        if c[1].button("Dismiss", key="vx_dock_x", icon=":material/close:", width="stretch"):
            st.session_state["_vx_dismissed"] = (latest or {}).get("id")
            st.rerun(scope="fragment")


if _FUTURES_MODE:
    st.markdown(vxv.CSS, unsafe_allow_html=True)
    if st.session_state.get("_vx_screen") != bot_choice:     # Vertex knows which page this laptop shows
        st.session_state["_vx_screen"] = bot_choice
        try:
            vxst.record_screen(_VX_DATA, _VIEWER, bot_choice)
        except OSError:
            pass
    with st.container(key="vx_dock_slot"):                   # out of the page's flow: no gap at the top
        _vertex_dock()


# --- VERTEX: the voice butler's own page -------------------------------------------------------------------------
if bot_choice == "VERTEX":
    st.markdown(vx.CSS, unsafe_allow_html=True)

    @st.cache_data(ttl=10, show_spinner=False)
    def _vx_summary() -> dict:
        return vxst.summary(_VX_DATA)

    def _vx_table(path: str) -> None:
        rows = vxst.lookup(_vx_summary(), path) or []
        cols = [c for c in vxv.TABLES.get(path, []) if any(c in r for r in rows)]
        if not rows or not cols:
            return
        df = pd.DataFrame(rows)[cols]
        if "ts" in df:
            df["ts"] = [vxv.when(t, _LOCAL_TZ) for t in df["ts"]]
        cfg = {c: st.column_config.NumberColumn(c.replace("_usd", "").replace("_", " ").capitalize(), format="dollar")
               for c in cols if c.endswith("_usd")}
        cfg.update({c: c.replace("_", " ").capitalize() for c in cols if not c.endswith("_usd")})
        st.markdown(f'<div class="vx-label" style="margin:10px 0 4px">{vxv.TABLE_TITLES.get(path, path)}</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(12), hide_index=True, width="stretch", column_config=cfg)

    def _vx_send(kind: str, viewer: str, **kw) -> dict | None:
        try:
            return vxst.record_vertex_event(_VX_DATA, {"kind": kind, "viewer": viewer, **kw})
        except (OSError, ValueError) as e:
            st.toast(f"Could not reach Vertex: {e}")
            return None

    def _vx_state(live: dict) -> tuple[str, dict, str]:
        who = _vx_who(live)
        me = live.get(who, {})
        return who, me, (me.get("state", "offline") if me.get("online") else "offline")

    @st.fragment(run_every=2)
    def _vertex_hero() -> None:
        who, me, state = _vx_state(vxst.vertex_live(_VX_DATA))
        title = {"listening": "Listening…", "thinking": "Thinking…", "speaking": "Speaking",
                 "idle": "At your service"}.get(state, "Not running")
        if state == "offline":
            sub = ("Vertex isn't running on any laptop right now. Start <b>Vertex Voice</b> on a laptop (desktop "
                   "icon; it lives by the clock), then say <b>“Hey Vertex”</b>.")
        else:
            sub = (f"Say <b>“Hey Vertex”</b> near <b>{vxv.esc(who)}</b>, or type below. Everything Vertex says "
                   "appears here and on every page, each number checked in the bots' records. It can also move "
                   "this dashboard: pages, tabs, accounts, the fleet plan, scrolling.")
        if state == "thinking" and me.get("detail") not in ("", None, "writing it down"):
            sub = f"“{vxv.esc(me.get('detail'))}”"
        st.markdown(f'<div class="vxh">{vxv.orb(state, 132)}<div class="vxh-t"><div class="vxh-name">VERTEX</div>'
                    f'<div class="vxh-state">{title}</div><div class="vxh-sub">{sub}</div></div></div>',
                    unsafe_allow_html=True)

    @st.fragment(run_every=2)
    def _vertex_conversation() -> None:
        feed = [e for e in vxst.vertex_feed(_VX_DATA, 300) if e.get("kind") in ("exchange", "announce")]
        wait = st.session_state.get("_vx_wait") or {}
        if wait and not any(e.get("reply_to") == wait.get("id") for e in feed):
            try:
                waited = (datetime.now(timezone.utc) - datetime.fromisoformat(wait["ts"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                waited = 1e9
            if waited < 90:
                st.markdown(f'<div class="vxc latest"><div class="vxc-m"><span>{vxv.esc(wait.get("viewer"))}'
                            f' · typed</span></div><div class="vxc-q"><span>YOU</span>{vxv.esc(wait.get("text"))}'
                            f'</div><div class="vxc-a" style="display:flex;align-items:center;gap:10px">'
                            f'{vxv.orb("thinking", 26)}<span style="width:auto">VERTEX IS THINKING…</span>'
                            f'</div></div>', unsafe_allow_html=True)
            else:
                st.session_state.pop("_vx_wait", None)
        if not feed and not wait:
            st.markdown(vx.banner("Nothing said yet. Ask something above — or say “Hey Vertex, how did we do "
                                  "today?”", "info"), unsafe_allow_html=True)
        for i, e in enumerate(reversed(feed[-40:])):
            st.markdown(vxv.entry_html(e, _LOCAL_TZ, latest=(i == 0)), unsafe_allow_html=True)
            if i == 0:
                for path in e.get("show") or []:
                    _vx_table(path)

    @st.fragment(run_every=2)
    def _vertex_laptops() -> None:
        live = vxst.vertex_live(_VX_DATA)
        if not live:
            st.markdown(vx.banner("No laptop has run Vertex yet. Install it with "
                                  "<code>vertex_voice/Install-Voice.ps1</code> (see the note "
                                  "“Vertex voice butler”).", "warn"), unsafe_allow_html=True)
        for name, v in sorted(live.items()):
            on = v.get("online")
            st_word = vxv.STATE_WORDS.get(v.get("state"), "") if on else "not running"
            pills = (vx.pill("voice on" if v.get("voice") else "muted", vx.OK if v.get("voice") else vx.WARN)
                     + " " + vx.pill("announcing" if v.get("announce") else "no announcements",
                                     vx.INFO if v.get("announce") else vx.TEXT_FAINT)) if on else ""
            st.markdown(f'<div class="vxl">{vxv.orb(v.get("state") if on else "offline", 30)}<div>'
                        f'<div class="vxl-n">{vxv.esc(name)}{" · this laptop" if name == _VIEWER else ""}</div>'
                        f'<div class="vxl-s">{st_word}</div></div><div class="vxl-f">{pills}</div></div>',
                        unsafe_allow_html=True)
            if on:
                c = st.columns(3)
                if c[0].button("Listen", key=f"vx_l_{name}", icon=":material/mic:", width="stretch",
                               help="Vertex on that laptop starts listening, as if you said “Hey Vertex”."):
                    _vx_send("listen", name)
                if c[1].button("Unmute" if not v.get("voice") else "Mute", key=f"vx_v_{name}",
                               icon=":material/volume_up:" if not v.get("voice") else ":material/volume_off:",
                               width="stretch"):
                    _vx_send("set", name, voice=not v.get("voice"))
                if c[2].button("Announce" if not v.get("announce") else "Quiet", key=f"vx_a_{name}",
                               icon=":material/campaign:", width="stretch",
                               help="Announcements: races, evaluations, payouts, accounts blown."):
                    _vx_send("set", name, announce=not v.get("announce"))

    _vertex_hero()
    _left, _right = st.columns([1.6, 1], gap="large")
    with _left:
        st.markdown(vx.section("Ask Vertex"), unsafe_allow_html=True)
        # inside a column the box stays here (at the bottom of the page Streamlit would keep the page scrolled
        # down to it, away from the orb)
        _ask = st.chat_input("e.g. how much did we make this month?", key="vx_ask")
        if _ask:
            _to = _vx_who(vxst.vertex_live(_VX_DATA))
            if not _to:
                st.toast("Vertex isn't running on any laptop -- start Vertex Voice first.", icon=":material/mic_off:")
            else:
                _e = _vx_send("ask", _to, text=_ask)
                if _e:
                    st.session_state["_vx_wait"] = _e
        st.markdown(vx.section("Conversation"), unsafe_allow_html=True)
        _vertex_conversation()
    with _right:
        st.markdown(vx.section("Laptops"), unsafe_allow_html=True)
        _vertex_laptops()
        st.markdown(vx.section("What you can say"), unsafe_allow_html=True)
        st.markdown(vxv.examples_html(), unsafe_allow_html=True)
    st.stop()

# --- STRATEGY_LAB: the forward-test harness, not a trading book --------------
#
# Deliberately a separate view with no P&L anywhere. The lab places no orders;
# it scores 296 strategies from subsequent bars. Showing it beside account
# equity would invite reading it as money, and its whole point is that its
# numbers are NOT actionable until the sample is large enough -- so the
# headline here is sample size and detectability, never a ranking.
if bot_choice in ("STRATEGY_LAB", "LAB"):
    st.markdown(vx.CSS, unsafe_allow_html=True)
    refresh_s = _refresh_seconds(30)

    @st.fragment(run_every=refresh_s)
    def _lab_view():
        import math

        LAB_STATE = _LAB_STATE_FILE
        st.markdown(vx.section("Strategy Lab &mdash; live forward test"
                               + ((" &middot; Lucid futures (internet prices, 10 min delay)" if _PAPER else
                                   " &middot; Lucid futures (NinjaTrader)") if _FUTURES_MODE else "")),
                    unsafe_allow_html=True)
        if not LAB_STATE.exists():
            st.info("No lab state yet. Start it with:  "
                    "`.venv\\Scripts\\python.exe scripts\\run_lab.py --configs 296`")
            return
        try:
            state = json.loads(LAB_STATE.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"{LAB_STATE.name} unreadable ({exc}).")
            return

        stats = state.get("stats", {})
        total = sum(d.get("n", 0) for d in stats.values())
        openn = sum(len(d.get("open_signals", [])) for d in stats.values())
        started = state.get("started", "")
        try:
            days = max(0.01, (datetime.now(timezone.utc)
                              - datetime.fromisoformat(started)).total_seconds() / 86400)
        except Exception:
            days = 0.01

        rows = []
        for cid, d in stats.items():
            n = d.get("n", 0)
            if n < 2:
                continue
            mean = d["sum_r"] / n
            var = (d["sum_r2"] - n * mean * mean) / (n - 1)
            sd = math.sqrt(var) if var > 0 else 0.0
            if sd <= 0:
                continue
            mde = (1.96 + 0.8416) * sd / math.sqrt(n)
            rows.append({"strategy": cid, "n": n, "mean_R": round(mean, 4),
                         "MDE": round(mde, 3),
                         "win%": round(100.0 * d.get("wins", 0) / n, 1),
                         "readable": "yes" if (n >= 30 and mean > mde) else "no"})

        ready = [r for r in rows if r["n"] >= 30]
        c = st.columns(4)
        c[0].markdown(vx.card("Strategies", f"{len(stats)}",
                              f"{total / days:,.0f} trades/day", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[1].markdown(vx.card("Resolved trades", f"{total:,}",
                              f"{openn:,} open", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[2].markdown(vx.card("Readable (n≥30)", f"{len(ready)}",
                              f"of {len(rows)} with data", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[3].markdown(vx.card("Running", f"{days:.1f} d",
                              "needs months", vx.WARN if days < 60 else vx.OK,
                              small=True), unsafe_allow_html=True)

        st.markdown(
            vx.banner(
                "This is a measurement instrument, not a signal source. With 296 "
                "strategies under test, roughly 15 will look significant by pure "
                "chance &mdash; so a strategy counts only once its mean exceeds its own "
                "detectable-effect floor (MDE) AND survives Benjamini-Hochberg "
                "correction. Expect that to read zero for months. "
                "Full detail: <code>scripts/lab_report.py</code>.",
                "warn"),
            unsafe_allow_html=True)

        t_str, t_chart, t_diag, t_trades, t_open, t_fam = st.tabs(
            [":material/science: Strategies", ":material/show_chart: Charts", ":material/monitor_heart: Diagnostics",
             ":material/receipt_long: Trades", ":material/radar: Open signals", ":material/category: By family"])

        # ---------- live charts ----------
        with t_chart:
            LAB_SIGNALS = _LAB_SIGNALS_FILE
            if not LAB_SIGNALS.exists():
                st.info("No resolved trades to chart yet.")
            else:
                recs = []
                for ln in LAB_SIGNALS.read_text(encoding="utf-8").splitlines()[-8000:]:
                    try:
                        j = json.loads(ln)
                        if j.get("r") is not None:
                            recs.append(j)
                    except Exception:
                        continue
                if len(recs) < 5:
                    st.info("Not enough resolved trades to chart yet.")
                else:
                    cd = pd.DataFrame(recs)
                    cd["closed"] = pd.to_datetime(cd["resolved"], utc=True, errors="coerce")
                    cd = cd.dropna(subset=["closed"]).sort_values("closed")
                    cd["family"] = cd["cid"].str.split("|").str[0]

                    # --- equity curves, plotted against TRADE NUMBER ------------
                    #
                    # The first version plotted these against wall-clock time and
                    # was unreadable: strategies trade at wildly different rates, so
                    # a 90-trade curve and a 6-trade curve share an x-axis where one
                    # is dense and the other is three disconnected steps. Time is
                    # not the comparison anyone wants here.
                    #
                    # Against trade number every curve starts at the same origin and
                    # slope IS expectancy, so they can be read against each other
                    # directly. Only strategies with a real sample are drawn -- a
                    # 4-trade line is noise wearing the costume of a trend.
                    st.markdown(vx.section("Equity curves &mdash; cumulative R by trade"),
                                unsafe_allow_html=True)
                    MIN_TR = 15
                    counts = cd["cid"].value_counts()
                    eligible = counts[counts >= MIN_TR].index.tolist()
                    if not eligible:
                        st.info(f"No strategy has {MIN_TR}+ resolved trades yet.")
                    else:
                        # One curve per FAMILY -- the best-sampled config in each.
                        # Taking the six largest samples outright gave four
                        # near-identical stochastic variants: a chart of the same
                        # line drawn four times. Deduping by family shows real
                        # variety, and it still selects on SAMPLE SIZE rather than
                        # on result, so the picture is not cherry-picked.
                        # 6 max: the categorical order is fixed and never cycled, so a
                        # 7th would have to reuse a hue and break identity.
                        seen, top = set(), []
                        for cid in counts[counts >= MIN_TR].index:
                            fam = cid.split("|")[0]
                            if fam in seen:
                                continue
                            seen.add(fam)
                            top.append(cid)
                            if len(top) == 6:
                                break
                        curve = go.Figure()
                        for i, cid in enumerate(top):
                            g = cd[cd["cid"] == cid].sort_values("closed")
                            y = g["r"].cumsum().values
                            nm = "  ".join(cid.split("|")[:1] + cid.split("|")[2:])
                            curve.add_trace(go.Scatter(
                                x=list(range(1, len(y) + 1)), y=y, mode="lines",
                                name=nm, line=dict(color=vx.SERIES[i], width=2,
                                                   shape="spline", smoothing=0.4),
                                hovertemplate=("trade %{x}<br>%{y:+.2f}R cumulative"
                                               "<extra>" + nm + "</extra>")))
                            # direct end-label: identity without hunting the legend
                            curve.add_trace(go.Scatter(
                                x=[len(y)], y=[y[-1]], mode="markers",
                                marker=dict(color=vx.SERIES[i], size=8,
                                            line=dict(color=vx.SURFACE, width=2)),
                                showlegend=False, hoverinfo="skip"))
                        curve.add_hline(y=0, line=dict(color=vx.TEXT_FAINT, width=1))
                        curve.update_layout(**vx.plotly_layout(
                            height=380, showlegend=True, hovermode="x unified",
                            legend=dict(orientation="h", y=-0.16, x=0,
                                        font=dict(size=11, color=vx.TEXT_MUTED))))
                        curve.update_xaxes(title_text="trade number")
                        curve.update_yaxes(title_text="cumulative R")
                        st.plotly_chart(curve, width="stretch",
                                        config={"displayModeBar": False})
                        st.caption(
                            f"Six largest samples with at least {MIN_TR} trades. X is trade "
                            "number, not time, so curves are comparable regardless of how "
                            "often a strategy fires — slope is expectancy. Cumulative R, "
                            "not money: the lab places no orders.")

                    n_show = st.slider("Strategies shown", 10, 40, 20, 5,
                                       key="lab_dot_n",
                                       help="Ranked by sample size, not by result.")
                    est = []
                    for cid, dd_ in stats.items():
                        n = dd_.get("n", 0)
                        if n < 3:
                            continue
                        mean = dd_["sum_r"] / n
                        var = (dd_["sum_r2"] - n * mean * mean) / (n - 1)
                        if var <= 0:
                            continue
                        se = math.sqrt(var) / math.sqrt(n)
                        est.append({"cid": cid, "n": n, "mean": mean, "se": se,
                                    "lo": mean - 1.96 * se, "hi": mean + 1.96 * se})
                    if not est:
                        st.info("No strategy has 3+ resolved trades yet.")
                    else:
                        ed = (pd.DataFrame(est).sort_values("n", ascending=False)
                              .head(n_show).sort_values("mean"))

                        def _verdict(r):
                            if r["lo"] > 0:
                                return "above zero"
                            if r["hi"] < 0:
                                return "below zero"
                            return "indistinguishable"
                        ed["verdict"] = ed.apply(_verdict, axis=1)
                        cmap = {"above zero": vx.POS, "below zero": vx.NEG,
                                "indistinguishable": vx.NEUTRAL}
                        # short, aligned labels -- the full id is in the hover
                        ed["label"] = ed["cid"].map(
                            lambda c: "  ".join(c.split("|")[:1] + c.split("|")[2:]))

                        st.markdown(
                            vx.section("Mean R per strategy, with 95% interval"),
                            unsafe_allow_html=True)
                        fig = go.Figure()
                        for v in ("indistinguishable", "above zero", "below zero"):
                            g = ed[ed["verdict"] == v]
                            if g.empty:
                                continue
                            fig.add_trace(go.Scatter(
                                x=g["mean"], y=g["label"], mode="markers", name=v,
                                marker=dict(color=cmap[v], size=9,
                                            line=dict(color=vx.SURFACE, width=2)),
                                error_x=dict(type="data", symmetric=False,
                                             array=g["hi"] - g["mean"],
                                             arrayminus=g["mean"] - g["lo"],
                                             color=cmap[v], thickness=2, width=0),
                                customdata=g[["n", "lo", "hi"]].values,
                                hovertemplate=("%{y}<br>mean %{x:+.3f}R<br>"
                                               "95%% CI %{customdata[1]:+.3f} to "
                                               "%{customdata[2]:+.3f}<br>"
                                               "n = %{customdata[0]}<extra></extra>")))
                        fig.add_vline(x=0, line=dict(color=vx.TEXT_FAINT, width=1))
                        fig.update_layout(**vx.plotly_layout(
                            height=max(300, 24 * len(ed) + 90), showlegend=True,
                            legend=dict(orientation="h", y=-0.09, x=0,
                                        font=dict(size=11, color=vx.TEXT_MUTED))))
                        fig.update_xaxes(title_text="mean R per trade")
                        fig.update_yaxes(tickfont=dict(size=10, family="JetBrains Mono, monospace"))
                        st.plotly_chart(fig, width="stretch",
                                        config={"displayModeBar": False})
                        nsig = int((ed["verdict"] != "indistinguishable").sum())
                        st.caption(
                            f"{len(ed) - nsig} of {len(ed)} intervals span zero — those "
                            "strategies are not yet distinguishable from random, whatever "
                            "their dot position. Bars are 95% intervals on the mean, "
                            "uncorrected for testing 296 strategies at once; "
                            "lab_report.py applies that correction.")

                    # --- sample accumulation (single series -> no legend) ------
                    st.markdown(vx.section("Sample accumulation"), unsafe_allow_html=True)
                    per = cd.set_index("closed")["r"].resample("15min").count().cumsum()
                    f2 = go.Figure(go.Scatter(
                        x=per.index, y=per.values, mode="lines",
                        line=dict(color=vx.ACCENT, width=2),
                        fill="tozeroy", fillcolor="rgba(76,110,245,0.12)",
                        hovertemplate="%{y:,} trades<extra></extra>"))
                    f2.update_layout(**vx.plotly_layout(height=210, hovermode="x"))
                    f2.update_yaxes(title_text="resolved trades")
                    st.plotly_chart(f2, width="stretch",
                                    config={"displayModeBar": False})

                    # --- R distribution: polarity -> diverging pair ------------
                    st.markdown(vx.section("Outcome distribution"), unsafe_allow_html=True)
                    f3 = go.Figure()
                    wins, losses = cd[cd["r"] > 0]["r"], cd[cd["r"] <= 0]["r"]
                    f3.add_trace(go.Histogram(x=losses, name="losses", nbinsx=40,
                                              marker=dict(color=vx.NEG)))
                    f3.add_trace(go.Histogram(x=wins, name="wins", nbinsx=40,
                                              marker=dict(color=vx.POS)))
                    f3.update_layout(**vx.plotly_layout(
                        height=230, showlegend=True, barmode="overlay",
                        legend=dict(orientation="h", y=-0.22, x=0,
                                    font=dict(size=11, color=vx.TEXT_MUTED))))
                    f3.update_traces(opacity=0.8)
                    f3.update_xaxes(title_text="R per trade")
                    st.plotly_chart(f3, width="stretch",
                                    config={"displayModeBar": False})

        # ---------- diagnostics: which strategies are even alive ----------
        with t_diag:
            st.markdown(vx.section("Signal production &mdash; is this strategy alive?"),
                        unsafe_allow_html=True)
            diag = []
            for cid, d in stats.items():
                n = d.get("n", 0)
                op = len(d.get("open_signals", []))
                total_sig = n + op
                rate = total_sig / days if days else 0.0
                if total_sig == 0:
                    status = "no signal yet"
                elif rate < 0.5:
                    status = "very rare"
                else:
                    status = "active"
                diag.append({"strategy": cid, "family": cid.split("|")[0],
                             "timeframe": cid.split("|")[2] if cid.count("|") >= 2 else "",
                             "resolved": n, "open": op,
                             "signals/day": round(rate, 2), "status": status})
            dd = pd.DataFrame(diag)
            k = st.columns(3)
            k[0].markdown(vx.card("Active", f"{(dd.status == 'active').sum()}",
                                  "≥0.5 signals/day", vx.OK, small=True),
                          unsafe_allow_html=True)
            k[1].markdown(vx.card("Very rare", f"{(dd.status == 'very rare').sum()}",
                                  "<0.5/day", vx.WARN, small=True),
                          unsafe_allow_html=True)
            k[2].markdown(vx.card("No signal yet", f"{(dd.status == 'no signal yet').sum()}",
                                  f"after {days:.1f} days", vx.TEXT_MUTED, small=True),
                          unsafe_allow_html=True)
            st.markdown(
                vx.banner(
                    f"<b>Do not prune on this yet.</b> The lab has run {days:.1f} days. "
                    "Session strategies (ORB, London) fire in one window per day, and "
                    "1h strategies see 24 bars a day &mdash; a strategy showing no signal "
                    "here has most likely just not had its window since the lab started. "
                    "A strategy is only genuinely dead after several full days covering "
                    "every session; <code>scripts/lab_prune.py</code> enforces that.",
                    "warn"),
                unsafe_allow_html=True)
            st.dataframe(dd.sort_values(["signals/day", "resolved"],
                                        ascending=[False, False]),
                         width="stretch", hide_index=True)

        with t_str:
            if not rows:
                st.info("No strategy has resolved 2+ trades yet.")
            else:
                df = pd.DataFrame(rows).sort_values(["n", "mean_R"],
                                                    ascending=[False, False])
                st.markdown(vx.section("By sample size &mdash; not a ranking"),
                            unsafe_allow_html=True)
                st.dataframe(df.head(60), width="stretch", hide_index=True)

        # ---------- every resolved trade ----------
        with t_trades:
            LAB_SIGNALS = _LAB_SIGNALS_FILE
            if not LAB_SIGNALS.exists():
                st.info("No trades resolved yet.")
            else:
                recs = []
                # Tail only: this file grows unboundedly over months of running,
                # and the dashboard re-reads it on every refresh.
                lines = LAB_SIGNALS.read_text(encoding="utf-8").splitlines()
                for ln in lines[-4000:]:
                    try:
                        recs.append(json.loads(ln))
                    except Exception:
                        continue
                if not recs:
                    st.info("No trades resolved yet.")
                else:
                    td = pd.DataFrame(recs)
                    td["opened"] = pd.to_datetime(td["ts"], utc=True, errors="coerce")
                    td["closed"] = pd.to_datetime(td["resolved"], utc=True, errors="coerce")
                    td["held"] = (td["closed"] - td["opened"]).dt.total_seconds() / 60.0
                    td["outcome"] = td["r"].map(lambda x: "win" if x > 0 else "loss")

                    f = st.columns(3)
                    syms = ["(all)"] + sorted(td["sym"].dropna().unique().tolist())
                    fams = ["(all)"] + sorted({c.split("|")[0] for c in td["cid"]})
                    pick_s = f[0].selectbox("Symbol", syms, key="lab_f_sym")
                    pick_f = f[1].selectbox("Family", fams, key="lab_f_fam")
                    pick_o = f[2].selectbox("Outcome", ["(all)", "win", "loss"],
                                            key="lab_f_out")
                    v = td
                    if pick_s != "(all)":
                        v = v[v["sym"] == pick_s]
                    if pick_f != "(all)":
                        v = v[v["cid"].str.startswith(pick_f + "|")]
                    if pick_o != "(all)":
                        v = v[v["outcome"] == pick_o]

                    k = st.columns(4)
                    k[0].markdown(vx.card("Trades shown", f"{len(v):,}",
                                          f"of {len(td):,} loaded", vx.TEXT_MUTED),
                                  unsafe_allow_html=True)
                    wr = 100.0 * (v["r"] > 0).mean() if len(v) else 0.0
                    k[1].markdown(vx.card("Win rate", f"{wr:.1f}%", "of shown",
                                          vx.TEXT_MUTED), unsafe_allow_html=True)
                    mr = v["r"].mean() if len(v) else 0.0
                    k[2].markdown(vx.card("Mean R", f"{mr:+.4f}",
                                          "not significance", vx.TEXT_MUTED),
                                  unsafe_allow_html=True)
                    k[3].markdown(vx.card("Median hold", f"{v['held'].median():.0f} min"
                                          if len(v) else "—", "signal to resolve",
                                          vx.TEXT_MUTED), unsafe_allow_html=True)

                    show = v.sort_values("closed", ascending=False)[
                        ["closed", "cid", "sym", "dir", "r", "held"]].head(500)
                    show = show.rename(columns={"closed": "resolved (UTC)",
                                                "cid": "strategy", "sym": "symbol",
                                                "dir": "side", "r": "R",
                                                "held": "held (min)"})
                    st.dataframe(
                        show.style.map(
                            lambda x: f"color:{vx.POS}" if isinstance(x, (int, float)) and x > 0
                            else (f"color:{vx.NEG}" if isinstance(x, (int, float)) and x < 0 else ""),
                            subset=["R"]),
                        width="stretch", hide_index=True)
                    st.caption("Newest 500 of the filtered set. Mean R here is "
                               "descriptive only — see the Strategies tab for whether "
                               "any of it clears its own detectability floor.")

        # ---------- signals still running ----------
        with t_open:
            op = []
            for cid, d in stats.items():
                for s in d.get("open_signals", []):
                    op.append({"strategy": cid, "symbol": s.get("sym"),
                               "side": s.get("dir"), "opened (UTC)": s.get("ts"),
                               "entry": s.get("entry"), "SL": s.get("sl"),
                               "TP": s.get("tp")})
            if not op:
                st.info("No open signals right now.")
            else:
                od = pd.DataFrame(op).sort_values("opened (UTC)", ascending=False)
                st.markdown(vx.section(f"{len(od):,} signals awaiting resolution"),
                            unsafe_allow_html=True)
                st.dataframe(od.head(500), width="stretch", hide_index=True)

        # ---------- family rollup ----------
        with t_fam:
            if not rows:
                st.info("Not enough data yet.")
            else:
                fr = pd.DataFrame(rows)
                fr["family"] = fr["strategy"].str.split("|").str[0]
                g = fr.groupby("family").agg(
                    strategies=("strategy", "count"),
                    trades=("n", "sum"),
                    mean_R=("mean_R", "mean"),
                    best_R=("mean_R", "max"),
                    readable=("readable", lambda s: (s == "yes").sum()),
                ).reset_index().sort_values("trades", ascending=False)
                st.markdown(vx.section("Rollup by strategy family"),
                            unsafe_allow_html=True)
                st.dataframe(g, width="stretch", hide_index=True)
                st.caption("Pooling families hides within-family variation and is for "
                           "orientation only — a family mean says nothing about whether "
                           "any single configuration in it has an edge.")

    _lab_view()
    st.stop()

from metatraderbot.nt_engine import NinjaTraderEngine as _NTEngine  # noqa: E402
def _engine(account: str, port: int, timeout_s: float):
    """What the bots trade through: NinjaTrader, or in paper mode their paper book (read-only here)."""
    if _PAPER:
        return _fresh_prices()[1].PaperEngine(account=account, read_only=True)       # internet prices
    return _NTEngine(account=account, port=port, timeout_s=timeout_s)

# ---------- trade charts for the race pages (TRIAL and FUNDED) ----------
# One price line, the levels that decide the trade drawn as labelled
# horizontal lines (target = status good, MLL line = status critical, buy
# price = muted ink), buy/sell markers. Labels wear text colours; the line
# colours only repeat what the labels say.
_CLOSE_REASON = {
    "tp": "target reached", "pass_while_away": "target reached while the laptop slept",
    "fail_while_away": "loss limit hit while the laptop slept", "sl": "safety stop",
    "fail_rescored": "loss limit hit (re-scored)",
    "exit_while_away": "16:00 New York close (laptop asleep)", "closed_by_bot": "closed by the bot",
    "manual": "closed by the bot", "external": "closed outside the bot", "gap": "closed at market",
    "protection_lost": "closed at market", "time_exit": "16:00 New York close",
    "line": "loss limit hit",
}

def _bars_around(sym: str, start_utc, end_utc, account: str, port: int):
    """Bars covering one trade -- 1-minute for recent trades, 5/15-minute for
    older ones (the bridge serves the last N bars up to now)."""
    age = pd.Timestamp.now(tz="UTC") - start_utc
    minutes = 1 if age <= pd.Timedelta(days=2) else (5 if age <= pd.Timedelta(days=10) else 15)
    count = int(age.total_seconds() // 60 // minutes) + 40
    tf = {1: "1m", 5: "5m", 15: "15m"}[minutes]
    try:
        if _PAPER:
            raise LookupError("paper mode: internet prices")
        b = _NTEngine(account=account, port=port, timeout_s=15.0).get_bars(sym, tf, max(60, min(count, 8000)))
    except Exception:  # noqa: BLE001 -- paper mode / NinjaTrader closed: the internet's bars
        b = _fresh_prices()[0].bars(tf, max(60, min(count, 8000)))
    pad = pd.Timedelta(minutes=20)
    return b[(b.index >= start_utc - pad) & (b.index <= end_utc + pad)], minutes

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _closed_trade_bars(sym: str, start_iso: str, end_iso: str, account: str, port: int):
    """A closed trade's bars never change -- fetched once, then cached. An answer without the trade's own
    minutes is not kept (2026-10-16: the internet's minutes missed the evening, two charts stayed blank)."""
    b, minutes = _bars_around(sym, pd.Timestamp(start_iso), pd.Timestamp(end_iso), account, port)
    inside = b[(b.index >= pd.Timestamp(start_iso)) & (b.index <= pd.Timestamp(end_iso))] if len(b) else b
    if not len(inside):
        raise LookupError("no prices for this trade's time yet")
    return b, minutes

def _trade_figure(b, minutes, tz, *, entry_t, entry_px, tp=None, line=None, exit_t=None, exit_px=None,
                  exit_text=None, exit_color=None, live_t=None, live_px=None, close_at=None,
                  height=300, title=None, tp_label="Target", line_label="Fail line", long=True):
    """long=False: a short race -- 'Sold short' at the entry, 'Bought back' at the exit."""
    in_word, out_word = ("Bought", "Sold") if long else ("Sold short", "Bought back")
    if exit_text and not long and exit_text.startswith("Sold "):
        exit_text = "Bought back " + exit_text[len("Sold "):]
    x = list((b.index + pd.Timedelta(minutes=minutes)).tz_convert(tz))     # bar CLOSE times
    y = [float(v) for v in b["close"]]
    if live_t is not None and live_px is not None:
        x.append(live_t)
        y.append(float(live_px))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="NQ price", line=dict(color=vx.POS, width=2),
                             hovertemplate="%{y:,.2f}<extra>NQ</extra>"))
    levels = [v for v in (tp, line, entry_px, exit_px, live_px) if v is not None]
    lo, hi = min(y + levels), max(y + levels)
    pad = max((hi - lo) * 0.10, 3.0)
    for val, col, dash, text in (
            (tp, vx.OK, "dot", f"{tp_label}<br>{tp:,.2f}" if tp else ""),
            (line, vx.DANGER, "dot", f"{line_label}<br>{line:,.2f}" if line else ""),
            (entry_px, vx.TEXT_MUTED, "dash", f"{in_word}<br>{entry_px:,.2f}")):
        if val is not None:          # label in the right margin, never on top of the price line
            fig.add_hline(y=val, line=dict(color=col, width=1.5, dash=dash))
            fig.add_annotation(x=1.0, xref="paper", xanchor="left", xshift=6, y=val, text=text,
                               showarrow=False, align="left", font=dict(color=vx.TEXT_MUTED, size=10.5))
    fig.add_trace(go.Scatter(x=[entry_t], y=[entry_px], mode="markers", name="Buy" if long else "Sell",
                             marker=dict(size=10, color=vx.TEXT, line=dict(color=vx.SURFACE, width=2)),
                             hovertemplate=in_word + " %{y:,.2f}<extra></extra>"))
    if exit_t is not None and exit_px is not None:
        fig.add_trace(go.Scatter(x=[exit_t], y=[exit_px], mode="markers+text", name="Sell" if long else "Buy",
                                 text=[exit_text or ""], textposition="top left",
                                 textfont=dict(color=vx.TEXT, size=11),
                                 marker=dict(size=11, color=exit_color or vx.TEXT,
                                             line=dict(color=vx.SURFACE, width=2)),
                                 hovertemplate=out_word + " %{y:,.2f}<extra></extra>"))
    if live_t is not None and live_px is not None:
        fig.add_trace(go.Scatter(x=[live_t], y=[live_px], mode="markers", name="Now",
                                 marker=dict(size=9, color=vx.POS, line=dict(color=vx.SURFACE, width=2)),
                                 hovertemplate="Now %{y:,.2f}<extra></extra>"))
    if close_at is not None:
        fig.add_shape(type="line", x0=close_at, x1=close_at, y0=0, y1=1, yref="paper",
                      line=dict(color=vx.TEXT_FAINT, width=1, dash="dash"))
        fig.add_annotation(x=close_at, y=1, yref="paper", text=f"auto-close {close_at:%H:%M}",
                           showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(color=vx.TEXT_MUTED, size=11))
        fig.update_xaxes(range=[x[0] if x else entry_t, close_at + pd.Timedelta(minutes=10)])
    lay = vx.plotly_layout(height=height, title=title)
    lay["margin"] = dict(lay.get("margin", {}), r=78)          # room for the level labels
    fig.update_layout(**lay, hovermode="x unified")
    fig.update_yaxes(range=[lo - pad, hi + pad], tickformat=",.0f")
    return fig


def _race_words(sg: float, qty: int, sym: str, entry: float, target: float, cur: float) -> str:
    """The open race in plain words, long (sg=+1) or short (sg=-1)."""
    gap = sg * (cur - entry)
    behind = (f" (now {-gap:,.2f} points {'below' if sg > 0 else 'above'} the entry price)" if gap < 0 else "")
    if sg > 0:
        return (f"Bought <b>{qty} {sym}</b> at <b>{entry:,.2f}</b> (betting on a rise). The bot sells automatically"
                f" at <b>{target:,.2f}</b> (+{target - entry:,.2f} points). Price now: <b>{cur:,.2f}</b>{behind}")
    return (f"Sold short <b>{qty} {sym}</b> at <b>{entry:,.2f}</b> (betting on a fall). The bot buys back"
            f" automatically at <b>{target:,.2f}</b> ({target - entry:,.2f} points). Price now:"
            f" <b>{cur:,.2f}</b>{behind}")


def _side_reason(info: dict) -> str:
    why = str(info.get("side_reason") or "")
    return f"<br>Why this direction: {why.split(': ', 1)[-1]}." if why else ""


_TREND_RULE_FROM = "2026-10-07"      # trading day the races began to follow the 200-day trend (always long before)


def _race_story(t: dict, prev: dict | None, rec: dict | None, local_tz, *, pv: float = 20.0,
                target_usd: float = 3_000.0, live: bool = False) -> list[tuple[str, str]]:
    """A trial race in plain words: why it started, why that direction, what it aimed for, how it ended.
    t: a trade-log row (or the open race's ticket, live=True); prev: the trade before it; rec: the
    brain's record of it (best / worst moves), if it has one yet."""
    from metatraderbot import lucid_rules as _lr_
    o = pd.Timestamp(t["opened_ts"])
    o_ny = o.tz_convert(_lr_.ET)

    def both(ts: pd.Timestamp) -> str:
        return f"{ts.tz_convert(_lr_.ET):%a %H:%M} New York ({ts.tz_convert(local_tz):%a %H:%M} your time)"

    n = t.get("attempt")
    short = t.get("direction") == "short"
    lots = float(t.get("lots") or 0)
    entry, tp, line = float(t["entry_price"]), t.get("tp"), t.get("line")
    out = []
    # why it started
    gap = (o - pd.Timestamp(prev["closed_ts"])).total_seconds() / 60 if prev and prev.get("closed_ts") else None
    if gap is not None and 0 <= gap <= 30:
        p_pnl = float(prev.get("pnl_usd") or 0)
        out.append(("Why it started", f"Evaluation #{prev.get('attempt')} had just ended ({'passed' if p_pnl > 0 else 'failed'}"
                                      f" at {both(pd.Timestamp(prev['closed_ts']))}), and on the simulator a new"
                                      " evaluation races straight away &mdash; your rule: keep going until one passes."))
    elif o_ny.hour == 19 and o_ny.minute < 15:
        out.append(("Why it started", "The day's planned race: every trading day at 19:00 New York (01:00 your time)."))
    else:
        out.append(("Why it started", f"It went in at {both(o)} instead of 19:00 New York: at 19:00 the bot, the"
                                      " laptop or NinjaTrader was not running or not connected, so it raced as soon as"
                                      " it could (entries are allowed until 12:00 New York)."))
    # why this direction
    why_side = str(t.get("side_reason") or "")
    if why_side:
        detail = why_side.split(": ", 1)[-1]
        out.append(("Why this direction", f"{'Sold short' if short else 'Bought'} because {detail} &mdash; the trend"
                                          " rule: buy when NQ is above its 200-day average, sell short below it."))
    elif short:
        out.append(("Why this direction", "Sold short because NQ was below its 200-day average (the trend rule)."))
    elif _lr_.trading_day(o.to_pydatetime()) >= _TREND_RULE_FROM:
        out.append(("Why this direction", "Bought because NQ was above its 200-day average (the trend rule: buy"
                                          " above it, sell short below it)."))
    else:
        out.append(("Why this direction", "Bought &mdash; before 7 Oct every race bought (the trend rule came later)."))
    # what it aimed for
    aim = f"{lots:g} NQ {'short' if short else 'long'} at {entry:,.2f}"
    if tp:
        tp_pts = abs(float(tp) - entry)
        aim += (f" &middot; target {float(tp):,.2f} ({tp_pts:,.2f} points, about +${tp_pts * lots * pv:,.0f} &mdash; what"
                f" the evaluation{' #' + str(n) if n else ''} still needed to reach +${target_usd:,.0f})")
    if line:
        ln_pts = abs(entry - float(line))
        aim += (f" &middot; loss line {float(line):,.2f} ({ln_pts:,.2f} points, about −${ln_pts * lots * pv:,.0f}: there"
                " the account is $2,000 below its best close and the evaluation fails)")
    out.append(("What it aimed for", aim + "."))
    if live:
        out.append(("How it can end", "At the target the evaluation passes; at the loss line it fails; if neither is"
                                      " reached, the bot closes the trade at 16:00 New York (22:00 your time) and the"
                                      " evaluation goes on with the next race."))
        return out
    # how it ended
    c = pd.Timestamp(t["closed_ts"])
    pnl, ex = float(t.get("pnl_usd") or 0), float(t["exit_price"])
    reason = str(t.get("reason") or "")
    sg = -1.0 if short else 1.0
    at_line = bool(line) and sg * (ex - float(line)) <= 0.5
    at_tp = bool(tp) and sg * (ex - float(tp)) >= -0.5
    moved = "fell" if not short else "rose"
    usd_ = f"{'+' if pnl >= 0 else '−'}${abs(pnl):,.0f}"
    if reason == "pass_while_away":
        end = (f"Price reached the target at {both(c)} while the laptop was asleep; on waking the bot booked it from"
               f" the price history, as Lucid's server saw it ({usd_}) &rarr; evaluation passed.")
    elif reason in ("tp",) or (at_tp and pnl > 0):
        end = f"Price reached the target at {both(c)} ({usd_}) &rarr; evaluation passed."
    elif reason == "fail_while_away":
        end = (f"Price {moved} to the loss line at {both(c)} while the laptop was asleep; booked from the price"
               f" history ({usd_}) &rarr; evaluation failed.")
    elif reason == "fail_rescored":
        end = ("NinjaTrader filled the target later, but the price history shows the loss line was touched first, at"
               f" {both(c)} &mdash; Lucid would have failed the account then ({usd_}) &rarr; counted as failed.")
    elif reason in ("line", "closed_by_bot", "manual") and at_line:
        end = (f"Price {moved} to the loss line at {both(c)}: the open loss reached $2,000 and the bot closed the"
               f" trade ({usd_}) &rarr; evaluation failed.")
    elif reason in ("time_exit", "exit_while_away"):
        end = (f"Neither the target nor the loss line by 16:00 New York; closed at {ex:,.2f} ({usd_}). The"
               " evaluation went on with the next race.")
    else:
        end = f"Closed by the bot at {ex:,.2f} at {both(c)} ({usd_})."
    if rec and rec.get("best_pts") is not None:
        end += (f" On the way: at best {float(rec['best_pts']):+,.2f} points, at worst −{abs(float(rec['worst_pts'])):,.2f}"
                " points.")
    out.append(("How it ended", end))
    return out


def _story_html(rows: list[tuple[str, str]]) -> str:
    return ('<div style="font-size:13.5px;line-height:1.6;color:%s;margin:-4px 0 14px">' % vx.TEXT_MUTED
            + "".join(f'<div><b style="color:{vx.TEXT}">{k}:</b> {v}</div>' for k, v in rows) + "</div>")


def _funded_path(balance: float, peak_eod: float, cycle: float, best: float, cycle_days: int,
                 n_paid: int, max_steps: int = 12) -> dict:
    """The trading days to the next payout if every race reaches its take-profit, by the bot's own rules
    (take = what the payout still needs, $150-$1,500; the payout at the close of the cycle's 3rd day).
    A race's rough chance is its distance to the loss line / (that + its distance to the take-profit):
    what a market with no direction gives. `cycle_days` counts the days already in the cycle, today
    included when no race is left in it."""
    k = min(n_paid, len(_F_WANT) - 1)
    bal, peak, cyc, bst, days = balance, peak_eod, cycle, best, cycle_days
    steps, chance = [], 1.0
    for _ in range(max_steps):
        need = _funded_need(bal, cyc, bst, n_paid)
        if need <= 0 and days >= 3:
            break
        floor = min(peak - _F_MLL, _F_LOCK)
        days += 1
        if need <= 0:                                   # money there: no race until the cycle has 3 days
            steps.append({"race": False, "take": 0.0, "balance": bal, "line": floor, "p": 1.0, "chance": chance})
            continue
        take = min(max(need, _F_TAKE_MIN), _F_TAKE_MAX)
        room = bal - floor
        p = room / (room + take) if room > 0 else 0.0
        chance *= p
        bal += take
        peak = max(peak, bal)
        cyc += take
        bst = max(bst, take)
        steps.append({"race": True, "take": take, "balance": bal, "line": min(peak - _F_MLL, _F_LOCK),
                      "p": p, "chance": chance})
    return {"steps": steps, "payout_no": n_paid + 1, "amount": max(0.0, min(_F_CAPS[k], bal - _F_KEEP)),
            "chance": chance}


def _funded_curve(trades: list[dict], payouts: list[dict], start_iso: str | None = None) -> pd.DataFrame:
    """One funded account's balance and loss line after every trade and payout. One race a day, so the
    balance after a trade is that day's close -- the one that moves the loss line."""
    ev = [(pd.Timestamp(t["closed_ts"]), float(t.get("pnl_usd") or 0.0), "trade") for t in trades if t.get("closed_ts")]
    ev += [(pd.Timestamp(p["ts"]), -float(p.get("amount_usd") or 0.0), "payout") for p in payouts if p.get("ts")]
    ev.sort(key=lambda e: e[0])
    bal = peak = _F_ANCHOR
    rows = []
    if start_iso:
        rows.append({"t": pd.Timestamp(start_iso), "balance": bal, "line": peak - _F_MLL, "what": "start", "amount": 0.0})
    for t, x, what in ev:
        bal += x
        if what == "trade":
            peak = max(peak, bal)
        rows.append({"t": t, "balance": bal, "line": min(peak - _F_MLL, _F_LOCK), "what": what, "amount": x})
    return pd.DataFrame(rows, columns=["t", "balance", "line", "what", "amount"])


def _nt_down_card() -> tuple[str, str, str]:
    """NinjaTrader unreachable: closed, or open but stuck at its login (2026-10-04/07: its Google login
    failed and it sat at the login screen for days while the bots waited)."""
    import subprocess
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq NinjaTrader.exe", "/NH"], capture_output=True,
                             text=True, timeout=5).stdout
        running = "NinjaTrader.exe" in out
    except Exception:  # noqa: BLE001
        running = False
    if running:
        return ("🔑", "NinjaTrader is open but not logged in",
                "Open the NinjaTrader window and log in (it signs in with your Google account and needs the "
                "internet). Until then the bots cannot trade. The numbers below are from the last saved state.")
    return ("⚠️", "NinjaTrader is not running",
            "The watchdog starts it within 15 minutes, or click the Vertex icon on your desktop. "
            "The numbers below are from the last saved state.")


# ---------- the trade brain's memory (scripts/trade_brain.py -> data/brain.json) ----------
@st.cache_data(show_spinner=False)
def _brain_memory(_mtime: float) -> dict:
    p = ROOT / "data" / "brain.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def _brain() -> dict:
    p = ROOT / "data" / "brain.json"
    return _brain_memory(p.stat().st_mtime if p.exists() else 0.0)


def _brain_caption(bot: str, t: dict) -> None:
    """The brain's one-line verdict under a trade chart."""
    rec = (_brain().get("trades") or {}).get(f"{bot}|{t.get('opened_ts')}")
    if rec and rec.get("verdict"):
        st.caption("🧠 " + rec["verdict"])


_PREDICTOR_NAMES = {   # scripts/run_prediction_lab.PREDICTORS
    "deployed": ("Deployed race", "buy every day at 19:00 New York (the benchmark; the bots follow Long-term trend)"),
    "first_hour": ("Follow the first hour", "at 10:30 New York, race the way NQ moved since 09:30"),
    "long_1030": ("Buy at 10:30", "buy every day at 10:30 New York (comparison for the line above)"),
    "events": ("Big-release days", "buy at 19:00 only before a Fed, CPI or jobs-report day"),
    "vwap_high": ("Above yesterday's VWAP", "buy only when price is high vs yesterday's average traded price"),
    "evening_buying": ("Evening buying", "buy only when 18:00-19:00 buying pressure is in its top third"),
    "treasury_buying": ("Treasury buying", "buy only when yesterday's Treasury buying was in its top third"),
    "timesfm_15m": ("TimesFM 15-min", "race up or down as Google's TimesFM forecasts 19:00->16:00"),
    "timesfm_daily": ("TimesFM daily", "race up or down as TimesFM forecasts the next daily close"),
    "breadth": ("Market breadth", "race up or down the way NYSE stocks leaned yesterday"),
    "trend200": ("Long-term trend", "race up above the 200-day average, down below it"),
    "skew_jump": ("SKEW jump", "buy only after Cboe's crash-protection price (SKEW) jumped the day before"),
}


# BRAIN: what every trade taught -- the Learning and Trade ledger notes, plus the prediction lab.
# FLEET (futures mode): many LucidPro account slots at once -- evaluation, funded account, payouts --
# simulated from real NQ minute prices (src/metatraderbot/lucid_fleet.py). No orders.
def _fleet_mod():
    """lucid_fleet, reloaded whenever its file changes. Streamlit reloads only the modules beside this
    script, so a running dashboard kept an older simulator (2026-10-08: "got an unexpected keyword
    argument 'start_slots'")."""
    import importlib
    from metatraderbot import lucid_fleet as _lf
    mtime = Path(_lf.__file__).stat().st_mtime
    if getattr(_lf, "_loaded_mtime", None) != mtime:
        _lf = importlib.reload(_lf)
        _lf._loaded_mtime = mtime
    return _lf


@st.cache_data(ttl=300, show_spinner=False)
def _fleet_minutes(since: str) -> pd.DataFrame:
    _lf = _fleet_mod()
    try:
        if _PAPER:
            raise LookupError("paper mode: internet prices")
        eng = _NTEngine(account=(_ENV.get("NT_ACCOUNT") or "Sim101").strip(),
                        port=int(_ENV.get("NT_BRIDGE_PORT") or 45973), timeout_s=30.0)
        return _lf.load_minutes(ROOT / "data" / "futures", eng, since=since)
    except Exception:  # noqa: BLE001 -- NinjaTrader closed: the stored bars + the internet's newest
        try:
            return _lf.load_minutes(ROOT / "data" / "futures", _fresh_prices()[0].FreeFeed(), since=since)
        except Exception:  # noqa: BLE001
            return _lf.load_minutes(ROOT / "data" / "futures", None, since=since)


@st.cache_data(ttl=300, show_spinner=False)
def _fleet_run(since: str, last_bar: str, slots: int, stagger: int, eval_fee: float, reset_fee: float,
               start_slots: int, step_usd: float, version: float = 0.0, step_on: str = "money",
               bots: str = "", stop_usd: float = 10_000.0, strategy: str = "") -> dict:
    _lf = _fleet_mod()            # `version` (the simulator file's time) makes a new simulator recompute
    # bots: the Real test races when and the way the paper bots raced (owner 2026-10-19: "we passed trial ... but
    # it doesn't show on the fleet page"); the value is the bots' ledgers' times, so a new race recomputes
    return _lf.run(_fleet_minutes(since), _lf.FleetConfig(
        slots=slots, start=since, stagger=stagger, eval_fee=eval_fee, reset_fee=reset_fee,
        start_slots=start_slots if start_slots < slots else None, step_usd=step_usd, step_on=step_on,
        bot_races=_lf.bot_races(ROOT / "data", since, *(_AD_TAGS if strategy == "ad" else ("", ""))) if bots else None,
        stop_usd=stop_usd or None))   # stop_usd: the owner's stop rule (2026-10-21), 0 = none


if bot_choice == "FLEET":
    st.markdown(vx.CSS, unsafe_allow_html=True)

    def _real_test_view(_lf, res, since, start_slots, slots, step_usd, eval_fee, reset_fee, stop_usd=10_000.0,
                        tags=("", ""), label="", same_days=""):
        """The real test in plain words: what your plan did on real prices since the bots started -- the money,
        where each account stands now, and every day's event as a sentence."""
        usd = lambda x: f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"  # noqa: E731
        sl, ev, daily = res["slots"], res["events"], res["daily"]
        plan50 = _lf.PLANS["50K"]
        fees = sum(s["fees"] for s in sl)
        to_you = 0.9 * sum(s["paid"] for s in sl)
        net = sum(s["net"] for s in sl)
        bought = sum(s["evals"] for s in sl)
        attempts = sum(s["passes"] + s["resets"] for s in sl) + sum(1 for s in sl if s["stage"] == "eval")
        passes = sum(s["passes"] for s in sl)
        blown = sum(s["blown"] for s in sl)
        payouts = sum(s["payouts"] for s in sl)
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Real test{" &middot; " + label if label else ""} &middot; your plan on real'
            f' prices since {pd.Timestamp(since):%a %d %b}</div>'
            f'<div class="vx-sub">One account to start, one more each time the money made reaches another '
            f'+${step_usd:,.0f} (up to {slots}) &middot; evaluation race &rarr; funded account &rarr; payouts, raced '
            f'when and the way your paper bots raced &middot; real NQ prices, Lucid&#39;s rules, your code prices (${eval_fee:,.0f} '
            f'evaluation, ${reset_fee:,.0f} reset)'
            + (f' &middot; stop rule: no more evaluations once down ${stop_usd:,.0f}' if stop_usd else '')
            + ' &middot; nothing bought for real</div></div>'
            f'<div class="vx-spacer"></div>{vx.pill("LIVE", vx.OK)}</div>', unsafe_allow_html=True)
        big = st.columns(3)
        for col, (label, value, colour, note) in zip(big, (
                ("Spent", f"${fees:,.0f}", vx.TEXT, f"{bought} evaluation{'s' if bought != 1 else ''} at "
                 f"${eval_fee:,.0f} + {sum(s['resets'] for s in sl)} reset{'s' if sum(s['resets'] for s in sl) != 1 else ''}"
                 f" at ${reset_fee:,.0f}"),
                ("Paid to you", f"${to_you:,.0f}", vx.POS if to_you > 0 else vx.TEXT,
                 f"{payouts} payout{'s' if payouts != 1 else ''} (you keep 90%)"),
                ("Result", usd(net), vx.pl_color(net), "paid to you minus spent"))):
            col.markdown(f'<div class="vx-card" style="padding:18px 20px"><div class="vx-label">{label}</div>'
                         f'<div class="vx-value" style="font-size:34px;color:{colour}">{value}</div>'
                         f'<div class="vx-note">{note}</div></div>', unsafe_allow_html=True)
        k = st.columns(4)
        k[0].markdown(vx.card("Evaluations", f"{passes} passed", note=f"of {attempts} raced (about 4 in 10 expected)"),
                      unsafe_allow_html=True)
        k[1].markdown(vx.card("Funded accounts", f"{passes}", note=f"{blown} blown so far"), unsafe_allow_html=True)
        k[2].markdown(vx.card("Payouts", f"{payouts}", note="history: the first after ~10 trading days, typically"),
                      unsafe_allow_html=True)
        k[3].markdown(vx.card("Running", f"{len(daily)} days", note=f"trading days since {pd.Timestamp(since):%d %b}"),
                      unsafe_allow_html=True)
        # --- where each account stands now ---
        nxt = "tonight at 01:00 your time" if datetime.now(_LOCAL_TZ).hour < 1 or datetime.now(_LOCAL_TZ).hour >= 18 \
            else "at 01:00 your time"
        if same_days:
            st.markdown(vx.banner(same_days, "info"), unsafe_allow_html=True)
        stopped_plan = any(e["what"].startswith("STOP RULE") for e in ev)
        lines = []
        for s_ in sl:
            if s_["stage"] == "stopped" or (s_["stage"] == "waiting" and stopped_plan):
                lines.append(f"<b>Account {s_['n']}</b> — stopped by your stop rule (money made down "
                             f"${stop_usd:,.0f}): nothing more is bought — time to look again.")
                continue
            if s_["stage"] == "waiting":
                lines.append(f"<b>Account {s_['n']}</b> — not bought yet: it comes once the money made reaches "
                             f"+${step_usd * (s_['n'] - start_slots):,.0f}.")
                continue
            if s_.get("open"):
                state = f"race running now ({usd(s_['open']['pnl'])})"
            else:
                state = f"next race {nxt}"
            if s_["stage"] == "eval":
                need = max(0.0, 3_000.0 - s_["bal"])
                room = s_["bal"] - s_["floor"]
                lines.append(f"<b>Account {s_['n']}</b> — in an evaluation: needs <b>+${need:,.0f}</b> to pass, can "
                             f"lose ${room:,.0f} before it fails · {state}.")
            else:
                want = plan50.want[min(s_["n_paid"], len(plan50.want) - 1)]
                need = max(0.0, plan50.keep + want - s_["bal"],                 # as lucid_fleet.Slot.need
                           s_["cycle_best"] / 0.40 - s_["cycle_pnl"] if s_["cycle_best"] > 0 else 0.0,
                           plan50.min_cycle - s_["cycle_pnl"])
                lines.append(f"<b>Account {s_['n']}</b> — <b>funded</b>, balance ${50_000 + s_['bal']:,.0f}: "
                             f"{'payout money reached — the payout is taken at a close' if need <= 0 else f'needs +${need:,.0f} for its next payout (${want:,.0f})'}"
                             f" · {s_['cycle_days']} of 3 cycle days · {state}.")
        st.markdown(vx.section("Right now"), unsafe_allow_html=True)
        st.markdown(vx.banner("<br>".join(lines), "info"), unsafe_allow_html=True)
        # --- every day, in words ---
        st.markdown(vx.section("What happened, day by day (newest first)"), unsafe_allow_html=True)
        words = {"evaluation bought": ("🛒", "Bought an evaluation"),
                 "evaluation failed -- reset": ("❌", "Evaluation race lost — reset bought"),
                 "evaluation PASSED -- funded account": ("✅", "Evaluation PASSED — the funded account starts"),
                 "evaluation race closed at 16:00": ("⏹", "Evaluation race closed at 16:00"),
                 "funded race won": ("💰", "Funded race won"),
                 "funded account blown -- new evaluation": ("💥", "Funded account blown — new evaluation bought"),
                 "no race -- payout money reached": ("⏸", "No race — payout money reached, waiting for the payout"),
                 "STOP RULE -- no more evaluations bought": ("🛑", "STOP RULE — the money made is down the limit: no"
                                                                  " more evaluations or resets are bought"),
                 "evaluation failed -- no reset (stop rule)": ("❌", "Evaluation race lost — no reset (stop rule)"),
                 "funded account blown -- no new evaluation (stop rule)": ("💥", "Funded account blown — no new"
                                                                                " evaluation (stop rule)")}
        # what both paper bots did each day (owner 2026-10-21: "the funded bot won a trade but it is not showing on
        # the fleet"): the plan's one account follows only one of them -- the trial bot while it is in an
        # evaluation, the funded bot while it is funded -- so the other bot's race is shown under the day
        bot_days: dict = {}
        for bot, fname in (("trial", f"lucid_futures_trades{tags[0]}.jsonl"),
                           ("funded", f"lucid_futures_funded_trades{tags[1]}.jsonl")):
            try:
                ledger = (ROOT / "data" / fname).read_text(encoding="utf-8").splitlines()
            except OSError:
                ledger = []
            for ln in ledger:
                try:
                    tr = json.loads(ln)
                    bot_days.setdefault(_lf.trading_day_of(tr["opened_ts"]), {}).setdefault(bot, []).append(
                        float(tr.get("pnl_usd") or 0.0))
                except (ValueError, KeyError, TypeError):
                    continue

        def bots_line(day: str) -> str:
            got = bot_days.get(str(day)) or {}
            parts = []
            for key, name in (("trial", "trial bot"), ("funded", "funded bot")):
                if key in got:
                    p_ = got[key]
                    bits = ", ".join(("won " if x > 0 else "lost ") + usd(x) for x in p_[:3])
                    parts.append(f"{name} {bits}" + (f" (+{len(p_) - 3} more)" if len(p_) > 3 else ""))
            if not parts:
                return ""
            return (f'<div style="display:grid;grid-template-columns:96px 28px 1fr;gap:8px;padding:0 12px 8px;'
                    f'font-size:12px;color:{vx.TEXT_FAINT}"><span></span><span>🤖</span>'
                    f'<span>Your paper bots that day: {" · ".join(parts)}</span></div>')

        rows = []
        for e in reversed(ev):
            w = e["what"]
            icon, text = words.get(w, ("🏦" if w.startswith("PAYOUT") else "•",
                                       (f"Payout {w.split('#')[-1]} taken" if w.startswith("PAYOUT") else
                                        w.replace(" -- ", " — ").capitalize())))
            if w.startswith("funded race") and w != "funded race won":
                icon, text = "📉", w.replace(" -- ", " — ").capitalize()
            race = f" (race {usd(e['race'])})" if e.get("race") is not None else ""
            amount = e["usd"]
            amt = (f'<span style="color:{vx.pl_color(amount)};font-family:JetBrains Mono,monospace">{usd(amount)}</span>'
                   if amount else "")
            who = f"Account {e['slot']}: " if e["slot"] else "Your plan: "       # slot 0: the plan (the stop rule)
            rows.append((str(e["day"]),
                         f'<div style="display:grid;grid-template-columns:96px 28px 1fr auto;gap:8px;padding:8px 12px;'
                         f'border-top:1px solid {vx.BORDER_SOFT};font-size:13.5px;color:{vx.TEXT}">'
                         f'<span style="color:{vx.TEXT_MUTED}">{pd.Timestamp(e["day"]):%a %d %b}</span><span>{icon}</span>'
                         f'<span>{who}{text}{race}</span>{amt}</div>'))
        shown = []
        for i, (day, html) in enumerate(rows):
            shown.append(html)
            if i + 1 == len(rows) or rows[i + 1][0] != day:      # the day's last line: what both bots did
                shown.append(bots_line(day))
        st.markdown(f'<div style="border:1px solid {vx.BORDER};border-radius:10px;overflow:hidden;background:{vx.SURFACE}">'
                    + "".join(shown[:160]) + "</div>", unsafe_allow_html=True)
        st.caption("Your plan replayed on the real NQ prices since the bots started, racing at the moments your paper "
                   "bots really raced and the way they raced: an evaluation like the trial bot, a funded race like the "
                   "funded bot, no race on a day they didn't (out at 16:00 New York; Lucid's loss line, payout rules "
                   "and your fees). Under each day, 🤖 shows what both paper bots did: your plan's one account follows "
                   "only one of them (the trial bot while it is in an evaluation, the funded bot while it is funded). "
                   "History for comparison — 102 start dates 2024–2026: the first payout typically after 10 trading "
                   "days and 5 evaluation attempts, 83% within a month, all within 3 months. Pick Last 3 / 12 months "
                   "above to see the same plan on history.")

    def _fleet_view():
        _lf = _fleet_mod()
        usd = lambda x: f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"  # noqa: E731
        forward = "2026-10-01"           # the real test began with the race bots (owner 2026-10-16: "those 2 weeks")
        today = datetime.now(timezone.utc).date()
        periods = {"Real test · since 1 Oct": forward,
                   "Last 3 months": str(today - timedelta(days=91)),
                   "Last 12 months": str(today - timedelta(days=365))}
        period = st.segmented_control("Period", list(periods), default="Real test · since 1 Oct", key="fleet_period",
                                      label_visibility="collapsed") or "Real test · since 1 Oct"
        # the 2-month side-by-side test (owner 2026-10-21): the Real test of either strategy's paper bots
        strat_pick = list(_STRATEGIES)[0]
        if _compare_on() and period == "Real test · since 1 Oct":
            strat_pick = st.segmented_control("Strategy", list(_STRATEGIES), default=strat_pick, key="fleet_strategy",
                                              label_visibility="collapsed") or strat_pick
            st.caption(_STRATEGIES[strat_pick]["what"])
        new_strat = _STRATEGIES[strat_pick]["funded_ad"]
        with st.expander("Plan settings — yours: start with 1 account, +1 each time the money made reaches another "
                         "+\\$2,500, up to 5 · \\$115 evaluation, \\$120 reset · stop when down \\$10,000"):  # \$: no formula
            c = st.columns([0.9, 0.9, 1.1, 0.9, 0.9, 0.9])
        start_slots = int(c[0].number_input("Start with", 1, 5, 1, key="fleet_start_slots",
                                            help="Accounts bought on day one (your plan since 14 Oct: 1)."))
        slots = int(c[1].number_input("Up to", 1, 5, 5, key="fleet_slots",
                                      help="At most 5 accounts at a time -- evaluations and funded accounts together."))
        rules = {"every +$2,500 made": ("money", 2_500.0),        # the owner's choice, 2026-10-09
                 "after every payout": ("payout", 2_500.0), "every +$5,000 made": ("money", 5_000.0)}
        rule = c[2].selectbox("Add an account", list(rules), key="fleet_rule",
                              help="Every +$2,500 made: your plan -- the next account is bought once the money made so"
                                   " far (payouts minus fees) reaches each step. After every payout: faster, deeper dips.")
        step_on, step_usd = rules[rule]
        stagger = 0                      # the starting accounts are bought together
        defaults = _lf.FleetConfig()
        eval_fee = float(c[3].number_input("Evaluation $", 0, 500, int(defaults.eval_fee), step=5, key="fleet_eval_fee",
                                           help="LucidPro 50K with your code: $115 with the daily loss limit (the"
                                                " race never needs it), $140 without. List price $185."))
        reset_fee = float(c[4].number_input("Reset $", 0, 500, int(defaults.reset_fee), step=5, key="fleet_reset_fee",
                                            help="After a failed evaluation. List price $120 -- lower it if your"
                                                 " code also covers resets."))
        stop_usd = float(c[5].number_input("Stop when down $", 0, 100_000, int(defaults.stop_usd or 0), step=1_000,
                                           key="fleet_stop_usd",
                                           help="Your stop rule (21 Oct): once the money made (payouts minus fees) is"
                                                " down this much, no more evaluations or resets are bought -- the"
                                                " accounts already running finish. 0 = no stop."))
        since = periods[period]
        live = period == "Real test · since 1 Oct"
        tags = _AD_TAGS if new_strat else ("", "")

        def bots_key(tg) -> str:          # the bots' ledgers and open races: a new race recomputes the Real test
            return "|".join(str(int((ROOT / "data" / f).stat().st_mtime)) for f in (
                f"lucid_futures_trades{tg[0]}.jsonl", f"lucid_futures_funded_trades{tg[1]}.jsonl",
                f"state_lucid_futures{tg[0]}.json", f"state_lucid_futures_funded{tg[1]}.json")
                if (ROOT / "data" / f).exists()) or "bots"

        until = _compare_until()
        if live and new_strat:            # the new strategy's Real test starts with its bots' first race
            ad_days = sorted(_lf.bot_races(ROOT / "data", None, *_AD_TAGS))
            if not ad_days:
                st.markdown(vx.banner(
                    "The new strategy's paper bots have not raced yet. They start with the next update on the 24/7"
                    " computer and race beside the current ones for 2 months"
                    + (f" (until {until:%d %b %Y})" if until else "") + "; this page fills after their first race.",
                    "info"), unsafe_allow_html=True)
                return
            since = ad_days[0]
        bars = _fleet_minutes(since)
        last_bar = str(bars.index[-1]) if len(bars) else ""
        res = _fleet_run(since, last_bar, slots, stagger, eval_fee, reset_fee, start_slots, step_usd,
                         getattr(_lf, "_loaded_mtime", 0.0), step_on, bots_key(tags) if live else "", stop_usd,
                         "ad" if new_strat else "")
        sl, ev, daily = res["slots"], res["events"], res["daily"]
        if live and not daily.empty:
            same_days = ""
            if new_strat:                 # the same days with the current strategy's bots, for the comparison
                cur = _fleet_run(since, last_bar, slots, stagger, eval_fee, reset_fee, start_slots, step_usd,
                                 getattr(_lf, "_loaded_mtime", 0.0), step_on, bots_key(("", "")), stop_usd, "")
                c_sl = cur["slots"]
                same_days = (f"<b>Side-by-side test</b>{f' until {until:%d %b %Y}' if until else ''}. The same days with"
                             f" the <b>current</b> strategy (200-day trend): result"
                             f" <b>{usd(sum(s['net'] for s in c_sl))}</b>, {sum(s['payouts'] for s in c_sl)} payout(s),"
                             f" {sum(s['passes'] for s in c_sl)} evaluation(s) passed &middot; the <b>new</b> strategy"
                             f" (below): <b>{usd(sum(s['net'] for s in sl))}</b>,"
                             f" {sum(s['payouts'] for s in sl)} payout(s), {sum(s['passes'] for s in sl)} passed.")
            _real_test_view(_lf, res, since, start_slots, slots, step_usd, eval_fee, reset_fee, stop_usd, tags,
                            "new strategy (A−D breadth)" if new_strat else "", same_days)
            return
        plan = (f"{start_slots} accounts, growing to {slots}" if start_slots < slots
                else f"{slots} account{'s' if slots > 1 else ''} at once")
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Fleet &middot; {plan}</div>'
            f'<div class="vx-sub">Each slot: evaluation race &rarr; funded account &rarr; payouts, one race a day at'
            f' 19:00 New York, on real NQ minute prices and Lucid&#39;s rules &mdash; simulated, no orders</div></div>'
            f'<div class="vx-spacer"></div>{vx.pill("LIVE SIMULATION" if live else "REPLAY OF HISTORY", vx.INFO)}</div>',
            unsafe_allow_html=True)
        if daily.empty:
            st.markdown(vx.banner(
                f"The fleet's first race starts tonight at <b>19:00 New York (01:00 your time)</b> with"
                f" <b>{min(start_slots, slots)} account{'s' if min(start_slots, slots) > 1 else ''}</b>"
                + ((f"; another one is bought after every payout, up to {slots}." if step_on == "payout" else
                    f"; another one is bought each time the money made reaches another <b>+${step_usd:,.0f}</b>,"
                    f" up to {slots}.") if start_slots < slots else ".")
                + " Results appear"
                " here after each race. To see what the same fleet would have done before, pick <b>Last 3 months</b>"
                " or <b>Last 12 months</b>.", "info"), unsafe_allow_html=True)
            return
        paid = sum(s["paid"] for s in sl)
        fees = sum(s["fees"] for s in sl)
        net = sum(s["net"] for s in sl)
        months = max(len(daily) / 21.0, 1 / 21.0)
        evals = sum(s["passes"] + s["resets"] for s in sl)
        passes = sum(s["passes"] for s in sl)
        k = st.columns(4)
        k[0].markdown(vx.card("Money to you", f'<span style="color:{vx.pl_color(net)}">{usd(net)}</span>',
                              note="payouts (90%) minus fees &middot; " + (f"about {usd(net / months)} a month" if len(daily) >= 21 else "too early for a monthly figure")),
                      unsafe_allow_html=True)
        k[1].markdown(vx.card("Payouts", f"{sum(s['payouts'] for s in sl)}",
                              note=f"${paid:,.0f} paid, ${0.9 * paid:,.0f} to you"), unsafe_allow_html=True)
        k[2].markdown(vx.card("Fees paid", f"${fees:,.0f}",
                              note=f"{sum(s['evals'] for s in sl)} evaluations bought, {sum(s['resets'] for s in sl)} resets"),
                      unsafe_allow_html=True)
        k[3].markdown(vx.card("Evaluations passed", f"{passes} of {evals}" if evals else "—",
                              note=(f"{passes / evals:.0%} &middot; " if evals else "") + "about 4 in 10 expected"),
                      unsafe_allow_html=True)
        k2 = st.columns(4)
        k2[0].markdown(vx.card("Funded accounts", f"{passes}",
                               note=f"{sum(s['blown'] for s in sl)} blown so far"), unsafe_allow_html=True)
        k2[1].markdown(vx.card("Right now", f"{sum(s['stage'] == 'funded' for s in sl)} funded",
                               note=f"{sum(s['stage'] == 'eval' for s in sl)} in an evaluation &middot; "
                                    f"{sum(s['stage'] == 'waiting' for s in sl)} not started"), unsafe_allow_html=True)
        k2[2].markdown(vx.card("Trading days", f"{len(daily)}", note=f"since {since}"), unsafe_allow_html=True)
        racing = [s for s in sl if s.get("open")]
        k2[3].markdown(vx.card("Races open now", f"{len(racing)}",
                               note=(f"{usd(sum(s['open']['pnl'] for s in racing))} open" if racing
                                     else "the next race starts 19:00 New York")), unsafe_allow_html=True)

        # --- money over time ---
        x = pd.to_datetime(daily["day"])
        fig = go.Figure()
        for name, y, col in (("Payouts to you (90%)", 0.9 * daily["paid"], vx.SERIES[0]),
                             ("Fees", daily["fees"], vx.SERIES[1]), ("Money to you", daily["net"], vx.SERIES[2])):
            fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=col, width=2, shape="hv"),
                                     hovertemplate="%{x|%a %d %b}: $%{y:,.0f}<extra>" + name + "</extra>"))
        fig.update_layout(**vx.plotly_layout(height=300))
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0, font=dict(color=vx.TEXT_MUTED)))
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, width="stretch", key="fleet_money", config={"displayModeBar": False})

        # --- every slot ---
        st.markdown(vx.section("Every slot"), unsafe_allow_html=True)
        stage_txt = {"waiting": "not started", "eval": "evaluation", "funded": "funded account",
                     "stopped": "stopped (stop rule)"}
        st.dataframe(pd.DataFrame([{
            "Slot": f"#{s['n']}",
            "Now": stage_txt.get(s["stage"], s["stage"])
                   + (f" · ${50_000 + s['bal']:,.0f}" if s["stage"] not in ("waiting", "stopped") else "")
                   + (f" · race open {usd(s['open']['pnl'])}" if s.get("open") else ""),
            "Evaluations bought": s["evals"], "Passed": s["passes"], "Failed (reset)": s["resets"],
            "Funded blown": s["blown"], "Payouts": s["payouts"], "To you": f"${0.9 * s['paid']:,.0f}",
            "Fees": f"${s['fees']:,.0f}", "Money to you": usd(s["net"]),
        } for s in sl]), width="stretch", hide_index=True)

        # --- what happened ---
        st.markdown(vx.section("What happened (newest first)"), unsafe_allow_html=True)
        try:
            local_tz = ZoneInfo((_ENV.get("VERTEX_LOCAL_TZ") or "Europe/Prague").strip())
        except Exception:  # noqa: BLE001
            local_tz = datetime.now().astimezone().tzinfo
        pick = st.segmented_control("Show", ["All", "Races", "Payouts", "Fees"], default="All", key="fleet_ev_kind")             or "All"
        keep = {"All": lambda e: True, "Races": lambda e: e.get("race") is not None,
                "Payouts": lambda e: e["what"].startswith("PAYOUT"), "Fees": lambda e: e["usd"] < 0}[pick]
        rows_ev = [e for e in reversed(ev) if keep(e)][:300]
        st.dataframe(pd.DataFrame([{
            "Trading day": f"{pd.Timestamp(e['day']):%a %d %b}",
            "Your time": f"{pd.Timestamp(e['ts']).tz_convert(local_tz):%a %H:%M}" if e.get("ts") else "",
            "Account": f"#{e['slot']}" if e["slot"] else "plan", "What": e["what"].replace(" -- ", " — "),
            "Race": usd(e["race"]) if e.get("race") is not None else "",
            "Fee / payout": usd(e["usd"]) if e["usd"] else "",
            "Why": e.get("why", ""),
        } for e in rows_ev]), width="stretch", hide_index=True, height=420,
            column_config={"Why": st.column_config.TextColumn("Why", width="large")})
        st.caption("Hover a row's \"Why\" to read it all. Each race: the side (the 200-day trend), the entry at 19:00"
                   " New York, the take-profit and the loss line, and which one the price reached first, when.")
        st.caption(
            "Simulated from NinjaTrader's 1-minute NQ prices the way the bots judge a race they did not watch: the loss"
            " line counts open trades and follows the best daily close; a minute touching both levels counts as the"
            " loss; the take-profit fills only once price trades through it; spread, commission and a tick of slippage"
            " are paid. One race per slot per trading day at 19:00 New York (restarts win less), out at 16:00. All"
            " slots race the same NQ at the same moment, so they win and lose together: more slots mean more money and"
            " bigger swings, not better odds. The long-run average (2019-2026 replay) is about $1,445 a month per slot"
            " at $115 / $120 ($1,650 if resets are $75 too); the last 12 months were better because NQ rose strongly.")

    _fleet_view()
    st.stop()


if bot_choice == "DESK":
    st.markdown(vx.CSS, unsafe_allow_html=True)

    def _desk_view():
        """The trading desk (owner 2026-10-21): four agents on their own paper account -- what they saw, argued and
        decided. DESK_DATA_DIR (tests only) reads another folder."""
        import os
        from metatraderbot import trading_desk as _td
        data = Path(os.environ.get("DESK_DATA_DIR") or (ROOT / "data"))

        def rd(name, default):
            try:
                return json.loads((data / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return default

        state, journal, book = rd("desk_state.json", {}), rd("desk_journal.json", []), rd("paper_Desk.json", {})
        st.markdown(
            '<div class="vx-top"><div class="vx-mark">V</div><div><div class="vx-name">Trading desk &middot; four '
            'agents, one paper account</div><div class="vx-sub">Scout (code) finds big moves &middot; News reads the '
            'headlines &middot; Chart plans the trade &middot; Skeptic argues with both and decides &middot; a Guard '
            'in code enforces the rules &middot; free prices, 10 minutes late &middot; paper only</div></div>'
            f'<div class="vx-spacer"></div>{vx.pill("PAPER", vx.INFO)}</div>', unsafe_allow_html=True)
        if not state:
            st.markdown(vx.banner(
                "The desk has not run yet. It starts with the next update on the 24/7 laptop, needs your Claude key "
                "there (DESK_ANTHROPIC_API_KEY in its settings) and then watches NQ every minute. Its notes go to the "
                "<b>Trading desk</b> folder in your Obsidian vault.", "info"), unsafe_allow_html=True)
            return
        usd = lambda x: f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"  # noqa: E731
        acct = state.get("account") or {}
        bal = float(book.get("balance", state.get("balance", _td.START)))
        floor = _td.floor_of(acct)
        days = state.get("days") or {}
        today = days.get(_td.trading_day()) or {}
        ev = state.get("evals") or {}
        bud = state.get("budget") or {}
        hb = state.get("heartbeat")
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds() if hb else None
        except ValueError:
            age = None
        if age is None or age > 600:
            st.markdown(vx.banner("The desk is not running on the 24/7 laptop right now (no sign of life for "
                                  f"{'a while' if age is None else f'{age / 60:.0f} minutes'}). The watchdog starts "
                                  "it again within 15 minutes.", "warn"), unsafe_allow_html=True)
        k = st.columns(4)
        k[0].markdown(vx.card(f"Evaluation #{acct.get('n', 1)}", f"${bal:,.0f}",
                              note=f"${_td.START + _td.TARGET - bal:,.0f} to the target · ${bal - floor:,.0f} above "
                                   f"the loss line"), unsafe_allow_html=True)
        k[1].markdown(vx.card("Today", f"{today.get('trades', 0)} of {_td.MAX_TRADES_DAY} trades",
                              note=f"{today.get('alerts', 0)} alarm(s) from the Scout · "
                                   f"{usd(float(today.get('pnl', 0.0)))}"), unsafe_allow_html=True)
        k[2].markdown(vx.card("Evaluations", f"{ev.get('passed', 0)} passed",
                              note=f"{ev.get('failed', 0)} failed · fees ${float(ev.get('fees', _td.EVAL_FEE)):,.0f}"),
                      unsafe_allow_html=True)
        k[3].markdown(vx.card("AI fees", f"${float(bud.get('spent', 0.0)):.2f}",
                              note=f"of ${_td.BUDGET_USD:g} in {bud.get('month', 'this month')} · "
                                   f"{int(bud.get('calls', 0))} calls"), unsafe_allow_html=True)
        opn = [x for x in (book.get("tickets") or {}).values() if x.get("open")]
        if opn:
            x = opn[0]
            st.markdown(vx.banner(
                f"<b>Open now:</b> {'long' if x['sign'] > 0 else 'short'} {x['qty']} MNQ at {x['entry']:,.2f} · stop "
                f"{x['sl']:,.2f} · target {x['tp']:,.2f}", "info"), unsafe_allow_html=True)
        st.markdown(vx.section("The desk's debates (newest first)"), unsafe_allow_html=True)
        if not journal:
            st.caption("No big move yet. The Scout wakes the others when NQ moves 0.45% in 15 minutes or 0.9% in "
                       "an hour (about 1.5 times a day).")
        for i, e in enumerate(journal[:15]):
            try:
                when = datetime.fromisoformat(e.get("at", "")).astimezone(_LOCAL_TZ)
                head = f"{when:%a %d %b %H:%M} — {_td.entry_title(e)}"
            except (TypeError, ValueError):
                head = _td.entry_title(e)
            with st.expander(head, expanded=(i == 0)):
                st.markdown("\n\n".join(_td.entry_lines(e)))
        trades = [e for e in journal if e.get("outcome")]
        if trades:
            st.markdown(vx.section("Trades"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "When": e.get("at", "")[:16].replace("T", " "), "Side": (e.get("order") or {}).get("direction"),
                "Micros": (e.get("order") or {}).get("qty"), "Entry": (e.get("order") or {}).get("entry"),
                "Exit": e["outcome"].get("exit"), "Why": e["outcome"].get("why"),
                "Result": usd(float(e["outcome"].get("pnl", 0.0)))} for e in trades]), width="stretch",
                hide_index=True)
        if state.get("lessons"):
            st.markdown(vx.section("What the desk learned"), unsafe_allow_html=True)
            st.markdown("\n".join(f"- **{x.get('day')}:** {x.get('text')}" for x in state["lessons"][:10]))
        st.caption("A paper experiment on free prices (10 minutes late): judge it on weeks of live results, never on a "
                   "few days. Every debate is also in the Obsidian folder Trading desk (Desk journal).")

    _desk_view()
    st.stop()

if bot_choice == "BRAIN":
    st.markdown(vx.CSS, unsafe_allow_html=True)

    def _brain_view():
        try:
            local_tz = ZoneInfo((_ENV.get("VERTEX_LOCAL_TZ") or "Europe/Prague").strip())
        except Exception:  # noqa: BLE001
            local_tz = datetime.now().astimezone().tzinfo
        mem = _brain()
        trades = sorted((mem.get("trades") or {}).values(), key=lambda x: x["opened"], reverse=True)
        fixes = mem.get("fixes") or {}
        lessons = mem.get("lessons") or {}
        usd = lambda x: f"{'+' if x >= 0 else '−'}${abs(x):,.2f}"  # noqa: E731
        try:
            upd = f"{datetime.fromisoformat(mem['updated']).astimezone(local_tz):%a %d %b %H:%M}"
        except (KeyError, ValueError):
            upd = "—"
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Brain &middot; what every trade taught</div>'
            f'<div class="vx-sub">Every trade of both bots, why it won or lost, the mistakes fixed for good and the '
            f'lessons tested against history &middot; updated {upd}</div></div>'
            f'<div class="vx-spacer"></div>{vx.pill(f"{len(trades)} TRADES", vx.INFO)}</div>',
            unsafe_allow_html=True)
        if not trades:
            st.markdown(vx.banner("No trades remembered yet - the brain writes after every trading day.", "info"),
                        unsafe_allow_html=True)
            return
        mistakes: dict = {}
        for x in trades:
            for m in x.get("mistakes") or []:
                mistakes.setdefault(m, []).append(x)
        losses = [x for x in trades if x["pnl"] <= 0]
        n_exec = sum(1 for x in losses if x["kind"] == "execution")
        verdicts = [v["verdict"] for v in lessons.values()]
        k = st.columns(4)
        k[0].markdown(vx.card("Trades remembered", f"{len(trades)}",
                              note=f"trial {sum(1 for x in trades if x['bot'] == 'Trial')} · funded "
                                   f"{sum(1 for x in trades if x['bot'] == 'Funded')}"), unsafe_allow_html=True)
        k[1].markdown(vx.card("Losses explained", f"{len(losses)}",
                              note=f"{len(losses) - n_exec} market losses · {n_exec} execution mistakes"),
                      unsafe_allow_html=True)
        k[2].markdown(vx.card("Mistake kinds fixed", f"{len(mistakes)}", note="each has a permanent fix"),
                      unsafe_allow_html=True)
        k[3].markdown(vx.card("Market lessons", f"{sum(1 for v in verdicts if v.startswith('CONFIRMED'))} confirmed",
                              note=f"{sum(1 for v in verdicts if v.startswith('watching'))} watching · "
                                   f"{sum(1 for v in verdicts if v.startswith('rejected'))} rejected"),
                      unsafe_allow_html=True)
        t_lessons, t_every, t_lab = st.tabs([":material/school: Lessons", ":material/receipt_long: Every trade",
                                             ":material/science: Prediction lab"])
        with t_lessons:
            st.markdown(vx.section("Never again — execution mistakes and their fixes"), unsafe_allow_html=True)
            if mistakes:
                st.dataframe(pd.DataFrame([{
                    "Mistake": (fixes.get(m) or {}).get("name", m), "Times": len(tl),
                    "Last seen": f"{pd.Timestamp(tl[0]['day']):%a %d %b} (trade #{tl[0]['no']})",
                    "What happened": (fixes.get(m) or {}).get("what", ""), "Fix": (fixes.get(m) or {}).get("fix", "")}
                    for m, tl in mistakes.items()]), width="stretch", hide_index=True)
            else:
                st.caption("No execution mistakes so far.")
            st.markdown(vx.section("Market lessons — tested against the race's history"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Condition before the race": v["name"],
                "Live: losses / trades": f"{v['live_losses']} / {v['live_trades']}",
                "History: wins with it / without": (f"{v['history']['with']:.0%} / {v['history']['without']:.0%}"
                                                    if v["history"]["n_with"] else "—"),
                "Verdict": v["verdict"]} for v in lessons.values()]), width="stretch", hide_index=True)
            st.caption("A losing day is only turned into a rule when 5+ live trades AND the race's history agree - "
                       "otherwise one unlucky day would teach the bots a superstition. Confirmed lessons go to the "
                       "prediction lab below to prove themselves before they change any trading.")
        with t_every:
            icon = {"target": "✅", "loss limit": "❌", "16:00 close": "⏹️", "closed": "⚠️"}
            st.dataframe(pd.DataFrame([{
                "#": x["no"], "Day (New York)": f"{pd.Timestamp(x['day']):%a %d %b}", "Bot": x["bot"],
                "Result": f"{icon.get(x['result'], '')} {x['result']}", "P&L": usd(x["pnl"]),
                "Why": (x.get("verdict") or "").split(" — ")[0]} for x in trades]),
                width="stretch", hide_index=True)
            pick = st.selectbox("Open a trade", range(len(trades)), key="brain_pick",
                                format_func=lambda n: (f"#{trades[n]['no']} · {trades[n]['bot']} · "
                                                       f"{pd.Timestamp(trades[n]['day']):%a %d %b} · {trades[n]['result']} · "
                                                       f"{usd(trades[n]['pnl'])}"))
            x = trades[pick]
            st.markdown(f"**Verdict:** {x.get('verdict', '')}")
            st.markdown("\n".join(f"- {line}" for line in (x.get("story") or [])))
            try:
                account = (_ENV.get("NT_ACCOUNT") or "Sim101").strip()
                port = int(_ENV.get("NT_BRIDGE_PORT") or 45973)
                o, c = pd.Timestamp(x["opened"]), pd.Timestamp(x["closed"])
                b, mins = _closed_trade_bars("NQ", x["opened"], x["closed"], account, port)
                fig = _trade_figure(b, mins, local_tz, entry_t=o.tz_convert(local_tz), entry_px=x["entry"], tp=x.get("tp"),
                                    line=x.get("line"), exit_t=c.tz_convert(local_tz), exit_px=x["exit"],
                                    exit_text=f"Sold {x['exit']:,.2f} · {usd(x['pnl'])}", exit_color=vx.pl_color(x["pnl"]),
                                    long=x.get("direction", "long") != "short", height=330, title=f"Trade #{x['no']} ({x['bot']}) - your time")
                st.plotly_chart(fig, width="stretch", key="brain_trade_chart", config={"displayModeBar": False})
            except Exception as e:  # noqa: BLE001 -- NinjaTrader closed / no bars
                st.caption(f"Chart unavailable right now ({e}).")
        with t_lab:
            st.markdown(vx.section("Prediction lab — up or down predictors, scored on days they never saw"),
                        unsafe_allow_html=True)
            lp_ = ROOT / "data" / "prediction_lab.json"
            try:
                lab = json.loads(lp_.read_text(encoding="utf-8")) if lp_.exists() else {}
            except (OSError, ValueError):
                lab = {}
            summ = lab.get("summary") or {}
            if not summ:
                st.caption("The prediction lab has not scored a day yet.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Predictor": _PREDICTOR_NAMES.get(key, (key, ""))[0], "What it does": _PREDICTOR_NAMES.get(key, ("", key))[1],
                    "Raced": v["raced"], "Won / resolved": f"{v['wins']} / {v['resolved']}",
                    "Win rate": f"{v['win_rate']:.0%}" if v["resolved"] else "—",
                    "Deployed race, same days": f"{v['deployed_same_days']:.0%}" if v["resolved"] else "—",
                    "Status": v["status"]} for key, v in summ.items()]), width="stretch", hide_index=True)
                st.caption(f"{len(lab.get('days') or {})} trading days scored since {lab.get('start', '?')}. A predictor "
                           "is worth switching to only after 60+ resolved races clearly better than the deployed race "
                           "(z >= 2) - until then every number here can be luck.")

    _brain_view()
    st.stop()


# FUNDED (futures mode): scripts/run_lucid_futures_funded.py -- the FUNDED
# RACE on NinjaTrader (since 2026-10-01). Written like the TRIAL page, for
# anyone: one plain "Today" card, the figures that decide the next payout,
# the results, and every trade as a chart; the technical detail folded away.
if bot_choice == "FUNDED":
    from metatraderbot import futures as _fut
    from metatraderbot import lucid_rules as _lr
    from metatraderbot import target_race as _tr
    from metatraderbot.nt_engine import BridgeError as _BridgeError

    st.markdown(vx.CSS, unsafe_allow_html=True)
    refresh_s = _refresh_seconds(5)
    _FUNDED_REASON = {
        "tp": "take-profit reached", "take_while_away": "take-profit reached while the laptop slept",
        "mll": "loss limit hit - account blown", "mll_while_away": "loss limit hit while the laptop slept",
        "mll_rescored": "loss limit hit first (re-scored)", "time_exit": "16:00 New York close",
        "exit_while_away": "16:00 New York close (laptop asleep)", "dll": "daily loss limit",
        "missed_flat_window": "closed after a missed flat window", "sl": "safety stop",
    }

    @st.fragment(run_every=refresh_s)
    def _race_funded_view():
        ANCHOR, KEEP, SPLIT = 50_000.0, 52_100.0, 0.90           # run_lucid_futures_funded's rules
        WANT = (500.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)
        CAPS = (2_000.0, 2_500.0, 2_500.0, 2_500.0, 2_500.0)
        # --- every funded account first, then the one picked, in detail ---
        insts = _funded_instances()
        # the current strategy and the new one side by side (owner 2026-10-21): each has its own funded account
        strat_name = list(_STRATEGIES)[0]
        if _compare_on():
            strat_name = st.segmented_control("Strategy", list(_STRATEGIES), default=strat_name, key="funded_strategy",
                                              label_visibility="collapsed") or strat_name
            st.caption(_STRATEGIES[strat_name]["what"])
        insts = [i for i in insts if _is_ad_account(i) == _STRATEGIES[strat_name]["funded_ad"]]
        if not insts:
            st.info("The new strategy's funded account (Sim102-AD) has not raced yet. The watchdog on the computer "
                    "that runs the bots starts it, and it appears here after its first race.")
            return
        st.markdown(vx.section(f"Funded accounts ({len(insts)})"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([_funded_overview(i) for i in insts]), width="stretch", hide_index=True)
        names = [i["account"] for i in insts]
        if len(insts) > 1:
            pick = st.segmented_control("Show this account in detail", names, default=names[0],
                                        key="funded_account_pick") or names[0]
        else:
            pick = names[0]
            st.caption("More funded accounts appear here by themselves: add each one's NinjaTrader account name to"
                       " LUCID_FUNDED_EXTRA_ACCOUNTS in the settings file (comma-separated) and the watchdog starts"
                       " a funded bot for it within 15 minutes.")
        inst = next(i for i in insts if i["account"] == pick)
        state = inst["state"]
        run = state.get("run") or {}
        strat = state.get("strategy") or {}
        cfg = _tr.RaceConfig(symbol="NQ", contracts=int(strat.get("contracts") or 1),
                             entry=str(strat.get("entry") or "19:00"))
        account = inst["account"]
        port = int(_ENV.get("NT_BRIDGE_PORT") or 45973)
        try:                                    # the owner's clock; DST-aware for future dates
            local_tz = ZoneInfo((_ENV.get("VERTEX_LOCAL_TZ") or "Europe/Prague").strip())
        except Exception:  # noqa: BLE001
            local_tz = datetime.now().astimezone().tzinfo
        now = datetime.now(timezone.utc)

        def usd(x: float) -> str:
            """+$1,234 / −$1,234"""
            return f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"

        def signed(x: float) -> str:
            return f'<span style="color:{vx.pl_color(x)}">{usd(x)}</span>'

        def when(t_utc: datetime) -> str:
            """'Wed 19:00 New York (Thu 01:00 your time)' -- 'now' if due."""
            if t_utc <= now + timedelta(seconds=30):
                return "now"
            e, loc = t_utc.astimezone(_lr.ET), t_utc.astimezone(local_tz)
            yours = f"{loc:%H:%M}" if loc.date() == e.date() else f"{loc:%a %H:%M}"
            return f"<b>{e:%a %H:%M}</b> New York ({yours} your time)"

        def at_ny(hhmm: str) -> str:
            h, m = (int(x) for x in hhmm.split(":"))
            e = now.astimezone(_lr.ET).replace(hour=h, minute=m, second=0, microsecond=0)
            return f"{hhmm} New York ({e.astimezone(local_tz):%H:%M} your time)"

        def local(iso: str | None) -> str:
            try:
                return f"{datetime.fromisoformat(iso).astimezone(local_tz):%a %d %b %H:%M}"
            except (TypeError, ValueError):
                return "—"

        def on_day(iso: str | None, day: str) -> bool:
            try:
                return _lr.trading_day(datetime.fromisoformat(iso)) == day
            except (TypeError, ValueError):
                return False

        # --- NinjaTrader, through the bridge ---
        eng = _engine(account, port, 3.0)
        bridge_err, acc, pos = None, None, None
        try:
            acc = eng.account_info()
            ps = eng.get_positions(9242)                       # run_lucid_futures_funded.MAGIC
            managed = {str(k) for k in (run.get("tickets") or {})}
            pos = next((p for p in ps if str(p.ticket) in managed), None)
            # open in NinjaTrader but no longer a race (its result already booked): the bot closes it
            leftover = [p for p in ps if str(p.ticket) not in managed]
        except (_BridgeError, OSError, ValueError) as e:
            bridge_err = str(e)
            leftover = []
        no_account = bool(bridge_err and "not found" in bridge_err)

        # --- the funded account, in Lucid's terms ---
        realized = float(run.get("realized_usd", 0.0))
        paid = float(run.get("paid_usd", 0.0))
        balance = ANCHOR + realized - paid
        floor = min(float(run.get("peak_eod", ANCHOR)) - 2_000.0, 50_100.0)
        open_pnl = float(pos.profit) if pos is not None else 0.0
        payouts = run.get("payouts") or []
        k = min(len(payouts), len(WANT) - 1)
        today_real = realized - float(run.get("day_start_realized", realized))
        cycle = float(run.get("cycle_pnl", 0.0)) + today_real
        best = max(float(run.get("cycle_best_day", 0.0)), today_real)
        need = max(KEEP + WANT[k] - balance, best / 0.40 - cycle if best > 0 else 0.0, 500.0 - cycle)
        cycle_days = int(run.get("cycle_days", 0))
        room = balance + open_pnl - floor
        sess = _lr.session_state(now)
        today = _lr.trading_day(now)
        nxt = _tr.next_entry(now, cfg, run.get("race_day"))
        history = state.get("history") or []
        sim = bool(acc and acc.is_demo) or (bridge_err is not None and account.lower().startswith("sim"))
        trades = []
        tlog = inst["trades_path"]
        if tlog.exists():
            for raw in tlog.read_text(encoding="utf-8").splitlines():
                try:
                    if raw.strip():
                        trades.append(json.loads(raw))
                except ValueError:
                    pass
        run_no = int(state.get("runs_total") or 1)
        last_today = next((t for t in reversed(trades)
                           if t.get("run") == run_no and on_day(t.get("closed_ts"), today)), None)
        last_end = history[-1] if history else None
        ended_today = bool(last_end and on_day(last_end.get("end"), today))
        paid_today = next((p for p in reversed(payouts) if on_day(p.get("ts"), today)), None)
        # the road to the next payout; today joins the cycle at its close even when no race is left in it
        no_race_left = pos is None and _lr.trading_day(nxt) != today
        path = _funded_path(balance, float(run.get("peak_eod", ANCHOR)), cycle, best,
                            cycle_days + (1 if no_race_left else 0), len(payouts))
        pv_race = _fut.point_value(cfg.symbol) * cfg.contracts            # $ per point of the race
        target_bal = KEEP + WANT[k]                                       # balance the next payout needs

        # --- header ---
        if no_account:
            status, scol = f"SET UP {account.upper()}", vx.WARN
        elif bridge_err:
            status, scol = "NINJATRADER OFF", vx.DANGER
        elif state.get("halted"):
            status, scol = "STOPPED", vx.DANGER
        elif pos is not None:
            status, scol = "RACE RUNNING", vx.OK
        elif leftover:
            status, scol = "OLD POSITION OPEN", vx.WARN
        elif need <= 0:
            status, scol = "PAYOUT DUE", vx.OK
        else:
            status, scol = "WAITING", vx.INFO
        mode = "simulator" if sim else "REAL Lucid account"
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Funded bot &middot; {account}</div>'
            f'<div class="vx-sub">Goal: payouts, fast &mdash; $500 first, then $2,500 each'
            + (f' &middot; paper account {account} (internet prices, 10 min delay)</div></div>' if _PAPER else
               f' &middot; NinjaTrader {account} ({mode})</div></div>')
            + f'<div class="vx-spacer"></div>{vx.pill(status, scol)}</div>',
            unsafe_allow_html=True)

        # --- TODAY: one card that says what is happening ---
        def today_card(icon: str, title: str, body: str, color: str) -> None:
            st.markdown(
                f'<div style="background:{vx.SURFACE};border:1px solid {vx.BORDER};border-left:4px solid {color};'
                f'border-radius:12px;padding:18px 22px;margin:2px 0 12px">'
                f'<div style="font-size:11px;font-weight:600;color:{vx.TEXT_FAINT};letter-spacing:.8px;'
                f'text-transform:uppercase;margin-bottom:6px">Today</div>'
                f'<div style="font-size:24px;font-weight:650;color:{vx.TEXT};margin-bottom:8px">{icon}&nbsp; {title}</div>'
                f'<div style="font-size:14.5px;line-height:1.65;color:{vx.TEXT_MUTED}">{body}</div></div>',
                unsafe_allow_html=True)

        take_next = min(max(need, _F_TAKE_MIN), _F_TAKE_MAX)
        progress = None
        if no_account:
            today_card("⚙️", f"Create the {account} account in NinjaTrader",
                       f"The funded bot trades its own simulation account, <b>{account}</b>, so it never mixes"
                       " with the evaluation bot. In NinjaTrader:<br>"
                       "1. Control Center &rarr; <b>Accounts</b> tab &rarr; right-click in the empty area under the"
                       f" accounts &rarr; <b>New simulation account...</b> &rarr; name it <b>{account}</b> &rarr; OK.<br>"
                       "2. <b>Connections</b> menu: disconnect and connect again &mdash; NinjaTrader switches a new"
                       " account on at the next connect.<br>"
                       "The bot is already running and starts on its own within a minute.", vx.WARN)
        elif bridge_err:
            today_card(*_nt_down_card(), vx.DANGER)
        elif state.get("halted"):
            today_card("⏹️", "The bot has stopped", str(state.get("halt_reason") or ""), vx.DANGER)
        elif pos is not None:
            entry, target, cur = float(pos.price_open), float(pos.tp), float(pos.price_current)
            info = next(iter((run.get("tickets") or {}).values()), {})
            sg = -1.0 if info.get("direction") == "short" else 1.0
            progress = max(0.0, min(1.0, sg * (cur - entry) / max(sg * (target - entry), 1e-9)))
            take = float(info.get("take_usd") or 0.0)
            today_card("🏁", f"Race running &mdash; aiming for +${take:,.0f}" if take else "Race running",
                       _race_words(sg, int(pos.volume), cfg.symbol, entry, target, cur)
                       + f" &mdash; what the next payout still needs, at most ${_F_TAKE_MAX:,.0f} a day."
                       + _side_reason(info) + "<br>"
                       f"If neither the take-profit nor the loss limit is reached, the bot closes the trade at"
                       f" {at_ny(cfg.exit)}.", vx.OK)
        elif paid_today is not None:
            amt = float(paid_today.get("amount_usd") or 0.0)
            how = " On the simulator it is counted, not paid." if sim else " Request it in Lucid's dashboard."
            today_card("💰", f"Payout #{paid_today.get('n')} taken: ${amt:,.0f}",
                       f"You would receive <b>${SPLIT * amt:,.0f}</b> (Lucid keeps 10%).{how}"
                       f" Next race: {when(nxt)}.", vx.OK)
        elif ended_today and last_end.get("outcome") == "blown":
            more = (f" On the simulator a new $50,000 funded account starts with the next race: {when(nxt)}."
                    if sim else "")
            today_card("💥", "Loss limit hit &mdash; funded account blown",
                       f"Funded account #{last_end.get('run')} ended today after"
                       f" {int(last_end.get('payouts') or 0)} payout(s).{more}", vx.DANGER)
        elif need <= 0:
            amt = min(CAPS[k], balance - KEEP)
            today_card("⏸️", "Payout money reached &mdash; waiting for the 3-day minimum",
                       f"This payout cycle has <b>{cycle_days} of 3</b> trading days. The bot does not trade until"
                       f" then, so nothing earned can be lost. The payout of about <b>${amt:,.0f}</b> is taken at"
                       " the close of the cycle's third day.", vx.OK)
        elif last_today is not None:
            pnl = float(last_today.get("pnl_usd") or 0.0)
            today_card("✅" if pnl > 0 else "⏸️", f"Today's race is finished: {usd(pnl)}",
                       f"The next payout still needs <b>${need:,.0f}</b>. Next race: {when(nxt)}, aiming for"
                       f" +${take_next:,.0f}.", vx.OK if pnl > 0 else vx.INFO)
        elif nxt <= now + timedelta(seconds=30):
            today_card("⏳", "Starting today's race", "The bot is placing the order right now.", vx.INFO)
        else:
            closed = " The market is closed right now." if sess == _lr.Session.CLOSED else ""
            today_card("💤", "Waiting for the next race",
                       f"Next race: {when(nxt)}, aiming for +${take_next:,.0f}.{closed}", vx.INFO)
        if progress is not None:
            st.progress(progress, text=f"{progress:.0%} of the way from the entry price to the take-profit")
        if leftover:
            nt_text, _ = _nt_status()
            fix = (" NinjaTrader has <b>no market connection</b> right now, so it refuses the close &mdash; in"
                   " NinjaTrader's Control Center open <b>Connections</b> and connect <b>Simulation</b> again."
                   if "no market connection" in nt_text else "")
            lp = ", ".join(f"{int(p.volume)} {cfg.symbol} at {float(p.price_open):,.2f} (now {usd(float(p.profit))})"
                           for p in leftover)
            st.markdown(vx.banner(
                f"⚠️ <b>A position from an earlier race is still open in NinjaTrader:</b> {lp}. Its result is"
                f" already booked as Lucid would have scored it, so it does not count for this account. The bot"
                f" closes it as soon as NinjaTrader accepts the order and starts no new race before that.{fix}",
                "warn"), unsafe_allow_html=True)

        # --- the next payout, as plain figures ---
        k_ = st.columns(4)
        k_[0].markdown(vx.card("Account balance", f"${balance:,.0f}",
                               note=f"{usd(balance - ANCHOR)} vs the $50,000 start"
                                    + (f", after ${paid:,.0f} paid out" if paid else "")),
                       unsafe_allow_html=True)
        if need > 0:
            k_[1].markdown(vx.card(f"Next payout (${WANT[k]:,.0f}+)", f"${need:,.0f} to go",
                                   note="profit still needed, Lucid's rules included"), unsafe_allow_html=True)
        else:
            k_[1].markdown(vx.card("Next payout", f"${min(CAPS[k], balance - KEEP):,.0f}",
                                   note="ready once the cycle has 3 days"), unsafe_allow_html=True)
        k_[2].markdown(vx.card("Payout cycle", f"{cycle_days} of 3 days",
                               note=f"profit {usd(cycle)}; best day {usd(best)} may be at most 40% of it"),
                       unsafe_allow_html=True)
        k_[3].markdown(vx.card("Can still lose", f"${max(0.0, room):,.0f}",
                               note="before the account is blown &mdash; open trade included"),
                       unsafe_allow_html=True)

        # --- the details, in tabs (the Today card and the numbers above stay in view) ---
        t_live, t_next, t_charts, t_res, t_how = st.tabs([":material/monitoring: Live race",
                                                         ":material/flag: Next payout",
                                                         ":material/candlestick_chart: Trade charts",
                                                         ":material/payments: Payouts & accounts",
                                                         ":material/help: How it works"])
        with t_live:
            # --- the open race, live ---
            if pos is not None:
                info = next(iter((run.get("tickets") or {}).values()), {})
                opened = (pd.Timestamp(info["opened_ts"]) if info.get("opened_ts")
                          else pd.Timestamp(now) - pd.Timedelta(hours=1))
                close_at = pd.Timestamp(_tr.exit_time(opened.to_pydatetime(), cfg)).tz_convert(local_tz)
                try:
                    b, mins = _bars_around(cfg.symbol, opened, pd.Timestamp(now), account, port)
                    fig = _trade_figure(b, mins, local_tz, entry_t=opened.tz_convert(local_tz),
                                        entry_px=float(pos.price_open), tp=float(pos.tp), line=info.get("line"),
                                        live_t=pd.Timestamp(now).tz_convert(local_tz), live_px=float(pos.price_current),
                                        close_at=close_at, height=340, title=f"Live: {cfg.symbol} since the entry (your time)",
                                        tp_label="Take-profit", line_label="Loss limit",
                                        long=info.get("direction", "long") != "short")
                    st.plotly_chart(fig, width="stretch", key="funded_live", config={"displayModeBar": False})
                except Exception as e:  # noqa: BLE001 -- no bars yet / bridge hiccup
                    st.caption(f"Live chart unavailable right now ({e}).")
                # --- the race in numbers ---
                entry, target, cur = float(pos.price_open), float(pos.tp), float(pos.price_current)
                sg = -1.0 if info.get("direction") == "short" else 1.0
                pv_pos = _fut.point_value(cfg.symbol) * float(pos.volume)
                to_tp = sg * (target - cur)
                rows_r = [
                    {"": "Sold short at" if sg < 0 else "Bought at",
                     "Price": f"{entry:,.2f}", "Points from entry": "—",
                     "Account": f"{int(pos.volume)} {cfg.symbol}, opened {opened.tz_convert(local_tz):%a %H:%M} your time"},
                    {"": "Take-profit", "Price": f"{target:,.2f}", "Points from entry": f"{sg * (target - entry):+,.2f}",
                     "Account": f"{usd(sg * (target - entry) * pv_pos)} if reached"},
                ]
                if info.get("line"):
                    ln = float(info["line"])
                    to_ln = sg * (cur - ln)
                    rows_r.append({"": "Loss line (account blown)", "Price": f"{ln:,.2f}",
                                   "Points from entry": f"{-abs(entry - ln):+,.2f}",
                                   "Account": f"{usd(-abs(entry - ln) * pv_pos)} if reached"})
                else:
                    to_ln = None
                rows_r.append({"": "Now", "Price": f"{cur:,.2f}", "Points from entry": f"{sg * (cur - entry):+,.2f}",
                               "Account": f"{usd(open_pnl)} open"})
                st.dataframe(pd.DataFrame(rows_r), width="stretch", hide_index=True)
                chance_now = (f" Rough chance it reaches the take-profit first, from here: <b>{to_ln / (to_ln + to_tp):.0%}</b>"
                              " (distance to the loss line ÷ both distances &mdash; what a market with no direction"
                              " gives)." if to_ln is not None and to_ln > 0 and to_tp > 0 else "")
                st.markdown(vx.banner(f"<b>{max(0.0, to_tp):,.2f} points</b> to the take-profit"
                                      + (f", <b>{max(0.0, to_ln):,.2f} points</b> to the loss line" if to_ln is not None else "")
                                      + f". Closed at the latest at <b>{close_at:%a %H:%M}</b> your time (16:00 New York)."
                                      + chance_now, "info"), unsafe_allow_html=True)
            else:
                plan = ""
                if need > 0 and not state.get("halted"):
                    room_next = balance - floor
                    plan = (f"<br>It will aim for <b>+${take_next:,.0f}</b> (about {take_next / pv_race:,.1f} points)."
                            f" The loss line will be about <b>{room_next / pv_race:,.1f} points</b> away"
                            f" (${room_next:,.0f}) &mdash; rough chance of a win"
                            f" <b>{room_next / (room_next + take_next):.0%}</b>.")
                elif need <= 0:
                    plan = "<br>No race until the payout cycle has its 3 days &mdash; see the <b>Next payout</b> tab."
                st.markdown(vx.banner(f"No race is open right now. Next race: <b>{when(nxt)}</b>.{plan} Its live"
                                      " chart appears here the moment it starts.", "info"),
                            unsafe_allow_html=True)

        with t_next:
            # --- Lucid's payout rules, one by one, and the road to the payout ---
            n_pay = len(payouts) + 1
            st.markdown(vx.section(f"What Lucid needs before payout #{n_pay} (${WANT[k]:,.0f} or more)"),
                        unsafe_allow_html=True)
            rule40 = cycle > 0 and best <= 0.40 * cycle
            checks = [
                (balance >= target_bal,
                 f"Balance at least <b>${target_bal:,.0f}</b> &mdash; $52,100 has to stay in the account",
                 f"now ${balance:,.0f}" + ("" if balance >= target_bal else f" &mdash; ${target_bal - balance:,.0f} to go")),
                (cycle_days >= 3, "At least <b>3 trading days</b> since the last payout (or since the start)",
                 f"{cycle_days} so far &mdash; a day counts at its 16:00 New York close"),
                (cycle >= 500, "At least <b>$500 profit</b> in those days", f"now {usd(cycle)}"),
                (rule40, "No single day above <b>40%</b> of that profit",
                 ("no profit yet" if cycle <= 0 and best <= 0 else
                  f"best day {usd(best)}" + (f" = {best / cycle:.0%} of the profit" if cycle > 0 else "")
                  + ("" if rule40 else f" &mdash; the profit must reach ${best / 0.40:,.0f}"))),
            ]
            st.markdown("".join(
                f'<div style="display:flex;gap:14px;align-items:flex-start;padding:9px 2px;'
                f'border-bottom:1px solid {vx.BORDER_SOFT}"><div style="font-size:18px;line-height:1.3">'
                f'{"✅" if ok else "⏳"}</div><div><div style="color:{vx.TEXT};font-size:14.5px">{what}</div>'
                f'<div style="color:{vx.TEXT_MUTED};font-size:13px">{now_}</div></div></div>'
                for ok, what, now_ in checks), unsafe_allow_html=True)

            st.markdown(vx.section("The road there, if every race wins"), unsafe_allow_html=True)
            steps = path["steps"]
            start_day = pd.Timestamp(today if pos is not None else _lr.trading_day(nxt))
            dates = pd.bdate_range(start_day, periods=len(steps)) if steps else []
            if steps:
                st.dataframe(pd.DataFrame([{
                    "Trading day": f"{d:%a %d %b}",
                    "Race": ((("today's race, running" if i == 0 and pos is not None else "race")
                              + f" — aims for +${s['take']:,.0f}") if s["race"] else "no race — waiting for the 3rd day"),
                    "Balance if it wins": f"${s['balance']:,.0f}",
                    "Loss line after": f"${s['line']:,.0f}",
                    "Chance this race wins": f"{s['p']:.0%}" if s["race"] else "—",
                    "Chance to get this far": f"{s['chance']:.0%}",
                } for i, (s, d) in enumerate(zip(steps, dates))]), width="stretch", hide_index=True)
            pay_day = dates[-1] if len(dates) else pd.Timestamp(today)
            amt_next = path["amount"]
            st.markdown(vx.banner(
                f"Then <b>payout #{n_pay}: about ${amt_next:,.0f}</b> (${SPLIT * amt_next:,.0f} to you), taken at the"
                f" close of <b>{pay_day:%a %d %b}</b>. Rough chance from here: <b>{path['chance']:.0%}</b>.", "ok"),
                unsafe_allow_html=True)
            later = _funded_path(KEEP, KEEP, 0.0, 0.0, 0, n_pay)
            st.caption(
                f"Each race starts at {at_ny(cfg.entry)} on the evening before its trading day; holidays are not"
                " counted. Rough chance = distance to the loss line ÷ (that + distance to the take-profit), what a"
                " market with no direction gives; the minute-by-minute replay of 2021-2026 gave 34-36% for a fresh"
                f" account's first payout. After payout #{n_pay} the loss line stays at $50,100 for good and each"
                f" later payout is $2,500: {sum(1 for s in later['steps'] if s['race'])} winning races,"
                f" rough chance {later['chance']:.0%} each time. Five payouts in all, then Lucid reviews the"
                " account for live trading.")

        with t_charts:
            # --- every closed trade, as a chart (newest first) ---
            done_ = [t for t in trades if t.get("opened_ts") and t.get("closed_ts")][::-1]

            def _trade_chart(t: dict, key: str, height: int = 290) -> None:
                o, c = pd.Timestamp(t["opened_ts"]), pd.Timestamp(t["closed_ts"])
                pnl = float(t.get("pnl_usd") or 0.0)
                ex_ = float(t["exit_price"])
                why = _FUNDED_REASON.get(t.get("reason") or "", "closed")
                ol, cl = o.tz_convert(local_tz), c.tz_convert(local_tz)
                span = (f"{ol:%a %d %b %H:%M} → {cl:%H:%M}" if ol.date() == cl.date()
                        else f"{ol:%a %d %b %H:%M} → {cl:%a %H:%M}")
                title = f"#{t.get('run')} · {span} · {usd(pnl)} · {why}"
                try:
                    b, mins = _closed_trade_bars(t["sym"], t["opened_ts"], t["closed_ts"], account, port)
                    fig = _trade_figure(b, mins, local_tz, entry_t=ol, entry_px=float(t["entry_price"]), tp=t.get("tp"),
                                        line=t.get("line"), exit_t=cl, exit_px=ex_,
                                        exit_text=f"Sold {ex_:,.2f} · {usd(pnl)}", exit_color=vx.pl_color(pnl),
                                        height=height, title=title, tp_label="Take-profit", line_label="Loss limit",
                                        long=t.get("direction", "long") != "short")
                    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
                except Exception as e:  # noqa: BLE001 -- NinjaTrader closed / no history
                    st.caption(f"{title} -- chart unavailable ({e}).")
                _brain_caption("Funded", t)

            if not done_:
                st.caption("No closed trades yet -- each one appears here as a chart.")
            else:
                latest = done_[:12]
                for i in range(0, len(latest), 2):
                    cols_ = st.columns(2)
                    for j, (col_, t) in enumerate(zip(cols_, latest[i:i + 2])):
                        with col_:
                            _trade_chart(t, key=f"funded_trade_{i + j}")
                if len(done_) > 12:
                    older = done_[12:]
                    pick = st.selectbox(
                        f"Older trades ({len(older)})", range(len(older)),
                        format_func=lambda n: (f"{pd.Timestamp(older[n]['opened_ts']).tz_convert(local_tz):%a %d %b %H:%M}"
                                               f" · {usd(float(older[n].get('pnl_usd') or 0))}"),
                        key="funded_older_pick")
                    _trade_chart(older[pick], key="funded_trade_older", height=320)

        with t_res:
            # --- results ---
            st.markdown(vx.section("Results so far"), unsafe_allow_html=True)
            paid_total = float(state.get("paid_total_usd") or 0.0)
            blown_unpaid = sum(1 for h in history if h.get("outcome") == "blown" and not h.get("payouts"))
            r = st.columns(4)
            r[0].markdown(vx.card("Funded accounts", f"{run_no}",
                                  note="started on the simulator" if sim else "on this Lucid account"),
                          unsafe_allow_html=True)
            r[1].markdown(vx.card("Got a payout", f"{int(state.get('runs_paid') or 0)}",
                                  note=f"{int(state.get('payouts_total') or 0)} payouts in total"),
                          unsafe_allow_html=True)
            r[2].markdown(vx.card("Blown before a payout", f"{blown_unpaid}"), unsafe_allow_html=True)
            r[3].markdown(vx.card("Paid out to you", f"${SPLIT * paid_total:,.0f}",
                                  note=f"90% of ${paid_total:,.0f}" + (" &mdash; virtual" if sim else "")),
                          unsafe_allow_html=True)
            reasons = [str(t.get("reason") or "") for t in trades]
            n_races = len(reasons)
            n_tp = sum(1 for x in reasons if x in ("tp", "take_while_away"))
            n_ml = sum(1 for x in reasons if x.startswith("mll"))
            n_16 = sum(1 for x in reasons if x in ("time_exit", "exit_while_away"))
            r2 = st.columns(4)
            r2[0].markdown(vx.card("Races", f"{n_races}", note="one a day, all funded accounts together"),
                           unsafe_allow_html=True)
            r2[1].markdown(vx.card("Take-profit reached", f"{n_tp}",
                                   note=(f"{n_tp / n_races:.0%} of races &mdash; " if n_races else "")
                                        + "about 2 in 3 expected"), unsafe_allow_html=True)
            r2[2].markdown(vx.card("Loss line hit", f"{n_ml}", note="each one ends a funded account"),
                           unsafe_allow_html=True)
            r2[3].markdown(vx.card("Closed at 16:00", f"{n_16}", note="neither level reached that day"),
                           unsafe_allow_html=True)
            st.caption("Expected, from the 12-month replay of your plan on real NQ 2021-2025 with the $1,500 limit:"
                       " about 30 in 100 funded accounts take a payout and about 7 in 10 are blown first; a payout"
                       " averages about $1,800 to you. A single race reaches a +$1,500 take-profit a bit more than"
                       " half the time — the first payout needs about 3 wins (then about $1,650, not $500).")

            # --- one account's balance, trade by trade ---
            st.markdown(vx.section("Account balance, trade by trade"), unsafe_allow_html=True)
            runs_ = sorted({int(t.get("run") or 0) for t in trades if t.get("run")}
                           | {int(h.get("run") or 0) for h in history if h.get("run")} | {run_no}, reverse=True)
            pick_run = st.selectbox("Funded account", runs_, key="funded_curve_pick",
                                    format_func=lambda n: f"#{n}" + (" (now)" if n == run_no else ""))
            if pick_run == run_no:
                pays, start_iso = payouts, run.get("start")
            else:
                h_ = next((h for h in history if h.get("run") == pick_run), {})
                pays, start_iso = h_.get("payout_list") or [], h_.get("start")
            cv = _funded_curve([t for t in trades if t.get("run") == pick_run], pays, start_iso)
            if pick_run == run_no and len(cv):
                cv = pd.concat([cv, pd.DataFrame([{"t": pd.Timestamp(now), "balance": cv["balance"].iloc[-1],
                                                   "line": cv["line"].iloc[-1], "what": "now", "amount": 0.0}])],
                               ignore_index=True)
            if len(cv):
                tl = pd.to_datetime(cv["t"], utc=True).dt.tz_convert(local_tz)
                label = {"start": "Start", "trade": "Race", "payout": "Payout", "now": "Now"}
                hover = [f"{label.get(w, w)} {tt:%a %d %b %H:%M}" + (f": {usd(a)}" if w in ("trade", "payout") else "")
                         + f"<br>balance ${b_:,.0f} · loss line ${l_:,.0f}"
                         for w, tt, a, b_, l_ in zip(cv["what"], tl, cv["amount"], cv["balance"], cv["line"])]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tl, y=cv["balance"], name="Balance", mode="lines+markers",
                                         line=dict(color=vx.ACCENT, width=2, shape="hv"), marker=dict(size=8),
                                         text=hover, hoverinfo="text"))
                fig.add_trace(go.Scatter(x=tl, y=cv["line"], name="Loss line", mode="lines",
                                         line=dict(color=vx.DANGER, width=2, dash="dash", shape="hv"), hoverinfo="skip"))
                if pick_run == run_no:
                    fig.add_hline(y=target_bal, line=dict(color=vx.OK, width=1.5, dash="dot"),
                                  annotation_text=f"payout #{len(payouts) + 1} needs ${target_bal:,.0f}",
                                  annotation_position="top left", annotation_font_color=vx.TEXT_MUTED)
                fig.add_hline(y=ANCHOR, line=dict(color=vx.TEXT_FAINT, width=1),
                              annotation_text="start $50,000", annotation_position="bottom left",
                              annotation_font_color=vx.TEXT_FAINT)
                fig.update_layout(**vx.plotly_layout(height=300))
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0,
                                                               font=dict(color=vx.TEXT_MUTED)))
                fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                st.plotly_chart(fig, width="stretch", key="funded_curve", config={"displayModeBar": False})
            if not (len(cv) and (cv["what"] == "trade").any()):
                st.caption(f"Funded account #{pick_run} has no closed race yet — its line grows with every race.")
            every = [(h.get("run"), p) for h in history for p in (h.get("payout_list") or [])]
            every += [(run_no, p) for p in payouts]
            if every:
                st.markdown(vx.section("Payouts"), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{
                    "Taken (your time)": local(p.get("ts")), "Funded account": f"#{rn}", "Payout": f"#{p.get('n')}",
                    "Amount": f"${float(p.get('amount_usd') or 0):,.0f}",
                    "To you (90%)": f"${float(p.get('to_you_usd') or 0):,.0f}",
                    "Cycle": f"{p.get('cycle_days')} days, {usd(float(p.get('cycle_pnl') or 0))}",
                } for rn, p in reversed(every)]), width="stretch", hide_index=True)
            if history:
                st.markdown(vx.section("Finished funded accounts"), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{
                    "Funded account": f"#{h.get('run')}", "Started": local(h.get("start")), "Ended": local(h.get("end")),
                    "Result": "💥 Blown" if h.get("outcome") == "blown" else "🏁 5 payouts",
                    "Payouts": int(h.get("payouts") or 0), "Paid": f"${float(h.get('paid_usd') or 0):,.0f}",
                    "Trading days": h.get("days"),
                } for h in reversed(history[-30:])]), width="stretch", hide_index=True)

        with t_how:
            # --- how it works, in plain words ---
            with st.expander("How the bot works", expanded=True):
                st.markdown(
                    f"1. Every trading day at **{at_ny(cfg.entry)}** the bot buys **{cfg.contracts} {cfg.symbol}**"
                    " (Nasdaq-100 futures) when NQ is above its 200-day average, or **sells short** when it is"
                    " below &mdash; one trade a day, never more.\n"
                    "2. At the same moment it places a closing order at the price where the account has made what the"
                    " **next payout still needs** &mdash; at most **$1,500** a day. Lucid pays only if no single"
                    " day is more than 40% of a payout cycle's profit, so after a big day the bot races on for"
                    " what the cycle still needs &mdash; the payout waits a little and is bigger.\n"
                    f"3. If that price is not reached, the bot closes the trade at **{at_ny(cfg.exit)}**.\n"
                    "4. The account is **blown** the moment its value &mdash; open trade included &mdash; falls"
                    " $2,000 below its best daily close. That line stops moving at $50,100 once a daily close"
                    " reaches $52,100.\n"
                    "5. **Payouts:** the first is **$500** &mdash; the balance must reach $52,600, because $52,100 has"
                    " to stay in the account. Every payout also needs at least 3 trading days and at least $500 of"
                    " profit in its cycle, with no day above 40% of it. Later payouts are **$2,500** each; after"
                    " five, Lucid reviews the account for live trading. You keep 90%.\n"
                    "6. Once the money for a payout is there, the bot **stops trading** until the 3 days have"
                    " passed, so it cannot lose what is earned.\n"
                    "7. On the simulator payouts are counted, not paid, and a blown account is replaced by a new"
                    " one at the next session. On a real Lucid account the bot tells you when to request a"
                    " payout, and stops if the account is blown.\n\n"
                    "Expected (minute-by-minute replay of 2021-2026): about **35 in 100** funded accounts get a"
                    " payout (median 4 trading days); about **2 in 3** are blown first. Trading more often does not"
                    " help: Lucid needs at least 3 days per payout and no day above 40% of it, which one race a day"
                    " already reaches. It is a race with a capped cost &mdash; the evaluation fee &mdash; not a"
                    " prediction of the market.")

            # --- technical details, folded away ---
            with st.expander("Technical details", expanded=False):
                et_now = now.astimezone(_lr.ET)
                rows_ = [
                    {"Item": "New York time", "Value": f"{et_now:%a %d %b %H:%M}"},
                    {"Item": "Lucid session", "Value": sess.value.replace("_", " ")},
                    {"Item": "Trade date", "Value": today},
                    {"Item": "Last race day (this account)", "Value": str(run.get("race_day") or "—")},
                    {"Item": "Race settings", "Value": f"{cfg.contracts} {cfg.symbol}, entries {cfg.entry}-"
                                                       f"{cfg.last_entry} ET, exit {cfg.exit} ET, safety stop "
                                                       f"{cfg.stop_points:g} pts, take $150-${_F_TAKE_MAX:,.0f}"},
                    {"Item": "Closed balance / loss limit", "Value": f"${balance:,.2f} / ${floor:,.2f}"},
                    {"Item": "Best end-of-day balance", "Value": f"${float(run.get('peak_eod', ANCHOR)):,.2f}"
                                                                 + (" (limit locked)" if run.get("mll_locked") else "")},
                    {"Item": "Payout cycle", "Value": f"{cycle_days} days, {usd(cycle)}, best day {usd(best)}"},
                ]
                if acc:
                    rows_ += [{"Item": "NinjaTrader account", "Value": f"{acc.login} ({acc.server})"},
                              {"Item": "NinjaTrader balance", "Value": f"${acc.balance:,.2f}"}]
                if pos is not None:
                    rows_ += [{"Item": "Open position", "Value": f"{pos.symbol} {pos.volume:g} @ {pos.price_open:,.2f}"
                                                                  f" | take {pos.tp:,.2f} | safety stop {pos.sl:,.2f}"}]
                if bridge_err:
                    rows_ += [{"Item": "Bridge", "Value": bridge_err}]
                st.dataframe(pd.DataFrame(rows_), width="stretch", hide_index=True)
                st.json(state, expanded=False)

    _race_funded_view()
    st.stop()


# LUCID_FUTURES is run_lucidpro_trial.py in futures mode (VERTEX_BROKER=
# ninjatrader): CME futures on Lucid through NinjaTrader 8. Same attempt/state
# shape as LUCIDPRO_TRIAL, but NOTHING here touches MetaTrader 5 -- positions
# and connection state come from the VertexBridge add-on, session state from
# lucid_rules -- so this page works on a machine where MT5 is not logged in.
if bot_choice in ("LUCID_FUTURES", "TRIAL"):
    from metatraderbot import lucid_rules as _lr
    from metatraderbot.nt_engine import BridgeError as _BridgeError
    from metatraderbot.nt_engine import NinjaTraderEngine as _NTEngine

    st.markdown(vx.CSS, unsafe_allow_html=True)
    refresh_s = _refresh_seconds(5)

    @st.fragment(run_every=refresh_s)
    def _lucid_futures_view():
        ANCHOR = 50_000.0
        TARGET_PCT, MLL_PCT, DLL_PCT = 0.06, 0.04, 0.024
        SLOTS = {9120: "6C", 9121: "6B", 9122: "6S", 9123: "MGC", 9124: "MNQ",
                 9125: "NQ"}   # run_lucidpro_trial.FUTURES_SYMBOL_MAGICS
        race = (_ENV.get("LUCID_TRIAL_STRATEGY") or "race").strip().lower() == "race"
        race_n = (_ENV.get("LUCID_RACE_CONTRACTS") or "4").strip()
        race_sym = (_ENV.get("LUCID_RACE_SYMBOL") or "NQ").strip().upper()
        strat_name = "Target race" if race else "ORB + trend"
        strat_desc = (f"Target race: {race_n} {race_sym} long once a day, limit at the target"
                      if race else "ORB + trend on 6 MNQ + 3 MGC")
        account = (_ENV.get("NT_ACCOUNT") or "Sim101").strip()
        port = int(_ENV.get("NT_BRIDGE_PORT") or 45973)

        path = ROOT / "data" / "state_lucid_futures.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            state = {}

        # ---------- NinjaTrader, through the bridge ----------
        eng = _engine(account, port, 3.0)
        bridge_err, acc, rows, acct_positions = None, None, [], {}
        try:
            acc = eng.account_info()
            for magic in SLOTS:
                for p in eng.get_positions(magic):
                    rows.append({
                        "Ticket": p.ticket, "Strategy": strat_name, "Symbol": p.symbol,
                        "Dir": UP if p.type == 0 else DOWN, "Volume": p.volume,
                        "Entry": p.price_open, "Current": p.price_current, "SL": p.sl, "TP": p.tp,
                        "Floating P&L": p.profit,
                        "Opened": datetime.fromtimestamp(p.time, tz=timezone.utc),
                    })
            acct_positions = eng.open_contracts()
        except (_BridgeError, OSError, ValueError) as e:
            bridge_err = str(e)
        open_df = pd.DataFrame(rows)
        n_open = len(rows)
        floating = float(sum(r["Floating P&L"] for r in rows))
        bot_contracts = {r["Symbol"] for r in rows}
        foreign = {k: v for k, v in acct_positions.items() if k not in bot_contracts}

        # ---------- Lucid's session ----------
        now = datetime.now(timezone.utc)
        sess = _lr.session_state(now)
        flat_by = _lr.flat_by(now.astimezone(_lr.ET).date())
        sess_label, sess_col = {
            _lr.Session.OPEN: ("SESSION OPEN", vx.OK),
            _lr.Session.NO_NEW_ENTRIES: ("NO NEW ENTRIES", vx.WARN),
            _lr.Session.FLATTEN: ("FLATTENING", vx.WARN),
            _lr.Session.CLOSED: ("SESSION CLOSED", vx.TEXT_FAINT),
        }[sess]
        if bridge_err:
            status, scol = "NO BRIDGE", vx.DANGER
        elif n_open:
            status, scol = "IN MARKET", vx.OK
        else:
            status, scol = "ARMED", vx.INFO

        who = f"{account} (simulator)" if (acc and acc.is_demo) else account
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Vertex Algo Bot</div>'
            f'<div class="vx-sub">Lucid Trial &middot; NinjaTrader {who} &middot; '
            f'{strat_desc} &middot; $50,000 evaluations</div></div>'
            f'<div class="vx-spacer"></div>{vx.pill(sess_label, sess_col)}&nbsp;{vx.pill(status, scol)}</div>',
            unsafe_allow_html=True)

        if bridge_err:
            st.markdown(vx.banner(
                "<b>NinjaTrader bridge unreachable.</b> Start NinjaTrader and make sure the"
                " VertexBridge add-on is compiled (NinjaScript Editor &rarr; F5). Figures below"
                " come from the last written state.", "warn"), unsafe_allow_html=True)
        if not _FUTURES_MODE:
            st.markdown(vx.banner(
                "The trial bot is running in <b>MT5 mode</b>. Set <code>VERTEX_BROKER=ninjatrader</code>"
                " in .env and restart it to trade Lucid futures.", "info"), unsafe_allow_html=True)
        if sess == _lr.Session.CLOSED:
            st.markdown(vx.banner(
                f"<b>Outside Lucid's trading hours.</b> Positions must be flat by {flat_by:%H:%M} ET;"
                " trading resumes at 18:00 ET, Sunday to Thursday.", "info"), unsafe_allow_html=True)
        elif sess in (_lr.Session.NO_NEW_ENTRIES, _lr.Session.FLATTEN):
            st.markdown(vx.banner(
                f"<b>End of Lucid's day.</b> No new entries; the bot closes everything before"
                f" {flat_by:%H:%M} ET.", "warn"), unsafe_allow_html=True)
        if foreign:
            st.markdown(vx.banner(
                "<b>The account holds positions the bot did not open:</b> "
                + ", ".join(f"{k} {v:+d}" for k, v in foreign.items())
                + ". They count toward Lucid's contract cap and block bot entries on those"
                " contracts. Close them in NinjaTrader if they are not yours.", "warn"),
                unsafe_allow_html=True)
        reading = (_ENV.get("LUCID_RULE_READING") or ("lenient" if race else "strict")).strip().lower()
        if race:
            st.markdown(vx.banner(
                "<b>Target race.</b> Replayed on LucidPro 50K: ~83% pass if Lucid judges its limits on the"
                " <b>closed balance</b>, ~39-43% if open positions count &mdash; ask Lucid support before"
                f" buying (docs/LUCIDPRO_PASS_RESEARCH.md). Simulator attempts here use the <b>{reading}</b>"
                " reading. Evaluation only &mdash; never on a funded account.", "info"),
                unsafe_allow_html=True)
        else:
            st.markdown(vx.banner(
                "ORB + trend: ~30-38% pass on LucidPro 50K in the replay"
                " (docs/LUCIDPRO_PASS_RESEARCH.md). Treat this simulator run as the test before paying"
                " for an evaluation.", "info"),
                unsafe_allow_html=True)

        # ---------- live attempt ----------
        att = state.get("attempt") or {}
        if att:
            a_real = float(att.get("realized_usd", 0.0))
            a_peak = float(att.get("peak_equity", ANCHOR))
            a_eq = ANCHOR + a_real + float(att.get("floating_usd", 0.0) or 0.0)
            mll_floor = a_peak - MLL_PCT * ANCHOR
            on_balance = reading == "lenient"
            room = (ANCHOR + a_real if on_balance else a_eq) - mll_floor
            room_frac = max(0.0, min(1.0, room / (MLL_PCT * ANCHOR)))
            target_eq = ANCHOR * (1 + TARGET_PCT)
            st.markdown(vx.section("Current attempt &mdash; live"), unsafe_allow_html=True)
            k = st.columns(4)
            k[0].markdown(vx.card("Attempt equity", f"${a_eq:,.0f}",
                                  f"{(a_eq - ANCHOR) / ANCHOR:+.2%} vs $50,000",
                                  vx.POS if a_eq >= ANCHOR else vx.NEG), unsafe_allow_html=True)
            k[1].markdown(vx.card("Room to MLL", f"${room:,.0f}",
                                  f"fails at ${mll_floor:,.0f} ({'closed balance' if on_balance else 'equity'})",
                                  vx.NEG if room_frac < 0.34 else (vx.WARN if room_frac < 0.67 else vx.OK)),
                          unsafe_allow_html=True)
            k[2].markdown(vx.card("To pass", f"${target_eq - a_eq:,.0f}", f"target ${target_eq:,.0f}",
                                  vx.TEXT_MUTED), unsafe_allow_html=True)
            k[3].markdown(vx.card("Trades this attempt", f"{int(att.get('total_entries', 0))}",
                                  f"{int(att.get('wins', 0))}W / {int(att.get('losses', 0))}L",
                                  vx.TEXT_MUTED, small=True), unsafe_allow_html=True)
            st.progress(room_frac, text=None)

        attempts = int(state.get("attempts", 0))
        passes, fails = int(state.get("passes", 0)), int(state.get("fails", 0))
        resolved = passes + fails
        st.markdown(vx.section("Evaluations"), unsafe_allow_html=True)
        c = st.columns(4)
        c[0].markdown(vx.card("Attempts resolved", f"{resolved}", f"{attempts} started", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[1].markdown(vx.card("Passed", f"{passes}", "evaluations cleared",
                              vx.POS if passes else vx.TEXT_MUTED), unsafe_allow_html=True)
        c[2].markdown(vx.card("Failed", f"{fails}", "hit the MLL",
                              vx.NEG if fails else vx.TEXT_MUTED), unsafe_allow_html=True)
        c[3].markdown(vx.card("Pass rate", f"{passes / resolved:.0%}" if resolved else "—",
                              "futures, measured live", vx.TEXT_MUTED), unsafe_allow_html=True)
        if state.get("halted"):
            st.markdown(vx.banner(f"<b>Trading halted</b> after the attempt {state['halted']}ed on a"
                                  " real Lucid account.", "warn"), unsafe_allow_html=True)
        if state.get("awaiting_reset"):
            st.markdown(vx.banner("<b>Attempt failed on the real Lucid account.</b> Reset it in Lucid's"
                                  " dashboard &mdash; the bot resumes racing by itself once the balance is"
                                  " back at $50,000.", "warn"), unsafe_allow_html=True)

        t_att, t_pos, t_trades, t_sys = st.tabs([":material/flag: Attempts", ":material/candlestick_chart: Positions",
                                                 ":material/receipt_long: Trades", ":material/settings: System"])

        with t_att:
            hist = state.get("history", []) or []
            if not hist:
                st.markdown(vx.banner("No resolved attempts yet.", "info"), unsafe_allow_html=True)
            else:
                h = pd.DataFrame(hist)
                cols = [c_ for c_ in ("outcome", "days", "trades", "wins", "losses", "final_pct",
                                      "opened_ts", "resolved_ts") if c_ in h.columns]
                st.dataframe(h[cols].iloc[::-1], width="stretch", hide_index=True)

        with t_pos:
            if bridge_err:
                st.markdown(vx.banner("NinjaTrader unreachable &mdash; cannot list positions.", "warn"),
                            unsafe_allow_html=True)
            elif open_df.empty:
                st.markdown(vx.banner("No open bot positions.", "info"), unsafe_allow_html=True)
            else:
                st.dataframe(
                    open_df.style.map(_color_pnl, subset=["Floating P&L"]).map(_color_dir, subset=["Dir"])
                    .format({"Entry": "{:,.5f}", "Current": "{:,.5f}", "SL": "{:,.5f}", "TP": "{:,.5f}",
                             "Floating P&L": "${:+,.2f}", "Volume": "{:.0f}"}),
                    width="stretch", hide_index=True)
                st.caption("Stops and targets are live orders in NinjaTrader; close a position there"
                           " (Positions tab) if you need to step in.")

        with t_trades:
            rows_ = []
            tp_ = ROOT / "data" / "lucid_futures_trades.jsonl"
            if tp_.exists():
                for line in tp_.read_text(encoding="utf-8").splitlines():
                    try:
                        if line.strip():
                            rows_.append(json.loads(line))
                    except ValueError:
                        pass
            st.markdown(vx.section("Closed trades"), unsafe_allow_html=True)
            t_ = _normalize_trades(pd.DataFrame(rows_)) if rows_ else pd.DataFrame()
            if t_.empty:
                st.caption("No closed trades yet.")
            else:
                view = t_.drop(columns=[c_ for c_ in ("Opened", "SL", "TP", "TF", "HoldH") if c_ in t_.columns])
                st.dataframe(
                    view.style.map(_result_color, subset=["Result"]).map(_color_pnl, subset=["P&L", "R"])
                    .map(_color_dir, subset=["Dir"])
                    .format({"Entry": "{:,.5f}", "Exit": "{:,.5f}", "R": "{:+.3f}", "P&L": "${:+,.2f}",
                             "Closed": lambda v: v.strftime("%Y-%m-%d %H:%M") if pd.notna(v) else ""}),
                    width="stretch", hide_index=True)
                render_trade_charts(pd.DataFrame(rows_), ANCHOR, key="futures")

        with t_sys:
            et_now = now.astimezone(_lr.ET)
            st.markdown(vx.section("Lucid session"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Item": "New York time", "Value": f"{et_now:%a %d %b %H:%M}"},
                {"Item": "Session", "Value": sess_label.title()},
                {"Item": "Flat by (today)", "Value": f"{flat_by:%H:%M} ET"},
                {"Item": "Trade date", "Value": _lr.trading_day(now)},
            ]), width="stretch", hide_index=True)
            st.markdown(vx.section("NinjaTrader"), unsafe_allow_html=True)
            if acc:
                st.dataframe(pd.DataFrame([
                    {"Item": "Account", "Value": acc.login},
                    {"Item": "Connection", "Value": acc.server},
                    {"Item": "Balance", "Value": f"${acc.balance:,.2f}"},
                    {"Item": "All positions", "Value": ", ".join(f"{k} {v:+d}" for k, v in acct_positions.items()) or "flat"},
                ]), width="stretch", hide_index=True)
            else:
                st.caption(bridge_err or "not connected")
            st.markdown(vx.section("Bot state"), unsafe_allow_html=True)
            st.json(state, expanded=False)

    # ------------------------------------------------------------------
    # TRIAL page for the TARGET RACE (the default trial strategy since
    # 2026-09-30). Written for anyone: one plain-language "Today" card, four
    # money figures, the results, and the technical detail folded away.
    # ------------------------------------------------------------------
    @st.fragment(run_every=refresh_s)
    def _race_trial_view():
        from metatraderbot import target_race as _tr

        ANCHOR, TARGET, MLL = 50_000.0, 3_000.0, 2_000.0
        cfg = _tr.RaceConfig(
            symbol=(_ENV.get("LUCID_RACE_SYMBOL") or "NQ").strip().upper(),
            contracts=int((_ENV.get("LUCID_RACE_CONTRACTS") or "2").strip()),
            entry=(_ENV.get("LUCID_RACE_ENTRY") or "19:00").strip())
        reading = (_ENV.get("LUCID_RULE_READING") or "lucid").strip().lower()
        account = (_ENV.get("NT_ACCOUNT") or "Sim101").strip()
        port = int(_ENV.get("NT_BRIDGE_PORT") or 45973)
        # the current strategy and the new one side by side (owner 2026-10-21: "one tab the current strategy and
        # one tab this new strategy and we will compare them"): which paper trial bot this page shows
        strat = list(_STRATEGIES)[0]
        if _compare_on():
            strat = st.segmented_control("Strategy", list(_STRATEGIES), default=strat, key="trial_strategy",
                                         label_visibility="collapsed") or strat
            st.caption(_STRATEGIES[strat]["what"])
        stag = _STRATEGIES[strat]["trial_tag"]
        account += stag.upper()
        try:                                    # the owner's clock; DST-aware for future dates
            local_tz = ZoneInfo((_ENV.get("VERTEX_LOCAL_TZ") or "Europe/Prague").strip())
        except Exception:  # noqa: BLE001
            local_tz = datetime.now().astimezone().tzinfo
        now = datetime.now(timezone.utc)

        def usd(x: float) -> str:
            """+$1,234 / −$1,234"""
            return f"{'+' if x >= 0 else '−'}${abs(x):,.0f}"

        def when(t_utc: datetime) -> str:
            """'Wed 19:00 New York (Thu 01:00 your time)' -- 'now' if due."""
            if t_utc <= now + timedelta(seconds=30):
                return "now"
            e, loc = t_utc.astimezone(_lr.ET), t_utc.astimezone(local_tz)
            yours = f"{loc:%H:%M}" if loc.date() == e.date() else f"{loc:%a %H:%M}"
            return f"<b>{e:%a %H:%M}</b> New York ({yours} your time)"

        def at_ny(hhmm: str) -> str:
            """Today's hh:mm New York time, with the local equivalent."""
            h, m = (int(x) for x in hhmm.split(":"))
            e = now.astimezone(_lr.ET).replace(hour=h, minute=m, second=0, microsecond=0)
            return f"{hhmm} New York ({e.astimezone(local_tz):%H:%M} your time)"

        path = ROOT / "data" / f"state_lucid_futures{stag}.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            state = {}
        att = state.get("attempt") or {}

        # --- NinjaTrader, through the bridge ---
        eng = _engine(account, port, 3.0)
        bridge_err, acc, pos = None, None, None
        try:
            acc = eng.account_info()
            for magic in (9125, 9124):              # race: NQ (or MNQ if configured)
                ps = eng.get_positions(magic)
                if ps:
                    pos = ps[0]
                    break
        except (_BridgeError, OSError, ValueError) as e:
            bridge_err = str(e)
        if pos is not None:                         # a race whose entry is its own, not the fill's
            _vi = (att.get("tickets") or {}).get(str(pos.ticket)) or {}
            if _vi.get("virtual_entry"):
                import dataclasses as _dc
                _e = float(_vi["entry_price"])
                _sg = -1.0 if _vi.get("direction") == "short" else 1.0
                pos = _dc.replace(pos, price_open=_e,
                                  profit=_sg * (float(pos.price_current) - _e) * float(pos.volume) * 20.0)

        # --- the numbers, in the evaluation's terms ---
        realized = float(att.get("realized_usd", 0.0))
        peak = float(att.get("peak_equity", ANCHOR))
        balance = ANCHOR + realized
        open_pnl = float(pos.profit) if pos is not None else 0.0
        fail_line = peak - MLL
        to_go = max(0.0, ANCHOR + TARGET - balance - open_pnl)
        sess = _lr.session_state(now)
        today = _lr.trading_day(now)
        history = state.get("history", []) or []
        passes, fails = int(state.get("passes", 0)), int(state.get("fails", 0))
        resolved = passes + fails
        last = history[-1] if history else None
        last_today = bool(last and last.get("resolved_ts")
                          and _lr.trading_day(datetime.fromisoformat(last["resolved_ts"])) == today)
        # the CURRENT attempt's race day (a new attempt may race again the same day)
        done_day = (state.get("race_day")       # one race a day (2026-10-15): the day's race is done for every attempt
                    if (_ENV.get("LUCID_TRIAL_ONE_RACE_A_DAY") or "1").strip() != "0"
                    or state.get("race_attempt", passes + fails + 1) == passes + fails + 1 else None)
        nxt = _tr.next_entry(now, cfg, done_day)
        sim = bool(acc and acc.is_demo) or (bridge_err is not None and account.lower().startswith("sim"))

        # --- header ---
        if bridge_err:
            status, scol = "NINJATRADER OFF", vx.DANGER
        elif state.get("halted") == "pass":
            status, scol = "PASSED - STOPPED", vx.OK
        elif state.get("awaiting_reset"):
            status, scol = "WAITING FOR RESET", vx.DANGER
        elif pos is not None:
            status, scol = "RACE RUNNING", vx.OK
        else:
            status, scol = "WAITING", vx.INFO
        mode = "simulator" if sim else "REAL Lucid account"
        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Evaluation bot &middot; Lucid 50K</div>'
            f'<div class="vx-sub">Goal: +$3,000 before the account value drops $2,000 below its best close'
            + (f' &middot; paper account {account} (internet prices, 10 min delay)</div></div>' if _PAPER else
               f' &middot; NinjaTrader {account} ({mode})</div></div>')
            + f'<div class="vx-spacer"></div>{vx.pill(status, scol)}</div>',
            unsafe_allow_html=True)

        # --- TODAY: one card that says what is happening ---
        def today_card(icon: str, title: str, body: str, color: str) -> None:
            st.markdown(
                f'<div style="background:{vx.SURFACE};border:1px solid {vx.BORDER};border-left:4px solid {color};'
                f'border-radius:12px;padding:18px 22px;margin:2px 0 12px">'
                f'<div style="font-size:11px;font-weight:600;color:{vx.TEXT_FAINT};letter-spacing:.8px;'
                f'text-transform:uppercase;margin-bottom:6px">Today</div>'
                f'<div style="font-size:24px;font-weight:650;color:{vx.TEXT};margin-bottom:8px">{icon}&nbsp; {title}</div>'
                f'<div style="font-size:14.5px;line-height:1.65;color:{vx.TEXT_MUTED}">{body}</div></div>',
                unsafe_allow_html=True)

        progress = None
        if bridge_err:
            today_card(*_nt_down_card(), vx.DANGER)
        elif state.get("halted") == "pass":
            today_card("🎉", "Evaluation passed", "The bot has stopped trading this account."
                       " Next step: the funded account.", vx.OK)
        elif state.get("awaiting_reset"):
            today_card("❌", "Evaluation failed &mdash; reset the account",
                       "Reset it in Lucid's dashboard. The bot starts racing again by itself as soon as the"
                       " balance is back at $50,000.", vx.DANGER)
        elif pos is not None:
            entry, target, cur = float(pos.price_open), float(pos.tp), float(pos.price_current)
            _ri = (att.get("tickets") or {}).get(str(pos.ticket)) or {}
            sg = -1.0 if _ri.get("direction") == "short" else 1.0
            progress = max(0.0, min(1.0, sg * (cur - entry) / max(sg * (target - entry), 1e-9)))
            today_card("🏁", f"Race running &mdash; ${to_go:,.0f} still to go",
                       _race_words(sg, int(pos.volume), cfg.symbol, entry, target, cur)
                       + ", which passes the evaluation." + _side_reason(_ri) + "<br>"
                       f"If the target is not reached, the bot closes the trade at {at_ny(cfg.exit)} and tries"
                       f" again in the next session.", vx.OK)
        elif last_today and last.get("outcome") == "pass":
            more = (f" On the simulator a new evaluation starts with the next race: {when(nxt)}."
                    if sim else "")
            today_card("✅", "Target reached &mdash; evaluation passed",
                       f"Evaluation #{resolved} passed today.{more}", vx.OK)
        elif last_today and last.get("outcome") == "fail":
            more = (f" On the simulator a new evaluation starts with the next race: {when(nxt)}."
                    if sim else "")
            today_card("❌", "Loss limit hit &mdash; evaluation failed",
                       f"Evaluation #{resolved} ended today: the open loss reached $2,000.{more}", vx.DANGER)
        elif done_day == today:
            today_card("⏸️", "Today's race is finished",
                       f"The evaluation is still going: ${to_go:,.0f} still needed. Next race: {when(nxt)}"
                       f" &mdash; it will aim for the full amount still missing.", vx.INFO)
        elif nxt <= now + timedelta(seconds=30):
            today_card("⏳", "Starting today's race", "The bot is placing the order right now.", vx.INFO)
        else:
            closed = " The market is closed right now." if sess == _lr.Session.CLOSED else ""
            today_card("💤", "Waiting for the next race", f"Next race: {when(nxt)}.{closed}", vx.INFO)
        if progress is not None:
            st.progress(progress, text=f"{progress:.0%} of the way from the entry price to the target")

        # --- where the evaluation stands, as distances (nothing to decode) ---
        lenient = reading == "lenient"
        room = balance - fail_line + (0.0 if lenient else open_pnl)

        def signed(x: float) -> str:
            return f'<span style="color:{vx.pl_color(x)}">{usd(x)}</span>'

        k = st.columns(4)
        k[0].markdown(vx.card("Made so far", signed(realized), note="this evaluation, closed trades"),
                      unsafe_allow_html=True)
        k[1].markdown(vx.card("Open trade", signed(open_pnl),
                              note="profit right now" if pos is not None else "no trade open"),
                      unsafe_allow_html=True)
        k[2].markdown(vx.card("Still needed to pass", f"${to_go:,.0f}", note="the target is +$3,000"),
                      unsafe_allow_html=True)
        k[3].markdown(vx.card("Can still lose", f"${max(0.0, room):,.0f}",
                              note=("before it fails &mdash; counted when a day closes" if lenient
                                    else "before it fails &mdash; open trade included")),
                      unsafe_allow_html=True)

        # --- the details, in tabs (the Today card and the numbers above stay in view) ---
        t_live, t_charts, t_res, t_how = st.tabs([":material/monitoring: Live race",
                                                 ":material/candlestick_chart: Trade charts",
                                                 ":material/history: Evaluations", ":material/help: How it works"])
        with t_live:
            # --- the open race, live ---
            if pos is not None:
                info = next((t for t in (att.get("tickets") or {}).values()), {})
                entry_px, target_px, cur = float(pos.price_open), float(pos.tp), float(pos.price_current)
                line_px = info.get("line")
                if line_px is None:                     # tickets opened before the line was recorded
                    from metatraderbot import futures as _fut
                    pv = _fut.point_value(cfg.symbol)
                    line_px = entry_px - (-1.0 if info.get("direction") == "short" else 1.0) * (
                        ANCHOR + realized - fail_line) / (max(float(pos.volume), 1.0) * pv)
                opened = pd.Timestamp(info["opened_ts"]) if info.get("opened_ts") else \
                    pd.Timestamp(now) - pd.Timedelta(hours=1)
                eh, em = (int(v) for v in cfg.exit.split(":"))
                close_at = (pd.Timestamp(datetime.combine(datetime.fromisoformat(_lr.trading_day(opened.to_pydatetime())).date(),
                                                          datetime.min.time().replace(hour=eh, minute=em)), tz=_lr.ET)
                            .tz_convert(local_tz))
                try:
                    b, mins = _bars_around(cfg.symbol, opened, pd.Timestamp(now), account, port)
                    fig = _trade_figure(b, mins, local_tz, entry_t=opened.tz_convert(local_tz), entry_px=entry_px,
                                        tp=target_px, line=line_px, live_t=pd.Timestamp(now).tz_convert(local_tz),
                                        live_px=cur, close_at=close_at, height=340,
                                        title=f"Live: {cfg.symbol} since the entry (your time)",
                                        long=info.get("direction", "long") != "short")
                    st.plotly_chart(fig, width="stretch", key="race_live", config={"displayModeBar": False})
                except Exception as e:  # noqa: BLE001 -- no bars yet / bridge hiccup
                    st.caption(f"Live chart unavailable right now ({e}).")
                # --- why this trade is open ---
                closed_ = []
                tlog_ = ROOT / "data" / f"lucid_futures_trades{stag}.jsonl"
                if tlog_.exists():
                    for raw in tlog_.read_text(encoding="utf-8").splitlines():
                        try:
                            if raw.strip():
                                closed_.append(json.loads(raw))
                        except ValueError:
                            pass
                live_t = {**info, "attempt": int(state.get("attempts", 0)) + 1, "entry_price": entry_px,
                          "tp": target_px, "line": line_px, "lots": float(pos.volume),
                          "opened_ts": info.get("opened_ts") or opened.isoformat()}
                st.markdown(vx.section("Why this trade is open"), unsafe_allow_html=True)
                st.markdown(_story_html(_race_story(live_t, closed_[-1] if closed_ else None, None, local_tz,
                                                    live=True)), unsafe_allow_html=True)
            else:
                st.markdown(vx.banner(f"No race is open right now. Next race: <b>{when(nxt)}</b>. Its live"
                                      " chart appears here the moment it starts.", "info"),
                            unsafe_allow_html=True)

        with t_charts:
            # --- every closed trade, as a chart (newest first) ---
            trades_ = []
            tlog = ROOT / "data" / f"lucid_futures_trades{stag}.jsonl"
            if tlog.exists():
                for raw in tlog.read_text(encoding="utf-8").splitlines():
                    try:
                        if raw.strip():
                            trades_.append(json.loads(raw))
                    except ValueError:
                        pass
            chrono_ = [t for t in trades_ if t.get("opened_ts") and t.get("closed_ts")]
            prev_of = {t["opened_ts"]: (chrono_[i - 1] if i else None) for i, t in enumerate(chrono_)}
            trades_ = chrono_[::-1]
            brain_trades = _brain().get("trades") or {}

            def _trade_chart(t: dict, key: str, height: int = 290) -> None:
                o, c = pd.Timestamp(t["opened_ts"]), pd.Timestamp(t["closed_ts"])
                pnl = float(t.get("pnl_usd") or 0.0)
                tp_, ln_, ex_ = t.get("tp"), t.get("line"), float(t["exit_price"])
                why = _CLOSE_REASON.get(t.get("reason") or "")
                sg_ = -1.0 if t.get("direction") == "short" else 1.0
                if why is None:
                    why = ("target reached" if tp_ and sg_ * (ex_ - tp_) >= -0.25 else
                           "loss limit hit" if ln_ and sg_ * (ex_ - ln_) <= 0.25 else "closed")
                ol, cl = o.tz_convert(local_tz), c.tz_convert(local_tz)
                span = f"{ol:%a %d %b %H:%M} → {cl:%H:%M}" if ol.date() == cl.date() else \
                    f"{ol:%a %d %b %H:%M} → {cl:%a %H:%M}"
                title = f"{span} · {usd(pnl)} · {why}"
                try:
                    b, mins = _closed_trade_bars(t["sym"], t["opened_ts"], t["closed_ts"], account, port)
                    fig = _trade_figure(b, mins, local_tz, entry_t=ol, entry_px=float(t["entry_price"]), tp=tp_,
                                        line=ln_, exit_t=cl, exit_px=ex_, exit_text=f"Sold {ex_:,.2f} · {usd(pnl)}",
                                        exit_color=vx.pl_color(pnl), height=height, title=title, long=sg_ > 0)
                    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
                except Exception as e:  # noqa: BLE001 -- NinjaTrader closed / no history
                    st.caption(f"{title} -- chart unavailable ({e}).")
                rec = brain_trades.get(f"Trial|{t.get('opened_ts')}")
                st.markdown(_story_html(_race_story(t, prev_of.get(t["opened_ts"]), rec, local_tz)),
                            unsafe_allow_html=True)

            if not trades_:
                st.caption("No closed trades yet -- each one appears here as a chart.")
            else:
                latest = trades_[:12]
                for i in range(0, len(latest), 2):
                    cols_ = st.columns(2)
                    for j, (col_, t) in enumerate(zip(cols_, latest[i:i + 2])):
                        with col_:
                            _trade_chart(t, key=f"race_trade_{i + j}")
                if len(trades_) > 12:
                    older = trades_[12:]
                    pick = st.selectbox(
                        f"Older trades ({len(older)})", range(len(older)),
                        format_func=lambda k: (f"{pd.Timestamp(older[k]['opened_ts']).tz_convert(local_tz):%a %d %b %H:%M}"
                                               f" · {usd(float(older[k].get('pnl_usd') or 0))}"),
                        key="race_older_pick")
                    _trade_chart(older[pick], key="race_trade_older", height=320)

        with t_res:
            # --- results ---
            st.markdown(vx.section("Results so far"), unsafe_allow_html=True)
            r = st.columns(4)
            r[0].markdown(vx.card("Evaluations passed", f"{passes}", note=None), unsafe_allow_html=True)
            r[1].markdown(vx.card("Evaluations failed", f"{fails}"), unsafe_allow_html=True)
            r[2].markdown(vx.card("Pass rate", f"{passes / resolved:.0%}" if resolved else "—",
                                  note=f"from {resolved} finished" if resolved else "none finished yet"),
                          unsafe_allow_html=True)
            r[3].markdown(vx.card("Expected pass rate", "≈ 41%" if not lenient else "≈ 83%",
                                  note="replay of 2019-2026 NQ data"),
                          unsafe_allow_html=True)
            if history:
                rows_ = []
                for h in reversed(history[-20:]):
                    try:
                        fin = datetime.fromisoformat(h["resolved_ts"]).astimezone(local_tz)
                        fin_s = f"{fin:%a %d %b %H:%M}"
                    except (KeyError, ValueError):
                        fin_s = "—"
                    took = max(1, int(-(-float(h.get("days") or 0) // 1)))
                    rows_.append({"Finished (your time)": fin_s,
                                  "Result": "✅ Passed" if h.get("outcome") == "pass" else "❌ Failed",
                                  "Took": f"{took} day" + ("s" if took > 1 else ""),
                                  "Profit": usd(float(h["final_usd"]) if h.get("final_usd") is not None
                                                else float(h.get("final_pct") or 0) / 100 * ANCHOR)})
                st.dataframe(pd.DataFrame(rows_), width="stretch", hide_index=True)
            else:
                st.caption("No evaluation has finished yet.")

        with t_how:
            # --- how it works, in plain words ---
            with st.expander("How the bot works", expanded=True):
                st.markdown(
                    f"1. Every trading day at **{at_ny(cfg.entry)}** the bot buys **{cfg.contracts} {cfg.symbol}**"
                    " (Nasdaq-100 futures).\n"
                    "2. At the same moment it places a sell order at the exact price where the account reaches"
                    " **+$3,000** &mdash; the evaluation target.\n"
                    "3. If the price gets there, the evaluation is **passed**. On a real Lucid account the bot then"
                    " stops for good.\n"
                    f"4. If not, it closes the trade at **{at_ny(cfg.exit)}** and tries again in the next session,"
                    " aiming for whatever is still missing.\n"
                    "5. The evaluation **fails** the moment the account's value &mdash; open trade included &mdash;"
                    " falls $2,000 below its best daily close (Lucid support confirmed this). So each race is"
                    " +$3,000 against &minus;$2,000.\n"
                    "6. After a failed evaluation the bot waits until you reset the account, then carries on.\n\n"
                    "Expected: about **4 in 10** evaluations passed (replay of 2019-2026) &mdash; close to the"
                    " mathematical limit for a $3,000 target against a $2,000 loss limit. Most passes happen on"
                    " the first day.")

            # --- technical details, folded away ---
            with st.expander("Technical details", expanded=False):
                et_now = now.astimezone(_lr.ET)
                rows_ = [
                    {"Item": "New York time", "Value": f"{et_now:%a %d %b %H:%M}"},
                    {"Item": "Lucid session", "Value": sess.value.replace("_", " ")},
                    {"Item": "Trade date", "Value": today},
                    {"Item": "Last race day", "Value": str(state.get("race_day") or "—")},
                    {"Item": "Race settings", "Value": f"{cfg.contracts} {cfg.symbol}, entries {cfg.entry}-"
                                                       f"{cfg.last_entry} ET, exit {cfg.exit} ET, safety stop "
                                                       f"{cfg.stop_points:g} pts"},
                    {"Item": "Rule reading (simulator)", "Value": reading},
                    {"Item": "Evaluation balance (closed)", "Value": f"${balance:,.2f}"},
                    {"Item": "Passes at / fails below", "Value": f"${ANCHOR + TARGET:,.0f} / ${fail_line:,.2f}"},
                    {"Item": "Peak closing balance", "Value": f"${peak:,.2f}"},
                ]
                if acc:
                    rows_ += [{"Item": "NinjaTrader account", "Value": f"{acc.login} ({acc.server})"},
                              {"Item": "NinjaTrader balance", "Value": f"${acc.balance:,.2f}"}]
                if pos is not None:
                    rows_ += [{"Item": "Open position", "Value": f"{pos.symbol} {pos.volume:g} @ {pos.price_open:,.2f}"
                                                                  f" | target {pos.tp:,.2f} | safety stop {pos.sl:,.2f}"}]
                if bridge_err:
                    rows_ += [{"Item": "Bridge error", "Value": bridge_err}]
                st.dataframe(pd.DataFrame(rows_), width="stretch", hide_index=True)
                tp_ = ROOT / "data" / f"lucid_futures_trades{stag}.jsonl"
                if tp_.exists():
                    trades = []
                    for line in tp_.read_text(encoding="utf-8").splitlines():
                        try:
                            if line.strip():
                                trades.append(json.loads(line))
                        except ValueError:
                            pass
                    if trades:
                        t_ = pd.DataFrame(trades)[["closed_ts", "sym", "lots", "entry_price", "exit_price", "pnl_usd"]]
                        st.caption("Closed trades")
                        st.dataframe(t_.iloc[::-1], width="stretch", hide_index=True)
                st.json(state, expanded=False)

    if (_ENV.get("LUCID_TRIAL_STRATEGY") or "race").strip().lower() == "race":
        _race_trial_view()
    else:
        _lucid_futures_view()
    st.stop()

# LUCIDPRO_TRIAL is a different shape of bot entirely -- repeated $50,000
# evaluation attempts with auto-reset on pass/fail (scripts/run_lucidpro_trial.py),
# tracked in its own JSON stats file, not the SQLite ledger the other three
# bots share via PortfolioRunner/Ledger. Render it separately and stop, rather
# than forcing it through the ledger-shaped rendering code below.
if bot_choice == "LUCIDPRO_TRIAL":
    st.markdown(vx.CSS, unsafe_allow_html=True)
    refresh_s = _refresh_seconds(5)

    @st.fragment(run_every=refresh_s)
    def _lucidpro_trial_view():
        ANCHOR = 50_000.0
        RISK_USD = 45_000.0          # RISK_PCT 0.90 x anchor
        TARGET_PCT, MLL_PCT, DLL_PCT = 0.06, 0.04, 0.024
        ROOT_P = ROOT

        path = ROOT_P / "data" / "state_lucidpro_trial.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            state = {}

        slots = {
            9100: {"name": "US500",  "strategy": "London Breakout"},
            9101: {"name": "NAS100", "strategy": "London Breakout"},
            9102: {"name": "XAUUSD", "strategy": "London Breakout"},
        }
        open_df = get_live_open_positions(slots, block_size=100)
        mt5_up = open_df is not None
        n_open = 0 if open_df is None else len(open_df)
        floating = 0.0 if (open_df is None or open_df.empty) else float(open_df["Floating P&L"].sum())

        attempts = int(state.get("attempts", 0))
        passes = int(state.get("passes", 0))
        fails = int(state.get("fails", 0))
        resolved = passes + fails
        pass_rate = (passes / resolved) if resolved else None
        hist = state.get("history", []) or []

        tw = int(state.get("total_wins", 0))
        tl = int(state.get("total_losses", 0))
        tt = int(state.get("total_trades", 0))
        # Resolved-only denominator: total_trades counts opens, wins/losses only
        # move on close, so dividing by total_trades understates the win rate
        # whenever a position is still running.
        t_resolved = tw + tl
        trade_wr = (tw / t_resolved) if t_resolved else None

        # ---------- market + header ----------
        ms = market_status(tuple(sorted({v["name"] for v in slots.values()})))
        if not mt5_up:
            status, scol = "NO CONNECTION", vx.WARN
        elif n_open > 0:
            status, scol = "IN MARKET", vx.OK
        else:
            status, scol = "ARMED", vx.INFO

        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Vertex Algo Bot</div>'
            f'<div class="vx-sub">LucidPro Trial &middot; repeated $50,000 evaluations '
            f'&middot; 90% risk ($45,000/trade)</div></div>'
            f'<div class="vx-spacer"></div>{market_pill(ms)}&nbsp;{vx.pill(status, scol)}</div>',
            unsafe_allow_html=True)

        if not ms["open"]:
            st.markdown(vx.banner(
                "<b>Market is closed.</b> The bot stays running and resumes at the"
                " open; no orders can be placed or closed until then.", "info"),
                unsafe_allow_html=True)

        if not mt5_up:
            st.markdown(vx.banner(
                "<b>MetaTrader 5 unreachable.</b> Live positions and order placement are"
                " unavailable; figures below come from the last written state.", "warn"),
                unsafe_allow_html=True)

        st.markdown(vx.banner(
            "This bot deliberately risks <b>90% of the anchor per trade</b> to clear"
            " evaluations as fast as possible. That is a variance strategy, not an income"
            " one &mdash; a single losing trade ends an attempt. Read pass/fail counts as"
            " the result; the per-trade dollar figures are not a survivable equity path.",
            "warn"), unsafe_allow_html=True)

        # ---------- live attempt: how close is it to resetting? ----------
        #
        # The reset itself is invisible after the fact -- the counters just tick
        # up. What the operator actually needs is the DISTANCE to the limit while
        # the attempt is still running, so a reset is something they saw coming
        # rather than something they discover afterwards.
        att = state.get("attempt") or {}
        if att:
            a_real = float(att.get("realized_usd", 0.0))
            a_peak = float(att.get("peak_equity", ANCHOR))
            a_eq = ANCHOR + a_real + float(att.get("floating_usd", 0.0) or 0.0)
            mll_floor = a_peak - MLL_PCT * ANCHOR
            room = a_eq - mll_floor
            room_frac = max(0.0, min(1.0, room / (MLL_PCT * ANCHOR))) if MLL_PCT else 1.0
            target_eq = ANCHOR * (1 + TARGET_PCT)
            to_target = target_eq - a_eq

            st.markdown(vx.section("Current attempt &mdash; live"), unsafe_allow_html=True)
            k = st.columns(4)
            k[0].markdown(vx.card("Attempt equity", f"${a_eq:,.0f}",
                                  f"{(a_eq - ANCHOR) / ANCHOR:+.2%} vs anchor",
                                  vx.POS if a_eq >= ANCHOR else vx.NEG),
                          unsafe_allow_html=True)
            k[1].markdown(vx.card("Room to reset", f"${room:,.0f}",
                                  f"resets at ${mll_floor:,.0f}",
                                  vx.NEG if room_frac < 0.34 else
                                  (vx.WARN if room_frac < 0.67 else vx.OK)),
                          unsafe_allow_html=True)
            k[2].markdown(vx.card("To pass", f"${to_target:,.0f}",
                                  f"target ${target_eq:,.0f}", vx.TEXT_MUTED),
                          unsafe_allow_html=True)
            k[3].markdown(vx.card("Trades this attempt",
                                  f"{int(att.get('total_entries', 0))}",
                                  f"{int(att.get('wins', 0))}W / "
                                  f"{int(att.get('losses', 0))}L", vx.TEXT_MUTED,
                                  small=True), unsafe_allow_html=True)
            st.progress(room_frac, text=None)
            st.caption(
                f"The attempt resets automatically when equity falls "
                f"{MLL_PCT:.0%} (${MLL_PCT * ANCHOR:,.0f}) below its trailing peak of "
                f"${a_peak:,.0f}. Checked every poll, not just at end of day — and "
                f"the attempt is now persisted across restarts, so a drawdown is no "
                f"longer forgotten when the process recycles.")

        # ---------- evaluation KPIs ----------
        st.markdown(vx.section("Evaluations"), unsafe_allow_html=True)
        c = st.columns(4)
        c[0].markdown(vx.card("Attempts resolved", f"{resolved}",
                              f"{attempts} started", vx.TEXT_MUTED), unsafe_allow_html=True)
        c[1].markdown(vx.card("Passed", f"{passes}",
                              "evaluations cleared", vx.POS if passes else vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[2].markdown(vx.card("Failed", f"{fails}",
                              "MLL or blow-up", vx.NEG if fails else vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[3].markdown(vx.card("Pass rate",
                              f"{pass_rate:.0%}" if pass_rate is not None else "—",
                              "backtested ref 67.4%",
                              vx.POS if (pass_rate or 0) >= .5 else vx.TEXT_MUTED),
                      unsafe_allow_html=True)

        # ---------- trade KPIs ----------
        st.markdown(vx.section("Trades"), unsafe_allow_html=True)
        d = st.columns(4)
        d[0].markdown(vx.card("Closed", f"{t_resolved}",
                              f"{tw}W / {tl}L" + (f" &middot; {n_open} open" if n_open else ""),
                              vx.TEXT_MUTED), unsafe_allow_html=True)
        d[1].markdown(vx.card("Win rate",
                              f"{trade_wr:.0%}" if trade_wr is not None else "—",
                              "closed trades only", vx.TEXT_MUTED), unsafe_allow_html=True)
        d[2].markdown(vx.card("Trades / attempt",
                              f"{tt / attempts:.1f}" if attempts else "—",
                              "an attempt resolves fast at 90% risk", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        d[3].markdown(vx.card("Floating", f"${floating:+,.2f}",
                              f"{n_open} open", vx.pl_color(floating)), unsafe_allow_html=True)

        # ---------- current attempt ----------
        start_iso = state.get("current_attempt_start")
        if start_iso:
            try:
                started = datetime.fromisoformat(start_iso)
                el = (datetime.now(timezone.utc) - started).total_seconds() / 86400.0
                st.markdown(vx.banner(
                    f"Current attempt running <b>{el:.2f} days</b> &mdash; "
                    f"{state.get('current_attempt_opened', 0)} entries opened this session. "
                    f"Target +{TARGET_PCT:.0%}, MLL &minus;{MLL_PCT:.0%} (EOD), "
                    f"DLL &minus;{DLL_PCT:.1%} (soft).", "info"), unsafe_allow_html=True)
            except ValueError:
                pass

        t_att, t_pos, t_trades, t_sys = st.tabs(
            ["Attempts", "Positions", "Trades", "System"])

        # =========================== ATTEMPTS ===========================
        with t_att:
            # Attempts resolved before 2026-08-29 cannot be read as a pass rate.
            # Until that date the FAIL check ran only on a day rollover while the
            # PASS check ran every poll, and the attempt was not persisted across
            # restarts -- so each restart erased the accumulated drawdown and a
            # losing attempt was structurally unable to reach the -4% limit. The
            # resulting 8/8 record measured state loss, not performance.
            _ARTIFACT_CUTOFF = "2026-08-29"
            _pre = sum(1 for e in hist
                       if str(e.get("resolved_ts", ""))[:10] < _ARTIFACT_CUTOFF)
            if _pre:
                st.markdown(vx.banner(
                    f"<b>{_pre} of these attempts predate the 2026-08-29 fix and are "
                    "not a valid pass rate.</b> Before that date the fail check ran "
                    "only at end of day while the pass check ran every poll, and the "
                    "attempt was not saved across restarts &mdash; so drawdown was "
                    "erased on every process recycle and an attempt could not fail. "
                    "Attempts resolved after the cutoff are the ones to judge.",
                    "warn"), unsafe_allow_html=True)
            if not hist:
                st.markdown(vx.banner("No resolved attempts yet.", "info"),
                            unsafe_allow_html=True)
            else:
                h = pd.DataFrame(hist)
                pref = [c_ for c_ in ("outcome", "days", "trades", "wins", "losses",
                                      "final_pct", "opened_ts", "resolved_ts")
                        if c_ in h.columns]
                view = h[pref].iloc[::-1]

                def _oc(v):
                    s_ = str(v).lower()
                    if s_ == "pass":
                        return f"color:{vx.POS};font-weight:600"
                    if s_ in ("fail", "blown"):
                        return f"color:{vx.NEG};font-weight:600"
                    return ""

                sty = view.style.map(_oc, subset=["outcome"])
                if "final_pct" in view.columns:
                    sty = sty.map(_color_pnl, subset=["final_pct"]).format({"final_pct": "{:+.2f}%"})
                if "days" in view.columns:
                    sty = sty.format({"days": "{:.2f}d"})
                st.dataframe(sty, width="stretch", hide_index=True)

                counts = h["outcome"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values, hole=.6,
                    marker=dict(colors=[vx.POS if str(o).lower() == "pass" else vx.NEG
                                        for o in counts.index],
                                line=dict(color=vx.SURFACE, width=2)),
                    textinfo="label+value",
                    hovertemplate="%{label}: %{value}<extra></extra>"))
                fig.update_layout(**vx.plotly_layout(280, "Outcomes"))
                st.plotly_chart(fig, width="stretch", key="vx_t_pie")

                if "final_pct" in h.columns:
                    b = go.Figure(go.Bar(
                        x=list(range(1, len(h) + 1)), y=h["final_pct"],
                        marker_color=[vx.pl_color(v) for v in h["final_pct"]],
                        hovertemplate="attempt %{x}<br>%{y:+.2f}%<extra></extra>"))
                    b.update_layout(**vx.plotly_layout(
                        240, "Result per attempt",
                        xaxis=dict(title="attempt", gridcolor=vx.GRID),
                        yaxis=dict(gridcolor=vx.GRID, ticksuffix="%")))
                    st.plotly_chart(b, width="stretch", key="vx_t_bar")

        # =========================== POSITIONS ===========================
        with t_pos:
            if open_df is None:
                st.markdown(vx.banner("MetaTrader 5 unreachable — cannot list positions.",
                                      "warn"), unsafe_allow_html=True)
            elif open_df.empty:
                st.markdown(vx.banner("No open positions.", "info"), unsafe_allow_html=True)
            else:
                k = st.columns(3)
                k[0].markdown(vx.card("Open", f"{len(open_df)}", "positions", vx.TEXT_MUTED),
                              unsafe_allow_html=True)
                k[1].markdown(vx.card("Floating", f"${floating:+,.2f}", "unrealised",
                                      vx.pl_color(floating)), unsafe_allow_html=True)
                winners = int((open_df["Floating P&L"] > 0).sum())
                k[2].markdown(vx.card("In profit", f"{winners}/{len(open_df)}", "currently green",
                                      vx.TEXT_MUTED), unsafe_allow_html=True)
                st.dataframe(
                    open_df.style
                    .map(_color_pnl, subset=["Floating P&L"])
                    .map(_color_dir, subset=["Dir"])
                    .format({"Entry": "{:,.5f}", "Current": "{:,.5f}", "SL": "{:,.5f}",
                             "TP": "{:,.5f}", "Floating P&L": "${:+,.2f}", "Volume": "{:.2f}"}),
                    width="stretch", hide_index=True)
                render_close_controls(open_df, book="trial")
                for _, row in open_df.iterrows():
                    with st.container(border=True):
                        is_long = row["Dir"] == UP
                        st.markdown(f"**{row['Strategy']}** — {row['Symbol']} "
                                    f"{dir_badge(is_long)} &nbsp;{row['Volume']:.2f} lots "
                                    f"@ {row['Entry']:.5f}", unsafe_allow_html=True)
                        components.html(build_live_chart_html(
                            row["Symbol"], int(row["Ticket"]), is_long,
                            default_timeframe="15m", height=380), height=500, scrolling=False)

        # =========================== TRADES ===========================
        with t_trades:
            rows = []
            tp_ = ROOT_P / "data" / "lucidpro_trades.jsonl"
            if tp_.exists():
                for line in tp_.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            pass
            df_ = pd.DataFrame(rows) if rows else None
            st.markdown(vx.section("Closed trades"), unsafe_allow_html=True)
            render_trade_table(df_, key="trial_real")
            if df_ is not None and not df_.empty:
                render_trade_charts(df_, ANCHOR, key="trial")

        # =========================== SYSTEM ===========================
        with t_sys:
            st.markdown(vx.section("Market sessions"), unsafe_allow_html=True)
            render_market_panel(ms)

            st.markdown(vx.section("Bot state"), unsafe_allow_html=True)
            st.json(state, expanded=False)

            st.markdown(vx.section("Danger zone"), unsafe_allow_html=True)
            st.markdown(vx.banner(
                "<b>Reset account</b> closes this book's open positions, archives its state"
                " and trade ledger, and clears the attempt history. Archives are kept and"
                " restorable &mdash; nothing is deleted. The reset aborts if positions"
                " cannot actually be closed (which is the case whenever the market is"
                " shut), so they are never left orphaned.", "danger"), unsafe_allow_html=True)
            conf = st.text_input('Type RESET to enable the button', key="vx_treset",
                                 placeholder="RESET")
            ca, cb = st.columns([1, 3])
            with ca:
                go_ = st.button("Reset account", type="primary",
                                disabled=(conf.strip().upper() != "RESET"), key="vx_tbtn")
            with cb:
                if conf.strip().upper() != "RESET":
                    st.caption("Button unlocks once the confirmation text matches.")
            if go_:
                try:
                    eng = MT5Engine(symbol_map={"NAS100": "USTEC"})
                    eng.connect()
                except Exception:
                    eng = None
                rep = vxr.reset_book(ROOT_P, "lucidpro_trial", range(9100, 9110),
                                     eng, anchor=ANCHOR)
                if rep.ok:
                    st.success(f"Account reset. Archive: {Path(rep.archive_dir).name}")
                    for s_ in rep.steps:
                        st.caption(f"· {s_}")
                    st.caption("Restart the bot so it picks up the new state.")
                else:
                    st.error(rep.error or "Reset failed.")
                    for s_ in rep.steps:
                        st.caption(f"· {s_}")

            arcs = vxr.list_archives(ROOT_P, "lucidpro_trial")
            if arcs:
                st.markdown(vx.section("Restore points"), unsafe_allow_html=True)
                pick = st.selectbox("Archive", [a["name"] for a in arcs], key="vx_tarc")
                if st.button("Restore selected archive", key="vx_trestore"):
                    ok_, msg = vxr.restore_archive(ROOT_P, pick)
                    (st.success if ok_ else st.error)(msg)

            st.markdown(vx.section("Logs"), unsafe_allow_html=True)
            lp = ROOT_P / "logs" / "lucidpro_trial.log"
            if lp.exists():
                st.code("\n".join(lp.read_text(encoding="utf-8",
                                               errors="replace").splitlines()[-80:]),
                        language=None)

    _lucidpro_trial_view()
    st.stop()

# LUCIDPRO_FUNDED added 2026-08-15 when this bot replaced FTMO. Same shape
# as LUCIDPRO_TRIAL (JSON state + JSONL trades, not the SQLite ledger) but a
# different bot entirely: no eval target, no attempts/passes/fails -- it
# runs indefinitely post-funding with an adaptive risk taper, and HALTS
# (rather than auto-resetting) on an MLL breach, so a halt is a real event
# that needs a human, not routine background noise.
if bot_choice == "LUCIDPRO_FUNDED":
    st.markdown(vx.CSS, unsafe_allow_html=True)
    refresh_s = _refresh_seconds(5)

    @st.fragment(run_every=refresh_s)
    def _lucidpro_funded_view():
        ANCHOR = 50_000.0
        DAILY_LIMIT = 2_000.0
        PAYOUT = 2_000.0
        ROOT_P = ROOT
        # Parsed from the runner so the UI always reports the live setting.
        # A hardcoded copy silently lied after the risk change on 2026-08-22.
        RISK_PCT = 0.010
        try:
            import re as _re
            _m = _re.search(r"^BASE_RISK_PCT\s*=\s*([0-9.]+)",
                            (ROOT_P / "scripts" / "run_lucidpro_funded.py")
                            .read_text(encoding="utf-8"), _re.M)
            if _m:
                RISK_PCT = float(_m.group(1))
        except OSError:
            pass

        path = ROOT_P / "data" / "state_lucidpro_funded.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            state = {}

        # ---------- live broker truth ----------
        slots = {
            9210: {"name": "US500",  "strategy": "Donch55_US500_1d"},
            9211: {"name": "XAUUSD", "strategy": "Donch55_XAUUSD_1d"},
            9212: {"name": "CHFJPY", "strategy": "Donch55_CHFJPY_1d"},
            9213: {"name": "USDJPY", "strategy": "Donch55_USDJPY_1d"},
            9214: {"name": "NAS100", "strategy": "Donch55_NAS100_4h"},
            9215: {"name": "XAUUSD", "strategy": "Donch55_XAUUSD_4h"},
            9216: {"name": "GBPJPY", "strategy": "MACD_GBPJPY_4h"},
            9217: {"name": "GBPUSD", "strategy": "RSI_GBPUSD_4h"},
        }
        open_df = get_live_open_positions(slots, block_size=100)
        mt5_up = open_df is not None

        # Positions carried by an already-blown run are shown separately and
        # excluded from THIS account's figures. Without this the dashboard kept
        # displaying the dead account's loss after a reset -- equity stuck at
        # -8.85%, today's P&L at -$6,354 -- which is exactly the "I can still
        # see the old attempt" symptom. The runner excludes the same ticket ids
        # from its own equity, so the two agree.
        carried_ids = set(state.get("excluded_tickets") or [])
        carried_df = None
        if carried_ids and open_df is not None and not open_df.empty \
                and "Ticket" in open_df.columns:
            mask = open_df["Ticket"].isin(carried_ids)
            carried_df = open_df[mask]
            open_df = open_df[~mask]

        n_open = 0 if open_df is None else len(open_df)
        floating = 0.0 if (open_df is None or open_df.empty) else float(open_df["Floating P&L"].sum())

        # ---------- real closed trades (CURRENT RUN only) ----------
        #
        # The ledger is append-only across the bot's whole life, so showing it
        # unfiltered made a freshly reset $50,000 account display the previous
        # account's trades and win rate. The ledger itself is never truncated --
        # every trade is kept for the record; this is a view filter.
        _run_start_iso = str(state.get("run_start") or "")
        try:
            _run_start_dt = (datetime.fromisoformat(_run_start_iso)
                             if _run_start_iso else None)
        except ValueError:
            _run_start_dt = None
        live_rows = []
        tp = ROOT_P / "data" / "lucidpro_funded_trades.jsonl"
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        _r = json.loads(line)
                    except ValueError:
                        continue
                    if _run_start_dt is not None:
                        _ts = _r.get("closed_ts") or _r.get("opened_ts")
                        try:
                            if _ts and datetime.fromisoformat(str(_ts)) < _run_start_dt:
                                continue
                        except ValueError:
                            pass
                    live_rows.append(_r)
        real_df = pd.DataFrame(live_rows) if live_rows else None
        real = vxs.compute(_normalize_trades(real_df) if real_df is not None else None,
                           ANCHOR, source="real", n_open=n_open, floating_pnl=floating)

        # Broker truth for headline money. The ledger above misses any position
        # closed by hand, so realised P&L, equity and the payout bar are driven
        # from the broker's own deal history instead.
        # Scoped to the CURRENT RUN, not the bot's lifetime. A blown account
        # starts a fresh $50,000 one, and carrying the previous run's realised
        # P&L into it makes the new account inherit profit it never earned --
        # the reset would show a $50,000 anchor next to +$2,329 of somebody
        # else's history. run_start is stamped at every reset; started_ts is
        # only the fallback for a state file predating runs.
        run_since = str(state.get("run_start") or state.get("started_ts")
                        or "2026-08-14T00:00:00+00:00")
        started = run_since
        bk = broker_realised(9210, 9217, started)
        bk_all = broker_realised(9200, 9217, started)   # includes the retired v5 book
        if bk.get("ok"):
            real.net_pnl = bk["pnl"]
            real.n_closed = bk["n"]
            real.wins, real.losses = bk["wins"], bk["losses"]
            resolved = bk["wins"] + bk["losses"]
            real.win_rate = (bk["wins"] / resolved) if resolved else None

        # ---------- simulated baseline (kept strictly separate) ----------
        fb = sim_baseline("funded") or {}
        sim_rows = fb.get("trades") or []
        sim_df = pd.DataFrame([{**t, "closed_ts": t.get("closed_at"),
                                "sym": t.get("symbol"), "label": t.get("leg")}
                               for t in sim_rows]) if sim_rows else None
        sim = vxs.compute(_normalize_trades(sim_df) if sim_df is not None else None,
                          ANCHOR, source="simulated")

        # ---------- market + header ----------
        ms = market_status(tuple(sorted({v["name"] for v in slots.values()})))
        halted = bool(state.get("halted"))
        if halted:
            status, scol = "HALTED", vx.DANGER
        elif not mt5_up:
            status, scol = "NO CONNECTION", vx.WARN
        elif n_open > 0:
            status, scol = "IN MARKET", vx.OK
        else:
            status, scol = "ARMED", vx.INFO

        day_start = float(state.get("day_start_equity") or ANCHOR)
        equity_real = ANCHOR + real.net_pnl + floating
        day_pnl = equity_real - day_start
        day_used = min(1.0, max(0.0, -day_pnl / DAILY_LIMIT)) if day_pnl < 0 else 0.0

        st.markdown(
            f'<div class="vx-top"><div class="vx-mark">V</div>'
            f'<div><div class="vx-name">Vertex Algo Bot</div>'
            f'<div class="vx-sub">LucidPro Funded &middot; v6 one-payout roster '
            f'&middot; $50,000 anchor</div></div>'
            f'<div class="vx-spacer"></div>{market_pill(ms)}&nbsp;{vx.pill(status, scol)}</div>',
            unsafe_allow_html=True)

        if not ms["open"]:
            st.markdown(vx.banner(
                "<b>Market is closed.</b> The bots stay running and will resume"
                " automatically at the open; no orders can be placed or closed until"
                " then. Session detail is in the System tab.", "info"),
                unsafe_allow_html=True)

        if halted:
            st.markdown(vx.banner(
                f"<b>Trading halted.</b> {state.get('halt_reason', 'no reason recorded')}"
                " &mdash; the bot stopped itself and will not trade until this is cleared"
                " in the System tab.", "danger"), unsafe_allow_html=True)
        if not mt5_up:
            st.markdown(vx.banner(
                "<b>MetaTrader 5 unreachable.</b> Live positions, prices and order"
                " placement are all unavailable. Figures below are from the last"
                " written state.", "warn"), unsafe_allow_html=True)

        # ---------- KPI row: TOTAL account money ----------
        #
        # Headline figures are ACCOUNT-level, not v6-book-level. Scoping them to
        # the active roster is technically defensible and in practice misleading:
        # on 2026-08-24 the panel showed $401 realised while $2,330 sat in the
        # balance, because $1,928 of it came from the retired v5 book. The number
        # a person checks has to be the number in their account; the per-roster
        # split belongs underneath, not instead of it.
        total_realised = bk_all["pnl"] if bk_all.get("ok") else real.net_pnl
        v6_realised = bk["pnl"] if bk.get("ok") else real.net_pnl
        other_realised = total_realised - v6_realised
        equity_total = ANCHOR + total_realised + floating

        st.markdown(vx.section("Account &mdash; real money"), unsafe_allow_html=True)
        c = st.columns(4)
        c[0].markdown(vx.card(
            "Equity", f"${equity_total:,.2f}",
            f"{(equity_total - ANCHOR) / ANCHOR:+.2%} vs anchor",
            vx.pl_color(equity_total - ANCHOR)), unsafe_allow_html=True)
        c[1].markdown(vx.card(
            "Realised P&L", f"${total_realised:+,.2f}",
            (f"v6 ${v6_realised:+,.0f} · retired ${other_realised:+,.0f}"
             if abs(other_realised) > 0.01 else
             (f"{bk.get('n', 0)} closed trades" if bk.get("ok") else "no closed trades")),
            vx.pl_color(total_realised)), unsafe_allow_html=True)
        c[2].markdown(vx.card(
            "Floating P&L", f"${floating:+,.2f}",
            f"{n_open} position{'s' if n_open != 1 else ''} open",
            vx.pl_color(floating)), unsafe_allow_html=True)
        c[3].markdown(vx.card(
            "Progress to payout", f"${max(0.0, equity_total - ANCHOR):,.0f} / ${PAYOUT:,.0f}",
            f"{min(100.0, max(0.0, (equity_total - ANCHOR) / PAYOUT * 100)):.0f}% of target",
            vx.POS if equity_total - ANCHOR >= PAYOUT else vx.TEXT_MUTED),
            unsafe_allow_html=True)

        if bk_all.get("ok"):
            st.caption(
                f"Realised P&L is read from the broker's deal history, so it includes "
                f"positions closed by hand as well as by the bot — "
                f"{bk_all.get('n', 0)} closed trades in total"
                + (f", of which {bk.get('n', 0)} on the active v6 roster "
                   f"(${v6_realised:+,.2f}) and the rest on the retired v5 book "
                   f"(${other_realised:+,.2f})." if abs(other_realised) > 0.01 else "."))

        # ---------- daily risk ----------
        st.markdown(vx.section("Daily loss limit"), unsafe_allow_html=True)
        c = st.columns(4)
        room = DAILY_LIMIT + min(0.0, day_pnl)
        c[0].markdown(vx.card("Today's P&L", f"${day_pnl:+,.2f}",
                              f"day opened ${day_start:,.2f}", vx.pl_color(day_pnl)),
                      unsafe_allow_html=True)
        c[1].markdown(vx.card("Allowance left", f"${room:,.0f}",
                              f"{day_used:.0%} of ${DAILY_LIMIT:,.0f} used",
                              vx.DANGER if day_used > 0.75 else (
                                  vx.WARN if day_used > 0.4 else vx.OK)),
                      unsafe_allow_html=True)
        # Read from the runner rather than hardcoded, so the card cannot drift
        # out of sync with what the bot is actually risking.
        c[2].markdown(vx.card("Risk per trade", f"${RISK_PCT * ANCHOR:,.0f}",
                              f"{RISK_PCT:.1%} of anchor", vx.TEXT_MUTED),
                      unsafe_allow_html=True)
        c[3].markdown(vx.card("Entries", "BLOCKED" if day_used >= 0.75 else "ALLOWED",
                              "stops at 75% of limit",
                              vx.DANGER if day_used >= 0.75 else vx.OK, small=True),
                      unsafe_allow_html=True)
        st.progress(day_used, text=None)

        # ---------- positions carried by a blown run ----------
        if carried_df is not None and not carried_df.empty:
            c_float = float(carried_df["Floating P&L"].sum())
            st.markdown(vx.banner(
                f"<b>{len(carried_df)} position(s) belong to a blown account, "
                f"not this one.</b> Floating ${c_float:+,.2f}, excluded from every "
                "figure above. They could not be closed when that account blew "
                "(the market was shut) and are still managed by the trailing and "
                "time stops &mdash; they will close on their own. Listed under "
                "Positions.", "warn"), unsafe_allow_html=True)

        # ---------- account runs: blow-ups and payouts ----------
        #
        # Before 2026-08-29 a daily-limit breach halted the bot terminally and
        # needed a hand edit of the state file to resume, so there was nothing
        # to count. It now blows the virtual account and opens a fresh one, which
        # makes "how often does this roster survive to a payout" an answerable
        # question -- but only if the outcomes are actually shown.
        runs = state.get("runs", []) or []
        r_total = int(state.get("runs_total", 0))
        r_blown = int(state.get("runs_blown", 0))
        r_paid = int(state.get("runs_paid", 0))
        if r_total or runs:
            st.markdown(vx.section("Account runs"), unsafe_allow_html=True)
            rc = st.columns(4)
            rc[0].markdown(vx.card("Runs", f"{r_total}", "virtual accounts used",
                                   vx.TEXT_MUTED), unsafe_allow_html=True)
            rc[1].markdown(vx.card("Reached payout", f"{r_paid}",
                                   f"${PAYOUT:,.0f} target",
                                   vx.POS if r_paid else vx.TEXT_MUTED),
                           unsafe_allow_html=True)
            rc[2].markdown(vx.card("Blown", f"{r_blown}",
                                   f"${DAILY_LIMIT:,.0f} daily limit",
                                   vx.NEG if r_blown else vx.TEXT_MUTED),
                           unsafe_allow_html=True)
            surv = (r_paid / r_total) if r_total else None
            rc[3].markdown(vx.card("Survival", f"{surv:.0%}" if surv is not None else "—",
                                   "payout before blow-up",
                                   vx.POS if (surv or 0) >= .5 else vx.WARN, small=True),
                           unsafe_allow_html=True)
            if runs:
                rdf = pd.DataFrame(runs)
                keep = [c_ for c_ in ("outcome", "reason", "day", "final_pct",
                                      "final_equity", "resolved_ts") if c_ in rdf.columns]
                rv = rdf[keep].iloc[::-1]

                def _run_oc(v):
                    s_ = str(v).lower()
                    if s_ == "payout":
                        return f"color:{vx.POS};font-weight:600"
                    if s_ in ("blown", "fail"):
                        return f"color:{vx.NEG};font-weight:600"
                    return ""
                rsty = rv.style.map(_run_oc, subset=["outcome"])
                if "final_pct" in rv.columns:
                    rsty = rsty.map(_color_pnl, subset=["final_pct"]).format(
                        {"final_pct": "{:+.2f}%"})
                st.dataframe(rsty, width="stretch", hide_index=True)
            st.caption(
                f"A breach of the ${DAILY_LIMIT:,.0f} daily loss limit now closes every "
                "position, records the run as blown, and starts a fresh "
                f"${ANCHOR:,.0f} virtual account — it no longer halts the bot. The broker "
                "balance is untouched; the anchor is bookkeeping, and every trade stays "
                "in the ledger.")

        # ---------- tabs ----------
        t_over, t_pos, t_trades, t_an, t_sys = st.tabs(
            ["Overview", "Positions", "Trades", "Analytics", "System"])

        # =========================== OVERVIEW ===========================
        with t_over:
            st.markdown(vx.section("Performance &mdash; closed trades"), unsafe_allow_html=True)
            if real.n_closed == 0:
                st.markdown(vx.banner(
                    "<b>No closed trades yet.</b> Every performance figure below will be"
                    " empty until a position closes. This is the correct reading of a new"
                    " book &mdash; it is not a data problem.", "info"), unsafe_allow_html=True)
            g = st.columns(4)
            g[0].markdown(vx.card("Win rate",
                                  f"{real.win_rate:.1%}" if real.win_rate is not None else "—",
                                  f"{real.wins}W / {real.losses}L"
                                  + (f" &middot; {real.n_open} open" if real.n_open else ""),
                                  vx.TEXT_MUTED), unsafe_allow_html=True)
            g[1].markdown(vx.card("Profit factor",
                                  f"{real.profit_factor:.2f}" if real.profit_factor else "—",
                                  "gross win / gross loss", vx.TEXT_MUTED),
                          unsafe_allow_html=True)
            g[2].markdown(vx.card("Expectancy",
                                  f"${real.expectancy:+,.2f}" if real.expectancy is not None else "—",
                                  f"{real.expectancy_r:+.3f}R" if real.expectancy_r is not None else "per trade",
                                  vx.pl_color(real.expectancy)), unsafe_allow_html=True)
            g[3].markdown(vx.card("Max drawdown", f"${real.max_drawdown:,.2f}",
                                  f"{real.max_drawdown_pct:.2f}% of peak"
                                  if real.max_drawdown_pct else "—", vx.TEXT_MUTED),
                          unsafe_allow_html=True)

            if real.equity_curve is not None and not real.equity_curve.empty:
                cur = real.equity_curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cur["ts"], y=cur["equity"], mode="lines",
                    line=dict(color=vx.POS, width=2), fill="tozeroy",
                    fillcolor="rgba(76,110,245,0.10)",
                    hovertemplate="%{x|%d %b %H:%M}<br>$%{y:,.2f}<extra></extra>"))
                fig.add_hline(y=ANCHOR, line_dash="dot", line_color=vx.TEXT_FAINT, opacity=.6)
                fig.update_layout(**vx.plotly_layout(280, "Equity"))
                st.plotly_chart(fig, width="stretch", key="vx_eq")

            st.markdown(vx.section("Reconstructed history &mdash; not real money"),
                        unsafe_allow_html=True)
            st.markdown(vx.banner(
                f"Order placement was broken from go-live until 2026-08-20 15:33 UTC."
                f" The figures here rebuild that period from real price data so the gap"
                f" is not simply blank. <b>They are a simulation and are excluded from"
                f" every account number above.</b>", "warn"), unsafe_allow_html=True)
            s = st.columns(4)
            s[0].markdown(vx.card("Simulated net", f"${sim.net_pnl:+,.2f}",
                                  f"{sim.total_r:+.2f}R", vx.TEXT_MUTED), unsafe_allow_html=True)
            s[1].markdown(vx.card("Simulated trades", f"{sim.n_closed}",
                                  f"{sim.wins}W / {sim.losses}L", vx.TEXT_MUTED),
                          unsafe_allow_html=True)
            s[2].markdown(vx.card("Simulated win rate",
                                  f"{sim.win_rate:.1%}" if sim.win_rate is not None else "—",
                                  "in-sample", vx.TEXT_MUTED), unsafe_allow_html=True)
            s[3].markdown(vx.card("Combined view",
                                  f"${real.net_pnl + sim.net_pnl:+,.2f}",
                                  "real + simulated", vx.TEXT_FAINT), unsafe_allow_html=True)

        # =========================== POSITIONS ===========================
        with t_pos:
            if open_df is None:
                st.markdown(vx.banner("MetaTrader 5 unreachable — cannot list positions.",
                                      "warn"), unsafe_allow_html=True)
            elif open_df.empty:
                st.markdown(vx.banner("No open positions.", "info"), unsafe_allow_html=True)
            else:
                enriched = _add_stop_columns(open_df)
                prot = int((enriched["Locked R"] > 0).sum()) if "Locked R" in enriched else 0
                k = st.columns(4)
                k[0].markdown(vx.card("Open", f"{len(open_df)}", "positions", vx.TEXT_MUTED),
                              unsafe_allow_html=True)
                k[1].markdown(vx.card("Floating", f"${floating:+,.2f}", "unrealised",
                                      vx.pl_color(floating)), unsafe_allow_html=True)
                k[2].markdown(vx.card("Stop in profit", f"{prot}/{len(open_df)}",
                                      "cannot lose", vx.OK if prot else vx.TEXT_MUTED),
                              unsafe_allow_html=True)
                winners = int((open_df["Floating P&L"] > 0).sum())
                k[3].markdown(vx.card("In profit", f"{winners}/{len(open_df)}",
                                      "currently green", vx.TEXT_MUTED), unsafe_allow_html=True)

                st.markdown(vx.section("Open positions"), unsafe_allow_html=True)
                st.dataframe(
                    enriched.style
                    .map(_color_pnl, subset=["Floating P&L"])
                    .map(_color_dir, subset=["Dir"])
                    .format({"Entry": "{:,.5f}", "Current": "{:,.5f}", "SL": "{:,.5f}",
                             "TP": "{:,.5f}", "SL orig": "{:,.5f}", "Locked R": "{:+.2f}",
                             "Floating P&L": "${:+,.2f}", "Volume": "{:.2f}"}),
                    width="stretch", hide_index=True)
                render_close_controls(open_df, book="funded")
                st.markdown(vx.section("Charts"), unsafe_allow_html=True)
                for _, row in open_df.iterrows():
                    with st.container(border=True):
                        is_long = row["Dir"] == UP
                        st.markdown(f"**{row['Strategy']}** — {row['Symbol']} "
                                    f"{dir_badge(is_long)} &nbsp;{row['Volume']:.2f} lots "
                                    f"@ {row['Entry']:.5f}", unsafe_allow_html=True)
                        components.html(build_live_chart_html(
                            row["Symbol"], int(row["Ticket"]), is_long,
                            default_timeframe="15m", height=380), height=500, scrolling=False)

        # =========================== TRADES ===========================
        with t_trades:
            st.markdown(vx.section("Closed trades &mdash; real"), unsafe_allow_html=True)
            render_trade_table(real_df, key="funded_real")
            if sim_rows:
                st.markdown(vx.section("Closed trades &mdash; reconstructed (simulated)"),
                            unsafe_allow_html=True)
                render_trade_table(sim_df, key="funded_sim")

        # =========================== ANALYTICS ===========================
        with t_an:
            src = real if real.n_closed else sim
            tag = "real" if real.n_closed else "simulated — no real trades yet"
            st.markdown(vx.section(f"Breakdown ({tag})"), unsafe_allow_html=True)
            if src.n_closed == 0:
                st.markdown(vx.banner("No closed trades to analyse yet.", "info"),
                            unsafe_allow_html=True)
            else:
                a = st.columns(4)
                a[0].markdown(vx.card("Avg win",
                                      f"${src.avg_win:,.2f}" if src.avg_win else "—",
                                      f"best ${src.largest_win:,.2f}" if src.largest_win else "",
                                      vx.POS), unsafe_allow_html=True)
                a[1].markdown(vx.card("Avg loss",
                                      f"${src.avg_loss:,.2f}" if src.avg_loss else "—",
                                      f"worst ${src.largest_loss:,.2f}" if src.largest_loss else "",
                                      vx.NEG), unsafe_allow_html=True)
                a[2].markdown(vx.card("Win/loss ratio",
                                      f"{src.win_loss_ratio:.2f}" if src.win_loss_ratio else "—",
                                      "avg win / avg loss", vx.TEXT_MUTED), unsafe_allow_html=True)
                a[3].markdown(vx.card("Avg hold",
                                      f"{src.avg_hold_hours:.1f}h" if src.avg_hold_hours else "—",
                                      "per trade", vx.TEXT_MUTED), unsafe_allow_html=True)
                b = st.columns(4)
                b[0].markdown(vx.card("Best streak", f"{src.max_win_streak}", "consecutive wins",
                                      vx.POS), unsafe_allow_html=True)
                b[1].markdown(vx.card("Worst streak", f"{src.max_loss_streak}",
                                      "consecutive losses", vx.NEG), unsafe_allow_html=True)
                b[2].markdown(vx.card("Current streak",
                                      f"{abs(src.current_streak)}"
                                      f"{'W' if src.current_streak > 0 else ('L' if src.current_streak < 0 else '')}",
                                      "running", vx.pl_color(src.current_streak)),
                              unsafe_allow_html=True)
                b[3].markdown(vx.card("Gross P/L",
                                      f"${src.gross_profit:,.0f} / ${src.gross_loss:,.0f}",
                                      "won / lost", vx.TEXT_MUTED, small=True),
                              unsafe_allow_html=True)

                if src.by_strategy is not None and not src.by_strategy.empty:
                    d = src.by_strategy.sort_values("net")
                    f1 = go.Figure(go.Bar(
                        x=d["net"], y=d["Strategy"], orientation="h",
                        marker_color=[vx.pl_color(v) for v in d["net"]],
                        hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>"))
                    f1.update_layout(**vx.plotly_layout(max(220, 34 * len(d)), "P&L by strategy",
                                                        xaxis=dict(gridcolor=vx.GRID,
                                                                   tickprefix="$")))
                    st.plotly_chart(f1, width="stretch", key="vx_leg")
                if src.daily is not None and not src.daily.empty:
                    d = src.daily
                    f2 = go.Figure(go.Bar(
                        x=d["date"], y=d["pnl"],
                        marker_color=[vx.pl_color(v) for v in d["pnl"]],
                        hovertemplate="%{x|%d %b}<br>$%{y:,.2f}<extra></extra>"))
                    f2.update_layout(**vx.plotly_layout(240, "Daily P&L",
                                                        yaxis=dict(gridcolor=vx.GRID,
                                                                   tickprefix="$")))
                    st.plotly_chart(f2, width="stretch", key="vx_daily")
                if src.by_symbol is not None and not src.by_symbol.empty:
                    st.markdown(vx.section("By symbol"), unsafe_allow_html=True)
                    st.dataframe(src.by_symbol.style
                                 .map(_color_pnl, subset=["net"])
                                 .format({"net": "${:+,.2f}", "avg": "${:+,.2f}",
                                          "win_rate": "{:.0f}%"}),
                                 width="stretch", hide_index=True)

            st.markdown(vx.section("Signal detail"), unsafe_allow_html=True)
            with st.expander("Raw strategy signals (no lifecycle modelling)"):
                render_weekly_reconstruction("funded")

        # =========================== SYSTEM ===========================
        with t_sys:
            st.markdown(vx.section("Components"), unsafe_allow_html=True)
            comp = []
            for name, pat in (("Funded bot", "run_lucidpro_funded"),
                              ("Trial bot", "run_lucidpro_trial"),
                              ("Chart API", "chart_api"),
                              ("Dashboard", "streamlit")):
                comp.append({"Component": name, "Status": "—"})
            try:
                import subprocess
                ps = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'} "
                     "| Select-Object -ExpandProperty CommandLine"],
                    capture_output=True, text=True, timeout=12).stdout
                for row, pat in zip(comp, ("run_lucidpro_funded", "run_lucidpro_trial",
                                           "chart_api", "streamlit")):
                    row["Status"] = "running" if pat in ps else "stopped"
            except Exception:
                pass
            comp.append({"Component": "MetaTrader 5",
                         "Status": "connected" if mt5_up else "unreachable"})
            st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)

            st.markdown(vx.section("Market sessions"), unsafe_allow_html=True)
            render_market_panel(ms)

            st.markdown(vx.section("Bot state"), unsafe_allow_html=True)
            st.json(state, expanded=False)

            if halted:
                st.markdown(vx.section("Clear halt"), unsafe_allow_html=True)
                st.markdown(vx.banner(
                    "The bot halted itself. Understand <b>why</b> before clearing —"
                    " the halt reason is shown at the top of this page.", "warn"),
                    unsafe_allow_html=True)
                if st.button("Clear halt flag and resume"):
                    state["halted"] = False
                    state["halt_reason"] = None
                    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                    st.success("Halt cleared. Restart the bot for it to take effect.")
                    st.rerun()

            # ---------------- danger zone ----------------
            st.markdown(vx.section("Danger zone"), unsafe_allow_html=True)
            st.markdown(vx.banner(
                "<b>Reset account</b> closes every open position for this book, archives"
                " the state file and trade ledger, and starts the book again from"
                f" ${ANCHOR:,.0f} with a clean history. Archived files are kept and can be"
                " restored below — nothing is deleted. A reset cannot proceed while the"
                " market is closed, because positions could not actually be closed and"
                " would be orphaned.", "danger"), unsafe_allow_html=True)

            confirm = st.text_input(
                'Type RESET to enable the button',
                key="vx_reset_confirm", placeholder="RESET",
                label_visibility="visible")
            col_a, col_b = st.columns([1, 3])
            with col_a:
                do_reset = st.button("Reset account", type="primary",
                                     disabled=(confirm.strip().upper() != "RESET"))
            with col_b:
                if confirm.strip().upper() != "RESET":
                    st.caption("Button unlocks once the confirmation text matches.")
            if do_reset:
                try:
                    eng = MT5Engine(symbol_map={"NAS100": "USTEC", "JP225": "JPN225"})
                    eng.connect()
                except Exception:
                    eng = None
                # Include the pinned v7 funded magics (9220-9226) as well as
                # retired legs; otherwise a reset can leave broker positions
                # open and make the fresh dashboard totals look wrong.
                rep = vxr.reset_book(ROOT_P, "lucidpro_funded", range(9200, 9227),
                                     eng, anchor=ANCHOR)
                if rep.ok:
                    st.success(f"Account reset. Archive: {Path(rep.archive_dir).name}")
                    for s_ in rep.steps:
                        st.caption(f"· {s_}")
                    st.caption("Restart the bot so it picks up the new state.")
                else:
                    st.error(rep.error or "Reset failed.")
                    for s_ in rep.steps:
                        st.caption(f"· {s_}")

            arcs = vxr.list_archives(ROOT_P, "lucidpro_funded")
            if arcs:
                st.markdown(vx.section("Restore points"), unsafe_allow_html=True)
                pick = st.selectbox("Archive",
                                    [a["name"] for a in arcs], key="vx_arc")
                if st.button("Restore selected archive"):
                    ok_, msg = vxr.restore_archive(ROOT_P, pick)
                    (st.success if ok_ else st.error)(msg)

            st.markdown(vx.section("Logs"), unsafe_allow_html=True)
            lp = ROOT_P / "logs" / "lucidpro_funded.log"
            if lp.exists():
                st.code("\n".join(lp.read_text(encoding="utf-8",
                                               errors="replace").splitlines()[-80:]),
                        language=None)

    _lucidpro_funded_view()
    st.stop()

# Everything below is the legacy FTMO ledger view. Each LucidPro branch above
# ends in st.stop(), so this is unreachable in a real Streamlit run -- but
# st.stop() does NOT halt a bare import, so anything importing this module (the
# dashboard tests do) falls through to here with bot_choice="LUCIDPRO_TRIAL" and
# used to die on a KeyError, taking the whole test session's collection with it.
# A default keeps the import survivable without changing the live FTMO path.
target_size = TARGET_ACCOUNT.get(bot_choice, 10_000.0)
target_size = st.sidebar.number_input(
    "Target account size ($)",
    min_value=100.0,
    max_value=1_000_000.0,
    value=target_size,
    step=100.0,
    help="Reference account size for computing return %.",
)
refresh_s = st.sidebar.slider("Live-update every (sec)", 2, 60, 5)

ledger_path = ROOT / "data" / LEDGER_FILE.get(bot_choice, LEDGER_FILE["FTMO_4DAY"])

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:6px;">
  <img class="app-logo" src="{_logo_data_uri(_LOGO_SMALL_PATH)}" style="filter:drop-shadow(0 0 10px rgba(34,211,238,0.6));" />
  <span style="font-size:2.4rem; font-weight:600; letter-spacing:0.5px;
        background:linear-gradient(90deg,#22d3ee,#7fb6ff,#ff4fd8);
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
        filter:drop-shadow(0 0 16px rgba(34,211,238,0.45));">
    Vertex Algo Bot
  </span>
  <span style="font-size:0.9rem; font-weight:600; color:#22d3ee; letter-spacing:1px;
        border:1px solid rgba(34,211,238,0.5); border-radius:20px; padding:3px 12px;
        box-shadow:0 0 14px rgba(34,211,238,0.3); text-shadow:0 0 8px rgba(34,211,238,0.6);">
    {bot_choice}
  </span>
  <span style="font-size:0.72rem; font-weight:600; color:#a3e635; letter-spacing:1px;
        border:1px solid rgba(163,230,53,0.4); border-radius:20px; padding:3px 10px; margin-left:2px;
        box-shadow:0 0 12px rgba(163,230,53,0.2);">SELF-HEALING</span>
</div>
""", unsafe_allow_html=True)

# --- Start/stop control — pauses new trade entries only. Existing open
# positions keep being managed (reconciliation, trailing stop, skyrocket
# exit) regardless of this flag; only new signal execution is gated. ---
control_path = get_control_path(bot_choice)
trading_enabled = read_trading_enabled(control_path)
col_btn, col_status = st.columns([1, 4])
with col_btn:
    if trading_enabled:
        if st.button("⏸ Stop scanning & trading"):
            write_trading_enabled(control_path, False)
            st.rerun()
    else:
        if st.button("▶ Start scanning & trading"):
            write_trading_enabled(control_path, True)
            st.rerun()
with col_status:
    if trading_enabled:
        st.success("Scanning & trading ACTIVE — new signals will open positions.", icon="🟢")
    else:
        st.warning("PAUSED — no new positions will open. Existing open positions are still fully managed (SL/TP/trailing stop).", icon="⏸️")

if not ledger_path.exists():
    st.error(f"Ledger not found: {ledger_path}. Has the {bot_choice} bot ever run?")
    st.stop()


@st.fragment(run_every=refresh_s)
def render_summary() -> None:
    """Liveness banner, ticker chips, and headline KPIs. Wrapping it in a
    fragment means the timer only reruns *this* function, not the whole
    script — no full-page flash or scroll-jump every tick."""
    _signals, _orders, closed, heartbeats = load_ledger(ledger_path)

    # --- Liveness banner ---
    if len(heartbeats):
        last_hb = pd.to_datetime(heartbeats.iloc[0]["ts"], format="ISO8601")
        age_s = (datetime.now(timezone.utc) - last_hb).total_seconds()
        if age_s < 120:
            st.success(f"Bot LIVE — last heartbeat {age_s:.0f}s ago")
        elif age_s < 900:
            st.warning(f"Bot may be stalled — last heartbeat {age_s / 60:.1f} min ago")
        else:
            st.error(f"Bot appears STOPPED — last heartbeat {age_s / 60:.0f} min ago")
    else:
        st.warning("No heartbeats logged yet — bot hasn't started against this ledger.")

    # --- Live open positions, fetched once and reused by both the KPIs above
    # and the "Open positions" section below, so the headline numbers move in
    # real time (mark-to-market) instead of only jumping when a trade closes. ---
    slot_names = load_slot_names(bot_choice)
    live_open = get_live_open_positions(slot_names, block_size=100)
    floating_pnl = float(live_open["Floating P&L"].sum()) if live_open is not None and len(live_open) else 0.0

    # --- Ticker chips, one per open position — quick at-a-glance strip. ---
    if live_open is not None and len(live_open):
        chips = []
        for _, r in live_open.iterrows():
            dot_color = "#4c6ef5" if r["Floating P&L"] >= 0 else "#f5426c"
            chips.append(
                f"<span class='ticker-chip'><span class='ticker-dot' style='background:{dot_color}'></span>"
                f"<b>{r['Symbol']}</b> {r['Dir']} <span style='color:{dot_color}'>${r['Floating P&L']:+,.2f}</span></span>"
            )
        st.markdown("<div style='margin-bottom:10px;'>" + "".join(chips) + "</div>", unsafe_allow_html=True)

    # --- Key metrics ---
    realized_pnl = float(closed["pnl_usd"].sum()) if len(closed) else 0.0
    total_pnl = realized_pnl + floating_pnl
    total_trades = len(closed)
    wins = closed[closed["pnl_usd"] > 0] if len(closed) else pd.DataFrame()
    losses = closed[closed["pnl_usd"] <= 0] if len(closed) else pd.DataFrame()
    wr = 100 * len(wins) / total_trades if total_trades else 0.0

    today = datetime.now(timezone.utc).date().isoformat()
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    day_pnl = floating_pnl
    week_pnl = floating_pnl
    if len(closed):
        day_pnl += float(closed[closed["closed_ts"].str.startswith(today)]["pnl_usd"].sum())
        week_pnl += float(closed[closed["closed_ts"] >= week_ago]["pnl_usd"].sum())

    open_positions = len(live_open) if live_open is not None else (
        int(heartbeats.iloc[0]["open_positions"]) if len(heartbeats) else 0
    )

    # --- Target-relative account view ---
    target_equity = target_size + total_pnl
    return_pct = (total_pnl / target_size * 100) if target_size else 0.0
    day_pct = (day_pnl / target_size * 100) if target_size else 0.0
    week_pct = (week_pnl / target_size * 100) if target_size else 0.0

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Virtual equity",
            f"${target_equity:,.2f}",
            delta=f"{return_pct:+.2f}%",
            help="Realized P&L + current floating P&L from open positions.",
        )
        c2.metric("Total P&L", f"${total_pnl:+,.2f}", delta=f"{return_pct:+.2f}%")
        c3.metric(
            "Floating (open)",
            f"${floating_pnl:+,.2f}",
            help="Unrealized P&L of currently open positions — moves live as price moves.",
        )
        c4.metric("Open positions", open_positions)

        c5, c6, c7 = st.columns(3)
        c5.metric("Today", f"${day_pnl:+,.2f}", delta=f"{day_pct:+.2f}%")
        c6.metric("Last 7d", f"${week_pnl:+,.2f}", delta=f"{week_pct:+.2f}%")
        c7.metric(
            "Win rate",
            f"{wr:.1f}%",
            help=f"{len(wins)}W / {len(losses)}L over {total_trades} trades (closed trades only)",
        )


@st.fragment(run_every=refresh_s)
def render_closed_positions() -> None:
    """Closed positions card only — split out from the rest of what used to
    be render_history (equity curve, per-strategy stats, raw tables) so that
    render_trade_review() (called right after this one, see the bottom of the
    file) renders its replay animation directly below the table it was
    opened from, instead of scrolled past all that other content too.
    Independently fetching the ledger — a small duplicate read, same pattern
    already used elsewhere in this file — keeps this section's own refresh
    timer independent of the others."""
    _signals, _orders, closed, _heartbeats = load_ledger(ledger_path)
    if len(closed):
        # Slovakia-local calendar date per trade — everything else in this
        # table is already shown in SK time, so the day filter needs to
        # bucket by the same clock or a trade near midnight could land in
        # the "wrong" day relative to what's displayed.
        closed = closed.copy()
        closed["_closed_date_sk"] = pd.to_datetime(closed["closed_ts"], format="ISO8601").apply(
            lambda d: to_slovakia(d).date()
        )

    with st.container(border=True):
        # --- Closed positions, one clear card per trade — realized_pnl above is
        # just the sum of these. Full raw table still in the "Closed positions"
        # tab further down for anyone who wants every column. ---
        st.subheader("🧾 Closed positions")
        if not len(closed):
            st.info("No closed positions yet.")
        else:
            available_days = sorted(closed["_closed_date_sk"].unique(), reverse=True)
            day_counts = closed["_closed_date_sk"].value_counts()
            day_options = ["All days"] + [
                f"{d.strftime('%Y-%m-%d (%a)')} — {day_counts[d]} trades" for d in available_days
            ]
            day_key_map = dict(zip(day_options[1:], available_days))
            selected_label = st.selectbox("Filter by day", day_options, key="closed_day_filter_select")
            selected_day = day_key_map.get(selected_label)  # None means "All days"
            display_closed = closed if selected_day is None else closed[closed["_closed_date_sk"] == selected_day]

            realized_pnl = float(display_closed["pnl_usd"].sum())
            n_win = int((display_closed["pnl_usd"] > 0).sum())
            n_loss = int((display_closed["pnl_usd"] <= 0).sum())
            day_suffix = "" if selected_day is None else f" on {selected_day.strftime('%Y-%m-%d')}"
            st.markdown(
                f"<small style='opacity:0.75'>{len(display_closed)} closed{day_suffix} &nbsp;|&nbsp; "
                f"{_dot('#4c6ef5')}{n_win} wins &nbsp;|&nbsp; {_dot('#f5426c')}{n_loss} losses &nbsp;|&nbsp; "
                f"Realized total: <b>${realized_pnl:+,.2f}</b></small>",
                unsafe_allow_html=True,
            )
            strat_by_name = load_strategy_by_config_name(bot_choice)
            col_widths = [1.6, 0.7, 0.4, 1.3, 0.9, 1.1, 1.1, 0.8, 0.6]
            hdr = st.columns(col_widths)
            for col, label in zip(hdr, ["Strategy", "Symbol", "Dir", "Entry → Exit", "P&L", "Opened (SK)", "Closed (SK)", "Duration", ""]):
                col.markdown(f"**{label}**")

            for _, row in display_closed.sort_values("closed_ts", ascending=False).iterrows():
                cols = st.columns(col_widths)
                cols[0].markdown(
                    strategy_cell_html(strat_by_name.get(row["config_name"], ""), row["config_name"], show_explain=False),
                    unsafe_allow_html=True,
                )
                cols[1].write(row["symbol"])
                is_long = str(row["direction"]).lower() in ("long", "buy")
                cols[2].markdown(dir_badge(is_long), unsafe_allow_html=True)
                cols[3].write(f"{row['entry_price']:.5f} → {row['exit_price']:.5f}")
                pnl = row["pnl_usd"]
                win = pnl > 0
                pnl_color = "#4c6ef5" if win else "#f5426c"
                reason = {"tp": "hit TP", "sl": "hit SL", "manual": "manual close"}.get(row["exit_reason"], row["exit_reason"])
                cols[4].markdown(
                    f"{_dot(pnl_color)}<span style='color:{pnl_color}; font-weight:600'>${pnl:+,.2f}</span>"
                    f"<br><small style='opacity:0.6'>{reason}</small>",
                    unsafe_allow_html=True,
                )
                opened_dt = pd.to_datetime(row["opened_ts"], format="ISO8601")
                closed_dt = pd.to_datetime(row["closed_ts"], format="ISO8601")
                opened_sk = to_slovakia(opened_dt)
                closed_sk = to_slovakia(closed_dt)
                cols[5].write(opened_sk.strftime("%Y-%m-%d %H:%M:%S"))
                cols[6].write(closed_sk.strftime("%Y-%m-%d %H:%M:%S"))
                cols[7].write(_fmt_duration(closed_dt - opened_dt))
                if cols[8].button("📈", key=f"review_{row['ticket']}", help="See how this trade played out"):
                    # Broker-clock offset + padded fetch range computed ONCE,
                    # right here, and frozen into session_state — NOT
                    # recomputed inside the chart builder on every fragment
                    # rerun (that used to bake a freshly-measured, slightly
                    # jittery timestamp into the HTML each cycle, which made
                    # Streamlit think the component's content changed and
                    # remount the iframe — restarting the animation and
                    # flashing every refresh tick).
                    offset = _broker_offset_seconds(row["symbol"])
                    opened_broker_ts = int(opened_dt.timestamp()) + offset
                    closed_broker_ts = int(closed_dt.timestamp()) + offset
                    duration_s = max(closed_broker_ts - opened_broker_ts, 60)
                    # Just the trade itself, plus a little room on each side
                    # to render the entry/exit bars cleanly and show what
                    # happened right after — no full surrounding context.
                    # Context padding made the y-axis stretch to cover
                    # whatever happened before/after the trade too, squashing
                    # the trade's own move down to an unreadable sliver.
                    pre_padding_s = 120
                    post_padding_s = 600
                    st.session_state["review_trade"] = {
                        "symbol": row["symbol"], "is_long": is_long,
                        "entry": float(row["entry_price"]), "sl": float(row["sl"]), "tp": float(row["tp"]),
                        "exit_price": float(row["exit_price"]), "pnl": float(pnl), "exit_reason": row["exit_reason"],
                        "opened_broker_ts": opened_broker_ts, "closed_broker_ts": closed_broker_ts,
                        "ts_from": opened_broker_ts - pre_padding_s, "ts_to": closed_broker_ts + post_padding_s,
                    }
                    # This fragment reruns on its own every refresh_s (see the
                    # decorator above) — the actual chart is rendered by
                    # render_trade_review(), a SEPARATE fragment with no timer,
                    # so the animation isn't restarted every few seconds. A
                    # plain st.rerun() (full-app scope, the default even when
                    # called from inside a fragment) is needed here once, to
                    # make that other fragment notice the new selection.
                    st.rerun()


@st.fragment(run_every=refresh_s)
def render_equity_and_stats() -> None:
    """Equity curve, per-strategy stats, and the raw signal/order/closed
    tables — everything else that used to live in render_history (see
    render_closed_positions above for why it was split out). Independently
    fetching the ledger + live-position data render_closed_positions and
    render_open_positions already fetch — a small duplicate read, same
    pattern already used elsewhere in this file."""
    signals, orders, closed, _heartbeats = load_ledger(ledger_path)
    slot_names = load_slot_names(bot_choice)
    live_open = get_live_open_positions(slot_names, block_size=100)
    floating_pnl = float(live_open["Floating P&L"].sum()) if live_open is not None and len(live_open) else 0.0
    realized_pnl = float(closed["pnl_usd"].sum()) if len(closed) else 0.0
    total_pnl = realized_pnl + floating_pnl
    target_equity = target_size + total_pnl

    with st.container(border=True):
        # --- Live equity curve, ALWAYS normalized to target account size ---
        st.subheader(f"Virtual equity — ${target_size:,.0f} account")

        # Built from THIS bot's own closed_positions (isolated to its own trades),
        # not from heartbeats. Heartbeats log the whole shared MT5 account's
        # equity, which includes the OTHER bot's floating P&L too (FTMO and YOLO
        # share one MT5 account) — that mismatch is exactly what made the chart
        # show a different number than the real per-bot balance. The final point
        # always equals target_equity above, so the chart and the KPI card can't
        # disagree.
        if len(closed):
            eq = closed[["closed_ts", "pnl_usd"]].copy().sort_values("closed_ts")
            eq["closed_ts"] = pd.to_datetime(eq["closed_ts"], format="ISO8601")
            eq["virtual_equity"] = target_size + eq["pnl_usd"].cumsum()
            now_point = pd.DataFrame([{
                "closed_ts": datetime.now(timezone.utc), "virtual_equity": target_equity,
            }])
            curve = pd.concat([eq[["closed_ts", "virtual_equity"]], now_point], ignore_index=True)
        else:
            curve = pd.DataFrame([
                {"closed_ts": datetime.now(timezone.utc) - timedelta(minutes=1), "virtual_equity": target_size},
                {"closed_ts": datetime.now(timezone.utc), "virtual_equity": target_equity},
            ])
        st.line_chart(curve.set_index("closed_ts")[["virtual_equity"]])

    with st.container(border=True):
        # --- Per-strategy improving equity curves — one neon line per
        # strategy, cumulative P&L over time (the "how each strategy is
        # improving" graph). ---
        st.subheader("📈 Strategy equity curves — each strategy over time")
        if len(closed):
            cc = closed[["config_name", "closed_ts", "pnl_usd"]].dropna(subset=["pnl_usd"]).copy()
            cc = cc.sort_values("closed_ts")
            # Group by STRATEGY, not per pair: "Donchian__XAUUSD" -> "Donchian".
            # All of a strategy's per-symbol legs sum into one climbing line.
            cc["strategy"] = cc["config_name"].map(strategy_label)
            series_by_config: dict[str, list] = {}
            for name, grp in cc.groupby("strategy"):
                cum = grp["pnl_usd"].cumsum()
                pts = [(_epoch(ts), float(v)) for ts, v in zip(grp["closed_ts"], cum) if _epoch(ts)]
                # de-dup identical timestamps (lightweight-charts requires strictly increasing time)
                seen = {}
                for t, v in pts:
                    seen[t] = v
                if len(seen) >= 1:
                    series_by_config[str(name)] = sorted(seen.items())
            if series_by_config:
                components.html(build_strategy_curves_html(series_by_config, height=360),
                                height=430, scrolling=False)
            else:
                st.caption("No closed trades yet — curves appear as trades close.")
        else:
            st.caption("No closed trades yet — curves appear as each strategy starts trading.")

    with st.container(border=True):
        # --- Self-healing / learning: the "why it happened" memory + what the
        # bot has learned to avoid. This is the honest version of the video's
        # "every win/loss AI answers why and improves the strategy". ---
        st.subheader("🧠 Self-healing — what the bot learned from each trade")
        recent_rv, agg_rv = load_reviews(ledger_path)
        if len(agg_rv):
            pills = []
            for _, r in agg_rv.sort_values("net_r").iterrows():
                avoid = (r["n"] >= 6 and r["net_r"] <= -3.0)
                cls = "avoid" if avoid else "ok"
                label = ("avoiding" if avoid else "ok")
                pills.append(
                    f"<span class='verdict-pill {cls}'>{r['config_name']} · {r['session']}: "
                    f"{int(r['n'])} trades, {r['net_r']:+.1f}R — {label}</span>")
            st.markdown("<div class='neon-panel'><b style='color:#c7d2e0;'>Learned session verdicts</b><br>"
                        + "".join(pills) + "</div>", unsafe_allow_html=True)
        if len(recent_rv):
            rows_html = []
            for _, r in recent_rv.iterrows():
                win = bool(r["was_win"])
                cls = "" if win else "loss"
                tag = "win" if win else "loss"
                rows_html.append(
                    f"<div class='why-row {cls}'><span class='why-tag {tag}'>"
                    f"{'WIN' if win else 'LOSS'} {r['r_multiple']:+.2f}R</span>"
                    f"<b>{r['config_name']}</b> · {r['symbol']} — {r['why']}</div>")
            st.markdown("<div style='margin-top:10px;'>" + "".join(rows_html) + "</div>",
                        unsafe_allow_html=True)
        elif not len(agg_rv):
            st.caption("The learning log fills in as trades close — each one gets a stored "
                       "post-mortem here, and repeated losses in a session bench that slot.")

    with st.container(border=True):
        # --- Per-strategy breakdown ---
        st.subheader("Per-strategy performance")
        if len(closed):
            closed_typed = closed.copy()
            # Robust for both new (Strategy__SYMBOL) and old per-pair names.
            closed_typed["strategy_type"] = closed_typed["config_name"].map(strategy_label)
            grp = (
                closed_typed.groupby("strategy_type")
                .agg(
                    trades=("id", "count"),
                    wins=("pnl_usd", lambda x: int((x > 0).sum())),
                    total_pnl=("pnl_usd", "sum"),
                )
                .reset_index()
            )
            grp["win_rate_%"] = (grp["wins"] / grp["trades"] * 100).round(1)
            grp = grp[["strategy_type", "win_rate_%", "total_pnl"]].round(2).sort_values("total_pnl", ascending=False)
            grp = grp.rename(columns={"strategy_type": "Strategy", "win_rate_%": "Win rate", "total_pnl": "P&L"})
            styled_grp = grp.style.map(_color_pnl, subset=["P&L"]).format({
                "P&L": "${:+,.2f}", "Win rate": "{:.1f}%",
            })
            st.dataframe(styled_grp, width="stretch", hide_index=True)
        else:
            st.info("No closed positions yet — waiting for first trade.")

    with st.container(border=True):
        # --- Signals / trades tabs ---
        tab_sig, tab_closed, tab_orders = st.tabs(
            ["Signals (last 30)", "Closed positions (last 30)", "Orders (last 30)"]
        )

        with tab_sig:
            if len(signals):
                display = signals[[
                    "ts", "config_name", "symbol", "timeframe", "direction",
                    "entry_price", "stop_loss", "take_profit", "lots",
                ]].head(30).copy()
                display["direction"] = display["direction"].map(_dir_icon)
                display["ts"] = pd.to_datetime(display["ts"], format="ISO8601").apply(
                    lambda d: to_slovakia(d).strftime("%Y-%m-%d %H:%M:%S")
                )
                display = display.rename(columns={"ts": "ts (SK time)"})
                styled_sig = display.style.map(_color_dir, subset=["direction"])
                st.dataframe(styled_sig, width="stretch", hide_index=True)
            else:
                st.info("No signals fired yet.")

        with tab_closed:
            if len(closed):
                display = closed[[
                    "closed_ts", "opened_ts", "config_name", "symbol", "direction",
                    "entry_price", "exit_price", "sl", "tp", "lots", "pnl_usd", "exit_reason",
                ]].head(30).copy()
                display["direction"] = display["direction"].map(_dir_icon)
                opened_dt = pd.to_datetime(display["opened_ts"], format="ISO8601")
                closed_dt = pd.to_datetime(display["closed_ts"], format="ISO8601")
                display.insert(2, "duration", (closed_dt - opened_dt).apply(_fmt_duration))
                display["opened_ts"] = opened_dt.apply(lambda d: to_slovakia(d).strftime("%Y-%m-%d %H:%M:%S"))
                display["closed_ts"] = closed_dt.apply(lambda d: to_slovakia(d).strftime("%Y-%m-%d %H:%M:%S"))
                display = display.rename(columns={"opened_ts": "opened (SK time)", "closed_ts": "closed (SK time)"})
                n_win = int((display["pnl_usd"] > 0).sum())
                n_loss = int((display["pnl_usd"] <= 0).sum())
                st.markdown(
                    f"<small style='opacity:0.75'>Of these {len(display)}: "
                    f"{_dot('#4c6ef5')}{n_win} wins &nbsp;|&nbsp; {_dot('#f5426c')}{n_loss} losses</small>",
                    unsafe_allow_html=True,
                )
                styled_closed = display.style.map(_color_pnl, subset=["pnl_usd"]).map(_color_dir, subset=["direction"]).format({
                    "pnl_usd": "${:+,.2f}", "entry_price": "{:.5f}", "exit_price": "{:.5f}",
                    "sl": "{:.5f}", "tp": "{:.5f}", "lots": "{:.2f}",
                })
                st.dataframe(styled_closed, width="stretch", hide_index=True)
            else:
                st.info("No closed positions yet.")

        with tab_orders:
            if len(orders):
                display = orders[[
                    "ts", "config_name", "symbol", "direction",
                    "requested_price", "filled_price", "requested_lots", "filled_lots",
                    "sl", "tp", "retcode", "comment",
                ]].head(30).copy()
                display["direction"] = display["direction"].map(_dir_icon)
                display["ts"] = pd.to_datetime(display["ts"], format="ISO8601").apply(
                    lambda d: to_slovakia(d).strftime("%Y-%m-%d %H:%M:%S")
                )
                display = display.rename(columns={"ts": "ts (SK time)"})
                display.insert(0, "status", display["retcode"].map(lambda r: "✅" if r == 10009 else "❌"))
                n_ok = int((display["retcode"] == 10009).sum())
                n_fail = len(display) - n_ok
                st.caption(f"Of these {len(display)}: ✅ **{n_ok} filled**  |  ❌ **{n_fail} rejected**")

                def _status_color(val):
                    return "background-color: rgba(245,66,108,0.18)" if val == "❌" else ""

                styled_orders = display.style.map(_status_color, subset=["status"]).map(_color_dir, subset=["direction"])
                st.dataframe(styled_orders, width="stretch", hide_index=True)
            else:
                st.info("No orders yet.")

    # --- Footer ---
    st.caption(
        f"Data source: `{ledger_path.name}`  |  "
        f"Target account size: ${target_size:,.0f}  |  "
        f"Live-updating every {refresh_s}s  |  "
        f"Refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
    )


@st.fragment
def render_open_positions() -> None:
    """Open positions — one card per position with its info AND its live
    chart together, instead of a separate table + a separate charts section
    repeating the same positions. No Streamlit-level auto-timer here on
    purpose: each chart is a self-contained JS component (lightweight-charts)
    that fetches from chart_api.py and updates ITSELF on its own JS interval
    (current price, floating P&L, and open-for duration are all rendered by
    that same JS loop, in the chart's stats bar) — completely independent of
    Streamlit's rerun cycle, which is what actually removes the flicker.
    The button below only exists to pick up newly opened/closed positions
    (a change in which tickets exist), not to refresh price data."""
    st.subheader("🔓 Open positions")
    st.button("🔄 Check for new/closed positions", key="refresh_charts_btn")
    slot_names = load_slot_names(bot_choice)
    live_open = get_live_open_positions(slot_names, block_size=100)
    if live_open is None:
        st.info("MT5 not reachable right now — open positions unavailable.")
        return
    if live_open.empty:
        st.info("No open positions right now.")
        return
    # One full-width panel per position, stacked — a 2-per-row grid made each
    # chart too small to read clearly.
    for _, row in live_open.iterrows():
        with st.container(border=True):
            info = STRATEGY_INFO.get(row["StrategyType"], {"icon": "❔", "label": row["StrategyType"] or "Unknown", "explain": ""})
            is_long = row["Dir"] == UP
            st.markdown(
                f"**{info['icon']} {info['label']}** — {row['Symbol']} {dir_badge(is_long)} "
                f"&nbsp;Vol {row['Volume']:.2f} &nbsp;Entry {row['Entry']:.5f}"
                f"<br><small style='opacity:0.65'>{info['explain']}</small>",
                unsafe_allow_html=True,
            )
            ticket = int(row["Ticket"])
            if st.button("❌ Close", key=f"close_{ticket}"):
                ok, msg = close_ticket(ticket)
                if ok:
                    st.success(f"Ticket {ticket} ({row['Symbol']}): {msg}")
                else:
                    st.error(f"Ticket {ticket} ({row['Symbol']}): {msg}")
                st.rerun()
            html = build_live_chart_html(row["Symbol"], ticket, is_long, default_timeframe="15m", height=420)
            # 420 (chart) + ~120 for the PNL row, toolbar row, and padding
            # above it — the previous 490 was ~20px short, clipping the
            # chart's own time-axis labels and watermark off the bottom
            # since components.html hard-clips overflow (scrolling=False).
            components.html(html, height=540, scrolling=False)


def _close_trade_review() -> None:
    st.session_state["review_trade"] = None


@st.dialog("🎬 Trade replay", width="large", on_dismiss=_close_trade_review)
def _trade_review_dialog(review: dict) -> None:
    """The actual replay dialog body — a real st.dialog modal (not an inline
    block) so it covers most of the viewport and needs no scrolling to reach,
    regardless of where in the closed-positions table the "📈" button was
    clicked. Sized up via the '.stApp div[data-testid="stDialog"]' CSS rule
    near the top of the file (the built-in "large" width is still only
    ~800px — nowhere near enough room to read a candlestick chart)."""
    if st.button("✕ Close replay"):
        st.session_state["review_trade"] = None
        st.rerun()
    html = build_closed_trade_chart_html(
        review["symbol"], review["is_long"], review["entry"], review["sl"], review["tp"],
        review["exit_price"], review["pnl"], review["exit_reason"],
        review["opened_broker_ts"], review["closed_broker_ts"],
        review["ts_from"], review["ts_to"],
        height=620,
    )
    components.html(html, height=680, scrolling=False)


def render_trade_review() -> None:
    """Triggers the trade-replay dialog. NOT a fragment — a plain function is
    enough since it only needs to run once on a genuine full-script rerun
    (which the "📈" button's st.rerun() call forces, see
    render_closed_positions); Streamlit itself then keeps the dialog open
    across any further reruns for as long as review_trade stays set, without
    this function needing to be called again."""
    review = st.session_state.get("review_trade")
    if review:
        _trade_review_dialog(review)


render_summary()
render_open_positions()
render_closed_positions()
render_trade_review()
render_equity_and_stats()
