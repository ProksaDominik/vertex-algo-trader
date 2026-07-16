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
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metatraderbot.mt5_engine import MT5Engine  # noqa: E402 — reuse filling-mode-aware close logic

# Target account sizes per bot — this is what P&L % is computed against,
# regardless of the actual MetaQuotes-Demo balance ($100k default).
TARGET_ACCOUNT = {
    "FTMO": 10_000.0,
    "YOLO": 200.0,
}

CONFIG_FILE = {
    "FTMO": "portfolio_ftmo.yaml",
    "YOLO": "portfolio_yolo.yaml",
}


def get_control_path(bot_choice: str) -> Path:
    """Path to the shared start/stop flag file — same naming convention
    run_paper.py uses (derived from the config filename's stem), so both
    processes agree on the same file without any other coordination."""
    stem = Path(CONFIG_FILE[bot_choice]).stem  # "portfolio_ftmo" / "portfolio_yolo"
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


def strategy_cell_html(strategy_type: str, config_name: str, show_explain: bool = True) -> str:
    """Small HTML block: strategy display name + icon (+ optional one-line
    explanation), used inline next to a trade so the 'which strategy' info
    travels with every row instead of living only in docs."""
    info = STRATEGY_INFO.get(strategy_type, {"icon": "❔", "label": strategy_type or "Unknown", "explain": ""})
    html = f"<span style='font-weight:600'>{info['icon']} {info['label']}</span>"
    if show_explain:
        html += f"<br><small style='opacity:0.65'>{info['explain']}</small>"
    html += f"<br><small style='opacity:0.4'>{config_name}</small>"
    return html


def load_slot_names(bot_choice: str) -> dict[int, dict]:
    """magic_number -> config info, so live MT5 positions can be labeled."""
    path = ROOT / "config" / CONFIG_FILE[bot_choice]
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        c["magic_number"]: {"name": c["name"], "symbol": c["symbol"], "timeframe": c["timeframe"], "strategy": c["strategy"]}
        for c in cfg.get("configs", [])
    }


def load_strategy_by_config_name(bot_choice: str) -> dict[str, str]:
    """config_name -> strategy type, for ledger rows (closed/signals/orders)
    which only store the config name, not the live magic number."""
    path = ROOT / "config" / CONFIG_FILE[bot_choice]
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {c["name"]: c["strategy"] for c in cfg.get("configs", [])}


def get_live_open_positions(slots: dict[int, dict]) -> pd.DataFrame | None:
    """Live open positions straight from MT5 (floating P&L, current price).

    Returns None (not an empty frame) when MT5 can't be reached, so callers
    can fall back to the heartbeat-based count instead of showing "0 open"
    when the truth is "unknown".
    """
    if not _MT5_AVAILABLE or not slots:
        return None
    if not mt5.initialize():
        return None
    positions = mt5.positions_get() or []
    rows = []
    for p in positions:
        if p.magic not in slots:
            continue
        rows.append({
            "Ticket": p.ticket,
            "Magic": p.magic,
            "Strategy": slots[p.magic]["name"],
            "StrategyType": slots[p.magic].get("strategy", ""),
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


# Blue(up)/red-pink(down) palette — used consistently everywhere (charts,
# P&L text, buttons) instead of the traditional green/red.
_BG = "#0a0d16"
_GRID = "#161b28"
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
<div id="chart_wrap_{ticket}" style="background:{_BG}; padding:8px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
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
  let slLine = null, tpLine = null, markersSet = false, hadPosition = false;
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
        }} else {{
          slLine.applyOptions({{price: p.sl}});
          tpLine.applyOptions({{price: p.tp}});
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
          const openedFor = fmtDuration(lastT - p.opened_time);
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

    reason_label = {"tp": "hit take-profit", "sl": "hit stop-loss (or trailing stop)", "manual": "closed manually"}.get(exit_reason, exit_reason)

    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  #chart_wrap_{div_id} * {{ font-family: 'Inter', sans-serif; }}
  #chart_wrap_{div_id} button:hover {{ background:rgba(255,255,255,0.1) !important; }}
  #chart_wrap_{div_id} button:disabled {{ opacity:0.35; cursor:default; }}
</style>
<div id="chart_wrap_{div_id}" style="background:{_BG}; padding:8px; border-radius:10px; border:1px solid rgba(255,255,255,0.06);">
  <div style="margin-bottom:8px; display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <div style="font-size:1.4rem; font-weight:700; color:{exit_color};">
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
      // candles get revealed each tick.
      let lo = Math.min({sl}, {tp}, {entry}, {exit_price});
      let hi = Math.max({sl}, {tp}, {entry}, {exit_price});
      for (const b of allBars) {{ lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); }}
      const firstT = allBars[0].time, lastT = allBars[allBars.length - 1].time;
      rangeSeries.setData([{{time: firstT, value: lo}}, {{time: lastT, value: hi}}]);
      chart.timeScale().setVisibleRange({{from: firstT, to: lastT}});

      // SL/TP/Entry/Exit as persistent reference lines, visible for the
      // entire animation regardless of how many candles have been revealed.
      series.createPriceLine({{price: {sl}, color: "{_SL_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: "SL"}});
      series.createPriceLine({{price: {tp}, color: "{_TP_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: "TP"}});
      series.createPriceLine({{price: {entry}, color: "{_ENTRY_COLOR}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, title: "Entry"}});
      series.createPriceLine({{price: {exit_price}, color: "{exit_color}", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, title: "Exit"}});

      entryBarTime = snapToBar(allBars, {opened_broker_ts});
      exitBarTime = snapToBar(allBars, {closed_broker_ts});

      // Reveal in real time where possible, but cap total animation length
      // so a multi-hour trade doesn't take minutes to play out.
      const targetTotalMs = 9000, minFrameMs = 15, maxFrameMs = 120;
      frameMs = Math.min(maxFrameMs, Math.max(minFrameMs, targetTotalMs / allBars.length));
      barsPerTick = frameMs === minFrameMs ? Math.max(1, Math.ceil((allBars.length * minFrameMs) / targetTotalMs)) : 1;

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


st.set_page_config(page_title="Vertex Algo Bot", layout="wide", page_icon=str(_LOGO_PATH))

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
    --bg: #0b0e16;
    --panel: #131722;
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

# --- Sidebar (outside the live fragment — changing these does a normal full rerun) ---
st.sidebar.header("Config")
bot_choice = st.sidebar.radio(
    "Bot",
    ("FTMO", "YOLO"),
    help="Pick which bot's ledger to view.",
)
target_size = TARGET_ACCOUNT[bot_choice]
target_size = st.sidebar.number_input(
    "Target account size ($)",
    min_value=100.0,
    max_value=1_000_000.0,
    value=target_size,
    step=100.0,
    help="Reference account size for computing return %.",
)
refresh_s = st.sidebar.slider("Live-update every (sec)", 2, 60, 5)

ledger_path = ROOT / "data" / (
    "paper_ledger.sqlite" if bot_choice == "FTMO" else "paper_ledger_yolo.sqlite"
)

st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:10px;">
  <img class="app-logo" src="{_logo_data_uri(_LOGO_SMALL_PATH)}" />
  <span style="font-size:2.3rem; font-weight:600; color:#eef2f6;">
    Vertex Algo Bot
  </span>
  <span style="font-size:1.05rem; color:#4c6ef5; opacity:0.9;">
    {bot_choice}
  </span>
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
    live_open = get_live_open_positions(slot_names)
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
    realized_pnl = float(closed["pnl_usd"].sum()) if len(closed) else 0.0

    with st.container(border=True):
        # --- Closed positions, one clear card per trade — realized_pnl above is
        # just the sum of these. Full raw table still in the "Closed positions"
        # tab further down for anyone who wants every column. ---
        st.subheader("🧾 Closed positions")
        if not len(closed):
            st.info("No closed positions yet.")
        else:
            n_win = int((closed["pnl_usd"] > 0).sum())
            n_loss = int((closed["pnl_usd"] <= 0).sum())
            st.markdown(
                f"<small style='opacity:0.75'>{len(closed)} closed &nbsp;|&nbsp; "
                f"{_dot('#4c6ef5')}{n_win} wins &nbsp;|&nbsp; {_dot('#f5426c')}{n_loss} losses &nbsp;|&nbsp; "
                f"Realized total: <b>${realized_pnl:+,.2f}</b></small>",
                unsafe_allow_html=True,
            )
            strat_by_name = load_strategy_by_config_name(bot_choice)
            col_widths = [1.6, 0.7, 0.4, 1.3, 0.9, 1.1, 1.1, 0.8, 0.6]
            hdr = st.columns(col_widths)
            for col, label in zip(hdr, ["Strategy", "Symbol", "Dir", "Entry → Exit", "P&L", "Opened (SK)", "Closed (SK)", "Duration", ""]):
                col.markdown(f"**{label}**")

            for _, row in closed.sort_values("closed_ts", ascending=False).iterrows():
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
                    # Pad by as much as the trade itself lasted on each side
                    # (min 30 min, max 3h) so the replay shows real market
                    # context before/after the trade instead of a tight crop
                    # that looks zoomed in on just the entry-to-exit move.
                    padding_s = int(min(max(duration_s * 1.0, 30 * 60), 3 * 3600))
                    st.session_state["review_trade"] = {
                        "symbol": row["symbol"], "is_long": is_long,
                        "entry": float(row["entry_price"]), "sl": float(row["sl"]), "tp": float(row["tp"]),
                        "exit_price": float(row["exit_price"]), "pnl": float(pnl), "exit_reason": row["exit_reason"],
                        "opened_broker_ts": opened_broker_ts, "closed_broker_ts": closed_broker_ts,
                        "ts_from": opened_broker_ts - padding_s, "ts_to": closed_broker_ts + padding_s,
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
    live_open = get_live_open_positions(slot_names)
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
        # --- Per-strategy breakdown ---
        st.subheader("Per-strategy performance")
        if len(closed):
            strat_by_name = load_strategy_by_config_name(bot_choice)
            closed_typed = closed.copy()
            closed_typed["strategy_type"] = closed_typed["config_name"].map(
                lambda n: STRATEGY_INFO.get(strat_by_name.get(n, ""), {}).get("label", strat_by_name.get(n, "Unknown"))
            )
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
            st.dataframe(styled_grp, use_container_width=True, hide_index=True)
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
                st.dataframe(styled_sig, use_container_width=True, hide_index=True)
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
                st.dataframe(styled_closed, use_container_width=True, hide_index=True)
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
                st.dataframe(styled_orders, use_container_width=True, hide_index=True)
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
    live_open = get_live_open_positions(slot_names)
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
