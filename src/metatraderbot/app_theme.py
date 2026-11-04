"""Design system for the Vertex desktop app.

One place for every colour, type scale and chart default, so the UI reads as a
single product rather than a pile of independently-styled widgets.

Colour decisions and why they are what they are:

* Profit/loss is a DIVERGING encoding (polarity around zero), so it gets two
  hues and a neutral midpoint -- never a single ramp. The pair is the project's
  existing blue/rose rather than the conventional green/red: it stays readable
  for red-green colour blindness, which green/red does not. Validated against
  the dark app surface (#0d1117): worst-pair CVD deltaE 21.9 protan / 35.7
  tritan, normal-vision 33.6, contrast >= 3:1 -- passes every gate with room to
  spare.
* Status colours (halted, warning, live) are RESERVED. They never double as a
  series colour, and they always ship with a text label rather than relying on
  colour alone.
* Series colours are assigned in fixed order and never cycled. A ninth series
  folds into "Other" rather than inventing a hue.
"""
from __future__ import annotations

import base64
from pathlib import Path


def _mark_uri() -> str:
    """The logo -- the black, white and grey split V (owner, 2026-10-21: "dark black and white or grey, more simple
    and more aesthetic") -- as an image for the small badge at the top of every page; "" without the file."""
    p = Path(__file__).resolve().parents[2] / "scripts" / "assets" / "vertex_v_icon_small.png"
    try:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except OSError:
        return ""


MARK_URI = _mark_uri()

# --- surfaces (dark-first: this is a always-on monitoring app) ---------------
BG          = "#0d1117"   # app background
SURFACE     = "#11161f"   # cards
SURFACE_ALT = "#161c27"   # nested / table header
BORDER      = "#1e2633"
BORDER_SOFT = "#171d28"

# --- ink ---------------------------------------------------------------------
TEXT        = "#e6edf3"
TEXT_MUTED  = "#8b98a9"
TEXT_FAINT  = "#5c6775"

# --- diverging: profit / loss ------------------------------------------------
POS         = "#4c6ef5"
POS_SOFT    = "#2b3f8f"
NEG         = "#f5426c"
NEG_SOFT    = "#8f2340"
NEUTRAL     = "#5c6775"

# --- reserved status ---------------------------------------------------------
OK          = "#2ea86b"
WARN        = "#d9a441"
DANGER      = "#e5484d"
INFO        = "#4c6ef5"

# --- categorical series, fixed order, never cycled ---------------------------
SERIES = ["#4c6ef5", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#9085e9", "#3fb6d3", "#c2705a"]

GRID = "#1a2130"
ACCENT = "#4c6ef5"


def pl_color(v: float | int | None) -> str:
    """Diverging colour for a P&L value. Neutral at exactly zero."""
    if v is None:
        return NEUTRAL
    try:
        f = float(v)
    except (TypeError, ValueError):
        return NEUTRAL
    return POS if f > 0 else (NEG if f < 0 else NEUTRAL)


def plotly_layout(height: int = 300, title: str | None = None, **kw) -> dict:
    """Shared Plotly layout: recessive axes, transparent surface, no chartjunk."""
    lay = dict(
        height=height,
        margin=dict(l=8, r=8, t=34 if title else 8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12,
                  family="Inter, 'Segoe UI', system-ui, sans-serif"),
        showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE_ALT, bordercolor=BORDER,
                        font=dict(color=TEXT, size=12)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER,
                   showspikes=False),
        yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER),
    )
    if title:
        lay["title"] = dict(text=title, font=dict(color=TEXT, size=13), x=0, xanchor="left")
    lay.update(kw)
    return lay


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
  font-family: Inter, 'Segoe UI', system-ui, sans-serif;
}}
.stApp {{ background: {BG}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.1rem 1.6rem 3rem; max-width: 1600px; }}

/* ---- app chrome ---- */
.vx-top {{
  display:flex; align-items:center; gap:14px;
  padding:0 0 14px; border-bottom:1px solid {BORDER}; margin-bottom:18px;
}}
.vx-mark {{
  width:30px; height:30px; border-radius:7px; overflow:hidden;
  background:#0e0e10 url("{MARK_URI}") center/cover no-repeat;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; color:{'transparent' if MARK_URI else '#fff'}; font-size:15px; letter-spacing:-0.5px;
}}
.vx-name {{ font-size:15px; font-weight:600; color:{TEXT}; letter-spacing:-0.2px; }}
.vx-sub  {{ font-size:12px; color:{TEXT_FAINT}; }}
.vx-spacer {{ flex:1; }}

.vx-pill {{
  display:inline-flex; align-items:center; gap:6px;
  padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600;
  letter-spacing:0.3px; border:1px solid;
}}
.vx-dot {{ width:6px; height:6px; border-radius:50%; }}

/* ---- metric cards ---- */
.vx-card {{
  background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;
  padding:13px 15px; height:100%;
}}
.vx-label {{
  font-size:10.5px; font-weight:600; color:{TEXT_FAINT};
  text-transform:uppercase; letter-spacing:0.7px; margin-bottom:6px;
}}
.vx-value {{
  font-size:23px; font-weight:600; color:{TEXT}; letter-spacing:-0.6px;
  font-family:'JetBrains Mono',ui-monospace,monospace; line-height:1.15;
}}
.vx-value.sm {{ font-size:18px; }}
.vx-delta {{ font-size:11.5px; font-weight:500; margin-top:4px; }}
.vx-note {{ font-size:11px; color:{TEXT_FAINT}; margin-top:3px; }}

/* ---- section headings ---- */
.vx-h {{
  font-size:12px; font-weight:600; color:{TEXT_MUTED};
  text-transform:uppercase; letter-spacing:0.8px;
  margin:22px 0 10px; padding-bottom:7px; border-bottom:1px solid {BORDER_SOFT};
}}

/* ---- banners ---- */
.vx-banner {{
  border-radius:9px; padding:11px 14px; font-size:12.5px; line-height:1.55;
  border:1px solid; margin-bottom:12px;
}}

/* ---- streamlit overrides ---- */
[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; }}
/* tabs as a segmented control (Streamlit 1.59: role=tablist, data-testid=stTab) */
.stTabs [role="tablist"] {{
  gap:4px; padding:4px; width:fit-content; max-width:100%; overflow-x:auto;
  background:{SURFACE}; border:1px solid {BORDER}; border-radius:11px; margin:12px 0 6px;
}}
.stTabs [data-testid="stTab"] {{
  height:34px; padding:0 14px; border-radius:8px; color:{TEXT_MUTED};
  display:flex; align-items:center; transition:background .15s ease, color .15s ease;
}}
.stTabs [data-testid="stTab"] p {{ font-size:13px; font-weight:500; color:inherit; white-space:nowrap; }}
.stTabs [data-testid="stTab"] span[role="img"] {{ font-size:17px !important; margin-right:5px; color:{TEXT_FAINT};
  vertical-align:-3px !important; }}
.stTabs [data-testid="stTab"]:hover {{ background:rgba(255,255,255,0.04); color:{TEXT}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{
  background:{SURFACE_ALT}; color:{TEXT}; box-shadow:inset 0 0 0 1px {BORDER}, 0 1px 2px rgba(0,0,0,0.35);
}}
.stTabs [data-testid="stTab"][aria-selected="true"] span[role="img"] {{ color:{ACCENT}; }}
.stTabs .react-aria-SelectionIndicator {{ display:none; }}
.stDataFrame {{ border:1px solid {BORDER}; border-radius:9px; }}
div[data-testid="stDataFrameResizable"] {{ background:{SURFACE}; }}
.stButton button {{
  border-radius:7px; font-size:12.5px; font-weight:500; border:1px solid {BORDER};
  background:{SURFACE_ALT}; color:{TEXT};
}}
.stButton button:hover {{ border-color:{ACCENT}; color:{TEXT}; }}
section[data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
hr {{ border-color:{BORDER_SOFT}; }}
</style>
"""


def card(label: str, value: str, delta: str | None = None,
         delta_color: str | None = None, note: str | None = None,
         small: bool = False) -> str:
    """One KPI card. Values are monospaced so columns of figures align."""
    d = (f'<div class="vx-delta" style="color:{delta_color or TEXT_MUTED}">{delta}</div>'
         if delta else "")
    n = f'<div class="vx-note">{note}</div>' if note else ""
    cls = "vx-value sm" if small else "vx-value"
    return (f'<div class="vx-card"><div class="vx-label">{label}</div>'
            f'<div class="{cls}">{value}</div>{d}{n}</div>')


def pill(text: str, color: str) -> str:
    return (f'<span class="vx-pill" style="color:{color};border-color:{color}44;'
            f'background:{color}14"><span class="vx-dot" style="background:{color}">'
            f'</span>{text}</span>')


def banner(text: str, kind: str = "info") -> str:
    c = {"info": INFO, "warn": WARN, "danger": DANGER, "ok": OK}.get(kind, INFO)
    return f'<div class="vx-banner" style="border-color:{c}44;background:{c}12;color:{TEXT}">{text}</div>'


def section(title: str) -> str:
    return f'<div class="vx-h">{title}</div>'
