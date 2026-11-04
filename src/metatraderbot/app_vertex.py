"""Vertex inside the dashboard (owner 2026-10-09: "keep it in the same style as the whole dashboard ... if not [a
new window] then keep it all in that dashboard" + "control everything with voice, also the whole dashboard").

The look of Vertex in the dashboard -- the orb, the card that slides in on every page while Vertex listens or
talks, the conversation with its proof -- and the small script that turns a voice command into a click on this
laptop's dashboard (a tab, scrolling). The data comes from metatraderbot.vertex_stats; the colours and fonts are
the dashboard's own (app_theme). Builds HTML only: nothing here talks to the bots.
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from . import app_theme as vx

GLOW = "#7c4cf5"                                   # the orb's second colour (the old logo's violet)
STATE_WORDS = {"idle": "Ready", "listening": "Listening", "thinking": "Thinking", "speaking": "Speaking",
               "offline": "Off"}
# Tables Vertex can put under an answer (the laptop app's TABLES): path -> columns
TABLES = {
    "trial.recent_trades": ["trading_day", "number", "direction", "entry", "exit", "result", "pnl_usd"],
    "funded.recent_trades": ["trading_day", "account", "number", "direction", "entry", "exit", "result", "pnl_usd"],
    "funded.payouts": ["ts", "account", "funded_account", "amount_usd", "to_you_usd"],
    "funded.accounts": ["account", "funded_account_number", "balance_usd", "next_payout_needs_usd", "today_pnl_usd"],
    "fleet_real_test.recent_events": ["trading_day", "account", "what", "race_usd", "fee_or_payout_usd"],
    "fleet_real_test.accounts": ["account", "now", "needs_to_pass_usd", "can_lose_usd", "balance_usd",
                                 "needs_for_next_payout_usd"],
}
TABLE_TITLES = {"trial.recent_trades": "Trial races", "funded.recent_trades": "Funded races",
                "funded.payouts": "Payouts", "funded.accounts": "Funded accounts",
                "fleet_real_test.recent_events": "Real test — what happened",
                "fleet_real_test.accounts": "Real test — the accounts now"}
EXAMPLES = [
    ("Numbers", ["How did we do today?", "And yesterday?", "How much did we make this month?",
                 "What's the pass rate since the start?", "When is the next race?",
                 "What can we expect from the plan?", "How is the real test going?",
                 "How much have I spent and been paid so far?"]),
    ("The screen", ["Show me the funded accounts", "Open the payouts tab", "Go to the trial trade charts",
                    "Show the fleet for the last 12 months with 3 accounts", "Scroll down", "Back to the top"]),
    ("Your PC", ["My girlfriend came over, we want a romantic evening", "Play some 90s rock", "Pause  ·  Next song",
                 "Volume 40  ·  Louder", "What's playing?", "Open Netflix", "Close Chrome",
                 "Remind me in 20 minutes to take the pizza out", "What's the weather tomorrow?",
                 "What's on my screen?", "Remember that I like jazz in the evening", "Lock the computer"]),
    ("Switches", ["Mute  ·  Voice on", "Announcements off", "Thank you  (ends the conversation)"]),
]

CSS = f"""
<style>
/* ---- the orb: Vertex's face, one look per state ---- */
.vxo {{ position:relative; width:var(--s); height:var(--s); flex:none; }}
.vxo > * {{ position:absolute; border-radius:50%; }}
.vxo-halo {{ inset:-18%; background:radial-gradient(circle, {vx.ACCENT}33 0%, {GLOW}14 40%, transparent 70%);
  animation:vxo-breathe 4.5s ease-in-out infinite; }}
.vxo-ring {{ inset:0; border:1px solid {vx.ACCENT}55; }}
.vxo-ring.r2 {{ inset:9%; border-color:{GLOW}40; }}
.vxo-ring.r3 {{ inset:-1px; border:none; opacity:0;
  background:conic-gradient(from 0deg, transparent 0 55%, {vx.ACCENT} 80%, {GLOW} 100%);
  -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px));
  mask:radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px)); }}
.vxo-wave {{ inset:18%; border:1px solid {vx.ACCENT}; opacity:0; }}
.vxo-core {{ inset:20%;
  background:radial-gradient(circle at 34% 28%, #c9d4ff 0%, #8ea3ff 12%, {vx.ACCENT} 38%, #3a2a9e 72%, #141a33 100%);
  box-shadow:0 0 calc(var(--s) * .22) {vx.ACCENT}88, 0 0 calc(var(--s) * .5) {GLOW}33,
             inset 0 0 calc(var(--s) * .08) rgba(255,255,255,.25);
  animation:vxo-breathe 4.5s ease-in-out infinite; }}
.vxo-bars {{ inset:20%; display:flex; align-items:center; justify-content:center; gap:calc(var(--s) * .035);
  opacity:0; }}
.vxo-bars i {{ display:block; width:calc(var(--s) * .035); height:22%; border-radius:9px; background:#fff;
  box-shadow:0 0 6px #fff; }}
.vxo.listening .vxo-wave {{ animation:vxo-ripple 1.8s ease-out infinite; }}
.vxo.listening .vxo-wave.w2 {{ animation-delay:.6s; }}
.vxo.listening .vxo-wave.w3 {{ animation-delay:1.2s; }}
.vxo.listening .vxo-core {{ animation:vxo-beat 1.2s ease-in-out infinite; }}
.vxo.thinking .vxo-ring.r3 {{ opacity:1; animation:vxo-spin 1.1s linear infinite; }}
.vxo.thinking .vxo-ring.r2 {{ animation:vxo-spin 3.5s linear infinite reverse; border-style:dashed; }}
.vxo.speaking .vxo-bars {{ opacity:.9; }}
.vxo.speaking .vxo-bars i {{ animation:vxo-eq .9s ease-in-out infinite; }}
.vxo.speaking .vxo-bars i:nth-child(2) {{ animation-delay:-.7s; }}
.vxo.speaking .vxo-bars i:nth-child(3) {{ animation-delay:-.35s; }}
.vxo.speaking .vxo-bars i:nth-child(4) {{ animation-delay:-.55s; }}
.vxo.speaking .vxo-bars i:nth-child(5) {{ animation-delay:-.15s; }}
.vxo.speaking .vxo-halo {{ animation:vxo-beat .9s ease-in-out infinite; }}
.vxo.offline {{ filter:grayscale(1) brightness(.55); }}
.vxo.offline * {{ animation:none !important; }}
@keyframes vxo-breathe {{ 0%,100% {{ transform:scale(.97); opacity:.92; }} 50% {{ transform:scale(1); opacity:1; }} }}
@keyframes vxo-beat {{ 0%,100% {{ transform:scale(.96); }} 50% {{ transform:scale(1.04); }} }}
@keyframes vxo-ripple {{ 0% {{ transform:scale(1); opacity:.9; }} 100% {{ transform:scale(2.1); opacity:0; }} }}
@keyframes vxo-spin {{ to {{ transform:rotate(360deg); }} }}
@keyframes vxo-eq {{ 0%,100% {{ height:16%; }} 50% {{ height:58%; }} }}
@media (prefers-reduced-motion: reduce) {{ .vxo * {{ animation:none !important; }} }}

/* ---- the card on every page while Vertex listens or talks ---- */
.st-key-vx_dock {{ position:fixed; top:58px; right:22px; width:min(430px, calc(100vw - 32px)); z-index:999990;
  background:linear-gradient(180deg, {vx.SURFACE}f2, {vx.BG}f2); backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px); border:1px solid {vx.BORDER}; border-radius:14px; padding:14px 16px 10px;
  box-shadow:0 24px 64px rgba(0,0,0,.6), 0 0 0 1px {vx.ACCENT}1f, 0 0 42px {vx.ACCENT}18;
  animation:vxd-in .35s cubic-bezier(.2,.8,.2,1); gap:6px; }}
.st-key-vx_dock::before {{ content:""; position:absolute; left:16px; right:16px; top:-1px; height:1px;
  background:linear-gradient(90deg, transparent, {vx.ACCENT}, {GLOW}, transparent); }}
@keyframes vxd-in {{ from {{ opacity:0; transform:translateY(-10px) scale(.98); }} to {{ opacity:1; transform:none; }} }}
.st-key-vx_dock [data-testid="stMarkdownContainer"] {{ margin-bottom:0 !important; }}  /* Streamlit pulls it 1rem up */
.st-key-vx_dock .stButton button {{ min-height:28px; padding:2px 12px; font-size:12px; }}
.st-key-vx_dock [data-testid="stHorizontalBlock"] {{ gap:8px; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-vx_dock_slot),
.st-key-vx_dock_slot {{ position:absolute !important; top:0; left:0; width:0 !important; height:0 !important; }}
[class*="st-key-vx_js"] {{ position:absolute !important;
  width:0 !important; height:0 !important; overflow:hidden !important; margin:0 !important; }}
.vxd-head {{ display:flex; align-items:center; gap:12px; }}
.vxd-name {{ font-size:11px; font-weight:700; letter-spacing:2.4px; color:{vx.TEXT}; }}
.vxd-state {{ font-size:12px; color:{vx.TEXT_MUTED}; margin-top:1px; }}
.vxd-state b {{ color:{vx.ACCENT}; font-weight:600; }}
.vxd-time {{ margin-left:auto; font-size:11px; color:{vx.TEXT_FAINT}; font-family:'JetBrains Mono',monospace; }}
.vxd-q {{ margin:12px 0 0; font-size:12.5px; color:{vx.TEXT_MUTED}; }}
.vxd-q span {{ color:{vx.TEXT_FAINT}; font-size:10.5px; font-weight:600; letter-spacing:.8px; margin-right:6px; }}
.vxd-a {{ margin:6px 0 2px; font-size:15px; line-height:1.5; color:{vx.TEXT}; font-weight:500; }}
.vxd-hint {{ margin:10px 0 2px; font-size:12px; color:{vx.TEXT_FAINT}; line-height:1.5; }}

/* ---- proof: every number with where it was read ---- */
.vxp {{ margin-top:10px; border:1px solid {vx.BORDER}; border-radius:10px; overflow:hidden; }}
.vxp-h {{ display:flex; justify-content:space-between; padding:6px 11px; background:{vx.SURFACE_ALT};
  font-size:10px; font-weight:600; letter-spacing:.8px; text-transform:uppercase; color:{vx.TEXT_FAINT}; }}
.vxp-r {{ display:grid; grid-template-columns:minmax(0, max-content) minmax(0, 1fr) 22px; gap:10px; align-items:center; padding:6px 11px;
  border-top:1px solid {vx.BORDER_SOFT}; font-size:12.5px; }}
.vxp-r .l {{ color:{vx.TEXT_MUTED}; min-width:0; max-width:240px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }}
.vxp-r .v {{ color:{vx.TEXT}; font-family:'JetBrains Mono',monospace; font-size:12.5px; text-align:right;
  line-height:1.45; }}  /* long values (a sentence) wrap */
.vxp-r .c {{ text-align:center; font-size:12px; }}
.vxp-r .c.ok {{ color:{vx.OK}; }}
.vxp-r .c.no {{ color:{vx.WARN}; }}
.vxp-r .p {{ grid-column:1 / -1; margin-top:-3px; font-size:10.5px; color:{vx.TEXT_FAINT};
  font-family:'JetBrains Mono',monospace; }}

/* ---- the Vertex page ---- */
.vxh {{ position:relative; display:flex; align-items:center; gap:30px; padding:26px 30px; border-radius:16px;
  border:1px solid {vx.BORDER}; overflow:hidden;
  background:radial-gradient(120% 140% at 0% 0%, {vx.ACCENT}1c 0%, transparent 55%),
             radial-gradient(90% 120% at 100% 100%, {GLOW}14 0%, transparent 60%), {vx.SURFACE}; }}
.vxh::after {{ content:""; position:absolute; inset:0; pointer-events:none; opacity:.35;
  background-image:linear-gradient({vx.BORDER_SOFT} 1px, transparent 1px),
                   linear-gradient(90deg, {vx.BORDER_SOFT} 1px, transparent 1px);
  background-size:28px 28px; -webkit-mask:linear-gradient(90deg, transparent, #000 60%);
  mask:linear-gradient(90deg, transparent, #000 60%); }}
.vxh-t {{ position:relative; z-index:1; min-width:0; }}
.vxh-name {{ font-size:13px; font-weight:700; letter-spacing:5px; color:{vx.TEXT}; }}
.vxh-state {{ font-size:26px; font-weight:600; letter-spacing:-.6px; color:{vx.TEXT}; margin:6px 0 4px; }}
.vxh-sub {{ font-size:13px; color:{vx.TEXT_MUTED}; line-height:1.55; max-width:640px; }}
.vxh-sub b {{ color:{vx.TEXT}; font-weight:600; }}
.vxl {{ display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid {vx.BORDER_SOFT}; }}
.vxl:last-child {{ border-bottom:none; }}
.vxl-n {{ font-size:13px; font-weight:600; color:{vx.TEXT}; }}
.vxl-s {{ font-size:12px; color:{vx.TEXT_MUTED}; }}
.vxl-f {{ margin-left:auto; display:flex; gap:6px; }}
.vxx {{ font-size:12.5px; color:{vx.TEXT_MUTED}; line-height:1.9; }}
.vxx b {{ display:block; margin-top:6px; font-size:10.5px; font-weight:600; letter-spacing:.8px;
  text-transform:uppercase; color:{vx.TEXT_FAINT}; }}
.vxx span {{ display:inline-block; margin:3px 6px 0 0; padding:2px 10px; border-radius:999px;
  border:1px solid {vx.BORDER}; background:{vx.SURFACE_ALT}; color:{vx.TEXT}; font-size:12px; }}
.vxc {{ position:relative; background:{vx.SURFACE}; border:1px solid {vx.BORDER}; border-radius:12px;
  padding:14px 16px; margin-bottom:10px; }}
.vxc.latest {{ border-color:{vx.ACCENT}55; box-shadow:0 0 0 1px {vx.ACCENT}22, 0 0 30px {vx.ACCENT}10; }}
.vxc.ev {{ padding:10px 16px; border-left:3px solid {vx.ACCENT}; }}
.vxc-m {{ display:flex; gap:8px; align-items:center; font-size:11px; color:{vx.TEXT_FAINT}; }}
.vxc-m .t {{ font-family:'JetBrains Mono',monospace; }}
.vxc-q {{ margin-top:8px; font-size:13px; color:{vx.TEXT_MUTED}; }}
.vxc-q span, .vxc-a span {{ display:inline-block; width:58px; font-size:10px; font-weight:700; letter-spacing:1.2px;
  color:{vx.TEXT_FAINT}; }}
.vxc-a {{ margin-top:6px; font-size:14.5px; line-height:1.55; color:{vx.TEXT}; }}
.vxc-a span {{ color:{vx.ACCENT}; }}
.vxc-ui {{ margin-top:8px; font-size:11.5px; color:{vx.TEXT_FAINT}; }}
.vxc-ui i {{ font-style:normal; color:{vx.TEXT_MUTED}; }}
.st-key-vx_ask textarea {{ font-size:14px; }}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def orb(state: str = "idle", size: int = 120) -> str:
    st = state if state in STATE_WORDS else "idle"
    return (f'<div class="vxo {st}" style="--s:{size}px"><div class="vxo-halo"></div><div class="vxo-wave"></div>'
            f'<div class="vxo-wave w2"></div><div class="vxo-wave w3"></div><div class="vxo-ring"></div>'
            f'<div class="vxo-ring r2"></div><div class="vxo-ring r3"></div><div class="vxo-core"></div>'
            f'<div class="vxo-bars"><i></i><i></i><i></i><i></i><i></i></div></div>')


def when(ts: str, tz: ZoneInfo, today: str | None = None) -> str:
    try:
        t = datetime.fromisoformat(ts).astimezone(tz)
    except (TypeError, ValueError):
        return ""
    return f"{t:%H:%M}" if t.strftime("%Y-%m-%d") == (today or datetime.now(tz).strftime("%Y-%m-%d")) \
        else f"{t:%d %b %H:%M}"


def proof_html(proof: list[dict], paths: bool = False) -> str:
    if not proof:
        return ""
    rows = []
    for p in proof:
        ok = bool(p.get("found"))
        rows.append(f'<div class="vxp-r"><span class="l">{esc(p.get("label"))}</span>'
                    f'<span class="v">{esc(p.get("value"))}</span>'
                    f'<span class="c {"ok" if ok else "no"}" title="{"read from the bots&#39; records" if ok else "not in the records"}">'
                    f'{"✓" if ok else "?"}</span>'
                    + (f'<span class="p">{esc(p.get("path"))}</span>' if paths else "") + "</div>")
    return ('<div class="vxp"><div class="vxp-h"><span>Proof</span><span>read from the bots\' records</span></div>'
            + "".join(rows) + "</div>")


def ui_words(ui: dict) -> str:
    """What a voice command did to the screen, in words ("funded · Payouts & accounts")."""
    bits = []
    if ui.get("page"):
        bits.append(str(ui["page"]).capitalize() + " page")
    if ui.get("tab"):
        bits.append(str(ui["tab"]))
    if ui.get("account"):
        bits.append("account " + str(ui["account"]))
    if ui.get("fleet_period"):
        bits.append(str(ui["fleet_period"]))
    if ui.get("fleet_start"):
        bits.append(f"start with {ui['fleet_start']}")
    if ui.get("fleet_up_to"):
        bits.append(f"up to {ui['fleet_up_to']}")
    if ui.get("scroll"):
        bits.append("scroll " + str(ui["scroll"]))
    return " · ".join(bits)


def did_html(did) -> str:
    """What Vertex did on the PC for this answer ("Done: music \"soft jazz\" · volume 30%")."""
    return f'<div class="vxc-ui">Done: <i>{esc(" · ".join(did))}</i></div>' if did else ""


def dock_html(entry: dict | None, state: str, viewer: str, tz: ZoneInfo, detail: str = "") -> str:
    """The card's text: orb + state, then the question and the answer (or what Vertex is doing)."""
    word = STATE_WORDS.get(state, "Ready")
    who = f" · {esc(viewer)}" if viewer else ""
    head = (f'<div class="vxd-head">{orb(state, 38)}<div><div class="vxd-name">VERTEX</div>'
            f'<div class="vxd-state"><b>{word}</b>{who}</div></div>'
            f'<div class="vxd-time">{when((entry or {}).get("ts", ""), tz)}</div></div>')
    if state == "listening" and not detail:
        return head + '<div class="vxd-hint">Go ahead — ask about the bots, tell me what to show, or what to do on ' \
                      'the PC.</div>'
    if state == "thinking" and detail and detail != "writing it down":
        return head + f'<div class="vxd-q"><span>YOU</span>{esc(detail)}</div>'
    if not entry:
        return head + f'<div class="vxd-hint">{"Writing it down…" if state == "thinking" else ""}</div>'
    if entry.get("kind") == "announce":
        return head + f'<div class="vxd-a">{esc(entry.get("text"))}</div>'
    q = entry.get("question")
    moved = ui_words(entry.get("ui") or {})
    return (head + (f'<div class="vxd-q"><span>YOU</span>{esc(q)}</div>' if q else "")
            + (f'<div class="vxd-a">{esc(entry.get("speech"))}</div>' if entry.get("speech") else "")
            + (f'<div class="vxc-ui">On screen: <i>{esc(moved)}</i></div>' if moved else "")
            + did_html(entry.get("did"))
            + proof_html((entry.get("proof") or [])[:5]))


def entry_html(e: dict, tz: ZoneInfo, latest: bool = False) -> str:
    """One item of the conversation on the Vertex page."""
    t = when(e.get("ts", ""), tz)
    if e.get("kind") == "announce":
        return (f'<div class="vxc ev"><div class="vxc-m"><span class="t">{t}</span><span>announcement</span></div>'
                f'<div class="vxc-a" style="margin-top:4px">{esc(e.get("text"))}</div></div>')
    src = {"typed": "typed", "voice": "spoken"}.get(e.get("source"), "")
    meta = " · ".join(x for x in (esc(e.get("viewer")), src) if x)
    q = e.get("question")
    moved = ui_words(e.get("ui") or {})
    return (f'<div class="vxc{" latest" if latest else ""}"><div class="vxc-m"><span class="t">{t}</span>'
            f'<span>{meta}</span></div>'
            + (f'<div class="vxc-q"><span>YOU</span>{esc(q)}</div>' if q else "")
            + (f'<div class="vxc-a"><span>VERTEX</span>{esc(e.get("speech"))}</div>' if e.get("speech") else "")
            + (f'<div class="vxc-ui">On screen: <i>{esc(moved)}</i></div>' if moved else "")
            + did_html(e.get("did"))
            + proof_html(e.get("proof") or [], paths=latest) + "</div>")


def examples_html() -> str:
    return '<div class="vxx">' + "".join(
        f"<b>{esc(h)}</b>" + "".join(f"<span>{esc(x)}</span>" for x in xs) for h, xs in EXAMPLES) + "</div>"


def command_script(act: dict, nonce: str) -> str:
    """A tiny page for components.html that works this laptop's dashboard the way a hand would: it clicks the page
    switch, the funded account, the fleet period and the tab, types the fleet's account numbers and scrolls -- so
    the dashboard behaves exactly as after real clicks. The steps run in the dashboard itself (the little frame
    may vanish when the page changes) and each waits for its button to be drawn (up to 12 s).
    act: {"page": label on the page switch, "account", "fleet_period", "fleet_start", "fleet_up_to", "tab",
    "scroll"} -- any of them."""
    payload = json.dumps({k: act.get(k) for k in ("page", "account", "fleet_period", "fleet_start", "fleet_up_to",
                                                    "tab", "scroll") if act.get(k)})
    nid = json.dumps(str(nonce))
    code = """(async function (A, ID) {
  window.__vx = window.__vx || {};
  if (window.__vx[ID]) return;
  window.__vx[ID] = 1;
  const doc = document, sleep = ms => new Promise(r => setTimeout(r, ms));
  const label = el => { const c = el.cloneNode(true);
    c.querySelectorAll('[data-testid="stIconMaterial"], [role="img"]').forEach(x => x.remove());
    return c.textContent.replace(/\\s+/g, ' ').trim(); };
  const live = el => !el.closest('[data-stale="true"]');
  async function find(sel, text, ms = 12000) {
    for (let t = 0; t < ms; t += 200) {
      const hit = [...doc.querySelectorAll(sel)].filter(live).find(e => text === undefined || label(e) === text);
      if (hit) return hit;
      await sleep(200);
    }
    return null;
  }
  const on = b => b.getAttribute('aria-checked') === 'true' || b.getAttribute('aria-selected') === 'true'
                  || /Active$/.test(b.getAttribute('data-testid') || '');
  async function press(sel, text) {
    const b = await find(sel, text);
    if (b && !on(b)) { b.click(); await sleep(900); }
    return b;
  }
  function scroller() {
    for (const s of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {
      const el = doc.querySelector(s);
      if (el && el.scrollHeight > el.clientHeight + 10) return el;
    }
    return doc.scrollingElement || doc.documentElement;
  }
  async function number(key, v) {
    const inp = await find('.st-key-' + key + ' input');
    if (!inp || String(inp.value) === String(v)) return;
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    inp.focus(); set.call(inp, String(v));
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    inp.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true}));
    inp.blur();
    await sleep(900);
  }
  if (A.page) {
    const b = await find('.st-key-nav_top button', A.page);
    if (b && !on(b)) { b.click(); await sleep(1500); if (!A.scroll) scroller().scrollTo({top: 0}); }
  }
  if (A.account) await press('.st-key-funded_account_pick button', A.account);
  if (A.fleet_period) await press('.st-key-fleet_period button', A.fleet_period);
  if (A.fleet_start) await number('fleet_start_slots', A.fleet_start);
  if (A.fleet_up_to) await number('fleet_slots', A.fleet_up_to);
  if (A.tab) {
    const t = await find('[data-testid="stTab"], button[role="tab"]', A.tab);
    if (t) { if (!on(t)) t.click(); await sleep(300); t.scrollIntoView({behavior: 'smooth', block: 'start'}); }
  }
  if (A.scroll) {
    await sleep(A.tab ? 500 : 0);
    const el = scroller(), page = el.clientHeight * 0.8;
    const to = {top: 0, bottom: el.scrollHeight, down: el.scrollTop + page, up: el.scrollTop - page}[A.scroll];
    if (to !== undefined) el.scrollTo({top: to, behavior: 'smooth'});
  }
})(""" + payload + ", " + nid + ");"
    return (f"<!-- {esc(nonce)} --><script>(function () {{ const s = window.parent.document.createElement('script');"
            f" s.textContent = {json.dumps(code)}; window.parent.document.body.appendChild(s); s.remove(); }})();"
            "</script>")
