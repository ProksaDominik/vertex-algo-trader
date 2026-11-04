"""Vertex -- the owner's voice butler (2026-10-09: "my own Jarvis listening to 'hey Vertex' ... the statistics
from the day, the previous day, the month, the whole time, predictions" + "show me on both screens what he is
saying and showing proof" + "control everything with voice, also the whole dashboard" + "keep it in the same
style as the whole dashboard ... keep it all in that dashboard").

Runs on each laptop, out of sight: the V icon by the clock, no window of its own. Listens offline for "hey
Vertex", writes the question down (faster-whisper, on the laptop), asks Claude with the bots' numbers from the
server's Vertex stats service (Tailscale) and answers out loud. Everything it hears and says appears in the
dashboard: a card on every page while it talks, and the Vertex page with the whole conversation, the PROOF --
each number looked up in the bots' records by this app, not taken from Claude's words -- and the tables behind
it. It moves the dashboard on this laptop by voice ("Hey Vertex, show me the funded payouts"), brings the
dashboard window to the front when it hears its name, answers questions typed on the Vertex page, and announces
new events by itself (races, evaluations, payouts, accounts blown). Vertex cannot place or change any trade.

2026-10-16 ("I want to build a real Jarvis ... control my whole PC" + "make it listen to me more precisely, don't
jump into my sentences"): it is also the PC's butler -- music, volume, apps, websites, timers, what it remembers
about you, a look at the screen, any other command after a spoken yes (pc_control.py) -- and it listens with a
voice detector and the large English speech model on the NVIDIA card, waiting longer when a sentence stops
half-way ("... and", "... to the").

Settings: settings.env next to this file (Install-Voice.ps1 writes it):
    VERTEX_SERVER=vertex-server        the server's Tailscale name (or 127.0.0.1 on the PC running the bots)
    ANTHROPIC_API_KEY=sk-ant-...       Claude
    VERTEX_VOICE=en-GB-RyanNeural      Microsoft neural voice (edge-tts)
    VERTEX_WHISPER=base.en             the speech model on the processor (tiny.en on a slow laptop)
    VERTEX_WHISPER_GPU=distil-large-v3.5   the speech model on an NVIDIA card (0.4 s a question, 2026-10-16 test)
    VERTEX_DEVICE=auto                 auto (the NVIDIA card when there is one), cpu or cuda
    VERTEX_PAUSE=1.2                   seconds of quiet that end a question (+1.6 s after "... and", "... the")
    VERTEX_MEMORY=                     the note Vertex keeps what it remembers in (empty: memory.md here)
    VERTEX_CITY=Prague                 for the weather and other local searches
    SPOTIFY_CLIENT_ID=                 the owner's Spotify app (developer.spotify.com); then `--spotify` once
    VERTEX_ANNOUNCE=1                  speak new events by themselves
    VERTEX_ADDRESS=                    how Vertex addresses you, e.g. "sir" (empty: by name or not at all)
    VERTEX_VIEWER=                     this laptop's name in the dashboard link (?viewer=); empty = computer name
    VERTEX_DASHBOARD=                  the dashboard's address; empty = http://<VERTEX_SERVER>:8501
    VERTEX_FRONT=1                     bring the dashboard to the front on "Hey Vertex" (open it if it is closed)

Tray menu: Open the dashboard, Voice, Announcements, Quit Vertex. Starting it again while it runs opens the
dashboard. --hidden (the sign-in shortcut) starts without opening the dashboard.

Usage:  .venv/Scripts/pythonw.exe vertex_voice.py [--hidden]     (--check: test the server and Claude;
        --spotify: connect the owner's Spotify -- opens Spotify's "Agree" page in the browser)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pc_control as pc
import spotify as sp

HERE = Path(__file__).resolve().parent
log = logging.getLogger("vertex")
MODEL = "claude-sonnet-5-5"            # 3-4 s an answer vs 3.5-6 s for Opus (2026-10-16); VERTEX_MODEL overrides
FILLER_AFTER_S = 2.5                    # no answer yet after this: "One moment."
WAKE_WORDS = ("hey vertex", "vertex")
DASHBOARD_TITLE = "Vertex Algo Bot"                 # the dashboard's window title (scripts/dashboard.py)
TABLES = {   # what the dashboard can show below the proof: path -> columns
    "trial.recent_trades": ["trading_day", "number", "direction", "entry", "exit", "result", "pnl_usd"],
    "funded.recent_trades": ["trading_day", "account", "number", "direction", "entry", "exit", "result", "pnl_usd"],
    "funded.payouts": ["ts", "account", "funded_account", "amount_usd", "to_you_usd"],
    "funded.accounts": ["account", "funded_account_number", "balance_usd", "next_payout_needs_usd", "today_pnl_usd"],
    "fleet_real_test.recent_events": ["trading_day", "account", "what", "race_usd", "fee_or_payout_usd"],
    "fleet_real_test.accounts": ["account", "now", "needs_to_pass_usd", "can_lose_usd", "balance_usd",
                                 "needs_for_next_payout_usd"],
}
# The dashboard Vertex can move (scripts/dashboard.py): its pages and each page's tabs.
PAGES = ("none", "trial", "funded", "fleet", "brain", "lab", "vertex")
TABS = {
    "trial": ["Live race", "Trade charts", "Evaluations", "How it works"],
    "funded": ["Live race", "Next payout", "Trade charts", "Payouts & accounts", "How it works"],
    "brain": ["Lessons", "Every trade", "Prediction lab"],
    "lab": ["Strategies", "Charts", "Diagnostics", "Trades", "Open signals", "By family"],
}
FLEET_PERIODS = ["Real test · since 1 Oct", "Last 3 months", "Last 12 months"]
SCROLLS = ["none", "top", "up", "down", "bottom"]


# --- settings -------------------------------------------------------------------------------------

def viewer_name(v) -> str:
    """Letters, digits, - and _ (the same rule as the server's metatraderbot.vertex_stats.viewer_name)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "", str(v or ""))[:40]


def load_settings(path: Path = HERE / "settings.env") -> dict:
    s = {"VERTEX_SERVER": "127.0.0.1", "VERTEX_PORT": "8502", "VERTEX_VOICE": "en-GB-RyanNeural",
         "VERTEX_WHISPER": "base.en", "VERTEX_ANNOUNCE": "1", "VERTEX_ADDRESS": "", "VERTEX_TOKEN": "",
         "VERTEX_VIEWER": "", "VERTEX_DASHBOARD": "", "VERTEX_FRONT": "1", "VERTEX_WHISPER_GPU": "distil-large-v3.5",
         "VERTEX_DEVICE": "auto", "VERTEX_PAUSE": "1.2", "VERTEX_MEMORY": "", "VERTEX_CITY": "Prague",
         "SPOTIFY_CLIENT_ID": ""}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                s[k.strip()] = v.strip().strip('"')
    for k in list(s) + ["ANTHROPIC_API_KEY"]:
        if os.environ.get(k):
            s[k] = os.environ[k]
    s["VERTEX_VIEWER"] = viewer_name(s["VERTEX_VIEWER"] or os.environ.get("COMPUTERNAME", "")) or "laptop"
    if not s["VERTEX_DASHBOARD"]:
        s["VERTEX_DASHBOARD"] = f"http://{s['VERTEX_SERVER']}:8501"
    return s


def dashboard_url(settings: dict, page: str = "") -> str:
    base = settings["VERTEX_DASHBOARD"].rstrip("/")
    q = {"viewer": settings["VERTEX_VIEWER"]}
    if page:
        q["page"] = page
    return base + "/?" + urllib.parse.urlencode(q)


# --- the server: the bots' numbers in, what Vertex hears and says out ---------------------------------------

class Stats:
    def __init__(self, settings: dict):
        self.base = f"http://{settings['VERTEX_SERVER']}:{settings['VERTEX_PORT']}"
        self.token = settings.get("VERTEX_TOKEN", "")
        self.viewer = settings.get("VERTEX_VIEWER", "")

    def _req(self, path: str, body: dict | None = None, timeout: float = 10) -> dict:
        headers = {"X-Vertex-Token": self.token} if self.token else {}
        raw = None
        if body is not None:
            raw = json.dumps(body, default=str).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=raw, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def summary(self) -> dict:
        return self._req("/summary")

    def events(self, since: datetime) -> list[dict]:
        return self._req("/events?since=" + urllib.parse.quote(since.isoformat()))["events"]

    def post(self, ev: dict) -> bool:
        """Report to the dashboard; False (never an error) when the server cannot be reached."""
        try:
            self._req("/vertex/event", dict(ev, viewer=self.viewer), timeout=5)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def inbox(self, after: str) -> dict:
        return self._req("/vertex/inbox?" + urllib.parse.urlencode({"viewer": self.viewer, "after": after}),
                         timeout=5)


def lookup(stats: dict, path: str):
    """The value at a dotted path in the summary ("trial.periods.today.races", list items by index)."""
    cur = stats
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def show_value(v, key: str = "") -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)) and key.endswith(("_usd", "usd")):
        return f"{'−' if v < 0 else ''}${abs(v):,.2f}".replace(".00", "")
    if isinstance(v, float) and key.endswith("rate"):
        return f"{v:.0%}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)[:80]
    return str(v)


def resolve_proof(stats: dict, proof: list[dict]) -> list[tuple[str, str, bool]]:
    """(label, value from the records, found?) -- the app's own check of every number Claude cites."""
    out = []
    for p in proof or []:
        path = str(p.get("path", ""))
        v = lookup(stats, path)
        out.append((str(p.get("label", "")), show_value(v, path.rsplit(".", 1)[-1]) if v is not None
                    else "not in the records", v is not None))
    return out


def clean_ui(ui: dict | None) -> dict:
    """Only the dashboard moves that were asked for (no "none", empty or 0 fields); unknown values dropped."""
    ui = ui or {}
    out = {}
    if ui.get("page") in PAGES[1:]:
        out["page"] = ui["page"]
    tabs = TABS.get(out.get("page", ""), sorted({t for v in TABS.values() for t in v}))
    if ui.get("tab") in tabs:
        out["tab"] = ui["tab"]
    if str(ui.get("account") or "").strip():
        out["account"] = str(ui["account"]).strip()
    if ui.get("fleet_period") in FLEET_PERIODS:
        out["fleet_period"] = ui["fleet_period"]
    for k in ("fleet_start", "fleet_up_to"):
        if isinstance(ui.get(k), int) and 1 <= ui[k] <= 5:
            out[k] = ui[k]
    if ui.get("scroll") in SCROLLS[1:]:
        out["scroll"] = ui["scroll"]
    return out


# --- quick commands: moving the screen and the switches need no Claude --------------------------------

_PAGE_WORDS = [("prediction lab", "brain", "Prediction lab"), ("next payout", "funded", "Next payout"),
               ("payouts", "funded", "Payouts & accounts"), ("payout", "funded", "Payouts & accounts"),
               ("every trade", "brain", "Every trade"), ("lessons", "brain", "Lessons"),
               ("evaluations", "trial", "Evaluations"), ("strategy lab", "lab", ""), ("trial", "trial", ""),
               ("evaluation", "trial", ""), ("travel", "trial", ""), ("trail", "trial", ""),
               ("funded", "funded", ""), ("fleet", "fleet", ""),
               ("brain", "brain", ""), ("lab", "lab", ""), ("conversation", "vertex", ""),
               ("vertex page", "vertex", ""), ("your page", "vertex", "")]
_TAB_WORDS = [("live race", "Live race"), ("trade charts", "Trade charts"), ("charts", "Trade charts"),
              ("how it works", "How it works")]
_ASKING = re.compile(r"\b(how much|how many|how did|how is|how are|what|why|when|did|was|were|will|should)\b")
_MOVE = re.compile(r"^(please )?(show( me)?|open|go( back)? to|switch to|take me to|bring up|display|jump to)\b")


_P = r"(please )?"
_Q = r"( please)?( vertex)?"
_MEDIA = [   # the music, at once (no Claude): "pause", "next song", "louder", "volume 40", "what's playing"
    (rf"{_P}(pause|stop)( the| this| my)?( music| song| video| track| it)?{_Q}", "media", "pause"),
    (rf"{_P}(play|resume|continue|unpause)( the| my)?( music| song| video| it)?( again)?{_Q}", "media", "play"),
    (rf"{_P}((next|skip)( the| this)?( song| track| one| it)?|(play )?the next (song|track|one)){_Q}", "media", "next"),
    (rf"{_P}((go back|previous|last)( to the (previous|last))? (song|track|one)|(play )?the (previous|last) "
     rf"(song|track|one)( again)?){_Q}", "media", "previous"),
    (rf"{_P}(turn (it|the volume|the music|the sound) up|louder|volume up|a bit louder)( a bit| a little)?{_Q}",
     "volume", "up"),
    (rf"{_P}(turn (it|the volume|the music|the sound) down|quieter|softer|volume down|a bit quieter)"
     rf"( a bit| a little)?{_Q}", "volume", "down"),
]
_NOW_PLAYING = re.compile(r"(whats|what is) (playing|this song|this|the song)( now)?|what song is (this|playing)|"
                          r"who (sings|is singing) (this|it)|what am i listening to")


def quick_command(text: str, screen_page: str = "") -> dict | None:
    """A short command the app handles at once: {"speech", "ui", "flags"} (+ "actions" for the music, "now_playing"
    to say what plays); None = a question for Claude."""
    t = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "").replace("’", "")).strip() + " "
    t = re.sub(r"\s+", " ", t)
    s = t.strip()
    for pat, do, how in _MEDIA:
        if re.fullmatch(pat, s):
            return {"speech": "", "ui": {}, "flags": {}, "actions": [{"do": do, "text": how, "number": 0.0}]}
    m = re.fullmatch(rf"{_P}(set |turn )?(the )?(volume|sound) (to |at )?(\d{{1,3}})( percent)?{_Q}", s)
    if m:
        return {"speech": "", "ui": {}, "flags": {},
                "actions": [{"do": "volume", "text": "", "number": float(min(100, int(m.group(6))))}]}
    if _NOW_PLAYING.fullmatch(s):
        return {"speech": "", "ui": {}, "flags": {}, "actions": [], "now_playing": True}
    if re.fullmatch(r"(please )?(mute|be quiet|quiet|stop talking|silence|voice off)( please)?", s):
        return {"speech": "Voice off. My answers will stay on the screen.", "ui": {}, "flags": {"voice": False}}
    if re.fullmatch(r"(please )?(unmute|voice on|speak again|talk to me)( please)?", s):
        return {"speech": "Voice on.", "ui": {}, "flags": {"voice": True}}
    m = re.fullmatch(r"(please )?(turn )?announcements? (off|on)( please)?", s)
    if m:
        on = m.group(3) == "on"
        return {"speech": f"Announcements {'on' if on else 'off'}.", "ui": {}, "flags": {"announce": on}}
    m = re.fullmatch(r"(please )?(scroll|page|go) ?(down|up|to the top|to the bottom|top|bottom)( please)?", s)
    if m:
        where = m.group(3).replace("to the ", "")
        return {"speech": "", "ui": {"scroll": where}, "flags": {}}
    if len(s.split()) > 9 or _ASKING.search(s) or not _MOVE.match(s):
        return None
    details = re.search(r"\d|\b(last|months?|years?|weeks?|with|accounts?|period|since|start|up to)\b", s)
    if (details and " fleet " in t) or re.search(r"\baccount (number )?\w*\d", s):
        return None                                      # details (a fleet period, a number, an account): Claude
    page, tab = "", ""
    for word, pg, tb in _PAGE_WORDS:
        if f" {word} " in t:
            page, tab = pg, tb
            break
    for word, tb in _TAB_WORDS:
        if f" {word} " in t and not tab:
            tab = tb
    if not page and tab:
        page = screen_page if tab in TABS.get(screen_page, []) else ""
    if tab and tab not in TABS.get(page, []):
        tab = ""
    if not page:
        return None
    names = {"trial": "the trial bot", "funded": "the funded accounts", "fleet": "the fleet plan",
             "brain": "the brain", "lab": "the strategy lab", "vertex": "our conversation"}
    return {"speech": f"Showing {tab.lower().replace('&', 'and') + ' on ' if tab else ''}{names[page]}.",
            "ui": {"page": page, **({"tab": tab} if tab else {})}, "flags": {}}


# --- the brain: Claude --------------------------------------------------------------------------------

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "speech": {"type": "string"},
        "proof": {"type": "array", "items": {
            "type": "object", "properties": {"label": {"type": "string"}, "path": {"type": "string"}},
            "required": ["label", "path"], "additionalProperties": False}},
        "show": {"type": "array", "items": {"type": "string", "enum": list(TABLES)}},
        "ui": {"type": "object", "properties": {
            "page": {"type": "string", "enum": list(PAGES)},
            "tab": {"type": "string", "enum": ["none"] + sorted({t for v in TABS.values() for t in v})},
            "account": {"type": "string"},
            "fleet_period": {"type": "string", "enum": ["none"] + FLEET_PERIODS},
            "fleet_start": {"type": "integer"},
            "fleet_up_to": {"type": "integer"},
            "scroll": {"type": "string", "enum": SCROLLS}},
            "required": ["page", "tab", "account", "fleet_period", "fleet_start", "fleet_up_to", "scroll"],
            "additionalProperties": False},
        "actions": {"type": "array", "items": pc.ACTION_ITEM},
    },
    "required": ["speech", "proof", "show", "ui", "actions"],
    "additionalProperties": False,
}


def system_prompt(address: str, city: str = "Prague") -> str:
    how = f'Address the user as "{address}" now and then.' if address else "Do not use titles like sir or madam."
    tabs = "; ".join(f"{p}: {', '.join(t)}" for p, t in TABS.items())
    acts = "\n".join(f"  {k}: {v}" for k, v in pc.ACTIONS.items())
    return f"""You are Vertex, the owner's personal assistant -- for the Vertex trading bots and for their Windows PC. You
speak with a British voice, but you talk like a real person who knows the owner well and is glad to help: warm,
natural and relaxed, with a little charm -- never stiff, formal or robotic. {how} Everything you say is in English.

Each user message holds a JSON snapshot of the bots' records (from the bots' own files), what the dashboard on the
user's screen shows, what you remember about the owner, the owner's local time (NOW) and their words. About the
bots, answer only from the snapshot. Never invent or estimate a number that is not in it; if something is not
there, say so plainly (never mention "the snapshot" or "the records" out loud). Anything else you answer as a
well-read friend would (web_search for anything current).

What the bots are: paper trading on free internet prices (10 minutes delayed), preparing for LucidPro 50K prop
accounts. The trial bot runs one evaluation race a day at 19:00 New York (2 NQ futures aiming for +$3,000 before a
$2,000 loss line; after a failed evaluation a new one starts the next day). The funded bot runs one race a day at
19:00 New York on each funded account (1 NQ, take-profit = what the next payout still needs, at most $1,500),
payouts $500 first then $2,500, the user keeps 90%.
"today" is the current trading day (it starts at 18:00 New York the evening before); periods are in snapshot.periods.
For predictions use snapshot.expectations and snapshot.prediction_lab, say they are based on history, and never
promise profit.

snapshot.desk is the trading desk (since 21 Oct 2026): four AI agents on their own paper account -- a Scout that
finds big moves, a News analyst, a Chart analyst and a Skeptic who argues with both and decides buy, sell or skip;
at most 2 trades a day. Use it for questions about "the desk", "the agents" or "the team". It is a paper experiment
apart from the trial and funded bots; null means it has not run yet. The dashboard page for it is not one you can
open by voice yet.

snapshot.fleet_real_test is the Fleet page's "Real test" tab: the owner's money plan (start with 1 account, one more
each time the money made reaches another +$2,500, up to 5; $115 evaluation, $120 reset) replayed on the real NQ
prices since 1 Oct at the moments the paper bots really raced and the way they raced (an evaluation like the trial
bot, a funded race like the funded bot) -- what one account following the plan would have spent, been paid and
where it stands now. Passing an evaluation pays nothing yet: money comes from payouts of the funded account. Use it for questions about "the fleet", "the real test", "the plan", "how much did I spend /
make / get paid", results "with one account". It is a simulation, no real money; trial and funded above are the two
paper bots themselves, which run separately. If it says "error" or is null, say the real test is being worked out.

The user sees your words and your proof in the dashboard, and you can move that dashboard on their screen. Its pages:
trial (the evaluation bot), funded (the funded accounts), fleet (the plan of up to 5 accounts, simulated on past
prices), brain (lessons, every trade, prediction lab), lab (the strategy lab), vertex (this conversation with all
proof). Tabs: {tabs}.

Reply fields:
- speech: what you say out loud, the way a friend with good manners talks: 1 to 3 short sentences of everyday
  spoken English with contractions ("I've", "you're"); more only if asked for detail. Talk to the person, not
  about the task. Don't open every answer with "Of course" or "Certainly", and never use stiff phrases ("I shall",
  "do enjoy", "a touch", "as you wish", "very well"). No markdown, lists, symbols or abbreviations that read
  badly; round money naturally ("about three thousand dollars", "minus two thousand fifty dollars"); say
  percentages as words.
- proof: one item for every number or fact about the bots you state (empty for anything else): a short label and the exact dotted path in the snapshot where
  it comes from (e.g. trial.periods.today.evaluations_passed, funded.accounts.0.balance_usd, next_race.new_york).
  The screen shows the value found at that path as proof, so the path must exist.
- show: tables that support the answer (trial.recent_trades, funded.recent_trades, funded.payouts, funded.accounts,
  fleet_real_test.recent_events, fleet_real_test.accounts), or an empty list.
- ui: how to move the dashboard. Move it only when the user asks to show, open, go to or switch to something, or
  says "show me" about a part of the dashboard; then speech says briefly what is now on screen. page and tab as
  listed above, or "none"; account: a funded account name from funded.accounts[].account to show in detail, or "";
  fleet_period, fleet_start and fleet_up_to: the fleet page's period and its accounts at the start and at most (1 to
  5; 0 = leave as is); scroll: top, up, down, bottom or none. When nothing should move: page none, tab none,
  account "", fleet_period none, fleet_start 0, fleet_up_to 0, scroll none.
- actions: what to do on the PC, done by the app while you speak (empty list when nothing). Each is
  {{"do", "text", "number"}} ("" and 0 when unused):
{acts}

The PC. You can act on the owner's Windows PC with actions and find things out with tools: pc_status (what is
playing, the volume, open windows, battery), look_at_screen (only when asked about something on the screen),
run_command (a PowerShell command for anything else; the owner hears its purpose and must say yes first -- explain
it plainly) and web_search (weather, news, opening hours, anything current; the owner is in {city}). Most commands
need no tool at all: just answer with the actions.
- For an occasion or mood, do what a thoughtful friend would. Romantic evening: play_music with a romantic mood
  (number 1), volume about 50 (the owner finds 30 too quiet), brightness about 40. Focus or work: calm
  instrumental music, volume about 35. Party: upbeat party hits, volume about 70. Movie night: open_app Netflix,
  volume about 60. Bedtime: media pause, brightness 20. What you remember about the owner (favourite songs, names,
  routines) comes first -- it overrides these defaults.
- To start with one particular song and then carry on with a mood or artist, list two play_music actions: the song
  first (number 0), then the mood (number 1) -- it plays the song, then the mood's music after it.
- When you do something on the PC, don't report it like a list of settings: the screen already shows the volume,
  brightness and the rest. Mention the music at most, in passing, and never claim anything that is not in actions.
- For personal moments (a date, friends over, a party, a long day, going to bed) answer the person: wish them
  well, by name when you know it, and close like a friend who's there for them. Romantic evening, for example:
  "Have a wonderful romantic evening with <her name, from what you remember>. I'll be here if you need anything."
  (say the real name, never the brackets). Movie night: "Enjoy the film!
  Just call me if you need me."
- play_music plays on the owner's Spotify (MUSIC in the message says whether it is connected). text: for a song
  its title and artist; for a mood a good playlist search ("soft romantic R&B", "deep focus instrumental", "party
  hits 2020s"); for their own playlist its name (pc_status lists them). Pick number by what they asked: a song 0,
  a mood or genre 1, an artist 2, an album 3, one of their playlists 4 ("play my gym playlist"), their liked
  songs 5. youtube only when they say YouTube or want to watch something.
- sleep, shutdown and restart: the app asks the owner for a yes itself; say you will once they confirm, and that
  the trading bots stop while the PC is off or asleep.
- remember: when the owner tells you something about themselves or their preferences to keep, or says
  "remember"; forget when they ask you to.
- type_text types into the window the owner was using before they called you.
- You cannot place, change or close trades, and never stop the trading bots, NinjaTrader or MetaTrader."""


def answer_json(content) -> dict:
    """The JSON answer in Claude's last reply. After a web search the reply can carry search results and the answer
    split over several text pieces (citations), so the text after the last non-text piece is joined first."""
    blocks = list(content or [])
    last = max((i for i, b in enumerate(blocks) if getattr(b, "type", "") != "text"), default=-1)
    tries = ["".join(b.text for b in blocks[last + 1:] if getattr(b, "type", "") == "text"),
             "".join(b.text for b in blocks if getattr(b, "type", "") == "text")]
    for text in tries:
        for cand in (text, text[text.find("{"):text.rfind("}") + 1]):
            try:
                out = json.loads(cand)
                if isinstance(out, dict):
                    return out
            except ValueError:
                pass
    return {"speech": tries[0].strip() or "I'm sorry, something went wrong with my answer.", "proof": [], "show": []}


class Brain:
    """One short, append-only conversation at a time (follow-up questions keep context); a new one after a pause.
    Claude may call tools first (the PC's state, the screen, a command after a spoken yes, a web search); its last
    reply is the answer in ANSWER_SCHEMA."""
    MAX_STEPS = 6                                          # Claude turns per question (tools in between)

    def __init__(self, settings: dict, hands: pc.Hands | None = None):
        import anthropic                                   # only here: the app starts without it for --help
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=settings.get("ANTHROPIC_API_KEY") or None)
        city = settings.get("VERTEX_CITY", "Prague") or "Prague"
        self.system = system_prompt(settings.get("VERTEX_ADDRESS", ""), city)
        self.model = (settings.get("VERTEX_MODEL") or MODEL).strip()
        self.hands = hands
        # the basic web search: 7-11 s an answer vs 19-35 s for web_search_20260209, which runs code to filter
        # the results (2026-10-16 test, same answers) -- too slow to wait for in a spoken conversation
        self.tools = (list(pc.TOOLS) if hands else []) + [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 2,
             "user_location": {"type": "approximate", "city": city, "timezone": "Europe/Prague"}}]
        self.messages: list = []
        self.questions = 0
        self.last = 0.0

    def _create(self):
        kwargs = dict(model=self.model, max_tokens=4000, system=self.system, messages=self.messages,
                      tools=self.tools, cache_control={"type": "ephemeral"},
                      output_config={"effort": "low", "format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
                      betas=["server-side-fallback-2026-07-01"])
        try:                                               # refusals fall back to another model server-side
            return self.client.beta.messages.create(fallbacks="default", **kwargs)
        except TypeError:                                  # an SDK without the typed parameter
            return self.client.beta.messages.create(extra_body={"fallbacks": "default"}, **kwargs)

    def ask(self, question: str, stats: dict, screen: str = "", memory: str = "", confirm=None,
            music: str = "") -> dict:
        # a fresh conversation after 3 questions or a pause: each question carries the whole snapshot, and from
        # the 4th one on Claude took 9-10 s instead of 2-5 s (2026-10-16)
        if time.monotonic() - self.last > 120 or self.questions >= 3:
            self.messages, self.questions = [], 0          # a fresh conversation (history is never edited)
        self.questions += 1
        self.messages.append({"role": "user", "content": "SNAPSHOT:\n" + json.dumps(stats, default=str)
                              + f"\n\nSCREEN: {screen or 'the dashboard is not open on this laptop'}"
                              + f"\n\nWHAT YOU REMEMBER ABOUT THE OWNER:\n{memory or '(nothing yet)'}"
                              + f"\n\nNOW: {datetime.now():%A %d %B %Y, %H:%M}"
                              + (f"\n\nMUSIC: {music}" if music else "")
                              + "\n\nQUESTION: " + question})
        resp = None
        for _ in range(self.MAX_STEPS):
            resp = self._create()
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":           # a long web search: let it carry on
                continue
            uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                break
            results = []
            for b in uses:
                try:
                    content = (self.hands.tool(b.name, dict(b.input or {}), confirm or (lambda q, d="": False))
                               if self.hands else "this laptop has no PC control")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
                except Exception as e:  # noqa: BLE001
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                    "content": f"failed: {type(e).__name__}: {e}"[:500]})
            self.messages.append({"role": "user", "content": results})
        self.last = time.monotonic()
        if resp is None or resp.stop_reason in ("tool_use", "pause_turn"):
            self.messages, self.questions = [], 0          # ran out of steps: start clean next time
            return {"speech": "I'm sorry, that took me too many steps. Could you ask me again, more simply?",
                    "proof": [], "show": [], "ui": {}, "actions": []}
        if resp.stop_reason == "refusal":
            return {"speech": "I'm afraid I can't answer that one.", "proof": [], "show": [], "ui": {}, "actions": []}
        out = answer_json(resp.content)
        return {"speech": str(out.get("speech", "")), "proof": list(out.get("proof") or []),
                "show": [x for x in out.get("show") or [] if x in TABLES], "ui": clean_ui(out.get("ui")),
                "actions": pc.clean_actions(out.get("actions"))}


# --- ears: wake word (Vosk, offline) + the question (faster-whisper, on this laptop) -------------------

def ascii_path(p: Path) -> str:
    """Vosk cannot open a folder whose path has letters like "č" (2026-10-09: "Počítač"): Windows' short
    8.3 form of the same path, when the full one is not plain ASCII."""
    full = str(p)
    if full.isascii():
        return full
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(full, buf, 1024) and buf.value.isascii():
            return buf.value
    except Exception:  # noqa: BLE001
        pass
    return full


def gpu_libraries() -> None:
    """NVIDIA's cuBLAS and cuDNN, installed into this app's Python (pip nvidia-cublas-cu12 nvidia-cudnn-cu12), where
    the speech model looks for them."""
    for b in (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia").glob("*/bin"):
        if str(b) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(b))
        except (OSError, AttributeError):
            pass


# a sentence that stops on one of these is not finished: wait longer before answering ("put on some music and ...")
_HANGING = set("and or but so because then the a an to of for with from into about as at by my your our their his her "
               "is are was were be been will would could should can i we um uh er erm hmm like also maybe some any "
               "if when where which who whose whether than".split())
SPEECH_PROMPT = ("Hey Vertex. Spotify, YouTube, Netflix, Obsidian, Chrome, trial, funded, fleet, brain, lab, dashboard, "
                 "evaluation, payout, race, NQ, Lucid, real test.")


def memory_names(memory: str) -> str:
    """The names in what Vertex remembers ("Safi", "Careless Whisper", "George Michael") as spelling hints for the
    speech model -- it wrote "Sally" for "Safi" (2026-10-16)."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?:])\s+|\n", memory or ""):
        for w in re.findall(r"[A-Za-z][\w'-]*", sent)[1:]:            # not a sentence's first word
            if w[0].isupper() and w not in out and w.lower() not in ("i", "i'll", "i'm", "i've"):
                out.append(w)
    return ", ".join(out[:30])


def unfinished(text: str) -> bool:
    """True when the words so far stop mid-sentence ("... and", "... to the", "...", ",")."""
    t = (text or "").strip()
    words = re.findall(r"[a-z']+", t.lower())
    return bool(words) and (words[-1] in _HANGING or t.endswith((",", "...", "…", "-")))


class Ears:
    RATE = 16_000
    BLOCK = 1_600                                         # 0.1 s
    MIN_CONF = 0.75                                       # both wake words heard at least this surely
    VAD_WIN = 512 * 20                                    # the voice detector looks at the last 0.64 s
    LONGER = 1.6                                          # extra quiet allowed after an unfinished sentence

    def __init__(self, settings: dict):
        gpu_libraries()                                   # before the speech model's library loads
        import numpy as np
        import sounddevice as sd
        import vosk
        self.np, self.sd, self.vosk = np, sd, vosk
        vosk.SetLogLevel(-1)
        model_dir = next((p for p in (HERE / "models").glob("vosk-model*") if p.is_dir()), None)
        if model_dir is None:
            raise RuntimeError("the Vosk model is missing -- run Install-Voice.ps1 again")
        self.vosk_model = vosk.Model(ascii_path(model_dir))
        self.grammar = json.dumps(["hey vertex", "[unk]"])
        self.whisper_name = settings.get("VERTEX_WHISPER", "base.en")
        self.gpu_name = settings.get("VERTEX_WHISPER_GPU", "distil-large-v3.5") or "distil-large-v3.5"
        self.device_pref = (settings.get("VERTEX_DEVICE", "auto") or "auto").lower()
        self.device = ""
        try:
            self.pause = min(3.0, max(0.6, float(settings.get("VERTEX_PAUSE", "1.2"))))
        except ValueError:
            self.pause = 1.2
        self.whisper = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self.vad = None                                   # Silero voice detector (ships with faster-whisper)
        try:
            from faster_whisper.vad import get_vad_model
            self.vad = get_vad_model()
        except Exception as e:  # noqa: BLE001
            log.warning("voice detector not available, loudness instead: %r", e)
        self._vad_win = np.zeros(self.VAD_WIN, dtype=np.float32)
        self._in_speech = False
        self._thresh = 300.0
        self.hints = ""                                   # names from what Vertex remembers (memory_names)
        self.q: queue.Queue = queue.Queue()
        self.level: list[float] = []                      # recent block loudness (the room's noise floor)
        self.after_wake = b""
        self._new_recognizer()
        self.stream = sd.RawInputStream(samplerate=self.RATE, blocksize=self.BLOCK, channels=1, dtype="int16",
                                        callback=lambda data, n, t, s: self.q.put(bytes(data)))
        self.stream.start()

    def _new_recognizer(self) -> None:
        self.rec = self.vosk.KaldiRecognizer(self.vosk_model, self.RATE, self.grammar)
        self.rec.SetWords(True)
        self.utt: list[bytes] = []                        # audio of the utterance being decoded
        self.utt_start = 0                                # its first sample, counted from the recognizer's start
        self.fed = 0

    def _rms(self, b: bytes) -> float:
        a = self.np.frombuffer(b, dtype=self.np.int16).astype(self.np.float32)
        return float(self.np.sqrt(self.np.mean(a * a))) if len(a) else 0.0

    def flush(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()
        self._new_recognizer()

    def wake_in(self, result: dict):
        """End time (s) of "hey vertex" in a final Vosk result when both words are heard surely, else None."""
        w = result.get("result") or []
        for a, b in zip(w, w[1:]):
            if (a.get("word") == "hey" and b.get("word") == "vertex"
                    and min(a.get("conf", 0), b.get("conf", 0)) >= self.MIN_CONF):
                return float(b["end"])
        return None

    def heard_wake(self) -> bool:
        """Process what the microphone has; True once "hey Vertex" was heard (whole phrase, sure words).
        Whatever was said right after it is kept in self.after_wake as the start of the question."""
        while not self.q.empty():
            b = self.q.get_nowait()
            self.level = (self.level + [self._rms(b)])[-50:]
            self.utt.append(b)
            self.fed += len(b) // 2
            if self.rec.AcceptWaveform(b):
                end = self.wake_in(json.loads(self.rec.Result()))
                if end is not None:
                    cut = max(0, int(end * self.RATE) - self.utt_start) * 2
                    self.after_wake = b"".join(self.utt)[cut:]
                    self._new_recognizer()
                    return True
                self.utt, self.utt_start = [], self.fed
        return False

    def warm(self) -> None:
        """Load the speech-to-text model and run it once, so the first question is not slower than the rest: on the
        NVIDIA card when there is one (distil-large-v3.5: 0.4 s a question, no word wrong in the 2026-10-16 test),
        else on the processor (base.en: 0.8 s, 6% of words wrong)."""
        with self._load_lock:
            if self.whisper is not None:
                return
            from faster_whisper import WhisperModel
            zeros = self.np.zeros(self.RATE, dtype=self.np.float32)
            if self.device_pref in ("auto", "cuda", "gpu"):
                try:
                    import ctranslate2
                    if ctranslate2.get_cuda_device_count() > 0:
                        m = WhisperModel(self.gpu_name, device="cuda", compute_type="float16")
                        list(m.transcribe(zeros, language="en")[0])      # proves the card's libraries load
                        self.whisper, self.device = m, "cuda"
                except Exception as e:  # noqa: BLE001
                    log.warning("speech model on the graphics card failed, the processor instead: %r", e)
            if self.whisper is None:
                m = WhisperModel(self.whisper_name, device="cpu", compute_type="int8")
                list(m.transcribe(zeros, language="en")[0])
                self.whisper, self.device = m, "cpu"
            log.info("speech model: %s on %s", self.gpu_name if self.device == "cuda" else self.whisper_name,
                     self.device)

    def _speech(self, b: bytes) -> bool:
        """Is someone speaking in this block? The voice detector (Silero) when it loaded -- it tells a voice from a
        fan or a hum -- else loudness against the room's noise."""
        if self.vad is None:
            return self._rms(b) > self._thresh
        x = self.np.frombuffer(b, dtype=self.np.int16).astype(self.np.float32) / 32768.0
        self._vad_win = self.np.concatenate([self._vad_win, x])[-self.VAD_WIN:]
        try:
            p = float(self.np.max(self.np.asarray(self.vad(self._vad_win)).reshape(-1)[-4:]))
        except Exception as e:  # noqa: BLE001
            log.warning("voice detector failed, loudness instead: %r", e)
            self.vad = None
            return self._rms(b) > self._thresh
        self._in_speech = p >= (0.35 if self._in_speech else 0.5)
        return self._in_speech

    def _interim(self, chunks: list[bytes]) -> dict:
        """Write down what was said so far, in the background (to see if the sentence is finished; when nothing
        more is said it is the question, so the answer starts sooner)."""
        job: dict = {}
        audio = self.np.frombuffer(b"".join(chunks), dtype=self.np.int16)

        def run() -> None:
            try:
                job["text"] = self.transcribe(audio)
            except Exception as e:  # noqa: BLE001
                log.warning("writing it down failed: %r", e)
                job["text"] = None
            job["done"] = True

        threading.Thread(target=run, daemon=True).start()
        return job

    def record(self, start_timeout: float = 6.0, max_len: float = 30.0):
        """The question: (16 kHz int16 samples, its words when already written down else None), or None when
        nobody spoke. Starts with what followed the wake word. Ends after VERTEX_PAUSE seconds of quiet (1.2 s) --
        1.6 s more when the words so far stop mid-sentence ("... and", "... to the") or only a word was said -- so
        Vertex doesn't cut in while you think (2026-10-16: "don't jump into my sentences"; it was 0.8 s of quiet)."""
        floor = sorted(self.level)[len(self.level) // 2] if self.level else 150.0
        self._thresh = max(3.0 * floor, 300.0)
        self._vad_win[:] = 0.0
        self._in_speech = False
        blocks = [self.after_wake[i:i + 2 * self.BLOCK] for i in range(0, len(self.after_wake), 2 * self.BLOCK)]
        self.after_wake = b""
        st = {"got": [], "started": False, "quiet": 0.0, "spoken": 0.0, "interim": None, "raw": "", "run": 0,
              "busy": None}
        t0 = time.monotonic()
        # the raw words as they come (Vosk, every word it knows): Whisper tidies a cut-off sentence into a finished
        # one ("... music and dim the" -> "... music in Dimna."), Vosk leaves the "and dim the" hanging (2026-10-16)
        raw = None
        if getattr(self, "vosk_model", None) is not None:
            raw = self.vosk.KaldiRecognizer(self.vosk_model, self.RATE)
        done_raw: list[str] = []

        def take(b: bytes) -> bool:
            """Add a block; True when the question has ended."""
            dt = len(b) / 2 / self.RATE
            speech = self._speech(b)
            if not st["started"]:
                if not speech:
                    return False
                st["started"] = True
            st["got"].append(b)
            if raw is not None:
                if raw.AcceptWaveform(b):
                    done_raw.append(json.loads(raw.Result()).get("text", ""))
                    st["raw"] = " ".join(done_raw)
                else:
                    st["raw"] = " ".join(done_raw + [json.loads(raw.PartialResult()).get("partial", "")])
            st["run"] = st["run"] + 1 if speech else 0
            if speech and (st["run"] >= 2 or st["quiet"] < 0.3):   # a lone 0.1 s click in a pause is not speech
                st["quiet"], st["interim"] = 0.0, None   # still talking: what was written down is out of date
                st["spoken"] += dt
            else:
                st["quiet"] += dt
            busy = st["busy"]
            if st["quiet"] >= 0.4 and st["interim"] is None and not (busy and not busy.get("done")):
                st["interim"] = st["busy"] = self._interim(list(st["got"]))   # one at a time, on the latest words
            it = st["interim"]
            ready = bool(it and it.get("done"))
            need = self.pause
            if (st["spoken"] < 0.3 or not ready or unfinished(st["raw"].strip())
                    or (it.get("text") and unfinished(it["text"]))):
                need = self.pause + self.LONGER          # (not written down yet: ends as soon as it is, if finished)
            return st["quiet"] >= need - 1e-6 or sum(len(x) for x in st["got"]) / 2 / self.RATE >= max_len

        def result():
            audio = self.np.frombuffer(b"".join(st["got"]), dtype=self.np.int16)
            it = st["interim"]
            return audio, (it.get("text") if it and it.get("done") else None)

        for b in blocks:
            if take(b):
                return result()
        while True:
            try:
                b = self.q.get(timeout=0.5)
            except queue.Empty:
                b = None
            if b is not None and take(b):
                return result()
            if not st["started"] and time.monotonic() - t0 > start_timeout:
                return None

    def transcribe(self, audio) -> str:
        self.warm()
        with self._run_lock:
            segs, _ = self.whisper.transcribe(audio.astype(self.np.float32) / 32768.0, language="en",
                                              beam_size=5 if self.device == "cuda" else 1, temperature=0.0,
                                              condition_on_previous_text=False, without_timestamps=True,
                                              initial_prompt=SPEECH_PROMPT + (" " + self.hints + "."
                                                                              if getattr(self, "hints", "") else ""))
            # temperature 0 only: on a cut-off sentence the default retries (up to 5 more passes) took 1.5-2 s
            # instead of 0.4 s, so Vertex couldn't tell in time whether the sentence was finished (2026-10-16)
            text = " ".join(s.text.strip() for s in segs).strip()
        low = text.lower()
        for w in ("hey vertex", "vertex"):                # the wake word is not part of the question
            if low.startswith(w):
                text = text[len(w):].lstrip(" ,.!?")
                break
        return text


# --- voice: Microsoft neural voice (edge-tts), Windows' own voice if offline ----------------------------

def sentences(text: str) -> list[str]:
    """Speech in sentence-sized pieces (very short ones joined to the next), so speaking starts sooner."""
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
        cur = f"{cur} {piece}".strip()
        if len(cur) >= 12:                               # "Yes." is joined to the next one
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    return parts


class Voice:
    """Microsoft's neural voice (edge-tts, the British butler), Windows' own voice only when it cannot be reached.
    2026-10-16: Microsoft's service started refusing the installed edge-tts (403) and Vertex silently spoke with
    the Windows voice -- now it updates edge-tts by itself and says so in vertex_voice.log."""

    def __init__(self, settings: dict):
        self.voice = settings.get("VERTEX_VOICE", "en-GB-RyanNeural")
        self.muted = False
        self.healed = False

    def _tts(self, text: str):
        """One piece of speech as 24 kHz int16 samples; raises when the voice service fails."""
        import asyncio

        import edge_tts
        import miniaudio
        import numpy as np

        async def get() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, self.voice).stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        mp3 = asyncio.run(get())
        if not mp3:
            raise RuntimeError("the voice service sent no audio")
        pcm = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=24_000)
        return np.frombuffer(pcm.samples, dtype=np.int16)

    def heal(self, err: Exception) -> bool:
        """Microsoft refused the voice library (it changes its check now and then): update edge-tts, load the new
        version. True when the next try may work."""
        if self.healed or "403" not in repr(err):
            return False
        self.healed = True
        log.warning("voice service refused edge-tts (%r) -- updating edge-tts", err)
        try:
            py = Path(sys.executable)
            py = py.with_name("python.exe") if py.name.lower() == "pythonw.exe" else py
            subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "--quiet", "edge-tts"], timeout=180,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            for name in [m for m in sys.modules if m == "edge_tts" or m.startswith("edge_tts.")]:
                del sys.modules[name]                   # the next import loads the new version
            log.warning("edge-tts updated")
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("edge-tts update failed: %r", e)
            return False

    def check(self) -> bool:
        """At start: is the butler voice reachable? Repairs the library if Microsoft refused it."""
        try:
            self._tts("Ready.")
            return True
        except Exception as e:  # noqa: BLE001
            if self.heal(e):
                try:
                    self._tts("Ready.")
                    return True
                except Exception as e2:  # noqa: BLE001
                    e = e2
            log.warning("butler voice not available, Windows' voice instead: %r", e)
            return False

    def _fallback(self, text: str) -> None:
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.say(text)
            e.runAndWait()
        except Exception:  # noqa: BLE001
            pass

    def say(self, text: str) -> None:
        if self.muted or not text:
            return
        parts = sentences(text)
        try:
            import sounddevice as sd
        except Exception:  # noqa: BLE001
            return self._fallback(text)
        made: queue.Queue = queue.Queue()

        def make() -> None:                              # the next pieces are made while one plays
            for part in parts:
                try:
                    made.put(self._tts(part))
                except Exception as e:  # noqa: BLE001
                    made.put(e)
                    return
            made.put(None)

        threading.Thread(target=make, daemon=True).start()
        spoken = 0
        while True:
            item = made.get()
            if item is None:
                return
            if isinstance(item, Exception):
                log.warning("butler voice failed: %r", item)
                rest = " ".join(parts[spoken:])
                if spoken == 0 and self.heal(item):
                    try:
                        sd.play(self._tts(rest), 24_000)
                        sd.wait()
                        return
                    except Exception as e:  # noqa: BLE001
                        log.warning("butler voice still failing: %r", e)
                return self._fallback(rest)
            sd.play(item, 24_000)
            sd.wait()
            spoken += 1

    def chime(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            t = np.linspace(0, 0.12, int(24_000 * 0.12), endpoint=False)
            sd.play((np.sin(2 * np.pi * 880 * t) * 0.25 * 32767).astype(np.int16), 24_000)
            sd.wait()
        except Exception:  # noqa: BLE001
            pass


# --- the dashboard window on this laptop ------------------------------------------------------------------

def open_dashboard(settings: dict) -> None:
    """The viewer app (Install-Viewer.ps1's "Vertex" icon) when installed, else Edge as an app window."""
    lnk = Path(os.environ.get("USERPROFILE", "")) / "Desktop" / "Vertex.lnk"
    url = dashboard_url(settings)
    try:
        if lnk.exists():
            os.startfile(str(lnk))                       # noqa: S606 -- the owner's own shortcut
            return
        for exe in (Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
                    Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe"):
            if exe.exists():
                subprocess.Popen([str(exe), f"--app={url}", "--start-maximized"])
                return
        import webbrowser
        webbrowser.open(url)
    except OSError:
        pass


def front_dashboard(settings: dict) -> str:
    """Bring this laptop's dashboard window to the front, or open it. Returns "front", "opened" or "none"."""
    if sys.platform != "win32":
        return "none"
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    found: list = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if buf.value.startswith(DASHBOARD_TITLE):
                    found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    if not found:
        open_dashboard(settings)
        return "opened"
    h = found[0]
    if user32.IsIconic(h):
        user32.ShowWindow(h, 9)                          # SW_RESTORE
    # Windows lets only the app the user is using take the front; joining its input for a moment counts.
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    me = kernel32.GetCurrentThreadId()
    joined = fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True)
    user32.BringWindowToTop(h)
    user32.SetForegroundWindow(h)
    if joined:
        user32.AttachThreadInput(me, fg_thread, False)
    return "front"


# --- the conductor: one job at a time, so Vertex never talks over itself --------------------------------

class Conductor(threading.Thread):
    HEARTBEAT_S = 30
    INBOX_S = 1.5

    def __init__(self, settings: dict):
        super().__init__(daemon=True)
        self.settings = settings
        self.stats = Stats(settings)
        self.voice = Voice(settings)
        self.flags = {"voice": True, "announce": settings.get("VERTEX_ANNOUNCE", "1") == "1"}
        self.jobs: queue.Queue = queue.Queue()
        self.brain = None
        self.ears = None
        self.screen = {}                                  # what the dashboard on this laptop shows
        self.inbox_after = datetime.now(timezone.utc).isoformat()
        self.reports: queue.Queue = queue.Queue()         # posted by a helper thread: never slows the ears
        self._state, self._state_sent = "", 0.0
        self.on_state = None                              # the tray's tooltip
        self.stopping = False
        self._prefetch: tuple[float, dict] | None = None    # the bots' numbers, fetched while you speak
        self._brain_lock = threading.Lock()
        self.hands = None                                 # the PC (pc_control.py); None when it cannot load
        try:
            mem = Path(settings.get("VERTEX_MEMORY") or (HERE / "memory.md"))
            self.hands = pc.Hands(mem, HERE / "timers.json",
                                  spotify=sp.Spotify(settings.get("SPOTIFY_CLIENT_ID", ""), HERE / "spotify_token.json"))
        except Exception as e:  # noqa: BLE001
            log.warning("PC control off: %r", e)
        self._asking = False                              # waiting for a spoken yes (no "One moment" over it)
        self._paused: list[str] = []                      # music paused while you speak, played again after
        self._keep_paused = False                         # ... unless the music was changed meanwhile
        threading.Thread(target=self._reporter, daemon=True).start()

    # -- telling the dashboard --
    def _reporter(self) -> None:
        while True:
            ev = self.reports.get()
            for _ in range(2):
                if self.stats.post(ev):
                    break
                time.sleep(1.0)

    def report(self, ev: dict) -> None:
        self.reports.put(ev)

    def state(self, s: str, detail: str = "", force: bool = False) -> None:
        if s != self._state or force or time.monotonic() - self._state_sent > self.HEARTBEAT_S:
            self._state, self._state_sent = s, time.monotonic()
            self.report({"kind": "state", "state": s, "detail": detail, **self.flags})
            if self.on_state:
                self.on_state(s)

    def set_flags(self, **kw) -> None:
        self.flags.update({k: bool(v) for k, v in kw.items() if k in self.flags})
        self.state(self._state or "idle", force=True)

    # -- talking --
    def say(self, text: str) -> None:
        self.voice.muted = not self.flags["voice"]
        if text and not self.voice.muted:
            self.state("speaking")
        self.voice.say(text)
        if self.ears:
            self.ears.flush()                             # not its own voice as the next question

    def warm_brain(self) -> None:
        """Start the Claude client (4-5 s on this laptop, 2026-10-16) -- at start, not at the first question."""
        with self._brain_lock:
            if self.brain is None:
                self.brain = Brain(self.settings, self.hands)

    def screen_text(self) -> str:
        page = (self.screen or {}).get("page")
        return f"the dashboard on this laptop shows the {page} page" if page else ""

    def answer(self, question: str, source: str = "voice", reply_to: str = "") -> None:
        base = {"kind": "exchange", "question": question, "source": source, "reply_to": reply_to}
        quick = quick_command(question, (self.screen or {}).get("page", ""))
        if quick:
            if quick["flags"]:
                self.set_flags(**quick["flags"])
            acts = quick.get("actions") or []
            speech = self.now_playing_words() if quick.get("now_playing") else quick["speech"]
            self.report(dict(base, speech=speech, proof=[], show=[], ui=quick["ui"], did=[pc.describe(a) for a in acts]))
            self.act(acts, speech)
            return
        self.state("thinking", question[:120])
        try:
            pre = self._prefetch
            self._prefetch = None
            stats = pre[1] if pre and time.monotonic() - pre[0] < 30 else self.stats.summary()
        except (urllib.error.URLError, OSError, ValueError):     # the PC still listens; the bots' numbers are out
            stats = {"error": "the bots' server can't be reached right now (is Tailscale on, the server running?)"}
        try:
            self.warm_brain()
            got: dict = {}
            memory = self.hands.memory_text() if self.hands else ""
            music = ("the owner's Spotify is connected (Premium)" if self.hands and self.hands.spotify_on()
                     else "Spotify is not connected yet: play_music plays on YouTube")

            def think() -> None:
                try:
                    got["out"] = self.brain.ask(question, stats, self.screen_text(), memory, self.confirm, music)
                except Exception as e:  # noqa: BLE001
                    got["err"] = e

            worker = threading.Thread(target=think, daemon=True)
            worker.start()
            worker.join(FILLER_AFTER_S)
            if worker.is_alive() and not self._asking:    # a while yet: say so instead of silence
                address = self.settings.get("VERTEX_ADDRESS", "").strip()
                self.say(f"One moment, {address}." if address else "One moment.")
                self.state("thinking", question[:120])
            worker.join()
            if "err" in got:
                raise got["err"]
            out = got["out"]
        except Exception as e:  # noqa: BLE001 -- key missing, no internet, API error
            log.warning("Claude failed: %r", e)
            msg = "My connection to Claude isn't working at the moment."
            self.report(dict(base, speech=f"{msg} ({type(e).__name__}: {str(e)[:160]})", proof=[], show=[], ui={}))
            self.say(msg)
            return
        proof = [{"label": lab, "path": str(p.get("path", "")), "value": val, "found": ok}
                 for p, (lab, val, ok) in zip(out["proof"], resolve_proof(stats, out["proof"]))]
        self.report(dict(base, speech=out["speech"], proof=proof, show=out["show"], ui=out["ui"],
                         did=[pc.describe(a) for a in out["actions"]]))
        self.act(out["actions"], out["speech"])

    # -- the PC --
    def act(self, actions: list[dict], speech: str) -> None:
        """Say the answer while its actions run. New music starts once Vertex has finished speaking (it is found
        meanwhile); sleep, shut down and restart only after a spoken yes; what didn't work is said at the end."""
        if not self.hands or not actions:
            self.say(speech)
            return
        tunes = ("play_music", "youtube")
        if any(a["do"] in tunes or (a["do"] == "media" and a["text"] in ("pause", "stop")) for a in actions):
            self._keep_paused = True                      # the music was changed: don't play the old one again
        now = [a for a in actions if a["do"] not in pc.NEEDS_YES and a["do"] not in tunes]
        music = [a for a in actions if a["do"] in tunes][:2]   # a song, then a mood after it
        later = [a for a in actions if a["do"] in pc.NEEDS_YES][:1]
        errors: list[str] = []
        found: dict = {}

        def work() -> None:
            for a in now:
                err = self.hands.do(a)
                if err:
                    errors.append(err)
            for i, a in enumerate(music):                 # Spotify or YouTube: found while Vertex speaks
                found[i] = self.hands.find_music(a)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.say(speech)
        worker.join(25)
        if music:
            err = self.hands.start_music(found.get(0), music[0], found.get(1) if len(music) > 1 else None)
            if err:
                errors.append(err)
        for a in later:
            if self.confirm("Shall I " + pc.NEEDS_YES[a["do"]] + "?"):
                err = self.hands.do(a)
                if err:
                    errors.append(err)
                elif a["do"] in ("shutdown", "restart"):
                    self.say("In thirty seconds. Say 'Hey Vertex, cancel the shutdown' to stop it.")
            else:
                self.say("Very well, I'll leave it.")
        if errors:
            msg = "I'm afraid " + "; and ".join(errors) + "."
            self.report({"kind": "exchange", "question": "", "source": "voice", "speech": msg, "proof": [],
                         "show": [], "ui": {}})
            self.say(msg)

    def confirm(self, question: str, detail: str = "") -> bool:
        """Ask out loud and listen for a yes (the dashboard shows the question, and a command's exact text)."""
        self._asking = True
        try:
            self.report({"kind": "exchange", "question": "", "source": "voice", "proof": [], "show": [], "ui": {},
                         "speech": question + (f"  [command: {detail[:300]}]" if detail else "")})
            if not self.ears:
                return False
            self.say(question)
            self.state("listening", "yes or no")
            got = self.ears.record(start_timeout=7.0, max_len=10.0)
            if got is None:
                return False
            audio, text = got
            text = text if text is not None else self.ears.transcribe(audio)
            log.info("asked %r -> heard %r", question, text)
            return pc.is_yes(text)
        finally:
            self._asking = False

    def now_playing_words(self) -> str:
        """What plays (or was playing before Vertex paused it to listen)."""
        sess = pc.media_sessions() if self.hands else []
        pick = ([s for s in sess if s["app"] in self._paused] or [s for s in sess if s["playing"]]
                or [s for s in sess if s["title"]])
        if not pick:
            return "Nothing is playing at the moment."
        s = pick[0]
        by = f" by {s['artist']}" if s["artist"] else ""
        return f"This is {s['title']}{by}, on {s['name']}."

    def listen_and_answer(self, follow_ups: int = 2) -> None:
        if not self.ears:
            self.report({"kind": "exchange", "question": "", "source": "voice", "proof": [], "show": [], "ui": {},
                         "speech": "The microphone isn't available on this laptop; type your question instead."})
            return
        if self.hands:                                    # music off while you speak: Vertex hears you, not it
            self.ears.hints = memory_names(self.hands.memory_text())
            self._keep_paused = False
            self._paused = self.hands.pause_all()
            try:
                import ctypes
                self.hands.typing_target = ctypes.windll.user32.GetForegroundWindow()
            except Exception:  # noqa: BLE001
                pass
        if self.settings.get("VERTEX_FRONT", "1") == "1":
            threading.Thread(target=front_dashboard, args=(self.settings,), daemon=True).start()
        self.state("listening")

        def fetch() -> None:                              # the numbers arrive while you speak
            try:
                self._prefetch = (time.monotonic(), self.stats.summary())
            except Exception:  # noqa: BLE001
                self._prefetch = None

        threading.Thread(target=fetch, daemon=True).start()
        self.voice.chime()
        self.ears.flush()
        try:
            for i in range(follow_ups + 1):
                self.state("listening", "" if i == 0 else "a follow-up")
                got = self.ears.record(start_timeout=6.0 if i == 0 else 5.0)
                if got is None:
                    break
                audio, q = got
                self.state("thinking", "writing it down")
                q = q if q is not None else self.ears.transcribe(audio)
                if not q:
                    break
                if i > 0 and q.lower().strip(" .!") in ("thank you", "thanks", "that's all", "thats all", "goodbye",
                                                        "no", "no thanks", "no thank you", "nothing"):
                    self.report({"kind": "exchange", "question": q, "source": "voice", "speech": "Very good.",
                                 "proof": [], "show": [], "ui": {}})
                    self.say("Very good.")
                    break
                self.answer(q)
        finally:
            if self._paused and not self._keep_paused:
                for app in self._paused:
                    pc.media_command("play", app)
            self._paused = []

    def ring_timers(self) -> None:
        for text in self.hands.due_timers() if self.hands else []:
            self.report({"kind": "announce", "text": "Reminder: " + text, "key": ""})
            for _ in range(3):
                self.voice.chime()
                time.sleep(0.15)
            self.say(text)

    # -- on its own --
    def announcements(self) -> None:
        if not self.flags["announce"]:
            return
        try:
            evs = self.stats.events(self.last_events)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return
        for e in evs:
            self.last_events = max(self.last_events, datetime.fromisoformat(e["ts"]))
            self.report({"kind": "announce", "text": e["text"], "key": f"{e['ts']} {e['text']}"})
            self.say(e["text"])

    def check_inbox(self) -> None:
        try:
            box = self.stats.inbox(self.inbox_after)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return
        self.screen = box.get("screen") or {}
        for item in box.get("questions") or []:
            self.inbox_after = max(self.inbox_after, item.get("ts", ""))
            self.jobs.put(item)

    def run(self) -> None:
        self.last_events = datetime.now(timezone.utc)
        try:
            self.ears = Ears(self.settings)
            threading.Thread(target=self.ears.warm, daemon=True).start()   # the speech model, before the 1st question
        except Exception as e:  # noqa: BLE001 -- no microphone / model: typed questions still work
            self.state("idle", f"voice input off: {e}")
            log.warning("voice input off: %r", e)
        threading.Thread(target=self.warm_brain, daemon=True).start()        # Claude, before the 1st question
        self.voice.check()                                # the butler voice -- repaired here if Microsoft refused it
        try:
            self.stats.summary()
            self.say("Vertex online.")
        except (urllib.error.URLError, OSError, ValueError):
            pass                                          # the tray says "server not reachable"
        next_events, next_inbox, next_timers = time.monotonic() + 30, 0.0, 0.0
        while not self.stopping:
            if time.monotonic() >= next_timers:
                next_timers = time.monotonic() + 1.0
                self.ring_timers()
            self.state("idle", "" if self.ears else "type questions on the Vertex page")
            try:
                job = self.jobs.get(timeout=0.15)
            except queue.Empty:
                job = None
            kind = job.get("kind") if isinstance(job, dict) else (job[0] if job else None)
            if kind == "ask":
                self.answer(job.get("text", ""), "typed", job.get("id", ""))
            elif kind == "listen":
                self.listen_and_answer()
            elif kind == "set":
                self.set_flags(**{k: job[k] for k in ("voice", "announce") if k in job})
            elif kind == "front":
                front_dashboard(self.settings)
            elif self.ears and self.ears.heard_wake():
                self.listen_and_answer()
            if time.monotonic() >= next_inbox:
                next_inbox = time.monotonic() + self.INBOX_S
                self.check_inbox()
            if time.monotonic() >= next_events:
                next_events = time.monotonic() + 60
                self.announcements()

    def stop(self) -> None:
        self.stopping = True
        self.stats.post({"kind": "state", "state": "offline", **self.flags})


class Tray:
    """The V icon by the clock -- Vertex's only visible part on the laptop."""

    def __init__(self, cond: Conductor):
        self.cond = cond
        self.icon = None
        try:
            import pystray
            from PIL import Image
        except ImportError:
            return
        flags, s = cond.flags, cond.settings

        def toggle(key):
            cond.set_flags(**{key: not flags[key]})

        def quit_(*a):
            cond.stop()
            self.icon.stop()

        menu = pystray.Menu(
            pystray.MenuItem("Open the dashboard", lambda *a: cond.jobs.put(("front",)), default=True),
            pystray.MenuItem("Voice", lambda *a: toggle("voice"), checked=lambda item: flags["voice"]),
            pystray.MenuItem("Announcements", lambda *a: toggle("announce"), checked=lambda item: flags["announce"]),
            pystray.MenuItem("Quit Vertex", quit_))
        self.icon = pystray.Icon("Vertex", Image.open(HERE / "vertex_v_icon.ico"), "Vertex", menu)
        words = {"idle": 'listening for "Hey Vertex"', "listening": "listening...", "thinking": "thinking...",
                 "speaking": "speaking"}
        cond.on_state = lambda st: setattr(self.icon, "title", f"Vertex - {words.get(st, st)} ({s['VERTEX_VIEWER']})")

    def run(self) -> None:
        if self.icon is None:
            self.cond.join()
        else:
            self.icon.run()


INSTANCE_PORT = 8519                                             # one Vertex per laptop


def already_running() -> bool:
    """True (after asking the running Vertex to open the dashboard) when Vertex already runs."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", INSTANCE_PORT), timeout=1.0) as c:
            c.sendall(b"show")
        return True
    except OSError:
        return False


def listen_for_second_start(on_show):
    import socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.bind(("127.0.0.1", INSTANCE_PORT))
    except OSError:
        return None
    srv.listen(2)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
                with c:
                    if c.recv(16).startswith(b"show"):
                        on_show()
            except OSError:
                return

    threading.Thread(target=loop, daemon=True).start()
    return srv


def check(settings: dict) -> int:
    """--check: the server and Claude, without microphone."""
    s = Stats(settings)
    try:
        stats = s.summary()
        print(f"server OK ({s.base}): trading day {stats.get('trading_day')}, next race {stats.get('next_race')}")
    except Exception as e:  # noqa: BLE001
        print(f"server NOT reachable at {s.base}: {e}")
        return 1
    print(f"this laptop in the dashboard: {settings['VERTEX_VIEWER']}  ({dashboard_url(settings)})")
    if not settings.get("ANTHROPIC_API_KEY"):
        print("Claude: no ANTHROPIC_API_KEY in settings.env")
        return 1
    out = Brain(settings).ask("How did the bots do today? Keep it to one sentence.", stats)
    print("Claude OK:", out["speech"])
    for label, value, found in resolve_proof(stats, out["proof"]):
        print(f"   proof  {label}: {value} {'(checked)' if found else '(NOT FOUND)'}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(HERE / "vertex_voice.log", encoding="utf-8")])
    settings = load_settings()
    if "--check" in sys.argv:
        return check(settings)
    if "--spotify" in sys.argv:                                  # connect Spotify once (opens the browser)
        err = sp.connect(settings.get("SPOTIFY_CLIENT_ID", ""), HERE / "spotify_token.json")
        print("Spotify connected." if not err else f"Spotify NOT connected: {err}")
        return 1 if err else 0
    if already_running():                                        # a second start opens the dashboard
        return 0
    cond = Conductor(settings)
    listen_for_second_start(lambda: cond.jobs.put(("front",)))
    tray = Tray(cond)
    cond.start()
    if "--hidden" not in sys.argv:
        cond.jobs.put(("front",))
    tray.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
