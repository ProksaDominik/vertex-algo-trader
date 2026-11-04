"""Vertex's hands on this PC (owner, 2026-10-16: "I want to build a real Jarvis ... I need Vertex to control my
whole PC, everything, something like 'hey Vertex, my girlfriend came over and we want a romantic evening' and it
turns the romantic music on ... a full butler who I can speak to and tell him what to do").

Two kinds of help, both chosen by Claude:
  ACTIONS  listed in Claude's answer and done at once: music, play / pause / skip, volume, screen brightness, open /
           close / switch apps, websites, typing, timers, remembering, lock; sleep, shut down and restart only after
           a spoken "yes".
  TOOLS    called when Claude needs to know something first: the PC's state (what plays, open windows, battery),
           a look at the screen, and any other PowerShell command -- the command is said out loud and runs only
           after a "yes".

What Vertex never does: close or stop the trading bots, NinjaTrader, MetaTrader, the dashboard, or VS Code (where
Claude works); run a command that touches the bots' files, the Vertex settings (the Claude key) or the disks'
partitions -- whatever the request.
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("vertex")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# --- what Claude can ask for -------------------------------------------------------------------------------------

ACTIONS = {   # name -> what it does, and what "text" / "number" mean for it (both are always sent; "" / 0 = unused)
    "play_music": 'play music on the owner\'s Spotify (Premium; YouTube when Spotify is not connected): text = what '
                  '("Spaceship Viktor Sheen", "Viktor Sheen", an album, a mood like "soft romantic R&B", or the name '
                  'of one of the owner\'s playlists); number = 0 a song, 1 a mood or genre (a playlist), 2 an '
                  'artist, 3 an album, 4 one of the owner\'s own playlists, 5 the owner\'s liked songs',
    "youtube": "play on YouTube in the browser instead -- only when the owner says YouTube or wants a video: "
               "text = what to search; number = 1 for a long mix, 0 for one song or video",
    "media": "the music or video playing now: text = play, pause, next, previous or stop",
    "volume": "speaker volume: number = 0-100 to set it; or text = up, down, mute or unmute",
    "brightness": "laptop screen brightness: number = 0-100",
    "open_app": 'start an app: text = its name ("Spotify", "Netflix", "Google Chrome", "Obsidian", "Steam")',
    "switch_to": "bring an open app's window to the front (opens the app when it is not open): text = its name",
    "close_app": "close an app's windows (it may ask to save): text = its name",
    "open_website": "open a web page in the browser: text = the full https:// address",
    "show_desktop": "minimise all windows",
    "type_text": "type text into the window in front (no Enter): text = what to type",
    "timer": "a reminder / timer: number = seconds from now, text = what to say when it is time",
    "cancel_timers": "cancel all timers and reminders",
    "remember": "keep a fact about the owner for later conversations: text = the fact, one short sentence",
    "forget": "drop a remembered fact: text = words from it",
    "lock": "lock the PC (the bots keep running)",
    "sleep": "put the PC to sleep (asks first; the trading bots stop while it sleeps)",
    "shutdown": "shut the PC down in 30 seconds (asks first; the trading bots stop)",
    "restart": "restart the PC in 30 seconds (asks first; the bots are started again by the watchdog after sign-in)",
    "cancel_shutdown": "cancel a shutdown or restart that is counting down",
}
NEEDS_YES = {"sleep": "put the computer to sleep? The trading bots stop while it sleeps",
             "shutdown": "shut the computer down? The trading bots stop until it is back on",
             "restart": "restart the computer?"}
ACTION_ITEM = {"type": "object", "properties": {"do": {"type": "string", "enum": list(ACTIONS)},
                                                 "text": {"type": "string"}, "number": {"type": "number"}},
               "required": ["do", "text", "number"], "additionalProperties": False}

TOOLS = [
    {"name": "pc_status", "strict": True,
     "description": "This PC now: local time, what music or video is playing (app, title, artist, playing or "
                    "paused), speaker volume, open app windows, battery, memory use, the timers set and the "
                    "owner's Spotify playlists.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "look_at_screen", "strict": True,
     "description": "A picture of what is on the PC's screens now. Only when the owner asks you to look at, read "
                    "or help with something on the screen.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "run_command", "strict": True,
     "description": "Run a Windows PowerShell command on this PC and get its output -- for anything the actions "
                    "and the other tools cannot do. Before it runs, the owner hears `purpose` and must say yes. "
                    "Prefer read-only commands; never delete without being asked; 60 s at most.",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string", "description": "the PowerShell command"},
         "purpose": {"type": "string", "description": 'what it does, in a few plain words for the owner, e.g. '
                                                      '"turn on dark mode"'}},
         "required": ["command", "purpose"], "additionalProperties": False}},
]

# never closed (the bots, the trading platforms, Vertex and the dashboard, Claude Code's editor, the Windows shell)
PROTECTED_EXE = {"python", "pythonw", "powershell", "pwsh", "cmd", "conhost", "windowsterminal", "openconsole",
                 "ninjatrader", "terminal64", "metaeditor64", "code", "taskmgr", "lsass", "winlogon", "csrss",
                 "dwm", "svchost", "searchhost", "startmenuexperiencehost", "shellexperiencehost", "textinputhost"}
PROTECTED_TITLES = ("vertex", "program manager")
BLOCKED_COMMANDS = [   # refused whatever the request (a yes cannot unlock these)
    (r"metatraderbot|vertexvoice|vertex_voice|lucid|ninjatrader|metatrader|terminal64", "it touches the trading bots or Vertex itself"),
    (r"settings\.env|\.env\b|api[_-]?key|anthropic", "it touches stored keys"),
    (r"\b(stop-process|taskkill|kill|spps)\b[^|;]*\b(python|pythonw|ninja|terminal64)", "it would stop the bots"),
    (r"(unregister|disable|stop)-scheduledtask|schtasks[^|;]*/(delete|end|change)", "it changes the bots' watchdog"),
    (r"\b(format-volume|clear-disk|initialize-disk|remove-partition|set-partition|diskpart|bcdedit|vssadmin|"
     r"cipher\s+/w|format\s+[a-z]:)", "it changes the disks"),
    (r"\b(shutdown|restart-computer|stop-computer)\b", "use the shut-down action, which asks first"),
]
YES = re.compile(r"^\s*(yes|yeah|yep|yup|sure|of course|do it|go ahead|please do|ok|okay|correct|confirm|affirmative|"
                 r"absolutely|definitely|right|go)\b", re.I)


def blocked(command: str) -> str:
    """Why a command is refused, or "" when it may be asked about."""
    low = command.lower()
    for pat, why in BLOCKED_COMMANDS:
        if re.search(pat, low):
            return why
    return ""


def clean_actions(items) -> list[dict]:
    """Only known actions, with text and number in their place."""
    out = []
    for a in items or []:
        if isinstance(a, dict) and a.get("do") in ACTIONS:
            num = a.get("number")
            out.append({"do": a["do"], "text": str(a.get("text") or "").strip(),
                        "number": float(num) if isinstance(num, (int, float)) else 0.0})
    return out


def describe(a: dict) -> str:
    """One plain line for the dashboard: what Vertex did."""
    d, t, n = a["do"], a.get("text", ""), a.get("number", 0)
    words = {"play_music": f'music: "{t}"', "youtube": f'YouTube: "{t}"', "media": f"music: {t}",
             "volume": f"volume {int(n)}%" if t not in ("up", "down", "mute", "unmute") else f"volume {t}",
             "brightness": f"screen brightness {int(n)}%", "open_app": f"open {t}", "switch_to": f"switch to {t}",
             "close_app": f"close {t}", "open_website": f"open {t}", "show_desktop": "show the desktop",
             "type_text": f'type "{t[:40]}"', "timer": f"reminder in {fmt_secs(n)}: {t}",
             "cancel_timers": "cancel the reminders", "remember": f"remember: {t}", "forget": f"forget: {t}",
             "lock": "lock the PC", "sleep": "sleep", "shutdown": "shut down", "restart": "restart",
             "cancel_shutdown": "cancel the shutdown"}
    return words.get(d, d)


def fmt_secs(s: float) -> str:
    s = int(round(s))
    if s >= 3600 and s % 3600 == 0:
        return f"{s // 3600} h"
    if s >= 60:
        return f"{s // 60} min" + (f" {s % 60} s" if s % 60 else "")
    return f"{s} s"


def app_match(name: str, names: list[str]) -> str | None:
    """The best match for a spoken app name: exact, then whole word, then start, then close spelling."""
    want = re.sub(r"\b(the|app|application|program)\b", " ", name.lower()).strip()
    want = re.sub(r"\s+", " ", want)
    if not want:
        return None
    low = {n.lower(): n for n in names}
    if want in low:
        return low[want]
    words = [n for n in names if re.search(rf"\b{re.escape(want)}\b", n.lower())]
    if words:
        return min(words, key=len)
    starts = [n for n in names if n.lower().startswith(want)]
    if starts:
        return min(starts, key=len)
    close = difflib.get_close_matches(want, list(low), n=1, cutoff=0.75)
    return low[close[0]] if close else None


# --- the hands -----------------------------------------------------------------------------------------------------

class Hands:
    def __init__(self, memory_path: Path, timers_path: Path, load_apps: bool = True, spotify=None):
        self.memory_path = Path(memory_path)
        self.timers_path = Path(timers_path)
        self.spotify = spotify                             # spotify.Spotify, or None
        self._apps: list[dict] | None = None
        self._mutex = threading.Lock()
        self.typing_target = 0                             # the window in front when "Hey Vertex" was heard
        self.timers: list[dict] = self._load_timers()
        if load_apps:
            threading.Thread(target=self.start_apps, daemon=True).start()     # the app list, before it is needed

    # -- doing what Claude asked --
    def do(self, a: dict) -> str:
        """Carry out one action; "" when done, else a short spoken reason why not."""
        d, t, n = a["do"], a.get("text", ""), a.get("number", 0)
        try:
            fn = getattr(self, "_" + d)
            return fn(t, n) or ""
        except Exception as e:  # noqa: BLE001
            log.warning("action %s failed: %r", d, e)
            return f"the {describe(a)} didn't work"

    def _play_music(self, query: str, kind: float) -> str:
        a = {"do": "play_music", "text": query, "number": kind}
        return self.start_music(self.find_music(a), a)

    def _youtube(self, query: str, long_mix: float) -> str:
        a = {"do": "youtube", "text": query, "number": long_mix}
        return self.start_music(self.find_music(a), a)

    def spotify_on(self) -> bool:
        return bool(self.spotify and self.spotify.ready())

    def find_music(self, a: dict) -> dict:
        """Where the music will come from (looked up while Vertex speaks): {"spotify": pick}, {"youtube": video or
        None} or {"error": reason}."""
        query, kind = a.get("text", "").strip(), int(a.get("number") or 0)
        if not query and kind != 5:
            return {"error": "I didn't catch which music"}
        if a["do"] == "play_music" and self.spotify_on():
            try:
                pick = self.spotify.find(query, kind)
                if pick:
                    return {"spotify": pick}
                if kind in (4, 5):
                    return {"error": f"I can't find {'your liked songs' if kind == 5 else query} on your Spotify"}
            except Exception as e:  # noqa: BLE001 -- no internet, Spotify down: YouTube instead
                log.warning("Spotify search failed: %r", e)
        return {"youtube": find_youtube(query, kind == 1)}

    def start_music(self, plan: dict | None, a: dict, then: dict | None = None) -> str:
        """Play what find_music found: on Spotify, else the YouTube video (its search page when none was found).
        then: a second find_music plan to carry on with after the first (a song, then a mood) -- Spotify only."""
        plan = plan or self.find_music(a)
        if "error" in plan:
            return plan["error"]
        query, long_mix = a.get("text", ""), int(a.get("number") or 0) == 1
        if "spotify" in plan:
            self.pause_all()                               # the browser's video, if one plays
            pick, after = plan["spotify"], None
            if then and "spotify" in then and pick.get("offset_uri"):
                after = then["spotify"]                    # a song, then the mood once the song ends
            try:
                err = self.spotify.play(pick)
                if not err:
                    log.info("Spotify plays %s%s", pick.get("name"), f", then {after.get('name')}" if after else "")
                    if after:
                        self.spotify.follow(pick, after)
                    return ""
            except Exception as e:  # noqa: BLE001
                err = f"Spotify didn't start ({type(e).__name__})"
                log.warning("Spotify play failed: %r", e)
            if int(a.get("number") or 0) in (4, 5):
                return err
            log.info("Spotify failed (%s): YouTube instead", err)
            plan = {"youtube": find_youtube(query, long_mix)}
        vid = plan.get("youtube")
        self.pause_all()                                   # not two kinds of music at once
        if vid is None:
            open_url("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query))
            return "I could only open the YouTube search for it"
        url = f"https://www.youtube.com/watch?v={vid['id']}"
        if not long_mix:
            url += f"&list=RD{vid['id']}"                  # a song: YouTube's mix carries on with similar ones
        open_url(url)
        threading.Thread(target=self._make_sure_it_plays, args=(vid.get("title", ""),), daemon=True).start()
        return ""

    def _make_sure_it_plays(self, title: str) -> None:
        """Browsers sometimes open a video paused: press play on it once it shows up."""
        key = title.lower()[:24]
        for _ in range(20):
            time.sleep(1.0)
            for s in media_sessions():
                if key and key in (s["title"] or "").lower():
                    if not s["playing"]:
                        media_command("play", s["app"])
                    return

    def _media(self, what: str, _n) -> str:
        what = what.lower().strip()
        if what not in ("play", "pause", "next", "previous", "stop"):
            return f"I don't know how to {what} the music"
        if what in ("pause", "stop"):
            self.pause_all()
            return ""
        return "" if media_command(what) else "nothing is playing"

    def pause_all(self) -> list[str]:
        """Pause whatever plays; the apps that were paused."""
        paused = []
        for s in media_sessions():
            if s["playing"] and media_command("pause", s["app"]):
                paused.append(s["app"])
        return paused

    def _volume(self, how: str, n: float) -> str:
        ev = _endpoint()
        how = how.lower().strip()
        if how == "mute":
            ev.SetMute(1, None)
        elif how == "unmute":
            ev.SetMute(0, None)
        elif how in ("up", "down"):
            now = ev.GetMasterVolumeLevelScalar()
            ev.SetMute(0, None)
            ev.SetMasterVolumeLevelScalar(min(1.0, max(0.0, now + (0.1 if how == "up" else -0.1))), None)
        else:
            ev.SetMute(0, None)
            ev.SetMasterVolumeLevelScalar(min(100.0, max(0.0, float(n))) / 100.0, None)
        return ""

    def _brightness(self, _t, n: float) -> str:
        pct = int(min(100, max(5, n)))
        r = ps("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods) | "
               f"Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1; Brightness={pct}}} | Out-Null",
               timeout=15)
        return "" if r.returncode == 0 else "this screen's brightness can't be set from here"

    def _open_app(self, name: str, _n) -> str:
        apps = self.start_apps()
        hit = app_match(name, [a["Name"] for a in apps])
        if not hit:
            return f"I can't find an app called {name}"
        app_id = next(a["AppID"] for a in apps if a["Name"] == hit)
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"], creationflags=NO_WINDOW)
        return ""

    def _switch_to(self, name: str, _n) -> str:
        wins = [w for w in windows() if not w["protected"] or w["exe"] == "code"]
        hit = app_match(name, [w["exe"] for w in wins]) or app_match(name, [w["title"] for w in wins])
        if not hit:
            return self._open_app(name, 0)
        w = next(w for w in wins if hit in (w["exe"], w["title"]))
        bring_to_front(w["hwnd"])
        return ""

    def _close_app(self, name: str, _n) -> str:
        wins = windows()
        hit = app_match(name, sorted({w["exe"] for w in wins})) or app_match(name, [w["title"] for w in wins])
        if not hit:
            return f"{name} doesn't seem to be open"
        targets = [w for w in wins if hit in (w["exe"], w["title"])]
        if all(w["protected"] for w in targets):
            return f"I keep my hands off {name}: the bots or Claude may need it. Please close it yourself"
        for w in targets:
            if not w["protected"]:
                post_close(w["hwnd"])
        return ""

    def _open_website(self, url: str, _n) -> str:
        if not re.match(r"^https?://", url.strip(), re.I):
            url = "https://" + url.strip()
        open_url(url)
        return ""

    def _show_desktop(self, _t, _n) -> str:
        ps("(New-Object -ComObject Shell.Application).MinimizeAll()")
        return ""

    def _type_text(self, text: str, _n) -> str:
        if self.typing_target:                             # back to the window the owner was in (not the dashboard)
            bring_to_front(self.typing_target)
            time.sleep(0.3)
        type_unicode(text)
        return ""

    def _timer(self, text: str, secs: float) -> str:
        if secs <= 0:
            return "I didn't catch when"
        with self._mutex:
            self.timers.append({"due": (datetime.now() + timedelta(seconds=secs)).isoformat(timespec="seconds"),
                                "text": text or "Your timer is up."})
            self._save_timers()
        return ""

    def _cancel_timers(self, _t, _n) -> str:
        with self._mutex:
            self.timers = []
            self._save_timers()
        return ""

    def _remember(self, fact: str, _n) -> str:
        fact = fact.strip().rstrip(".")
        if not fact:
            return ""
        if not self.memory_path.exists():
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(MEMORY_HEADER, encoding="utf-8")
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(f"- {fact}. ({datetime.now():%Y-%m-%d})\n")
        return ""

    def _forget(self, words: str, _n) -> str:
        if not self.memory_path.exists():
            return "there's nothing to forget"
        lines = self.memory_path.read_text(encoding="utf-8").splitlines(keepends=True)
        facts = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
        want = words.lower().strip()
        hit = [i for i in facts if want and want in lines[i].lower()]
        if not hit:
            best = difflib.get_close_matches(want, [lines[i][2:].lower() for i in facts], n=1, cutoff=0.4)
            hit = [i for i in facts if best and lines[i][2:].lower() == best[0]]
        if not hit:
            return "I don't find that in what I remember"
        self.memory_path.write_text("".join(ln for i, ln in enumerate(lines) if i not in hit), encoding="utf-8")
        return ""

    def _lock(self, _t, _n) -> str:
        subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"], creationflags=NO_WINDOW)
        return ""

    def _sleep(self, _t, _n) -> str:
        ps("Add-Type -AssemblyName System.Windows.Forms; "
           "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false) | Out-Null")
        return ""

    def _shutdown(self, _t, _n) -> str:
        subprocess.run(["shutdown", "/s", "/t", "30"], creationflags=NO_WINDOW, check=False)
        return ""

    def _restart(self, _t, _n) -> str:
        subprocess.run(["shutdown", "/r", "/t", "30"], creationflags=NO_WINDOW, check=False)
        return ""

    def _cancel_shutdown(self, _t, _n) -> str:
        r = subprocess.run(["shutdown", "/a"], creationflags=NO_WINDOW, check=False, capture_output=True)
        return "" if r.returncode == 0 else "no shutdown was counting down"

    # -- what Claude can look up --
    def tool(self, name: str, args: dict, confirm) -> list | str:
        """A tool's result for Claude (text, or text and a picture). confirm(question) -> True when the owner
        said yes."""
        if name == "pc_status":
            return json.dumps(self.status(), default=str)
        if name == "look_at_screen":
            return screenshot_blocks()
        if name == "run_command":
            cmd, purpose = str(args.get("command", "")), str(args.get("purpose", "")).strip() or "run a command"
            why = blocked(cmd)
            if why:
                return f"Refused without asking the owner: {why}. Tell the owner you can't do this one."
            if not confirm(f"To {purpose.rstrip('.')}, I'd run a command on the computer. Shall I?", cmd):
                return "The owner said no (or didn't answer). Don't run it; say you've left it."
            r = ps(cmd, timeout=60)
            out = (r.stdout or "") + (("\nERRORS:\n" + r.stderr) if r.stderr.strip() else "")
            log.info("command run (exit %s): %s", r.returncode, cmd[:300])
            return f"exit code {r.returncode}\n{out.strip()[:4000] or '(no output)'}"
        return f"unknown tool {name}"

    def status(self) -> dict:
        out = {"local_time": datetime.now().strftime("%A %d %B %Y, %H:%M")}
        try:
            out["playing"] = [{k: s[k] for k in ("app", "title", "artist", "playing")} for s in media_sessions()]
        except Exception as e:  # noqa: BLE001
            out["playing"] = f"unknown ({type(e).__name__})"
        try:
            ev = _endpoint()
            out["volume_percent"] = round(ev.GetMasterVolumeLevelScalar() * 100)
            out["muted"] = bool(ev.GetMute())
        except Exception:  # noqa: BLE001
            pass
        out["open_windows"] = [f"{w['exe']}: {w['title'][:70]}" for w in windows()][:25]
        if self.spotify_on():
            try:
                out["spotify_playlists"] = [p["name"] for p in self.spotify.my_playlists()][:40]
            except Exception as e:  # noqa: BLE001
                out["spotify_playlists"] = f"unknown ({type(e).__name__})"
        else:
            out["spotify"] = "not connected: music plays on YouTube"
        out.update(power_and_memory())
        out["timers"] = [f"{t['due'][11:16]} {t['text']}" for t in self.timers]
        return out

    # -- timers --
    def due_timers(self) -> list[str]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._mutex:
            due = [t for t in self.timers if t["due"] <= now]
            if due:
                self.timers = [t for t in self.timers if t["due"] > now]
                self._save_timers()
        return [t["text"] for t in due]

    def _load_timers(self) -> list[dict]:
        try:
            return [t for t in json.loads(self.timers_path.read_text(encoding="utf-8")) if "due" in t]
        except (OSError, ValueError):
            return []

    def _save_timers(self) -> None:
        try:
            self.timers_path.write_text(json.dumps(self.timers), encoding="utf-8")
        except OSError:
            pass

    # -- memory --
    def memory_text(self, limit: int = 6000) -> str:
        try:
            text = self.memory_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        facts = [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]
        return "\n".join(f"- {f}" for f in facts)[-limit:]

    # -- apps --
    def start_apps(self) -> list[dict]:
        """The Start menu's apps (name, id) -- Store apps like Spotify and Netflix too. Read once (~1 s)."""
        if self._apps is None:
            try:
                r = ps("Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress", timeout=30)
                data = json.loads(r.stdout or "[]")
                self._apps = data if isinstance(data, list) else [data]
            except (ValueError, OSError, subprocess.TimeoutExpired):
                return []
        return self._apps


MEMORY_HEADER = """# Vertex remembers

What Vertex, the voice butler, knows about you -- it reads this before every conversation. Say "Hey Vertex,
remember that ..." to add to it, "forget ..." to remove; or edit this note yourself (one fact per line,
starting with "- ").

"""


# --- Windows plumbing ------------------------------------------------------------------------------------------------

def ps(command: str, timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-Command", command], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=NO_WINDOW)


def open_url(url: str) -> None:
    os.startfile(url)                                       # noqa: S606 -- the default browser


def find_youtube(query: str, long_mix: bool) -> dict | None:
    """The best YouTube video for a request: {"id", "title", "duration"}; None when the search fails."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True,
                               "skip_download": True, "socket_timeout": 10}) as y:
            res = y.extract_info(f"ytsearch8:{query}", download=False)
        vids = [e for e in res.get("entries") or [] if e.get("id") and len(e["id"]) == 11]
    except Exception as e:  # noqa: BLE001
        log.warning("YouTube search failed: %r", e)
        return None
    if not vids:
        return None
    dur = [(v, v.get("duration") or 0) for v in vids]
    pick = (next((v for v, d in dur if d >= 1200), None) if long_mix
            else next((v for v, d in dur if 60 <= d <= 900), None))
    v = pick or vids[0]
    return {"id": v["id"], "title": v.get("title") or "", "duration": v.get("duration")}


def _endpoint():
    from pycaw.pycaw import AudioUtilities
    spk = AudioUtilities.GetSpeakers()
    ev = getattr(spk, "EndpointVolume", None)              # pycaw 2025+
    if ev is None:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        ev = cast(spk.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None), POINTER(IAudioEndpointVolume))
    return ev


def _run(coro):
    return asyncio.run(coro)


def media_sessions() -> list[dict]:
    """What plays or is paused in Windows' media controls (Spotify, browsers, players)."""
    try:
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Mgr
    except ImportError:
        return []

    async def get():
        mgr = await Mgr.request_async()
        out = []
        for s in mgr.get_sessions():
            title = artist = ""
            try:
                p = await s.try_get_media_properties_async()
                title, artist = p.title or "", p.artist or ""
            except Exception:  # noqa: BLE001
                pass
            app = s.source_app_user_model_id or ""
            out.append({"app": app, "playing": int(s.get_playback_info().playback_status) == 4,
                        "title": title, "artist": artist,
                        "name": "Spotify" if "spotify" in app.lower() else app.split("!")[0].split(".")[0]})
        return out

    try:
        return _run(get())
    except Exception as e:  # noqa: BLE001
        log.warning("media sessions: %r", e)
        return []


def media_command(what: str, app: str = "") -> bool:
    """play / pause / next / previous / stop on one app's media (default: the one Windows calls current);
    media keys when Windows' media controls are not available."""
    try:
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager as Mgr
    except ImportError:
        return media_key(what)

    async def go():
        mgr = await Mgr.request_async()
        sess = next((s for s in mgr.get_sessions() if s.source_app_user_model_id == app), None) if app else None
        sess = sess or mgr.get_current_session()
        if sess is None:
            return False
        fn = {"play": sess.try_play_async, "pause": sess.try_pause_async, "next": sess.try_skip_next_async,
              "previous": sess.try_skip_previous_async, "stop": sess.try_stop_async}[what]
        return bool(await fn())

    try:
        return _run(go())
    except Exception as e:  # noqa: BLE001
        log.warning("media %s: %r", what, e)
        return media_key(what)


def media_key(what: str) -> bool:
    import ctypes
    vk = {"play": 0xB3, "pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}.get(what)
    if vk is None:
        return False
    ctypes.windll.user32.keybd_event(vk, 0, 1, 0)
    ctypes.windll.user32.keybd_event(vk, 0, 3, 0)
    return True


def windows() -> list[dict]:
    """The app windows a person would see in Alt+Tab: hwnd, exe (lower case, no .exe), title, protected."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32, dwm = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.dwmapi
    out = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):          # GW_OWNER: dialogs, popups
            return True
        if user32.GetWindowLongW(hwnd, -20) & 0x80:                                 # WS_EX_TOOLWINDOW
            return True
        cloaked = wintypes.DWORD()
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if cloaked.value:
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = ""
        h = kernel32.OpenProcess(0x1000, False, pid.value)                           # QUERY_LIMITED_INFORMATION
        if h:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, path, ctypes.byref(size)):
                exe = Path(path.value).stem.lower()
            kernel32.CloseHandle(h)
        title = buf.value
        prot = exe in PROTECTED_EXE or title.lower().startswith(PROTECTED_TITLES) or (exe == "explorer" and not
                                                                                     _is_folder(hwnd))
        out.append({"hwnd": hwnd, "exe": exe, "title": title, "protected": prot})
        return True

    user32.EnumWindows(each, 0)
    return out


def _is_folder(hwnd) -> bool:
    import ctypes
    buf = ctypes.create_unicode_buffer(64)
    ctypes.windll.user32.GetClassNameW(hwnd, buf, 64)
    return buf.value == "CabinetWClass"                    # a File Explorer window, not the desktop or taskbar


def post_close(hwnd) -> None:
    import ctypes
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE: as if its X was clicked


def bring_to_front(hwnd) -> None:
    import ctypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)                         # SW_RESTORE
    fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    me = kernel32.GetCurrentThreadId()
    joined = fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    if joined:
        user32.AttachThreadInput(me, fg_thread, False)


def type_unicode(text: str) -> None:
    """Type text into the window in front, letter by letter (new lines are left out: no Enter)."""
    import ctypes
    from ctypes import wintypes

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    for ch in text.replace("\r", "").replace("\n", " "):
        for flags in (0x0004, 0x0004 | 0x0002):            # KEYEVENTF_UNICODE, then key up
            inp = INPUT(type=1)
            inp.ki = KEYBDINPUT(0, ord(ch), flags, 0, 0)
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.004)


def power_and_memory() -> dict:
    import ctypes
    from ctypes import wintypes
    out = {}

    class SPS(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte),
                    ("sys", ctypes.c_ubyte), ("life", wintypes.DWORD), ("full", wintypes.DWORD)]

    s = SPS()
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        out["plugged_in"] = s.ac == 1
        if s.pct != 255:
            out["battery_percent"] = s.pct

    class MS(ctypes.Structure):
        _fields_ = [("len", wintypes.DWORD), ("load", wintypes.DWORD), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                    ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]

    m = MS(len=ctypes.sizeof(MS))
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        out["memory_used_percent"] = int(m.load)
    return out


def screenshot_blocks() -> list:
    """All screens as one JPEG for Claude (at most 1600 px wide)."""
    from PIL import ImageGrab
    img = ImageGrab.grab(all_screens=True).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(buf.getvalue()).decode("ascii")}},
            {"type": "text", "text": f"The screens at {datetime.now():%H:%M}."}]


def is_yes(text: str) -> bool:
    return bool(YES.match(text or "")) and not re.search(r"\b(no|not|don't|dont|cancel|stop|wait)\b", text.lower())
