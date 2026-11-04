"""Vertex plays music on the owner's Spotify (owner, 2026-10-16: "I have Spotify Premium, it's open right now").

Spotify's Web API starts any song, artist, album, mood playlist, the owner's own playlists or liked songs on the
Spotify app of this PC (Premium needed). Connecting is done once: the owner registers a small app at
developer.spotify.com (its Client ID goes into settings.env as SPOTIFY_CLIENT_ID) and agrees in the browser
(`vertex_voice.py --spotify`); the key Spotify gives back is kept in spotify_token.json next to Vertex (not in
OneDrive) and renewed by itself. No password is ever seen by Vertex (sign-in with PKCE, no client secret).
"""
from __future__ import annotations

import base64
import difflib
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("vertex")
REDIRECT = "http://127.0.0.1:8520/callback"
SCOPES = ("user-read-playback-state user-modify-playback-state user-read-currently-playing playlist-read-private "
          "playlist-read-collaborative user-library-read")
ACCOUNTS = "https://accounts.spotify.com"
API = "https://api.spotify.com/v1"
KINDS = {0: "song", 1: "mood", 2: "artist", 3: "album", 4: "own playlist", 5: "liked songs"}


class SpotifyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge) for the sign-in without a client secret (RFC 7636, S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    return verifier, challenge_for(verifier)


def challenge_for(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


def _post_form(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode("ascii"),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SpotifyError(e.code, e.read().decode("utf-8", "replace")[:300]) from e


def connect(client_id: str, token_path: Path, open_browser=None, timeout: float = 300) -> str:
    """Sign in once: opens Spotify's "Agree" page in the browser and waits for the answer on 127.0.0.1:8520.
    Returns "" when connected, else what went wrong."""
    import http.server
    import webbrowser
    if not client_id:
        return "no SPOTIFY_CLIENT_ID in settings.env"
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if not self.path.startswith("/callback"):
                self.send_response(404)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got and got.get("state") == state
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = ("Vertex is now connected to your Spotify. You can close this tab." if ok else
                   "Spotify did not connect: " + got.get("error", "unexpected answer") + ". You can close this tab.")
            self.wfile.write(f"<html><body style='font:20px sans-serif;background:#111;color:#eee;padding:40px'>"
                             f"<b>Vertex</b><br><br>{msg}</body></html>".encode("utf-8"))

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 8520), Handler)
    server.timeout = 1.0
    url = ACCOUNTS + "/authorize?" + urllib.parse.urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": REDIRECT, "scope": SCOPES,
        "code_challenge_method": "S256", "code_challenge": challenge, "state": state})
    (open_browser or webbrowser.open)(url)
    end = time.monotonic() + timeout
    try:
        while not got and time.monotonic() < end:
            server.handle_request()
    finally:
        server.server_close()
    if not got:
        return "no answer from Spotify within 5 minutes"
    if got.get("state") != state or "code" not in got:
        return "Spotify said: " + got.get("error", "unexpected answer")
    tok = _post_form(ACCOUNTS + "/api/token", {"grant_type": "authorization_code", "code": got["code"],
                                               "redirect_uri": REDIRECT, "client_id": client_id,
                                               "code_verifier": verifier})
    save_token(token_path, tok)
    return ""


def save_token(path: Path, tok: dict, old: dict | None = None) -> dict:
    out = {"access_token": tok["access_token"],
           "refresh_token": tok.get("refresh_token") or (old or {}).get("refresh_token", ""),
           "expires_at": time.time() + float(tok.get("expires_in", 3600)), "scope": tok.get("scope", "")}
    Path(path).write_text(json.dumps(out), encoding="utf-8")
    return out


class Spotify:
    """The owner's Spotify: find what to play, and play it on this PC's Spotify app."""

    def __init__(self, client_id: str, token_path: Path, http=None):
        self.client_id = (client_id or "").strip()
        self.token_path = Path(token_path)
        self._http = http                                   # tests: a fake (method, url, body) -> (status, dict)
        self._sleep = time.sleep
        try:
            self.tok = json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.tok = {}

    def ready(self) -> bool:
        return bool(self.client_id and self.tok.get("refresh_token"))

    # -- the API --
    def _access(self) -> str:
        if not self.tok.get("access_token") or time.time() > float(self.tok.get("expires_at", 0)) - 60:
            new = _post_form(ACCOUNTS + "/api/token", {"grant_type": "refresh_token", "client_id": self.client_id,
                                                       "refresh_token": self.tok.get("refresh_token", "")})
            self.tok = save_token(self.token_path, new, self.tok)   # Spotify may hand out a new refresh key
        return self.tok["access_token"]

    def call(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        if self._http:
            status, data = self._http(method, url, body)
            if status >= 400:
                raise SpotifyError(status, json.dumps(data))
            return data
        for attempt in range(2):
            req = urllib.request.Request(url, method=method, data=json.dumps(body).encode("utf-8") if body is not None
                                         else (b"" if method in ("PUT", "POST") else None),
                                         headers={"Authorization": "Bearer " + self._access(),
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    raw = r.read()
                try:
                    return json.loads(raw.decode("utf-8")) if raw.strip() else {}
                except ValueError:                           # some commands answer with plain text
                    return {}
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:          # the key ran out early: renew once
                    self.tok["expires_at"] = 0
                    continue
                msg = e.read().decode("utf-8", "replace")
                try:
                    msg = json.loads(msg)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    pass
                raise SpotifyError(e.code, str(msg)[:200]) from e
        return {}

    # -- what to play --
    def find(self, query: str, kind: int) -> dict | None:
        """{"name", "context_uri" (+ "offset_uri": the song to start at), "shuffle"}; None when nothing fits.
        kind: 0 a song, 1 a mood or genre (a playlist), 2 an artist, 3 an album, 4 one of the owner's playlists,
        5 the owner's liked songs."""
        q = (query or "").strip()
        if kind == 5:                                       # Liked Songs, as a whole (it played, 2026-10-16)
            uid = self.call("GET", "/me").get("id")
            return {"name": "your liked songs", "context_uri": f"spotify:user:{uid}:collection", "shuffle": True}                 if uid else None
        if kind == 4:
            mine = self.my_playlists()
            names = [p["name"] for p in mine]
            hit = _best(q, names)
            if hit is None:
                return None
            p = next(p for p in mine if p["name"] == hit)
            return {"name": p["name"], "context_uri": p["uri"], "shuffle": True}
        types = {0: "track", 1: "playlist", 2: "artist", 3: "album"}.get(kind, "track")
        res = self.call("GET", "/search", {"q": q, "type": types, "limit": 10})
        items = [x for x in (res.get(types + "s") or {}).get("items") or [] if x]
        if not items:
            return None
        if kind == 1:                                       # a mood: a real playlist, not a 3-song one
            good = [p for p in items if ((p.get("tracks") or {}).get("total") or 0) >= 15] or items
            p = good[0]
            return {"name": p.get("name", ""), "context_uri": p["uri"], "shuffle": True}
        if kind == 0:                                       # its album, started at the song: Spotify accepts a
            t = items[0]                                    # list of songs ("uris") but plays nothing (2026-10-16)
            artists = ", ".join(a.get("name", "") for a in t.get("artists") or [])
            return {"name": f"{t.get('name', '')} by {artists}".strip(),
                    "context_uri": (t.get("album") or {}).get("uri", ""), "offset_uri": t["uri"]}
        x = items[0]
        return {"name": x.get("name", ""), "context_uri": x["uri"], "shuffle": kind == 2}

    def follow(self, first: dict, after: dict, device_id: str | None = None, give_up_s: float = 1200) -> None:
        """When `first` (a song) ends, start `after` (a mood playlist, artist or album) -- a romantic evening that
        opens with Careless Whisper and carries on with love songs (owner, 2026-10-16). Spotify lets this app play
        other people's playlists but not read their songs (403), so it can't be one list. If the owner plays
        something else meanwhile, it leaves their choice alone. Runs in the background."""
        song = first.get("offset_uri", "")
        if not song or not after.get("context_uri"):
            return
        threading.Thread(target=self._follow, args=(song, after, device_id, give_up_s), daemon=True).start()

    def _follow(self, song: str, after: dict, device_id: str | None, give_up_s: float) -> bool:
        end = time.monotonic() + give_up_s
        while time.monotonic() < end:
            try:
                now = self.call("GET", "/me/player/currently-playing") or {}
            except SpotifyError as e:
                log.info("Spotify follow: %s", e)
                now = {}
            item = now.get("item") or {}
            if now and item.get("uri") != song:            # the owner chose something else: leave it
                return False
            left = (item.get("duration_ms") or 0) - (now.get("progress_ms") or 0)
            if item and now.get("is_playing") and 0 < left <= 4000:
                self._sleep(max(0.0, left / 1000 - 0.3))
                body = {"context_uri": after["context_uri"]}
                self.call("PUT", "/me/player/play", {"device_id": device_id} if device_id else None, body)
                if after.get("shuffle"):
                    try:
                        self.call("PUT", "/me/player/shuffle", {"state": "true"})
                    except SpotifyError:
                        pass
                log.info("Spotify: %s after the song", after.get("name"))
                return True
            self._sleep(1.0 if 0 < left <= 15000 else 5.0)
        return False

    def my_playlists(self) -> list[dict]:
        items = self.call("GET", "/me/playlists", {"limit": 50}).get("items") or []
        return [{"name": p.get("name", ""), "uri": p["uri"]} for p in items if p and p.get("uri")]

    # -- playing it --
    def device(self, wait: float = 12.0) -> str | None:
        """This PC's Spotify app (started when it is closed); None when it doesn't show up."""
        me = os.environ.get("COMPUTERNAME", "").lower()
        end = time.monotonic() + wait
        started = False
        while True:
            devs = self.call("GET", "/me/player/devices").get("devices") or []
            pcs = [d for d in devs if (d.get("type") or "").lower() == "computer"]
            pick = (next((d for d in pcs if (d.get("name") or "").lower() == me), None)
                    or next((d for d in devs if d.get("is_active")), None) or (pcs[0] if pcs else None))
            if pick:
                return pick["id"]
            if not started:
                try:
                    os.startfile("spotify:")                # noqa: S606 -- the owner's Spotify app
                except OSError:
                    return None
                started = True
            if time.monotonic() > end:
                return None
            time.sleep(1.0)

    def play(self, pick: dict) -> str:
        """Play what find() chose on this PC; "" when it plays, else a short reason."""
        dev = self.device()
        if not dev:
            return "Spotify didn't open on this computer"
        body = {"context_uri": pick["context_uri"]}
        if pick.get("offset_uri"):
            body["offset"] = {"uri": pick["offset_uri"]}      # the chosen song first
        try:
            self.call("PUT", "/me/player/play", {"device_id": dev}, body)
        except SpotifyError as e:
            if e.status == 404:                              # no active player yet: wake this one, then play
                self.call("PUT", "/me/player", None, {"device_ids": [dev], "play": False})
                time.sleep(0.5)
                self.call("PUT", "/me/player/play", {"device_id": dev}, body)
            elif e.status == 403:
                return "Spotify refused to play it (it needs Premium)"
            else:
                raise
        if pick.get("shuffle") is not None:
            try:
                self.call("PUT", "/me/player/shuffle", {"state": "true" if pick["shuffle"] else "false",
                                                         "device_id": dev})
            except SpotifyError:
                pass
        return ""


def _best(want: str, names: list[str]) -> str | None:
    """The owner's playlist a spoken name means ("my gym playlist" -> "GYM 2026")."""
    w = re.sub(r"\b(my|the|playlist|please|songs?)\b", " ", want.lower())
    w = re.sub(r"\s+", " ", w).strip()
    if not w or not names:
        return None
    low = {n.lower(): n for n in names}
    if w in low:
        return low[w]
    hits = [n for n in names if w in n.lower()]
    if hits:
        return min(hits, key=len)
    close = difflib.get_close_matches(w, list(low), n=1, cutoff=0.5)
    return low[close[0]] if close else None
