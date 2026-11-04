"""Vertex stats service (owner, 2026-10-09: a voice butler "like Jarvis" on both laptops). Runs next to the
bots and answers the laptops' Vertex app over Tailscale. Nothing about the bots can be changed through it.

    GET  /summary                       everything Vertex can tell (src/metatraderbot/vertex_stats.py)
    GET  /events?since=<iso>            things worth announcing since a time (races, evaluations, payouts, ...)
    GET  /health                        "ok"
    POST /vertex/event                  a laptop's Vertex reports an answer / announcement / its state; the
                                        dashboard shows them (data/vertex_log.jsonl, data/vertex_state.json)
    GET  /vertex/inbox?viewer=&after=   typed questions and switches the dashboard left for that laptop

Port 8502 (VERTEX_API_PORT). With VERTEX_API_TOKEN set in .env, every request must send the header
X-Vertex-Token: <token>. On the server the firewall lets only Tailscale addresses reach it (Setup-Server.ps1).

Usage:  .venv/Scripts/python.exe scripts/vertex_api.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from metatraderbot import vertex_stats  # noqa: E402

DATA = ROOT / "data"
PORT = int(os.environ.get("VERTEX_API_PORT", "8502"))
log = logging.getLogger("vertex_api")


class Handler(BaseHTTPRequestHandler):
    server_version = "VertexAPI/1"

    def _send(self, code: int, body: object) -> None:
        raw = json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 -- http.server's name
        token = os.environ.get("VERTEX_API_TOKEN", "").strip()
        if token and self.headers.get("X-Vertex-Token", "") != token:
            self._send(401, {"error": "wrong or missing X-Vertex-Token"})
            return
        url = urlparse(self.path)
        try:
            if url.path == "/health":
                self._send(200, {"ok": True})
            elif url.path == "/summary":
                self._send(200, vertex_stats.summary(DATA))
            elif url.path == "/vertex/inbox":
                q = parse_qs(url.query)
                viewer = q.get("viewer", [""])[0]
                self._send(200, {"questions": vertex_stats.vertex_inbox(DATA, viewer, q.get("after", [""])[0]),
                                 "screen": vertex_stats.vertex_screen(DATA, viewer)})
            elif url.path == "/events":
                q = parse_qs(url.query)
                try:
                    since = datetime.fromisoformat(q.get("since", [""])[0].strip().replace(" ", "+"))  # "+" arrives as a space
                    if since.tzinfo is None:
                        since = since.replace(tzinfo=timezone.utc)
                except ValueError:
                    since = datetime.now(timezone.utc) - timedelta(hours=1)
                self._send(200, {"events": vertex_stats.events(DATA, since),
                                 "now": datetime.now(timezone.utc).isoformat()})
            else:
                self._send(404, {"error": "unknown path", "paths": ["/summary", "/events?since=", "/health", "/vertex/inbox?viewer=&after="]})
        except Exception as e:  # noqa: BLE001 -- the service must keep answering
            log.exception("request failed")
            self._send(500, {"error": repr(e)})

    def do_POST(self) -> None:  # noqa: N802
        """The laptops' Vertex reports what it does (shown in the dashboard; owner 2026-10-09: "keep it all in
        that dashboard"): POST /vertex/event {viewer, kind: exchange|announce|state, ...}. Only these files are
        written -- nothing about the bots can be changed here."""
        token = os.environ.get("VERTEX_API_TOKEN", "").strip()
        if token and self.headers.get("X-Vertex-Token", "") != token:
            self._send(401, {"error": "wrong or missing X-Vertex-Token"})
            return
        if urlparse(self.path).path != "/vertex/event":
            self._send(404, {"error": "unknown path"})
            return
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 200_000)
            ev = json.loads(self.rfile.read(n).decode("utf-8"))
            vertex_stats.record_vertex_event(DATA, ev)
            self._send(200, {"ok": True})
        except (ValueError, OSError) as e:
            self._send(400, {"error": repr(e)})

    def log_message(self, fmt, *args) -> None:  # quiet: one line per request in the log file only
        log.info("%s %s", self.address_string(), fmt % args)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / "vertex_api.log", encoding="utf-8")])
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log.warning(f"Vertex stats service on port {PORT}")
    vertex_stats.real_test(DATA)          # the Fleet page's Real test, worked out in the background from the start
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
