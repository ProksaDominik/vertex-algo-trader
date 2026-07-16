"""
Tiny local JSON API feeding the dashboard's live JS chart (lightweight-charts).

Runs as its own process on localhost:8765. Exists because Streamlit itself
can't update a chart smoothly — every rerun resends the whole component and
the browser repaints it, which is what caused the flashing. A chart that
updates itself via JavaScript on its own timer, independent of Streamlit's
rerun cycle, needs something to fetch fresh candles from — this is that.

Usage:
    python scripts/chart_api.py
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import MetaTrader5 as mt5  # noqa: E402
from metatraderbot.mt5_engine import TIMEFRAME_MAP  # noqa: E402

PORT = 8765


def get_bars(symbol: str, timeframe: str, count: int) -> list[dict]:
    tf_attr = TIMEFRAME_MAP.get(timeframe)
    if tf_attr is None:
        return []
    tf = getattr(mt5, tf_attr)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None:
        return []
    return [
        {"time": int(r["time"]), "open": float(r["open"]), "high": float(r["high"]),
         "low": float(r["low"]), "close": float(r["close"])}
        for r in rates
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # keep the console quiet — this gets hit every ~1s per open chart

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/chart":
            self._send_json({"error": "not found"}, 404)
            return

        qs = parse_qs(parsed.query)
        symbol = qs.get("symbol", [""])[0]
        timeframe = qs.get("timeframe", ["1m"])[0]
        ticket = qs.get("ticket", [""])[0]
        count = int(qs.get("count", ["150"])[0])

        if not mt5.initialize():
            self._send_json({"error": f"MT5 not reachable: {mt5.last_error()}"}, 503)
            return

        bars = get_bars(symbol, timeframe, count)

        position = None
        if ticket:
            positions = mt5.positions_get(ticket=int(ticket))
            if positions:
                p = positions[0]
                opened_time = int(p.time)
                # Markers in lightweight-charts must land exactly on a bar's
                # time value or they're silently dropped — a position almost
                # never opens exactly on a candle boundary, so snap to the
                # bar it actually opened within (last bar at/before open time).
                entry_bar_time = opened_time
                if bars:
                    on_or_before = [b["time"] for b in bars if b["time"] <= opened_time]
                    entry_bar_time = on_or_before[-1] if on_or_before else bars[0]["time"]
                position = {
                    "entry": p.price_open, "sl": p.sl, "tp": p.tp,
                    "current": p.price_current, "is_long": p.type == 0,
                    "profit": p.profit, "opened_time": opened_time,
                    "entry_bar_time": entry_bar_time,
                }

        self._send_json({"bars": bars, "position": position})


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[OK] Chart API listening on http://127.0.0.1:{PORT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
