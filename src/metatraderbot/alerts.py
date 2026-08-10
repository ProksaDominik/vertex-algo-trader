"""Operational alerts — log always, push to Telegram when configured.

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable pushes.
Uses stdlib urllib only (no new dependency); failures are swallowed after
logging — an alert channel outage must never take the bot down with it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request

log = logging.getLogger(__name__)


def alert(message: str, *, book: str = "") -> None:
    tag = f"[{book}] " if book else ""
    log.warning(f"ALERT {tag}{message}")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    def _send() -> None:
        try:
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=json.dumps({"chat_id": chat_id, "text": f"{tag}{message}"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # noqa: BLE001 — alerting must never crash the bot
            log.warning(f"Telegram alert failed: {e}")

    # Fire-and-forget: never block the trading loop on a network call.
    threading.Thread(target=_send, daemon=True).start()
