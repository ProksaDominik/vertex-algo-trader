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


def alert(message: str, *, book: str = "", html: bool = False) -> None:
    """html=True lets `message` use Telegram's HTML subset (<b>, <i>, <code>)
    for emphasis -- Telegram has no actual text-color support, so bold +
    emoji is the real way to make an alert visually pop. Defaults to False
    (plain text, original behaviour) so every existing caller is unaffected
    -- only opt in a message you've checked doesn't contain a bare '<' or
    '&' outside of a tag you intended, since Telegram's HTML parser rejects
    malformed markup outright rather than degrading gracefully."""
    tag = f"<b>[{book}]</b> " if (book and html) else (f"[{book}] " if book else "")
    log.warning(f"ALERT {tag}{message}")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    def _send() -> None:
        try:
            payload: dict = {"chat_id": chat_id, "text": f"{tag}{message}"}
            if html:
                payload["parse_mode"] = "HTML"
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # noqa: BLE001 — alerting must never crash the bot
            log.warning(f"Telegram alert failed: {e}")

    # Fire-and-forget: never block the trading loop on a network call.
    threading.Thread(target=_send, daemon=True).start()
