"""Persistent bot state — survives crashes, restarts, and VPS reboots.

One JSON file per book (data/state_<book>.json), written atomically
(tmp + os.replace) so a crash mid-write can never corrupt it. Everything
the runner must NOT lose across a restart lives here:

- initial_balance_anchor: the account balance captured ONCE on first run —
  the fixed reference for the overall drawdown guard. Before this existed,
  the anchor was re-captured at every process start, so each restart
  silently reset the total-DD limit (AUDIT C2).
- daily anchor: CE(S)T day key + realized-P&L / equity snapshot at rollover.
- halt: guard-trip state (daily halts auto-clear next day; total halts
  require manual reset).
- equity_high_water: book-level HWM for the YOLO circuit breaker.
- quarantine: slot-health bench state.
- pending_signals: NoWick-style pending setups (in-memory only before —
  AUDIT M8).
- broker_offset_s: last KNOWN broker-clock offset, used when the live tick
  is stale (weekends) instead of computing garbage (AUDIT M4).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULTS: dict[str, Any] = {
    "version": 1,
    "initial_balance_anchor": None,
    "anchor_captured_ts": None,
    "daily": {},          # {"day": "2026-07-21", "realized_at_start": 0.0, "equity_at_start": null}
    "halt": {},           # {"active": true, "reason": "...", "scope": "daily"|"total"|"breaker", "day": "..."}
    "equity_high_water": None,
    "quarantine": {},     # {slot_name: iso_until}
    "pending_signals": {},  # {slot_name: [ {...PendingSignal fields...} ]}
    "broker_offset_s": 0,
    "target_alerted": False,
}


class BotState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, Any] = json.loads(json.dumps(_DEFAULTS))
        self._load()

    def _load(self) -> None:
        try:
            on_disk = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(on_disk, dict):
                self.data.update(on_disk)
                log.info(f"State restored from {self.path.name}")
        except FileNotFoundError:
            log.info(f"No prior state at {self.path.name} — starting fresh")
        except (OSError, ValueError) as e:
            # A corrupt state file must not brick the bot; keep the corrupt
            # copy for forensics and start from defaults.
            log.error(f"State file unreadable ({e}) — preserving as .corrupt and starting fresh")
            try:
                self.path.replace(self.path.with_suffix(".json.corrupt"))
            except OSError:
                pass

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            log.error(f"State save failed: {e}")

    # Convenience accessors -------------------------------------------------

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value

    @property
    def halted(self) -> bool:
        return bool((self.data.get("halt") or {}).get("active"))

    def set_halt(self, *, reason: str, scope: str, day: str | None = None) -> None:
        self.data["halt"] = {"active": True, "reason": reason, "scope": scope, "day": day}
        self.save()

    def clear_halt(self) -> None:
        self.data["halt"] = {}
        self.save()
