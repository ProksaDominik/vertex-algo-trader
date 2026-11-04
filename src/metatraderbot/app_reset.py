"""Account reset for the Vertex app.

Wipes a book back to a clean slate: closes its live positions, archives its
ledger and state, and writes a fresh state file.

This is the most destructive action in the product, so it is built to be
survivable when it goes wrong:

* **Everything is archived, never deleted.** State and trade ledger are copied
  into data/archive/<book>_<timestamp>/ before anything is touched, so a reset
  fired by mistake can be reversed by copying two files back.
* **Positions are closed FIRST, and a failure aborts the reset.** Resetting the
  state while positions are still open would orphan them -- the runner would
  forget it owns them while the broker still does. A close that does not
  actually close therefore has to stop the whole operation, which is why this
  checks the broker afterwards instead of trusting the return value: a close
  attempted while the market is shut returns retcode=0/"Done" and does nothing
  (see MT5Engine.close_position).
* **It reports what it did, per step.** No silent partial success.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ResetReport:
    ok: bool = False
    archive_dir: str | None = None
    positions_found: int = 0
    positions_closed: int = 0
    realized: float = 0.0
    files_archived: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    error: str | None = None


def reset_book(root: Path, book: str, magics: range | list[int],
               engine, anchor: float = 50_000.0,
               close_positions: bool = True) -> ResetReport:
    """Reset one book. `engine` is a connected MT5Engine (None to skip closing)."""
    rep = ResetReport()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    state_p = root / "data" / f"state_{book}.json"
    trades_p = root / "data" / f"{book}_trades.jsonl"
    arch = root / "data" / "archive" / f"{book}_{ts}"

    try:
        # 1. Close positions before touching state, and VERIFY against the
        #    broker -- a reported close is not a completed one.
        if close_positions and engine is not None:
            import MetaTrader5 as mt5
            live = [p for p in (mt5.positions_get() or []) if p.magic in magics]
            rep.positions_found = len(live)
            if live:
                for p in live:
                    if engine.close_position(p.ticket):
                        rep.positions_closed += 1
                        rep.realized += float(p.profit)
                still = [q for q in (mt5.positions_get() or []) if q.magic in magics]
                if still:
                    rep.error = (
                        f"{len(still)} of {len(live)} positions are still open after the "
                        f"close attempt — the market is most likely shut. Reset ABORTED so "
                        f"they are not orphaned; retry once the market is open."
                    )
                    rep.steps.append(f"close positions: FAILED ({len(still)} remain)")
                    return rep
                rep.steps.append(f"closed {rep.positions_closed} positions "
                                 f"(realised ${rep.realized:+,.2f})")
            else:
                rep.steps.append("no open positions to close")
        else:
            rep.steps.append("position closing skipped by request")

        # 2. Archive, never delete.
        arch.mkdir(parents=True, exist_ok=True)
        rep.archive_dir = str(arch)
        for p in (state_p, trades_p):
            if p.exists():
                shutil.copy2(p, arch / p.name)
                rep.files_archived.append(p.name)
        rep.steps.append(f"archived {len(rep.files_archived)} file(s) -> {arch.name}")

        # 3. Fresh state.
        fresh = {
            "peak_equity": anchor, "realized_usd": 0.0, "current_day": None,
            "day_start_equity": anchor, "halted": False, "halt_reason": None,
            "total_trades": 0, "total_wins": 0, "total_losses": 0,
            "payout_ready": False,
            "started_ts": datetime.now(timezone.utc).isoformat(),
            "reset_from_archive": arch.name,
        }
        state_p.parent.mkdir(parents=True, exist_ok=True)
        state_p.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
        rep.steps.append("wrote fresh state file")

        # 4. Clear the ledger (the archived copy above is the record).
        if trades_p.exists():
            trades_p.write_text("", encoding="utf-8")
            rep.steps.append("cleared trade ledger")

        rep.ok = True
        return rep
    except Exception as e:  # noqa: BLE001 - surface any failure to the operator
        rep.error = f"{type(e).__name__}: {e}"
        return rep


def list_archives(root: Path, book: str | None = None) -> list[dict]:
    """Available restore points, newest first."""
    base = root / "data" / "archive"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if book and not d.name.startswith(book):
            continue
        out.append({"name": d.name, "path": str(d),
                    "files": [f.name for f in d.iterdir() if f.is_file()],
                    "when": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    return out


def restore_archive(root: Path, archive_name: str) -> tuple[bool, str]:
    """Copy an archived state/ledger back over the live files."""
    src = root / "data" / "archive" / archive_name
    if not src.is_dir():
        return False, f"Archive not found: {archive_name}"
    n = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, root / "data" / f.name)
            n += 1
    return True, f"Restored {n} file(s) from {archive_name}. Restart the bot to load it."
