"""One-shot: close every position left over from the retired v5 roster.

Magics 9200-9207 belong to the v5 book, retired 2026-08-22 in favour of the v6
one-payout roster (9210-9217). The new runner does not manage them, so they sit
unattended while still consuming the account's $2,000 DAILY loss allowance --
an old position stopping out could halt the new roster through no fault of its
own. This closes them and then exits.

Safe to run repeatedly: it only ever touches 9200-9207 and does nothing if none
are open. Intended to run at market open, since a close attempted while the
market is shut silently does nothing (the broker returns retcode=0/"Done" with
no deal ticket -- see MT5Engine.close_position).
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("close_legacy")
from metatraderbot.mt5_engine import MT5Engine  # noqa: E402
import MetaTrader5 as mt5  # noqa: E402

LEGACY = range(9200, 9208)
MAX_WAIT_MIN = 180   # keep retrying this long after start, then give up

def main() -> int:
    eng = MT5Engine(); eng.connect()
    deadline = time.time() + MAX_WAIT_MIN * 60
    while time.time() < deadline:
        open_pos = [p for p in (mt5.positions_get() or []) if p.magic in LEGACY]
        if not open_pos:
            log.info("No legacy v5 positions remain. Done.")
            return 0
        floating = sum(p.profit for p in open_pos)
        log.info(f"{len(open_pos)} legacy positions open (floating ${floating:+,.2f}) -- closing")
        closed = 0
        for p in open_pos:
            if eng.close_position(p.ticket):
                closed += 1
                log.info(f"  closed {p.ticket} {p.symbol} magic={p.magic} P&L ${p.profit:+,.2f}")
        left = [q for q in (mt5.positions_get() or []) if q.magic in LEGACY]
        log.info(f"closed {closed}, {len(left)} remaining")
        if not left:
            return 0
        # Market probably still shut; wait and retry rather than reporting success.
        time.sleep(60)
    log.error("Gave up: legacy positions still open after the retry window.")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
