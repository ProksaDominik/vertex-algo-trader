"""Close the retired v6 gold positions at the next possible moment.

WHY THIS EXISTS
---------------
Four XAUUSD positions opened by the retired v6 roster (magics 9211 and 9215)
were sized off a broken multiplier: this broker reports trade_tick_value=0.1 for
XAUUSD when live P&L proves the true figure is 100 USD per price unit per lot,
so each position carries ~4,722 USD of risk where 500 was intended -- about
19,188 USD in total against a 2,000 USD daily loss limit.

The user asked for them closed. The attempt on Friday 2026-08-28 20:31 UTC was
rejected with retcode 10018 (market closed): gold had just gone into the weekly
close, while still reporting trade_mode=4 and streaming ticks. A non-disabled
trade_mode does not mean tradeable -- only a fill does.

So this retries until the positions are actually gone, then exits. It is
deliberately a separate process rather than logic inside run_lucidpro_funded.py:
this is a one-off cleanup of a retired roster, and threading a special case
through the live runner's entry path to do it once would be a worse trade.

Verification is by RE-READING POSITIONS, never by trusting a return code. That
is the lesson of the earlier "closed 21/21" incident, where every close reported
retcode=0/"Done" with no deal ticket and all 21 positions were still open.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import MetaTrader5 as mt5                              # noqa: E402
from metatraderbot.mt5_engine import MT5Engine         # noqa: E402
from metatraderbot.alerts import alert                 # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "close_legacy_gold.log",
                                  encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("close_legacy")

TARGET_MAGICS = {9211, 9215}
POLL_S = 60
MAX_HOURS = 96          # give up after four days rather than poll forever


def remaining():
    return [p for p in (mt5.positions_get() or []) if p.magic in TARGET_MAGICS]


def main() -> int:
    eng = MT5Engine(symbol_map={"NAS100": "USTEC"})
    if not eng.connect():
        log.error("MT5 connect failed")
        return 1
    start = time.time()
    attempts = 0
    log.warning("watching %d legacy gold positions (magics %s)",
                len(remaining()), sorted(TARGET_MAGICS))

    while True:
        pos = remaining()
        if not pos:
            log.warning("ALL LEGACY GOLD POSITIONS CLOSED after %d attempts", attempts)
            try:
                alert("Legacy v6 gold positions closed (oversized by the XAUUSD "
                      "multiplier bug). v7 can now open positions.")
            except Exception:
                pass
            return 0
        if (time.time() - start) / 3600 > MAX_HOURS:
            log.error("giving up after %.0f hours; %d still open", MAX_HOURS, len(pos))
            return 2

        attempts += 1
        floating = sum(p.profit for p in pos)
        closed_now = 0
        for p in pos:
            res = eng.close_position(p.ticket)
            if getattr(res, "ok", False):
                closed_now += 1
                log.warning("closed ticket %d magic %d %.2f lots at %+.2f",
                            p.ticket, p.magic, p.volume, p.profit)
            time.sleep(0.3)

        # Trust the book, not the return codes.
        left = remaining()
        if closed_now and not left:
            log.warning("verified: nothing left on those magics")
        elif attempts % 10 == 1:
            log.info("attempt %d: %d still open, floating %+.2f (market likely shut)",
                     attempts, len(left), floating)
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
