"""K9 (CLAUDE.md kill switches §11.1): "server time drift > 5s from NTP ->
HALT". A minimal, stdlib-only SNTP client -- no new pip dependency for one
UDP round-trip. Fails OPEN (returns None) on any network problem: an
unreachable NTP server says nothing about whether THIS machine's clock is
wrong, so it must never be treated as "drift detected".
"""
from __future__ import annotations

import socket
import struct
import time

_NTP_EPOCH_OFFSET = 2_208_988_800  # seconds between 1900-01-01 and 1970-01-01
_NTP_PORT = 123


def ntp_offset_seconds(server: str = "pool.ntp.org", timeout: float = 2.0) -> float | None:
    """Local clock minus true NTP time, in seconds. Positive = local clock
    is AHEAD of true time. None if the server couldn't be reached in time."""
    try:
        packet = b"\x1b" + 47 * b"\0"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(packet, (server, _NTP_PORT))
            local_recv = time.time()
            data, _ = sock.recvfrom(48)
        if len(data) < 48:
            return None
        transmit_ts = struct.unpack("!12I", data)[10]
        # A zero (or absurdly small) transmit timestamp means the reply was
        # never filled in -- a Kiss-o'-Death packet, a captive-portal reply, or
        # a truncated/dropped response. Arithmetic on it produces a ~127-year
        # "drift" (0 - _NTP_EPOCH_OFFSET flips sign), which K9 then reads as
        # real and HALTS the book. Seen live: the FTMO bot sat halted on
        # "K9 clock drift +3994752758.8s" and never placed a trade.
        # This module's contract is to fail OPEN, so an unusable reply must
        # return None rather than a number.
        if transmit_ts < _NTP_EPOCH_OFFSET:
            return None
        ntp_now = transmit_ts - _NTP_EPOCH_OFFSET
        offset = local_recv - ntp_now
        # Any genuine desync worth halting for is seconds-to-minutes. A value
        # beyond a day means the packet is garbage, not that the clock is a
        # year out -- again, fail open rather than halt on nonsense.
        if abs(offset) > 86_400:
            return None
        return offset
    except (OSError, struct.error):
        return None
