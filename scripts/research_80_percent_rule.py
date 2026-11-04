"""The Market Profile '80% rule' (from a futures broker's article, owner, 2026-10-08), checked and traded.

Rule: if New York opens (09:30) OUTSIDE the previous New York session's value area (70% of its volume around
the point of control) and then gets accepted back inside it for two consecutive 30-minute periods, price has
an 80% chance to travel to the OTHER side of the value area.
  Acceptance (*): 'traded inside' -- price traded inside the value area in two consecutive 30-minute periods
                  (the classic two-TPO reading); or 'closed inside' -- both periods closed inside.
Part 1: how often the other side is reached by 16:00 after acceptance (the claimed 80%).
Part 2: the trade -- at the close of the second period, enter toward the far edge; target the far edge;
  stop 1 tick beyond the day's extreme so far (where it came from); out at 16:40 latest. One trade a day,
  acceptance counted until 14:00. Costs: $3.50 round trip (0.175 pt NQ, 0.07 ES) + 0.25 pt spread; 1 tick
  slippage on stops; stop first if a candle reaches both.
Value areas from NinjaTrader bars with real volume (each bar's volume spread evenly over its range).
Data: NQ 5-min 2021-2026 (split 2024-01-01), ES 5-min Dec 2023 - Oct 2026 (split 2025-04-01), NQ 1-min
Mar 2024 - Oct 2026 (split 2025-07-01).

Usage:  .venv/Scripts/python.exe scripts/research_80_percent_rule.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_volume_profile import value_area  # noqa: E402

TICK, SPREAD = 0.25, 0.25
DATA = {
    "NQ 5-min 2021-26": ("NQ_5m_long_nt.parquet", 0.175, "2024-01-01"),
    "ES 5-min 2023-26": ("ES_5m_long_nt.parquet", 0.07, "2025-04-01"),
    "NQ 1-min 2024-26": ("NQ_1m_long_nt.parquet", 0.175, "2025-07-01"),
}


def run(name: str, acceptance: str):
    fname, comm, split = DATA[name]
    df = pd.read_parquet(ROOT / "data" / "futures" / fname)[["open", "high", "low", "close", "volume"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    df = df[~df.index.duplicated()].sort_index()
    m = df.index.hour * 60 + df.index.minute
    rth = df[(m >= 570) & (m < 960)]
    days = sorted(set(rth.index.date))
    stats, trades = [], []
    prev_va = None
    for d in days:
        x = rth[rth.index.date == d]
        if len(x) < 20:
            continue
        o, h, l, c = (x[k].to_numpy() for k in ("open", "high", "low", "close"))
        xm = np.asarray(x.index.hour * 60 + x.index.minute)
        if prev_va is not None:
            val, poc, vah = prev_va
            op = o[0]
            side = -1 if op > vah else (1 if op < val else 0)        # trade direction after re-entry
            if side != 0:
                period = (xm - 570) // 30
                accepted_at = None
                prev_in = False
                for p in range(int(period.max()) + 1):
                    sel = period == p
                    if not sel.any() or 570 + 30 * p >= 840:
                        break
                    if acceptance == "traded inside":
                        inside = (l[sel].min() <= vah) if side < 0 else (h[sel].max() >= val)
                    else:
                        inside = val <= c[sel][-1] <= vah
                    if inside and prev_in:
                        accepted_at = np.flatnonzero(sel)[-1]           # last bar of the second period
                        break
                    prev_in = inside
                if accepted_at is not None:
                    far = val if side < 0 else vah
                    rest = slice(accepted_at + 1, len(c))
                    reached = ((l[rest] <= far).any() if side < 0 else (h[rest] >= far).any())
                    stats.append((pd.Timestamp(d), reached))
                    ent = c[accepted_at]
                    ext = h[:accepted_at + 1].max() if side < 0 else l[:accepted_at + 1].min()
                    stp = ext + TICK if side < 0 else ext - TICK
                    risk, rew = side * (ent - stp), side * (far - ent)
                    if risk >= 1.0 and rew > 0:
                        res = None
                        for k in range(accepted_at + 1, len(c)):
                            if (h[k] >= stp) if side < 0 else (l[k] <= stp):
                                res = (-risk - TICK, "stop")
                                break
                            if (l[k] <= far) if side < 0 else (h[k] >= far):
                                res = (rew, "far edge")
                                break
                        if res is None:
                            res = (side * (c[-1] - ent), "16:00 close")
                        trades.append((pd.Timestamp(d), (res[0] - comm - SPREAD) / risk, rew / risk, res[1]))
        prev_va = value_area(l, h, x["volume"].to_numpy(dtype=float)) if x["volume"].sum() > 0 else None
    return pd.DataFrame(stats, columns=["day", "reached"]), pd.DataFrame(trades, columns=["day", "R", "planned_rr", "how"]), split


def main() -> int:
    for name in DATA:
        for acc in ("traded inside", "closed inside"):
            st, tr, split = run(name, acc)
            sp = pd.Timestamp(split)
            a, b = tr[tr.day < sp], tr[tr.day >= sp]
            z = lambda r: r.mean() / (r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 2 else float("nan")  # noqa: E731
            print(f"{name}, acceptance '{acc}': {len(st)} days qualified, the far side reached by 16:00 on "
                  f"{st['reached'].mean():.0%} (claim: 80%)")
            print(f"   trade: {len(tr)} trades, planned reward/risk median {tr.planned_rr.median():.2f}, winning "
                  f"{(tr.R > 0).mean():.0%}, average {tr.R.mean():+.3f} R | first part {a.R.mean():+.3f} R (n {len(a)}) "
                  f"-> later {b.R.mean():+.3f} R (n {len(b)}, z {z(b.R):+.2f}); exits {tr.how.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
