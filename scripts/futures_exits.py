"""Exit variants for the ORB + trend-filter legs (MNQ, MGC).

Entries stay exactly as deployed (45-min range, entries until 12:00 ET, first
breakout decides, trend filter). Only the exit changes:

  target      1R, 1.5R, 2R, 3R, or none (time exit)
  breakeven   move the stop to entry once price has gone +1R (or never)
  time exit   13:00, 14:00 or 15:45 ET bar (the 15:45 bar closes at 16:00)

Scored like everything else in this research: futures costs, a bar spanning
stop and target counts as the stop, 10.5 years, both halves, share of positive
years, plus a walk-forward pick (best exit on the years before, traded next
year) so an exit is only adopted if choosing it in advance would have paid.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402

M = fh.M
MARKETS = {"MNQ": "nas100", "MGC": "xauusd"}
TARGETS = (1.0, 1.5, 2.0, 3.0, None)
BREAKEVEN = (None, 1.0)
TIME_EXIT = (M(13, 0), M(14, 0), M(15, 45))


def walk_be(x, start, side, stop, target, last_exit, cost, entry, be_r):
    """Like fh.walk, plus a breakeven move once the trade has run be_r x risk."""
    risk = abs(entry - stop)
    moved = False
    for i in range(start, len(x.m)):
        hit_stop = (x.l[i] <= stop) if side > 0 else (x.h[i] >= stop)
        hit_tgt = target is not None and ((x.h[i] >= target) if side > 0 else (x.l[i] <= target))
        if hit_stop:
            return stop - side * cost.tick, ("be" if moved else "sl")
        if hit_tgt:
            return target, "tp"
        if x.m[i] >= last_exit:
            return x.c[i] - side * cost.tick, "time"
        if be_r and not moved:
            fav = (x.h[i] - entry) if side > 0 else (entry - x.l[i])
            if fav >= be_r * risk:
                stop, moved = entry, True          # effective from the NEXT bar (no look-ahead)
    return x.c[-1] - side * cost.tick, "time"


def orb_exit(days, dd, cost, target_r, be_r, last_exit) -> list:
    out = []
    for x in days:
        atr, trend = dd.at[x.date, "atr14"], dd.at[x.date, "trend"]
        if pd.isna(atr):
            continue
        orh, orl = x.h[:3].max(), x.l[:3].min()
        if orh - orl <= 0:
            continue
        for i in range(3, len(x.m) - 1):
            if x.m[i] >= M(12, 0):
                break
            side = 1 if x.c[i] > orh else (-1 if x.c[i] < orl else 0)
            if side == 0:
                continue
            if side != trend:
                break
            entry = x.o[i + 1]
            stop = orl if side > 0 else orh
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                break
            tgt = entry + side * target_r * abs(entry - stop) if target_r else None
            px, why = walk_be(x, i + 1, side, stop, tgt, last_exit, cost, entry, be_r)
            fh.trade(out, x.date, side, entry, px, stop, why, cost)
            break
    return out


def main() -> int:
    fh.keep_awake()
    all_rows = []
    for contract, proxy in MARKETS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = fh.daily_frame(days)
        per_cell = {}
        for tr, be, tx in product(TARGETS, BREAKEVEN, TIME_EXIT):
            t = pd.DataFrame(orb_exit(days, dd, fh.Cost(contract), tr, be, tx))
            t2 = pd.DataFrame(orb_exit(days, dd, fh.Cost(contract, 2.0), tr, be, tx))
            m = fh.metrics(t)
            t["year"] = pd.to_datetime(t["date"]).dt.year
            per_cell[(tr, be, tx)] = t
            all_rows.append({"contract": contract, "target": tr or "time", "breakeven": be or "-",
                             "time_exit": f"{tx // 60}:{tx % 60:02d}", **{k: m[k] for k in
                             ("n", "usd_mean", "win", "t", "h1", "h2", "years_pos", "max_dd_usd")},
                             "usd_2x": t2["usd"].mean(), "sd": t["usd"].std()})
        # walk-forward: pick the best exit on past years, trade the next
        wf = []
        for y in range(2019, 2027):
            train = {c: t[t.year < y]["usd"].mean() for c, t in per_cell.items()}
            pick = max(train, key=train.get)
            oos = per_cell[pick][per_cell[pick].year == y]["usd"]
            wf.append((y, pick, oos.sum(), len(oos)))
        tot = sum(w[2] for w in wf)
        n = sum(w[3] for w in wf)
        print(f"{contract} walk-forward over exits: ${tot:,.0f} over {n} trades (${tot / max(n, 1):.2f}/trade), "
              f"positive years {np.mean([w[2] > 0 for w in wf]):.0%}; picks: "
              + ", ".join(f"{w[0]}:{w[1][0] or 'time'}R/be{w[1][1] or '-'}/{w[1][2] // 60}h" for w in wf))
    res = pd.DataFrame(all_rows)
    res["sharpe_like"] = res["usd_mean"] / res["sd"]
    res.to_csv(fh.OUT / "exits_results.csv", index=False)
    pd.set_option("display.width", 220)
    for c in MARKETS:
        r = res[res.contract == c].sort_values("t", ascending=False)
        print(f"\n=== {c}: exits by t (deployed = 2R / no breakeven / 15:45) ===")
        print(r.drop(columns=["contract", "sd"]).head(12).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        dep = r[(r.target == 2.0) & (r.breakeven == "-") & (r.time_exit == "15:45")]
        print("deployed:", dep.drop(columns=["contract", "sd"]).to_string(index=False, header=False,
                                                                          float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
