"""Stop/target geometry, optimised SEPARATELY for each book.

The two books want different geometry because they are scored differently:

  FTMO  — at risk r with a target of tp_r, one winning trade moves the account
          by r * tp_r. To pass (+10%) on a single win you need r * tp_r >= 0.10.
          That can be bought as 10% risk x 1R, or 5% x 2R, or 3.3% x 3R...
          and those are NOT equivalent: a 5% loss survives the 10% overall
          rule and lets the book try again tomorrow, whereas a 10% loss ends
          the evaluation immediately. So the right FTMO geometry is whichever
          (stop, target, risk) triple maximises P(pass within the deadline),
          which trades win rate against number of attempts.

  YOLO  — scored on growth, so it wants the geometry with the best
          expectancy x frequency, subject to surviving.

Both are measured, not assumed: the sweep runs real backtests over ~10.5
years with the project's cost model, splits train/OOS, and reports the whole
surface so a plateau can be told apart from a lucky cell.

Usage:
    python scripts/rsi_geometry_sweep.py [--tf 1d 1h] [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

DAILY_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]
H1_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY", "GBPJPY"]

# Wider stops mechanically improve the cost-to-risk ratio, which
# scripts/loss_attribution.py identified as the dominant loss source
# (<1 ATR -0.167R -> 3+ ATR +0.181R). The sweep tests that directly.
SL_LOOKBACKS = [5, 10, 14, 20, 30]
TP_RS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

RNG = np.random.default_rng(97)


def _cell(job):
    tf, sl_lb, tp_r, sym = job
    try:
        bars = load(sym, tf)
        if bars is None:
            return None
        # Drop the partial Sunday bar: it is excluded live via skip_weekdays,
        # so leaving it in here would measure a book we do not run.
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        cfg = {
            "symbol": sym,
            "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                        "max_positions": 2, "daily_stop_loss_usd": -1e9},
            "strategy": {"rsi": {"period": 14, "oversold": 30, "overbought": 70},
                         "session": None,
                         "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": sl_lb}},
        }
        out = {}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 250:
                continue
            bt = Backtest(cfg, b, strategy=Windowed(RSIReversionStrategy(cfg)))
            bt.run()
            out[half] = {"rs": [t.pnl_usd / RISK_USD for t in bt.closed],
                         "days": len(set(b.index.date))}
        return {"tf": tf, "sl_lb": sl_lb, "tp_r": tp_r, "sym": sym, **out}
    except Exception:
        return None


def agg(cells, half):
    rs = [r for c in cells for r in c.get(half, {}).get("rs", [])]
    days = max((c.get(half, {}).get("days", 0) for c in cells), default=0)
    if not rs:
        return None
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gl = -sum(losses)
    return {"n": len(rs), "per_day": len(rs) / days if days else 0,
            "win": len(wins) / len(rs), "e": float(np.mean(rs)),
            "total": float(sum(rs)), "pf": (sum(wins) / gl) if gl > 0 else float("inf")}


def sim_ftmo(win, tp_r, per_day, risk, days, n=12_000):
    """Day-by-day FTMO sim for one geometry+risk combo."""
    init = 10_000.0
    eq = np.full(n, init)
    st = np.zeros(n, np.int8)
    for _ in range(days):
        day0 = eq.copy()
        k = RNG.poisson(per_day, n)
        for t in range(int(k.max()) if k.max() else 0):
            act = (st == 0) & (k > t)
            if not act.any():
                continue
            stake = eq[act] * risk
            w = RNG.random(act.sum()) < win
            eq[act] = eq[act] + np.where(w, stake * tp_r, -stake)
            idx = np.where(act)[0]
            st[idx[(eq[idx] <= day0[idx] * 0.95) & (st[idx] == 0)]] = 2
            st[idx[(eq[idx] <= init * 0.90) & (st[idx] == 0)]] = 3
            st[idx[(eq[idx] >= init * 1.10) & (st[idx] == 0)]] = 1
    return (st == 1).mean(), ((st == 2) | (st == 3)).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=["1d", "1h"])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = []
    for tf in args.tf:
        syms = DAILY_SYMBOLS if tf == "1d" else H1_SYMBOLS
        for sl_lb, tp_r in itertools.product(SL_LOOKBACKS, TP_RS):
            for s in syms:
                jobs.append((tf, sl_lb, tp_r, s))

    print("=" * 104)
    print("RSI STOP/TARGET GEOMETRY SWEEP - optimised per book")
    print(f"{len(jobs)} cells on {args.workers} workers (Sunday bars excluded)")
    print("=" * 104, flush=True)

    cells = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            done += 1
            if c:
                cells.append(c)
            if done % 200 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)

    results = []
    for tf in args.tf:
        print(f"\n{'=' * 104}\n### TIMEFRAME {tf} - out-of-sample surface\n{'=' * 104}")
        print(f"{'SL swing':>9}{'TP (R)':>8}{'n':>7}{'/day':>7}{'win%':>7}"
              f"{'E_net':>9}{'PF':>7}{'totR':>9}{'trainE':>9}")
        print("-" * 104)
        for sl_lb, tp_r in itertools.product(SL_LOOKBACKS, TP_RS):
            mine = [c for c in cells
                    if c["tf"] == tf and c["sl_lb"] == sl_lb and c["tp_r"] == tp_r]
            o, t = agg(mine, "oos"), agg(mine, "train")
            if not o:
                continue
            pf = f"{o['pf']:.2f}" if o["pf"] != float("inf") else "inf"
            print(f"{sl_lb:>9}{tp_r:>8.2f}{o['n']:>7}{o['per_day']:>7.2f}"
                  f"{o['win']*100:>7.1f}{o['e']:>+9.3f}{pf:>7}{o['total']:>+9.1f}"
                  f"{(t['e'] if t else float('nan')):>+9.3f}")
            results.append({"tf": tf, "sl_lb": sl_lb, "tp_r": tp_r,
                            "oos": o, "train": t})

    # ---- FTMO objective -------------------------------------------------
    print(f"\n{'=' * 104}\n### FTMO: best (stop, target, risk) for P(pass)\n{'=' * 104}")
    print("Only combos where ONE win reaches +10% (risk x TP >= 0.10) can pass on a")
    print("single trade; lower risk survives a loss and gets more attempts.\n")
    d1 = [r for r in results if r["tf"] == "1d" and r["oos"]["n"] >= 100
          and r["train"] and r["train"]["e"] > 0]
    scored = []
    for r in d1:
        for risk in (0.03, 0.04, 0.05, 0.075, 0.10, 0.15):
            if risk * r["tp_r"] < 0.10:
                continue
            for days in (10, 30):
                p, f = sim_ftmo(r["oos"]["win"], r["tp_r"], r["oos"]["per_day"],
                                risk, days)
                scored.append({"sl_lb": r["sl_lb"], "tp_r": r["tp_r"], "risk": risk,
                               "days": days, "p": p, "f": f,
                               "win": r["oos"]["win"], "e": r["oos"]["e"]})
    for days in (10, 30):
        top = sorted([s for s in scored if s["days"] == days],
                     key=lambda s: -s["p"])[:8]
        print(f"  --- within {days} trading days ---")
        print(f"  {'SL':>4}{'TP':>7}{'risk':>8}{'win%':>8}{'P(pass)':>10}{'P(fail)':>10}")
        for s in top:
            print(f"  {s['sl_lb']:>4}{s['tp_r']:>7.2f}{s['risk']*100:>7.1f}%"
                  f"{s['win']*100:>8.1f}{s['p']:>10.1%}{s['f']:>10.1%}")
        print()

    # ---- YOLO objective -------------------------------------------------
    print(f"{'=' * 104}\n### YOLO: best geometry by R-per-day (growth proxy)\n{'=' * 104}")
    print(f"  {'tf':>4}{'SL':>5}{'TP':>7}{'/day':>7}{'win%':>8}{'E_net':>9}"
          f"{'R/day':>9}{'trainE':>9}")
    ranked = [r for r in results if r["oos"]["n"] >= 80 and r["train"]]
    for r in sorted(ranked, key=lambda x: -(x["oos"]["e"] * x["oos"]["per_day"]))[:12]:
        rpd = r["oos"]["e"] * r["oos"]["per_day"]
        print(f"  {r['tf']:>4}{r['sl_lb']:>5}{r['tp_r']:>7.2f}{r['oos']['per_day']:>7.2f}"
              f"{r['oos']['win']*100:>8.1f}{r['oos']['e']:>+9.3f}{rpd:>+9.4f}"
              f"{r['train']['e']:>+9.3f}")

    out = ROOT / "backtest" / "rsi_geometry.json"
    with open(out, "w") as f:
        json.dump({"surface": results, "ftmo": scored}, f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
