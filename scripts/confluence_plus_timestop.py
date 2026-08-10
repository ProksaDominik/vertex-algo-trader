"""Combine the two things that actually worked, and push each further.

Two independent discoveries so far:
  1. HARD TIME STOP  -- cuts median hold 18d -> 1-2d, freeing position slots
     ~10x faster. Lifted the 30-day pass rate 31.9% -> 43.1%.
  2. CONFLUENCE      -- requiring RSI(14) and RSI(7) to agree lifts the OOS
     win rate 33.7% -> 65.6%, producing a much smoother equity path. Lifted
     the 10-day rate 32.3% -> 39.1%.

They have never been combined. Confluence currently runs with NO time stop,
so it still holds positions for days; adding one should compound the two
effects rather than trade them off.

Also pushed here:
  * THREE-WAY agreement (RSI14 + RSI7 + Stochastic / Bollinger). Higher
    conviction still -- the question is whether win rate keeps climbing or
    the sample just collapses.
  * A finer target sweep (0.5 - 1.5R) around the region where the high-win
    profile lives, rather than the coarse grid used before.

The binding constraint is NOT expectancy -- it is that anything volatile
enough to make +10% fast also produces 5% down days. So the target here is a
high win rate at the LARGEST payoff that keeps it, since pass rate needs both.

Every result must be positive in BOTH halves. Confluence uses only bars
closed before entry on every leg, so agreement introduces no lookahead.

Usage:
    python scripts/confluence_plus_timestop.py [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
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
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_stochastic import StochasticStrategy  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
           "AUDCHF", "EURAUD", "AUDCAD"]

SIG = {
    "RSI14": (RSIReversionStrategy,
              {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "RSI7": (RSIReversionStrategy,
             {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}),
    "RSI21": (RSIReversionStrategy,
              {"rsi": {"period": 21, "oversold": 35, "overbought": 65}}),
    "Boll20": (BollingerBounceStrategy,
               {"bollinger": {"period": 20, "std_dev": 2.0}}),
    "Stoch": (StochasticStrategy, {"stochastic": {}}),
}
# base signal -> partners that must also agree
COMBOS = [
    ("RSI14", ("RSI7",)),
    ("RSI14", ("Stoch",)),
    ("RSI14", ("RSI7", "Stoch")),
    ("RSI14", ("RSI7", "Boll20")),
    ("RSI14", ("RSI7", "RSI21")),
    ("RSI14", ("RSI7", "RSI21", "Stoch")),
]
TPS = (0.5, 0.75, 1.0, 1.25, 1.5)
HOLDS = (None, 1, 2, 3)


def _trades(name, tp, sym, bars):
    cls, params = SIG[name]
    strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return bt.closed


def _cell(job):
    ci, tp, sym = job
    base, partners = COMBOS[ci]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        cutoff = bars.index[split]
        closed = _trades(base, tp, sym, bars)
        agree = None
        for p in partners:
            s = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name)
                 for t in _trades(p, tp, sym, bars)}
            agree = s if agree is None else (agree & s)
        cl = bars["close"]
        out = {h: {"train": [], "oos": []} for h in HOLDS}
        for t in closed:
            o = pd.Timestamp(t.entered_at).normalize()
            if agree is not None and (o, t.direction.name) not in agree:
                continue
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            half = "train" if o < cutoff else "oos"
            for h in HOLDS:
                if h is None or len(path) <= h + 1:
                    r = float(path[-1])
                else:
                    cost = path[-1] - (seg.values[-1] - t.entry_price) * sg / rp
                    r = float(path[h] + cost)
                out[h][half].append(r)
        return {"ci": ci, "tp": tp, "out": out}
    except Exception:
        return None


def stat(rs):
    if not rs:
        return None
    a = np.array(rs)
    w = a[a > 0]
    gl = -a[a <= 0].sum()
    return {"n": len(a), "win": len(w) / len(a), "e": float(a.mean()),
            "pf": float(w.sum() / gl) if gl > 0 else float("inf")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print("=" * 106)
    print("CONFLUENCE x TIME STOP — combining the two things that worked")
    print("=" * 106, flush=True)

    jobs = [(ci, tp, s) for ci in range(len(COMBOS)) for tp in TPS for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=10):
            if c:
                cells.append(c)

    print(f"\n{'agreement':<34}{'TP':>5}{'hold':>6}{'n_oos':>7}{'win_tr':>8}"
          f"{'win_oos':>9}{'E_tr':>9}{'E_oos':>9}{'PF':>7}  gate")
    print("-" * 106)
    keep = []
    for ci, (base, partners) in enumerate(COMBOS):
        label = "+".join((base,) + partners)
        for tp in TPS:
            mine = [c for c in cells if c["ci"] == ci and c["tp"] == tp]
            if not mine:
                continue
            for h in HOLDS:
                tr = stat([r for c in mine for r in c["out"][h]["train"]])
                oo = stat([r for c in mine for r in c["out"][h]["oos"]])
                if not tr or not oo or oo["n"] < 60:
                    continue
                ok = tr["e"] > 0 and oo["e"] > 0
                if ok:
                    keep.append((label, tp, h, tr, oo))
                print(f"{label[:33]:<34}{tp:>5.2f}{str(h or '-'):>6}{oo['n']:>7}"
                      f"{tr['win']*100:>8.1f}{oo['win']*100:>9.1f}{tr['e']:>+9.3f}"
                      f"{oo['e']:>+9.3f}{oo['pf']:>7.2f}  {'KEEP' if ok else ''}")

    print("\n" + "=" * 106)
    print(f"SURVIVORS (positive in BOTH halves): {len(keep)}")
    print("=" * 106)
    print("Ranked by win_oos x E_oos -- pass rate needs BOTH a high win rate")
    print("AND enough payoff per win; either alone is not sufficient.\n")
    for lbl, tp, h, tr, oo in sorted(
            keep, key=lambda r: -(r[4]["win"] * max(r[4]["e"], 0)))[:12]:
        print(f"  {lbl[:30]:<31} TP {tp:<5} hold {str(h or '-'):<3} "
              f"win {oo['win']*100:>5.1f}%  E_oos {oo['e']:+.3f}R  "
              f"E_tr {tr['e']:+.3f}R  n={oo['n']}")
    if keep:
        bw = max(keep, key=lambda r: r[4]["win"])
        print(f"\n  highest win rate: {bw[4]['win']*100:.1f}% "
              f"({bw[0]}, TP {bw[1]}, hold {bw[2] or '-'}, E_oos {bw[4]['e']:+.3f}R)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
