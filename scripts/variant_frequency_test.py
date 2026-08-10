"""Can more trades/day be bought WITHOUT losing the edge?

Frequency cannot come from a faster timeframe -- every canonical strategy is
negative at 1h and 4h, and every (stop, target) pair on the hourly chart is
negative too. The only remaining route is more strategies/parameterisations on
the DAILY chart.

This measures whether that route actually works, which needs three answers,
not one:

  1. Does each VARIANT hold up on its own, in BOTH halves of history?
     (scripts/rsi_robustness.py found 45 of 48 daily RSI parameter cells
     positive, so this is plausible -- but "plausible" is not "measured".)
  2. How much NEW frequency does each add? Two RSI settings on the same
     symbol often fire on the same day; overlapping signals add correlation
     without adding opportunity.
  3. What happens to expectancy when they are pooled?

The overlap number is the one that decides it. The 17-pair basket is already
worth only ~4.9 independent bets (scripts/correlation_audit.py); variants that
fire together make that worse, and a drawdown built from simultaneous losers
is what turned +$374 into -$832 in the live ledger.

Usage:
    python scripts/variant_frequency_test.py [--workers 6]
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

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]

# DEPLOYED legs first, then candidate variants around them. Parameters are
# spread deliberately -- adjacent settings fire on the same bars and buy
# nothing.
VARIANTS = [
    ("RSI14_30/70 *deployed*", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 5.0),
    ("RSI7_25/75", RSIReversionStrategy,
     {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}, 5.0),
    ("RSI21_35/65", RSIReversionStrategy,
     {"rsi": {"period": 21, "oversold": 35, "overbought": 65}}, 5.0),
    ("Boll20_2.0 *deployed*", BollingerBounceStrategy,
     {"bollinger": {"period": 20, "std_dev": 2.0}}, 4.0),
    ("Boll10_2.5", BollingerBounceStrategy,
     {"bollinger": {"period": 10, "std_dev": 2.5}}, 4.0),
    ("Boll40_1.5", BollingerBounceStrategy,
     {"bollinger": {"period": 40, "std_dev": 1.5}}, 4.0),
]


def _cell(job):
    idx, sym = job
    name, cls, params, tp = VARIANTS[idx]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        strat = {"session": None,
                 "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym,
               "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                           "max_positions": 2, "daily_stop_loss_usd": -1e9},
               "strategy": strat}
        out = {"name": name, "sym": sym, "trades": {}}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 250:
                continue
            bt = Backtest(cfg, b, strategy=Windowed(cls(cfg)))
            bt.run()
            out["trades"][half] = [
                (str(pd.Timestamp(t.entered_at).date()), t.pnl_usd / RISK_USD)
                for t in bt.closed]
            out.setdefault("days", {})[half] = len(set(b.index.date))
        return out
    except Exception:
        return None


def agg(cells, half):
    rs = [r for c in cells for _, r in c["trades"].get(half, [])]
    days = max((c.get("days", {}).get(half, 0) for c in cells), default=0)
    if not rs:
        return None
    wins = [x for x in rs if x > 0]
    gl = -sum(x for x in rs if x <= 0)
    return {"n": len(rs), "per_day": len(rs) / days if days else 0,
            "win": len(wins) / len(rs), "e": float(np.mean(rs)),
            "pf": (sum(wins) / gl) if gl > 0 else float("inf")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = [(i, s) for i in range(len(VARIANTS)) for s in SYMBOLS]
    print("=" * 100)
    print("VARIANT FREQUENCY TEST -- more trades/day without losing the edge?")
    print(f"{len(VARIANTS)} variants x {len(SYMBOLS)} symbols = {len(jobs)} cells")
    print("=" * 100, flush=True)

    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=6):
            if c:
                cells.append(c)

    print(f"\n(1) EACH VARIANT ALONE\n{'-' * 100}")
    print(f"  {'variant':<26}{'n':>7}{'/day':>8}{'win%':>8}{'E_oos':>9}"
          f"{'E_train':>10}{'PF':>7}  verdict")
    survivors = []
    for i, (name, *_rest) in enumerate(VARIANTS):
        mine = [c for c in cells if c["name"] == name]
        o, t = agg(mine, "oos"), agg(mine, "train")
        if not o or not t:
            continue
        ok = o["e"] > 0 and t["e"] > 0
        if ok:
            survivors.append(name)
        print(f"  {name:<26}{o['n']:>7}{o['per_day']:>8.2f}{o['win']*100:>8.1f}"
              f"{o['e']:>+9.3f}{t['e']:>+10.3f}{o['pf']:>7.2f}"
              f"  {'KEEP' if ok else 'reject (a half is negative)'}")

    # ---- (2) overlap: do variants fire on the SAME symbol+day? ----------
    print(f"\n(2) SIGNAL OVERLAP -- new opportunity, or the same bet twice?\n{'-' * 100}")
    day_sets = {}
    for name in survivors:
        s = set()
        for c in [c for c in cells if c["name"] == name]:
            for d, _ in c["trades"].get("oos", []):
                s.add((c["sym"], d))
        day_sets[name] = s
    print(f"  {'pair':<52}{'overlap':>10}{'verdict':>22}")
    for a, b in itertools.combinations(survivors, 2):
        inter = len(day_sets[a] & day_sets[b])
        union = len(day_sets[a] | day_sets[b])
        jac = inter / union if union else 0
        verdict = ("mostly the SAME bet" if jac > 0.5 else
                   "partly overlapping" if jac > 0.2 else "largely INDEPENDENT")
        print(f"  {a[:24]:<25} vs {b[:24]:<25}{jac:>10.0%}{verdict:>22}")

    # ---- (3) pooled ----------------------------------------------------
    print(f"\n(3) POOLED (all survivors together)\n{'-' * 100}")
    pooled = [c for c in cells if c["name"] in survivors]
    o, t = agg(pooled, "oos"), agg(pooled, "train")
    dep = [c for c in cells if "*deployed*" in c["name"]]
    od, td = agg(dep, "oos"), agg(dep, "train")
    if o and od:
        print(f"  {'':<26}{'/day':>8}{'win%':>8}{'E_oos':>9}{'E_train':>10}"
              f"{'R/day':>9}")
        print(f"  {'DEPLOYED (2 legs)':<26}{od['per_day']:>8.2f}"
              f"{od['win']*100:>8.1f}{od['e']:>+9.3f}{td['e']:>+10.3f}"
              f"{od['e']*od['per_day']:>+9.4f}")
        print(f"  {'ALL SURVIVORS':<26}{o['per_day']:>8.2f}"
              f"{o['win']*100:>8.1f}{o['e']:>+9.3f}{t['e']:>+10.3f}"
              f"{o['e']*o['per_day']:>+9.4f}")
        print(f"\n  frequency {od['per_day']:.2f} -> {o['per_day']:.2f}/day "
              f"({o['per_day']/od['per_day']:.2f}x)")
        print(f"  expectancy {od['e']:+.3f}R -> {o['e']:+.3f}R "
              f"({(o['e']/od['e']-1)*100:+.0f}%)")
        print(f"  R/day {od['e']*od['per_day']:+.4f} -> "
              f"{o['e']*o['per_day']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
