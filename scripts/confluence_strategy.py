"""Can CONFLUENCE raise the win rate enough to matter?

The FTMO 10-day pass rate is bounded by win rate: measured 33.7% -> 31.4%
pass; a hypothetical 66.8% -> 63.8%. Every attempt to raise EXPECTANCY has
failed, but win rate is a different axis and one structural idea remains
untested here: require two INDEPENDENT strategies to agree before entering.

The theory: RSI and Bollinger overlap on only ~10% of signal-days
(scripts/variant_frequency_test.py), so an agreement filter is genuinely
selective rather than a relabelling of the same signal. If the agreeing subset
wins materially more often, that is exactly the lever the pass rate needs.

The theory could also fail in a specific, informative way: agreement may
select for STRONG moves, which mean-reversion strategies are precisely wrong
about. That is worth knowing either way.

Also tested: tight targets on DAILY bars (0.3-0.75R). Those produce ~70% win
rates but were negative on the train half in the geometry sweep -- included
here to see whether confluence rescues them, since a high-win-rate/small-R
profile is theoretically the best fit for a 5%-daily-loss constraint.

Every result must be positive in BOTH halves. Windows are computed strictly
from bars closed before entry.

Usage:
    python scripts/confluence_strategy.py [--workers 6]
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

SIGNALS = {
    "RSI14": (RSIReversionStrategy,
              {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "RSI7": (RSIReversionStrategy,
             {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}),
    "Boll20": (BollingerBounceStrategy,
               {"bollinger": {"period": 20, "std_dev": 2.0}}),
    "Stoch": (StochasticStrategy, {"stochastic": {}}),
}
TPS = (0.5, 0.75, 1.0, 2.0, 3.0, 4.0)


def _cell(job):
    name, tp, sym = job
    cls, params = SIGNALS[name]
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
        out = {"name": name, "tp": tp, "sym": sym, "halves": {}}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 250:
                continue
            bt = Backtest(cfg, b, strategy=Windowed(cls(cfg)))
            bt.run()
            out["halves"][half] = [
                {"d": pd.Timestamp(t.entered_at).normalize(),
                 "dir": t.direction.name,
                 "r": t.pnl_usd / RISK_USD}
                for t in bt.closed]
        return out
    except Exception:
        return None


def stats(rs):
    if not rs:
        return None
    a = np.array(rs)
    wins = a[a > 0]
    gl = -a[a <= 0].sum()
    return {"n": len(a), "win": len(wins) / len(a), "e": float(a.mean()),
            "pf": float(wins.sum() / gl) if gl > 0 else float("inf")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print("=" * 100)
    print("CONFLUENCE — does requiring two strategies to agree raise the win rate?")
    print("=" * 100, flush=True)

    jobs = [(n, tp, s) for n in SIGNALS for tp in TPS for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=10):
            if c:
                cells.append(c)

    # index: (name, tp) -> {half -> {(sym, date, dir) -> r}}
    idx = {}
    for c in cells:
        k = (c["name"], c["tp"])
        d = idx.setdefault(k, {"train": {}, "oos": {}})
        for half, trs in c["halves"].items():
            for t in trs:
                d[half][(c["sym"], t["d"], t["dir"])] = t["r"]

    print(f"\n{'setup':<34}{'TP':>5}{'n_oos':>7}{'win_tr':>8}{'win_oos':>9}"
          f"{'E_tr':>9}{'E_oos':>9}{'PF_oos':>8}  gate")
    print("-" * 100)

    rows = []
    # (a) each signal alone, for the baseline
    for name in SIGNALS:
        for tp in TPS:
            d = idx.get((name, tp))
            if not d:
                continue
            tr, oo = stats(list(d["train"].values())), stats(list(d["oos"].values()))
            if not tr or not oo or oo["n"] < 80:
                continue
            gate = "KEEP" if (tr["e"] > 0 and oo["e"] > 0) else ""
            rows.append((f"{name} alone", tp, oo, tr, gate))
            print(f"{name+' alone':<34}{tp:>5.2f}{oo['n']:>7}{tr['win']*100:>8.1f}"
                  f"{oo['win']*100:>9.1f}{tr['e']:>+9.3f}{oo['e']:>+9.3f}"
                  f"{oo['pf']:>8.2f}  {gate}")

    print()
    # (b) confluence: both signals fired on the same symbol/day/direction
    for a, b in itertools.combinations(SIGNALS, 2):
        for tp in TPS:
            da, db = idx.get((a, tp)), idx.get((b, tp))
            if not da or not db:
                continue
            res = {}
            for half in ("train", "oos"):
                keys = set(da[half]) & set(db[half])
                # average the two legs' outcome on an agreed setup
                res[half] = [(da[half][k] + db[half][k]) / 2 for k in keys]
            tr, oo = stats(res["train"]), stats(res["oos"])
            if not tr or not oo or oo["n"] < 40:
                continue
            gate = "KEEP" if (tr["e"] > 0 and oo["e"] > 0) else ""
            label = f"{a}+{b} agree"
            rows.append((label, tp, oo, tr, gate))
            print(f"{label:<34}{tp:>5.2f}{oo['n']:>7}{tr['win']*100:>8.1f}"
                  f"{oo['win']*100:>9.1f}{tr['e']:>+9.3f}{oo['e']:>+9.3f}"
                  f"{oo['pf']:>8.2f}  {gate}")

    print("\n" + "=" * 100)
    keep = [r for r in rows if r[4] == "KEEP"]
    print(f"SURVIVORS (positive in BOTH halves): {len(keep)}")
    print("=" * 100)
    print("Ranked by OOS WIN RATE -- the axis the FTMO pass rate depends on:")
    for lbl, tp, oo, tr, _ in sorted(keep, key=lambda r: -r[2]["win"])[:10]:
        print(f"  {lbl:<32} TP {tp:<5} win_oos {oo['win']*100:>5.1f}%  "
              f"E_oos {oo['e']:+.3f}R  E_train {tr['e']:+.3f}R  n={oo['n']}")
    if keep:
        best = max(keep, key=lambda r: r[2]["win"])
        print(f"\n  Best win rate among survivors: {best[2]['win']*100:.1f}%")
        print(f"  Needed for a 60%+ 10-day pass rate: ~65%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
