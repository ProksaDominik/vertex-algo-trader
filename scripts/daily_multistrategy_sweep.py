"""Every daily strategy x every target -- to build a multi-strategy stack.

Why this exists: canonical_strategy_test.py ran all mean-reversion strategies
at a 1R target, and rsi_geometry_sweep.py then showed 1R was badly wrong for
RSI (+0.114R at 1R vs +0.274R at 3R). That correction was never applied to the
OTHER strategies, so any of them may have been dismissed at the wrong
geometry -- and some fire far more often than RSI does (Bollinger ~0.94
trades/day vs RSI ~0.38).

That matters because trade frequency at POSITIVE expectancy is the binding
constraint on both books. Frequency cannot come from a faster timeframe (all
16 canonical strategies are negative at 1h and 4h). It can only come from
running MORE INDEPENDENT STRATEGIES on the daily timeframe -- which is what
this sweep is for.

Everything is out-of-sample, real costs, Sundays excluded, with the train-half
number printed alongside so a result that only works in one era is visible.

Usage:
    python scripts/daily_multistrategy_sweep.py [--workers 6]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_smc_bos import SMCBOSStrategy  # noqa: E402
from metatraderbot.strategies_stochastic import StochasticStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]
TP_RS = [3.0, 4.0, 5.0, 6.0, 8.0]

STRATS = {  # extension run: RSI + Bollinger only
    "RSI":        (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "Bollinger":  (BollingerBounceStrategy, {"bollinger": {}}),
    "Stochastic": (StochasticStrategy, {"stochastic": {}}),
    "Supertrend": (SupertrendStrategy, {"supertrend": {"period": 10, "multiplier": 3.0}}),
    "SMACross":   (SMACrossStrategy, {"ma_cross": {}}),
    "MACD":       (MACDTrendStrategy, {"macd": {}}),
    "Donchian20": (DonchianBreakoutStrategy, {"donchian": {"period": 20}}),
    "EMACross":   (EMACrossoverStrategy, {"ema_cross": {"fast_period": 20, "slow_period": 50}}),
    "Ichimoku":   (IchimokuStrategy, {"ichimoku": {}}),
    "SMC_BOS":    (SMCBOSStrategy, {"smc_bos": {}}),
}


def _cell(job):
    name, tp_r, sym = job
    cls, params = STRATS[name]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]     # partial Sunday bar
        split = int(len(bars) * 0.70)
        strat = {"session": None,
                 "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym,
               "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                           "max_positions": 2, "daily_stop_loss_usd": -1e9},
               "strategy": strat}
        out = {"name": name, "tp_r": tp_r, "sym": sym}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 250:
                continue
            bt = Backtest(cfg, b, strategy=Windowed(cls(cfg)))
            bt.run()
            out[half] = {"rs": [t.pnl_usd / RISK_USD for t in bt.closed],
                         "days": len(set(b.index.date)),
                         "dates": sorted({str(t.entered_at)[:10] for t in bt.closed})}
        return out
    except Exception:
        return None


def agg(cells, half):
    rs = [r for c in cells for r in c.get(half, {}).get("rs", [])]
    days = max((c.get(half, {}).get("days", 0) for c in cells), default=0)
    if not rs:
        return None
    wins = [x for x in rs if x > 0]
    gl = -sum(x for x in rs if x <= 0)
    return {"n": len(rs), "per_day": len(rs) / days if days else 0,
            "win": len(wins) / len(rs), "e": float(np.mean(rs)),
            "total": float(sum(rs)),
            "pf": (sum(wins) / gl) if gl > 0 else float("inf"),
            "r_per_day": float(np.mean(rs)) * (len(rs) / days if days else 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    jobs = [(n, tp, s) for n in STRATS for tp in TP_RS for s in SYMBOLS]
    print("=" * 104)
    print("DAILY MULTI-STRATEGY x TARGET SWEEP")
    print(f"{len(STRATS)} strategies x {len(TP_RS)} targets x {len(SYMBOLS)} symbols "
          f"= {len(jobs)} cells")
    print("=" * 104, flush=True)

    cells, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            done += 1
            if c:
                cells.append(c)
            if done % 300 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)

    results = []
    print(f"\n{'strategy':<12}{'TP':>6}{'n':>7}{'/day':>7}{'win%':>7}{'E_net':>9}"
          f"{'PF':>7}{'R/day':>9}{'trainE':>9}  verdict")
    print("-" * 104)
    for name in STRATS:
        for tp in TP_RS:
            mine = [c for c in cells if c["name"] == name and c["tp_r"] == tp]
            o, t = agg(mine, "oos"), agg(mine, "train")
            if not o:
                continue
            both = (o["e"] > 0 and t and t["e"] > 0)
            verdict = "KEEP (both halves +)" if both else ("oos+ only" if o["e"] > 0 else "")
            pf = f"{o['pf']:.2f}" if o["pf"] != float("inf") else "inf"
            print(f"{name:<12}{tp:>6.1f}{o['n']:>7}{o['per_day']:>7.2f}"
                  f"{o['win']*100:>7.1f}{o['e']:>+9.3f}{pf:>7}{o['r_per_day']:>+9.4f}"
                  f"{(t['e'] if t else float('nan')):>+9.3f}  {verdict}")
            results.append({"name": name, "tp_r": tp, "oos": o, "train": t,
                            "both_positive": both})
        print()

    keep = [r for r in results if r["both_positive"] and r["oos"]["n"] >= 100]
    print("=" * 104)
    print("SURVIVORS -- positive out-of-sample AND positive in-sample, n>=100")
    print("=" * 104)
    if not keep:
        print("  none.")
        return 0
    best = {}
    for r in sorted(keep, key=lambda x: -x["oos"]["r_per_day"]):
        if r["name"] not in best:
            best[r["name"]] = r
        print(f"  {r['name']:<12} TP {r['tp_r']:<4} {r['oos']['per_day']:>5.2f}/day "
              f"E={r['oos']['e']:+.3f}R  R/day={r['oos']['r_per_day']:+.4f}  "
              f"train={r['train']['e']:+.3f}R")

    print("\n" + "=" * 104)
    print("BEST STACK -- one geometry per surviving strategy")
    print("=" * 104)
    tot_day = sum(b["oos"]["per_day"] for b in best.values())
    tot_rday = sum(b["oos"]["r_per_day"] for b in best.values())
    for n, b in sorted(best.items(), key=lambda kv: -kv[1]["oos"]["r_per_day"]):
        print(f"  {n:<12} TP {b['tp_r']:<4} {b['oos']['per_day']:>5.2f}/day  "
              f"E={b['oos']['e']:+.3f}R  R/day={b['oos']['r_per_day']:+.4f}")
    print(f"\n  COMBINED: {tot_day:.2f} trades/day   {tot_rday:+.4f} R/day")
    print(f"  blended expectancy ~ {tot_rday/tot_day:+.3f}R per trade"
          if tot_day else "")
    print("\n  NOTE: these strategies trade the same 17 correlated pairs, so their")
    print("  signals are NOT independent -- see scripts/correlation_audit.py (the")
    print("  basket is worth ~4.9 independent bets). Expect the live combined")
    print("  drawdown to be worse than summing the individual ones suggests.")

    out = ROOT / "backtest" / "daily_multistrategy.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
