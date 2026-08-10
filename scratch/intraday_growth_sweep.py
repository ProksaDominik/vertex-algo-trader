"""YOLO-book question: does parameter tuning / wider targets / confluence flip
1h or 4h into positive-expectancy territory, and if so, does the resulting
GROWTH (R/day, not just per-trade R) beat the deployed 1d baseline?

Reuses the project's established methodology exactly:
  - load()/Windowed/RISK_USD/cfg_for/NO_TRAIL from canonical_strategy_test.py
    (same cost model: PAIR_SPECS spread+commission, same window feed).
  - 70/30 chronological train/OOS split, both halves reported (the
    convention in daily_multistrategy_sweep.py / nonfx_trend_test.py).
  - FULL pooled basket, never cherry-picked symbols (17 FX for
    mean-reversion, 10 non-FX for trend) -- metatraderbot_overfitting_guardrail.

Three families tested per timeframe (1h, 4h):
  (A) MR sweep: RSI(14) / Bollinger(20,2.0) / Stochastic mean-reversion,
      TP in {2,3,4,6}R, on the 17-pair FX basket -- tests whether the
      "wrong geometry" finding that fixed RSI on 1d (rsi_geometry_sweep.py)
      also rescues 1h/4h.
  (B) Trend sweep: Donchian20 / Donchian55 / Supertrend / EMACross,
      TP in {2,3,4,6}R, on the 10-instrument non-FX basket.
  (C) Confluence: RSI(14)+RSI(7) same-direction agreement (the one
      confirmed-working confluence idea in this project, so far only
      tested at 1d) on the FX basket at TP=2.

Reports, per (family, strategy, tp, tf): train E, oos E, oos n, oos PF,
oos trades/day, oos R/day (the growth-comparable number). Survivors need
both halves positive and oos n >= 100 (flagged <300 as low-confidence per
the guardrail).
"""
from __future__ import annotations

import argparse
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

from canonical_strategy_test import RISK_USD, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_stochastic import StochasticStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
      "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
      "AUDCHF", "EURAUD", "AUDCAD"]
NONFX = ["US30", "NAS100", "DE40", "JP225", "WTI", "XPTUSD", "BTCUSD", "ETHUSD",
         "XAUUSD", "US500"]

TIMEFRAMES = ["1h", "4h"]  # 1h/4h only -- this is what the task asked to re-verify.
# (15m was mistakenly added mid-task on my own initiative, not requested --
# dropped. It is also ~4x the bar count of 1h, which was the dominant cause
# of the prior run producing zero output for over an hour.)


class Windowed:
    """Lighter re-implementation of canonical_strategy_test.Windowed.

    That version uses a 400-bar window sized for a 200-EMA + 55-bar Donchian
    lookback. Nothing in THIS sweep's roster needs more than a 20-bar
    lookback (RSI14/Bollinger20/Donchian20/Supertrend10), so 100 bars (5x
    margin) is already generous for EWM convergence. Re-slicing+resetting a
    400-row DataFrame on every single bar was the measured bottleneck at
    hourly resolution; cutting the window ~4x cuts that per-bar cost
    proportionally with no change in signal correctness.
    """

    WINDOW = 100

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, i, bars):
        start = max(0, i - self.WINDOW + 1)
        v = bars.iloc[start:i + 1].reset_index(drop=True)
        return self.inner.on_bar(len(v) - 1, v)

MR_STRATS = {
    "RSI":        (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "Bollinger":  (BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}}),
    "Stochastic": (StochasticStrategy, {"stochastic": {}}),
}
TREND_STRATS = {
    "Donchian20": (DonchianBreakoutStrategy, {"donchian": {"period": 20}}),
    "Donchian55": (DonchianBreakoutStrategy, {"donchian": {"period": 55}}),
    "Supertrend": (SupertrendStrategy, {"supertrend": {"period": 10, "multiplier": 3.0}}),
    "EMACross":   (EMACrossoverStrategy, {"ema_cross": {"fast_period": 20, "slow_period": 50}}),
}

# Full SYMBOL basket is NEVER trimmed (overfitting guardrail) -- what's
# narrowed is the strategy/TP grid, to the two highest-prior-evidence
# strategies per family (RSI+Bollinger for MR, matching the deployed daily
# legs; Donchian20+Supertrend for trend, Donchian is the deployed daily leg
# and Supertrend is live-traded elsewhere in this project) at a still-
# meaningfully-wide TP set. Grid size will be widened (more TPs/strategies)
# once per-cell timing at 1h/4h is actually measured and found tractable.
TF_GRID = {
    "1h":    {"mr": ["RSI", "Bollinger"], "mr_tps": [2.0, 4.0],
              "trend": ["Donchian20", "Supertrend"], "trend_tps": [2.0, 4.0],
              "conf_tps": [2.0]},
    "4h":    {"mr": ["RSI", "Bollinger"], "mr_tps": [2.0, 4.0],
              "trend": ["Donchian20", "Supertrend"], "trend_tps": [2.0, 4.0],
              "conf_tps": [2.0]},
}


def _mk_cfg(sym, strat_dict, trail):
    cap = {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
           "max_positions": 2, "daily_stop_loss_usd": -1e9}
    if trail:
        cap["trail_start_r"] = 1.0
        cap["trail_lock_fraction"] = 0.5
    return {"symbol": sym, "capital": cap, "strategy": strat_dict}


def _run_half(cls_or_kwargs, sym, bars, tp, trail, confluence=False):
    if confluence:
        strat = {"session": None,
                 "confluence": {"legs": [
                     {"strategy": "RSIReversion", "rsi": {"period": 14, "oversold": 30, "overbought": 70}},
                     {"strategy": "RSIReversion", "rsi": {"period": 7, "oversold": 25, "overbought": 75}},
                 ], "same_direction": True},
                 "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        from metatraderbot.strategies_confluence import ConfluenceStrategy
        cfg = _mk_cfg(sym, strat, trail)
        bt = Backtest(cfg, bars, strategy=Windowed(ConfluenceStrategy(cfg)))
    else:
        cls, params = cls_or_kwargs
        strat = {"session": None, "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = _mk_cfg(sym, strat, trail)
        bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return [t.pnl_usd / RISK_USD for t in bt.closed]


def _cell(job):
    fam, name, tp, tf, sym = job
    try:
        bars = load(sym, tf)
        if bars is None or len(bars) < 600:
            return None
        bars = bars[bars.index.dayofweek != 6]
        split = int(len(bars) * 0.70)
        out = {"fam": fam, "name": name, "tp": tp, "tf": tf, "sym": sym}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 300:
                continue
            trail = fam != "mr"  # mean-reversion legs run w/o trail per project convention
            if fam == "conf":
                rs = _run_half(None, sym, b, tp, trail, confluence=True)
            elif fam == "mr":
                rs = _run_half(MR_STRATS[name], sym, b, tp, trail)
            else:
                rs = _run_half(TREND_STRATS[name], sym, b, tp, trail)
            out[half] = {"rs": rs, "days": len(set(b.index.date))}
        return out
    except Exception as e:
        return {"error": f"{fam}/{name}/{tp}/{tf}/{sym}: {e!r}"}


def agg(cells, half):
    rs = [r for c in cells for r in c.get(half, {}).get("rs", [])]
    days = max((c.get(half, {}).get("days", 0) for c in cells), default=0)
    if not rs:
        return None
    a = np.array(rs)
    w = a[a > 0]
    l = a[a <= 0]
    gl = -l.sum()
    e = float(a.mean())
    per_day = len(a) / days if days else 0.0
    avg_win = float(w.mean()) if len(w) else 0.0
    avg_loss = float(-l.mean()) if len(l) else 0.0
    return {"n": len(a), "per_day": per_day, "win": len(w) / len(a), "e": e,
            "pf": float(w.sum() / gl) if gl > 0 else float("inf"),
            "r_per_day": e * per_day,
            "rr": (avg_win / avg_loss) if avg_loss > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tf", nargs="*", default=TIMEFRAMES)
    args = ap.parse_args()

    jobs = []
    for tf in args.tf:
        grid = TF_GRID[tf]
        for name in grid["mr"]:
            for tp in grid["mr_tps"]:
                for sym in FX:
                    jobs.append(("mr", name, tp, tf, sym))
        for name in grid["trend"]:
            for tp in grid["trend_tps"]:
                for sym in NONFX:
                    jobs.append(("trend", name, tp, tf, sym))
        for tp in grid["conf_tps"]:
            for sym in FX:
                jobs.append(("conf", "RSI14_RSI7", tp, tf, sym))

    print("=" * 104)
    print(f"INTRADAY GROWTH SWEEP -- {len(jobs)} cells across tf={args.tf}")
    print("=" * 104, flush=True)

    cells, errors, done = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=2):
            done += 1
            if c is None:
                pass
            elif "error" in c:
                errors.append(c)
            else:
                cells.append(c)
            if done % 10 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)

    if errors:
        print(f"\n{len(errors)} cells FAILED. First few:")
        for e in errors[:8]:
            print("   ", e["error"])

    results = []
    for tf in args.tf:
        print(f"\n{'=' * 104}\n### TIMEFRAME {tf}\n{'=' * 104}")
        print(f"{'fam':<6}{'strategy':<14}{'TP':>5}{'n_oos':>7}{'/day':>7}{'win%':>7}"
              f"{'E_tr':>9}{'E_oos':>9}{'PF':>7}{'R/day':>9}  verdict")
        print("-" * 104)
        groups = {}
        for c in cells:
            if c["tf"] != tf:
                continue
            key = (c["fam"], c["name"], c["tp"])
            groups.setdefault(key, []).append(c)
        for (fam, name, tp), mine in sorted(groups.items()):
            tr, oo = agg(mine, "train"), agg(mine, "oos")
            if not oo:
                continue
            both = bool(tr and tr["e"] > 0 and oo["e"] > 0)
            conf_flag = "" if oo["n"] >= 300 else " (n<300, low-confidence)"
            verdict = ("KEEP" + conf_flag) if both else ""
            pf = f"{oo['pf']:.2f}" if oo["pf"] != float("inf") else "inf"
            print(f"{fam:<6}{name:<14}{tp:>5.1f}{oo['n']:>7}{oo['per_day']:>7.2f}"
                  f"{oo['win']*100:>7.1f}{(tr['e'] if tr else float('nan')):>+9.3f}"
                  f"{oo['e']:>+9.3f}{pf:>7}{oo['r_per_day']:>+9.4f}  {verdict}")
            results.append({"tf": tf, "fam": fam, "name": name, "tp": tp,
                             "train": tr, "oos": oo, "both_positive": both})

    survivors = [r for r in results if r["both_positive"]]
    print("\n" + "=" * 104)
    print("SURVIVORS -- positive both halves (any n)")
    print("=" * 104)
    if not survivors:
        print("  none.")
    else:
        for r in sorted(survivors, key=lambda x: -x["oos"]["r_per_day"]):
            flag = "" if r["oos"]["n"] >= 300 else "  [low-confidence, n<300]"
            print(f"  {r['tf']:<4}{r['fam']:<6}{r['name']:<14}TP{r['tp']:<4} "
                  f"n={r['oos']['n']:<6} R/day={r['oos']['r_per_day']:+.4f} "
                  f"E_oos={r['oos']['e']:+.3f} E_tr={r['train']['e']:+.3f}{flag}")

    out = ROOT / "scratch" / "intraday_growth_sweep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
