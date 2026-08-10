"""THE canonical-strategy test: well-known strategies, textbook settings,
multi-year data, multi-timeframe, honest out-of-sample split.

Why this script exists
----------------------
Every prior comparison in this repo ran on ONE calendar year (2024) of 15m
bars, mostly at 30m. Both of those choices work against the strategies that
public evidence and academic work actually support: classic trend/momentum
edges (Turtle/Donchian breakout, time-series momentum) live on H1/H4/D1 and
need multiple market regimes to be distinguishable from noise.

So this runs the WELL-KNOWN set at their TEXTBOOK settings across ~10 years
(`data/<sym>_15m_long.parquet`, see scripts/download_history.py) on H1, H4
and D1, with a strict chronological train/OOS split and the project's real
cost model (per-pair spread + commission from instruments.PAIR_SPECS).

Deliberately NOT a parameter sweep. Each strategy is tested at the settings
it is famous for, plus (for Donchian) the two classic filter variants. A
sweep over one dataset is how you manufacture a false positive; the whole
point here is to see which ideas survive without tuning.

Usage:
    python scripts/canonical_strategy_test.py                # all TFs
    python scripts/canonical_strategy_test.py --tf 4h        # one TF
    python scripts/canonical_strategy_test.py --oos-frac 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Each worker is single-threaded by design: the parallelism here is across
# (strategy, symbol) cells, so letting OpenBLAS also spin a thread pool per
# process just multiplies memory until allocation fails. Must be set BEFORE
# numpy/pandas is imported, in both parent and (spawned) children.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_smc_bos import SMCBOSStrategy  # noqa: E402
from metatraderbot.strategies_sr_retest import SRBreakoutRetestStrategy  # noqa: E402
from metatraderbot.strategies_stochastic import StochasticStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

RISK_USD = 100.0  # 1R = $100, so pnl/RISK_USD reads directly as R multiples

BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "EURGBP", "XAUUSD", "US500",
]

TIMEFRAMES = ["1h", "4h", "1d"]

# --- The canonical roster -------------------------------------------------
# (label, class, strategy_params). Settings are the textbook/classic ones.
CANDIDATES: list[tuple[str, type, dict]] = [
    # Turtle: the original 20-bar breakout, full opposite-channel stop.
    ("Donchian20_turtle", DonchianBreakoutStrategy, {
        "donchian": {"period": 20}, "exit": {"tp_r_multiple": 2.0}}),
    # Turtle slow system (55-bar) — the other half of the original rules.
    ("Donchian55_turtle", DonchianBreakoutStrategy, {
        "donchian": {"period": 55}, "exit": {"tp_r_multiple": 2.0}}),
    # The modern consensus fix for 20-bar breakouts: a 200-EMA trend filter.
    ("Donchian20_ema200", DonchianBreakoutStrategy, {
        "donchian": {"period": 20, "trend_ema_period": 200},
        "exit": {"tp_r_multiple": 2.0}}),
    # Trend filter + require the break to CLEAR the channel decisively.
    ("Donchian20_ema200_atr", DonchianBreakoutStrategy, {
        "donchian": {"period": 20, "trend_ema_period": 200,
                     "min_breakout_atr_mult": 0.5},
        "exit": {"tp_r_multiple": 2.0}}),
    # Same, but an ATR stop instead of the very wide channel stop.
    ("Donchian20_ema200_atrstop", DonchianBreakoutStrategy, {
        "donchian": {"period": 20, "trend_ema_period": 200,
                     "sl_mode": "atr", "sl_atr_mult": 2.0},
        "exit": {"tp_r_multiple": 2.0}}),
    ("Donchian55_ema200", DonchianBreakoutStrategy, {
        "donchian": {"period": 55, "trend_ema_period": 200},
        "exit": {"tp_r_multiple": 2.0}}),
    # Incumbent (Phase 11) — carried for a like-for-like comparison.
    ("Supertrend_10_3", SupertrendStrategy, {
        "supertrend": {"period": 10, "multiplier": 3.0},
        "exit": {"tp_r_multiple": 2.0}}),
    ("MACD_classic", MACDTrendStrategy, {
        "macd": {}, "exit": {"tp_r_multiple": 2.0}}),
    ("EMACross_20_50", EMACrossoverStrategy, {
        "ema_cross": {"fast_period": 20, "slow_period": 50},
        "exit": {"tp_r_multiple": 2.0}}),
    ("SMACross", SMACrossStrategy, {
        "ma_cross": {}, "exit": {"tp_r_multiple": 2.0}}),
    ("Ichimoku", IchimokuStrategy, {
        "ichimoku": {}, "exit": {"tp_r_multiple": 2.0}}),
    ("SRRetest", SRBreakoutRetestStrategy, {
        "sr_retest": {}, "exit": {"tp_r_multiple": 2.0}}),
    ("SMC_BOS", SMCBOSStrategy, {
        "smc_bos": {}, "exit": {"tp_r_multiple": 1.5}}),
    # Mean-reversion family: tight target, no trail (their standard form).
    ("Bollinger", BollingerBounceStrategy, {
        "bollinger": {}, "exit": {"tp_r_multiple": 1.0}}),
    ("RSIReversion", RSIReversionStrategy, {
        "rsi": {}, "exit": {"tp_r_multiple": 1.0}}),
    ("Stochastic", StochasticStrategy, {
        "stochastic": {}, "exit": {"tp_r_multiple": 1.0}}),
]

# Mean-reversion strategies run without a trailing stop (see STRATEGIES.md).
NO_TRAIL = {"Bollinger", "RSIReversion", "Stochastic"}


class Windowed:
    """Feed each strategy a rolling window so per-bar cost stays bounded.

    350 bars is comfortably more than the longest lookback in the roster
    (200-EMA + 55-bar channel), so no strategy is starved of history.
    """

    WINDOW = 400

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, i, bars):
        start = max(0, i - self.WINDOW + 1)
        v = bars.iloc[start:i + 1].reset_index(drop=True)
        return self.inner.on_bar(len(v) - 1, v)


def load(sym: str, tf: str) -> pd.DataFrame | None:
    """Multi-year 15m bars resampled to `tf`.

    Falls back to the legacy 2024-only file ONLY when --allow-short is set:
    pooling a 10-year symbol with a 1-year one silently weights the basket
    toward whichever regime the short symbol happens to sit in.
    """
    f = ROOT / "data" / f"{sym.lower()}_15m_long.parquet"
    if not f.exists():
        if not _ALLOW_SHORT:
            return None
        f = ROOT / "data" / f"{sym.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    out = df.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out if len(out) > 500 else None


def cfg_for(sym: str, params: dict, trail: bool) -> dict:
    # sl_swing_lookback is required by the MA/MACD/RSI/Stochastic family
    # (they stop at the recent swing); the others ignore it harmlessly.
    strat: dict = {"session": None,
                   "exit": {"tp_r_multiple": 2.0, "sl_swing_lookback": 10}}
    for k, v in params.items():
        if k == "exit":
            strat["exit"].update(v)
        else:
            strat[k] = v
    cap = {
        "risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
        "max_positions": 2, "daily_stop_loss_usd": -1e9,
    }
    if trail:
        cap["trail_start_r"] = 1.0
        cap["trail_lock_fraction"] = 0.5
    return {"symbol": sym, "capital": cap, "strategy": strat}


def run_one(cls, params: dict, sym: str, bars: pd.DataFrame, trail: bool) -> list[float]:
    cfg = cfg_for(sym, params, trail)
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return [t.pnl_usd / RISK_USD for t in bt.closed]


# --- Parallel plumbing ----------------------------------------------------
# One (strategy, symbol, timeframe) cell per worker task. Each worker reloads
# its own bars from parquet: sending frames through pickle costs more than
# re-reading them, and it keeps memory flat across 12 processes.

_CLASS_BY_LABEL = {label: cls for label, cls, _ in CANDIDATES}
_PARAMS_BY_LABEL = {label: params for label, _, params in CANDIDATES}


def _cell(job: tuple[str, str, str]) -> dict:
    """Run one (timeframe, strategy, symbol) cell. Returns its OOS R-list."""
    tf, label, sym = job
    try:
        bars = load(sym, tf)
        if bars is None:
            return {"tf": tf, "strategy": label, "sym": sym, "rs": [], "days": 0, "skip": True}
        split = int(len(bars) * (1 - _OOS_FRAC))
        oos = bars.iloc[split:]
        if len(oos) < 300:
            return {"tf": tf, "strategy": label, "sym": sym, "rs": [], "days": 0, "skip": True}
        rs = run_one(_CLASS_BY_LABEL[label], _PARAMS_BY_LABEL[label], sym, oos,
                     label not in NO_TRAIL)
        return {"tf": tf, "strategy": label, "sym": sym, "rs": rs,
                "days": len(set(oos.index.date)), "skip": False}
    except Exception as e:
        return {"tf": tf, "strategy": label, "sym": sym, "rs": [], "days": 0,
                "skip": True, "error": repr(e)}


_OOS_FRAC = 0.30      # both rebound in main() before the pool spawns
_ALLOW_SHORT = False


def _init_worker(oos_frac: float, allow_short: bool) -> None:
    global _OOS_FRAC, _ALLOW_SHORT
    _OOS_FRAC = oos_frac
    _ALLOW_SHORT = allow_short


def stats(rs: list[float], trading_days: int) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gross_loss = -sum(losses)
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    ser = pd.Series(rs)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    return {
        "n": n,
        "per_day": n / trading_days if trading_days else 0.0,
        "win_rate": len(wins) / n,
        # Reward:risk of the AVERAGE win vs the AVERAGE loss — together with
        # win_rate this is exactly the (W, RR) pair research/mc3.py needs to
        # simulate FTMO pass probability.
        "rr": (avg_win / avg_loss) if avg_loss > 0 else 0.0,
        "e_net": sum(rs) / n,
        "total_r": sum(rs),
        "pf": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
        "max_dd_r": max_dd,
        # Per-trade Sharpe — comparable across strategies with different counts.
        "sharpe": (ser.mean() / ser.std()) if n > 2 and ser.std() > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=TIMEFRAMES)
    ap.add_argument("--oos-frac", type=float, default=0.30)
    ap.add_argument("--out", default="backtest/canonical_results.json")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--allow-short", action="store_true",
                    help="also use legacy 2024-only files for symbols with no long history")
    args = ap.parse_args()

    global _ALLOW_SHORT
    _ALLOW_SHORT = args.allow_short

    print("=" * 104)
    print("CANONICAL WELL-KNOWN STRATEGY TEST — multi-year, multi-timeframe, out-of-sample")
    print(f"basket={len(BASKET)} symbols   OOS = last {args.oos_frac:.0%} of history   1R = ${RISK_USD:.0f}")
    print("=" * 104)

    jobs = [(tf, label, sym) for tf in args.tf for label, _, _ in CANDIDATES for sym in BASKET]
    workers = min(args.workers, os.cpu_count() or 4)
    print(f"\nRunning {len(jobs)} (timeframe x strategy x symbol) cells on {workers} workers...\n",
          flush=True)

    cells: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(args.oos_frac, args.allow_short)) as ex:
        for cell in ex.map(_cell, jobs, chunksize=4):
            cells.append(cell)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} cells", flush=True)

    for err in [c for c in cells if c.get("error")][:5]:
        print(f"  [ERROR] {err['strategy']} {err['sym']} {err['tf']}: {err['error']}", flush=True)

    results = []
    for tf in args.tf:
        tf_cells = [c for c in cells if c["tf"] == tf and not c.get("skip")]
        if not tf_cells:
            print(f"\n### {tf}: no data")
            continue

        n_syms = len({c["sym"] for c in tf_cells})
        print(f"\n{'=' * 104}\n### TIMEFRAME {tf}   ({n_syms} symbols)\n{'=' * 104}")
        print(f"{'strategy':<28}{'n':>7}{'/day':>7}{'win%':>7}{'E_net':>9}{'PF':>7}"
              f"{'totR':>9}{'maxDD':>8}{'Sharpe':>8}{'sym+':>7}")
        print("-" * 104)

        rows = []
        for label, _, _ in CANDIDATES:
            mine = [c for c in tf_cells if c["strategy"] == label]
            if not mine:
                continue
            oos_rs = [r for c in mine for r in c["rs"]]
            pos_syms = sum(1 for c in mine if c["rs"] and sum(c["rs"]) > 0)
            tested_syms = len(mine)
            oos_days = max((c["days"] for c in mine), default=0)

            st = stats(oos_rs, oos_days)
            if st["n"] == 0:
                print(f"{label:<28}{'0':>7}   (no trades)")
                continue
            pf_s = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
            print(f"{label:<28}{st['n']:>7}{st['per_day']:>7.2f}{st['win_rate']*100:>7.1f}"
                  f"{st['e_net']:>+9.3f}{pf_s:>7}{st['total_r']:>+9.1f}"
                  f"{st['max_dd_r']:>8.1f}{st['sharpe']:>8.3f}{f'{pos_syms}/{tested_syms}':>7}")
            row = {"tf": tf, "strategy": label, "pos_syms": pos_syms,
                   "tested_syms": tested_syms,
                   "per_symbol": {c["sym"]: round(sum(c["rs"]), 2) for c in mine if c["rs"]},
                   **st}
            rows.append(row)
            results.append(row)

        if rows:
            best = sorted(rows, key=lambda r: r["e_net"], reverse=True)[:3]
            print(f"\n  best by OOS expectancy @ {tf}: " +
                  ", ".join(f"{r['strategy']} ({r['e_net']:+.3f}R, PF {r['pf']:.2f})" for r in best))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {len(results)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
