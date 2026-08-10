"""What the DEPLOYED FTMO config actually does: trades/day and win rate.

The headline numbers quoted during development came from the raw strategy
sweep. The live book adds filters that change both figures:

  * skip_weekdays [6]      -- no entries on the partial Sunday bar
  * daily_stop_action block -- no NEW entries once the day is down 1R
  * max_trades_per_day 2

This applies those in order and reports the result per leg and combined, so
the deployed expectation is measured rather than inherited from an earlier,
looser test.

NOT modelled here: cost_gate_r 0.04 (a runner-side gate with no equivalent in
the Backtest class) and the 2-12x live slippage gap measured by
scripts/daily_review.py. Both push the real numbers DOWN, so treat what
follows as an optimistic bound.

Usage:
    python scripts/deployed_stats.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
LEGS = [
    ("Bollinger_Daily", BollingerBounceStrategy, {"bollinger": {}}, 4.0),
    ("RSI_Daily", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 5.0),
]
MAX_TRADES_DAY = 2
DAILY_STOP_R = 1.0


def collect() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        bars = load(sym, "1d")
        if bars is None:
            continue
        bars = bars[bars.index.dayofweek != 6]        # skip_weekdays [6]
        for name, cls, params, tp in LEGS:
            strat = {"session": None,
                     "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
            strat.update(params)
            cfg = {"symbol": sym,
                   "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                               "max_positions": 2, "daily_stop_loss_usd": -1e9},
                   "strategy": strat}
            try:
                bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
                bt.run()
            except Exception:
                continue
            for t in bt.closed:
                rows.append({"date": pd.Timestamp(t.entered_at).normalize(),
                             "leg": name, "sym": sym,
                             "r": t.pnl_usd / RISK_USD})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def describe(rs, days, label):
    rs = list(rs)
    if not rs:
        print(f"  {label:<28} no trades")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gl = -sum(losses)
    print(f"  {label:<28}{len(rs):>7}{len(rs)/days:>9.2f}"
          f"{len(wins)/len(rs)*100:>8.1f}{float(np.mean(rs)):>+9.3f}"
          f"{(sum(wins)/gl if gl > 0 else float('inf')):>7.2f}"
          f"{sum(rs):>+9.1f}")


def main() -> int:
    df = collect()
    n_days = df["date"].nunique()
    span_days = (df["date"].max() - df["date"].min()).days
    trading_days = span_days * 5 / 7          # weekdays in the span

    print("=" * 92)
    print("DEPLOYED FTMO CONFIG -- measured over the full 10.5y history")
    print(f"{len(SYMBOLS)} FX symbols | Sundays excluded | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    print("=" * 92)
    print(f"  {'':<28}{'n':>7}{'/day':>9}{'win%':>8}{'E_net':>9}{'PF':>7}{'totR':>9}")
    print("-" * 92)

    print("\n  RAW (every signal, no daily stop or per-day cap):")
    for name, *_ in LEGS:
        describe(df[df["leg"] == name]["r"], trading_days, name)
    describe(df["r"], trading_days, "COMBINED")

    # --- apply the deployed gating, chronologically ---------------------
    kept_rows = []
    for _, day in df.groupby("date"):
        day_r = 0.0
        taken = 0
        for _, tr in day.iterrows():
            if taken >= MAX_TRADES_DAY:
                break
            if day_r <= -DAILY_STOP_R:          # daily stop: block new entries
                break
            kept_rows.append(tr)
            day_r += tr["r"]
            taken += 1
    kept = pd.DataFrame(kept_rows)

    print("\n  DEPLOYED (daily stop at -1R + max 2 trades/day):")
    for name, *_ in LEGS:
        describe(kept[kept["leg"] == name]["r"], trading_days, name)
    describe(kept["r"], trading_days, "COMBINED")

    print("\n" + "=" * 92)
    raw_e, dep_e = df["r"].mean(), kept["r"].mean()
    print(f"  trades/day  {len(df)/trading_days:.2f} raw  ->  "
          f"{len(kept)/trading_days:.2f} deployed "
          f"({len(kept)/len(df):.0%} of signals survive the gating)")
    print(f"  win rate    {(df['r'] > 0).mean()*100:.1f}%  ->  "
          f"{(kept['r'] > 0).mean()*100:.1f}%")
    print(f"  E_net       {raw_e:+.3f}R  ->  {dep_e:+.3f}R")
    print(f"  R per day   {raw_e*len(df)/trading_days:+.4f}  ->  "
          f"{dep_e*len(kept)/trading_days:+.4f}")
    print("\n  Reminder: the cost gate (0.04) and the 2-12x live slippage gap")
    print("  are NOT in these figures. Both push the real numbers down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
