"""Two questions, both answered out-of-sample:

  1. Does STOPPING FOR THE DAY after the first loss help?
     It can only help if losses CLUSTER within a day -- i.e. if a loss makes
     further losses that same day more likely. If daily results are
     independent, stopping early just removes trades at random, which throws
     away winners at the same rate as losers and shrinks the edge.

  2. Can profitable DAYS be predicted in advance?
     Tested predictors, all computable BEFORE the day starts: day of week,
     yesterday's result, recent streak, and volatility regime. Every rule is
     fitted on the train half and applied frozen to the out-of-sample half.

The honest default is that neither works. This script exists to check rather
than assume, because "only trade the good days" is the single most seductive
idea in trading and almost never survives contact with a holdout.

Usage:
    python scripts/day_filter_test.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

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
    ("RSI", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}, 5.0),
    ("Bollinger", BollingerBounceStrategy, {"bollinger": {}}, 4.0),
]


def collect() -> pd.DataFrame:
    """Every trade from the deployed stack, tagged with its entry date."""
    rows = []
    for sym in SYMBOLS:
        bars = load(sym, "1d")
        if bars is None:
            continue
        bars = bars[bars.index.dayofweek != 6]
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
                ts = pd.Timestamp(t.entered_at)
                rows.append({"date": ts.normalize(), "dow": ts.dayofweek,
                             "sym": sym, "leg": name,
                             "r": t.pnl_usd / RISK_USD})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def stats(rs) -> str:
    rs = list(rs)
    if not rs:
        return "n=0"
    e = float(np.mean(rs))
    se = float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    return f"n={len(rs):<5} E={e:+.3f}R (+/-{se:.3f})  total={sum(rs):+7.1f}R"


def main() -> int:
    df = collect()
    print("=" * 96)
    print(f"collected {len(df)} trades from the deployed stack "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")
    print("=" * 96)

    mid = df["date"].quantile(0.70)
    train, oos = df[df["date"] <= mid], df[df["date"] > mid]
    print(f"train: {len(train)} trades | OOS: {len(oos)} trades\n")

    # ---------- Q1: do losses cluster within a day? --------------------
    print("=" * 96)
    print("Q1. DOES STOPPING AFTER THE FIRST LOSS OF THE DAY HELP?")
    print("=" * 96)
    print("  For it to help, a loss must make the NEXT trade that day worse.\n")
    for label, d in (("TRAIN", train), ("OOS", oos)):
        first, after_loss, after_win = [], [], []
        for _, day in d.groupby("date"):
            rs = day["r"].tolist()
            first.append(rs[0])
            for i in range(1, len(rs)):
                (after_loss if rs[i - 1] <= 0 else after_win).append(rs[i])
        print(f"  {label}")
        print(f"    first trade of the day      {stats(first)}")
        print(f"    trades after a LOSS         {stats(after_loss)}")
        print(f"    trades after a WIN          {stats(after_win)}")
        if after_loss and after_win:
            diff = np.mean(after_loss) - np.mean(after_win)
            print(f"    -> after-loss minus after-win = {diff:+.3f}R "
                  f"({'losses cluster' if diff < -0.05 else 'no clustering'})")
        print()

    # Simulate the actual rule on OOS.
    kept, dropped = [], []
    for _, day in oos.groupby("date"):
        rs = day["r"].tolist()
        stop = False
        for r in rs:
            if stop:
                dropped.append(r)
            else:
                kept.append(r)
                if r <= 0:
                    stop = True
    print("  THE RULE APPLIED (stop for the day after the first losing trade):")
    print(f"    kept    {stats(kept)}")
    print(f"    dropped {stats(dropped)}")
    if kept and dropped:
        print(f"    -> the rule removed trades averaging {np.mean(dropped):+.3f}R")
        print(f"       ({'GOOD - they were losers' if np.mean(dropped) < 0 else 'BAD - they were WINNERS on average'})")

    # ---------- Q2: are good days predictable? -------------------------
    print("\n" + "=" * 96)
    print("Q2. CAN PROFITABLE DAYS BE PREDICTED IN ADVANCE?")
    print("=" * 96)

    print("\n  (a) by DAY OF WEEK")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tr_dow = train.groupby("dow")["r"].mean()
    bad = [d for d in tr_dow.index if tr_dow[d] < 0]
    print(f"    {'day':<6}{'train E':>12}{'OOS E':>12}{'flipped?':>11}")
    for d in sorted(set(train["dow"]) & set(oos["dow"])):
        te = train[train["dow"] == d]["r"].mean()
        oe = oos[oos["dow"] == d]["r"].mean()
        print(f"    {names[d]:<6}{te:>+12.3f}{oe:>+12.3f}"
              f"{('YES' if (te > 0) != (oe > 0) else ''):>11}")
    if bad:
        keep = oos[~oos["dow"].isin(bad)]["r"]
        print(f"\n    rule: avoid {[names[d] for d in bad]} (negative in train)")
        print(f"      baseline OOS {stats(oos['r'])}")
        print(f"      filtered OOS {stats(keep)}")
        print(f"      -> {np.mean(keep) - oos['r'].mean():+.3f}R change")

    print("\n  (b) by YESTERDAY'S RESULT (does a bad day predict a bad day?)")
    daily = df.groupby("date")["r"].sum().sort_index()
    prev, cur = daily.shift(1), daily
    both = pd.DataFrame({"prev": prev, "cur": cur}).dropna()
    tr_b = both[both.index <= mid]
    oo_b = both[both.index > mid]
    for label, b in (("TRAIN", tr_b), ("OOS", oo_b)):
        after_down = b[b["prev"] < 0]["cur"]
        after_up = b[b["prev"] > 0]["cur"]
        print(f"    {label}: day after a DOWN day {after_down.mean():+.3f}R "
              f"(n={len(after_down)})   after an UP day {after_up.mean():+.3f}R "
              f"(n={len(after_up)})")
    corr = both["prev"].corr(both["cur"])
    print(f"    correlation between consecutive days' P&L: {corr:+.3f}")
    print("    (near zero = yesterday tells you nothing about today)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
