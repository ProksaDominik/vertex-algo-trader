"""Does picking the profitable symbols actually work?

The intuitive move after seeing per-symbol results is to drop the losers and
trade only the winners. That is only correct if a symbol's past profitability
PREDICTS its future profitability. This measures whether it does, instead of
assuming it either way.

Method (no hindsight anywhere):
  1. Split the 10.5-year history into an EARLY half and a LATE half.
  2. Rank symbols purely on EARLY-half results.
  3. Trade the early-half winners in the LATE half and see what happens.
  4. Compare against trading the whole basket, and against the early-half
     LOSERS, over that same late half.

If selection works, the early winners beat the full basket later. If symbol
performance is mostly noise, the winners regress to the mean and the whole
exercise just shrinks the sample for nothing.

Also reports the standard error on each per-symbol estimate, which is the
quickest way to see whether the ranking is measuring skill or sampling noise.

Usage:
    python scripts/winner_picking_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import BASKET, RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402


def trades_for(sym: str, bars: pd.DataFrame) -> list[float]:
    cfg = {
        "symbol": sym,
        "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                    "max_positions": 2, "daily_stop_loss_usd": -1e9},
        "strategy": {"rsi": {"period": 14, "oversold": 30, "overbought": 70},
                     "session": None,
                     "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10}},
    }
    bt = Backtest(cfg, bars, strategy=Windowed(RSIReversionStrategy(cfg)))
    bt.run()
    return [t.pnl_usd / RISK_USD for t in bt.closed]


def summarize(rs: list[float]) -> str:
    if not rs:
        return "no trades"
    e = float(np.mean(rs))
    se = float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    return (f"n={len(rs):<5} E_net={e:+.3f}R  (+/-{se:.3f} SE)  "
            f"total={sum(rs):+.1f}R  win={sum(1 for x in rs if x > 0)/len(rs):.1%}")


def main() -> int:
    print("=" * 100)
    print("DOES PICKING THE PROFITABLE SYMBOLS WORK?")
    print("Rank on the EARLY half, then trade that ranking in the LATE half.")
    print("=" * 100)

    early: dict[str, list[float]] = {}
    late: dict[str, list[float]] = {}
    for s in BASKET:
        bars = load(s, "1d")
        if bars is None:
            continue
        mid = len(bars) // 2
        try:
            early[s] = trades_for(s, bars.iloc[:mid])
            late[s] = trades_for(s, bars.iloc[mid:])
        except Exception as e:
            print(f"  [skip] {s}: {e!r}")

    syms = [s for s in early if early[s] and late[s]]
    print(f"\n{len(syms)} symbols with trades in both halves\n")

    print("--- per-symbol EARLY half (the information you would actually have) ---")
    print(f"{'symbol':<9}{'early':<52}")
    ranked = sorted(syms, key=lambda s: -sum(early[s]))
    for s in ranked:
        print(f"{s:<9}{summarize(early[s])}")

    # Note the standard errors: if SE is comparable to the spread in E_net,
    # the ranking is mostly noise.
    spread = np.std([np.mean(early[s]) for s in ranked])
    mean_se = np.mean([np.std(early[s], ddof=1) / np.sqrt(len(early[s]))
                       for s in ranked])
    print(f"\n  spread of per-symbol E_net across the basket : {spread:.3f}R")
    print(f"  average standard error of one symbol estimate: {mean_se:.3f}R")
    if mean_se >= spread * 0.8:
        print("  -> the ranking spread is roughly the size of its own error bars:")
        print("     the ordering is mostly SAMPLING NOISE, not symbol skill.")

    half = max(1, len(ranked) // 2)
    winners, losers = ranked[:half], ranked[half:]
    print(f"\n  early winners: {', '.join(winners)}")
    print(f"  early losers : {', '.join(losers)}")

    print("\n" + "=" * 100)
    print("--- what happens NEXT (late half) ---")
    print("=" * 100)
    w_late = [r for s in winners for r in late[s]]
    l_late = [r for s in losers for r in late[s]]
    a_late = [r for s in syms for r in late[s]]
    print(f"  trade ONLY early winners : {summarize(w_late)}")
    print(f"  trade ONLY early losers  : {summarize(l_late)}")
    print(f"  trade the WHOLE basket   : {summarize(a_late)}")

    if w_late and a_late:
        diff = np.mean(w_late) - np.mean(a_late)
        print(f"\n  >>> selecting winners changed expectancy by {diff:+.3f}R/trade")
        print(f"      and cut the sample from {len(a_late)} to {len(w_late)} trades")
        if diff <= 0:
            print("      -> selection made it WORSE. The early ranking did not persist.")
        else:
            print("      -> selection helped, but check it against the SE above.")

    # Rank persistence: does early rank predict late rank at all?
    e_rank = pd.Series({s: np.mean(early[s]) for s in syms}).rank()
    l_rank = pd.Series({s: np.mean(late[s]) for s in syms}).rank()
    rho = e_rank.corr(l_rank, method="spearman")
    print(f"\n  Spearman rank correlation early vs late: {rho:+.3f}")
    print("  (+1 = past winners stay winners, 0 = past tells you nothing,")
    print("   negative = past winners tend to become losers)")

    print("\n--- per-symbol, both halves side by side ---")
    print(f"{'symbol':<9}{'early E_net':>13}{'late E_net':>13}{'flipped?':>11}")
    for s in ranked:
        e, l = np.mean(early[s]), np.mean(late[s])
        flip = "YES" if (e > 0) != (l > 0) else ""
        print(f"{s:<9}{e:>+13.3f}{l:>+13.3f}{flip:>11}")
    flips = sum(1 for s in syms if (np.mean(early[s]) > 0) != (np.mean(late[s]) > 0))
    print(f"\n  >>> {flips}/{len(syms)} symbols FLIPPED sign between halves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
