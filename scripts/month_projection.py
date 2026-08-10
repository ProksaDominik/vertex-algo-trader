"""One-month dollar projection for both books, from the MEASURED backtest edge.

Answers "what would a month look like?" the only way that is honest: as a
distribution. A single expected-value number hides that the same edge which
averages +$X also produces a losing month a large fraction of the time.

Two things this deliberately does NOT do:
  - It does not assume the backtested edge is real. It is a projection OF the
    backtest, and the backtest's robustness audit is still running.
  - It does not model swap (unmodelled in the cost stack) or slippage beyond
    the spread already in Backtest._costs().

It DOES check the thing that actually decides whether the $200 book can run
a daily strategy at all: the broker minimum lot (0.01) against the strategy's
real stop widths. If one minimum-lot position already risks more than the
book's target risk, the risk setting is fiction and the real risk is whatever
the lot floor imposes.

Usage:
    python scripts/month_projection.py
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

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

TRADING_DAYS = 21          # one month
MIN_LOT = 0.01             # standard MT5 broker minimum
N_SIMS = 40_000

# Measured out-of-sample stats (backtest/canonical_majors.json, 10.5y, 9 syms)
EDGES = {
    "RSI daily only":      {"win": 0.612, "rr": 0.94, "per_day_9sym": 0.25},
    "Daily stack (3 strat)": {"win": 0.534, "rr": 1.00, "per_day_9sym": 1.40},
}
SYMBOL_SCALE = 20 / 9      # deploying on 20 symbols vs the 9 measured


def measure_stop_widths() -> pd.DataFrame:
    """Actual RSI-daily stop distances -> $ risk at the 0.01 lot floor."""
    rows = []
    for sym in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GBPJPY", "EURJPY"]:
        bars = load(sym, "1d")
        if bars is None:
            continue
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
        if not bt.closed:
            continue
        mult = PAIR_SPECS[sym]["contract_mult"]
        dists = [abs(t.entry_price - t.stop_loss) for t in bt.closed]
        med = float(np.median(dists))
        rows.append({
            "symbol": sym,
            "median_stop_price": med,
            "risk_at_min_lot": med * mult * MIN_LOT,
            "n": len(dists),
        })
    return pd.DataFrame(rows)


def simulate(win: float, rr: float, n_trades: int, risk_frac: float,
             start: float, rng) -> np.ndarray:
    """Compound `n_trades` bernoulli outcomes, return end equity per sim."""
    eq = np.full(N_SIMS, start)
    for _ in range(n_trades):
        stake = eq * risk_frac
        wins = rng.random(N_SIMS) < win
        eq = eq + np.where(wins, stake * rr, -stake)
        eq = np.maximum(eq, 0.0)
    return eq


def report(title: str, start: float, risk_frac: float, win: float, rr: float,
           n_trades: int, rng, ftmo: bool = False) -> None:
    end = simulate(win, rr, n_trades, risk_frac, start, rng)
    profit = end - start
    pct = np.percentile(profit, [5, 25, 50, 75, 95])
    print(f"\n  {title}")
    print(f"    {n_trades} trades in {TRADING_DAYS} trading days, "
          f"risk {risk_frac*100:.2f}%/trade, win {win:.1%}, RR {rr:.2f}")
    print(f"    mean profit   ${profit.mean():>10,.2f}")
    print(f"    median profit ${pct[2]:>10,.2f}")
    print(f"    5th pct       ${pct[0]:>10,.2f}      95th pct  ${pct[4]:>10,.2f}")
    print(f"    25th pct      ${pct[1]:>10,.2f}      75th pct  ${pct[3]:>10,.2f}")
    print(f"    P(month is profitable) = {(profit > 0).mean():.1%}")
    if ftmo:
        print(f"    P(hit +10% target = +${start*0.10:,.0f}) = {(profit >= start*0.10).mean():.1%}")
        print(f"    P(breach -10% max loss)               = {(profit <= -start*0.10).mean():.1%}")
    else:
        print(f"    P(lose >half the account) = {(profit <= -start*0.5).mean():.1%}")


def main() -> int:
    rng = np.random.default_rng(11)
    print("=" * 88)
    print("ONE-MONTH PROJECTION — from the measured out-of-sample edge")
    print(f"{TRADING_DAYS} trading days | {N_SIMS:,} Monte Carlo paths")
    print("=" * 88)

    print("\n### FIRST: can a $200 account even trade this? (lot-floor check)")
    df = measure_stop_widths()
    if not df.empty:
        print(f"\n{'symbol':<9}{'median stop':>14}{'$ risk at 0.01 lot':>21}"
              f"{'% of $200':>12}{'% of $10k':>12}")
        for _, r in df.iterrows():
            print(f"{r['symbol']:<9}{r['median_stop_price']:>14.5f}"
                  f"{r['risk_at_min_lot']:>21.2f}{r['risk_at_min_lot']/200*100:>11.1f}%"
                  f"{r['risk_at_min_lot']/10000*100:>11.2f}%")
        avg = df["risk_at_min_lot"].mean()
        print(f"\n  average forced risk per trade at the 0.01 lot floor: ${avg:.2f}")
        print(f"    -> on $200 that is {avg/200*100:.1f}% per trade "
              f"(book TARGETS 2.5% = $5.00)")
        print(f"    -> on $10k that is {avg/10000*100:.2f}% per trade "
              f"(book targets 0.5% = $50.00)")
        forced_200 = avg / 200
    else:
        forced_200 = 0.025
        print("  (no data)")

    for label, e in EDGES.items():
        per_day = e["per_day_9sym"] * SYMBOL_SCALE
        n_trades = max(1, round(per_day * TRADING_DAYS))
        print("\n" + "=" * 88)
        print(f"### {label}   (~{per_day:.2f} trades/day on 20 symbols)")
        print("=" * 88)

        report("FTMO $10,000 @ 0.5% risk", 10_000, 0.005, e["win"], e["rr"],
               n_trades, rng, ftmo=True)

        report("YOLO $200 @ 2.5% risk (the CONFIGURED target)", 200, 0.025,
               e["win"], e["rr"], n_trades, rng)

        report(f"YOLO $200 @ {forced_200*100:.1f}% risk (what the LOT FLOOR actually forces)",
               200, min(forced_200, 0.99), e["win"], e["rr"], n_trades, rng)

    print("\n" + "=" * 88)
    print("Caveats: projection OF the backtest, not proof of it. Excludes swap")
    print("(daily holds run 2-4 days) and assumes trades are independent, which")
    print("they are not — correlated FX pairs cluster wins and losses together,")
    print("so real spread is WIDER than shown here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
