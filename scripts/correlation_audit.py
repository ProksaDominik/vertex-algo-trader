"""How many INDEPENDENT bets is a 20-symbol basket really making?

The live ledger for 2026-07-15 showed 17 simultaneous trades that decomposed
into roughly one position — long CHF, short JPY — expressed 13 ways. They all
won together (+$374); six days later the same structure lost together
(-$832). Position count is therefore a bad proxy for risk, and the config's
per-currency cap was clearly not binding.

This measures the real thing, three ways:

  1. CORRELATION of daily returns across the deployed basket.
  2. EFFECTIVE BET COUNT via the participation ratio of the correlation
     matrix's eigenvalues: sum(w)^2 / sum(w^2). Twenty perfectly independent
     symbols score 20; twenty copies of one symbol score 1.
  3. SIGNAL CLUSTERING: when the daily RSI strategy actually fires, how many
     symbols fire on the SAME DAY in the SAME DIRECTION? That is the number
     the position limits have to survive.

Output feeds max_currency_exposure_r / max_concurrent_positions_total.

Usage:
    python scripts/correlation_audit.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
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


def effective_bets(corr: np.ndarray) -> float:
    w = np.linalg.eigvalsh(corr)
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


def currencies(sym: str) -> list[str]:
    return [sym[:3], sym[3:]] if len(sym) == 6 and sym.isalpha() else [sym]


def main() -> int:
    print("=" * 96)
    print("CORRELATION AUDIT — how many independent bets does the basket make?")
    print("=" * 96)

    closes = {}
    for s in BASKET:
        b = load(s, "1d")
        if b is not None:
            closes[s] = b["close"]
    px = pd.DataFrame(closes).dropna(how="all")
    rets = px.pct_change().dropna(how="all")
    print(f"\n{len(px.columns)} symbols, {len(rets)} daily observations "
          f"({rets.index[0].date()} -> {rets.index[-1].date()})")

    corr = rets.corr().fillna(0.0).values
    n = corr.shape[0]
    eff = effective_bets(corr)
    print(f"\n--- (1)+(2) return correlation ---")
    off = corr[~np.eye(n, dtype=bool)]
    print(f"  mean |correlation| between symbols : {np.abs(off).mean():.3f}")
    print(f"  max  |correlation|                 : {np.abs(off).max():.3f}")
    print(f"  EFFECTIVE INDEPENDENT BETS         : {eff:.1f}  (out of {n} symbols)")
    print(f"  -> position sizing that treats {n} positions as {n} independent")
    print(f"     risks is overstating diversification by ~{n/eff:.1f}x")

    # Most-correlated pairs, the ones that silently double a bet.
    pairs = []
    cols = list(px.columns)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((abs(corr[i, j]), cols[i], cols[j], corr[i, j]))
    pairs.sort(reverse=True)
    print("\n  most-correlated pairs (these are ONE bet, not two):")
    for _, a, b, c in pairs[:10]:
        print(f"     {a:<8} / {b:<8}  r = {c:+.3f}")

    # --- (3) signal clustering -------------------------------------------
    print("\n--- (3) signal clustering: when RSI daily fires, how many fire together? ---")
    by_day: dict[pd.Timestamp, list[tuple[str, str]]] = defaultdict(list)
    for s in BASKET:
        bars = load(s, "1d")
        if bars is None:
            continue
        cfg = {
            "symbol": s,
            "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                        "max_positions": 2, "daily_stop_loss_usd": -1e9},
            "strategy": {"rsi": {"period": 14, "oversold": 30, "overbought": 70},
                         "session": None,
                         "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10}},
        }
        try:
            bt = Backtest(cfg, bars, strategy=Windowed(RSIReversionStrategy(cfg)))
            bt.run()
        except Exception as e:
            print(f"   [skip] {s}: {e!r}")
            continue
        for t in bt.closed:
            day = pd.Timestamp(t.entered_at).normalize()
            by_day[day].append((s, t.direction.name))

    if by_day:
        sizes = [len(v) for v in by_day.values()]
        print(f"  days with at least one signal : {len(by_day)}")
        print(f"  mean signals per active day   : {np.mean(sizes):.2f}")
        print(f"  95th percentile               : {np.percentile(sizes, 95):.0f}")
        print(f"  worst day                     : {max(sizes)} simultaneous signals")
        hist = Counter(sizes)
        print("\n  distribution of simultaneous signals:")
        for k in sorted(hist):
            bar = "#" * min(60, hist[k])
            print(f"     {k:>2} signal(s): {hist[k]:>4} days {bar}")

        # Currency concentration on the busiest days.
        print("\n  worst-case CURRENCY exposure on the busiest days:")
        worst = sorted(by_day.items(), key=lambda kv: -len(kv[1]))[:5]
        for day, sigs in worst:
            cc: Counter = Counter()
            for sym, direction in sigs:
                for c in currencies(sym):
                    cc[c] += 1
            top = ", ".join(f"{c}x{k}" for c, k in cc.most_common(4))
            print(f"     {day.date()}  {len(sigs):>2} signals -> {top}")
            mx = cc.most_common(1)[0]
            print(f"        -> single currency '{mx[0]}' carried by {mx[1]} of "
                  f"{len(sigs)} positions")

    print("\n" + "=" * 96)
    print("Implication for the configs: cap RISK PER CURRENCY, not position count.")
    print("The 2026-07-15 book held 13 positions that were ~2 bets; a position")
    print("limit cannot see that, a per-currency R limit can.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
