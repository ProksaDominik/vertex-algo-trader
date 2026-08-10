"""Where do the losses actually come from — and does removing them help?

Two different questions, and only the second one matters for deployment:

  A. ATTRIBUTION: which buckets (instrument, direction, session, volatility
     regime, stop width) carry the losses?
  B. DOES EXCLUSION SURVIVE? A bucket that lost money in the past only helps
     if it KEEPS losing. So every candidate exclusion here is derived from the
     EARLY half of history and then applied, frozen, to the LATE half.

(A) alone is how you talk yourself into curve fitting: any dataset has a worst
bucket, and removing it always improves that dataset. (B) is the test that
tells you whether the bucket was structurally bad or just unlucky.

Usage:
    python scripts/loss_attribution.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
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


def collect(sym: str, bars: pd.DataFrame) -> list[dict]:
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
    out = []
    atr = (bars["high"] - bars["low"]).rolling(14).mean()
    for t in bt.closed:
        ts = pd.Timestamp(t.entered_at)
        stop_dist = abs(t.entry_price - t.stop_loss)
        a = float(atr.reindex([ts.normalize()]).ffill().iloc[0]) if len(atr) else np.nan
        out.append({
            "sym": sym,
            "r": t.pnl_usd / RISK_USD,
            "dir": t.direction.name,
            "reason": t.exit_reason,
            "dow": ts.dayofweek,
            "month": ts.month,
            "stop_atr": (stop_dist / a) if a and a == a and a > 0 else np.nan,
        })
    return out


def stat(rs: list[float]) -> str:
    if not rs:
        return "n=0"
    e = float(np.mean(rs))
    se = float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    return f"n={len(rs):<5} E={e:+.3f}R (+/-{se:.3f})  total={sum(rs):+7.1f}R"


def main() -> int:
    early: list[dict] = []
    late: list[dict] = []
    for s in BASKET:
        bars = load(s, "1d")
        if bars is None:
            continue
        mid = len(bars) // 2
        try:
            early += collect(s, bars.iloc[:mid])
            late += collect(s, bars.iloc[mid:])
        except Exception as e:
            print(f"  [skip] {s}: {e!r}")

    print("=" * 100)
    print("(A) WHERE THE LOSSES COME FROM  — early half (2016-2021)")
    print("=" * 100)

    def bucket(rows, key):
        d = defaultdict(list)
        for row in rows:
            d[row[key]].append(row["r"])
        return d

    for key, label in [("sym", "INSTRUMENT"), ("dir", "DIRECTION"),
                       ("reason", "EXIT REASON"), ("dow", "DAY OF WEEK")]:
        print(f"\n--- by {label} ---")
        d = bucket(early, key)
        for k, rs in sorted(d.items(), key=lambda kv: sum(kv[1])):
            print(f"  {str(k):<10}{stat(rs)}")

    print("\n--- by STOP WIDTH (in ATRs) ---")
    d = defaultdict(list)
    for row in early:
        sa = row["stop_atr"]
        if sa != sa:
            continue
        b = "<1 ATR" if sa < 1 else "1-2" if sa < 2 else "2-3" if sa < 3 else "3+ ATR"
        d[b].append(row["r"])
    for k in ["<1 ATR", "1-2", "2-3", "3+ ATR"]:
        if k in d:
            print(f"  {k:<10}{stat(d[k])}")

    # Concentration: how much of the total loss is in the worst instruments?
    per_sym = bucket(early, "sym")
    losses = sorted(((sum(v), k) for k, v in per_sym.items()))
    tot_neg = sum(x for x, _ in losses if x < 0)
    print(f"\n  total negative R across losing instruments: {tot_neg:+.1f}R")
    run = 0.0
    for x, k in losses:
        if x >= 0:
            break
        run += x
        print(f"    {k:<8}{x:>+8.1f}R   cumulative {run/tot_neg:>5.0%} of all losses")

    print("\n" + "=" * 100)
    print("(B) DOES EXCLUDING THEM HELP LATER?  rule from EARLY, applied to LATE")
    print("=" * 100)

    base = [r["r"] for r in late]
    print(f"\n  baseline — whole basket, late half : {stat(base)}")

    early_by_sym = bucket(early, "sym")
    worst = [k for k, v in sorted(early_by_sym.items(), key=lambda kv: sum(kv[1]))]

    for n_drop in (1, 2, 3, 5):
        drop = set(worst[:n_drop])
        kept = [r["r"] for r in late if r["sym"] not in drop]
        e_base, e_kept = np.mean(base), np.mean(kept)
        se = np.sqrt(np.var(base, ddof=1) / len(base) + np.var(kept, ddof=1) / len(kept))
        verdict = "HELPED" if e_kept > e_base else "HURT"
        sig = "significant" if abs(e_kept - e_base) > 2 * se else "within noise"
        print(f"\n  drop worst {n_drop} early instrument(s) {sorted(drop)}")
        print(f"    late result: {stat(kept)}")
        print(f"    vs baseline: {e_kept - e_base:+.3f}R  -> {verdict}, {sig}")

    # The non-FX instruments, singled out — they are mechanically different.
    nonfx = {"XAUUSD", "US500"}
    kept = [r["r"] for r in late if r["sym"] not in nonfx]
    e_base, e_kept = np.mean(base), np.mean(kept)
    se = np.sqrt(np.var(base, ddof=1) / len(base) + np.var(kept, ddof=1) / len(kept))
    print(f"\n  drop NON-FX (XAUUSD, US500) — a mechanistic reason, not a data one")
    print(f"    late result: {stat(kept)}")
    print(f"    vs baseline: {e_kept - e_base:+.3f}R  "
          f"({'beyond' if abs(e_kept-e_base) > 2*se else 'within'} 2 SE)")

    # Direction rule.
    for d in ("LONG", "SHORT"):
        e_rows = [r["r"] for r in early if r["dir"] == d]
        if e_rows and np.mean(e_rows) < 0:
            kept = [r["r"] for r in late if r["dir"] != d]
            print(f"\n  early {d} was negative ({np.mean(e_rows):+.3f}R) -> drop {d}s")
            print(f"    late result: {stat(kept)}  vs baseline {np.mean(base):+.3f}R")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
