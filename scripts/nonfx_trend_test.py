"""Do indices/commodities/crypto give a second, UNCORRELATED return stream?

Everything tested so far is FX mean-reversion: 17 pairs worth only ~4.9
independent bets (scripts/correlation_audit.py). That cap on independence is
what pins the portfolio's daily mu/sigma at 0.03-0.12 and therefore caps the
FTMO pass rate -- scripts/variance_reduction_race.py showed that cutting
volatility just cuts drift with it, because the bets are not independent.

The only mechanism that RAISES mu/sigma rather than trading one against the
other is a genuinely uncorrelated second stream. Indices, commodities and
crypto qualify on two counts: different instruments, and they TREND rather
than mean-revert, so trend-following (which loses on FX) is the right tool.

Three questions, in order, because a NO at any stage kills the idea:
  1. Does trend-following actually work on these instruments, both halves?
  2. Is the resulting daily return stream uncorrelated with the FX stream?
  3. Does combining them raise mu/sigma above the 0.03-0.12 ceiling?

Question 2 is the one that matters. Everything selling off together in
risk-off periods is a real possibility, in which case this is the same wall
wearing a different hat.

COST CAVEAT: PAIR_SPECS has no entries for most of these, so costs are
ESTIMATED from live MT5 ticks and marked as such. Costs have been the binding
constraint all session, so treat any positive result as provisional until the
specs are verified against real fills.

Usage:
    python scripts/nonfx_trend_test.py [--workers 5]
"""
from __future__ import annotations

import argparse
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
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

# Tradeable on the live MT5 account (verified: 10/12 offered). BRENT and
# COPPER downloaded but are NOT offered by the broker, so they are research
# only -- a backtested edge you cannot trade is worthless.
NONFX = ["US30", "NAS100", "DE40", "JP225", "WTI", "XPTUSD", "BTCUSD", "ETHUSD",
         "XAUUSD", "US500"]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
      "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
      "AUDCHF", "EURAUD", "AUDCAD"]

TREND = {
    "Donchian20": (DonchianBreakoutStrategy, {"donchian": {"period": 20}}),
    "Donchian55": (DonchianBreakoutStrategy, {"donchian": {"period": 55}}),
    "Supertrend": (SupertrendStrategy,
                   {"supertrend": {"period": 10, "multiplier": 3.0}}),
    "MACD": (MACDTrendStrategy, {"macd": {}}),
    "EMACross": (EMACrossoverStrategy,
                 {"ema_cross": {"fast_period": 20, "slow_period": 50}}),
}
TPS = (2.0, 3.0, 4.0, 6.0)


def _cell(job):
    kind, name, tp, sym = job
    if kind == "trend":
        cls, params = TREND[name]
    else:
        cls, params = RSIReversionStrategy, {
            "rsi": {"period": 14, "oversold": 30, "overbought": 70}}
    try:
        bars = load(sym, "1d")
        if bars is None or len(bars) < 600:
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
        out = {"kind": kind, "name": name, "tp": tp, "sym": sym,
               "train": [], "oos": [], "daily": {}}
        for half, sl in (("train", slice(0, split)), ("oos", slice(split, None))):
            b = bars.iloc[sl]
            if len(b) < 200:
                continue
            bt = Backtest(cfg, b, strategy=Windowed(cls(cfg)))
            bt.run()
            out[half] = [t.pnl_usd / RISK_USD for t in bt.closed]
            if half == "oos":
                for t in bt.closed:
                    d = pd.Timestamp(t.entered_at).normalize()
                    out["daily"][d] = out["daily"].get(d, 0.0) + t.pnl_usd / RISK_USD
        return out
    except Exception as e:
        # Surface failures. A bare `return None` here previously hid the fact
        # that EVERY non-FX cell was throwing on a missing PAIR_SPECS entry,
        # and the run reported "no strategy survives" when nothing had run.
        return {"error": f"{kind}/{name}/{sym}: {e!r}"}


def agg(rs):
    if not rs:
        return None
    a = np.array(rs)
    w = a[a > 0]
    gl = -a[a <= 0].sum()
    return {"n": len(a), "win": len(w) / len(a), "e": float(a.mean()),
            "pf": float(w.sum() / gl) if gl > 0 else float("inf")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    print("=" * 100)
    print("NON-FX TREND FOLLOWING — is there a second, uncorrelated stream?")
    print("=" * 100, flush=True)

    jobs = [("trend", n, tp, s) for n in TREND for tp in TPS for s in NONFX]
    jobs += [("mr", "RSI14", 4.0, s) for s in FX]
    cells, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            if not c:
                continue
            (errors if "error" in c else cells).append(c)
    if errors:
        print(f"\n{len(errors)} of {len(jobs)} cells FAILED. First few:")
        for e in errors[:5]:
            print("   ", e["error"])
        print()

    print(f"\n(1) TREND STRATEGIES ON NON-FX (pooled over {len(NONFX)} instruments)")
    print(f"{'strategy':<14}{'TP':>5}{'n_oos':>7}{'win_tr':>8}{'win_oos':>9}"
          f"{'E_tr':>9}{'E_oos':>9}{'PF':>7}  gate")
    print("-" * 100)
    survivors = []
    for name in TREND:
        for tp in TPS:
            mine = [c for c in cells if c["kind"] == "trend"
                    and c["name"] == name and c["tp"] == tp]
            tr = agg([r for c in mine for r in c["train"]])
            oo = agg([r for c in mine for r in c["oos"]])
            if not tr or not oo or oo["n"] < 100:
                continue
            ok = tr["e"] > 0 and oo["e"] > 0
            if ok:
                survivors.append((name, tp, mine, tr, oo))
            print(f"{name:<14}{tp:>5.1f}{oo['n']:>7}{tr['win']*100:>8.1f}"
                  f"{oo['win']*100:>9.1f}{tr['e']:>+9.3f}{oo['e']:>+9.3f}"
                  f"{oo['pf']:>7.2f}  {'KEEP' if ok else ''}")
        print()

    if not survivors:
        print("=" * 100)
        print("NO trend strategy survives both halves on non-FX instruments.")
        print("The second-stream idea dies here -- no point measuring the")
        print("correlation of a stream that has no edge.")
        return 0

    # (2) correlation of the best trend stream vs the FX mean-reversion stream
    print("=" * 100)
    print("(2) CORRELATION with the FX mean-reversion stream (OOS daily R)")
    print("=" * 100)
    fx_daily = {}
    for c in [c for c in cells if c["kind"] == "mr"]:
        for d, r in c["daily"].items():
            fx_daily[d] = fx_daily.get(d, 0.0) + r
    fx_s = pd.Series(fx_daily).sort_index()

    print(f"  {'trend setup':<24}{'overlap days':>14}{'corr vs FX':>13}{'verdict':>26}")
    best = None
    for name, tp, mine, tr, oo in survivors:
        td = {}
        for c in mine:
            for d, r in c["daily"].items():
                td[d] = td.get(d, 0.0) + r
        ts = pd.Series(td).sort_index()
        # Align on the UNION of days with 0 for "no trade that day". Requiring
        # both streams to trade the same day (dropna) is the wrong test: two
        # streams that rarely coincide are not un-measurable, they are
        # evidence OF independence. For portfolio purposes what matters is the
        # daily P&L series including flat days.
        idx = fx_s.index.union(ts.index)
        both = pd.DataFrame({"fx": fx_s.reindex(idx).fillna(0.0),
                             "tr": ts.reindex(idx).fillna(0.0)})
        if len(both) < 40:
            continue
        corr = both["fx"].corr(both["tr"])
        v = ("GENUINELY INDEPENDENT" if abs(corr) < 0.3 else
             "partly correlated" if abs(corr) < 0.6 else "SAME BET")
        print(f"  {name+' @'+str(tp):<24}{len(both):>14}{corr:>13.3f}{v:>26}")
        if best is None or abs(corr) < abs(best[0]):
            best = (corr, name, tp, ts)

    # (3) does combining raise mu/sigma?
    if best:
        print("\n" + "=" * 100)
        print("(3) DOES COMBINING RAISE mu/sigma?  (the number that caps pass rate)")
        print("=" * 100)
        corr, name, tp, ts = best
        idx = fx_s.index.union(ts.index)
        both = pd.DataFrame({"fx": fx_s.reindex(idx).fillna(0.0),
                             "tr": ts.reindex(idx).fillna(0.0)})
        for lbl, s in (("FX mean-reversion alone", both["fx"]),
                       (f"{name}@{tp} trend alone", both["tr"]),
                       ("50/50 combined", 0.5 * both["fx"] + 0.5 * both["tr"])):
            mu, sd = s.mean(), s.std()
            print(f"  {lbl:<30} mu {mu:>+7.4f}R/day  sigma {sd:>6.4f}"
                  f"  mu/sigma {(mu/sd if sd else 0):>+7.4f}")
        print("\n  The FX-only book measures mu/sigma 0.03-0.12 daily.")
        print("  60% pass in 10 days needs roughly 0.7-2.0. If the combined")
        print("  figure is not materially above the FX-only one, diversifying")
        print("  across asset classes does not break the ceiling either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
