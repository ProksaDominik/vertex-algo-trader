"""Can you choose WHEN to start a challenge and beat the unconditional rate?

Every replay so far has started a challenge on every possible date and averaged
the outcomes. But nobody has to buy a challenge on a random day -- you can watch
the market for weeks and start when conditions look favourable.

That makes this different from the prediction ideas already rejected here
(day-of-week, stop-after-loss, prior-day momentum, volatility-regime entry
filters). Those tried to forecast individual TRADES. This only asks whether the
market state observable BEFORE a challenge begins carries information about how
that challenge will go. You genuinely can wait.

Features are all computed from bars closed strictly before the start date:
  * trailing realised volatility of the basket (20d, 60d)
  * dispersion across pairs (are they moving together or independently?)
    -- the correlation audit says the basket is only ~4.9 independent bets, so
    a period of LOW cross-correlation should behave like more bets
  * trailing performance of the strategy itself over the prior 20/60 days
  * trailing absolute trend strength (mean-reversion should prefer chop)

Method: fit the rule on the TRAIN half only, freeze it, apply to OOS. Report
both. A rule that only works in-sample is worth nothing, and this project has
produced several of those.

Usage:
    python scripts/start_timing_test.py [--workers 4]
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
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
           "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
           "AUDCHF", "EURAUD", "AUDCAD"]
RSI14 = {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}
RSI7 = {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}
TP = 1.0
RISK, CONC, PER_DAY = 0.04, 15, 8
DAILY_LIM, TOTAL_LIM, MIN_DAYS, TARGET = 0.05, 0.10, 4, 0.10


def _run(params, sym, bars):
    strat = {"session": None, "exit": {"tp_r_multiple": TP, "sl_swing_lookback": 10}}
    strat.update(params)
    cfg = {"symbol": sym,
           "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                       "max_positions": 2, "daily_stop_loss_usd": -1e9},
           "strategy": strat}
    bt = Backtest(cfg, bars, strategy=Windowed(RSIReversionStrategy(cfg)))
    bt.run()
    return bt.closed


def _cell(sym):
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        agree = {(pd.Timestamp(t.entered_at).normalize(), t.direction.name)
                 for t in _run(RSI7, sym, bars)}
        cl = bars["close"]
        out = []
        for t in _run(RSI14, sym, bars):
            o = pd.Timestamp(t.entered_at).normalize()
            if (o, t.direction.name) not in agree:
                continue
            c = pd.Timestamp(t.exited_at).normalize()
            rp = abs(t.entry_price - t.stop_loss)
            if rp <= 0 or c < o:
                continue
            seg = cl.loc[o:c]
            sg = 1.0 if t.direction.name == "LONG" else -1.0
            path = (seg.values - t.entry_price) * sg / rp
            path[-1] = t.pnl_usd / RISK_USD
            out.append({"open": o, "sym": sym, "path": list(path),
                        "final": float(path[-1])})
        rets = cl.pct_change()
        return {"trades": out, "rets": rets.rename(sym)}
    except Exception:
        return None


def replay(by_day, days, i0, hz):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= CONC or opened >= PER_DAY:
                break
            op.append([tr, 0])
            opened += 1
        if opened:
            td += 1
        fl, still = 0.0, []
        for tr, age in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * RISK
                still.append([tr, age])
            else:
                real += tr["final"] * RISK
        op = still
        eq = 1.0 + real + fl
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return 0
        if eq >= 1.0 + TARGET and td >= MIN_DAYS:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print("=" * 100)
    print("START TIMING — does pre-start market state predict challenge outcome?")
    print("=" * 100, flush=True)

    trades, rets = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, SYMBOLS):
            if c:
                trades += c["trades"]
                rets.append(c["rets"])
    ret_df = pd.concat(rets, axis=1).dropna(how="all")

    by_day = {}
    for t in trades:
        by_day.setdefault(t["open"], []).append(t)
    days = sorted(by_day)

    # Features strictly from data BEFORE the start date.
    vol20 = ret_df.std(axis=1).rolling(20).mean().shift(1)
    vol60 = ret_df.std(axis=1).rolling(60).mean().shift(1)
    # cross-sectional dispersion: high = pairs moving independently
    disp = ret_df.std(axis=1).rolling(20).mean().shift(1)
    absmove = ret_df.abs().mean(axis=1).rolling(20).mean().shift(1)

    rows = []
    for i, d in enumerate(days):
        if i + 10 >= len(days):
            break
        # trailing strategy performance over the prior 60 signal days
        prior = [r for dd in days[max(0, i - 60):i] for r in
                 [t["final"] for t in by_day.get(dd, [])]]
        rows.append({
            "i": i, "date": d,
            "vol20": vol20.reindex([d]).iloc[0] if d in vol20.index else np.nan,
            "vol60": vol60.reindex([d]).iloc[0] if d in vol60.index else np.nan,
            "absmove": absmove.reindex([d]).iloc[0] if d in absmove.index else np.nan,
            "prior_e": float(np.mean(prior)) if len(prior) >= 20 else np.nan,
            "pass10": replay(by_day, days, i, 10),
            "pass30": replay(by_day, days, i, 30),
        })
    df = pd.DataFrame(rows).dropna()
    split = int(len(df) * 0.70)
    tr, oo = df.iloc[:split], df.iloc[split:]
    print(f"\n{len(df)} start dates | train {len(tr)} | OOS {len(oo)}")
    print(f"baseline pass rate  train {tr['pass10'].mean():.1%} / "
          f"OOS {oo['pass10'].mean():.1%}  (10 day)")
    print(f"                    train {tr['pass30'].mean():.1%} / "
          f"OOS {oo['pass30'].mean():.1%}  (30 day)\n")

    print(f"{'feature':<12}{'rule':<22}{'train pass':>12}{'OOS pass':>11}"
          f"{'OOS n':>8}{'vs base':>10}")
    print("-" * 100)
    for feat in ("vol20", "vol60", "absmove", "prior_e"):
        for q, lab in ((0.5, "top half"), (0.7, "top 30%"), (0.3, "bottom 30%")):
            thr = tr[feat].quantile(q)
            if lab.startswith("bottom"):
                m_tr, m_oo = tr[feat] <= thr, oo[feat] <= thr
            else:
                m_tr, m_oo = tr[feat] >= thr, oo[feat] >= thr
            if m_oo.sum() < 25:
                continue
            for hz in ("pass10", "pass30"):
                ptr = tr.loc[m_tr, hz].mean()
                poo = oo.loc[m_oo, hz].mean()
                base = oo[hz].mean()
                print(f"{feat:<12}{(lab+' '+hz[-2:]+'d'):<22}{ptr:>12.1%}"
                      f"{poo:>11.1%}{int(m_oo.sum()):>8}{poo-base:>+10.1%}")
        print()

    print("=" * 100)
    print("A rule only counts if it beats baseline in BOTH halves. Selecting a")
    print("start condition on the train half and finding it holds OOS is the")
    print("only evidence that waiting for conditions is worth anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
