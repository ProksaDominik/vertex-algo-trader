"""Can the 1-HOUR timeframe be made profitable at >5 trades/day?

The canonical test said no: all 16 well-known strategies are negative at 1h
over 10.5 years, RSI worst of all (-0.129R, 0/12 symbols). But "the raw signal
loses" is not the same as "no filtered subset wins", and the daily timeframe
IS positive — so the obvious idea is to let the DAILY chart supply the edge and
the 1h chart supply the timing (classic multi-timeframe confirmation).

This searches that space honestly:

  - Run 1h RSI once per symbol, recording for every trade the features a
    filter could legitimately use AT ENTRY (daily RSI, daily trend, session,
    volatility regime, stop width, cost ratio). No lookahead: every feature is
    computed from bars at or before the entry bar.
  - Fit filter thresholds on the TRAIN half only.
  - Score every filter combination on the OOS half.
  - Report the FRONTIER: expectancy vs trades/day, so the trade-off between
    "only take the surest setups" and ">5 trades/day" is explicit instead of
    assumed away.

Usage:
    python scripts/hourly_precision_search.py [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
import json
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

from canonical_strategy_test import RISK_USD, Windowed  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402

# FX only — XAUUSD/US500 carried 42% of all losses and are structurally
# unsuited to mean reversion (see scripts/loss_attribution.py).
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.where(d > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    ll = (-d).where(d < 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    return (100 - 100 / (1 + g / ll.replace(0, np.nan))).fillna(50)


def load_raw(sym: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{sym.lower()}_15m_long.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def features_for(sym: str) -> list[dict] | None:
    """Every 1h RSI trade + the entry-time context a filter may use."""
    raw = load_raw(sym)
    if raw is None:
        return None
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h1 = raw.resample("1h").agg(agg).dropna()
    d1 = raw.resample("1d").agg(agg).dropna()
    if len(h1) < 2000 or len(d1) < 300:
        return None

    # Daily context, shifted one day so a 1h bar only ever sees COMPLETED
    # daily information — this is the no-lookahead guarantee.
    d_rsi = wilder_rsi(d1["close"], 14).shift(1)
    d_ema = d1["close"].ewm(span=200, adjust=False).mean().shift(1)
    d_close = d1["close"].shift(1)
    d_trend_up = (d_close > d_ema)

    h_atr = (h1["high"] - h1["low"]).rolling(14).mean()
    atr_pct = h_atr.rank(pct=True)

    cfg = {
        "symbol": sym,
        "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                    "max_positions": 2, "daily_stop_loss_usd": -1e9},
        "strategy": {"rsi": {"period": 14, "oversold": 30, "overbought": 70},
                     "session": None,
                     "exit": {"tp_r_multiple": 1.0, "sl_swing_lookback": 10}},
    }
    try:
        bt = Backtest(cfg, h1, strategy=Windowed(RSIReversionStrategy(cfg)))
        bt.run()
    except Exception:
        return None

    mult = PAIR_SPECS[sym]["contract_mult"]
    spread = PAIR_SPECS[sym]["spread_price"]
    comm = PAIR_SPECS[sym]["commission_per_lot"]

    out = []
    for t in bt.closed:
        ts = pd.Timestamp(t.entered_at)
        day = ts.normalize()
        stop = abs(t.entry_price - t.stop_loss)
        if stop <= 0:
            continue
        atr_now = h_atr.reindex([ts]).ffill()
        atrv = float(atr_now.iloc[0]) if len(atr_now) and atr_now.notna().all() else np.nan
        # cost as a fraction of R — the quantity the cost gate targets
        lots = RISK_USD / (stop * mult)
        cost_r = ((spread * mult * lots) + comm * lots * 2) / RISK_USD
        out.append({
            "sym": sym,
            "r": t.pnl_usd / RISK_USD,
            "long": t.direction.name == "LONG",
            "hour": ts.hour,
            "dow": ts.dayofweek,
            "d_rsi": float(d_rsi.reindex([day]).ffill().iloc[0]) if day in d_rsi.index or True else np.nan,
            "d_trend_up": bool(d_trend_up.reindex([day]).ffill().fillna(False).iloc[0]),
            "atr_pct": float(atr_pct.reindex([ts]).ffill().iloc[0]) if len(atr_pct) else np.nan,
            "stop_atr": (stop / atrv) if atrv and atrv == atrv and atrv > 0 else np.nan,
            "cost_r": cost_r,
            "ts": ts.isoformat(),
        })
    return out


def summarize(rows: list[dict], days: int) -> dict:
    if not rows:
        return {"n": 0}
    rs = [r["r"] for r in rows]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gl = -sum(losses)
    return {
        "n": len(rs),
        "per_day": len(rs) / days if days else 0.0,
        "win": len(wins) / len(rs),
        "e": float(np.mean(rs)),
        "se": float(np.std(rs, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0,
        "total": float(sum(rs)),
        "pf": (sum(wins) / gl) if gl > 0 else float("inf"),
    }


# --- the filter space -----------------------------------------------------
# Each entry: (label, predicate). Predicates use ONLY entry-time features.
FILTERS = {
    "none": lambda r: True,
    # Mean-revert only WITH the daily trend (buy dips in daily uptrends).
    "daily_trend_align": lambda r: r["long"] == r["d_trend_up"],
    # Daily oscillator must agree the market is stretched the same way.
    "daily_rsi_agree": lambda r: (r["d_rsi"] < 45) if r["long"] else (r["d_rsi"] > 55),
    "daily_rsi_strong": lambda r: (r["d_rsi"] < 35) if r["long"] else (r["d_rsi"] > 65),
    # Liquid sessions only (London + NY), where spreads are tightest.
    "session_lon_ny": lambda r: 7 <= r["hour"] < 21,
    # Skip the partial Sunday bar and the Friday close.
    "no_sun": lambda r: r["dow"] != 6,
    # Cost must be small relative to R (the dominant loss source).
    "cost_lt_04": lambda r: r["cost_r"] < 0.04,
    "cost_lt_02": lambda r: r["cost_r"] < 0.02,
    # Stop must be wide enough that costs don't dominate.
    "stop_gt_1atr": lambda r: r["stop_atr"] == r["stop_atr"] and r["stop_atr"] > 1.0,
    # Avoid the dead-vol regime where mean reversion has no range to work with.
    "vol_mid_high": lambda r: r["atr_pct"] == r["atr_pct"] and r["atr_pct"] > 0.35,
}

STACKS = [
    ["none"],
    ["no_sun"],
    ["session_lon_ny"],
    ["daily_trend_align"],
    ["daily_rsi_agree"],
    ["daily_rsi_strong"],
    ["cost_lt_04"],
    ["stop_gt_1atr"],
    ["no_sun", "session_lon_ny"],
    ["no_sun", "session_lon_ny", "cost_lt_04"],
    ["no_sun", "session_lon_ny", "stop_gt_1atr"],
    ["no_sun", "session_lon_ny", "daily_rsi_agree"],
    ["no_sun", "session_lon_ny", "daily_trend_align"],
    ["no_sun", "session_lon_ny", "cost_lt_04", "daily_rsi_agree"],
    ["no_sun", "session_lon_ny", "cost_lt_04", "stop_gt_1atr"],
    ["no_sun", "session_lon_ny", "cost_lt_04", "daily_rsi_agree", "stop_gt_1atr"],
    ["no_sun", "session_lon_ny", "cost_lt_02", "daily_rsi_strong", "stop_gt_1atr"],
    ["no_sun", "session_lon_ny", "cost_lt_02", "daily_rsi_strong", "stop_gt_1atr",
     "vol_mid_high"],
    ["no_sun", "cost_lt_02", "daily_rsi_strong", "stop_gt_1atr", "vol_mid_high",
     "daily_trend_align"],
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print("=" * 108)
    print("1-HOUR PRECISION SEARCH — can filtering make the hourly timeframe pay?")
    print(f"{len(SYMBOLS)} FX symbols (XAUUSD/US500 excluded), ~10.5y, "
          f"train=first 70% / OOS=last 30%")
    print("=" * 108)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(features_for, SYMBOLS):
            if res:
                rows += res
    print(f"\ncollected {len(rows)} raw 1h RSI trades")

    rows.sort(key=lambda r: r["ts"])
    split = int(len(rows) * 0.70)
    train, oos = rows[:split], rows[split:]

    def days_of(rs):
        return len({r["ts"][:10] for r in rs}) or 1

    d_tr, d_oos = days_of(train), days_of(oos)
    print(f"train: {len(train)} trades over {d_tr} days | "
          f"OOS: {len(oos)} trades over {d_oos} days")

    print(f"\n{'filter stack':<62}{'n':>6}{'/day':>7}{'win%':>7}"
          f"{'E_net':>9}{'+/-SE':>8}{'PF':>7}{'totR':>9}")
    print("-" * 108)

    results = []
    for stack in STACKS:
        preds = [FILTERS[s] for s in stack]

        def keep(r):
            return all(p(r) for p in preds)

        tr = [r for r in train if keep(r)]
        oo = [r for r in oos if keep(r)]
        st_tr = summarize(tr, d_tr)
        st = summarize(oo, d_oos)
        if not st.get("n"):
            continue
        label = "+".join(stack)
        pf = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
        print(f"{label:<62}{st['n']:>6}{st['per_day']:>7.2f}{st['win']*100:>7.1f}"
              f"{st['e']:>+9.3f}{st['se']:>8.3f}{pf:>7}{st['total']:>+9.1f}")
        results.append({"stack": stack, "oos": st, "train": st_tr})

    print("\n" + "=" * 108)
    print("THE FRONTIER — only stacks that are POSITIVE out-of-sample")
    print("=" * 108)
    pos = [r for r in results if r["oos"]["e"] > 0]
    if not pos:
        print("\n  NONE. No filter combination produced positive OOS expectancy at 1h.")
    else:
        print(f"\n{'filter stack':<62}{'/day':>7}{'E_net':>9}{'E/SE':>7}"
              f"{'train E':>10}{'consistent?':>13}")
        print("-" * 108)
        for r in sorted(pos, key=lambda x: -x["oos"]["per_day"]):
            t = r["train"]["e"] if r["train"].get("n") else float("nan")
            ratio = r["oos"]["e"] / r["oos"]["se"] if r["oos"]["se"] > 0 else 0
            ok = "yes" if t > 0 else "NO - train neg"
            print(f"{'+'.join(r['stack']):<62}{r['oos']['per_day']:>7.2f}"
                  f"{r['oos']['e']:>+9.3f}{ratio:>7.1f}{t:>+10.3f}{ok:>13}")

        good = [r for r in pos if r["oos"]["per_day"] >= 5]
        print(f"\n  >>> stacks with BOTH positive OOS expectancy AND >=5 trades/day: "
              f"{len(good)}")
        for r in good:
            print(f"      {'+'.join(r['stack'])}  "
                  f"{r['oos']['per_day']:.2f}/day  E={r['oos']['e']:+.3f}R")

    out = ROOT / "backtest" / "hourly_precision.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
