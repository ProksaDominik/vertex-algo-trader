"""Geometry sweep — tests the hypothesis "the SL and TP are both too far".

For each active Donchian symbol, on the VALIDATION window only (last 30% of
~5.5 months of bars, strictly out-of-sample), sweeps:
  - stop placement: full opposite channel (wide, original) vs tighter
    ATR- and channel-fraction stops
  - take-profit: 1R, 1.5R, 2R, 3R
with the live trail (1R engage) modeled. Reports expectancy, profit factor,
win rate and max drawdown per combo so the choice is evidence, not opinion.

Usage:  python scripts/geometry_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402

# Active Donchian slots from the trimmed live configs.
SYMBOLS = [("USDJPY", "30m", 100.0), ("US30", "30m", 100.0),
           ("EURAUD", "30m", 100.0), ("XAUUSD", "30m", 30.0)]

SL_MODES = [
    ("channel(1.0)", {"sl_mode": "channel"}),
    ("channel(0.5)", {"sl_mode": "channel_frac", "sl_channel_frac": 0.5}),
    ("atr(1.5)", {"sl_mode": "atr", "sl_atr_mult": 1.5}),
    ("atr(1.0)", {"sl_mode": "atr", "sl_atr_mult": 1.0}),
]
TP_MULTS = [1.0, 1.5, 2.0, 3.0]


class Windowed:
    WINDOW = 350

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, i, bars):
        start = max(0, i - self.WINDOW + 1)
        view = bars.iloc[start:i + 1].reset_index(drop=True)
        return self.inner.on_bar(len(view) - 1, view)


def load(symbol, tf):
    f = ROOT / "data" / f"{symbol.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0]); df.index = pd.to_datetime(df.index, utc=True)
    if tf == "30m":
        df = df.resample("30min").agg({"open": "first", "high": "max", "low": "min",
                                       "close": "last", "volume": "sum"}).dropna()
    return df.tail(6000)


def cfg_for(symbol, risk, sl_kw, tp):
    d = {"period": 10}; d.update(sl_kw)
    return {
        "symbol": symbol,
        "capital": {"risk_usd_per_trade": risk, "initial_usd": 10_000,
                    "max_positions": 2, "daily_stop_loss_usd": -1e9,
                    "trail_start_r": 1.0, "trail_lock_fraction": 0.5},
        "strategy": {"donchian": d, "exit": {"tp_r_multiple": tp}, "session": None,
                     "trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "orb": {}},
    }


def run(symbol, tf, risk, sl_kw, tp, bars):
    c = cfg_for(symbol, risk, sl_kw, tp)
    bt = Backtest(c, bars, strategy=Windowed(DonchianBreakoutStrategy(c)))
    bt.run(); r = bt.report()
    wins = sum(t.pnl_usd for t in bt.closed if t.pnl_usd > 0)
    loss = -sum(t.pnl_usd for t in bt.closed if t.pnl_usd < 0)
    r["pf"] = (wins / loss) if loss > 0 else float("inf") if wins > 0 else 0.0
    return r


def main() -> int:
    rows = []
    for symbol, tf, risk in SYMBOLS:
        bars = load(symbol, tf)
        if bars is None:
            print(f"[skip] {symbol}: no data"); continue
        valid = bars.iloc[int(len(bars) * 0.70):]
        print(f"\n===== {symbol} {tf} (validation, {len(valid)} bars) =====")
        print(f"{'stop':16}{'TP':>5}{'trades':>8}{'WR%':>7}{'exp$':>9}{'PF':>7}{'pnl$':>10}{'maxDD':>10}")
        for sl_name, sl_kw in SL_MODES:
            for tp in TP_MULTS:
                r = run(symbol, tf, risk, sl_kw, tp, valid)
                rows.append({"symbol": symbol, "stop": sl_name, "tp": tp,
                             "trades": r["trades"], "wr": round(r["win_rate_pct"], 1),
                             "exp": round(r["expectancy_usd_per_trade"], 2),
                             "pf": round(r["pf"], 2), "pnl": round(r["total_pnl_usd"], 2),
                             "dd": round(r["max_drawdown_usd"], 2)})
                print(f"{sl_name:16}{tp:5.1f}{r['trades']:8d}{r['win_rate_pct']:7.1f}"
                      f"{r['expectancy_usd_per_trade']:9.2f}{r['pf']:7.2f}"
                      f"{r['total_pnl_usd']:10.2f}{r['max_drawdown_usd']:10.2f}")
    df = pd.DataFrame(rows)
    out = ROOT / "backtest" / "geometry_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out.name}")
    print("\n=== best combo per symbol (by expectancy) ===")
    for symbol in df["symbol"].unique():
        s = df[df["symbol"] == symbol].sort_values("exp", ascending=False).iloc[0]
        print(f"  {symbol:8} stop={s['stop']:14} tp={s['tp']}  exp=${s['exp']:.2f}  "
              f"pf={s['pf']}  pnl=${s['pnl']:.0f}  (current: channel(1.0) tp=3.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
