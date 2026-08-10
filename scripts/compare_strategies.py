"""Compare every strategy head-to-head — which one is best.

Runs each strategy across a common symbol basket on the VALIDATION window
(last 30% of ~5.5 months of 30m bars, strictly out-of-sample), with the live
1R trail modelled, and ranks them by out-of-sample expectancy, profit factor,
net P&L and how many symbols each is positive on.

Usage:  python scripts/compare_strategies.py [--start N --count M]
Writes: backtest/strategy_comparison.csv + strategy_comparison.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bos import MarketOpenBOSStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_sweep_reclaim import SweepReclaimStrategy  # noqa: E402
from metatraderbot.strategies_tod_reversion import TODReversionStrategy  # noqa: E402
from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy  # noqa: E402

BASKET = ["XAUUSD", "US30", "DE40", "USDJPY", "EURUSD", "GBPUSD",
          "AUDUSD", "NZDUSD", "USDCHF", "EURAUD"]

# (label, class, extra strategy_params, style)
# style controls the exit given to each strategy for a FAIR test:
#   "trend"     -> TP 2R with a 1R ratchet trail (let winners run)
#   "reversion" -> TP 1R, no trail (take the bounce back to fair value fast)
# Testing a mean-reversion fade with a trend-following exit understates it,
# and vice-versa, so each strategy is judged with the exit that fits its idea.
STRATS = [
    ("Donchian", DonchianBreakoutStrategy, {"donchian": {"period": 10}}, "trend"),
    ("MACD", MACDTrendStrategy, {"macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9,
                                          "require_zero_side": True}, "exit": {"sl_swing_lookback": 5}}, "trend"),
    ("BOS", MarketOpenBOSStrategy, {"bos": {"session_open_hour_utc": 13, "trade_window_hours": 4,
                                            "swing_lookback": 40, "swing_strength": 2}}, "trend"),
    ("SweepReclaim", SweepReclaimStrategy, {"sweep": {"lookback": 20, "sl_buffer_atr": 0.25}}, "reversion"),
    ("TODReversion", TODReversionStrategy, {"tod": {"revert_hour_utc": 20, "stretch_atr": 1.5}}, "reversion"),
    ("VWAPReversion", VWAPReversionStrategy, {"vwap": {"band_atr": 2.0}}, "reversion"),
]


class Windowed:
    WINDOW = 350

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, i, bars):
        start = max(0, i - self.WINDOW + 1)
        v = bars.iloc[start:i + 1].reset_index(drop=True)
        return self.inner.on_bar(len(v) - 1, v)


def load(sym):
    f = ROOT / "data" / f"{sym.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0]); df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}).dropna().tail(6000)


def cfg_for(sym, extra, style):
    tp = 2.0 if style == "trend" else 1.0
    trail = 1.0 if style == "trend" else 0.0
    strat = {"trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "donchian": {}, "orb": {},
             "macd": {}, "bos": {}, "sweep": {}, "tod": {}, "vwap": {}, "session": None,
             "exit": {"tp_r_multiple": tp}}
    for k, v in extra.items():
        if k == "exit":
            strat["exit"].update(v)
        else:
            strat[k] = v
    return {"symbol": sym, "capital": {"risk_usd_per_trade": 100, "initial_usd": 10000,
            "max_positions": 2, "daily_stop_loss_usd": -1e9, "trail_start_r": trail,
            "trail_lock_fraction": 0.5}, "strategy": strat}


def run(label, cls, extra, style, sym, bars):
    c = cfg_for(sym, extra, style)
    bt = Backtest(c, bars, strategy=Windowed(cls(c)))
    bt.run(); r = bt.report()
    w = sum(t.pnl_usd for t in bt.closed if t.pnl_usd > 0)
    l = -sum(t.pnl_usd for t in bt.closed if t.pnl_usd < 0)
    r["pf"] = (w / l) if l > 0 else (float("inf") if w > 0 else 0.0)
    return r


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=99)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out_csv = ROOT / "backtest" / "strategy_comparison.csv"

    if not args.report_only:
        rows = []
        idx = -1
        for label, cls, extra, style in STRATS:
            for sym in BASKET:
                idx += 1
                if idx < args.start or idx >= args.start + args.count:
                    continue
                bars = load(sym)
                if bars is None:
                    continue
                valid = bars.iloc[int(len(bars) * 0.70):]
                r = run(label, cls, extra, style, sym, valid)
                rows.append({"strategy": label, "symbol": sym, "trades": r["trades"],
                             "wr": round(r["win_rate_pct"], 1),
                             "exp": round(r["expectancy_usd_per_trade"], 2),
                             "pf": round(r["pf"], 2), "pnl": round(r["total_pnl_usd"], 2),
                             "dd": round(r["max_drawdown_usd"], 2)})
                print(f"{label:14}{sym:8} n={r['trades']:4d} wr={r['win_rate_pct']:5.1f} "
                      f"exp=${r['expectancy_usd_per_trade']:7.2f} pf={r['pf']:5.2f} "
                      f"pnl=${r['total_pnl_usd']:8.0f}", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"appended {len(df)} rows", flush=True)
        return 0

    # report
    df = pd.read_csv(out_csv)
    md = ["# Strategy comparison — which is best (out-of-sample)", "",
          f"Validation window, {len(BASKET)}-symbol basket, live 1R trail modelled, TP 2R.", "",
          "| strategy | symbols | trades | net P&L | avg exp $/trade | avg PF | positive symbols |",
          "|---|---|---|---|---|---|---|"]
    agg = []
    for label in [s[0] for s in STRATS]:
        sub = df[df["strategy"] == label]
        if not len(sub):
            continue
        pf = sub["pf"].replace([float("inf")], 10.0)
        agg.append((label, sub["pnl"].sum(), sub["exp"].mean(), pf.mean(),
                    int((sub["pnl"] > 0).sum()), len(sub), int(sub["trades"].sum())))
    for label, pnl, exp, pf, pos, n, tr in sorted(agg, key=lambda x: x[2], reverse=True):
        md.append(f"| {label} | {n} | {tr} | {pnl:+.0f} | {exp:+.2f} | {pf:.2f} | {pos}/{n} |")
    (ROOT / "backtest" / "strategy_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
