"""Phase 5 walk-forward validation — before vs after exit stacks, strictly
out-of-sample.

For every deployed slot in both portfolio YAMLs:
- load the pair's 15m parquet, resample to the slot's timeframe,
- split 70% train / 30% validation by time,
- run the OLD exit stack (tp 0.333R, trail from 0.15R) and the NEW one
  (tp 3.0R, trail from 1.0R) on BOTH windows,
- report trades, win rate, profit factor, expectancy, max drawdown.

No parameters are optimized here — this validates the deployed configs on
data the (original) sweeps never selected on. Fills are bar-close idealized
(see backtest.py); partial-take is not modeled.

Usage:  python scripts/walk_forward.py
Writes: backtest/walkforward_phase5.csv + backtest/walkforward_phase5.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategy import NoWickStrategy  # noqa: E402

STRATS = {
    "DonchianBreakout": DonchianBreakoutStrategy,
    "MACDTrend": MACDTrendStrategy,
    "NoWick": NoWickStrategy,
}

BOOKS = [
    ("FTMO", "config/portfolio_ftmo.yaml", 100.0, 0.0),
    ("YOLO", "config/portfolio_yolo.yaml", 30.0, 2.5),   # skyrocket 2.5 live
]
EXIT_STACKS = {
    "before": {"tp": 0.333, "trail_start": 0.15, "trail_lock": 0.5},
    "after":  {"tp": 3.0,   "trail_start": 1.0,  "trail_lock": 0.5},
}


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{symbol.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    if timeframe == "30m":
        df = df.resample("30min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    # Cap history: strategies recompute indicators over the FULL slice each
    # bar (O(n^2)) — ~5.5 months of 30m bars keeps runs tractable while the
    # validation tail stays a meaningful ~7 weeks out-of-sample.
    return df.tail(6000)


def strat_config(symbol: str, params: dict, risk: float, exits: dict,
                 skyrocket: float) -> dict:
    p = params or {}
    cfg = {
        "symbol": symbol,
        "capital": {
            "risk_usd_per_trade": risk,
            "initial_usd": 10_000,
            "max_positions": 2,
            "daily_stop_loss_usd": -1e9,
            "trail_start_r": exits["trail_start"],
            "trail_lock_fraction": exits["trail_lock"],
            "skyrocket_multiple": skyrocket,
        },
        "strategy": {
            "trend": p.get("trend", {}) or {},
            "signal": p.get("signal", {}) or {},
            "entry": p.get("entry", {}) or {},
            "exit": dict(p.get("exit", {}) or {}),
            "session": p.get("session"),
            "bollinger": p.get("bollinger", {}),
            "donchian": p.get("donchian", {}),
            "macd": p.get("macd", {}),
            "orb": p.get("orb", {}),
        },
    }
    s = cfg["strategy"]
    s["trend"].setdefault("ema_period", 50)
    s["signal"].setdefault("wick_max_pct", 0.05)
    s["signal"].setdefault("body_min_pct", 0.60)
    s["entry"].setdefault("retrace_bars", 5)
    s["exit"].setdefault("sl_swing_lookback", 5)
    s["exit"]["tp_r_multiple"] = exits["tp"]
    return cfg


class Windowed:
    """Feed the wrapped strategy a sliding 350-bar window instead of the
    full history. The deployed strategies look back at most ~60 bars
    (EMA50 / Donchian10 / MACD 26+9 / swing 10), so results are identical
    to 4+ decimals while the full-history recompute cost drops from
    O(n^2) to O(n). Harness-only optimization — live code is untouched."""

    WINDOW = 350

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, bar_index: int, bars: pd.DataFrame):
        start = max(0, bar_index - self.WINDOW + 1)
        view = bars.iloc[start:bar_index + 1].reset_index(drop=True)
        return self.inner.on_bar(len(view) - 1, view)


def run_one(cfg: dict, strat_name: str, bars: pd.DataFrame) -> dict:
    strat_cls = STRATS[strat_name]
    bt = Backtest(cfg, bars, strategy=Windowed(strat_cls(cfg)))
    bt.run()
    r = bt.report()
    wins = sum(t.pnl_usd for t in bt.closed if t.pnl_usd > 0)
    losses = -sum(t.pnl_usd for t in bt.closed if t.pnl_usd < 0)
    r["profit_factor"] = (wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
    return r


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="first slot index (global, both books)")
    ap.add_argument("--count", type=int, default=10_000, help="number of slots to run")
    ap.add_argument("--report-only", action="store_true", help="just rebuild the md from the csv")
    args = ap.parse_args()

    out_csv = ROOT / "backtest" / "walkforward_phase5.csv"
    if args.report_only:
        return write_report(pd.read_csv(out_csv), out_csv)

    slot_idx = -1
    rows = []
    for book, yaml_path, risk, skyrocket in BOOKS:
        with open(ROOT / yaml_path) as f:
            book_cfg = yaml.safe_load(f)
        for entry in book_cfg["configs"]:
            if entry["strategy"] not in STRATS:
                continue
            slot_idx += 1
            if slot_idx < args.start or slot_idx >= args.start + args.count:
                continue
            bars = load_bars(entry["symbol"], entry["timeframe"])
            if bars is None or len(bars) < 500:
                print(f"[skip] {entry['name']}: no data")
                continue
            split = int(len(bars) * 0.70)
            windows = {"train": bars.iloc[:split], "valid": bars.iloc[split:]}
            for stack_name, exits in EXIT_STACKS.items():
                for win_name, win_bars in windows.items():
                    cfg = strat_config(entry["symbol"], entry.get("strategy_params"),
                                       risk, exits, skyrocket)
                    r = run_one(cfg, entry["strategy"], win_bars)
                    rows.append({
                        "book": book, "slot": entry["name"], "symbol": entry["symbol"],
                        "strategy": entry["strategy"], "tf": entry["timeframe"],
                        "stack": stack_name, "window": win_name,
                        "trades": r["trades"], "win_rate": round(r["win_rate_pct"], 1),
                        "pnl": round(r["total_pnl_usd"], 2),
                        "expectancy": round(r["expectancy_usd_per_trade"], 2),
                        "profit_factor": round(r["profit_factor"], 2),
                        "max_dd": round(r["max_drawdown_usd"], 2),
                    })
                    print(f"{book} {entry['name']:34s} {stack_name:6s} {win_name:5s} "
                          f"n={r['trades']:4d} wr={r['win_rate_pct']:5.1f}% "
                          f"pnl={r['total_pnl_usd']:9.2f} pf={r['profit_factor']:5.2f}")

    df = pd.DataFrame(rows)
    header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=header, index=False)
    print(f"\nAppended {len(df)} rows to {out_csv.name}")
    return 0


def write_report(df: pd.DataFrame, out_csv: Path) -> int:
    md = ["# Walk-forward validation — Phase 5", "",
          "70% train / 30% validation split per pair (last ~5.5 months of 30m",
          "bars; validation = final ~7 weeks, strictly out-of-sample for the",
          "original sweeps). Fills bar-close idealized; partial-take not",
          "modeled; trail modeled at bar-close granularity (live trails",
          "tighter). before = tp 0.333R / trail 0.15R; after = tp 3R / trail 1R.", ""]
    for win in ("train", "valid"):
        md.append(f"## {win} window\n")
        md.append("| book | stack | slots | trades | net P&L | avg PF | positive slots |")
        md.append("|---|---|---|---|---|---|---|")
        for book in ("FTMO", "YOLO"):
            for stack in ("before", "after"):
                sub = df[(df["book"] == book) & (df["stack"] == stack) & (df["window"] == win)]
                if not len(sub):
                    continue
                pf = sub["profit_factor"].replace([float("inf")], 10.0)
                md.append(
                    f"| {book} | {stack} | {len(sub)} | {int(sub['trades'].sum())} "
                    f"| {sub['pnl'].sum():+.0f} | {pf.mean():.2f} "
                    f"| {(sub['pnl'] > 0).sum()}/{len(sub)} |")
        md.append("")
    md.append("Per-slot rows: see walkforward_phase5.csv")
    (ROOT / "backtest" / "walkforward_phase5.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote walkforward_phase5.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
