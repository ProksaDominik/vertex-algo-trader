"""
Run the no-wick backtest against downloaded XAUUSD data and print a report.

Prerequisites:
    python scripts/download_data.py

Usage:
    python scripts/run_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.backtest import Backtest  # noqa: E402


def main() -> int:
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)

    data_path = ROOT / cfg["backtest"]["data_path"]
    if not data_path.exists():
        print(f"[FAIL] Data file missing: {data_path}")
        print("       Run: python scripts/download_data.py")
        return 2

    bars = pd.read_parquet(data_path)
    print(f"[INFO] Loaded {len(bars):,} bars from {data_path.name}")
    print(f"       Range: {bars.index[0]} -> {bars.index[-1]}")

    bt = Backtest(cfg, bars)
    bt.run()
    r = bt.report()

    print()
    print("=" * 64)
    print("NO-WICK STRATEGY — BACKTEST RESULTS")
    print("=" * 64)
    print(f"Symbol / TF:       {cfg['symbol']} / {cfg['timeframe']}")
    print(f"Initial capital:   ${r['initial_capital']:>10.2f}")
    print(f"Final equity:      ${r['final_equity']:>10.2f}   "
          f"({r['return_pct']:+.1f}%)"
          + ("   [BLOWN UP]" if r["blown_up"] else ""))
    print(f"Total P&L:         ${r['total_pnl_usd']:>+10.2f}")
    print()
    print(f"Trades:            {r['trades']:>10d}")
    print(f"  Wins:            {r['wins']:>10d}   ({r['win_rate_pct']:.1f}%)")
    print(f"  Losses:          {r['losses']:>10d}")
    print(f"  Avg win:         ${r['avg_win_usd']:>+10.2f}")
    print(f"  Avg loss:        ${r['avg_loss_usd']:>+10.2f}")
    print(f"  Expectancy:      ${r['expectancy_usd_per_trade']:>+10.2f}  per trade")
    print()
    print(f"Max drawdown:      ${r['max_drawdown_usd']:>+10.2f}   ({r['max_drawdown_pct']:.1f}%)")
    print()
    print(f"Sub-0.01-lot fills: {r['sub_min_lot_trades']:>10d}   ({r['sub_min_lot_pct']:.1f}% of trades)")
    print(f"  ^ These would REJECT on a real broker at $200 capital.")
    print()
    print("Exit reason breakdown:")
    for reason in ("tp", "sl", "end-of-data"):
        count = r["by_exit_reason"].get(reason, 0)
        pnl = r["pnl_by_exit_reason"].get(reason, 0.0)
        avg = (pnl / count) if count else 0.0
        print(f"  {reason:<12s} {count:>4d} trades   total ${pnl:>+8.2f}   avg ${avg:>+7.2f}")
    print("=" * 64)

    if r["trades"] == 0:
        print()
        print("[WARN] Zero trades fired. Likely causes:")
        print("       - Wickless threshold too strict (try wick_max_pct: 0.10)")
        print("       - Body threshold too high (try body_min_pct: 0.50)")
        print("       - Trend filter too strict (try require_slope_sign: false)")
        print("       Adjust config/settings.yaml and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
