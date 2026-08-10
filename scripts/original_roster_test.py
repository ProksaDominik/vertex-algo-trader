"""Rigorously re-test the EXACT original strategy roster from the initial
commit (git 6862e33, config/portfolio_ftmo.yaml + portfolio_yolo.yaml,
before any of this session's "new strategies" work) -- Donchian/MACD/NoWick
with tp_r_multiple=0.333 (TP 3x CLOSER than SL -- a tight-TP/high-win-rate
shape) and a 0.15R/50%-lock ratchet trail, exactly as originally deployed.

This is DIFFERENT from full_strategy_comparison.py's test of these same
three strategy CLASSES, which used a generic wide-TP "trend" style (my own
reasonable default, not the historically-tuned live parameters). If the
original tight-TP parameterization shows genuine edge that the wide-TP
version didn't, that's a real, distinct finding -- not covered by the
earlier comparison.

Usage: python scripts/original_roster_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategy import NoWickStrategy  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "US30", "US500", "DE40", "XAUUSD",
]

# Exact params from git 6862e33's config/portfolio_ftmo.yaml + portfolio_yolo.yaml
STRATS = [
    ("Donchian_orig", DonchianBreakoutStrategy, {
        "donchian": {"period": 10}, "exit": {"tp_r_multiple": 0.333},
    }),
    ("MACD_orig", MACDTrendStrategy, {
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "require_zero_side": True},
        "exit": {"sl_swing_lookback": 5, "tp_r_multiple": 0.333},
    }),
    ("NoWick_orig", NoWickStrategy, {
        "trend": {"ema_period": 50}, "signal": {"wick_max_pct": 0.05, "body_min_pct": 0.60},
        "entry": {"retrace_bars": 5}, "exit": {"sl_swing_lookback": 5, "tp_r_multiple": 0.333},
    }),
]


class Windowed:
    WINDOW = 350

    def __init__(self, inner):
        self.inner = inner

    def on_bar(self, i, bars):
        start = max(0, i - self.WINDOW + 1)
        v = bars.iloc[start:i + 1].reset_index(drop=True)
        return self.inner.on_bar(len(v) - 1, v)


def load(sym: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{sym.lower()}_15m.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    return df.resample("30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def cfg_for(sym: str, params: dict) -> dict:
    strat = {
        "trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "donchian": {}, "orb": {},
        "macd": {}, "bos": {}, "sweep": {}, "tod": {}, "vwap": {}, "session": None,
        "exit": {"tp_r_multiple": 0.333},
    }
    for k, v in params.items():
        if k == "exit":
            strat["exit"].update(v)
        else:
            strat[k] = v
    return {
        "symbol": sym,
        "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                    "max_positions": 5, "daily_stop_loss_usd": -1e9,
                    "trail_start_r": 0.15, "trail_lock_fraction": 0.5},  # exact original ratchet
        "strategy": strat,
    }


def run_one(cls, params: dict, sym: str, bars: pd.DataFrame) -> dict:
    cfg = cfg_for(sym, params)
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    r = bt.report()
    r["rs"] = [t.pnl_usd / RISK_USD for t in bt.closed]
    return r


def main() -> None:
    print("=" * 96)
    print(f"ORIGINAL ROSTER RE-TEST -- exact params from git 6862e33 (tp_r_multiple=0.333, "
          f"trail 0.15R/50%), {len(BASKET)} symbols, 70/30 train/OOS")
    print("=" * 96)

    summary = []
    for name, cls, params in STRATS:
        train_rs: list[float] = []
        oos_rs: list[float] = []
        oos_trades = oos_wins = 0
        oos_dd_pcts: list[float] = []
        per_symbol = []

        for sym in BASKET:
            bars = load(sym)
            if bars is None:
                continue
            split = int(len(bars) * 0.70)
            train_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]
            try:
                r_train = run_one(cls, params, sym, train_bars)
                r_oos = run_one(cls, params, sym, oos_bars)
            except Exception as e:
                print(f"  [ERROR] {name} {sym}: {e!r}")
                continue

            train_rs += r_train["rs"]
            oos_rs += r_oos["rs"]
            oos_trades += r_oos["trades"]
            oos_wins += r_oos["wins"]
            oos_dd_pcts.append(r_oos["max_drawdown_pct"])
            per_symbol.append((sym, r_oos["trades"], r_oos["win_rate_pct"], r_oos["expectancy_usd_per_trade"]))

        print(f"\n--- {name}: OOS aggregate ({oos_trades} trades across {len(BASKET)} symbols) ---")
        print("  per-symbol (top 6 by trade count):")
        for sym, n, wr, exp in sorted(per_symbol, key=lambda x: -x[1])[:6]:
            print(f"    {sym:8} n={n:5d}  wr={wr:5.1f}%  exp=${exp:+.2f}/trade")

        if oos_trades < 10:
            print(f"  too few OOS trades ({oos_trades}) -- cannot evaluate.")
            summary.append((name, oos_trades, None, None, None, None))
            continue

        oos_wr = oos_wins / oos_trades
        win_rs = [x for x in oos_rs if x > 0]
        loss_rs = [x for x in oos_rs if x <= 0]
        oos_avg_win = (sum(win_rs) / len(win_rs)) if win_rs else 0.0
        oos_avg_loss = (-sum(loss_rs) / len(loss_rs)) if loss_rs else 0.0
        oos_rr = (oos_avg_win / oos_avg_loss) if oos_avg_loss > 0 else 0.0
        oos_e_r = sum(oos_rs) / len(oos_rs)
        oos_pf = (sum(win_rs) / -sum(loss_rs)) if loss_rs and sum(loss_rs) < 0 else (
            float("inf") if win_rs else 0.0)
        oos_max_dd = min(oos_dd_pcts) if oos_dd_pcts else 0.0

        train_e_r = (sum(train_rs) / len(train_rs)) if train_rs else float("nan")
        wf_eff = (oos_e_r / train_e_r) if train_e_r and train_e_r > 0 else float("nan")

        if oos_rr > 0 and 0 < oos_wr < 1:
            p_pass, _, _, _ = mc_sim(0.005, oos_wr, oos_rr, tpd=4)
        else:
            p_pass = float("nan")

        pf_s = f"{oos_pf:.2f}" if oos_pf != float("inf") else "inf"
        wf_s = f"{wf_eff:.2f}" if wf_eff == wf_eff else "n/a"
        pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
        print(f"  POOLED: n={oos_trades:5d}  win_rate={oos_wr:.1%}  avg_RR={oos_rr:.2f}  "
              f"E_net={oos_e_r:+.3f}R  PF={pf_s}  max_dd={oos_max_dd:.1f}%  "
              f"wf_eff={wf_s}  MC_P(pass)={pp_s}")
        summary.append((name, oos_trades, oos_e_r, oos_pf, oos_wr, p_pass))

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for name, n, e_r, pf, wr, pp in summary:
        if e_r is None:
            print(f"  {name:16} n={n:<6} insufficient trades")
            continue
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        pp_s = f"{pp:.1%}" if pp == pp else "n/a"
        flag = "  <-- POSITIVE EDGE" if e_r > 0 else ""
        print(f"  {name:16} n={n:<6} win_rate={wr:.1%}  E_net={e_r:+.3f}R  PF={pf_s}  MC_P(pass)={pp_s}{flag}")


if __name__ == "__main__":
    main()
