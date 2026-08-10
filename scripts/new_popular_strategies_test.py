"""Test 3 new, genuinely popular strategies not covered by
full_strategy_comparison.py's 18: SMC-style Liquidity-Sweep+BOS+Order-Block,
Ichimoku Cloud, and Supertrend. Same rigor: 30m bars, full 21-symbol basket,
70/30 train/OOS, real costs, pooled Definition-of-Done metrics, PLUS the
reversed-split robustness check (evaluate directly on the first 30% too)
that has already caught several false positives this session.

Usage: python scripts/new_popular_strategies_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_smc_bos import SMCBOSStrategy  # noqa: E402
from metatraderbot.strategies_ichimoku import IchimokuStrategy  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "US30", "US500", "DE40", "XAUUSD",
]

STRATS = [
    ("SMC_BOS", SMCBOSStrategy, {
        "smc_bos": {"swing_lookback": 15, "sweep_buffer_pips": 1.0, "bos_lookback": 8,
                     "max_wait_bars": 15, "sl_buffer_pips": 3.0},
        "exit": {"tp_r_multiple": 1.5},
    }),
    ("Ichimoku", IchimokuStrategy, {
        "ichimoku": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52,
                      "displacement": 26, "sl_buffer_atr_mult": 0.5},
        "exit": {"tp_r_multiple": 2.0},
    }),
    ("Supertrend", SupertrendStrategy, {
        "supertrend": {"period": 10, "multiplier": 3.0},
        "exit": {"tp_r_multiple": 2.0},
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
    ).dropna().tail(6000)


def cfg_for(sym: str, params: dict) -> dict:
    strat = {
        "smc_bos": {}, "ichimoku": {}, "supertrend": {}, "session": None,
        "exit": {"tp_r_multiple": 2.0},
    }
    for k, v in params.items():
        if k == "exit":
            strat["exit"].update(v)
        else:
            strat[k] = v
    return {
        "symbol": sym,
        "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                    "max_positions": 2, "daily_stop_loss_usd": -1e9,
                    "trail_start_r": 1.0, "trail_lock_fraction": 0.5},
        "strategy": strat,
    }


def run_one(cls, params: dict, sym: str, bars: pd.DataFrame) -> dict:
    cfg = cfg_for(sym, params)
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    r = bt.report()
    r["rs"] = [t.pnl_usd / RISK_USD for t in bt.closed]
    return r


def eval_segment(cls, params, bars_by_symbol: dict) -> dict:
    rs, trades, wins = [], 0, 0
    for sym, bars in bars_by_symbol.items():
        try:
            r = run_one(cls, params, sym, bars)
        except Exception as e:
            print(f"    [ERROR] {sym}: {e!r}")
            continue
        rs += r["rs"]
        trades += r["trades"]
        wins += r["wins"]
    return {"rs": rs, "trades": trades, "wins": wins}


def summarize(label: str, res: dict) -> tuple:
    n = res["trades"]
    if n < 10:
        print(f"  {label}: n={n} -- too few")
        return (n, None, None)
    rs = res["rs"]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    wr = res["wins"] / n
    e_r = sum(rs) / n
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    if rr > 0 and 0 < wr < 1:
        p_pass, _, _, _ = mc_sim(0.005, wr, rr, tpd=4)
    else:
        p_pass = float("nan")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
    print(f"  {label}: n={n}  win_rate={wr:.1%}  avg_RR={rr:.2f}  E_net={e_r:+.3f}R  PF={pf_s}  MC_P(pass)={pp_s}")
    return (n, e_r, pf)


def main() -> None:
    print("=" * 96)
    print(f"NEW POPULAR STRATEGIES -- SMC BOS, Ichimoku, Supertrend x {len(BASKET)} symbols, 30m bars")
    print("=" * 96)

    all_bars = {}
    for sym in BASKET:
        b = load(sym)
        if b is not None:
            all_bars[sym] = b

    summary = []
    for name, cls, params in STRATS:
        print(f"\n--- {name} ---")
        train_bars = {s: b.iloc[:int(len(b) * 0.70)] for s, b in all_bars.items()}
        oos_bars = {s: b.iloc[int(len(b) * 0.70):] for s, b in all_bars.items()}
        first30_bars = {s: b.iloc[:int(len(b) * 0.30)] for s, b in all_bars.items()}

        print("  [forward test: OOS = last 30%]")
        res_oos = eval_segment(cls, params, oos_bars)
        n, e_r, pf = summarize("OOS", res_oos)

        print("  [robustness check: evaluate directly on first 30%]")
        res_first = eval_segment(cls, params, first30_bars)
        summarize("first30", res_first)

        summary.append((name, n, e_r, pf))

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for name, n, e_r, pf in summary:
        if e_r is None:
            print(f"  {name:12} n={n:<6} insufficient trades")
            continue
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        flag = "  <-- POSITIVE EDGE" if e_r > 0 else ""
        print(f"  {name:12} n={n:<6} E_net={e_r:+.3f}R  PF={pf_s}{flag}")


if __name__ == "__main__":
    main()
