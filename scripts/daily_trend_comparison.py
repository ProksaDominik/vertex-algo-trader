"""Classic trend-following (Donchian breakout, MA crossover, MACD) tested on
DAILY bars, 2015-2026 -- the timeframe these strategies are actually
designed for and where real CTA/managed-futures funds run them. Every
earlier comparison in this project used 15m/30m bars, where these same
signal types failed; this tests whether the timeframe (not the signal
logic) was the missing piece.

Same rigor as the rest of this project's research: 70/30 train/OOS split,
real transaction costs, pooled Definition-of-Done metrics, Monte Carlo
P(pass) via research/mc3.py.

Usage: python scripts/daily_trend_comparison.py
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
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD", "US500"]

# Classic CTA-style params: longer lookbacks than the 30m versions, since a
# "20-day breakout" / "50/200 MA cross" is the actual popular daily-bar
# convention (the 30m versions used period=10 / fast=10,slow=30 -- far too
# short for daily bars, which is itself part of why testing the RIGHT
# timeframe with RIGHT-sized lookbacks matters here).
STRATS = [
    ("Donchian20_trend2R", DonchianBreakoutStrategy, {"donchian": {"period": 20}}, 2.0, 1.0),
    ("Donchian55_trend3R", DonchianBreakoutStrategy, {"donchian": {"period": 55}}, 3.0, 1.0),  # classic Turtle system-2 length
    ("MACross50_200_trend2R", SMACrossStrategy, {
        "ma_cross": {"fast_period": 50, "slow_period": 200}, "exit": {"sl_swing_lookback": 20},
    }, 2.0, 1.0),
    ("MACross20_50_trend2R", SMACrossStrategy, {
        "ma_cross": {"fast_period": 20, "slow_period": 50}, "exit": {"sl_swing_lookback": 20},
    }, 2.0, 1.0),
    ("MACD_daily_trend2R", MACDTrendStrategy, {
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "require_zero_side": True},
        "exit": {"sl_swing_lookback": 20},
    }, 2.0, 1.0),
]


def load(sym: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{sym.lower()}_1d.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.dropna()


def cfg_for(sym: str, params: dict, tp: float, trail: float) -> dict:
    strat = {
        "trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "donchian": {}, "orb": {},
        "macd": {}, "bos": {}, "sweep": {}, "tod": {}, "vwap": {}, "session": None,
        "ma_cross": {}, "exit": {"tp_r_multiple": tp},
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
                    "trail_start_r": trail, "trail_lock_fraction": 0.5},
        "strategy": strat,
    }


def run_one(cls, params: dict, tp: float, trail: float, sym: str, bars: pd.DataFrame) -> dict:
    cfg = cfg_for(sym, params, tp, trail)
    bt = Backtest(cfg, bars, strategy=cls(cfg))  # no Windowed wrapper needed -- daily bars, short series
    bt.run()
    r = bt.report()
    r["rs"] = [t.pnl_usd / RISK_USD for t in bt.closed]
    return r


def run_split(name, cls, params, tp, trail, first_frac):
    """Evaluate directly on the FIRST first_frac of each symbol's bars (the
    portion that was 'train' in the forward test) -- a clean reversed check
    since these strategies use fixed, not-fit-to-data parameters, so there
    is no overfitting-to-parameters risk in either half; this only checks
    temporal consistency of the sign/magnitude of edge."""
    oos_rs, oos_trades, oos_wins = [], 0, 0
    for sym in SYMBOLS:
        bars = load(sym)
        if bars is None:
            continue
        cut = int(len(bars) * first_frac)
        seg = bars.iloc[:cut]
        try:
            r = run_one(cls, params, tp, trail, sym, seg)
        except Exception:
            continue
        oos_rs += r["rs"]
        oos_trades += r["trades"]
        oos_wins += r["wins"]
    if oos_trades < 5:
        print(f"  n={oos_trades} -- too few")
        return
    wins = [x for x in oos_rs if x > 0]
    losses = [x for x in oos_rs if x <= 0]
    wr = oos_wins / oos_trades
    e_r = sum(oos_rs) / oos_trades
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  n={oos_trades}  win_rate={wr:.1%}  E_net={e_r:+.3f}R  PF={pf_s}")


def main() -> None:
    print("=" * 96)
    print(f"DAILY-TIMEFRAME TREND-FOLLOWING -- {len(STRATS)} strategies x {len(SYMBOLS)} symbols, "
          "2015-2026, 70/30 train/OOS")
    print("=" * 96)

    summary = []
    for name, cls, params, tp, trail in STRATS:
        train_rs: list[float] = []
        oos_rs: list[float] = []
        oos_trades = oos_wins = 0
        oos_dd_pcts: list[float] = []
        per_symbol = []

        for sym in SYMBOLS:
            bars = load(sym)
            if bars is None:
                continue
            split = int(len(bars) * 0.70)
            train_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]
            try:
                r_train = run_one(cls, params, tp, trail, sym, train_bars)
                r_oos = run_one(cls, params, tp, trail, sym, oos_bars)
            except Exception as e:
                print(f"  [ERROR] {name} {sym}: {e!r}")
                continue
            train_rs += r_train["rs"]
            oos_rs += r_oos["rs"]
            oos_trades += r_oos["trades"]
            oos_wins += r_oos["wins"]
            oos_dd_pcts.append(r_oos["max_drawdown_pct"])
            per_symbol.append((sym, r_oos["trades"], r_oos["expectancy_usd_per_trade"]))

        print(f"\n--- {name}: OOS aggregate ({oos_trades} trades across {len(SYMBOLS)} symbols) ---")
        for sym, n, exp in per_symbol:
            print(f"    {sym:8} n={n:4d}  exp=${exp:+.2f}/trade")
        if oos_trades < 10:
            print(f"  too few OOS trades ({oos_trades})")
            summary.append((name, oos_trades, None, None, None))
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
            p_pass, _, _, _ = mc_sim(0.005, oos_wr, oos_rr, tpd=1)
        else:
            p_pass = float("nan")
        pf_s = f"{oos_pf:.2f}" if oos_pf != float("inf") else "inf"
        wf_s = f"{wf_eff:.2f}" if wf_eff == wf_eff else "n/a"
        pp_s = f"{p_pass:.1%}" if p_pass == p_pass else "n/a"
        print(f"  POOLED: n={oos_trades}  win_rate={oos_wr:.1%}  avg_RR={oos_rr:.2f}  "
              f"E_net={oos_e_r:+.3f}R  PF={pf_s}  max_dd={oos_max_dd:.1f}%  wf_eff={wf_s}  MC_P(pass)={pp_s}")
        summary.append((name, oos_trades, oos_e_r, oos_pf, p_pass))

    print("\n" + "=" * 96)
    print("SUMMARY (ranked by E_net)")
    print("=" * 96)
    ranked = sorted([s for s in summary if s[2] is not None], key=lambda x: x[2], reverse=True)
    for name, n, e_r, pf, pp in ranked:
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        pp_s = f"{pp:.1%}" if pp == pp else "n/a"
        flag = "  <-- POSITIVE EDGE" if e_r > 0 else ""
        print(f"  {name:24} n={n:<6} E_net={e_r:+.3f}R  PF={pf_s}  MC_P(pass)={pp_s}{flag}")

    print("\n" + "=" * 96)
    print("ROBUSTNESS CHECK: evaluate directly on the FIRST 30% of each symbol's bars")
    print("=" * 96)
    for name, cls, params, tp, trail in STRATS:
        print(f"--- {name} ---")
        run_split(name, cls, params, tp, trail, first_frac=0.30)


if __name__ == "__main__":
    main()
