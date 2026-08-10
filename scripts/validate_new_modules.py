"""Definition-of-Done validation (CLAUDE.md spec Section 12.3) for the 4 new
modules: AsianFade, LondonBreakout, LiquiditySweep, H4Pullback.

Real backtest (metatraderbot.backtest.Backtest) over cached 15m parquet
data, resampled to 30m, 70% train / 30% out-of-sample validation split,
across the 5-symbol basket both books trade. Feeds the OOS win-rate +
average reward:risk into research/mc3.py's own Monte Carlo simulator
(verbatim from the pasted CLAUDE.md research) for P(pass).

Approximation: mc3.py's sim() assumes each trade risks a fixed fraction and
wins RR*risk or loses risk. Real trade R-multiples vary (partial exits via
the trail, early stop-outs); avg-win-R / avg-loss-R is used as a single RR
input, the same simplification professional MC validators typically make.

Usage: python scripts/validate_new_modules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_asian_fade import AsianFadeStrategy  # noqa: E402
from metatraderbot.strategies_h4_pullback import H4PullbackStrategy  # noqa: E402
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_sweep_reclaim import SweepReclaimStrategy  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402 -- research/mc3.py

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
RISK_USD = 100.0  # matches portfolio_ftmo.yaml's fallback risk_usd_per_trade

STRATS = [
    ("AsianFade", AsianFadeStrategy, {
        "asian_fade": {"session_start_hour_utc": 0, "session_end_hour_utc": 7,
                        "fade_start_hour_utc": 7, "fade_end_hour_utc": 16,
                        "min_range_pips": 15.0, "sl_buffer_pips": 3.0, "lookback_bars": 96},
        "exit": {"tp_r_multiple": 1.5},
    }, "reversion"),
    ("LondonBreakout", LondonBreakoutStrategy, {
        "london_breakout": {"session_start_hour_utc": 0, "session_end_hour_utc": 7,
                             "breakout_start_hour_utc": 7, "breakout_end_hour_utc": 10,
                             "min_clearance_pips": 2.0, "sl_buffer_pips": 3.0,
                             "min_d1_atr_pips": 40.0, "lookback_bars": 96},
        "exit": {"tp_r_multiple": 2.0},
    }, "trend"),
    ("LiquiditySweep", SweepReclaimStrategy, {
        "sweep": {"lookback": 20, "sl_buffer_atr": 0.25},
        "exit": {"tp_r_multiple": 1.0},
    }, "reversion"),
    ("H4Pullback", H4PullbackStrategy, {
        "h4_pullback": {"htf_multiplier": 8, "htf_lookback": 8, "fast_period": 21,
                         "slow_period": 50, "pullback_tolerance_pips": 3.0,
                         "sl_swing_lookback": 10, "confirmation_min_body_pct": 0.4},
        "exit": {"tp_r_multiple": 2.0},
    }, "trend"),
]


class Windowed:
    """Sliding window so O(n^2) strategy recompute stays tractable -- same
    technique compare_strategies.py / walk_forward.py already use."""
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


def cfg_for(sym: str, params: dict, style: str) -> dict:
    trail = 1.0 if style == "trend" else 0.0
    strat = {
        "trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "donchian": {}, "orb": {},
        "macd": {}, "bos": {}, "sweep": {}, "tod": {}, "vwap": {}, "session": None,
        "asian_fade": {}, "london_breakout": {}, "h4_pullback": {},
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
                    "trail_start_r": trail, "trail_lock_fraction": 0.7},
        "strategy": strat,
    }


def run_one(cls, params: dict, style: str, sym: str, bars: pd.DataFrame) -> dict:
    cfg = cfg_for(sym, params, style)
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    r = bt.report()
    r["rs"] = [t.pnl_usd / RISK_USD for t in bt.closed]
    return r


def main() -> None:
    print("=" * 78)
    print("DEFINITION-OF-DONE VALIDATION -- 4 new modules, real backtest")
    print("70% train / 30% out-of-sample split, 5-symbol basket, up to last")
    print("6000 30m bars per symbol (~4-5 months, ~37-day OOS tail per symbol)")
    print("=" * 78)

    summary = []
    for name, cls, params, style in STRATS:
        train_rs: list[float] = []
        oos_rs: list[float] = []
        oos_trades = oos_wins = 0
        oos_dd_pcts: list[float] = []

        for sym in SYMBOLS:
            bars = load(sym)
            if bars is None:
                print(f"  [skip] {name} {sym}: no cached data")
                continue
            split = int(len(bars) * 0.70)
            train_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]

            r_train = run_one(cls, params, style, sym, train_bars)
            r_oos = run_one(cls, params, style, sym, oos_bars)

            train_rs += r_train["rs"]
            oos_rs += r_oos["rs"]
            oos_trades += r_oos["trades"]
            oos_wins += r_oos["wins"]
            oos_dd_pcts.append(r_oos["max_drawdown_pct"])

            print(f"  {name:16}{sym:8} train n={r_train['trades']:4d} "
                  f"pnl=${r_train['total_pnl_usd']:8.0f}   "
                  f"OOS n={r_oos['trades']:4d} pnl=${r_oos['total_pnl_usd']:8.0f} "
                  f"dd={r_oos['max_drawdown_pct']:6.1f}%")

        print(f"\n--- {name}: OOS aggregate ({oos_trades} trades across {len(SYMBOLS)} symbols) ---")
        if oos_trades == 0:
            print("  ZERO OOS trades -- cannot evaluate. FAIL (insufficient sample).")
            summary.append((name, oos_trades, None, None, False))
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

        print(f"  win_rate={oos_wr:.1%}  avg_RR={oos_rr:.2f}  E_net={oos_e_r:+.3f}R  "
              f"PF={oos_pf:.2f}  max_dd={oos_max_dd:.1f}%  wf_eff={wf_eff:.2f}  "
              f"MC_P(pass)={p_pass:.1%}" if p_pass == p_pass else "  (MC P(pass): n/a)")

        dod = {
            "trade count (OOS) >= 200": (oos_trades, oos_trades >= 200),
            "E_net per trade >= 0.15R": (oos_e_r, oos_e_r >= 0.15),
            "profit factor >= 1.25": (oos_pf, oos_pf >= 1.25),
            "max drawdown <= 8%": (oos_max_dd, oos_max_dd >= -8.0),
            "walk-forward efficiency > 0.6": (wf_eff, (wf_eff > 0.6) if wf_eff == wf_eff else False),
            "Monte Carlo P(pass) >= 65%": (p_pass, (p_pass >= 0.65) if p_pass == p_pass else False),
        }
        all_pass = True
        for label, (val, ok) in dod.items():
            all_pass &= ok
            shown = f"{val:.3f}" if isinstance(val, float) else str(val)
            print(f"    [{'PASS' if ok else 'FAIL'}] {label}: {shown}")
        verdict = "CLEARS" if all_pass else "DOES NOT CLEAR"
        print(f"  => {name}: {verdict} the Definition of Done")
        summary.append((name, oos_trades, oos_e_r, oos_pf, all_pass))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, n, e_r, pf, ok in summary:
        e_r_s = f"{e_r:+.3f}R" if e_r is not None else "n/a"
        pf_s = f"{pf:.2f}" if pf is not None else "n/a"
        print(f"  {name:16} OOS trades={n:<5} E_net={e_r_s:<8} PF={pf_s:<6} "
              f"{'CLEARS' if ok else 'DOES NOT CLEAR'} the Definition of Done")


if __name__ == "__main__":
    main()
