"""Comprehensive Definition-of-Done comparison across EVERY strategy in
STRATEGY_CLASSES x the full available symbol basket (intersection of cached
15m parquet data and instruments.PAIR_SPECS).

Same rigor as scripts/validate_new_modules.py (which only covered the 4
newest modules): 70/30 train/OOS split, OOS win_rate/avg_RR/E_net(R)/PF/
max_dd/walk-forward-efficiency, then research/mc3.py's real FTMO challenge
Monte Carlo simulator for P(pass). Ranks by OOS E_net(R) so genuine edge
(not guesswork) drives the roster.

Usage: python scripts/full_strategy_comparison.py [--symbols N]
  --symbols N   use only the first N symbols (for a quick smoke run)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategy import NoWickStrategy  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_donchian import DonchianBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_macd import MACDTrendStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402
from metatraderbot.strategies_ma_cross import SMACrossStrategy  # noqa: E402
from metatraderbot.strategies_stochastic import StochasticStrategy  # noqa: E402
from metatraderbot.strategies_orb import ORBStrategy  # noqa: E402
from metatraderbot.strategies_bos import MarketOpenBOSStrategy  # noqa: E402
from metatraderbot.strategies_sweep_reclaim import SweepReclaimStrategy  # noqa: E402
from metatraderbot.strategies_tod_reversion import TODReversionStrategy  # noqa: E402
from metatraderbot.strategies_vwap_reversion import VWAPReversionStrategy  # noqa: E402
from metatraderbot.strategies_bos_pullback import TrendPullbackBOSStrategy  # noqa: E402
from metatraderbot.strategies_sr_retest import SRBreakoutRetestStrategy  # noqa: E402
from metatraderbot.strategies_ema_cross import EMACrossoverStrategy  # noqa: E402
from metatraderbot.strategies_asian_fade import AsianFadeStrategy  # noqa: E402
from metatraderbot.strategies_london_breakout import LondonBreakoutStrategy  # noqa: E402
from metatraderbot.strategies_h4_pullback import H4PullbackStrategy  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402 -- research/mc3.py

# Intersection of cached data/*.parquet AND instruments.PAIR_SPECS (excludes
# FBTC/GBTC/IBIT crypto-ETF tickers and pairs missing contract specs).
BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "US30", "US500", "DE40", "XAUUSD",
]

RISK_USD = 100.0

# (label, class, extra strategy_params, style)
# style: "trend" -> TP 2R + 1R ratchet trail; "reversion" -> TP 1R (or the
# module's own tuned default, if given via exit=), no trail.
STRATS = [
    ("NoWick", NoWickStrategy, {
        "trend": {"ema_period": 50}, "signal": {"wick_max_pct": 0.25, "body_min_pct": 0.5},
        "entry": {"retrace_bars": 3}, "exit": {"sl_swing_lookback": 10},
    }, "trend"),
    ("BollingerBounce", BollingerBounceStrategy, {
        "bollinger": {"period": 20, "std_dev": 2.0, "sl_buffer_pct": 0.20},
    }, "reversion"),
    ("Donchian", DonchianBreakoutStrategy, {"donchian": {"period": 10}}, "trend"),
    ("MACD", MACDTrendStrategy, {
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "require_zero_side": True},
        "exit": {"sl_swing_lookback": 5},
    }, "trend"),
    ("RSIReversion", RSIReversionStrategy, {
        "rsi": {"period": 14, "oversold": 30, "overbought": 70}, "exit": {"sl_swing_lookback": 10},
    }, "reversion"),
    ("SMACross", SMACrossStrategy, {
        "ma_cross": {"fast_period": 10, "slow_period": 30}, "exit": {"sl_swing_lookback": 10},
    }, "trend"),
    ("Stochastic", StochasticStrategy, {
        "stochastic": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
        "exit": {"sl_swing_lookback": 10},
    }, "reversion"),
    ("ORB", ORBStrategy, {"orb": {"range_start_hour_utc": 8, "range_minutes": 30, "trade_window_hours": 4}}, "trend"),
    ("MarketOpenBOS", MarketOpenBOSStrategy, {
        "bos": {"session_open_hour_utc": 13, "trade_window_hours": 4, "swing_lookback": 40, "swing_strength": 2},
    }, "trend"),
    ("SweepReclaim", SweepReclaimStrategy, {"sweep": {"lookback": 20, "sl_buffer_atr": 0.25}}, "reversion"),
    ("TODReversion", TODReversionStrategy, {"tod": {"revert_hour_utc": 20, "stretch_atr": 1.5}}, "reversion"),
    ("VWAPReversion", VWAPReversionStrategy, {"vwap": {"band_atr": 2.0}}, "reversion"),
    ("TrendPullbackBOS", TrendPullbackBOSStrategy, {
        "bos_pullback": {"htf_multiplier": 4, "htf_lookback": 20, "swing_lookback": 20,
                          "fib_low": 0.5, "fib_high": 0.618, "sl_buffer_pips": 4.0, "max_wait_bars": 20},
    }, "trend"),
    ("SRRetest", SRBreakoutRetestStrategy, {
        "sr_retest": {"level_lookback": 80, "swing_arm": 2, "min_touches": 2, "retest_tolerance_pips": 5.0},
    }, "trend"),
    ("EMACross", EMACrossoverStrategy, {
        "ema_cross": {"fast_period": 20, "slow_period": 50, "pullback_tolerance_pips": 3.0,
                       "sl_swing_lookback": 10, "max_wait_bars": 15, "confirmation_min_body_pct": 0.4},
    }, "trend"),
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
    ("H4Pullback", H4PullbackStrategy, {
        "h4_pullback": {"htf_multiplier": 8, "htf_lookback": 8, "fast_period": 21,
                         "slow_period": 50, "pullback_tolerance_pips": 3.0,
                         "sl_swing_lookback": 10, "confirmation_min_body_pct": 0.4},
        "exit": {"tp_r_multiple": 2.0},
    }, "trend"),
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


def cfg_for(sym: str, params: dict, style: str) -> dict:
    tp = 2.0 if style == "trend" else 1.0
    trail = 1.0 if style == "trend" else 0.0
    strat = {
        "trend": {}, "signal": {}, "entry": {}, "bollinger": {}, "donchian": {}, "orb": {},
        "macd": {}, "bos": {}, "sweep": {}, "tod": {}, "vwap": {}, "session": None,
        "rsi": {}, "ma_cross": {}, "stochastic": {}, "bos_pullback": {}, "sr_retest": {},
        "ema_cross": {}, "asian_fade": {}, "london_breakout": {}, "h4_pullback": {},
        "exit": {"tp_r_multiple": tp},
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=len(BASKET))
    args = ap.parse_args()
    symbols = BASKET[: args.symbols]

    print("=" * 96)
    print(f"FULL STRATEGY COMPARISON -- {len(STRATS)} strategies x {len(symbols)} symbols, "
          "70/30 train/OOS, real Definition-of-Done + FTMO Monte Carlo")
    print("=" * 96)

    summary = []
    for name, cls, params, style in STRATS:
        train_rs: list[float] = []
        oos_rs: list[float] = []
        oos_trades = oos_wins = 0
        oos_dd_pcts: list[float] = []

        for sym in symbols:
            bars = load(sym)
            if bars is None:
                continue
            split = int(len(bars) * 0.70)
            train_bars, oos_bars = bars.iloc[:split], bars.iloc[split:]
            try:
                r_train = run_one(cls, params, style, sym, train_bars)
                r_oos = run_one(cls, params, style, sym, oos_bars)
            except Exception as e:
                print(f"  [ERROR] {name} {sym}: {e!r}")
                continue

            train_rs += r_train["rs"]
            oos_rs += r_oos["rs"]
            oos_trades += r_oos["trades"]
            oos_wins += r_oos["wins"]
            oos_dd_pcts.append(r_oos["max_drawdown_pct"])

        print(f"\n--- {name}: OOS aggregate ({oos_trades} trades across {len(symbols)} symbols) ---", flush=True)
        if oos_trades < 10:
            print(f"  too few OOS trades ({oos_trades}) -- cannot evaluate.")
            summary.append((name, oos_trades, None, None, None, None, False))
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
        print(f"  n={oos_trades:5d}  win_rate={oos_wr:.1%}  avg_RR={oos_rr:.2f}  "
              f"E_net={oos_e_r:+.3f}R  PF={pf_s}  max_dd={oos_max_dd:.1f}%  "
              f"wf_eff={wf_s}  MC_P(pass)={pp_s}", flush=True)

        summary.append((name, oos_trades, oos_e_r, oos_pf, oos_wr, p_pass, True))

    print("\n" + "=" * 96)
    print("RANKED SUMMARY (by OOS E_net per trade, descending)")
    print("=" * 96)
    ranked = sorted(
        [s for s in summary if s[2] is not None],
        key=lambda x: x[2], reverse=True,
    )
    print(f"{'strategy':<18}{'n':>7}{'E_net(R)':>11}{'PF':>8}{'win_rate':>10}{'MC_P(pass)':>12}")
    for name, n, e_r, pf, wr, pp, _ in ranked:
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        pp_s = f"{pp:.1%}" if pp == pp else "n/a"
        flag = "  <-- POSITIVE EDGE" if e_r > 0 else ""
        print(f"{name:<18}{n:>7}{e_r:>+10.3f}R{pf_s:>8}{wr:>9.1%}{pp_s:>12}{flag}")
    skipped = [s for s in summary if s[2] is None]
    for name, n, *_ in skipped:
        print(f"{name:<18}{n:>7}  (insufficient OOS trades)")


if __name__ == "__main__":
    main()
