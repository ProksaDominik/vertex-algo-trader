"""Apply the project's REAL, already-built SessionLearner (trade_review.py)
retroactively to the closest-to-breakeven candidates found so far, to see
if it can push any of them into genuine positive expectancy.

Methodology (no lookahead / no cherry-picking):
  1. Run the strategy on TRAIN (first 70%) and TEST (last 30%) separately.
  2. Feed EVERY train trade's (symbol, session, r_multiple) into a
     SessionLearner. This derives an avoid-list using ONLY train data.
  3. Replay TEST trades through that (frozen) learner: any trade whose
     (symbol, session) is already flagged 'avoid' from train is dropped;
     trades in never-before-seen-as-bad sessions still count.
  4. Compare TEST E_net/PF before vs after the filter.

This is the honest way to test whether the existing session-avoidance
mechanism actually helps -- the avoid-list is derived strictly from data
the filtered evaluation never sees.

Usage: python scripts/session_filter_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_supertrend import SupertrendStrategy  # noqa: E402
from metatraderbot.strategies_smc_bos import SMCBOSStrategy  # noqa: E402
from metatraderbot.trade_review import SessionLearner, classify_session  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

RISK_USD = 100.0
BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "US30", "US500", "DE40", "XAUUSD",
]

STRATS = [
    ("Supertrend", SupertrendStrategy, {
        "supertrend": {"period": 10, "multiplier": 3.0}, "exit": {"tp_r_multiple": 2.0},
    }),
    ("SMC_BOS", SMCBOSStrategy, {
        "smc_bos": {"swing_lookback": 15, "sweep_buffer_pips": 1.0, "bos_lookback": 8,
                     "max_wait_bars": 15, "sl_buffer_pips": 3.0},
        "exit": {"tp_r_multiple": 1.5},
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
    strat = {"smc_bos": {}, "ichimoku": {}, "supertrend": {}, "session": None,
             "exit": {"tp_r_multiple": 2.0}}
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


def run_one(cls, params: dict, sym: str, bars: pd.DataFrame) -> list[dict]:
    """Returns per-trade dicts: {symbol, session, r}."""
    cfg = cfg_for(sym, params)
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    out = []
    for t in bt.closed:
        hour = pd.Timestamp(t.entered_at).hour
        session = classify_session(hour)
        out.append({"symbol": sym, "session": session, "r": t.pnl_usd / RISK_USD})
    return out


def summarize(label: str, rs: list[float]) -> None:
    n = len(rs)
    if n < 5:
        print(f"  {label}: n={n} -- too few")
        return
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    wr = len(wins) / n
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
    print(f"  {label}: n={n}  win_rate={wr:.1%}  E_net={e_r:+.3f}R  PF={pf_s}  MC_P(pass)={pp_s}")


def main() -> None:
    print("=" * 96)
    print("SESSION-FILTER RETROACTIVE TEST -- can the existing SessionLearner rescue a candidate?")
    print("=" * 96)

    for name, cls, params in STRATS:
        print(f"\n--- {name} ---")
        train_trades, test_trades = [], []
        for sym in BASKET:
            bars = load(sym)
            if bars is None:
                continue
            split = int(len(bars) * 0.70)
            train_bars, test_bars = bars.iloc[:split], bars.iloc[split:]
            try:
                train_trades += run_one(cls, params, sym, train_bars)
                test_trades += run_one(cls, params, sym, test_bars)
            except Exception as e:
                print(f"  [ERROR] {sym}: {e!r}")

        raw_rs = [t["r"] for t in test_trades]
        summarize("TEST unfiltered", raw_rs)

        for min_trades, avoid_r, label in [
            (6, -3.0, "loose (min_trades=6, avoid<=-3.0R)"),
            (10, -5.0, "stricter (min_trades=10, avoid<=-5.0R)"),
            (15, -8.0, "strictest (min_trades=15, avoid<=-8.0R)"),
        ]:
            learner = SessionLearner(min_trades=min_trades, avoid_below_net_r=avoid_r)
            for t in train_trades:
                learner.observe(t["symbol"], t["session"], t["r"])
            n_avoid = sum(
                1 for sym in {t["symbol"] for t in train_trades}
                for sess in ("asian", "london", "ny_overlap", "ny_pm", "late")
                if learner.should_avoid(sym, sess)
            )
            filtered_rs = [t["r"] for t in test_trades if not learner.should_avoid(t["symbol"], t["session"])]
            print(f"  [{label}] learned {n_avoid} avoid-pairs")
            summarize(f"  TEST filtered ({label})", filtered_rs)


if __name__ == "__main__":
    main()
