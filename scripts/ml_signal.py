"""ML-based direction signal: the last unexplored edge-source class after
18 single-symbol TA strategies (full_strategy_comparison.py) and 17-pair
stat-arb (pairs_stat_arb.py) both failed rigorous, pooled OOS testing.

Design, chosen specifically to resist overfitting on this small dataset:
  - ONE pooled logistic regression across ALL symbols (not per-symbol) --
    forces the model to learn a signal that generalizes across markets
    rather than memorizing one symbol's noise. Strong L2 regularization.
  - A small, economically-motivated feature set (returns, vol regime, RSI,
    MACD, Bollinger z-score, distance from EMA, session/day cyclical
    encoding) -- not hundreds of engineered features.
  - Purely causal (no lookahead): every feature at bar i uses only data up
    to and including bar i; the label is the FORWARD return over the next
    H bars, so training pairs never see their own future.
  - TIME-based walk-forward split (not random shuffle): train strictly
    precedes test in calendar time, pooled across symbols on the same
    2024 calendar so no symbol's test period overlaps another's train.
  - The OOS trading simulation (not just classification accuracy) is
    scored with the same Definition-of-Done metrics + real costs as the
    other two comparison scripts, so results are directly comparable.

Usage: python scripts/ml_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from metatraderbot.indicators import atr, ema  # noqa: E402
from metatraderbot.instruments import PAIR_SPECS  # noqa: E402
from mc3 import sim as mc_sim  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RISK_USD = 100.0
HORIZON = 8          # bars forward the label/exit looks (8 * 30m = 4h)
SL_ATR_MULT = 1.5
TP_ATR_MULT = 1.5
ENTRY_PROB = 0.52    # only trade when the model is this confident

BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "US30", "US500", "DE40", "XAUUSD",
]


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


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"]
    a = atr(bars, 14)
    ema20, ema50 = ema(c, 20), ema(c, 50)
    macd_line = ema(c, 12) - ema(c, 26)
    macd_sig = ema(macd_line, 9)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()

    f = pd.DataFrame(index=bars.index.rename(None))
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = (np.log(c) - np.log(c.shift(n))) / (a / c).replace(0, np.nan)
    f["vol_ratio"] = a / a.rolling(100).mean()
    f["rsi_z"] = (rsi(c, 14) - 50) / 50
    f["macd_hist"] = (macd_line - macd_sig) / a
    f["bb_z"] = (c - sma20) / std20.replace(0, np.nan)
    f["dist_ema20"] = (c - ema20) / a
    f["dist_ema50"] = (c - ema50) / a
    hour = bars.index.hour + bars.index.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = bars.index.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return f


FEATURE_COLS = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "vol_ratio", "rsi_z",
                 "macd_hist", "bb_z", "dist_ema20", "dist_ema50",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def main() -> None:
    print("=" * 96)
    print(f"ML SIGNAL -- pooled logistic regression, {len(BASKET)} symbols, "
          f"H={HORIZON} bars, entry_prob={ENTRY_PROB}")
    print("=" * 96)

    all_data = {}
    for sym in BASKET:
        bars = load(sym)
        if bars is None:
            continue
        feats = build_features(bars)
        fwd_ret = (np.log(bars["close"].shift(-HORIZON)) - np.log(bars["close"])) / (atr(bars, 14) / bars["close"])
        feats["label"] = (fwd_ret > 0).astype(int)
        feats["symbol"] = sym
        feats["close"] = bars["close"]
        feats["atr"] = atr(bars, 14)
        feats["timestamp"] = bars.index
        all_data[sym] = feats.dropna()

    pooled = pd.concat(all_data.values(), axis=0)
    pooled = pooled.sort_values("timestamp")
    split_ts = pooled["timestamp"].quantile(0.70, interpolation="nearest")
    train = pooled[pooled["timestamp"] < split_ts]
    test = pooled[pooled["timestamp"] >= split_ts]
    print(f"train: {len(train)} rows (< {split_ts.date()})   test: {len(test)} rows (>= {split_ts.date()})")

    scaler = StandardScaler().fit(train[FEATURE_COLS])
    X_train = scaler.transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    clf = LogisticRegression(penalty="l2", C=0.05, max_iter=2000, class_weight="balanced")
    clf.fit(X_train, train["label"])

    train_acc = clf.score(X_train, train["label"])
    test_acc = clf.score(X_test, test["label"])
    print(f"train accuracy: {train_acc:.3f}   test accuracy: {test_acc:.3f}   "
          f"(baseline 0.500 = coin flip)")
    coefs = dict(zip(FEATURE_COLS, clf.coef_[0]))
    print("feature coefficients:", {k: round(v, 3) for k, v in coefs.items()})

    p_up = clf.predict_proba(X_test)[:, 1]
    test = test.copy()
    test["p_up"] = p_up

    # --- trading simulation on OOS predictions, per symbol, with real costs ---
    rs_all: list[float] = []
    wins_all = 0
    per_symbol = []
    for sym in BASKET:
        sub = test[test["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if sub.empty or sym not in PAIR_SPECS:
            continue
        spec = PAIR_SPECS[sym]
        cm, spread, comm = spec["contract_mult"], spec["spread_price"], spec["commission_per_lot"]
        rs_sym: list[float] = []
        i = 0
        n = len(sub)
        while i < n:
            p = sub["p_up"].iloc[i]
            direction = 1 if p >= ENTRY_PROB else (-1 if p <= 1 - ENTRY_PROB else 0)
            if direction == 0:
                i += 1
                continue
            entry_price = sub["close"].iloc[i]
            a = sub["atr"].iloc[i]
            if pd.isna(a) or a <= 0:
                i += 1
                continue
            sl = entry_price - direction * SL_ATR_MULT * a
            tp = entry_price + direction * TP_ATR_MULT * a
            lots = RISK_USD / (SL_ATR_MULT * a * cm)
            exit_price = None
            j = i + 1
            while j < min(i + HORIZON, n):
                px = sub["close"].iloc[j]
                if direction == 1 and (px <= sl or px >= tp):
                    exit_price = px
                    break
                if direction == -1 and (px >= sl or px <= tp):
                    exit_price = px
                    break
                j += 1
            if exit_price is None:
                j = min(i + HORIZON, n) - 1
                exit_price = sub["close"].iloc[j]
            pnl = direction * (exit_price - entry_price) * lots * cm
            cost = (spread * cm + comm) * lots * 2
            r = (pnl - cost) / RISK_USD
            rs_sym.append(r)
            i = j + 1
        rs_all += rs_sym
        wins_all += sum(1 for x in rs_sym if x > 0)
        if rs_sym:
            per_symbol.append((sym, len(rs_sym), sum(rs_sym) / len(rs_sym)))

    print("\nPer-symbol OOS (top/bottom by trade count):")
    for sym, n, e in sorted(per_symbol, key=lambda x: -x[1])[:8]:
        print(f"  {sym:8} n={n:5d}  E_net={e:+.3f}R")

    n_tr = len(rs_all)
    if n_tr == 0:
        print("\nNo OOS trades generated.")
        return
    wins = [x for x in rs_all if x > 0]
    losses = [x for x in rs_all if x <= 0]
    wr = wins_all / n_tr
    e_r = sum(rs_all) / n_tr
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else (float("inf") if wins else 0.0)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(losses) / len(losses)) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    eq = np.cumsum(rs_all)
    peak = np.maximum.accumulate(eq)
    max_dd_r = float((eq - peak).min())

    if rr > 0 and 0 < wr < 1:
        p_pass, _, _, _ = mc_sim(0.005, wr, rr, tpd=4)
    else:
        p_pass = float("nan")

    print(f"\n{'=' * 96}\nPOOLED OOS TRADING RESULT (all {len(BASKET)} symbols combined)\n{'=' * 96}")
    print(f"  n={n_tr}  win_rate={wr:.1%}  avg_RR={rr:.2f}  E_net={e_r:+.3f}R  "
          f"PF={pf:.2f}  max_dd={max_dd_r:.1f}R  MC_P(pass)={p_pass:.1%}" if p_pass == p_pass
          else f"  n={n_tr}  win_rate={wr:.1%}  E_net={e_r:+.3f}R  PF={pf:.2f}")


if __name__ == "__main__":
    main()
