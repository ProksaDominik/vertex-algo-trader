"""Fallback track: edges with holds of 1-3 days (NOT tradeable on LucidPro).

The owner accepts holds of up to ~3 days if nothing intraday works. Lucid
closes everything at 4:45 PM ET, so these would need a firm/account that allows
overnight positions. This measures whether that is worth it: are documented
multi-day edges clearly stronger than the intraday ones?

  RSI2  (Connors): long at the 16:00 ET close when close > 200-day SMA and
        RSI(2) < 10; exit at the close once close > 5-day SMA, or after 3 days.
  TOM   turn of the month: long at the close of the month's last trading day,
        out at the close of the 2nd trading day of the new month (3 days).
  BO20  20-day closing-high breakout long / low breakout short, out after 3
        days (no stop -- the hold is the risk control being measured).

Daily bars are the 09:30-16:00 ET session from the 10-year 15m proxies; costs
are the futures ones (tick + Lucid commission both ways). Overnight gaps are
real P&L here -- exactly the risk a multi-day account carries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run(dd: pd.DataFrame, contract: str, rule: str) -> pd.DataFrame:
    cost = fh.Cost(contract)
    c = dd["close"]
    sma200, sma5 = c.rolling(200).mean(), c.rolling(5).mean()
    r2 = rsi(c, 2)
    hi20, lo20 = c.rolling(20).max().shift(1), c.rolling(20).min().shift(1)
    dates = list(dd.index)
    month = pd.Series([d.month for d in dates], index=dd.index)
    last_of_month = month != month.shift(-1)
    rows, i = [], 0
    while i < len(dates) - 1:
        side, hold = 0, 3
        if rule == "RSI2" and c.iloc[i] > sma200.iloc[i] and r2.iloc[i] < 10:
            side = 1
        elif rule == "TOM" and last_of_month.iloc[i]:
            side = 1
        elif rule == "BO20" and not pd.isna(hi20.iloc[i]):
            side = 1 if c.iloc[i] > hi20.iloc[i] else (-1 if c.iloc[i] < lo20.iloc[i] else 0)
        if side == 0:
            i += 1
            continue
        entry = c.iloc[i] + side * cost.tick
        j = i
        for k in range(1, hold + 1):
            j = i + k
            if j >= len(dates) - 1:
                break
            if rule == "RSI2" and c.iloc[j] > sma5.iloc[j]:
                break
        exit_px = c.iloc[j] - side * cost.tick
        pts = (exit_px - entry) * side - cost.comm_pts
        rows.append({"date": dates[i], "side": side, "usd": pts * cost.pv, "days": j - i, "risk_pts": np.nan})
        i = j + 1                                    # one position at a time
    return pd.DataFrame(rows)


def main() -> int:
    rows = []
    for mkt, (proxy, contract) in fh.MARKETS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = fh.daily_frame(days)
        for rule in ("RSI2", "TOM", "BO20"):
            t = run(dd, contract, rule)
            m = fh.metrics(t) if not t.empty else {"n": 0}
            rows.append({"market": mkt, "contract": contract, "rule": rule, **m,
                         "avg_days": t["days"].mean() if not t.empty else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(fh.OUT / "multiday_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(res[["market", "rule", "n", "usd_mean", "win", "pf", "t", "h1", "h2", "years_pos",
               "max_dd_usd", "per_week", "avg_days"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
