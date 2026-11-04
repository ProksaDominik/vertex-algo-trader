"""Stress the best intraday candidates from futures_hypotheses.py.

Plateau test (backtest-expert): a real effect should hold across a
neighbourhood of settings and on related markets, not at one lucky value.

  * Volatility breakout (open +/- k x prior-day range, trend-filtered), k from
    0.4 to 1.2, on all five markets.
  * Opening-range breakout with the daily-trend filter, opening range of 15 to
    60 minutes, last entry 11:00 to 13:00 ET.

Prints, for each: trades, $/trade (base and 2x costs), t, both halves and the
share of positive years, plus a year-by-year table for the strongest.
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

M = fh.M


def orb_variant(days, dd, cost, or_bars: int, last_entry: int, target_r, filt: str) -> list:
    out = []
    for x in days:
        atr, trend = dd.at[x.date, "atr14"], dd.at[x.date, "trend"]
        if pd.isna(atr):
            continue
        orh, orl = x.h[:or_bars].max(), x.l[:or_bars].min()
        if orh - orl <= 0:
            continue
        for i in range(or_bars, len(x.m) - 1):
            if x.m[i] >= last_entry:
                break
            side = 1 if x.c[i] > orh else (-1 if x.c[i] < orl else 0)
            if side == 0:
                continue
            if filt == "trend" and side != trend:
                break
            entry = x.o[i + 1]
            stop = orl if side > 0 else orh
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                break
            tgt = entry + side * target_r * abs(entry - stop) if target_r else None
            px, why = fh.walk(x, i + 1, side, stop, tgt, M(15, 45), cost)
            fh.trade(out, x.date, side, entry, px, stop, why, cost)
            break
    return out


def summarize(rows: list) -> dict:
    t = pd.DataFrame(rows)
    return fh.metrics(t) if not t.empty else {"n": 0}


def main() -> int:
    fh.keep_awake()
    data = {}
    for mkt, (proxy, contract) in fh.MARKETS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        data[mkt] = (days, fh.daily_frame(days), contract)

    rows = []
    for mkt, (days, dd, contract) in data.items():
        for k in (0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.2):
            for filt in ("none", "trend"):
                b = summarize(fh.h4_volbreak(days, dd, fh.Cost(contract), k, filt))
                s = summarize(fh.h4_volbreak(days, dd, fh.Cost(contract, 2.0), k, filt))
                rows.append({"test": "volbreak", "market": mkt, "k": k, "filt": filt, **b,
                             "usd_2x": s.get("usd_mean")})
        for or_bars in (1, 2, 3, 4):
            for last in (M(11, 0), M(12, 0), M(13, 0)):
                for tr in (2.0, None):
                    b = summarize(orb_variant(days, dd, fh.Cost(contract), or_bars, last, tr, "trend"))
                    s = summarize(orb_variant(days, dd, fh.Cost(contract, 2.0), or_bars, last, tr, "trend"))
                    rows.append({"test": "orb_trend", "market": mkt, "or_min": 15 * or_bars,
                                 "last_entry": f"{last // 60}:{last % 60:02d}", "target": tr or "EOD",
                                 **b, "usd_2x": s.get("usd_mean")})
    res = pd.DataFrame(rows)
    res.to_csv(fh.OUT / "robustness_results.csv", index=False)
    pd.set_option("display.width", 250)
    cols_v = ["market", "k", "filt", "n", "usd_mean", "usd_2x", "t", "h1", "h2", "years_pos", "per_week"]
    v = res[res.test == "volbreak"]
    print("=== Volatility breakout neighbourhood (share of cells positive by market/filter) ===")
    print(v.groupby(["market", "filt"]).agg(cells=("usd_mean", "size"),
                                             positive=("usd_mean", lambda s: (s > 0).mean()),
                                             both_halves=("h1", lambda s: 0),
                                             mean_usd=("usd_mean", "mean"), mean_t=("t", "mean"))
          .drop(columns="both_halves").to_string(float_format=lambda x: f"{x:.2f}"))
    print(v[v.filt == "trend"][cols_v].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    o = res[res.test == "orb_trend"]
    print("\n=== ORB + trend filter neighbourhood (share of cells positive by market) ===")
    print(o.groupby("market").agg(cells=("usd_mean", "size"), positive=("usd_mean", lambda s: (s > 0).mean()),
                                  both_halves=("h1", lambda s: 0), mean_usd=("usd_mean", "mean"),
                                  mean_t=("t", "mean")).drop(columns="both_halves")
          .to_string(float_format=lambda x: f"{x:.2f}"))
    top = o.sort_values("t", ascending=False).head(8)
    print(top[["market", "or_min", "last_entry", "target", "n", "usd_mean", "usd_2x", "t", "h1", "h2",
               "years_pos"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # year by year for the best volatility-breakout cell
    best = v[v.filt == "trend"].sort_values("t", ascending=False).iloc[0]
    days, dd, contract = data[best.market]
    t = pd.DataFrame(fh.h4_volbreak(days, dd, fh.Cost(contract), best.k, "trend"))
    yr = t.groupby(pd.to_datetime(t["date"]).dt.year)["usd"].agg(["size", "mean", "sum"])
    print(f"\n=== year by year: volbreak {best.market} k={best.k} trend ($ per 1 micro) ===")
    print(yr.to_string(float_format=lambda x: f"{x:.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
