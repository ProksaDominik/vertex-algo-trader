"""Candidate intraday stack -> real-futures check -> LucidPro Monte Carlo.

Stack: opening-range breakout with the daily-trend filter (the only intraday
candidate whose whole parameter neighbourhood was positive -- 24/24 cells on
NQ and on GC in futures_robustness.py), at the CENTRE of that plateau rather
than its best cell: 45-minute range, last entry 12:00 ET, 2R target, flat by
15:55 ET. Markets: MNQ (micro Nasdaq) and MGC (micro gold).

1. 10-year proxy trades (2016-2026) per leg, $ per 1 micro contract.
2. The same rules on the REAL futures bars from NinjaTrader (~15 months),
   compared with the proxy over the overlapping months.
3. LucidPro 50K Monte Carlo over contract counts: evaluation pass/fail odds and
   days, funded payout-before-blow odds and days.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402
import futures_robustness as fr  # noqa: E402
import lucid_montecarlo as mc  # noqa: E402

LEGS = {"MNQ": "nas100", "MGC": "xauusd"}
PARAMS = dict(or_bars=3, last_entry=fh.M(12, 0), target_r=2.0, filt="trend")


def leg_trades(bars: pd.DataFrame, contract: str) -> pd.DataFrame:
    days = fh.load_days(bars)
    dd = fh.daily_frame(days)
    t = pd.DataFrame(fr.orb_variant(days, dd, fh.Cost(contract), **PARAMS))
    t["leg"] = contract
    return t


def main() -> int:
    fh.keep_awake()
    proxy = {c: leg_trades(fh.bars_from_parquet(ROOT / "data" / f"{p}_15m_long.parquet"), c)
             for c, p in LEGS.items()}
    real = {c: leg_trades(fh.bars_from_parquet(fh.OUT / f"{c}_15m.parquet"), c) for c in LEGS}

    print("=== 1. ten-year proxy, per 1 micro contract ===")
    for c, t in proxy.items():
        m = fh.metrics(t)
        print(f"{c}: n={m['n']} ${m['usd_mean']:.2f}/trade win={m['win']:.0%} t={m['t']:.2f} "
              f"halves ${m['h1']:.2f}/${m['h2']:.2f} years+ {m['years_pos']:.0%} "
              f"maxDD ${m['max_dd_usd']:,.0f} {m['per_week']:.2f}/wk")

    print("\n=== 2. real futures bars (NinjaTrader) vs proxy, same months ===")
    for c in LEGS:
        r = real[c]
        start, end = pd.to_datetime(r["date"]).min(), pd.to_datetime(r["date"]).max()
        p = proxy[c][(pd.to_datetime(proxy[c]["date"]) >= start) & (pd.to_datetime(proxy[c]["date"]) <= end)]
        same = pd.merge(r[["date", "side", "usd"]], p[["date", "side", "usd"]], on="date", suffixes=("_fut", "_proxy"))
        agree = (same["side_fut"] == same["side_proxy"]).mean() if len(same) else np.nan
        print(f"{c}: futures n={len(r)} ${r['usd'].mean():.2f}/trade (sum ${r['usd'].sum():,.0f}) | "
              f"proxy n={len(p)} ${p['usd'].mean():.2f}/trade (sum ${p['usd'].sum():,.0f}) | "
              f"same-day trades {len(same)}, same direction {agree:.0%} | {start:%Y-%m-%d} -> {end:%Y-%m-%d}")

    trades = pd.concat(proxy.values(), ignore_index=True)
    trades.to_parquet(fh.OUT / "stack_orb_trend_trades.parquet")
    cal = pd.to_datetime(sorted({d for t in proxy.values() for d in t["date"]} |
                                {d.date() for d in pd.bdate_range("2016-01-04", "2026-08-14")}))
    cal = pd.DatetimeIndex(sorted(set(cal)))

    print("\n=== 3. LucidPro 50K Monte Carlo (20,000 runs each, 5-day blocks) ===")
    rows = []
    for n_mnq, n_mgc in product((1, 2, 3, 4), (0, 2, 4, 6)):
        pnl, low = mc.daily_series(trades, {"MNQ": n_mnq, "MGC": n_mgc}, cal)
        ev = mc.simulate(pnl, low, "eval", horizon=120)
        fu = mc.simulate(pnl, low, "funded", horizon=120)
        rows.append({"MNQ": n_mnq, "MGC": n_mgc, "units": n_mnq + n_mgc,
                     "usd_per_week": pnl.mean() * 5,
                     "eval_pass": ev["p_success"], "eval_fail": ev["p_fail"], "eval_open": ev["p_open"],
                     "eval_days_med": ev["median_days"], "fails_per_pass": ev["p_fail"] / max(ev["p_success"], 1e-9),
                     "funded_pay": fu["p_success"], "funded_blow": fu["p_fail"], "funded_days_med": fu["median_days"]})
    res = pd.DataFrame(rows)
    res.to_csv(fh.OUT / "stack_mc_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
