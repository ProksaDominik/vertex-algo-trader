"""The free data sources as race predictors (2026-10-07, owner: "do all the free ones").

Data from scripts/fetch_free_data.py (data/free/):
  * CFTC Traders in Financial Futures, E-mini Nasdaq-100, weekly since 2006 -- what hedge
    funds ("leveraged funds"), asset managers, dealers and small traders hold. Used only
    after its Friday release (positions are as of the Tuesday before).
  * Cboe option gauges, daily: VIX (now from 1990 -- before, NinjaTrader gave it from 2021
    only), VXN (Nasdaq VIX), the volatility curve VIX9D / VIX / VIX3M, SKEW, VVIX, VIX1D.
  * Cboe put/call ratios, daily since 2006 (+ put/call open interest from Oct 2019).
Every daily value is from a session that closed before the 19:00 ET race.

Same test as every earlier search (lucid_vix_tick_search.evaluate): race 2 NQ +75.5 / -50
pts, rules by sign and by past-only thirds, ranked on 2016-2022, judged on 2023-2026 (days
never seen), false-discovery check, then full evaluation / funded replays of the best.

Usage:  .venv/Scripts/python.exe -I scripts/lucid_free_data_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402
import lucid_vix_tick_search as vt  # noqa: E402

FREE = ROOT / "data" / "free"
A19 = lp.S(19)


def z(x: pd.Series, n: int = 252, min_n: int = 120) -> pd.Series:
    """How unusual today's value is vs the past year (0 = normal; the window ends today)."""
    m, s = x.rolling(n, min_periods=min_n).mean(), x.rolling(n, min_periods=min_n).std()
    return (x - m) / s.replace(0, np.nan)


def cot_signals(idx: pd.DatetimeIndex) -> dict:
    c = pd.read_parquet(FREE / "cot_nq_tff.parquet")
    oi = c["open_interest_all"]
    net = {
        "hedge funds": (c["lev_money_positions_long"] - c["lev_money_positions_short"]) / oi,
        "asset managers": (c["asset_mgr_positions_long"] - c["asset_mgr_positions_short"]) / oi,
        "dealers": (c["dealer_positions_long_all"] - c["dealer_positions_short_all"]) / oi,
        "small traders": (c["nonrept_positions_long_all"] - c["nonrept_positions_short_all"]) / oi,
    }
    out = {}
    for who, x in net.items():
        out[f"CFTC {who}: net position vs past year"] = (A19, vt.known_before(z(x, 52, 26), idx), "cont")
        out[f"CFTC {who}: weekly change"] = (A19, vt.known_before(x.diff(), idx), "cont")
    out["CFTC open interest: weekly change"] = (A19, vt.known_before(oi.pct_change(), idx), "cont")
    return out


def cboe_signals(idx: pd.DatetimeIndex) -> dict:
    i = pd.read_parquet(FREE / "cboe_indices.parquet")
    vix, vxn = i["VIX"].dropna(), i["VXN"].dropna()
    rank = vix.rolling(252, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    curve = (i["VIX3M"] / i["VIX"] - 1).dropna()                         # > 0 calm, < 0 fear now
    short_end = (i["VIX9D"] / i["VIX"] - 1).dropna()                     # > 0 stress this week
    nas = (vxn / vix.reindex(vxn.index) - 1).dropna()                   # Nasdaq fear vs S&P fear
    one_day = (i["VIX1D"] / i["VIX"] - 1).dropna()
    kb = vt.known_before
    return {
        "VIX level vs past year (rank)": (A19, kb(rank - 0.5, idx), "cont"),
        "VIX change yesterday": (A19, kb(vix.pct_change(), idx), "cont"),
        "VIX vs its 10-day average": (A19, kb(vix / vix.rolling(10).mean() - 1, idx), "cont"),
        "VIX curve 3M/1M (calm > 0)": (A19, kb(curve, idx), "cont"),
        "VIX curve vs past year": (A19, kb(z(curve), idx), "cont"),
        "VIX 9-day over 30-day (stress > 0)": (A19, kb(short_end, idx), "cont"),
        "VXN (Nasdaq VIX) change yesterday": (A19, kb(vxn.pct_change(), idx), "cont"),
        "VXN over VIX vs past year": (A19, kb(z(nas), idx), "cont"),
        "SKEW vs past year": (A19, kb(z(i["SKEW"].dropna()), idx), "cont"),
        "SKEW change yesterday": (A19, kb(i["SKEW"].dropna().diff(), idx), "cont"),
        "VVIX vs past year": (A19, kb(z(i["VVIX"].dropna()), idx), "cont"),
        "VVIX over VIX vs past year": (A19, kb(z((i["VVIX"] / i["VIX"]).dropna()), idx), "cont"),
        "VIX1D over VIX (from 2022)": (A19, kb(z(one_day, 252, 60), idx), "cont"),
    }


def putcall_signals(idx: pd.DatetimeIndex) -> dict:
    p = pd.read_parquet(FREE / "cboe_putcall.parquet")
    kb = vt.known_before
    out = {}
    for k, name in (("pc_total", "total"), ("pc_equity", "equity (stocks)"), ("pc_index", "index")):
        x = p[k].dropna()
        out[f"Put/call {name} vs past year"] = (A19, kb(z(x), idx), "cont")
        out[f"Put/call {name}, 5-day average vs past year"] = (A19, kb(z(x.rolling(5).mean()), idx), "cont")
        out[f"Put/call {name} change yesterday"] = (A19, kb(x.diff(), idx), "cont")
    oi = p["oi_pc_equity"].dropna()
    out["Put/call open interest, stocks (from 2019) vs past year"] = (A19, kb(z(oi, 252, 60), idx), "cont")
    return out


def main() -> int:
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    parts = [("CFTC positions (hedge funds, asset managers, dealers, small traders)", cot_signals(idx)),
             ("Cboe option gauges (VIX family, SKEW, VVIX)", cboe_signals(idx)),
             ("Cboe put/call ratios", putcall_signals(idx))]
    found = []
    for title, sigs in parts:
        res = vt.evaluate(g, sigs, 2023, title)
        found.append(res.assign(part=title))
    allr = pd.concat(found, ignore_index=True)
    ok = allr["A n"] >= 60
    allr["BH all"] = False
    allr.loc[ok, "BH all"] = ss.bh_fdr(allr.loc[ok, "A p"].to_numpy(), 0.10)
    held = ok & (allr["A z"] >= 2) & (allr["B z"] >= 1.65)
    print(f"\nALL FREE DATA: {int(ok.sum())} rules; {int(allr['BH all'].sum())} pass the false-discovery check "
          f"across all of them; {int(held.sum())} rank well on 2016-22 AND hold on 2023-26 (z >= 1.65).")
    print("\nFull replays (evaluation pass 2019-22 / 2023-26, funded $ to you) of the 6 best-ranked rules:")
    rows = [{"rule": "always buy", **ss.replays(g, A19, np.ones(len(idx), int))}]
    tr = np.array([1 if v else -1 for v in (g.daily.reindex(idx)["prev_close"] >=
                                             g.daily.reindex(idx)["prev_close"].rolling(200, min_periods=160).mean())])
    rows.append({"rule": "trend 200 (deployed since 7 Oct)", **ss.replays(g, A19, tr)})
    for _, r in allr[ok].sort_values("A z", ascending=False).head(6).iterrows():
        side = r["_side"].reindex(idx).fillna(0).to_numpy().astype(int)
        rows.append({"rule": f"{r['signal']} | {r['rule']}"[:72], **ss.replays(g, int(r["_a"]), side)})
    pd.set_option("display.width", 230)
    print(pd.DataFrame(rows).to_string(index=False,
                                        float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    allr.drop(columns=["_side", "_a"]).to_csv(FREE / "free_data_search.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
