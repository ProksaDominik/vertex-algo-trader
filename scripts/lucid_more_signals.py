"""More predictors (2026-10-03, owner: "keep looking, find the best prediction strategy").

  M  machine learning on REAL exchange data (NinjaTrader bars, Apr 2021 - Oct 2026):
     every order-flow / cross-market signal of lucid_orderflow_search plus NQ's
     own price features, logistic regression and gradient boosting, walk-forward
     by calendar year (train on all earlier years, test the next; 2023-2026 out
     of sample). Rules: race the side the model favours / long only when it is
     confident.
  K  calendar effects not tested before, on the 2016-2026 proxy history:
     options-expiry Fridays (monthly and quarterly), the day before a US market
     holiday, and Bitcoin's weekend move as the signal for Monday's race.

Scoring as in lucid_signal_search (race 2 NQ, +75.5 / -50 pts).

Usage:  .venv/Scripts/python.exe scripts/lucid_more_signals.py
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import lucid_orderflow_search as of  # noqa: E402
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402
from metatraderbot import lucid_rules  # noqa: E402

warnings.filterwarnings("ignore")
S, K, B = lp.S, lp.K, lp.RTH_LAST


def zscore(p, p0, n):
    return (p - p0) / math.sqrt(p0 * (1 - p0) / n) if n else float("nan")


def ml_on_real_data() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    G = {m: lp.load_grid(m, path=p) for m, p in of.NT.items() if p.exists()}
    g = G["NQ"]
    idx = pd.to_datetime(g.dates)
    VV = {m: of.volume_grid(of.NT[m], G[m].dates) for m in G}
    for m in list(G):                                  # other grids onto NQ's dates
        if m == "NQ":
            continue
        pos = pd.Series(np.arange(len(G[m].dates)), index=pd.to_datetime(G[m].dates)).reindex(idx).to_numpy()
        ok = ~np.isnan(pos)
        ti = np.where(ok, pos, 0).astype(int)
        for attr in ("O", "H", "L", "C"):
            arr = getattr(G[m], attr)[ti].copy()
            arr[~ok] = np.nan
            setattr(G[m], attr, arr)
        v = VV[m][ti].copy()
        v[~ok] = 0
        VV[m] = v
    sig = of.signals(G, VV, idx)
    X = pd.DataFrame({k: v[1] for k, v in sig.items() if v[0] == S(19) and v[2] == "cont"}, index=idx)
    d = g.daily.reindex(idx)
    c = d["close"].ffill()
    X["rth1"] = d["prev_rth_ret"]
    X["r5"] = (c / c.shift(5) - 1).shift(1)
    X["r20"] = (c / c.shift(20) - 1).shift(1)
    X["atr"] = d["atr14"] / d["prev_close"]
    X["dow"] = idx.dayofweek
    lab = ss.race_labels(g, S(19))
    yL, yS = pd.Series(lab[1], index=idx), pd.Series(lab[-1], index=idx)
    X = X.replace([np.inf, -np.inf], np.nan)
    years = idx.year
    models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
              "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=200,
                                                                 l2_regularization=2.0, min_samples_leaf=40)}
    print(f"M: {X.shape[1]} features on {len(X)} real-data days ({idx[0].date()} .. {idx[-1].date()})")
    for name, fn in models.items():
        preds = {}
        for tgt, y in (("L", yL), ("S", yS)):
            p = pd.Series(np.nan, index=idx)
            for Y in range(2023, 2027):
                tr = (years < Y) & (idx < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=7)) & y.notna() & X.notna().all(1)
                te = (years == Y) & X.notna().all(1)
                if tr.sum() < 150 or te.sum() == 0:
                    continue
                m = fn()
                m.fit(X[tr], y[tr].astype(int))
                p[te] = m.predict_proba(X[te])[:, 1]
            preds[tgt] = p
        oos = (years >= 2023) & preds["L"].notna() & preds["S"].notna()
        okL = oos & yL.notna()
        auc = roc_auc_score(yL[okL], preds["L"][okL]) if okL.sum() > 50 else float("nan")
        base = float(yL[oos].mean())
        side = np.where(preds["L"] >= preds["S"], 1, -1)
        yy = np.where(side > 0, yL, yS)
        msk = oos & ~np.isnan(yy)
        w = float(np.nanmean(yy[msk]))
        print(f"  {name}: AUC (long race) {auc:.3f} | side by model wins {w:.3f} vs always-long {base:.3f} "
              f"(z {zscore(w, base, int(msk.sum())):+.2f}, n {int(msk.sum())})")
        for q in (0.5, 0.67):
            thr = preds["L"][oos].quantile(q)
            sel = oos & (preds["L"] >= thr) & yL.notna()
            wl = float(yL[sel].mean())
            print(f"     long only when the model's long-win estimate is in its top {1 - q:.0%}: wins {wl:.3f} "
                  f"(z {zscore(wl, base, int(sel.sum())):+.2f}, n {int(sel.sum())})")


def calendar() -> None:
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    years = np.asarray(idx.year)
    lab = ss.race_labels(g, S(19))
    base = {p: ss.score(pd.Series(1, index=idx), lab, years, P)[0] for p, P in (("A", (2016, 2022)), ("B", (2023, 2026)))}
    days = pd.Series(idx, index=idx)
    third_fri = days[(idx.dayofweek == 4) & (idx.day >= 15) & (idx.day <= 21)]
    opex = days.isin(third_fri)
    qopex = opex & idx.month.isin([3, 6, 9, 12])
    week_opex = pd.Series(idx.to_period("W").isin(third_fri.index.to_period("W")), index=idx)
    hol = pd.DatetimeIndex(sorted(pd.to_datetime([str(d) for d in lucid_rules.HOLIDAY_SESSIONS])))
    nxt = pd.Series(idx, index=idx).shift(-1)
    pre_hol = pd.Series([any((h > d) and (h <= n) for h in hol) if pd.notna(n) else False
                         for d, n in zip(idx, nxt)], index=idx)
    # Bitcoin's weekend: Friday 16:00 ET -> Sunday 18:00 ET, for Monday's race
    btc = pd.read_parquet(ROOT / "data" / "btcusd_15m_long.parquet")
    bidx = pd.DatetimeIndex(btc.index if "timestamp" not in btc.columns else btc["timestamp"])
    bidx = bidx.tz_convert(lucid_rules.ET) if bidx.tz is not None else bidx
    bc = pd.Series(btc["close"].to_numpy(), index=bidx).sort_index()
    wk = pd.Series(np.nan, index=idx)
    for d in idx[idx.dayofweek == 0]:
        fri16 = pd.Timestamp(d.date() - pd.Timedelta(days=3)) + pd.Timedelta(hours=16)
        sun18 = pd.Timestamp(d.date() - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)
        if bc.index.tz is not None:
            fri16, sun18 = fri16.tz_localize(lucid_rules.ET), sun18.tz_localize(lucid_rules.ET)
        a_, b_ = bc[:fri16], bc[:sun18]
        if len(a_) and len(b_) and b_.index[-1] > fri16:
            wk[d] = b_.iloc[-1] / a_.iloc[-1] - 1
    sigs = {"options-expiry Friday": (opex, "bin"), "quarterly expiry Friday": (qopex, "bin"),
            "options-expiry week": (week_opex, "bin"), "day before a market holiday": (pre_hol, "bin"),
            "Bitcoin weekend move (Mondays)": (wk, "cont")}
    rows = []
    for name, (x, kind) in sigs.items():
        x = x.astype(float)
        rules = ss.rules_for(name, x, kind)
        if kind == "cont":                              # Mondays only: sign rules + long if BTC rose
            mon = idx.dayofweek == 0
            rules = {"side = sign": pd.Series(np.where(mon, np.sign(x.fillna(0)), 0), index=idx),
                     "long if BTC rose, else skip": pd.Series(np.where(mon & (x > 0), 1, 0), index=idx),
                     "long if BTC fell, else skip": pd.Series(np.where(mon & (x < 0), 1, 0), index=idx)}
        for r, side in rules.items():
            pa, na, _ = ss.score(side, lab, years, (2016, 2022))
            pb, nb, _ = ss.score(side, lab, years, (2023, 2026))
            rows.append({"signal": name, "rule": r, "2016-22 win": pa, "n": na, "z": zscore(pa, base["A"], na),
                         "2023-26 win": pb, "n ": nb, "z ": zscore(pb, base["B"], nb)})
    pd.set_option("display.width", 220)
    print(f"\nK: calendar (proxy 2016-2026); deployed race wins {base['A']:.3f} (2016-22) / {base['B']:.3f} (2023-26)")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.2f}" if abs(v) > 1 else f"{v:.3f}"))


if __name__ == "__main__":
    ml_on_real_data()
    calendar()
