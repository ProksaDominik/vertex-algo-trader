"""Can machine learning lift the race above the ~41% LucidPro ceiling? (2026-10-01)

Each Lucid day the race goes in at 19:00 ET with 2 NQ: it passes if price
travels the +$3,000 target (75.5 pts) before the -$2,000 limit (50 pts, open
equity -- Lucid's confirmed rule). A model can only help by knowing, at 19:00
ET, which SIDE (long/short) will win or which DAYS to skip.

Method (ml-pipeline skill, anti-leakage checklist): features use only data
available at 19:00 ET; walk-forward by calendar year (train on all earlier
years, 5-day embargo, test on the next year); no shuffling. Models: logistic
regression and gradient boosting. The best rule is then put through the full
evaluation replay (lucid_pass_research.replay, confirmed rules, MLL lock).

Usage:  .venv/Scripts/python.exe scripts/lucid_ml_race.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402

warnings.filterwarnings("ignore")
S = lp.S
A, B = S(19), lp.RTH_LAST            # entry: open of the 19:00 ET bar; exit: 16:00 ET
UP, DN = 75.5, 50.0                  # 2 NQ: +$3,000 target / -$2,000 open-equity limit, in points


def race_outcome(g: lp.Grid, side: int) -> np.ndarray:
    """1 = pass, 0 = fail, nan = neither by 16:00 (or no data). Adverse first in a bar."""
    n = len(g.dates)
    out = np.full(n, np.nan)
    for i in range(n):
        if not (g.has[i, A] and g.has[i, B]):
            continue
        e = g.O[i, A]
        for k in range(A, B + 1):
            if side > 0:
                bad, good = g.L[i, k] <= e - DN, g.H[i, k] >= e + UP
            else:
                bad, good = g.H[i, k] >= e + DN, g.L[i, k] <= e - UP
            if bad:
                out[i] = 0.0
                break
            if good:
                out[i] = 1.0
                break
    return out


def features(G: dict) -> pd.DataFrame:
    g = G["NQ"]
    dd = g.daily
    c, o = dd["close"], dd["open"]
    f = pd.DataFrame(index=dd.index)
    prev = lambda s, k=1: s.shift(k)                                  # noqa: E731
    f["rth1"] = prev(c / o - 1)
    f["cc1"] = prev(c.pct_change())
    f["r5"] = prev(c / c.shift(5) - 1)
    f["r20"] = prev(c / c.shift(20) - 1)
    f["sma20"] = prev(c / c.rolling(20, min_periods=15).mean() - 1)
    f["atr"] = dd["atr14"] / dd["prev_close"]                        # atr14 is already as-of yesterday
    f["rng1"] = prev(dd["range"]) / dd["atr14"]
    last_hour = pd.Series(g.C[:, B] / g.O[:, S(15)] - 1, index=dd.index)
    f["last_hour"] = prev(last_hour)
    on = pd.Series(g.O[:, lp.RTH_OPEN] / g.O[:, A] - 1, index=dd.index)
    f["on1"] = prev(on)
    f["gap19"] = pd.Series(g.O[:, A], index=dd.index) / dd["prev_close"] - 1   # 16:00 -> 19:00 move
    rets = c.pct_change()
    f["vol_ratio"] = prev(rets.rolling(5).std() / rets.rolling(20).std())
    f["dow"] = dd.index.dayofweek
    first, last = lp.month_position(g.dates)
    f["tom"] = np.where(first <= 3, first, np.where(last >= -2, last, 0))
    for m in ("ES", "GC", "CL"):
        d2 = G[m].daily.reindex(dd.index)
        f[f"{m}_rth1"] = (d2["close"] / d2["open"] - 1).shift(1)
        f[f"{m}_sma20"] = (d2["close"] / d2["close"].rolling(20, min_periods=15).mean() - 1).shift(1)
    return f


def walk_forward(X: pd.DataFrame, y: pd.Series, model_fn, years) -> pd.Series:
    p = pd.Series(np.nan, index=X.index)
    for Y in years:
        test = X.index.year == Y
        cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=7)          # embargo
        train = (X.index < cut) & y.notna() & X.notna().all(1)
        tmask = test & X.notna().all(1)
        if train.sum() < 200 or tmask.sum() == 0:
            continue
        m = model_fn()
        m.fit(X[train], y[train].astype(int))
        p[tmask] = m.predict_proba(X[tmask])[:, 1]
    return p


def main() -> int:
    G = {m: lp.load_grid(m) for m in ("NQ", "ES", "GC", "CL")}
    g = G["NQ"]
    idx = pd.to_datetime(g.dates)
    yL = pd.Series(race_outcome(g, +1), index=idx)
    yS = pd.Series(race_outcome(g, -1), index=idx)
    X = features(G)
    keep = idx.year >= 2018
    X, yL, yS = X[keep], yL[keep], yS[keep]
    years = range(2020, 2027)
    oos = X.index.year >= 2020
    res_L, res_S = yL[oos], yS[oos]
    print(f"days 2020-26: {oos.sum()} | long race resolved {res_L.notna().mean():.0%}, "
          f"pass among resolved {res_L.mean():.3f} | short {res_S.mean():.3f}")
    either = ((res_L == 1) | (res_S == 1))[res_L.notna() | res_S.notna()]
    print(f"perfect foresight of the winning side (upper bound): {either.mean():.3f}\n")

    models = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)),
        "boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                           l2_regularization=1.0, min_samples_leaf=40),
    }
    rows = []
    best = None
    for name, fn in models.items():
        pL = walk_forward(X, yL, fn, years)
        pS = walk_forward(X, yS, fn, years)
        ok = oos & pL.notna() & pS.notna()
        aucL = roc_auc_score(yL[ok & yL.notna()], pL[ok & yL.notna()])
        aucS = roc_auc_score(yS[ok & yS.notna()], pS[ok & yS.notna()])
        for rule in ("side", "long_only"):
            for t in (0.0, 0.40, 0.45, 0.50, 0.55):
                if rule == "side":
                    side = np.where(pL >= pS, 1, -1)
                    pmax = np.maximum(pL, pS)
                else:
                    side = np.ones(len(pL), int)
                    pmax = pL.to_numpy()
                trade = ok & (pmax >= t)
                outcome = np.where(side > 0, yL, yS)
                o = pd.Series(outcome, index=X.index)[trade]
                o = o[o.notna()]
                if len(o) < 30:
                    continue
                rows.append({"model": name, "rule": rule, "min_prob": t, "days_traded": int(trade.sum()),
                             "share_of_days": trade.sum() / ok.sum(), "pass_resolved": o.mean(),
                             "aucL": aucL, "aucS": aucS})
                if best is None or o.mean() > best[0]:
                    best = (o.mean(), name, rule, t, side, pmax, ok)
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    r.to_csv(ROOT / "data" / "futures" / "ml_race_oos.csv", index=False)

    # --- full evaluation replay: baseline vs the best ML rule (2020-26, confirmed rules) ---
    _, name, rule, t, side, pmax, ok = best
    full = np.zeros(len(g.dates), int)
    pos = np.searchsorted(g.dates, X.index.values.astype("datetime64[D]"))
    chosen = np.where(ok & (pmax >= t), side, 0)
    full[pos] = chosen
    n = len(g.dates)
    ok_bars = g.has[:, A] & g.has[:, B]
    stop_long, stop_short = g.O[:, A] - 400.0, g.O[:, A] + 400.0
    out = {}
    for label, sides in (("always long (current bot)", np.where(ok_bars, 1, 0)),
                         (f"ML best: {name}/{rule}/p>={t}", np.where(ok_bars, full, 0))):
        stop = np.where(sides > 0, stop_long, stop_short)
        leg = lp.build_leg(g, label, sides, np.full(n, A), np.full(n, B), stop=stop)
        k = pd.to_datetime(g.dates).year >= 2020
        c, w, b = 20 * leg.close[k], 20 * leg.worst[k], 20 * leg.best[k] - 10
        st = np.arange(0, k.sum() - 120)
        rules = lp.Rules(max_days=120, pass_on="best", mll_on="equity", dll_on="balance", lock=100.0,
                         **lp.PLANS["LucidPro 50K"])
        out[label] = lp.summarise(lp.replay(c, w, rules, st, best=b))
    print("\nFull LucidPro 50K evaluation replay, starts 2020-26:")
    for label, s in out.items():
        print(f"  {label:45s} pass {s['pass_resolved']:.3f} | failed evals per pass {s['fails_per_pass']:.2f} "
              f"| median days {s['med_days']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
