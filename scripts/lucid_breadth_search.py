"""Market breadth as race predictors: Nasdaq TICK, TRIN, advancers/decliners, up/down volume (2026-10-04).

NinjaTrader indexes fetched 2026-10-04 (data/futures/*_nt.parquet): $TICKQ (Nasdaq
stocks ticking up minus down), $TRIN / $TRINQ (Arms index: breadth vs volume,
> 1 = selling dominates), $ADV / $DECL (advancing / declining NYSE stocks),
$UVOL / $DVOL (volume in rising / falling NYSE stocks) -- daily from 2011,
15-minute TICKQ / TRIN from mid-2020. Daily bars carry their own session date
(lucid_vix_tick_search.load); every value is used only once it was known.

Same scoring as lucid_vix_tick_search.evaluate (race 2 NQ, rank 2016-2022 on the
proxy history, judge 2023-2026), then full replays for the best.

Usage:  .venv/Scripts/python.exe scripts/lucid_breadth_search.py
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

S = lp.S


def main() -> int:
    d = {n: vt.load(f"{n}_1d") for n in ("TICK", "TICKQ", "TRIN", "TRINQ", "ADV", "DECL", "UVOL", "DVOL")}
    tq = vt.rth_15m_stats(vt.load("TICKQ_15m"))
    tn = vt.rth_15m_stats(vt.load("TRIN_15m"))
    tk = vt.rth_15m_stats(vt.load("TICK_15m"))
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    a19, a1030 = S(19), S(10, 30)
    kb = lambda s: vt.known_before(s, idx)           # noqa: E731 -- yesterday's value at tonight's race

    def ser(df, col):
        s = df[col].copy()
        s.index = pd.DatetimeIndex(s.index)
        return s

    adv_ratio = np.log(d["ADV"]["close"] / d["DECL"]["close"])
    vol_ratio = np.log(d["UVOL"]["close"] / d["DVOL"]["close"])
    tick_sess, tickq_sess = ser(tk, "session"), ser(tq, "session")
    sigs = {
        "Advancers / decliners yesterday": (a19, kb(adv_ratio), "cont"),
        "Up / down volume yesterday": (a19, kb(vol_ratio), "cont"),
        "Advancers / decliners, 3-day": (a19, kb(adv_ratio.rolling(3).mean()), "cont"),
        "Up / down volume, 3-day": (a19, kb(vol_ratio.rolling(3).mean()), "cont"),
        "TRIN yesterday (close)": (a19, kb(np.log(d["TRIN"]["close"].clip(lower=0.05))), "cont"),
        "TRINQ yesterday (close)": (a19, kb(np.log(d["TRINQ"]["close"].clip(lower=0.05))), "cont"),
        "Nasdaq TICK yesterday (close)": (a19, kb(d["TICKQ"]["close"]), "cont"),
        "Nasdaq TICK yesterday: session average": (a19, kb(tickq_sess), "cont"),
        "NYSE + Nasdaq TICK agree, yesterday": (a19, kb(np.sign(tick_sess) + np.sign(tickq_sess.reindex(tick_sess.index))),
                                                 "cont"),
        "TRIN yesterday: session average": (a19, kb(ser(tn, "session")), "cont"),
        "Nasdaq TICK opening hour (enter 10:30)": (a1030, vt.first_hour_today(tq, idx), "cont"),
        "TRIN opening hour (enter 10:30)": (a1030, vt.first_hour_today(tn, idx), "cont"),
        "Breadth extreme: advancers/decliners top or bottom 10% (past year)":
            (a19, kb(adv_ratio.rolling(252, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)), "cont"),
    }
    res = vt.evaluate(g, sigs, 2023, "Breadth on the proxy NQ history 2016-2026")
    print("\nFull replays of the 5 best-ranked rules vs the deployed race:")
    rep = [{"rule": "DEPLOYED race", **ss.replays(g, a19, np.ones(len(idx), int))}]
    for _, r in res[res["A n"] >= 60].sort_values("A z", ascending=False).head(5).iterrows():
        rep.append({"rule": f"{r['signal']} | {r['rule']}"[:72],
                    **ss.replays(g, int(r["_a"]), r["_side"].reindex(idx).fillna(0).to_numpy().astype(int))})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rep).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    res.drop(columns=["_side", "_a"]).to_csv(ROOT / "data" / "futures" / "breadth_search.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
