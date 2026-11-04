"""VIX, VIX futures and NYSE TICK as race predictors (2026-10-04).

NinjaTrader serves inputs never tested before (fetched 2026-10-04 into
data/futures/*_nt.parquet): the VIX index ($VIX, daily from Oct 2021, 15-min
from Sep 2023), VIX futures (VX, daily from Jun 2023) and NYSE TICK ($TICK --
stocks ticking up minus down, market breadth: daily from 2011, 15-min from Jul
2017). Every value is used only once it was known: yesterday's session at the
19:00 ET race, the 09:30-10:30 opening hour for a 10:30 race.

Scoring as in lucid_signal_search (race 2 NQ, +75.5 / -50 pts, rules by sign
and by past-only thirds), ranked on the first part of each history and judged
on the rest, then full replays for the best.

Usage:  .venv/Scripts/python.exe scripts/lucid_vix_tick_search.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_orderflow_search as of  # noqa: E402
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402

S = lp.S
F = ROOT / "data" / "futures"
ET = "America/New_York"


def load(name: str) -> pd.DataFrame:
    """Intraday bars on New York time. Daily bars: NinjaTrader stamps them 00:00 UTC OF THEIR
    OWN SESSION DATE, so keep that date as is -- converting to New York time would move every
    session back a day and leak the race day's own value into "yesterday" (found 2026-10-04:
    a z of +10 that was pure look-ahead)."""
    df = pd.read_parquet(F / f"{name}_nt.parquet")
    idx = pd.DatetimeIndex(df.index)
    if name.endswith("_1d"):
        df.index = pd.DatetimeIndex(idx.tz_convert("UTC").date)
    else:
        df.index = idx.tz_convert(ET)
    return df


def known_before(daily: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """For each Lucid day d: the last daily value from a session BEFORE d (known at the 19:00 race)."""
    s = daily.copy()
    s.index = pd.DatetimeIndex(s.index.date)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    left = pd.DataFrame({"d": pd.DatetimeIndex(idx).as_unit("ns")})       # one time unit on both sides
    right = pd.DataFrame({"d": pd.DatetimeIndex(s.index).as_unit("ns"), "v": s.to_numpy()})
    m = pd.merge_asof(left, right, on="d", allow_exact_matches=False)
    return pd.Series(m["v"].to_numpy(), index=idx)


def rth_15m_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per ET session date: mean 15-min close over the session, its last hour and its first hour."""
    mins = df.index.hour * 60 + df.index.minute
    rth = df[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]
    rm = rth.index.hour * 60 + rth.index.minute
    day = pd.Series(rth.index.date, index=rth.index)
    return pd.DataFrame({
        "session": rth["close"].groupby(day.to_numpy()).mean(),
        "last_hour": rth["close"][rm >= 15 * 60].groupby(day[rm >= 15 * 60].to_numpy()).mean(),
        "first_hour": rth["close"][rm < 10 * 60 + 30].groupby(day[rm < 10 * 60 + 30].to_numpy()).mean(),
    })


def first_hour_today(stats: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """The 09:30-10:30 value of the SAME date -- known at a 10:30 race (the bars end at 10:30)."""
    s = stats["first_hour"].copy()
    s.index = pd.DatetimeIndex(s.index)
    return s.reindex(idx)


def evaluate(g: lp.Grid, sigs: dict, split_year: int, title: str) -> pd.DataFrame:
    idx = pd.to_datetime(g.dates)
    years = np.asarray(idx.year)
    first = int(years.min())
    A, B = (first, split_year - 1), (split_year, 2026)
    labels = {a: ss.race_labels(g, a) for a in {v[0] for v in sigs.values()} | {S(19)}}
    base = {p: ss.score(pd.Series(1, index=idx), labels[S(19)], years, P)[0] for p, P in (("A", A), ("B", B))}
    rows = []
    for sname, (a, x, kind) in sigs.items():
        for rname, side in ss.rules_for(sname, x.reindex(idx), kind).items():
            pa, na, _ = ss.score(side, labels[a], years, A)
            pb, nb, _ = ss.score(side, labels[a], years, B)
            za = (pa - base["A"]) / math.sqrt(base["A"] * (1 - base["A"]) / na) if na else np.nan
            zb = (pb - base["B"]) / math.sqrt(base["B"] * (1 - base["B"]) / nb) if nb else np.nan
            rows.append({"signal": sname, "rule": rname, "A win": pa, "A n": na, "A z": za,
                         "B win": pb, "B n": nb, "B z": zb, "_side": side, "_a": a})
    res = pd.DataFrame(rows)
    res["A p"] = [ss.norm_sf(z) if np.isfinite(z) else 1.0 for z in res["A z"]]
    ok = res["A n"] >= 60
    res["BH 10%"] = False
    if ok.any():
        res.loc[ok, "BH 10%"] = ss.bh_fdr(res.loc[ok, "A p"].to_numpy(), 0.10)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_colwidth", 44)
    cols = ["signal", "rule", "A win", "A n", "A z", "B win", "B n", "B z", "BH 10%"]
    f = {c: (lambda v: f"{v:.3f}") for c in ("A win", "B win")}
    f.update({c: (lambda v: f"{v:+.2f}") for c in ("A z", "B z")})
    print(f"\n=== {title}: rank {A[0]}-{A[1]}, judge {B[0]}-2026; deployed race wins "
          f"{base['A']:.3f} / {base['B']:.3f}; {len(res)} rules, {int(res['BH 10%'].sum())} pass the "
          f"false-discovery check")
    print(res[ok].sort_values("A z", ascending=False).head(12)[cols].to_string(index=False, formatters=f))
    held = ok & (res["A z"] >= 2) & (res["B z"] >= 1.65)
    print(f"A z >= 2: {int((ok & (res['A z'] >= 2)).sum())}; also B z >= 1.65: {int(held.sum())}")
    return res


def main() -> int:
    tick_d, tick_15 = load("TICK_1d"), load("TICK_15m")
    vix_d, vx_d = load("VIX_1d"), load("VX_1d")
    tstats = rth_15m_stats(tick_15)
    # --- TICK on the 2016-2026 proxy history (TICK 15-min from mid-2017) ---
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    a19, a1030 = S(19), S(10, 30)
    sess = tstats["session"].copy()
    sess.index = pd.DatetimeIndex(sess.index)
    lasth = tstats["last_hour"].copy()
    lasth.index = pd.DatetimeIndex(lasth.index)
    sig_t = {
        "TICK yesterday: session average": (a19, known_before(sess, idx), "cont"),
        "TICK yesterday: last-hour average": (a19, known_before(lasth, idx), "cont"),
        "TICK yesterday: close": (a19, known_before(tick_d["close"], idx), "cont"),
        "TICK yesterday: high + low (net extreme)": (a19, known_before(tick_d["high"] + tick_d["low"], idx), "cont"),
        "TICK 3-day session average": (a19, known_before(sess.rolling(3).mean(), idx), "cont"),
        "TICK opening hour (enter 10:30)": (a1030, first_hour_today(tstats, idx), "cont"),
    }
    rt = evaluate(g, sig_t, 2023, "NYSE TICK, proxy NQ 2017-2026")
    # --- VIX / VX on the real NinjaTrader bars (2021-2026) ---
    gr = lp.load_grid("NQ", path=of.NT["NQ"])
    idr = pd.to_datetime(gr.dates)
    vc = vix_d["close"]
    vix_rank = vc.rolling(252, min_periods=120).apply(lambda w: (w[:-1] < w[-1]).mean(), raw=True)
    term = (vx_d["close"] / vc.reindex(vx_d.index, method="ffill") - 1)          # >0 calm (contango)
    sig_v = {
        "VIX level vs past year": (a19, known_before(vix_rank, idr), "cont"),
        "VIX change yesterday": (a19, known_before(vc.pct_change(), idr), "cont"),
        "VIX vs its 10-day average": (a19, known_before(vc / vc.rolling(10).mean() - 1, idr), "cont"),
        "VIX futures over VIX (calm > 0)": (a19, known_before(term, idr), "cont"),
        "TICK yesterday: session average": (a19, known_before(sess, idr), "cont"),
        "TICK opening hour (enter 10:30)": (a1030, first_hour_today(tstats, idr), "cont"),
    }
    rv = evaluate(gr, sig_v, 2024, "VIX / VIX futures / TICK, real NinjaTrader NQ 2021-2026")
    # --- full replays for the best-ranked rules ---
    print("\nFull replays of the 3 best-ranked rules of each part vs the deployed race:")
    rep = [{"rule": "DEPLOYED (proxy 2019-22 / 23-26)", **ss.replays(g, a19, np.ones(len(idx), int))}]
    for _, r in rt[rt["A n"] >= 60].sort_values("A z", ascending=False).head(3).iterrows():
        rep.append({"rule": f"{r['signal']} | {r['rule']}"[:70],
                    **ss.replays(g, int(r["_a"]), r["_side"].reindex(idx).fillna(0).to_numpy().astype(int))})
    print(pd.DataFrame(rep).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    rep = [{"rule": "DEPLOYED (real 2021-23 / 24-26)", **of.replays(gr, a19, np.ones(len(idr), int))}]
    for _, r in rv[rv["A n"] >= 60].sort_values("A z", ascending=False).head(3).iterrows():
        rep.append({"rule": f"{r['signal']} | {r['rule']}"[:70],
                    **of.replays(gr, int(r["_a"]), r["_side"].reindex(idr).fillna(0).to_numpy().astype(int))})
    print(pd.DataFrame(rep).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    pd.concat([rt.assign(part="TICK proxy"), rv.assign(part="VIX real")]).drop(columns=["_side", "_a"]).to_csv(
        F / "vix_tick_search.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
