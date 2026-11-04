"""Order flow on REAL futures volume: does buying/selling pressure predict the race? (2026-10-03)

Every earlier predictor (lucid_ml_race, lucid_signal_search, TimesFM) used CFD
proxy bars whose "volume" is not exchange volume. NinjaTrader serves real CME
15-minute bars with volume back to April 2021 (data/futures/<root>_15m_long_nt.parquet,
fetched 2026-10-03): NQ, ES, RTY, GC and ZN (10-year Treasuries -- rates were
never tested before). Signals, all known at the entry:

  V  volume / pressure (bar-level estimate: volume x where the bar closed in
     its range): yesterday's session and last hour, activity vs 20-day normal,
     price at 19:00 vs yesterday's VWAP, the 18:00-19:00 hour;
  C  the same pressure for ES / RTY / GC / ZN, and Treasury moves;
  D  later entry (10:00 / 10:30 ET) on the opening hour's real pressure.

Same rules and scoring as lucid_signal_search: race 2 NQ +75.5 / -50 pts,
rank on 2021-2023, judge on 2024-2026, Benjamini-Hochberg across all rules,
then full evaluation + funded replays on the real bars.

Usage:  .venv/Scripts/python.exe scripts/lucid_orderflow_search.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402

S, K, B = lp.S, lp.K, lp.RTH_LAST
A_YEARS, B_YEARS = (2021, 2023), (2024, 2026)
NT = {m: ROOT / "data" / "futures" / f"{m}_15m_long_nt.parquet" for m in ("NQ", "ES", "RTY", "GC", "ZN")}
ENTRY = {"19:00": S(19), "10:00": S(10), "10:30": S(10, 30)}
for m in ("RTY", "ZN"):                         # signal-only roots: any spec will do for the grid
    lp.MARKETS.setdefault(m, ("nas100", "MNQ", False))


def volume_grid(path: Path, dates: np.ndarray) -> np.ndarray:
    """(n, K) exchange volume per Lucid slot, the same slotting as lucid_pass_research.load_grid."""
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index).tz_convert(lp.ET if hasattr(lp, "ET") else "America/New_York")
    mins = np.asarray(idx.hour * 60 + idx.minute)
    keep = (mins < 16 * 60 + 45) | (mins >= 18 * 60)
    idx, mins, vol = idx[keep], mins[keep], df["volume"].to_numpy(float)[keep]
    base = np.asarray(idx.tz_localize(None).normalize().values.astype("datetime64[D]"))
    lucid = base + (mins >= 1080).astype("timedelta64[D]")
    slot = np.where(mins >= 1080, (mins - 1080) // 15, (mins + 360) // 15)
    pos = {d: i for i, d in enumerate(dates.astype("datetime64[D]"))}
    out = np.zeros((len(dates), K))
    for d, s_, v in zip(lucid, slot, vol):
        i = pos.get(d)
        if i is not None and 0 <= s_ < K:
            out[i, s_] += v
    return out


def pressure(g: lp.Grid, V: np.ndarray, a: int, b: int) -> np.ndarray:
    """Estimated net buying in slots [a, b): volume x close location in the bar's range,
    as a share of that volume (-1 all selling .. +1 all buying)."""
    H, L, C = g.H[:, a:b], g.L[:, a:b], g.C[:, a:b]
    rng = np.where(H - L > 0, H - L, np.nan)
    clv = np.nan_to_num(((C - L) - (H - C)) / rng)
    v = V[:, a:b]
    tot = v.sum(1)
    return np.where(tot > 0, (clv * v).sum(1) / np.where(tot > 0, tot, 1), np.nan)


def vwap(g: lp.Grid, V: np.ndarray, a: int, b: int) -> np.ndarray:
    typ = (g.H[:, a:b] + g.L[:, a:b] + g.C[:, a:b]) / 3
    v = V[:, a:b]
    tot = v.sum(1)
    return np.where(tot > 0, (typ * v).sum(1) / np.where(tot > 0, tot, 1), np.nan)


def signals(G: dict, VV: dict, idx: pd.DatetimeIndex) -> dict:
    g, V = G["NQ"], VV["NQ"]
    a19, op, cl = S(19), lp.RTH_OPEN, lp.RTH_LAST + 1
    prev = lambda x: pd.Series(x, index=idx).shift(1)              # yesterday's session, known tonight
    sig = {}
    rth_vol = pd.Series(V[:, op:cl].sum(1), index=idx).replace(0, np.nan)
    sig["V pressure, yesterday's session"] = (a19, prev(pressure(g, V, op, cl)), "cont")
    sig["V pressure, yesterday's last hour"] = (a19, prev(pressure(g, V, S(15), cl)), "cont")
    sig["V volume vs 20-day normal"] = (a19, (rth_vol / rth_vol.rolling(20, min_periods=10).mean()).shift(1), "cont")
    sig["V volume 5d / 20d"] = (a19, (rth_vol.rolling(5).mean() / rth_vol.rolling(20).mean()).shift(1), "cont")
    sig["V last-hour share of volume"] = (a19, prev(V[:, S(15):cl].sum(1) / np.maximum(V[:, op:cl].sum(1), 1)),
                                          "cont")
    vw = prev(vwap(g, V, op, cl))
    atr = g.daily["atr14"].reindex(idx)
    sig["V price at 19:00 vs yesterday's VWAP"] = (a19, (pd.Series(g.O[:, a19], index=idx) - vw) / atr, "cont")
    sig["V 18:00-19:00 pressure"] = (a19, pd.Series(pressure(g, V, 0, a19), index=idx), "cont")
    eve = pd.Series(V[:, 0:a19].sum(1), index=idx).replace(0, np.nan)
    sig["V 18:00-19:00 volume vs normal"] = (a19, eve / eve.rolling(20, min_periods=10).mean().shift(1), "cont")
    sig["V 3-day pressure"] = (a19, pd.Series(pressure(g, V, op, cl), index=idx).rolling(3).mean().shift(1), "cont")
    for m in ("ES", "RTY", "GC", "ZN"):
        if m in G:
            sig[f"C {m} pressure, yesterday"] = (a19, prev(pressure(G[m], VV[m], op, cl)), "cont")
            d = G[m].daily.reindex(idx)
            sig[f"C {m} yesterday's session move"] = (a19, d["prev_rth_ret"], "cont")
            sig[f"C {m} 16:00->19:00 move"] = (a19, pd.Series(G[m].O[:, a19], index=idx) / d["prev_close"] - 1, "cont")
    if "ZN" in G:
        rz = G["ZN"].daily.reindex(idx)["close"].ffill().pct_change()
        rn = g.daily.reindex(idx)["close"].ffill().pct_change()
        sig["C NQ-Treasury 20-day correlation"] = (a19, rn.rolling(20).corr(rz).shift(1), "cont")
    for ename in ("10:00", "10:30"):
        a = ENTRY[ename]
        sig[f"D opening pressure 09:30->{ename} (enter {ename})"] = (a, pd.Series(pressure(g, V, op, a), index=idx),
                                                                      "cont")
        sig[f"D opening move 09:30->{ename} (enter {ename})"] = (a, pd.Series(g.O[:, a] / g.O[:, op] - 1, index=idx),
                                                                  "cont")
        ov = pd.Series(V[:, op:a].sum(1), index=idx).replace(0, np.nan)
        sig[f"D opening volume vs normal (enter {ename})"] = (a, ov / ov.rolling(20, min_periods=10).mean().shift(1),
                                                              "cont")
        sig[f"D baseline (enter {ename})"] = (a, pd.Series(1.0, index=idx), "bin")
    return sig


def replays(g: lp.Grid, a: int, side: np.ndarray) -> dict:
    n = len(g.dates)
    ok = g.has[:, a] & g.has[:, B]
    sd = np.where(ok, side, 0).astype(int)
    e = g.O[:, a]
    leg = lp.build_leg(g, "rule", sd, np.full(n, a), np.full(n, B), stop=np.where(sd >= 0, e - 400.0, e + 400.0))
    yrs = np.asarray(pd.to_datetime(g.dates).year)
    starts = np.arange(0, n - 120)
    sy = yrs[starts]
    ev = lp.replay(20 * leg.close, 20 * leg.worst,
                   lp.Rules(max_days=120, pass_on="best", mll_on="equity", dll_on="balance", lock=100.0,
                            **lp.PLANS["LucidPro 50K"]), starts, best=20 * leg.best - 10)
    fr = lp.replay_funded(10 * leg.close, 10 * leg.worst, 10 * leg.best - 5,
                          lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0), max_days=120),
                          starts, day_tp=1040.0, adaptive=True, pause_when_met=True)
    out = {}
    for lab_, (y0, y1) in (("21-23", A_YEARS), ("24-26", B_YEARS)):
        m = (sy >= y0) & (sy <= y1)
        got = fr.payouts[m] >= 1
        out[f"eval pass {lab_}"] = float((ev.status[m] == 1).mean())
        out[f"funded paid<=10d {lab_}"] = float((got & (fr.first_day[m] <= 10)).mean())
        out[f"funded $ to you {lab_}"] = float(0.9 * fr.paid[m].mean())
    return out


def main() -> int:
    t0 = time.time()
    G = {m: lp.load_grid(m, path=p) for m, p in NT.items() if p.exists()}
    g = G["NQ"]
    idx = pd.to_datetime(g.dates)
    VV = {m: volume_grid(NT[m], G[m].dates) for m in G}
    # other markets' grids have their own dates: put them on NQ's
    for m in list(G):
        if m == "NQ":
            continue
        pos = pd.Series(np.arange(len(G[m].dates)), index=pd.to_datetime(G[m].dates)).reindex(idx)
        take = pos.to_numpy()
        ok = ~np.isnan(take)
        ti = np.where(ok, take, 0).astype(int)
        for attr in ("O", "H", "L", "C"):
            arr = getattr(G[m], attr)[ti].copy()
            arr[~ok] = np.nan
            setattr(G[m], attr, arr)
        v = VV[m][ti].copy()
        v[~ok] = 0
        VV[m] = v
    years = np.asarray(idx.year)
    print(f"real NinjaTrader bars: {', '.join(G)}; NQ days {len(idx)} ({idx[0].date()} .. {idx[-1].date()}); "
          f"NQ session volume median {np.median(VV['NQ'][:, lp.RTH_OPEN:B + 1].sum(1)):,.0f} contracts")
    labels = {e: ss.race_labels(g, a) for e, a in ENTRY.items()}
    base = {p: ss.score(pd.Series(1, index=idx), labels["19:00"], years, P)[0]
            for p, P in (("A", A_YEARS), ("B", B_YEARS))}
    print(f"deployed race (19:00 long), win rate among resolved days: 2021-23 {base['A']:.3f} | 2024-26 {base['B']:.3f}")
    rows = []
    for sname, (a, x, kind) in signals(G, VV, idx).items():
        ename = next(k for k, v in ENTRY.items() if v == a)
        for rname, side in ss.rules_for(sname, x.reindex(idx), kind).items():
            pa, na, cova = ss.score(side, labels[ename], years, A_YEARS)
            pb, nb, covb = ss.score(side, labels[ename], years, B_YEARS)
            za = (pa - base["A"]) / math.sqrt(base["A"] * (1 - base["A"]) / na) if na else np.nan
            zb = (pb - base["B"]) / math.sqrt(base["B"] * (1 - base["B"]) / nb) if nb else np.nan
            rows.append({"family": sname[0], "signal": sname[2:], "rule": rname, "entry": ename,
                         "A win": pa, "A n": na, "A z": za, "B win": pb, "B n": nb, "B lift": pb - base["B"],
                         "B z": zb, "_side": side, "_a": a})
    res = pd.DataFrame(rows)
    res["A p"] = [ss.norm_sf(z) if np.isfinite(z) else 1.0 for z in res["A z"]]
    usable = (res["A n"] >= 100)
    res["BH 10%"] = False
    res.loc[usable, "BH 10%"] = ss.bh_fdr(res.loc[usable, "A p"].to_numpy(), 0.10)
    res.drop(columns=["_side", "_a"]).to_csv(ROOT / "data" / "futures" / "orderflow_search.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 46)
    cols = ["family", "signal", "rule", "entry", "A win", "A n", "A z", "B win", "B n", "B lift", "B z", "BH 10%"]
    fmt = {c: (lambda v: f"{v:.3f}") for c in ("A win", "B win", "B lift")}
    fmt.update({c: (lambda v: f"{v:+.2f}") for c in ("A z", "B z")})
    print(f"\n{len(res)} rules, {int(usable.sum())} with >= 100 resolved days on 2021-23; "
          f"{int(res['BH 10%'].sum())} pass the false-discovery check")
    print("\nTop 20 RANKED ON 2021-23, judged on 2024-26:")
    print(res[usable].sort_values("A z", ascending=False).head(20)[cols].to_string(index=False, formatters=fmt))
    sel = usable & (res["A z"] >= 2.0)
    print(f"\nA z >= 2: {int(sel.sum())} rules; of these, B z >= 1.65: {int((sel & (res['B z'] >= 1.65)).sum())}")
    print("\nBest on BOTH halves (min of A z, B z) -- for information, this selection uses B:")
    both = res[usable].assign(minz=np.minimum(res["A z"], res["B z"])).sort_values("minz", ascending=False).head(8)
    print(both[cols].to_string(index=False, formatters=fmt))
    fin = res[usable].sort_values("A z", ascending=False).head(6)
    rep = [{"rule": "DEPLOYED race 19:00", **replays(g, S(19), np.ones(len(idx), int))}]
    for _, r in fin.iterrows():
        rep.append({"rule": f"{r['signal']} | {r['rule']} | {r['entry']}"[:72],
                    **replays(g, int(r["_a"]), r["_side"].reindex(idx).fillna(0).to_numpy().astype(int))})
    print("\nFull replays on the real bars (top 6 by 2021-23 rank):")
    print(pd.DataFrame(rep).to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
