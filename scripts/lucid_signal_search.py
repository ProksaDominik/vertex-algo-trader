"""Can anything we can see at entry time beat the race's coin flip? (2026-10-03)

Owner, 2026-10-03: "try everything" with the data we have, plus the skills
added today (market-regime-classifier, correlation-regime-switcher,
hypothesis-tester / statistical-power; TimesFM is tested separately in
lucid_timesfm_race.py). The earlier ML test (lucid_ml_race.py) covered NQ's own
returns/volatility, day of week, turn of month and ES/GC/CL; this covers what
it did not:

  X  cross-market: YM, copper, Nikkei, DAX, Brent, platinum, Bitcoin, Ether
     and 12 currency pairs -- yesterday's US session and their 16:00 -> 19:00
     move, at the 19:00 ET race entry;
  O  overnight: enter later (02:00 / 04:00 / 08:00 ET) and use what happened
     since 19:00 (NQ, Nikkei, DAX, yen, Bitcoin) to pick the side or skip;
  R  regimes (market-regime-classifier / correlation-regime-switcher): trend
     strength (ADX), Bollinger bandwidth, ATR level, 20-day NQ correlation with
     gold / yen / bonds-proxy, each as a past-only percentile;
  E  events: Fed decision, CPI and jobs-report days (official dates,
     data/futures/event_dates.json).

Each signal becomes rules -- side by sign (momentum / reversal), and "race only
when the signal is in its low / middle / high third" (thirds from the past 252
days only) -- scored on the race's own outcome: 2 NQ, +75.5 pts (the $3,000
target) before -50 pts (the $2,000 limit), adverse first inside a 15-min bar,
out 16:00 ET. Ranked on 2016-2022, judged on 2023-2026, with a
Benjamini-Hochberg false-discovery check across all rules. The survivors go
through the full evaluation and funded replays.

Usage:  .venv/Scripts/python.exe scripts/lucid_signal_search.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402

S, K, B = lp.S, lp.K, lp.RTH_LAST
UP, DN = 75.5, 50.0
A_YEARS, B_YEARS = (2016, 2022), (2023, 2026)
ENTRIES = {"19:00": S(19), "02:00": S(2), "04:00": S(4), "08:00": S(8)}

# signal-only markets (no trading): proxy file, any known contract for the spec
EXTRA = {"DAX": "de40", "BRENT": "brent", "PLAT": "xptusd", "BTC": "btcusd", "ETH": "ethusd",
         "EURJPY": "eurjpy", "AUDJPY": "audjpy", "GBPJPY": "gbpjpy", "NZDUSD": "nzdusd",
         "EURCHF": "eurchf", "EURGBP": "eurgbp", "AUDCAD": "audcad"}
for k, proxy in EXTRA.items():
    lp.MARKETS.setdefault(k, (proxy, "MNQ", False))
CROSS = ["ES", "YM", "GC", "CL", "HG", "NKD", "6E", "6B", "6A", "6J", "6C", "6S"] + list(EXTRA)


# --- outcomes ------------------------------------------------------------------

def race_labels(g: lp.Grid, a: int) -> dict[int, np.ndarray]:
    """1 win / 0 fail / nan unresolved or no data, for a long (+1) and short (-1) race from slot a."""
    kk = np.arange(K)[None, :]
    live = (kk >= a) & (kk <= B)
    e = g.O[:, a][:, None]
    ok = g.has[:, a] & g.has[:, B]
    out = {}
    for side in (1, -1):
        adv, fav = (g.L, g.H) if side > 0 else (g.H, g.L)
        bad = live & (side * (adv - e) <= -DN)
        good = live & (side * (fav - e) >= UP)
        fb = np.where(bad.any(1), bad.argmax(1), K)
        fg = np.where(good.any(1), good.argmax(1), K)
        y = np.where((fb == K) & (fg == K), np.nan, np.where(fb <= fg, 0.0, 1.0))
        y[~ok] = np.nan
        out[side] = y
    return out


# --- signals -------------------------------------------------------------------

def slot_series(G, m, slot, idx):
    """Market m's price at the open of `slot` (forward-filled inside the day), on NQ's dates."""
    g = G[m]
    return pd.Series(g.O[:, slot], index=pd.to_datetime(g.dates)).reindex(idx)


def daily_of(G, m, idx):
    return G[m].daily.reindex(idx)


def adx(high, low, close, n=14):
    up, dn = high.diff(), -low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=close.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def build_signals(G: dict, idx: pd.DatetimeIndex) -> dict[str, tuple[int, pd.Series, str]]:
    """name -> (entry slot, value known at that entry, 'cont' | 'bin')."""
    sig: dict[str, tuple[int, pd.Series, str]] = {}
    a19 = S(19)
    nq = daily_of(G, "NQ", idx)
    # X: cross-market, known at 19:00
    for m in CROSS:
        if m not in G:
            continue
        d = daily_of(G, m, idx)
        sig[f"X {m} prev US session"] = (a19, d["prev_rth_ret"], "cont")
        sig[f"X {m} 16:00->19:00"] = (a19, slot_series(G, m, a19, idx) / d["prev_close"] - 1, "cont")
    sig["X NQ minus ES prev session"] = (a19, nq["prev_rth_ret"] - daily_of(G, "ES", idx)["prev_rth_ret"], "cont")
    # O: overnight, decided at a later entry
    for ename in ("02:00", "04:00", "08:00"):
        a = ENTRIES[ename]
        for m in ("NQ", "NKD", "DAX", "6J", "BTC", "GC"):
            if m in G:
                move = slot_series(G, m, a, idx) / slot_series(G, m, a19, idx) - 1
                sig[f"O {m} 19:00->{ename} (enter {ename})"] = (a, move, "cont")
        sig[f"O baseline (enter {ename})"] = (a, pd.Series(1.0, index=idx), "bin")
    # R: regimes (past-only; daily frame values are as of yesterday's close)
    c = nq["close"].ffill()
    h, low = nq["high"].ffill(), nq["low"].ffill()
    sig["R ADX trend strength"] = (a19, adx(h, low, c).shift(1), "cont")
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    sig["R Bollinger bandwidth"] = (a19, (4 * sd / mid).shift(1), "cont")
    sig["R ATR % of price"] = (a19, nq["atr14"] / nq["prev_close"], "cont")
    sig["R 5d/20d volatility"] = (a19, (c.pct_change().rolling(5).std() / c.pct_change().rolling(20).std()).shift(1),
                                   "cont")
    r = c.pct_change()
    for m in ("GC", "6J", "ES", "CL"):
        if m in G:
            rm = daily_of(G, m, idx)["close"].ffill().pct_change()
            sig[f"R corr NQ-{m} 20d"] = (a19, r.rolling(20).corr(rm).shift(1), "cont")
    sig["R distance from 50d high"] = (a19, (c / c.rolling(50).max() - 1).shift(1), "cont")
    # E: events (a release at 08:30 / 14:00 ET on date D falls inside Lucid day D's race)
    ev = json.loads((ROOT / "data" / "futures" / "event_dates.json").read_text(encoding="utf-8"))
    days = pd.Series(idx, index=idx)
    for name in ("fomc", "cpi", "nfp"):
        dts = pd.to_datetime(ev[name])
        dts = pd.DatetimeIndex([d + pd.offsets.BDay(0) for d in dts])   # weekend -> next weekday
        on = days.isin(dts).astype(float)
        sig[f"E {name.upper()} day"] = (a19, on, "bin")
        sig[f"E day before {name.upper()}"] = (a19, on.shift(-1).fillna(0.0), "bin")   # dates are scheduled ahead
        sig[f"E day after {name.upper()}"] = (a19, on.shift(1).fillna(0.0), "bin")
    anyev = sum(days.isin(pd.DatetimeIndex([d + pd.offsets.BDay(0) for d in pd.to_datetime(ev[n])])).astype(float)
                for n in ("fomc", "cpi", "nfp")).clip(upper=1)
    sig["E any big release day"] = (a19, anyev, "bin")
    return sig


# --- rules and scoring ---------------------------------------------------------

def past_terciles(x: pd.Series) -> pd.Series:
    """0 / 1 / 2 = low / middle / high third, thresholds from the previous 252 days only."""
    q1 = x.rolling(252, min_periods=120).quantile(1 / 3).shift(1)
    q2 = x.rolling(252, min_periods=120).quantile(2 / 3).shift(1)
    t = pd.Series(np.where(x <= q1, 0, np.where(x <= q2, 1, 2)), index=x.index, dtype=float)
    t[x.isna() | q1.isna()] = np.nan
    return t


def rules_for(name: str, x: pd.Series, kind: str) -> dict[str, pd.Series]:
    """rule -> side per day (+1 long race, -1 short race, 0 skip)."""
    out = {}
    if kind == "bin":
        on = x.fillna(0) > 0
        out["long on these days only"] = pd.Series(np.where(on, 1, 0), index=x.index)
        if "baseline" not in name:
            out["long, skip these days"] = pd.Series(np.where(on, 0, 1), index=x.index)
            out["short on these days only"] = pd.Series(np.where(on, -1, 0), index=x.index)
        return out
    sgn = np.sign(x.fillna(0.0))
    out["side = sign (momentum)"] = pd.Series(sgn, index=x.index)
    out["side = -sign (reversal)"] = pd.Series(-sgn, index=x.index)
    t = past_terciles(x)
    for k, lab in ((0, "low"), (1, "middle"), (2, "high")):
        out[f"long only in {lab} third"] = pd.Series(np.where(t == k, 1, 0), index=x.index)
        out[f"short only in {lab} third"] = pd.Series(np.where(t == k, -1, 0), index=x.index)
    return out


def score(side: pd.Series, lab: dict[int, np.ndarray], years: np.ndarray, period) -> tuple[float, int, float]:
    """(win rate among resolved raced days, resolved raced days, share of days raced)."""
    sel = (years >= period[0]) & (years <= period[1])
    s = side.to_numpy()
    y = np.where(s > 0, lab[1], np.where(s < 0, lab[-1], np.nan))
    raced = sel & (s != 0)
    res = raced & ~np.isnan(y)
    n = int(res.sum())
    return (float(np.nanmean(y[res])) if n else np.nan), n, float(raced.sum() / max(sel.sum(), 1))


def bh_fdr(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: which hypotheses survive at false-discovery rate q."""
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


# --- full replays for the finalists ----------------------------------------------

def replays(g: lp.Grid, a: int, side: np.ndarray) -> dict:
    n = len(g.dates)
    ok = g.has[:, a] & g.has[:, B]
    sd = np.where(ok, side, 0).astype(int)
    e = g.O[:, a]
    leg = lp.build_leg(g, "rule", sd, np.full(n, a), np.full(n, B), stop=np.where(sd >= 0, e - 400.0, e + 400.0))
    yrs = pd.to_datetime(g.dates).year
    keep = np.asarray(yrs >= 2019)
    starts = np.arange(0, keep.sum() - 120)
    sy = np.asarray(yrs[keep])[starts]
    c, w, bb = 20 * leg.close[keep], 20 * leg.worst[keep], 20 * leg.best[keep] - 10
    ev = lp.replay(c, w, lp.Rules(max_days=120, pass_on="best", mll_on="equity", dll_on="balance", lock=100.0,
                                  **lp.PLANS["LucidPro 50K"]), starts, best=bb)
    c, w, bb = 10 * leg.close[keep], 10 * leg.worst[keep], 10 * leg.best[keep] - 5
    fr = lp.replay_funded(c, w, bb, lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0), max_days=120),
                          starts, day_tp=1040.0, adaptive=True, pause_when_met=True)
    out = {}
    for lab_, (y0, y1) in (("19-22", (2019, 2022)), ("23-26", (2023, 2026))):
        m = (sy >= y0) & (sy <= y1)
        e_, f_ = ev[m], fr[m]
        got = f_.payouts >= 1
        out[f"eval pass {lab_}"] = float((e_.status == 1).mean())
        out[f"eval days {lab_}"] = float(e_.days[e_.status == 1].mean()) if (e_.status == 1).any() else np.nan
        out[f"funded paid<=10d {lab_}"] = float((got & (f_.first_day <= 10)).mean())
        out[f"funded $ to you {lab_}"] = float(0.9 * f_.paid.mean())
    return out


def main() -> int:
    t0 = time.time()
    names = ["NQ"] + CROSS
    G = {}
    for m in names:
        try:
            G[m] = lp.load_grid(m)
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: not loaded ({e})")
    g = G["NQ"]
    idx = pd.to_datetime(g.dates)
    years = np.asarray(idx.year)
    print(f"loaded {len(G)} markets in {time.time() - t0:.0f}s; NQ days {len(idx)} ({idx[0].date()} .. {idx[-1].date()})")
    labels = {e: race_labels(g, a) for e, a in ENTRIES.items()}
    base = {}
    for e in ENTRIES:
        for per, P in (("A", A_YEARS), ("B", B_YEARS)):
            base[(e, per)] = score(pd.Series(1, index=idx), labels[e], years, P)[0]
    print("baseline long race, win rate among resolved days: " +
          ", ".join(f"{e}: A {base[(e, 'A')]:.3f} / B {base[(e, 'B')]:.3f}" for e in ENTRIES))
    sigs = build_signals(G, idx)
    rows = []
    for sname, (a, x, kind) in sigs.items():
        ename = next(k for k, v in ENTRIES.items() if v == a)
        lab = labels[ename]
        for rname, side in rules_for(sname, x.reindex(idx), kind).items():
            pa, na, cova = score(side, lab, years, A_YEARS)
            pb, nb, covb = score(side, lab, years, B_YEARS)
            ba, bb_ = base[("19:00", "A")], base[("19:00", "B")]      # beat the DEPLOYED 19:00 race
            za = (pa - ba) / math.sqrt(ba * (1 - ba) / na) if na else np.nan
            zb = (pb - bb_) / math.sqrt(bb_ * (1 - bb_) / nb) if nb else np.nan
            rows.append({"family": sname[0], "signal": sname[2:], "rule": rname, "entry": ename,
                         "A win": pa, "A n": na, "A raced": cova, "A lift": pa - ba, "A z": za,
                         "B win": pb, "B n": nb, "B raced": covb, "B lift": pb - bb_, "B z": zb,
                         "_side": side, "_a": a})
    res = pd.DataFrame(rows)
    res["A p"] = [norm_sf(z) if np.isfinite(z) else 1.0 for z in res["A z"]]
    usable = (res["A n"] >= 150) & (res["A raced"] >= 0.15)
    res["BH 10%"] = False
    res.loc[usable, "BH 10%"] = bh_fdr(res.loc[usable, "A p"].to_numpy(), 0.10)
    out = res.drop(columns=["_side", "_a"])
    out.to_csv(ROOT / "data" / "futures" / "signal_search.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    cols = ["family", "signal", "rule", "entry", "A win", "A n", "A z", "B win", "B n", "B lift", "B z", "BH 10%"]
    fmt = {c: (lambda v: f"{v:.3f}") for c in ("A win", "B win", "B lift")}
    fmt.update({c: (lambda v: f"{v:+.2f}") for c in ("A z", "B z")})
    print(f"\n{len(res)} rules from {len(sigs)} signals ({usable.sum()} with enough days); "
          f"{int(res['BH 10%'].sum())} pass the false-discovery check on 2016-22")
    print("\nTop 25 RANKED ON 2016-22 (A), judged on 2023-26 (B never used to rank):")
    top = res[usable].sort_values("A z", ascending=False).head(25)
    print(top[cols].to_string(index=False, formatters=fmt))
    held = res[usable & (res["A z"] >= 2.0) & (res["B z"] >= 1.65)]
    print(f"\nrules with A z >= 2 that ALSO hold on B (z >= 1.65): {len(held)} of "
          f"{int((usable & (res['A z'] >= 2.0)).sum())}")
    # expected by luck alone: ~5% of the A-selected rules clear z >= 1.65 on B
    fin = res[usable].sort_values("A z", ascending=False).head(8)
    print("\nFull evaluation + funded replays of the 8 best-ranked rules (vs the deployed race):")
    rep = [{"rule": "DEPLOYED: long race 19:00 every day", **replays(g, S(19), np.ones(len(idx), int))}]
    for _, r in fin.iterrows():
        rep.append({"rule": f"{r['signal']} | {r['rule']} | {r['entry']}"[:70],
                    **replays(g, int(r["_a"]), r["_side"].reindex(idx).fillna(0).to_numpy().astype(int))})
    rp = pd.DataFrame(rep)
    print(rp.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 5 else f"{v:,.0f}"))
    rp.to_csv(ROOT / "data" / "futures" / "signal_search_replays.csv", index=False)
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
