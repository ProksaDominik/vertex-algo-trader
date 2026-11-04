"""Two testable ideas from the QuantifiedStrategies 'Best futures trading strategies' article (2026-10-08).

A. IBS mean reversion. IBS = (close - low) / (high - low) of the New York session (09:30-16:00).
   A1 classic (holds overnight -- NOT allowed on LucidPro, shown for reference): buy at the 16:00 close when
      IBS < 0.2, sell at the next 16:00 close; short when IBS > 0.8.
   A2 Lucid-compatible: the same signal, in at 19:00 New York, out at 16:00 the next day (the race window).
   A3 as a race filter / side: the race (2 NQ, +75.5 / -50 pts, 19:00 -> 16:00) only after low-IBS days, or
      long after low and short after high IBS -- scored with lucid_signal_search (choose on 2016-2022, judge
      on 2023-2026) and full evaluation / funded replays.
B. NQ / ES pairs (statistical arbitrage): spread = log(NQ) - b log(ES), b from the last 2,000 bars; z = the
   spread's distance from its 100-bar mean in standard deviations. z > +2: sell NQ, buy ES (equal dollars);
   z < -2: the reverse. Out at z = 0, at |z| = 4 (stop), or on the session's last bar before 16:40.
   Entries 09:30-15:30 New York. Costs on both legs (micro-contract commissions + 1 tick spread each way).
   P&L in dollars per $100,000 of each leg.
Data: NQ / ES 15-min stand-ins 2016-2026, NinjaTrader NQ / ES 5-min Dec 2023 - Oct 2026.

Usage:  .venv/Scripts/python.exe scripts/research_ibs_pairs.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402
import lucid_vix_tick_search as vt  # noqa: E402

ET = "America/New_York"


def load15(name: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / f"{name}_15m_long.parquet")[["open", "high", "low", "close"]]
    df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
    return df[~df.index.duplicated()].sort_index()


def rth_days(df: pd.DataFrame) -> pd.DataFrame:
    m = df.index.hour * 60 + df.index.minute
    r = df[(m >= 570) & (m < 960)]
    d = r.groupby(r.index.date).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    d.index = pd.DatetimeIndex(d.index)
    d["ibs"] = (d["close"] - d["low"]) / (d["high"] - d["low"]).replace(0, np.nan)
    return d


def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else np.nan


def part_a(market: str, proxy: str, cost_pts: float) -> None:
    df = load15(proxy)
    d = rth_days(df)
    # A1 classic close-to-close
    nxt = d["close"].shift(-1) - d["close"]
    print(f"\n=== {market}: IBS, 10 years ({len(d)} sessions) ===")
    for lo, hi in ((2016, 2022), (2023, 2026)):
        sel = (d.index.year >= lo) & (d.index.year <= hi)
        base = nxt[sel].dropna().to_numpy()
        lng = nxt[sel & (d["ibs"] < 0.2)].dropna().to_numpy() - cost_pts
        sht = -nxt[sel & (d["ibs"] > 0.8)].dropna().to_numpy() - cost_pts
        print(f"  A1 classic {lo}-{hi}: every day {base.mean():+.1f} pts | buy after IBS<0.2: {lng.mean():+.1f} pts/trade "
              f"(n {len(lng)}, t {tstat(lng):+.2f}) | short after IBS>0.8: {sht.mean():+.1f} pts (n {len(sht)}, t {tstat(sht):+.2f})")
    # A2: 19:00 -> 16:00 next day
    m = df.index.hour * 60 + df.index.minute
    o19 = df[m == 1140]["open"]
    o19.index = pd.DatetimeIndex(o19.index.date)
    c16 = df[m == 945]["close"]                                   # the 15:45 bar closes at 16:00
    c16.index = pd.DatetimeIndex(c16.index.date)
    rows = []
    for day in d.index:
        nxt_day = d.index[d.index > day]
        if not len(nxt_day) or day not in o19.index or nxt_day[0] not in c16.index:
            continue
        rows.append((day, d.loc[day, "ibs"], c16[nxt_day[0]] - o19[day]))
    a2 = pd.DataFrame(rows, columns=["day", "ibs", "move"]).set_index("day")
    for lo, hi in ((2016, 2022), (2023, 2026)):
        x = a2[(a2.index.year >= lo) & (a2.index.year <= hi)]
        base, lng, sht = x["move"].to_numpy(), x[x.ibs < 0.2]["move"].to_numpy() - cost_pts, -x[x.ibs > 0.8]["move"].to_numpy() - cost_pts
        print(f"  A2 Lucid window (19:00 -> 16:00) {lo}-{hi}: every day {base.mean():+.1f} pts | after IBS<0.2 long {lng.mean():+.1f} "
              f"(n {len(lng)}, t {tstat(lng):+.2f}) | after IBS>0.8 short {sht.mean():+.1f} (n {len(sht)}, t {tstat(sht):+.2f})")
    return d


def part_a3(d: pd.DataFrame) -> None:
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    a19 = lp.S(19)
    ibs = vt.known_before(d["ibs"], idx)                         # yesterday's New York session, known at 19:00
    sigs = {"IBS of yesterday's session (0.5 - IBS: + = closed low)": (a19, 0.5 - ibs, "cont")}
    res = vt.evaluate(g, sigs, 2023, "A3: IBS as the race's filter / side (NQ race, 2 NQ)")
    n = len(idx)
    print("\n  full replays (evaluation pass 2019-22 / 2023-26, funded $ to you):")
    rows = [{"rule": "always long (deployed race, trend200 ~ same)", **ss.replays(g, a19, np.ones(n, int))}]
    x = ibs.to_numpy()
    for name, side in (("race only after IBS < 0.5 (closed low)", np.where(x < 0.5, 1, 0)),
                       ("race only after IBS < 0.2", np.where(x < 0.2, 1, 0)),
                       ("long after IBS < 0.5, short after IBS > 0.5", np.where(np.isnan(x), 1, np.where(x < 0.5, 1, -1)))):
        rows.append({"rule": name, **ss.replays(g, a19, side)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 5 else f"{v:,.0f}"))


def pairs(nq: pd.DataFrame, es: pd.DataFrame, label: str, comm_usd: float, ticks_usd: float) -> None:
    j = nq[["close"]].join(es[["close"]], lsuffix="_nq", rsuffix="_es", how="inner").dropna()
    ln, le = np.log(j["close_nq"].to_numpy()), np.log(j["close_es"].to_numpy())
    m = np.asarray(j.index.hour * 60 + j.index.minute)
    day = np.asarray((j.index + pd.Timedelta(hours=6)).date)
    n = len(j)
    rl, re_ = pd.Series(ln).diff(), pd.Series(le).diff()
    beta = (rl.rolling(2000, min_periods=500).cov(re_) / re_.rolling(2000, min_periods=500).var()).shift(1).to_numpy()
    out = []
    pos = None
    for i in range(2100, n):
        b = beta[i]
        if np.isnan(b):
            continue
        spread = ln[i - 100:i + 1] - b * le[i - 100:i + 1]
        mu, sd = spread[:-1].mean(), spread[:-1].std()
        if sd <= 0:
            continue
        z = (spread[-1] - mu) / sd
        last = i + 1 >= n or day[i + 1] != day[i] or m[i + 1] >= 1000
        if pos is not None:
            s, i0 = pos
            if (s < 0 and z <= 0) or (s > 0 and z >= 0) or abs(z) >= 4 or last:
                # s = +1: long NQ / short ES. $ on $100k each leg
                pnl = s * 100_000 * ((ln[i] - ln[i0]) - (le[i] - le[i0])) - 2 * (comm_usd + ticks_usd)
                out.append((j.index[i0], pnl, "stop" if abs(z) >= 4 else ("16:40" if last else "back to normal")))
                pos = None
        if pos is None and 570 <= m[i] < 930 and not last:
            if z > 2:
                pos = (-1, i)
            elif z < -2:
                pos = (1, i)
    t = pd.DataFrame(out, columns=["opened", "usd", "how"])
    yrs = (t.opened.max() - t.opened.min()).days / 365.25
    print(f"\n=== B pairs NQ / ES, {label}: {len(t)} trades ({len(t) / yrs:.0f} a year) ===")
    split = t.opened.min() + (t.opened.max() - t.opened.min()) / 2 if "15-min" not in label else pd.Timestamp("2023-01-01", tz=ET)
    for part, x in (("first half", t[t.opened < split]), ("second half", t[t.opened >= split])):
        u = x["usd"].to_numpy()
        print(f"  {part}: average ${u.mean():+.1f} per trade per $100k a leg (t {tstat(u):+.2f}), winning {np.mean(u > 0):.0%}, "
              f"exits {x['how'].value_counts().to_dict()}")


def main() -> int:
    d_nq = part_a("NQ", "nas100", 0.175 + 0.25)
    part_a("ES", "us500", 0.07 + 0.25)
    part_a3(d_nq)
    # costs per leg of $100k: NQ $100k ~ 1.6 MNQ... use micros: commission ~$0.50 per micro per side; $100k of
    # NQ ~ 1.6 NQ-notional = 16 MNQ -> ~$16 round trip + 1 tick ($0.50 x 16 = $8) each way
    nq, es = load15("nas100"), load15("us500")
    pairs(nq, es, "15-min 2016-2026", comm_usd=16.0, ticks_usd=16.0)
    f = ROOT / "data" / "futures"
    n5 = pd.read_parquet(f / "NQ_5m_long_nt.parquet")[["close"]]
    e5 = pd.read_parquet(f / "ES_5m_long_nt.parquet")[["close"]]
    for x in (n5, e5):
        x.index = pd.DatetimeIndex(x.index).tz_convert(ET)
    pairs(n5, e5, "5-min Dec 2023 - Oct 2026", comm_usd=16.0, ticks_usd=16.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
