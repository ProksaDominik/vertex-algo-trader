"""Search for a SECOND intraday edge to stack with ORB + trend (MNQ/MGC).

Two independent edges raise a prop account's pass rate far more than tuning
one, so each idea here trades at a different time of day than the ORB legs:

  H5 afternoon continuation (NY): at 12:00 or 13:00 ET, if the move from the
     09:30 open is >= k x ATR14 in the direction of the daily trend, join it;
     stop at the open (or halfway back); out by the 15:45 bar.
  H6 London breakout WITH the daily-trend filter: the Asian range (18:00-03:00
     ET) breaking between 03:00 and 06:00/08:00 ET; stop at the far side; 2R or
     time exit at 11:30 / 15:45 ET. Gold and FX (the failed unfiltered version
     is in futures_roster_search.py).
  H7 crude after the Wednesday 10:30 ET inventory report: the 10:30-11:00
     range, breakouts until 12:00/13:00 with the trend filter; out at 14:30
     (pit close) or the 15:45 bar.

Same costs and pessimism as futures_hypotheses.py; judged by the plateau test
(share of neighbouring cells positive), then both halves / positive years.
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

M = fh.M
ET = fh.ET
# extra proxies for H6 (FX futures): proxy file -> contract whose costs apply
FX = {"6E": ("eurusd", "6E"), "6B": ("gbpusd", "6B"), "6J": ("usdjpy", "6J"), "6A": ("audusd", "6A")}


# --- overnight session days (18:00 ET -> 16:45 ET), minutes since 18:00 ET ----

def S(h: int, m: int) -> int:
    """Session minutes since 18:00 ET (so the overnight never wraps)."""
    return (h * 60 + m - 18 * 60) % 1440


def load_sessions(df: pd.DataFrame) -> dict:
    et_min = df.index.hour * 60 + df.index.minute
    smin = (et_min - 18 * 60) % 1440
    trade_date = (df.index + pd.Timedelta(hours=6)).date          # 18:00 ET starts the next trade date
    out = {}
    frame = pd.DataFrame({"s": smin, "o": df["open"].to_numpy(), "h": df["high"].to_numpy(),
                          "l": df["low"].to_numpy(), "c": df["close"].to_numpy()}, index=df.index)
    for d, g in frame.groupby(trade_date):
        g = g[g["s"] < S(16, 45)]
        if len(g) < 60:
            continue
        out[d] = fh.Day(d, g["s"].to_numpy(), g["o"].to_numpy(), g["h"].to_numpy(),
                        g["l"].to_numpy(), g["c"].to_numpy())
    return out


def h6_london(sessions: dict, dd: pd.DataFrame, cost, range_end: int, entry_end: int,
              target_r, exit_at: int) -> list:
    out = []
    for d, x in sessions.items():
        if d not in dd.index or pd.isna(dd.at[d, "trend"]):
            continue
        trend = dd.at[d, "trend"]
        asian = x.m < range_end
        if asian.sum() < 20:
            continue
        hi, lo = x.h[asian].max(), x.l[asian].min()
        start = int(np.flatnonzero(~asian)[0]) if (~asian).any() else None
        if start is None:
            continue
        for i in range(start, len(x.m) - 1):
            if x.m[i] >= entry_end:
                break
            side = 1 if x.c[i] > hi else (-1 if x.c[i] < lo else 0)
            if side == 0:
                continue
            if side != trend:
                break
            entry = x.o[i + 1]
            stop = lo if side > 0 else hi
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                break
            tgt = entry + side * target_r * abs(entry - stop) if target_r else None
            px, why = fh.walk(x, i + 1, side, stop, tgt, exit_at, cost)
            fh.trade(out, d, side, entry, px, stop, why, cost)
            break
    return out


def h5_afternoon(days, dd, cost, at: int, k: float, stop_mode: str) -> list:
    out = []
    for x in days:
        atr, trend = dd.at[x.date, "atr14"], dd.at[x.date, "trend"]
        if pd.isna(atr) or pd.isna(trend) or trend == 0:
            continue
        i = x.at(at)
        if i is None:
            continue
        op, px_now = x.o[0], x.c[i - 1]
        move = (px_now - op) * trend
        if move < k * atr:
            continue
        side = int(trend)
        entry = x.o[i]
        stop = op if stop_mode == "open" else (op + px_now) / 2.0
        if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
            continue
        p, why = fh.walk(x, i, side, stop, None, M(15, 45), cost)
        fh.trade(out, x.date, side, entry, p, stop, why, cost)
    return out


def h7_eia(days, dd, cost, entry_end: int, exit_at: int, target_r) -> list:
    out = []
    for x in days:
        if pd.Timestamp(x.date).weekday() != 2:                # Wednesdays
            continue
        trend = dd.at[x.date, "trend"]
        i0 = x.at(M(10, 30))
        if i0 is None or pd.isna(trend):
            continue
        hi, lo = x.h[i0:i0 + 2].max(), x.l[i0:i0 + 2].min()
        for i in range(i0 + 2, len(x.m) - 1):
            if x.m[i] >= entry_end:
                break
            side = 1 if x.c[i] > hi else (-1 if x.c[i] < lo else 0)
            if side == 0:
                continue
            if side != trend:
                break
            entry = x.o[i + 1]
            stop = lo if side > 0 else hi
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                break
            tgt = entry + side * target_r * abs(entry - stop) if target_r else None
            p, why = fh.walk(x, i + 1, side, stop, tgt, exit_at, cost)
            fh.trade(out, x.date, side, entry, p, stop, why, cost)
            break
    return out


def score(rows: list) -> dict:
    t = pd.DataFrame(rows)
    return fh.metrics(t) if not t.empty else {"n": 0}


def main() -> int:
    fh.keep_awake()
    res = []
    # H5 on the NY markets, H7 on crude
    for mkt, (proxy, contract) in fh.MARKETS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = fh.daily_frame(days)
        for at, k, sm in product((M(12, 0), M(13, 0)), (0.3, 0.5, 0.7), ("open", "half")):
            m = score(h5_afternoon(days, dd, fh.Cost(contract), at, k, sm))
            m2 = score(h5_afternoon(days, dd, fh.Cost(contract, 2.0), at, k, sm))
            res.append({"hyp": "H5_afternoon", "market": mkt, "cell": f"{at // 60}h k{k} {sm}", **m,
                        "usd_2x": m2.get("usd_mean")})
        if mkt == "CL":
            for ee, ex, tr in product((M(12, 0), M(13, 0)), (M(14, 30), M(15, 45)), (2.0, None)):
                m = score(h7_eia(days, dd, fh.Cost(contract), ee, ex, tr))
                m2 = score(h7_eia(days, dd, fh.Cost(contract, 2.0), ee, ex, tr))
                res.append({"hyp": "H7_eia", "market": mkt, "cell": f"to{ee // 60}h ex{ex // 60}:{ex % 60:02d} "
                            f"{tr or 'time'}", **m, "usd_2x": m2.get("usd_mean")})
    # H6 on gold + FX
    h6_markets = {"GC": ("xauusd", "MGC"), **FX}
    for mkt, (proxy, contract) in h6_markets.items():
        bars = fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet")
        dd = fh.daily_frame(fh.load_days(bars))
        sessions = load_sessions(bars)
        for re_, ee, tr, ex in product((S(2, 0), S(3, 0)), (S(6, 0), S(8, 0)), (2.0, None), (S(11, 30), S(15, 45))):
            if ee <= re_:
                continue
            m = score(h6_london(sessions, dd, fh.Cost(contract), re_, ee, tr, ex))
            m2 = score(h6_london(sessions, dd, fh.Cost(contract, 2.0), re_, ee, tr, ex))
            res.append({"hyp": "H6_london_trend", "market": mkt,
                        "cell": f"range->{(re_ // 60 + 18) % 24}h entry->{(ee // 60 + 18) % 24}h "
                                f"{tr or 'time'} exit{(ex // 60 + 18) % 24}:{ex % 60:02d}",
                        **m, "usd_2x": m2.get("usd_mean")})
    r = pd.DataFrame(res)
    r.to_csv(fh.OUT / "second_edge_results.csv", index=False)
    pd.set_option("display.width", 240)
    summ = r.groupby(["hyp", "market"]).agg(cells=("usd_mean", "size"),
                                              positive=("usd_mean", lambda s: (s > 0).mean()),
                                              pos_2x=("usd_2x", lambda s: (s > 0).mean()),
                                              mean_usd=("usd_mean", "mean"), mean_t=("t", "mean"),
                                              best_t=("t", "max"))
    print("=== plateau summary (share of neighbouring cells positive) ===")
    print(summ.to_string(float_format=lambda v: f"{v:.2f}"))
    good = summ[(summ["positive"] >= 0.8) & (summ["mean_t"] > 0.8)]
    print("\n=== markets whose whole neighbourhood is positive ===")
    for (hyp, mkt) in good.index:
        cells = r[(r.hyp == hyp) & (r.market == mkt)].sort_values("t", ascending=False)
        print(f"-- {hyp} {mkt}")
        print(cells[["cell", "n", "usd_mean", "usd_2x", "t", "h1", "h2", "years_pos", "per_week"]]
              .head(6).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
