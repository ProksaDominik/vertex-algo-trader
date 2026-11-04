"""Futures-native INTRADAY hypotheses for Lucid, tested on 10 years of data.

Why
---
The CFD strategy library showed no edge on Lucid futures (futures_roster_search
.py: 1,776 tests, 16% positive, fewer t>2 results than chance). This script
tests a different kind of idea: effects documented on US index futures that
live inside ONE New York session, so they fit Lucid's flat-by-4:45 PM ET rule.

Each hypothesis is stated once, codified with no discretion, and given a small
parameter neighbourhood -- enough to see a plateau, not enough to curve-fit.

  H1 intraday momentum   (Gao, Han, Li & Zhou 2018): the return from the prior
     16:00 ET close to 10:00 ET predicts the 15:30-16:00 ET return.
  H2 opening-range breakout: the 09:30-10:00 ET range, breakouts until 12:00 ET,
     with a narrow-range filter and/or a daily-trend filter.
  H3 opening gap: fade gaps back toward the prior close, or follow them.
  H4 volatility breakout (Williams): open +/- k x prior-day range, flat by 15:55.

Data
----
10.5 years (2016-2026) of 15m bars from data/*_15m_long.parquet (Dukascopy
cash-index / spot data) as proxies for the futures -- same price scale and
intraday dynamics -- with FUTURES costs applied: one tick of slippage on every
market fill and stop, Lucid's commission both ways, and a bar that spans stop
and target counted as the stop. Survivors are then re-checked on the real
futures bars from NinjaTrader (data/futures/, ~15 months).

Output: data/futures/hypotheses_results.csv (one row per config x market),
data/futures/hypotheses_trades.parquet (every trade, for stacking / Monte Carlo).

Usage:
    .venv/Scripts/python.exe scripts/futures_hypotheses.py
"""
from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import futures  # noqa: E402

OUT = ROOT / "data" / "futures"
ET = "America/New_York"

# research market -> (10-year proxy file, futures contract whose costs apply)
MARKETS = {
    "ES": ("us500", "MES"),
    "NQ": ("nas100", "MNQ"),
    "YM": ("us30", "MYM"),
    "GC": ("xauusd", "MGC"),
    "CL": ("wti", "MCL"),
}


def M(h: int, m: int) -> int:
    """Minutes after midnight, New York time."""
    return h * 60 + m


def keep_awake() -> None:
    """Stop Windows sleeping while the research runs (released on exit)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


# --- data ------------------------------------------------------------------------

@dataclass
class Day:
    date: object
    m: np.ndarray        # bar open time, minutes after midnight ET
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray

    def at(self, minute: int) -> int | None:
        k = np.flatnonzero(self.m == minute)
        return int(k[0]) if len(k) else None


def bars_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index).tz_convert(ET)
    return df[["open", "high", "low", "close"]].astype(float)


def load_days(df: pd.DataFrame) -> list[Day]:
    minutes = df.index.hour * 60 + df.index.minute
    rth = df[(minutes >= M(9, 30)) & (minutes < M(16, 0))]
    rmin = rth.index.hour * 60 + rth.index.minute
    days = []
    for d, idx in pd.Series(np.arange(len(rth)), index=rth.index.date).groupby(level=0):
        pos = idx.to_numpy()
        mm = np.asarray(rmin[pos])
        if len(pos) < 24 or mm[0] != M(9, 30) or mm[-1] != M(15, 45):
            continue                                  # half days / gaps: skip, never guess
        g = rth.iloc[pos]
        days.append(Day(d, mm, g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(),
                        g["close"].to_numpy()))
    return days


def daily_frame(days: list[Day]) -> pd.DataFrame:
    dd = pd.DataFrame({"open": [x.o[0] for x in days], "high": [x.h.max() for x in days],
                       "low": [x.l.min() for x in days], "close": [x.c[-1] for x in days]},
                      index=[x.date for x in days])
    dd["range"] = dd["high"] - dd["low"]
    dd["atr14"] = dd["range"].rolling(14).mean().shift(1)         # known before today's open
    dd["sma20"] = dd["close"].rolling(20).mean().shift(1)
    dd["prev_close"] = dd["close"].shift(1)
    dd["prev_range"] = dd["range"].shift(1)
    dd["trend"] = np.sign(dd["prev_close"] - dd["sma20"])         # +1 up / -1 down, as of yesterday
    return dd


# --- execution model -----------------------------------------------------------

class Cost:
    def __init__(self, contract: str, mult: float = 1.0):
        s = futures.spec(contract)
        self.tick = s["tick_size"] * mult                 # slippage per market/stop fill
        self.comm_pts = 2 * s["commission"] / s["point_value"] * mult
        self.pv = s["point_value"]


def walk(x: Day, start: int, side: int, stop: float | None, target: float | None,
         last_exit: int, cost: Cost) -> tuple[float, str]:
    """Hold from bar `start` until stop / target / the time exit."""
    for i in range(start, len(x.m)):
        hit_stop = stop is not None and ((x.l[i] <= stop) if side > 0 else (x.h[i] >= stop))
        hit_tgt = target is not None and ((x.h[i] >= target) if side > 0 else (x.l[i] <= target))
        if hit_stop:
            return stop - side * cost.tick, "sl"          # stop fills a tick worse; also when both hit
        if hit_tgt:
            return target, "tp"                            # limit order: no slippage
        if x.m[i] >= last_exit:
            return x.c[i] - side * cost.tick, "time"
    return x.c[-1] - side * cost.tick, "time"


def trade(rows: list, date, side: int, entry_raw: float, exit_px: float, stop: float | None,
          reason: str, cost: Cost) -> None:
    entry = entry_raw + side * cost.tick                    # market / stop entry a tick worse
    pts = (exit_px - entry) * side - cost.comm_pts
    risk = abs(entry - stop) if stop is not None else np.nan
    rows.append({"date": date, "side": side, "entry": entry, "exit": exit_px, "pts": pts,
                 "risk_pts": risk, "usd": pts * cost.pv, "reason": reason})


# --- hypotheses ------------------------------------------------------------------

def h1_momentum(days, dd, cost, variant: str, stop_mult: float) -> list:
    """Sign of (10:00 ET close / prior 16:00 close) -> trade 15:30-16:00 ET."""
    r1 = {}
    for x in days:
        pc, i = dd.at[x.date, "prev_close"], x.at(M(9, 45))
        if not pd.isna(pc) and i is not None:
            r1[x.date] = x.c[i] / pc - 1.0
    med = np.nanmedian(np.abs(list(r1.values()))) if r1 else 0.0
    out = []
    for x in days:
        atr = dd.at[x.date, "atr14"]
        if x.date not in r1 or pd.isna(atr):
            continue
        if variant == "strong" and abs(r1[x.date]) < med:
            continue
        side = 1 if r1[x.date] > 0 else -1
        i = x.at(M(15, 30))
        if i is None:
            continue
        entry = x.o[i]
        stop = entry - side * stop_mult * atr / 4.0
        px, why = walk(x, i, side, stop, None, M(15, 45), cost)
        trade(out, x.date, side, entry, px, stop, why, cost)
    return out


def h2_orb(days, dd, cost, stop_mode: str, target_r: float | None, filt: str) -> list:
    out = []
    for x in days:
        atr, trend = dd.at[x.date, "atr14"], dd.at[x.date, "trend"]
        if pd.isna(atr):
            continue
        orh, orl = x.h[:2].max(), x.l[:2].min()
        width = orh - orl
        if width <= 0 or (filt in ("narrow", "both") and width > 0.5 * atr):
            continue
        for i in range(2, len(x.m) - 1):
            if x.m[i] >= M(12, 0):
                break
            side = 1 if x.c[i] > orh else (-1 if x.c[i] < orl else 0)
            if side == 0:
                continue
            if filt in ("trend", "both") and side != trend:
                break                                        # first break against trend: no trade today
            entry = x.o[i + 1]
            stop = (orl if side > 0 else orh) if stop_mode == "range" else (orh + orl) / 2.0
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                break
            tgt = entry + side * target_r * abs(entry - stop) if target_r else None
            px, why = walk(x, i + 1, side, stop, tgt, M(15, 45), cost)
            trade(out, x.date, side, entry, px, stop, why, cost)
            break
    return out


def h3_gap(days, dd, cost, mode: str, thresh: float) -> list:
    out = []
    for x in days:
        pc, atr = dd.at[x.date, "prev_close"], dd.at[x.date, "atr14"]
        if pd.isna(pc) or pd.isna(atr) or atr <= 0:
            continue
        op = x.o[0]
        gap = op - pc
        if abs(gap) < thresh * atr:
            continue
        gdir = 1 if gap > 0 else -1
        if mode == "fade":
            side = -gdir
            stop, tgt = op + gdir * abs(gap), pc              # stop one gap further, target the fill
            px, why = walk(x, 0, side, stop, tgt, M(15, 45), cost)
            trade(out, x.date, side, op, px, stop, why, cost)
        else:                                                  # follow, confirmed at 10:00
            i = x.at(M(10, 0))
            if i is None or (x.c[i - 1] - op) * gdir <= 0:
                continue
            side = gdir
            entry = x.o[i]
            stop = x.l[:2].min() if side > 0 else x.h[:2].max()
            if (side > 0 and stop >= entry) or (side < 0 and stop <= entry):
                continue
            px, why = walk(x, i, side, stop, None, M(15, 45), cost)
            trade(out, x.date, side, entry, px, stop, why, cost)
    return out


def h4_volbreak(days, dd, cost, k: float, filt: str) -> list:
    out = []
    for x in days:
        pr, trend = dd.at[x.date, "prev_range"], dd.at[x.date, "trend"]
        if pd.isna(pr) or pr <= 0:
            continue
        op = x.o[0]
        up, dn = op + k * pr, op - k * pr
        for i in range(len(x.m)):
            if x.m[i] >= M(15, 0):
                break
            side = 1 if x.h[i] >= up else (-1 if x.l[i] <= dn else 0)
            if side == 0:
                continue
            if filt == "trend" and side != trend:
                break
            entry = up if side > 0 else dn                       # stop entry at the level...
            stop = op                                            # ...risking back to the open
            # The trigger bar itself can reach the stop; start the walk on it.
            px, why = walk(x, i, side, stop, None, M(15, 45), cost)
            trade(out, x.date, side, entry, px, stop, why, cost)
            break
    return out


CONFIGS: list[tuple[str, dict]] = (
    [("H1_momentum", {"variant": v, "stop_mult": s}) for v in ("all", "strong") for s in (1.0, 2.0)]
    + [("H2_orb", {"stop_mode": sm, "target_r": tr, "filt": f})
       for sm in ("range", "mid") for tr in (1.0, 2.0, None) for f in ("none", "narrow", "trend", "both")]
    + [("H3_gap", {"mode": m, "thresh": t}) for m in ("fade", "follow") for t in (0.25, 0.5)]
    + [("H4_volbreak", {"k": k, "filt": f}) for k in (0.25, 0.5, 0.75) for f in ("none", "trend")]
)
FUNCS = {"H1_momentum": h1_momentum, "H2_orb": h2_orb, "H3_gap": h3_gap, "H4_volbreak": h4_volbreak}


def metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    t = t.sort_values("date")
    r = t["usd"]
    years = pd.to_datetime(t["date"]).dt.year
    by_year = r.groupby(years).mean()
    half = len(t) // 2
    sd = r.std(ddof=1)
    eq = r.cumsum()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    span_days = (pd.Timestamp(t["date"].iloc[-1]) - pd.Timestamp(t["date"].iloc[0])).days
    return {
        "n": len(t), "usd_mean": r.mean(), "win": (r > 0).mean(),
        "pf": wins / losses if losses > 0 else np.inf,
        "t": r.mean() / (sd / np.sqrt(len(t))) if sd > 0 else 0.0,
        "h1": r.iloc[:half].mean(), "h2": r.iloc[half:].mean(),
        "years_pos": (by_year > 0).mean(), "n_years": len(by_year),
        "worst_year": by_year.min(), "max_dd_usd": (eq.cummax() - eq).max(),
        "per_week": len(t) / max(1.0, span_days / 7),
        "risk_usd_med": float(np.nanmedian(t["risk_pts"])) if t["risk_pts"].notna().any() else np.nan,
    }


def main() -> int:
    keep_awake()
    results, all_trades = [], []
    for mkt, (proxy, contract) in MARKETS.items():
        days = load_days(bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = daily_frame(days)
        print(f"{mkt}: {len(days)} full RTH days {days[0].date} -> {days[-1].date}", flush=True)
        pv = futures.point_value(contract)
        for fam, params in CONFIGS:
            cfg = f"{fam}|{json.dumps(params, sort_keys=True)}"
            base = pd.DataFrame(FUNCS[fam](days, dd, Cost(contract), **params))
            stress = pd.DataFrame(FUNCS[fam](days, dd, Cost(contract, mult=2.0), **params))
            m = metrics(base)
            results.append({"market": mkt, "contract": contract, "family": fam, "config": cfg, **m,
                            "risk_usd_med": m.get("risk_usd_med", np.nan) * pv,
                            "usd_mean_2x": metrics(stress).get("usd_mean", np.nan)})
            if not base.empty:
                base["market"], base["config"], base["contract"] = mkt, cfg, contract
                all_trades.append(base)
    res = pd.DataFrame(results)
    res.to_csv(OUT / "hypotheses_results.csv", index=False)
    pd.concat(all_trades).to_parquet(OUT / "hypotheses_trades.parquet")
    show = res[res["n"] >= 100].sort_values("t", ascending=False)
    cols = ["market", "config", "n", "usd_mean", "usd_mean_2x", "win", "pf", "t", "h1", "h2",
            "years_pos", "worst_year", "max_dd_usd", "risk_usd_med", "per_week"]
    pd.set_option("display.width", 260)
    print(show[cols].head(30).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n{len(res)} tests; n>=100: {len(show)}; t>2: {(show['t'] > 2).sum()} "
          f"(chance ~{0.023 * len(show):.0f}); t>3: {(show['t'] > 3).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
