"""YOLO book sizing: growth-optimal risk fraction, currency caps and concurrency.

The book
--------
A separate $200 paper account with NO drawdown rule, NO profit target and no
deadline. Objective: maximise growth over a ONE-WEEK horizon; secondarily the
best win rate that does not cost growth. The only hard constraint is that ruin
ends compounding, so the question is genuinely "where is the Kelly peak and how
far before it should we stop".

Why this is not a Monte Carlo
-----------------------------
research/mc3.py credited each trade's full R at ENTRY. Measured median hold on
the daily stack is 10-18 days, so over a 5-day horizon most positions have NOT
resolved: the week's return is dominated by FLOATING P&L, not closed R. Crediting
R instantly overstates both the speed of compounding and the sharpness of the
Kelly peak. That mistake already misled this project once (see
scripts/ftmo_race_backtest.py, whose replay pattern this script copies).

So: every figure here comes from replaying actual history.
  * signals come from the real strategies on real daily bars (Sunday dropped)
  * positions open subject to risk fraction, concurrency and currency caps
  * stake is fixed-fractional: risk_frac x equity AT THE MOMENT OF ENTRY
  * equity is marked to market EVERY DAY off real closes, floating P&L included
  * equity <= 0 is ruin: positions are abandoned and the book stops trading

Two replay modes, both reported:
  COLD  -- the book starts flat on day D and runs 5 trading days. Every trading
           day in 10.6 years is a start date. This is literally "if I switch
           this on Monday, where am I Friday".
  CONT  -- one continuous 10.6-year path per config; weekly returns are read off
           that equity curve. This is the steady-state book (positions already
           running at the start of each week) and is the fairer read on
           long-run compounding.

Stack: the 4 legs that passed the train/OOS gate (RSI14 30/70, RSI7 25/75,
Boll20 2.0, Boll40 1.5) on DAILY, over the 17-pair FX basket.

Usage:
    python scripts/yolo_kelly_sizing.py --build      # generate + cache trades
    python scripts/yolo_kelly_sizing.py              # analyse (uses cache)
    python scripts/yolo_kelly_sizing.py --haircut 0.10   # slippage sensitivity
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]
LEGS = [
    ("Boll20", BollingerBounceStrategy, {"bollinger": {"period": 20, "std_dev": 2.0}}),
    ("RSI14", RSIReversionStrategy,
     {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    ("RSI7", RSIReversionStrategy,
     {"rsi": {"period": 7, "oversold": 25, "overbought": 75}}),
    ("Boll40", BollingerBounceStrategy, {"bollinger": {"period": 40, "std_dev": 1.5}}),
]

TPS = (2.0, 3.0, 4.0)          # 4R is the established best expectancy; 2/3R test
                               # whether faster resolution beats it over 5 days
WEEK = 5                       # trading days in the horizon
CACHE = ROOT / "backtest" / "yolo_trades.pkl"

RISKS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075, 0.10, 0.125,
         0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CONCS = (2, 4, 6, 10, 15, 20, 30)
# Currency-exposure caps. "cnt" counts any position touching the currency;
# "net" caps SIGNED exposure, so long EURUSD + short EURJPY offset in EUR
# instead of both consuming budget. 99 = no cap at all.
CCY_CAPS = (("none", 99, False), ("cnt2", 2, False), ("cnt1", 1, False),
            ("net2", 2, True), ("net1", 1, True))
HORIZONS = (5, 10, 21, 63)     # 1wk / 2wk / 1mo / 1qtr, in trading days
# Time stop, in trading days held. 0 = none (the deployed behaviour).
# This matters far more than it looks: 7.5% of trades on this stack run past
# 120 days and the longest runs 1661, so with no time stop a handful of zombie
# positions own the concurrency budget permanently and the book stops trading.
MAX_HOLDS = (0, 60, 30, 20, 10, 5)


# --------------------------------------------------------------------------
# Stage 1: generate the trade universe (slow, cached)
# --------------------------------------------------------------------------

def _cell(job):
    """Every trade for one (leg, symbol, tp), each carrying a DAILY R path."""
    li, sym, tp = job
    name, cls, params = LEGS[li]
    try:
        bars = load(sym, "1d")
        if bars is None:
            return None
        bars = bars[bars.index.dayofweek != 6]
        strat = {"session": None,
                 "exit": {"tp_r_multiple": tp, "sl_swing_lookback": 10}}
        strat.update(params)
        cfg = {"symbol": sym,
               "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                           "max_positions": 2, "daily_stop_loss_usd": -1e9},
               "strategy": strat}
        bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
        bt.run()
        closes = bars["close"]
        out = []
        for t in bt.closed:
            o = pd.Timestamp(t.entered_at).normalize()
            c = pd.Timestamp(t.exited_at).normalize()
            risk_px = abs(t.entry_price - t.stop_loss)
            if risk_px <= 0 or c < o:
                continue
            seg = closes.loc[o:c]
            if not len(seg):
                continue
            sign = 1.0 if t.direction.name == "LONG" else -1.0
            # daily unrealised R off real closes; final bar carries the real R
            path = (seg.values - t.entry_price) * sign / risk_px
            final = t.pnl_usd / RISK_USD
            path = np.asarray(path, dtype=float)
            path[-1] = final
            out.append({"leg": name, "sym": sym, "tp": tp, "dir": sign,
                        "open": o, "close": c,
                        # stop distance as a fraction of price: converts an
                        # annual carry rate into R per day held, which is how
                        # the uncharged swap cost gets sized in the report
                        "rpx_frac": risk_px / t.entry_price,
                        "dates": seg.index.to_numpy(), "path": path,
                        "final": final})
        return out
    except Exception:
        return None


def build(workers: int) -> None:
    jobs = [(li, s, tp) for tp in TPS for li in range(len(LEGS)) for s in SYMBOLS]
    print(f"building {len(jobs)} (leg x symbol x tp) cells on {workers} workers ...",
          flush=True)
    trades, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_cell, jobs, chunksize=2):
            done += 1
            if res:
                trades.extend(res)
            if done % 20 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} cells, {len(trades)} trades",
                      flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(trades, f)
    print(f"wrote {len(trades)} trades -> {CACHE}")


# --------------------------------------------------------------------------
# Stage 2a: classic Kelly / optimal-f on the measured R distribution
# --------------------------------------------------------------------------

def kelly_curve(rs: np.ndarray, fs: np.ndarray) -> np.ndarray:
    """E[log(1 + f*R)] per trade. -inf once f can wipe the account out."""
    out = np.empty(len(fs))
    for i, f in enumerate(fs):
        w = 1.0 + f * rs
        out[i] = -np.inf if (w <= 0).any() else float(np.mean(np.log(w)))
    return out


def report_kelly(rs: np.ndarray, label: str) -> float:
    fs = np.arange(0.005, 1.001, 0.005)
    g = kelly_curve(rs, fs)
    best = int(np.nanargmax(np.where(np.isfinite(g), g, -np.inf)))
    f_star = fs[best]
    # where growth per trade turns negative again
    neg = [fs[i] for i in range(best, len(fs)) if not np.isfinite(g[i]) or g[i] <= 0]
    f_zero = neg[0] if neg else float("nan")
    print(f"  {label:<22} n={len(rs):>5}  E={rs.mean():+.4f}R  "
          f"win={np.mean(rs > 0):.1%}  minR={rs.min():+.2f}  maxR={rs.max():+.2f}")
    print(f"  {'':<22} full-Kelly f*={f_star:.1%}  (g={g[best]:+.5f}/trade)   "
          f"g<=0 at f>={f_zero:.1%}   half-Kelly={f_star / 2:.1%}")
    return f_star


# --------------------------------------------------------------------------
# Stage 2b: historical replay with daily mark-to-market
# --------------------------------------------------------------------------

class Universe:
    """Trades for one TP, indexed onto a single global trading calendar."""

    def __init__(self, trades, haircut: float = 0.0):
        dates = sorted({d for t in trades for d in t["dates"]})
        self.cal = np.array(dates)
        pos = {d: i for i, d in enumerate(self.cal)}
        self.paths: list[np.ndarray] = []
        self.ccy: list[tuple[str, str]] = []
        self.sgn: list[float] = []
        self.finals: list[float] = []
        by_day: dict[int, list[int]] = {}
        for t in trades:
            i0, i1 = pos[t["dates"][0]], pos[t["dates"][-1]]
            arr = np.full(i1 - i0 + 1, np.nan)
            for d, r in zip(t["dates"], t["path"]):
                arr[pos[d] - i0] = r
            # holidays in one symbol but not another: carry the last mark
            m = np.isnan(arr)
            if m.any():
                idx = np.where(~m, np.arange(len(arr)), 0)
                np.maximum.accumulate(idx, out=idx)
                arr = arr[idx]
            arr = arr - haircut          # slippage/cost haircut, in R
            k = len(self.paths)
            self.paths.append(arr)
            self.finals.append(float(arr[-1]))
            s = t["sym"]
            self.ccy.append((s[:3], s[3:]))
            self.sgn.append(float(t["dir"]))
            by_day.setdefault(i0, []).append(k)
        # one fixed candidate ordering per day, shared by every config, so
        # configs are compared on identical signal streams
        rng = np.random.default_rng(20260804)
        self.by_day = {d: list(rng.permutation(v)) for d, v in sorted(by_day.items())}
        self.finals_arr = np.array(self.finals)

    def __len__(self):
        return len(self.cal)


def run_path(u: Universe, start: int, n_days: int, risk: float,
             max_conc: int, ccy_cap: int, signed: bool, max_hold: int = 0,
             track_equity=None):
    """Fixed-fractional replay with daily mark-to-market.

    Stake is risk * equity at the moment of entry; equity is 1.0 + realised +
    floating, revalued every day off real closes. Equity <= 0 is ruin: open
    positions are abandoned and the book stops trading for good.

    Returns (end_equity, min_equity, n_opened, list_of_closed_R).
    """
    eq = 1.0
    realised = 0.0
    open_pos: list[list] = []        # [path, age, stake, trade_idx]
    ccy_n: dict[str, float] = {}     # exposure budget consumed per currency
    min_eq = 1.0
    opened = 0
    closed_rs: list[float] = []
    end = min(start + n_days, len(u.cal))
    for i in range(start, end):
        # --- open new positions at this day's signals -------------------
        for k in u.by_day.get(i, ()):
            if len(open_pos) >= max_conc:
                break
            a, b = u.ccy[k]
            if signed:
                # long EURUSD is +1 EUR / -1 USD; offsetting trades free budget
                d = u.sgn[k]
                if abs(ccy_n.get(a, 0.0) + d) > ccy_cap or \
                   abs(ccy_n.get(b, 0.0) - d) > ccy_cap:
                    continue
                ccy_n[a] = ccy_n.get(a, 0.0) + d
                ccy_n[b] = ccy_n.get(b, 0.0) - d
            else:
                if ccy_n.get(a, 0.0) >= ccy_cap or ccy_n.get(b, 0.0) >= ccy_cap:
                    continue
                ccy_n[a] = ccy_n.get(a, 0.0) + 1
                ccy_n[b] = ccy_n.get(b, 0.0) + 1
            pth = u.paths[k]
            # time stop: exit at that day's close, i.e. the mark already on the
            # path. One extra spread, not modelled, see the caveats.
            lim = len(pth) - 1 if max_hold <= 0 else min(len(pth) - 1, max_hold)
            open_pos.append([pth, 0, risk * eq, k, lim])
            opened += 1
        # --- mark to market off real closes -----------------------------
        floating = 0.0
        still = []
        for p in open_pos:
            path, age, stake, k, lim = p
            if age >= lim:
                realised += path[lim] * stake
                closed_rs.append(path[lim])
                a, b = u.ccy[k]
                d = u.sgn[k] if signed else 1.0
                ccy_n[a] -= d
                ccy_n[b] -= -d if signed else 1.0
            else:
                floating += path[age] * stake
                p[1] = age + 1
                still.append(p)
        open_pos = still
        eq = 1.0 + realised + floating
        if eq <= 0.0:                # ruin: abandon everything, stop trading
            if track_equity is not None:
                track_equity.append(0.0)
            return 0.0, 0.0, opened, closed_rs
        min_eq = min(min_eq, eq)
        if track_equity is not None:
            track_equity.append(eq)
    return eq, min_eq, opened, closed_rs


def _growth(r: np.ndarray) -> float:
    """Per-period expected log growth -- the ACTUAL Kelly objective.

    Median is only a proxy for it, and for a 23%-win / 4R-target distribution
    the two disagree violently. A single ruin week makes this -inf, which is
    the correct verdict: you cannot compound through a zero.
    """
    if (r <= -1.0).any():
        return float("-inf")
    return float(np.mean(np.log1p(r)))


def cold_stats(u: Universe, risk: float, max_conc: int, ccy_cap: int,
               signed: bool, max_hold: int = 0, step: int = 1,
               horizon: int = WEEK):
    """Every trading day is a start date for a flat book run for `horizon` days."""
    rets, mins, wins, nop = [], [], [], []
    for s in range(0, len(u.cal) - horizon, step):
        eq, mn, op, crs = run_path(u, s, horizon, risk, max_conc, ccy_cap,
                                   signed, max_hold)
        rets.append(eq - 1.0)
        mins.append(mn)
        nop.append(op)
        wins.extend(crs)
    r = np.array(rets)
    m = np.array(mins)
    g = _growth(r)
    return {
        "n": len(r),
        "median": float(np.median(r)),
        "mean": float(r.mean()),
        "geo": float(np.expm1(g)) if np.isfinite(g) else float("-inf"),
        "p_win": float(np.mean(r > 0)),
        "p_half": float(np.mean(m <= 0.5)),
        "p_ruin": float(np.mean(m <= 0.0)),
        "p05": float(np.percentile(r, 5)),
        "p95": float(np.percentile(r, 95)),
        "opened": float(np.mean(nop)),
        "trade_win": float(np.mean(np.array(wins) > 0)) if wins else float("nan"),
    }


def cont_stats(u: Universe, risk: float, max_conc: int, ccy_cap: int,
               signed: bool, max_hold: int = 0):
    """One continuous 10.6y path; weekly returns read off that equity curve."""
    eqs: list[float] = []
    _, _, taken, crs = run_path(u, 0, len(u.cal), risk, max_conc, ccy_cap,
                                signed, max_hold, track_equity=eqs)
    e = np.array(eqs)
    ruined = bool(len(e) < len(u.cal) - 1 or (e <= 0).any())
    ruin_day = int(len(e)) if ruined else -1
    base = {"taken": taken,
            "trade_win": float(np.mean(np.array(crs) > 0)) if crs else float("nan"),
            "trade_e": float(np.mean(crs)) if crs else float("nan")}
    if len(e) < WEEK + 2:                        # blown up almost immediately
        return {**base, "sub": [float("nan")] * 5, "sub_pos": 0, "median": -1.0,
                "mean": -1.0, "geo": float("-inf"), "p_win": 0.0, "p_half": 1.0,
                "final": 0.0, "cagr": -1.0, "maxdd": 1.0, "ruined": True,
                "ruin_day": ruin_day}
    # overlapping WEEK-day windows of the steady-state book
    a, b = e[:-WEEK], e[WEEK:]
    with np.errstate(divide="ignore", invalid="ignore"):
        wk = np.where(a > 0, b / a - 1.0, -1.0)
    peak = np.maximum.accumulate(np.maximum(e, 1e-12))
    dd = float(np.max(1.0 - e / peak))
    years = len(u.cal) / 252.0
    final = 0.0 if ruined else float(e[-1])
    cagr = (final ** (1 / years) - 1.0) if final > 0 else -1.0
    g = _growth(wk)
    # Sub-period stability: a 10.6y CAGR off ONE path can be a single lucky
    # stretch. Split the equity curve into 5 chunks and annualise each.
    sub = []
    for c0, c1 in zip(np.linspace(0, len(e), 6)[:-1].astype(int),
                      np.linspace(0, len(e), 6)[1:].astype(int)):
        if c1 - c0 < 20 or e[c0] <= 0:
            sub.append(float("nan"))
            continue
        sub.append(float((e[c1 - 1] / e[c0]) ** (252.0 / (c1 - c0)) - 1.0)
                   if e[c1 - 1] > 0 else -1.0)
    return {
        **base,
        "sub": sub,
        "sub_pos": int(np.sum(np.array(sub) > 0)),
        "median": float(np.median(wk)),
        "mean": float(np.mean(wk)),
        "geo": float(np.expm1(g)) if np.isfinite(g) else float("-inf"),
        "p_win": float(np.mean(wk > 0)),
        "p_half": float(np.mean(wk <= -0.5)),
        "final": final,
        "cagr": cagr,
        "maxdd": dd,
        "ruined": ruined,
        "ruin_day": ruin_day,
    }



# --------------------------------------------------------------------------
# parallel driver
# --------------------------------------------------------------------------

_U: Universe | None = None


def _init(trades, haircut):
    global _U
    _U = Universe(trades, haircut)


def _sweep_cell(job):
    risk, conc, cname, cap, signed, hold, step = job
    return {"risk": risk, "conc": conc, "cap": cname, "hold": hold,
            "cold": cold_stats(_U, risk, conc, cap, signed, hold, step),
            "cont": cont_stats(_U, risk, conc, cap, signed, hold)}


def _horizon_cell(job):
    risk, conc, cname, cap, signed, hold, horizon, step = job
    return {"risk": risk, "conc": conc, "cap": cname, "hold": hold, "h": horizon,
            "cold": cold_stats(_U, risk, conc, cap, signed, hold, step, horizon)}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

W = 104
HOLD_L = {0: "none"}


def hl(h):
    return HOLD_L.get(h, f"{h}d")


def _g(v, w=8):
    return f"{'-inf':>{w}}" if v == float("-inf") else f"{v:>+{w}.2%}"


def _c(k, w=9):
    return f"{'RUIN':>{w}}" if k["ruined"] else f"{k['cagr']:>+{w}.1%}"


ROW_HDR = (f"{'median':>8}{'mean':>8}{'geo':>8}{'P(win)':>7}{'P(-50%)':>8}"
           f"{'P(ruin)':>8}{'p05':>8} |{'median':>8}{'P(win)':>7}{'CAGR':>9}"
           f"{'maxDD':>7}{'trades':>7}")
ROW_SUB = (f"{'COLD START: flat book, all 2743 weeks':^55} |"
           f"{'CONTINUOUS 10.6y BOOK':^38}")


def row(r):
    c, k = r["cold"], r["cont"]
    return (f"{c['median']:>+8.2%}{c['mean']:>+8.2%}{_g(c['geo'])}{c['p_win']:>7.1%}"
            f"{c['p_half']:>8.2%}{c['p_ruin']:>8.2%}{c['p05']:>+8.1%} |"
            f"{k['median']:>+8.2%}{k['p_win']:>7.1%}{_c(k)}{k['maxdd']:>7.1%}"
            f"{k['taken']:>7}")


def table(sel, lead_hdr, lead):
    """Print one block; `lead` maps a row to its left-hand label columns."""
    print(f"{'':>{len(lead_hdr)}} |{ROW_SUB}")
    print(f"{lead_hdr} |{ROW_HDR}")
    print("-" * W)
    for r in sel:
        print(f"{lead(r)} |{row(r)}")


def peaks(rows, key, lbl, n=6):
    sel = [r for r in rows if np.isfinite(key(r))]
    print(f"\n  {lbl}")
    if not sel:
        print("    (nothing qualifies)")
        return
    print(f"    {'conc':>5}{'ccy':>6}{'hold':>6}{'risk':>7}{'medW':>9}{'meanW':>9}"
          f"{'geoW':>9}{'P(win)':>8}{'P(-50%)':>9}{'P(ruin)':>9}{'CAGR':>9}"
          f"{'maxDD':>8}{'trades':>8}")
    for r in sorted(sel, key=key, reverse=True)[:n]:
        c, k = r["cold"], r["cont"]
        print(f"    {r['conc']:>5}{r['cap']:>6}{hl(r['hold']):>6}{r['risk']:>7.1%}"
              f"{c['median']:>+9.2%}{c['mean']:>+9.2%}{_g(c['geo'], 9)}"
              f"{c['p_win']:>8.1%}{c['p_half']:>9.2%}{c['p_ruin']:>9.2%}"
              f"{_c(k)}{k['maxdd']:>8.1%}{k['taken']:>8}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--tp", type=float, default=4.0)
    ap.add_argument("--haircut", type=float, default=0.0,
                    help="R subtracted from every trade, to model live slippage")
    ap.add_argument("--step", type=int, default=1, help="start-date stride")
    args = ap.parse_args()

    if args.build:
        build(min(args.workers, os.cpu_count() or 4))
        return 0
    if not CACHE.exists():
        print("no cache; run with --build first")
        return 1
    with open(CACHE, "rb") as f:
        all_trades = pickle.load(f)
    trades = [t for t in all_trades if t["tp"] == args.tp]

    print("=" * W)
    print("YOLO BOOK ($200, no drawdown rule, no target) -- growth-optimal sizing")
    print(f"4-leg daily stack x {len(SYMBOLS)} FX pairs | TP={args.tp:.0f}R | "
          f"horizon = {WEEK} trading days | slippage haircut = {args.haircut:.2f}R")
    print("Every number is a historical replay with DAILY mark-to-market. No Monte Carlo.")
    print("=" * W)

    # ---- (1) per-trade R distribution and classic optimal-f --------------
    print("\n### 1. PER-TRADE R DISTRIBUTION AND CLASSIC OPTIMAL-f")
    print("(single bet resolving instantly -- an UPPER bound on safe f, and already")
    print(" far below the 10-15% the earlier crude work suggested)\n")
    for tp in TPS:
        rs = np.array([t["final"] for t in all_trades if t["tp"] == tp]) - args.haircut
        report_kelly(rs, f"4-leg stack TP={tp:.0f}R")
    print()
    for name, _, _ in LEGS:
        rs = np.array([t["final"] for t in all_trades
                       if t["tp"] == args.tp and t["leg"] == name]) - args.haircut
        report_kelly(rs, name)

    holds = np.array([len(t["path"]) - 1 for t in trades])
    print(f"\n  HOLD TIME (trading days): median {np.median(holds):.0f}  "
          f"mean {holds.mean():.1f}  p75 {np.percentile(holds, 75):.0f}  "
          f"p90 {np.percentile(holds, 90):.0f}  p99 {np.percentile(holds, 99):.0f}  "
          f"max {holds.max():.0f}")
    print(f"  -> {np.mean(holds > WEEK):.0%} of trades are still OPEN after one week: "
          f"weekly P&L is mostly FLOATING, not closed R.")
    print(f"  -> {np.mean(holds > 120):.1%} run past 120 days. With no time stop these "
          f"zombies own the concurrency budget forever.")

    print("\n  EFFECT OF A TIME STOP on the per-trade distribution "
          "(exit at that day's close):")
    for h in MAX_HOLDS:
        rs = np.array([t["path"][len(t["path"]) - 1 if h <= 0
                                 else min(len(t["path"]) - 1, h)] for t in trades])
        report_kelly(rs - args.haircut, f"time stop {hl(h)}")

    # ---- (2)(3) the full grid -------------------------------------------
    jobs = [(r, c, nm, cap, sg, h, args.step)
            for h in MAX_HOLDS for nm, cap, sg in CCY_CAPS
            for c in CONCS for r in RISKS]
    print(f"\nreplaying {len(jobs)} configs x {len(trades)} trades, every start "
          f"week of 10.6 years ...", flush=True)
    pool = dict(max_workers=min(args.workers, os.cpu_count() or 4),
                initializer=_init, initargs=(trades, args.haircut))
    with ProcessPoolExecutor(**pool) as ex:
        rows = list(ex.map(_sweep_cell, jobs, chunksize=6))

    # the geometry that maximises long-run compounding among survivors
    live = [r for r in rows if not r["cont"]["ruined"]]
    best = max(live, key=lambda r: r["cont"]["cagr"])
    bh, bc, bcap = best["hold"], best["conc"], best["cap"]

    # ---- (3) concurrency x risk -----------------------------------------
    print(f"\n{'=' * W}\n### 3. CONCURRENCY x RISK   (ccy cap = {bcap}, "
          f"time stop = {hl(bh)})\n{'=' * W}")
    for conc in CONCS:
        table([r for r in rows if r["hold"] == bh and r["cap"] == bcap
               and r["conc"] == conc],
              f"{'conc':>5}{'risk':>7}",
              lambda r: f"{r['conc']:>5}{r['risk']:>7.1%}")
        print()

    # ---- (2) currency caps ----------------------------------------------
    print(f"{'=' * W}\n### 2. CURRENCY EXPOSURE CAP   (conc = {bc}, "
          f"time stop = {hl(bh)})")
    print("### cnt2/cnt1 = max 2/1 open positions touching a currency")
    print("### net2/net1 = same but SIGNED, so offsetting trades free the budget")
    print("=" * W)
    for nm, _, _ in CCY_CAPS:
        table([r for r in rows if r["hold"] == bh and r["conc"] == bc
               and r["cap"] == nm],
              f"{'ccy':>6}{'risk':>7}",
              lambda r: f"{r['cap']:>6}{r['risk']:>7.1%}")
        print()

    # ---- time stop -------------------------------------------------------
    print(f"{'=' * W}\n### TIME STOP   (conc = {bc}, ccy cap = {bcap})")
    print(f"### 'trades' = how many of the {len(trades)} available signals the book "
          f"could actually take in 10.6y")
    print("=" * W)
    for h in MAX_HOLDS:
        table([r for r in rows if r["hold"] == h and r["conc"] == bc
               and r["cap"] == bcap],
              f"{'hold':>6}{'risk':>7}",
              lambda r: f"{hl(r['hold']):>6}{r['risk']:>7.1%}")
        print()

    # ---- (4) where each objective peaks ----------------------------------
    print("=" * W)
    print(f"### 4. WHERE EACH OBJECTIVE PEAKS  (over all {len(rows)} "
          f"risk x conc x ccy-cap x time-stop configs)")
    print("=" * W)
    peaks(rows, lambda r: r["cold"]["median"], "MAX MEDIAN weekly return, cold start")
    peaks(rows, lambda r: r["cold"]["geo"],
          "MAX EXPECTED LOG GROWTH per week, cold start   <-- the true Kelly objective")
    peaks(rows, lambda r: r["cont"]["median"],
          "MAX MEDIAN weekly return, continuous book")
    peaks(live, lambda r: r["cont"]["cagr"],
          "MAX realised 10.6y CAGR, survivors only        <-- best actual compounding")
    peaks([r for r in live if r["cold"]["p_half"] <= 0.005],
          lambda r: r["cont"]["cagr"],
          "MAX CAGR s.t. P(lose half in a week) <= 0.5%   <-- the deployable answer")
    peaks([r for r in live if r["cold"]["p_half"] <= 0.0001],
          lambda r: r["cont"]["cagr"],
          "MAX CAGR s.t. P(lose half in a week) == 0")
    peaks(rows, lambda r: r["cold"]["p_win"], "MAX weekly WIN RATE")

    # ---- the measured Kelly curve ---------------------------------------
    print("\n" + "=" * W)
    print("### THE MEASURED KELLY CURVE: growth vs risk fraction")
    print("### peak = argmax over the risk grid; 0@ = first f above the peak where "
          "the metric turns <= 0")
    print("=" * W)
    print(f"  {'hold':>5}{'conc':>5}{'ccy':>6} | {'COLD median':^25} | "
          f"{'COLD log-growth':^25} | {'CONT CAGR':^21}")
    for h in MAX_HOLDS:
        shown = False
        for nm, _, _ in CCY_CAPS:
            for conc in CONCS:
                if nm != bcap and conc != bc:
                    continue
                s = [x for x in rows if x["hold"] == h and x["cap"] == nm
                     and x["conc"] == conc]
                if not s:
                    continue

                def pz(get, ss=s):
                    v = [(x["risk"], get(x)) for x in ss]
                    pk = max(v, key=lambda t: t[1] if np.isfinite(t[1]) else -9e9)
                    z = next((f for f, m in v
                              if f > pk[0] and (not np.isfinite(m) or m <= 0)), None)
                    return pk, ("n/a" if z is None else f"{z:.1%}")

                pm, zm = pz(lambda x: x["cold"]["median"])
                pg, zg = pz(lambda x: x["cold"]["geo"])
                pc, _z = pz(lambda x: float("-inf") if x["cont"]["ruined"]
                            else x["cont"]["cagr"])
                pcv = "all RUIN" if not np.isfinite(pc[1]) else f"{pc[1]:>+6.1%}"
                print(f"  {hl(h):>5}{conc:>5}{nm:>6} | peak {pm[0]:>5.1%} "
                      f"({pm[1]:>+7.2%}) 0@{zm:>5} | peak {pg[0]:>5.1%} "
                      f"({pg[1]:>+7.2%}) 0@{zg:>5} | peak {pc[0]:>5.1%} ({pcv})")
                shown = True
        if shown:
            print()

    # ---- robustness of the top configs -----------------------------------
    print("=" * W)
    print("### ROBUSTNESS OF THE TOP CONFIGS")
    print("### sub-period CAGR = the 10.6y path cut into 5 equal chunks, each")
    print("### annualised. A real edge is positive in most of them; one lucky")
    print("### stretch is not. 'yr' columns are cold-start 252-day replays.")
    print("=" * W)
    top = sorted(live, key=lambda r: r["cont"]["cagr"], reverse=True)[:6]
    yjobs = [(r["risk"], r["conc"], r["cap"],
              *[(c, s) for n, c, s in CCY_CAPS if n == r["cap"]][0],
              r["hold"], 252, max(args.step, 3)) for r in top]
    with ProcessPoolExecutor(**pool) as ex:
        yrows = list(ex.map(_horizon_cell, yjobs, chunksize=1))
    print(f"  {'conc':>5}{'ccy':>6}{'hold':>6}{'risk':>7}{'CAGR':>9}{'+sub':>6}"
          f"   {'sub-period CAGRs (5 x ~2.1y)':<44}"
          f"{'yr med':>9}{'yr P(win)':>10}{'yr P(ruin)':>11}")
    for r, y in zip(top, yrows):
        s = "  ".join(f"{v:>+6.1%}" if np.isfinite(v) else "  n/a " for v in
                      r["cont"]["sub"])
        c = y["cold"]
        print(f"  {r['conc']:>5}{r['cap']:>6}{hl(r['hold']):>6}{r['risk']:>7.1%}"
              f"{r['cont']['cagr']:>+9.1%}{r['cont']['sub_pos']:>4}/5   {s:<44}"
              f"{c['median']:>+9.1%}{c['p_win']:>10.1%}{c['p_ruin']:>11.2%}")

    # ---- where the P&L actually comes from -------------------------------
    print("\n  WHERE THE P&L COMES FROM, by hold length (all "
          f"{len(trades)} TP={args.tp:.0f}R trades, unweighted R):")
    print(f"    {'hold bucket':<14}{'n':>7}{'share of n':>12}{'total R':>10}"
          f"{'share of R':>12}{'E[R]':>9}{'win%':>8}")
    fr = np.array([t["final"] for t in trades])
    tot = fr.sum()
    for lo, hi, lbl in ((0, 5, "<= 1 week"), (5, 20, "1-4 weeks"),
                        (20, 60, "1-3 months"), (60, 252, "3-12 months"),
                        (252, 10 ** 9, "> 1 year")):
        m = (holds > lo) & (holds <= hi)
        if not m.any():
            continue
        print(f"    {lbl:<14}{m.sum():>7}{m.mean():>11.1%}{fr[m].sum():>+10.1f}"
              f"{fr[m].sum() / tot:>11.1%}{fr[m].mean():>+9.3f}"
              f"{np.mean(fr[m] > 0):>8.1%}")
    print("    NOTE: multi-month holds are NOT charged swap/financing anywhere in")
    print("    this project's cost model. Any share of P&L they contribute is")
    print("    overstated by an unknown amount. Sized below.")

    # ---- how big is the uncharged financing cost? ------------------------
    # 1R = the stop distance. As a fraction of price that is rpx_frac, so an
    # annual carry rate of x costs (x/252)/rpx_frac R per day held. Long holds
    # are therefore charged in R far more heavily than short ones.
    rpx = np.array([t.get("rpx_frac", np.nan) for t in trades])
    ok = np.isfinite(rpx) & (rpx > 0)
    if ok.any():
        print(f"\n  UNCHARGED SWAP / FINANCING SENSITIVITY")
        print(f"    stop distance as a fraction of price: median "
              f"{np.median(rpx[ok]):.3%}  mean {rpx[ok].mean():.3%}")
        print(f"    so 1R ~= {np.median(rpx[ok]):.2%} of notional, and one year of "
              f"carry at rate x costs x/{np.median(rpx[ok]):.4f} R")
        print(f"    {'adverse carry':<16}{'R/day (med)':>13}{'stack E[R]':>13}"
              f"{'total R':>11}{'verdict':>28}")
        for rate in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
            per_day = (rate / 252.0) / rpx
            cost = np.where(ok, per_day * holds, 0.0)
            e = float((fr - cost).mean())
            tr_ = float((fr - cost).sum())
            v = ("edge intact" if e > 0.06 else
                 "edge halved" if e > 0.03 else
                 "edge marginal" if e > 0 else "EDGE GONE")
            print(f"    {rate:<16.2%}{np.median(per_day[ok]):>13.4f}{e:>+13.3f}"
                  f"{tr_:>+11.0f}{v:>28}")
        print("    Sign is NOT knowable from price data alone -- these legs are")
        print("    mean-reverting and may sit on either side of the differential.")
        print("    Magnitude is the point: a 2-3% adverse differential erases the")
        print("    entire measured edge, because the edge lives in long holds.")

    # ---- horizon sensitivity --------------------------------------------
    bcap_i, bsg = [(cap, sg) for nm, cap, sg in CCY_CAPS if nm == bcap][0]
    print("=" * W)
    print(f"### HORIZON SENSITIVITY at conc={bc} ccy={bcap} timestop={hl(bh)} "
          f"(cold start, book begins flat)")
    print("### one week is SHORT next to the hold time -- this is how much of the "
          "edge is not yet visible at the bell")
    print("=" * W)
    hjobs = [(r, bc, bcap, bcap_i, bsg, bh, h, args.step)
             for h in HORIZONS for r in (0.02, 0.05, 0.10, 0.15, 0.25)]
    with ProcessPoolExecutor(**pool) as ex:
        hrows = list(ex.map(_horizon_cell, hjobs, chunksize=2))
    print(f"  {'horizon':>9}{'risk':>7}{'median':>10}{'mean':>10}{'geo':>10}"
          f"{'P(win)':>9}{'P(-50%)':>10}{'P(ruin)':>10}")
    for h in HORIZONS:
        for r in [x for x in hrows if x["h"] == h]:
            c = r["cold"]
            print(f"  {f'{h}d':>9}{r['risk']:>7.1%}{c['median']:>+10.2%}"
                  f"{c['mean']:>+10.2%}{_g(c['geo'], 10)}{c['p_win']:>9.1%}"
                  f"{c['p_half']:>10.2%}{c['p_ruin']:>10.2%}")
        print()

    out = ROOT / "backtest" / f"yolo_sizing_tp{args.tp:.0f}.json"
    with open(out, "w") as f:
        json.dump(rows, f, default=lambda o: None if o == float("-inf") else o)
    print(f"wrote {len(rows)} config rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
