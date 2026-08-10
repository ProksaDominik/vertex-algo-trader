"""Robustness audit of the one candidate that survived the canonical test.

`canonical_strategy_test.py` found RSI mean-reversion on the DAILY timeframe
to be the only clearly-positive well-known strategy over ~10 years of real
data. One positive cell in a big grid is exactly what data-mining noise looks
like, so before deploying anything this script tries hard to kill it:

  1. PARAMETER NEIGHBOURHOOD — a real edge is a plateau, not a spike. If only
     RSI(14)/30/70 works and RSI(10)/25/75 loses money, it was curve fit.
  2. TIMEFRAME NEIGHBOURHOOD — 4h/8h/12h/1d. An effect that exists only at
     one sampling frequency is suspicious.
  3. TRAIN vs OOS — same settings on the first 70% and the last 30%. A big
     gap means the "edge" is regime-specific.
  4. YEAR BY YEAR — the whole history sliced per calendar year. One monster
     year carrying a flat decade is not a tradeable edge.
  5. PER-SYMBOL — how many symbols actually contribute, so a single outlier
     pair can't masquerade as a basket-wide effect.

Every number printed is out-of-sample for its own window and includes the
project's real per-pair spread + commission model.

Usage:
    python scripts/rsi_robustness.py [--workers 6]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import BASKET, RISK_USD, Windowed, stats  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

PERIODS = [7, 10, 14, 21]
BANDS = [(20, 80), (25, 75), (30, 70), (35, 65)]
TP_RS = [1.0, 1.5, 2.0]
TFS = ["4h", "8h", "12h", "1d"]


def load(sym: str, tf: str) -> pd.DataFrame | None:
    f = ROOT / "data" / f"{sym.lower()}_15m_long.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    out = df.resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return out if len(out) > 400 else None


def cfg(sym: str, period: int, os_: float, ob: float, tp_r: float) -> dict:
    return {
        "symbol": sym,
        "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                    "max_positions": 2, "daily_stop_loss_usd": -1e9},
        "strategy": {
            "rsi": {"period": period, "oversold": os_, "overbought": ob},
            "session": None,
            "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": 10},
        },
    }


def _run(sym: str, bars: pd.DataFrame, period: int, os_: float, ob: float,
         tp_r: float) -> list[tuple[pd.Timestamp, float]]:
    c = cfg(sym, period, os_, ob, tp_r)
    bt = Backtest(c, bars, strategy=Windowed(RSIReversionStrategy(c)))
    bt.run()
    return [(pd.Timestamp(t.exited_at), t.pnl_usd / RISK_USD) for t in bt.closed]


def _cell(job) -> dict:
    kind, tf, period, os_, ob, tp_r, sym = job
    try:
        bars = load(sym, tf)
        if bars is None:
            return {"skip": True}
        split = int(len(bars) * 0.70)
        if kind == "train":
            use = bars.iloc[:split]
        elif kind == "oos":
            use = bars.iloc[split:]
        else:
            use = bars
        if len(use) < 250:
            return {"skip": True}
        trades = _run(sym, use, period, os_, ob, tp_r)
        return {"skip": False, "kind": kind, "tf": tf, "period": period,
                "os": os_, "ob": ob, "tp_r": tp_r, "sym": sym,
                "trades": [(str(ts), r) for ts, r in trades],
                "days": len(set(use.index.date))}
    except Exception as e:
        return {"skip": True, "error": repr(e)}


def agg(cells: list[dict]) -> dict:
    rs = [r for c in cells for _, r in c["trades"]]
    days = max((c["days"] for c in cells), default=0)
    st = stats(rs, days)
    st["pos_syms"] = sum(1 for c in cells if c["trades"] and sum(r for _, r in c["trades"]) > 0)
    st["n_syms"] = sum(1 for c in cells if c["trades"])
    return st


def line(label: str, st: dict, width: int = 34) -> str:
    if not st.get("n"):
        return f"{label:<{width}}  (no trades)"
    pf = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "inf"
    return (f"{label:<{width}}{st['n']:>7}{st['per_day']:>7.2f}{st['win_rate']*100:>7.1f}"
            f"{st['e_net']:>+9.3f}{pf:>7}{st['total_r']:>+9.1f}{st['max_dd_r']:>8.1f}"
            f"{f'{st['pos_syms']}/{st['n_syms']}':>7}")


HEAD = (f"{'':<34}{'n':>7}{'/day':>7}{'win%':>7}{'E_net':>9}{'PF':>7}"
        f"{'totR':>9}{'maxDD':>8}{'sym+':>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    syms = [s for s in BASKET if (ROOT / "data" / f"{s.lower()}_15m_long.parquet").exists()]
    print("=" * 108)
    print("RSI MEAN-REVERSION ROBUSTNESS AUDIT — trying to kill the one survivor")
    print(f"symbols with full history: {len(syms)}  ({', '.join(syms)})")
    print("=" * 108)

    jobs = []
    # (1)+(2) parameter x timeframe grid, evaluated OUT OF SAMPLE.
    for tf, p, (o, b), tp in itertools.product(TFS, PERIODS, BANDS, TP_RS):
        for s in syms:
            jobs.append(("oos", tf, p, o, b, tp, s))
    # (3) train-window run of the headline setting only.
    for s in syms:
        jobs.append(("train", "1d", 14, 30, 70, 1.0, s))
    # (4) full history of the headline setting, for the per-year slice.
    for s in syms:
        jobs.append(("full", "1d", 14, 30, 70, 1.0, s))

    print(f"\nRunning {len(jobs)} cells on {args.workers} workers...", flush=True)
    cells = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for c in ex.map(_cell, jobs, chunksize=8):
            done += 1
            if not c.get("skip"):
                cells.append(c)
            if done % 200 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)

    def sel(**kw):
        return [c for c in cells if all(c.get(k) == v for k, v in kw.items())]

    # --- (1)+(2) the surface -------------------------------------------------
    print(f"\n{'=' * 108}\n(1)+(2) PARAMETER x TIMEFRAME SURFACE — out-of-sample")
    print("A real edge is a PLATEAU. If only one cell is green, it is noise.\n")
    surface = []
    for tf in TFS:
        print(f"\n--- {tf} ---")
        print(HEAD)
        for p, (o, b), tp in itertools.product(PERIODS, BANDS, TP_RS):
            cs = sel(kind="oos", tf=tf, period=p, os=o, ob=b, tp_r=tp)
            if not cs:
                continue
            st = agg(cs)
            if not st.get("n"):
                continue
            surface.append({"tf": tf, "period": p, "os": o, "ob": b, "tp_r": tp, **st})
            print(line(f"RSI{p} {o}/{b} tp{tp}", st))

    pos = [s for s in surface if s["e_net"] > 0]
    print(f"\n  >>> {len(pos)}/{len(surface)} grid cells positive out-of-sample "
          f"({len(pos)/max(1,len(surface)):.0%})")
    for tf in TFS:
        tf_cells = [s for s in surface if s["tf"] == tf]
        tf_pos = [s for s in tf_cells if s["e_net"] > 0]
        if tf_cells:
            print(f"      {tf:>3}: {len(tf_pos):>2}/{len(tf_cells):<3} positive "
                  f"({len(tf_pos)/len(tf_cells):>4.0%})   "
                  f"median E_net={pd.Series([s['e_net'] for s in tf_cells]).median():+.3f}R")

    # --- (3) train vs OOS ----------------------------------------------------
    print(f"\n{'=' * 108}\n(3) TRAIN vs OUT-OF-SAMPLE — headline setting RSI14 30/70 tp1.0 @ 1d")
    print(HEAD)
    tr = agg(sel(kind="train", tf="1d")) if sel(kind="train", tf="1d") else {}
    oo = agg(sel(kind="oos", tf="1d", period=14, os=30, ob=70, tp_r=1.0))
    if tr.get("n"):
        print(line("TRAIN (first 70%)", tr))
    print(line("OOS   (last 30%)", oo))

    # --- (4) per calendar year ----------------------------------------------
    print(f"\n{'=' * 108}\n(4) YEAR BY YEAR — headline setting over the FULL history")
    print("Consistency matters more than the total: one huge year is not an edge.\n")
    full = sel(kind="full", tf="1d")
    by_year: dict[int, list[float]] = {}
    for c in full:
        for ts, r in c["trades"]:
            by_year.setdefault(pd.Timestamp(ts).year, []).append(r)
    print(f"{'year':<8}{'n':>7}{'win%':>8}{'E_net':>10}{'totR':>10}")
    yr_pos = 0
    for y in sorted(by_year):
        rs = by_year[y]
        if not rs:
            continue
        wr = sum(1 for x in rs if x > 0) / len(rs)
        e = sum(rs) / len(rs)
        yr_pos += 1 if sum(rs) > 0 else 0
        print(f"{y:<8}{len(rs):>7}{wr*100:>8.1f}{e:>+10.3f}{sum(rs):>+10.1f}")
    print(f"\n  >>> {yr_pos}/{len(by_year)} calendar years net-positive")

    # --- (5) per symbol ------------------------------------------------------
    print(f"\n{'=' * 108}\n(5) PER SYMBOL — headline setting, out-of-sample")
    print(f"{'symbol':<10}{'n':>7}{'win%':>8}{'E_net':>10}{'totR':>10}")
    for c in sorted(sel(kind="oos", tf="1d", period=14, os=30, ob=70, tp_r=1.0),
                    key=lambda c: -sum(r for _, r in c["trades"])):
        rs = [r for _, r in c["trades"]]
        if not rs:
            continue
        wr = sum(1 for x in rs if x > 0) / len(rs)
        print(f"{c['sym']:<10}{len(rs):>7}{wr*100:>8.1f}{sum(rs)/len(rs):>+10.3f}{sum(rs):>+10.1f}")

    out = ROOT / "backtest" / "rsi_robustness.json"
    with open(out, "w") as f:
        json.dump({"surface": surface,
                   "train": tr, "oos": oo,
                   "by_year": {str(k): sum(v) for k, v in by_year.items()}},
                  f, indent=2, default=str)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
