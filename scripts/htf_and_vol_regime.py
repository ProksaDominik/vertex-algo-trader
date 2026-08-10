"""Two untested ways to RAISE the expectancy of the daily survivors.

Background (established, not re-tested here)
--------------------------------------------
All 16 canonical strategies are negative at 1h and 4h. Only the daily
timeframe survives, and only two strategies on it clear the "positive in BOTH
the train and the OOS half" gate: RSI(14) 30/70 and Bollinger(20, 2.0), best
at a 4R target. Costs bind everything -- SL exits realise about -1.054R rather
than -1.000R, and live slippage runs 2-12x the modelled figure -- so a gross
edge under roughly 0.15R is not real.

This script does not look for a new strategy. It asks two questions about the
two that already work.

(A) HIGHER TIMEFRAME. Nothing above 1d has ever been tested in this repo.
    The a-priori reason to expect a gain is purely mechanical: the stop
    distance on a weekly bar is several times a daily stop, so the fixed
    spread + commission is a much smaller fraction of 1R. If the underlying
    signal has any edge at all, less cost drag per unit of R should show up as
    higher expectancy. Tested on 2-day, 3-day and weekly bars against a 1d
    control, targets 2R-6R.

    The counterweight, which the results must be read against: fewer bars means
    fewer trades (a weekly RSI-14 crossing 30 is rare), and the hold time
    stretches from weeks to months, which is why MEDIAN HOLD IN DAYS and the
    end-of-data censoring rate are reported alongside expectancy.

(B) VOLATILITY REGIME CONDITIONING. Bucket the EXISTING daily RSI/Bollinger
    trades by the volatility regime in force at entry. If expectancy is
    concentrated in particular regimes, a filter raises the edge without
    inventing anything.

    The regime variable is the trailing percentile rank of ATR(14)/close within
    the prior 252 bars, then shifted one bar. Both properties matter:

      * rolling + trailing means it never sees the future, and it needs no
        train/test normalisation because it is self-scaling;
      * the extra .shift(1) means the value attached to a trade is computable
        strictly BEFORE the entry bar even existed.

    That second point is deliberate. This project has already been bitten once
    by a "stop after a losing day" rule that decided whether to open a trade
    using an outcome that was not knowable until 18 days later. Any conditioning
    variable here must be knowable before the entry bar prints, and shift(1)
    makes that unarguable.

    The quintile threshold is chosen on the TRAIN half alone and then applied
    frozen to OOS. A per-symbol raw-ATR threshold variant is run as a robustness
    check on the same rule.

MANDATORY GATE: a result counts only if expectancy is positive in BOTH halves.
Train and OOS are printed side by side for every row, including the failures.

Usage:
    python scripts/htf_and_vol_regime.py                 # both parts
    python scripts/htf_and_vol_regime.py --part A
    python scripts/htf_and_vol_regime.py --part B --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Workers are single-threaded on purpose: parallelism is across cells, so a
# BLAS thread pool per process only multiplies memory.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_strategy_test import RISK_USD, Windowed, load  # noqa: E402
from metatraderbot.backtest import Backtest  # noqa: E402
from metatraderbot.indicators import atr  # noqa: E402
from metatraderbot.strategies_bollinger import BollingerBounceStrategy  # noqa: E402
from metatraderbot.strategies_rsi import RSIReversionStrategy  # noqa: E402

# The 17 FX pairs with full 10.5y history. Deliberately the whole basket: small
# symbol samples in this repo have produced false-positive edge before.
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD",
]

STRATS = {
    "RSI":       (RSIReversionStrategy, {"rsi": {"period": 14, "oversold": 30, "overbought": 70}}),
    "Bollinger": (BollingerBounceStrategy, {"bollinger": {}}),
}

TIMEFRAMES = ["1d", "2d", "3d", "1w"]   # 1d is the control, already known-good
TP_RS = [2.0, 3.0, 4.0, 5.0, 6.0]

OOS_FRAC = 0.30
MIN_BARS_PER_HALF = 60          # weekly train ~387 / oos ~166, so this is slack
SURVIVOR_TP = 4.0               # the established best daily geometry, frozen

VOL_LOOKBACK = 252              # ~1 trading year of trailing daily bars
N_QUINTILES = 5
MIN_TRAIN_RETAIN = 0.40         # a filter must keep >=40% of train trades


# --- bar construction -----------------------------------------------------

def daily_bars(sym: str) -> pd.DataFrame | None:
    """Daily bars with the partial Sunday bar removed.

    The Sunday bar is only the 21:00-24:00 UTC re-open stub; leaving it in
    fabricates a bar with a fraction of a real session's range, which distorts
    both the indicators and any ATR-based volatility measure.
    """
    d = load(sym, "1d")
    if d is None:
        return None
    d = d[d.index.dayofweek != 6]
    return d if len(d) > 500 else None


def to_tf(d: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate Sunday-free daily bars up to `tf`.

    N-day bars group N CONSECUTIVE TRADING days rather than N calendar days:
    a calendar resample would silently emit bars that straddle a weekend with
    only one session's worth of data in them. Weekly uses W-FRI so each bucket
    is exactly one Mon-Fri trading week.
    """
    if tf == "1d":
        return d
    if tf == "1w":
        return d.resample("W-FRI").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    n = int(tf[:-1])                     # "2d" -> 2, "3d" -> 3
    grp = np.arange(len(d)) // n
    out = d.groupby(grp).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"})
    out.index = d.index[::n][:len(out)]  # label each bar by its FIRST day
    return out


def split_ts(d: pd.DataFrame) -> pd.Timestamp:
    """One calendar split point per symbol, taken from the daily bars.

    Splitting each timeframe at its own 70% BAR index would put the boundary on
    a slightly different date per timeframe, so a weekly-vs-daily comparison
    would be partly a comparison of different eras. A shared date removes that.
    """
    return d.index[int(len(d) * (1 - OOS_FRAC))]


def make_cfg(sym: str, params: dict, tp_r: float, sl_lookback: int = 10) -> dict:
    strat = {"session": None,
             "exit": {"tp_r_multiple": tp_r, "sl_swing_lookback": sl_lookback}}
    strat.update(params)
    return {"symbol": sym,
            "capital": {"risk_usd_per_trade": RISK_USD, "initial_usd": 10_000,
                        "max_positions": 2, "daily_stop_loss_usd": -1e9},
            "strategy": strat}


def run_bt(cls, cfg: dict, bars: pd.DataFrame) -> list:
    """Mean-reversion strategies run without a trailing stop (see STRATEGIES.md)."""
    bt = Backtest(cfg, bars, strategy=Windowed(cls(cfg)))
    bt.run()
    return bt.closed


# --- stats ----------------------------------------------------------------

def summarise(trades: list, trading_days: int) -> dict | None:
    if not trades:
        return None
    rs = [t.pnl_usd / RISK_USD for t in trades]
    holds = [(t.exited_at - t.entered_at).total_seconds() / 86400.0 for t in trades]
    wins = [r for r in rs if r > 0]
    gross_loss = -sum(r for r in rs if r <= 0)
    eod = sum(1 for t in trades if t.exit_reason == "end-of-data")
    return {
        "n": len(rs),
        "per_day": len(rs) / trading_days if trading_days else 0.0,
        "win": len(wins) / len(rs),
        "e": float(np.mean(rs)),
        "total": float(sum(rs)),
        "pf": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
        "med_hold_d": float(np.median(holds)),
        "eod_pct": eod / len(rs),
    }


def agg_stats(rs: list[float]) -> dict | None:
    if not rs:
        return None
    wins = [r for r in rs if r > 0]
    gl = -sum(r for r in rs if r <= 0)
    return {"n": len(rs), "win": len(wins) / len(rs), "e": float(np.mean(rs)),
            "total": float(sum(rs)),
            "pf": (sum(wins) / gl) if gl > 0 else float("inf")}


def _pf(v: float) -> str:
    return f"{v:.2f}" if v != float("inf") else "inf"


# ==========================================================================
# PART A -- higher timeframes
# ==========================================================================

def _cell_a(job):
    name, tf, tp_r, sym = job
    cls, params = STRATS[name]
    try:
        d = daily_bars(sym)
        if d is None:
            return None
        cut = split_ts(d)
        bars = to_tf(d, tf)
        cfg = make_cfg(sym, params, tp_r)
        out = {"name": name, "tf": tf, "tp_r": tp_r, "sym": sym}
        for half, b in (("train", bars[bars.index < cut]), ("oos", bars[bars.index >= cut])):
            if len(b) < MIN_BARS_PER_HALF:
                continue
            # Trading-day denominator comes from the DAILY bars covering the
            # same span, so trades/day means the same thing at every timeframe.
            dd = d[(d.index >= b.index[0]) & (d.index <= b.index[-1])]
            trades = run_bt(cls, cfg, b)
            out[half] = {
                "rs": [t.pnl_usd / RISK_USD for t in trades],
                "holds": [(t.exited_at - t.entered_at).total_seconds() / 86400.0
                          for t in trades],
                "eod": sum(1 for t in trades if t.exit_reason == "end-of-data"),
                "days": len(set(dd.index.date)),
            }
        return out
    except Exception as e:
        return {"name": name, "tf": tf, "tp_r": tp_r, "sym": sym, "error": repr(e)}


def pool_half(cells: list, half: str) -> dict | None:
    rs = [r for c in cells for r in c.get(half, {}).get("rs", [])]
    holds = [h for c in cells for h in c.get(half, {}).get("holds", [])]
    eod = sum(c.get(half, {}).get("eod", 0) for c in cells)
    days = max((c.get(half, {}).get("days", 0) for c in cells), default=0)
    st = agg_stats(rs)
    if st is None:
        return None
    st["per_day"] = len(rs) / days if days else 0.0
    st["med_hold_d"] = float(np.median(holds)) if holds else 0.0
    st["eod_pct"] = eod / len(rs)
    return st


def part_a(workers: int) -> list[dict]:
    jobs = [(n, tf, tp, s) for n in STRATS for tf in TIMEFRAMES
            for tp in TP_RS for s in SYMBOLS]
    print("=" * 118)
    print("PART A -- HIGHER TIMEFRAME  (RSI + Bollinger, 2d / 3d / weekly bars, 1d control)")
    print(f"{len(STRATS)} strategies x {len(TIMEFRAMES)} timeframes x {len(TP_RS)} targets "
          f"x {len(SYMBOLS)} symbols = {len(jobs)} cells")
    print("=" * 118, flush=True)

    cells, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(_cell_a, jobs, chunksize=8):
            done += 1
            if c:
                cells.append(c)
            if done % 200 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)}", flush=True)

    for e in [c for c in cells if c.get("error")][:5]:
        print(f"  [ERROR] {e['name']} {e['tf']} {e['sym']}: {e['error']}")

    results = []
    hdr = (f"{'strategy':<11}{'tf':>4}{'TP':>5}{'n_tr':>6}{'n_oos':>6}{'/day':>7}"
           f"{'winTR%':>8}{'winOOS%':>8}{'E_train':>9}{'E_oos':>9}{'PF_oos':>8}"
           f"{'medHold_d':>11}{'eod%':>6}  verdict")
    print(f"\n{hdr}\n{'-' * 118}")
    for name in STRATS:
        for tf in TIMEFRAMES:
            for tp in TP_RS:
                mine = [c for c in cells if c.get("name") == name
                        and c.get("tf") == tf and c.get("tp_r") == tp]
                t, o = pool_half(mine, "train"), pool_half(mine, "oos")
                if not o or not t:
                    print(f"{name:<11}{tf:>4}{tp:>5.1f}   (insufficient trades)")
                    continue
                both = t["e"] > 0 and o["e"] > 0
                verdict = "PASS both halves" if both else (
                    "oos+ only" if o["e"] > 0 else ("train+ only" if t["e"] > 0 else "fail"))
                print(f"{name:<11}{tf:>4}{tp:>5.1f}{t['n']:>6}{o['n']:>6}{o['per_day']:>7.3f}"
                      f"{t['win']*100:>8.1f}{o['win']*100:>8.1f}{t['e']:>+9.3f}{o['e']:>+9.3f}"
                      f"{_pf(o['pf']):>8}{o['med_hold_d']:>11.1f}{o['eod_pct']*100:>6.1f}"
                      f"  {verdict}")
                results.append({"part": "A", "name": name, "tf": tf, "tp_r": tp,
                                "train": t, "oos": o, "both_positive": both})
            print()
    return results


# ==========================================================================
# PART B -- volatility regime conditioning on the daily survivors
# ==========================================================================

def vol_rank_series(d: pd.DataFrame) -> pd.Series:
    """Trailing percentile rank of ATR(14)/close, shifted one bar.

    ATR is divided by close so the measure is a percentage volatility and is
    comparable across a 150-JPY pair and a 1.1-EUR pair. The rolling rank makes
    it comparable across eras without any train-half normalisation, which is
    what keeps the OOS half genuinely untouched.

    .shift(1) is the anti-lookahead guarantee: the value carried by a bar is
    built only from bars that closed strictly before it.
    """
    ap = atr(d, 14) / d["close"]
    return ap.rolling(VOL_LOOKBACK).rank(pct=True).shift(1)


def _cell_b(job):
    name, sym = job
    cls, params = STRATS[name]
    try:
        d = daily_bars(sym)
        if d is None:
            return None
        cut = split_ts(d)
        vr = vol_rank_series(d)
        ap = (atr(d, 14) / d["close"]).shift(1)      # raw, for the robustness variant
        cfg = make_cfg(sym, params, SURVIVOR_TP)
        rows = []
        for half, b in (("train", d[d.index < cut]), ("oos", d[d.index >= cut])):
            for t in run_bt(cls, cfg, b):
                ts = pd.Timestamp(t.entered_at)
                rank = vr.get(ts, np.nan)
                raw = ap.get(ts, np.nan)
                rows.append({"name": name, "sym": sym, "half": half,
                             "r": t.pnl_usd / RISK_USD,
                             "rank": float(rank) if pd.notna(rank) else None,
                             "raw": float(raw) if pd.notna(raw) else None})
        return rows
    except Exception:
        return None


def part_b(workers: int) -> list[dict]:
    jobs = [(n, s) for n in STRATS for s in SYMBOLS]
    print("\n" + "=" * 118)
    print("PART B -- VOLATILITY REGIME CONDITIONING on the existing daily survivors")
    print(f"RSI + Bollinger, daily, TP {SURVIVOR_TP:.0f}R (frozen). Regime = trailing "
          f"{VOL_LOOKBACK}-bar percentile rank of ATR(14)/close, shifted 1 bar.")
    print("=" * 118, flush=True)

    trades = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(_cell_b, jobs, chunksize=4):
            if rows:
                trades.extend(rows)
    print(f"  collected {len(trades)} trades "
          f"({sum(1 for t in trades if t['rank'] is None)} without a regime value "
          f"-- warm-up bars, excluded from the bucket tables)")

    results = []
    edges = np.linspace(0, 1, N_QUINTILES + 1)
    for name in STRATS:
        mine = [t for t in trades if t["name"] == name and t["rank"] is not None]
        base = {h: agg_stats([t["r"] for t in mine if t["half"] == h])
                for h in ("train", "oos")}
        if not base["train"] or not base["oos"]:
            continue

        print(f"\n### {name} daily TP{SURVIVOR_TP:.0f}R -- expectancy by volatility quintile")
        print(f"{'quintile (ATR pctile)':<26}{'n_train':>9}{'E_train':>10}"
              f"{'n_oos':>9}{'E_oos':>10}{'winOOS%':>9}{'PF_oos':>9}")
        print("-" * 118)

        buckets = []
        for q in range(N_QUINTILES):
            lo, hi = edges[q], edges[q + 1]
            sel = [t for t in mine if lo <= t["rank"] < hi or (q == N_QUINTILES - 1 and t["rank"] == 1.0)]
            st = {h: agg_stats([t["r"] for t in sel if t["half"] == h]) for h in ("train", "oos")}
            buckets.append(st)
            label = f"Q{q + 1}  {lo:.1f}-{hi:.1f}" + ("  (calmest)" if q == 0 else
                                                      "  (wildest)" if q == N_QUINTILES - 1 else "")
            tr, oo = st["train"], st["oos"]
            print(f"{label:<26}{(tr['n'] if tr else 0):>9}"
                  f"{(tr['e'] if tr else float('nan')):>+10.3f}"
                  f"{(oo['n'] if oo else 0):>9}{(oo['e'] if oo else float('nan')):>+10.3f}"
                  f"{(oo['win'] * 100 if oo else float('nan')):>9.1f}"
                  f"{(_pf(oo['pf']) if oo else '-'):>9}")

        # --- threshold chosen on TRAIN ONLY, then frozen -------------------
        # Restricted to CONTIGUOUS quintile bands: "volatility between X and Y"
        # is a regime statement, whereas an arbitrary subset of quintiles is
        # just curve-fitting to whichever buckets happened to pay. The >=40%
        # retention floor stops the rule collapsing onto one lucky bucket.
        n_train_all = base["train"]["n"]
        best = None
        for a in range(N_QUINTILES):
            for b in range(a, N_QUINTILES):
                if a == 0 and b == N_QUINTILES - 1:
                    continue                                  # that is the baseline
                sel = [t for t in mine if a <= min(int(t["rank"] * N_QUINTILES), N_QUINTILES - 1) <= b]
                tr = agg_stats([t["r"] for t in sel if t["half"] == "train"])
                if not tr or tr["n"] < MIN_TRAIN_RETAIN * n_train_all:
                    continue
                if best is None or tr["e"] > best[2]["e"]:
                    best = (a, b, tr, sel)

        print(f"\n  baseline (all quintiles)   train E={base['train']['e']:+.3f} "
              f"(n={base['train']['n']})   oos E={base['oos']['e']:+.3f} (n={base['oos']['n']})")
        if best is None:
            print("  no contiguous band clears the 40% train-retention floor.")
            continue
        a, b, tr_sel, sel = best
        oo_sel = agg_stats([t["r"] for t in sel if t["half"] == "oos"])
        thr_lo, thr_hi = edges[a], edges[b + 1]
        print(f"  TRAIN-CHOSEN filter        keep ATR pctile in [{thr_lo:.1f}, {thr_hi:.1f}] "
              f"(Q{a + 1}-Q{b + 1})   <- selected on the train half only, then frozen")
        print(f"  filtered                   train E={tr_sel['e']:+.3f} (n={tr_sel['n']})   "
              f"oos E={(oo_sel['e'] if oo_sel else float('nan')):+.3f} "
              f"(n={oo_sel['n'] if oo_sel else 0})")
        d_tr = tr_sel["e"] - base["train"]["e"]
        d_oo = (oo_sel["e"] - base["oos"]["e"]) if oo_sel else float("nan")
        gate = (oo_sel and tr_sel["e"] > 0 and oo_sel["e"] > 0)
        print(f"  delta vs baseline          train {d_tr:+.3f}R   oos {d_oo:+.3f}R"
              f"   -> {'HELPS OOS' if d_oo > 0 else 'HURTS OOS'}"
              f", {'passes' if gate else 'FAILS'} both-halves gate")

        # --- robustness: per-symbol raw ATR threshold from the train half ---
        # Same rule, different regime encoding: instead of a cross-era rolling
        # rank, freeze each symbol's own train-half median ATR% and split on it.
        # If the quintile result is real it should not vanish here.
        med = {}
        for sym in SYMBOLS:
            vals = [t["raw"] for t in mine if t["sym"] == sym and t["half"] == "train"
                    and t["raw"] is not None]
            if vals:
                med[sym] = float(np.median(vals))
        lo_side, hi_side = {}, {}
        for h in ("train", "oos"):
            lo_side[h] = agg_stats([t["r"] for t in mine if t["half"] == h
                                    and t["raw"] is not None and t["sym"] in med
                                    and t["raw"] <= med[t["sym"]]])
            hi_side[h] = agg_stats([t["r"] for t in mine if t["half"] == h
                                    and t["raw"] is not None and t["sym"] in med
                                    and t["raw"] > med[t["sym"]]])
        pick = "low" if (lo_side["train"] and hi_side["train"]
                         and lo_side["train"]["e"] > hi_side["train"]["e"]) else "high"
        chosen = lo_side if pick == "low" else hi_side
        print(f"  robustness (per-symbol train-median ATR% split, train prefers "
              f"{pick.upper()} vol):")
        print(f"    low-vol  half   train E={lo_side['train']['e']:+.3f} "
              f"(n={lo_side['train']['n']})   oos E={lo_side['oos']['e']:+.3f} "
              f"(n={lo_side['oos']['n']})")
        print(f"    high-vol half   train E={hi_side['train']['e']:+.3f} "
              f"(n={hi_side['train']['n']})   oos E={hi_side['oos']['e']:+.3f} "
              f"(n={hi_side['oos']['n']})")
        print(f"    frozen pick     oos E={chosen['oos']['e']:+.3f} vs baseline "
              f"{base['oos']['e']:+.3f}  -> "
              f"{'HELPS' if chosen['oos']['e'] > base['oos']['e'] else 'HURTS'}")

        results.append({
            "part": "B", "name": name, "baseline": base,
            "quintiles": buckets,
            "train_chosen_band": [int(a), int(b)],
            "filtered": {"train": tr_sel, "oos": oo_sel},
            "robustness": {"pick": pick, "low": lo_side, "high": hi_side},
            "both_positive": bool(gate),
        })
    return results


# ==========================================================================
# PART C -- diagnostics that decide how to read A and B
# ==========================================================================

def _cell_c1(job):
    """Measure the realised COST DRAG in R per trade at each timeframe.

    This is the whole premise of Part A. Cost in dollars is
    (2 x commission x lots) + (spread x lots x mult), and lots is
    risk_usd / (stop_distance x mult) -- so cost measured in R is inversely
    proportional to the stop distance. A weekly stop is several times a daily
    stop, so the drag per trade MUST fall. The question this answers is not
    whether it falls but by how much, i.e. the maximum expectancy the higher
    timeframe could ever have bought.
    """
    from metatraderbot.instruments import get_specs
    name, tf, sym = job
    cls, params = STRATS[name]
    try:
        d = daily_bars(sym)
        if d is None:
            return None
        cfg = make_cfg(sym, params, SURVIVOR_TP)
        specs = get_specs(cfg)
        mult, spread, comm = specs["contract_mult"], specs["spread_price"], specs["commission_per_lot"]
        trades = run_bt(cls, cfg, to_tf(d, tf))
        if not trades:
            return None
        cost_r = [((comm * t.lots * 2) + (spread * t.lots * mult)) / RISK_USD for t in trades]
        sl_dist = [abs(t.entry_price - t.stop_loss) / t.entry_price for t in trades]
        return {"name": name, "tf": tf, "cost_r": cost_r, "sl_pct": sl_dist}
    except Exception:
        return None


def boot_delta(base_by_sym: dict, filt_by_sym: dict, n_boot: int = 5000, seed: int = 0):
    """Bootstrap the filtered-minus-baseline expectancy delta by RESAMPLING SYMBOLS.

    Resampling individual trades would assume they are independent. They are
    not: the basket's 17 pairs share currency legs and are worth roughly 4.9
    independent bets (scripts/correlation_audit.py), and trades within one
    symbol overlap in time. Resampling whole symbols keeps both kinds of
    clustering intact and is the conservative choice.
    """
    rng = np.random.default_rng(seed)
    syms = sorted(set(base_by_sym) & set(filt_by_sym))
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(len(syms), size=len(syms), replace=True)
        b = [r for i in pick for r in base_by_sym[syms[i]]]
        f = [r for i in pick for r in filt_by_sym[syms[i]]]
        if b and f:
            deltas.append(np.mean(f) - np.mean(b))
    if not deltas:
        return None
    deltas = np.array(deltas)
    return {"mean": float(deltas.mean()),
            "lo": float(np.percentile(deltas, 2.5)),
            "hi": float(np.percentile(deltas, 97.5)),
            "p_gt0": float((deltas > 0).mean())}


def part_c(workers: int) -> list[dict]:
    print("\n" + "=" * 118)
    print("PART C -- DIAGNOSTICS")
    print("=" * 118, flush=True)

    # --- C1: does the cost-drag mechanism behind Part A actually exist? ----
    jobs = [(n, tf, s) for n in STRATS for tf in TIMEFRAMES for s in SYMBOLS]
    cells = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in ex.map(_cell_c1, jobs, chunksize=8):
            if c:
                cells.append(c)

    print("\nC1. Realised cost drag per trade, by timeframe "
          "(the stated reason to expect higher TFs to win)")
    print(f"{'strategy':<11}{'tf':>4}{'med stop (% px)':>17}{'mean cost (R)':>15}"
          f"{'median cost (R)':>17}{'vs 1d':>10}")
    print("-" * 118)
    c1 = []
    for name in STRATS:
        base_cost = None
        for tf in TIMEFRAMES:
            mine = [c for c in cells if c["name"] == name and c["tf"] == tf]
            if not mine:
                continue
            costs = [x for c in mine for x in c["cost_r"]]
            slp = [x for c in mine for x in c["sl_pct"]]
            mc = float(np.mean(costs))
            if tf == "1d":
                base_cost = mc
            rel = f"{mc / base_cost:.2f}x" if base_cost else "-"
            print(f"{name:<11}{tf:>4}{np.median(slp) * 100:>16.3f}%{mc:>15.4f}"
                  f"{np.median(costs):>17.4f}{rel:>10}")
            c1.append({"name": name, "tf": tf, "mean_cost_r": mc,
                       "median_stop_pct": float(np.median(slp))})
        print()

    # --- C2: does the Part B filter survive clustered resampling? ---------
    print("C2. Is the Part B volatility filter distinguishable from noise?")
    print("    (5000 bootstrap replications, resampling the 17 SYMBOLS with replacement)")
    print("-" * 118)
    jobs_b = [(n, s) for n in STRATS for s in SYMBOLS]
    trades = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(_cell_b, jobs_b, chunksize=4):
            if rows:
                trades.extend(rows)

    c2 = []
    for name in STRATS:
        mine = [t for t in trades if t["name"] == name and t["rank"] is not None]
        # The band the TRAIN half selected in Part B was Q3-Q5 for both.
        lo, hi = 0.4, 1.0
        for half in ("train", "oos"):
            base_by, filt_by = {}, {}
            for t in mine:
                if t["half"] != half:
                    continue
                base_by.setdefault(t["sym"], []).append(t["r"])
                if lo <= t["rank"] <= hi:
                    filt_by.setdefault(t["sym"], []).append(t["r"])
            bs = boot_delta(base_by, filt_by)
            if not bs:
                continue
            verdict = ("significant" if bs["lo"] > 0 else
                       "NOT distinguishable from zero")
            print(f"  {name:<11}{half:<6} delta E = {bs['mean']:+.3f}R   "
                  f"95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}]   "
                  f"P(delta>0) = {bs['p_gt0']:.2f}   -> {verdict}")
            c2.append({"name": name, "half": half, **bs})
        print()
    return [{"part": "C", "cost_drag": c1, "bootstrap": c2}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C", "both", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="backtest/htf_and_vol_regime.json")
    args = ap.parse_args()
    workers = min(args.workers, os.cpu_count() or 4)

    results = []
    if args.part in ("A", "both", "all"):
        results += part_a(workers)
    if args.part in ("B", "both", "all"):
        results += part_b(workers)
    if args.part in ("C", "all"):
        results += part_c(workers)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {len(results)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
