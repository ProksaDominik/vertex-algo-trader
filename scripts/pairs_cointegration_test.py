"""Market-neutral statistical arbitrage (cointegration pairs trading) on the FX basket.

HYPOTHESIS
----------
A spread between two cointegrated FX pairs mean-reverts more reliably than either
leg's own price. If true this should produce the profile an FTMO challenge needs:
high win rate (>65%), shallow drawdown, fast resolution (median hold < 5 days).
The prior correlation audit found mean |corr| 0.337, max 0.879 (NZDJPY/AUDJPY
+0.879, AUDUSD/NZDUSD +0.853), so the raw material exists.

WHAT THIS DOES DIFFERENTLY FROM scripts/pairs_stat_arb.py
---------------------------------------------------------
1. Uses the 10.5-year `*_15m_long.parquet` files, not the 2024-only ones.
2. Uses a proper cointegration spread (rolling OLS hedge ratio on log prices),
   not a raw price ratio. A ratio implicitly assumes beta == 1.
3. Pair SELECTION happens on the TRAIN half only (Engle-Granger ADF on the
   train residual), then the selection is FROZEN and tested on OOS. Picking
   pairs on full history is itself a form of fitting.
4. All 190 candidate pairs are screened, not a hand-picked list -- a hand-picked
   list carries the author's knowledge of the full sample.

MECHANICS
---------
  logA, logB          natural logs of close
  rolling window L:   beta_t = cov(logA,logB)/var(logB), alpha_t = mA - beta*mB
                      resid_t = logA_t - alpha_t - beta_t*logB_t
                      sigma_t = sqrt(varA - beta_t^2 * varB)     (OLS resid sd)
                      z_t     = resid_t / sigma_t
  Everything is TRAILING -- no lookahead in the signal.
  z >= +entry_z  ->  SHORT A / LONG  B (beta-weighted)
  z <= -entry_z  ->  LONG  A / SHORT B
  exit on |z| <= exit_z (win side), |z| >= stop_z (loss side), or max_hold bars.
  Signal on bar close, FILL AT NEXT BAR OPEN, both legs.

SIZING / R CONVENTION
---------------------
Position scaled so a move from entry_z to stop_z equals RISK_USD, i.e. 1R is the
stop distance -- directly comparable to every other R number in this repo.
  K = RISK_USD / ((stop_z - entry_z) * sigma_t)
  lots_A = K / (cm_A * P_A)      lots_B = K * beta / (cm_B * P_B)

COSTS (the whole point)
-----------------------
Per leg, mirroring metatraderbot.backtest exactly:
    cost_leg = commission_per_lot * lots * 2 + spread_price * lots * contract_mult
Charged on BOTH legs. A pairs trade therefore pays roughly double a single trade.
`--slip-mult` re-runs with the measured live-slippage multiple applied to spread.

RESULT: HYPOTHESIS REJECTED
---------------------------
Run 2026-08-04 over 2016-01..2026-08, 19 symbols, 171 candidate pairs, 48 configs
per timeframe, on 1h / 4h / 1d. NOT ONE configuration is positive in both halves
at the default 20-day max hold, on any timeframe. Three failure modes, in order
of how badly each one kills the idea:

1. REVERSION IS FAR TOO SLOW. Of the 171 candidate pairs, ~18% are cointegrated
   at the EG 5% level on train -- but the median mean-reversion half-life of
   those is 82 days (min 42, p25 74, p75 87). ZERO pairs have a half-life under
   30 days, let alone the <5 days an FTMO 10-day sprint needs. FX majors do
   co-move, but the spread wanders for months before it closes. This alone
   disqualifies the approach for a 10-day challenge regardless of profitability.

2. THERE IS NO GROSS EDGE TO PAY THE COSTS WITH. Pre-cost OOS expectancy across
   the whole 4h sweep sits between -0.02R and +0.17R, and is under +0.05R for
   40 of 48 configs. Costs run 0.05-0.16R per trade (both legs, spread +
   commission). Costs are typically 2-10x the gross edge; on 1h they are 505%
   of it. The spread reverts, but it reverts by less than the round trip costs.

3. THE HIGH-WIN-RATE PROFILE IS REAL AND STILL LOSES MONEY. This is the trap.
   L=162(40d) entry=1.5 exit=1.0 gives OOS win rate 66.9% with a 2.17-day median
   hold -- exactly the profile the challenge wants -- and E_net = -0.047R in
   BOTH halves (gross +0.003R, cost 0.051R). Win rate 55-67% is easy here; it is
   bought with a 1:2.3 reward:risk that the costs then tip negative. A high win
   rate is not an edge.

Robustness of the few apparent survivors:
   max_hold 10d ->  0/48 survive     max_hold 20d ->  0/48
   max_hold 40d ->  3/48 survive     max_hold 40d + 3x slippage -> 0/48
The survivor set only appears at a 40-day hold, has 8-11 day median holds (too
slow for the challenge anyway), and is wiped out by 3x slippage -- well inside
the 2-12x live slippage measured on this account. It is max-hold-horizon noise,
not an edge.

Do not tune this further. If the spread needs 82 days to revert, no z-threshold
fixes that.

Usage:
    python scripts/pairs_cointegration_test.py --tf 4h
    python scripts/pairs_cointegration_test.py --tf 1h --slip-mult 3
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.instruments import PAIR_SPECS  # noqa: E402

RISK_USD = 100.0
TRAIN_FRAC = 0.70

BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "NZDJPY", "AUDJPY", "CHFJPY", "GBPCHF", "EURCHF",
    "AUDCHF", "EURAUD", "AUDCAD", "EURGBP", "XAUUSD", "US500",
]

# Nominal bars/day per TF. The EFFECTIVE figure (weekends and the dropped
# Sunday stub removed) is measured from the panel at runtime -- using the
# nominal one would make every "days" number ~40% too small.
BARS_PER_DAY = {"1h": 24.0, "2h": 12.0, "4h": 6.0, "1d": 1.0}

# Engle-Granger (residual-based) ADF critical values, 1 regressor, constant,
# large sample (MacKinnon 1991). More negative than the plain ADF values
# because the residual is itself estimated.
EG_CRIT = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(sym: str, tf: str) -> pd.DataFrame | None:
    """10.5y 15m bars resampled to `tf`, Sunday stub bars dropped."""
    f = ROOT / "data" / f"{sym.lower()}_15m_long.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index, utc=True)
    out = (df.resample(tf)
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna())
    out = out[out.index.dayofweek != 6]          # Sunday bar is a ~3h stub
    return out if len(out) > 2000 else None


def build_panel(tf: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (opens, closes) aligned on the common index across the basket."""
    opens, closes, syms = {}, {}, []
    for s in BASKET:
        b = load(s, tf)
        if b is None or s not in PAIR_SPECS:
            continue
        opens[s], closes[s] = b["open"], b["close"]
        syms.append(s)
    o = pd.DataFrame(opens).dropna()
    c = pd.DataFrame(closes).dropna()
    idx = o.index.intersection(c.index)
    return o.loc[idx], c.loc[idx], syms


# --------------------------------------------------------------------------
# cointegration statistics (statsmodels is not installed in this venv)
# --------------------------------------------------------------------------
def adf_tstat(x: np.ndarray, lags: int = 1) -> float:
    """ADF t-statistic, constant / no trend. Returns +inf if degenerate."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    n = len(dx) - lags
    if n < 100:
        return np.inf
    y = dx[lags:]
    cols = [x[lags:-1], np.ones(n)]
    for k in range(1, lags + 1):
        cols.append(dx[lags - k:-k])
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    resid = y - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return np.inf
    s2 = resid @ resid / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.inf
    se = np.sqrt(s2 * xtx_inv[0, 0])
    return float(beta[0] / se) if se > 0 else np.inf


def half_life_bars(x: np.ndarray) -> float:
    """AR(1) mean-reversion half-life in bars. inf if not mean-reverting."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    lag = x[:-1]
    X = np.column_stack([lag, np.ones(len(lag))])
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    b = beta[0]
    if b >= 0:
        return np.inf
    return float(-np.log(2.0) / b)


def screen_pairs(log_close_train: pd.DataFrame, syms: list[str],
                 bpd: float, top_n: int, max_hl_days: float) -> tuple[list, list]:
    """Engle-Granger screen on the TRAIN half only. Returns the frozen list."""
    rows = []
    for a, b in itertools.combinations(syms, 2):
        la = log_close_train[a].to_numpy()
        lb = log_close_train[b].to_numpy()
        X = np.column_stack([lb, np.ones(len(lb))])
        coef, *_ = np.linalg.lstsq(X, la, rcond=None)
        beta, alpha = float(coef[0]), float(coef[1])
        resid = la - alpha - beta * lb
        t = adf_tstat(resid, lags=1)
        hl = half_life_bars(resid)
        rows.append({"a": a, "b": b, "beta": beta, "adf_t": t,
                     "hl_days": hl / bpd,
                     "corr": float(np.corrcoef(la, lb)[0, 1])})
    rows.sort(key=lambda r: r["adf_t"])
    # Frozen selection rule, applied to TRAIN statistics only.
    sel = [r for r in rows
           if r["adf_t"] < EG_CRIT["5%"] and 0 < r["hl_days"] <= max_hl_days]
    return rows, sel[:top_n]


# --------------------------------------------------------------------------
# rolling spread / z-score (fully trailing)
# --------------------------------------------------------------------------
def rolling_z(la: pd.Series, lb: pd.Series, L: int):
    """Trailing rolling-OLS residual z-score plus the rolling beta and sigma."""
    mA = la.rolling(L).mean()
    mB = lb.rolling(L).mean()
    vA = la.rolling(L).var(ddof=1)
    vB = lb.rolling(L).var(ddof=1)
    cAB = la.rolling(L).cov(lb, ddof=1)
    beta = cAB / vB
    resid = la - (mA - beta * mB) - beta * lb
    var_res = (vA - beta * cAB).clip(lower=0.0)          # = vA - beta^2 * vB
    sigma = np.sqrt(var_res * (L - 1) / max(L - 2, 1))
    z = resid / sigma
    return (z.to_numpy(), beta.to_numpy(), sigma.to_numpy())


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------
def backtest(sym_a: str, sym_b: str,
             open_a: np.ndarray, open_b: np.ndarray,
             ts: pd.DatetimeIndex,
             z: np.ndarray, beta: np.ndarray, sigma: np.ndarray,
             entry_z: float, exit_z: float, stop_z: float,
             max_hold: int, slip_mult: float, end_i: int) -> dict:
    """Event-driven: jump straight to the next entry / exit index.

    `end_i` is the last bar index this segment may touch -- without it a trade
    opened at the end of the train half would be closed with OOS prices.
    """
    spec_a, spec_b = PAIR_SPECS[sym_a], PAIR_SPECS[sym_b]
    cm_a, cm_b = spec_a["contract_mult"], spec_b["contract_mult"]
    sp_a = spec_a["spread_price"] * slip_mult
    sp_b = spec_b["spread_price"] * slip_mult
    com_a, com_b = spec_a["commission_per_lot"], spec_b["commission_per_lot"]

    n = min(len(z), end_i + 1)
    az = np.abs(z)
    ok = np.isfinite(z) & np.isfinite(beta) & (sigma > 0)
    ok = ok[:n]
    az = az[:n]
    zz = z[:n]
    entry_idx = np.flatnonzero(ok & (az >= entry_z))
    # SIGNED exit conditions. Using |z| <= exit_z would make exit_z = 0.0
    # untriggerable (z is never exactly zero) and would also fail to close a
    # spread that blew straight through the mean to the other side. What a
    # pairs trader actually means by "exit at the mean" is a zero CROSSING.
    #   entered short-A (z > 0): exit when z <= +exit_z  or z >= +stop_z
    #   entered long-A  (z < 0): exit when z >= -exit_z  or z <= -stop_z
    exit_hi = np.flatnonzero(ok & ((zz <= exit_z) | (zz >= stop_z)))
    exit_lo = np.flatnonzero(ok & ((zz >= -exit_z) | (zz <= -stop_z)))

    rs, costs_r, holds, reasons = [], [], [], []
    cursor = 0
    zwidth = stop_z - entry_z
    while True:
        p = np.searchsorted(entry_idx, cursor, side="left")
        if p >= len(entry_idx):
            break
        i = int(entry_idx[p])
        fi = i + 1                                   # fill at NEXT bar open
        if fi >= n - 1:
            break

        # exit search starts strictly after the fill bar, on the arm that
        # matches the entry side
        short_a = zz[i] > 0
        exit_idx = exit_hi if short_a else exit_lo
        q = np.searchsorted(exit_idx, fi + 1, side="left")
        if q < len(exit_idx):
            j = int(exit_idx[q])
            reverted = (zz[j] <= exit_z) if short_a else (zz[j] >= -exit_z)
            reason = "revert" if reverted else "stop"
        else:
            j = n - 2
            reason = "eod"
        if j - fi >= max_hold:
            j = fi + max_hold
            reason = "timeout"
        fj = min(j + 1, n - 1)
        if fj <= fi:
            cursor = i + 1
            continue

        d = -1.0 if short_a else 1.0                 # +1 = long A / short B
        b_t = float(beta[i])
        sg = float(sigma[i])
        if not np.isfinite(b_t) or sg <= 0 or b_t == 0:
            cursor = i + 1
            continue

        K = RISK_USD / (zwidth * sg)                 # $ per 1.0 of log-resid
        pa, pb = float(open_a[fi]), float(open_b[fi])
        lots_a = K / (cm_a * pa)
        lots_b = abs(K * b_t) / (cm_b * pb)
        if not np.isfinite(lots_a) or not np.isfinite(lots_b):
            cursor = i + 1
            continue

        # P&L: long A leg + (sign-flipped, beta-weighted) B leg
        sgn_b = -np.sign(b_t)
        pnl_a = d * (float(open_a[fj]) - pa) * lots_a * cm_a
        pnl_b = d * sgn_b * (float(open_b[fj]) - pb) * lots_b * cm_b
        cost = (com_a * lots_a * 2 + sp_a * lots_a * cm_a
                + com_b * lots_b * 2 + sp_b * lots_b * cm_b)
        pnl = pnl_a + pnl_b - cost

        rs.append(pnl / RISK_USD)
        costs_r.append(cost / RISK_USD)
        holds.append((ts[fj] - ts[fi]).total_seconds() / 86400.0)
        reasons.append(reason)
        cursor = fj + 1

    return {"rs": np.array(rs), "costs_r": np.array(costs_r),
            "holds": np.array(holds), "reasons": reasons}


def stats(res: dict, span_days: float) -> dict:
    rs = res["rs"]
    n = len(rs)
    if n == 0:
        return {"n": 0, "wr": 0.0, "e": 0.0, "gross": 0.0, "cost": 0.0,
                "hold": 0.0, "tpd": 0.0, "pf": 0.0, "maxdd": 0.0, "revert": 0.0}
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    eq = np.cumsum(rs)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if n else 0.0
    rr = res["reasons"]
    return {
        # fraction of trades that actually MEAN-REVERTED rather than timing out
        # or stopping out -- the hypothesis stands or falls on this number.
        "revert": sum(1 for x in rr if x == "revert") / n,
        "n": n,
        "wr": float(len(wins) / n),
        "e": float(rs.mean()),
        "gross": float(rs.mean() + res["costs_r"].mean()),
        "cost": float(res["costs_r"].mean()),
        "hold": float(np.median(res["holds"])),
        "tpd": n / span_days if span_days > 0 else 0.0,
        "pf": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "maxdd": dd,
    }


def pool(results: list[dict], span_days: float) -> dict:
    if not results:
        return stats({"rs": np.array([]), "costs_r": np.array([]),
                      "holds": np.array([]), "reasons": []}, span_days)
    merged = {
        "rs": np.concatenate([r["rs"] for r in results]),
        "costs_r": np.concatenate([r["costs_r"] for r in results]),
        "holds": np.concatenate([r["holds"] for r in results]),
        "reasons": [x for r in results for x in r["reasons"]],
    }
    return stats(merged, span_days)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h", choices=list(BARS_PER_DAY))
    ap.add_argument("--top-n", type=int, default=12, help="pairs kept from the TRAIN screen")
    ap.add_argument("--max-hl-days", type=float, default=30.0)
    ap.add_argument("--slip-mult", type=float, default=1.0,
                    help="multiply modelled spread (live slippage was 2-12x)")
    ap.add_argument("--max-hold-days", type=float, default=20.0)
    args = ap.parse_args()

    tf = args.tf

    print("=" * 108)
    print(f"COINTEGRATION PAIRS STAT-ARB  tf={tf}  slip_mult={args.slip_mult}x  "
          f"max_hold={args.max_hold_days:.0f} calendar days  "
          f"1R = (stop_z-entry_z) spread sigma")
    print("=" * 108)

    opens, closes, syms = build_panel(tf)
    n = len(closes)
    split = int(n * TRAIN_FRAC)
    # effective bars per CALENDAR day (weekends + Sunday stub already removed)
    bpd = n / ((closes.index[-1] - closes.index[0]).total_seconds() / 86400.0)
    max_hold = int(round(args.max_hold_days * bpd))
    lc = np.log(closes)
    print(f"panel: {len(syms)} symbols, {n} aligned {tf} bars "
          f"({bpd:.2f} bars/calendar-day effective) "
          f"{closes.index[0].date()} -> {closes.index[-1].date()}   "
          f"train {closes.index[0].date()}..{closes.index[split - 1].date()}  "
          f"OOS {closes.index[split].date()}..{closes.index[-1].date()}")

    span_tr = (closes.index[split - 1] - closes.index[0]).total_seconds() / 86400
    span_oos = (closes.index[-1] - closes.index[split]).total_seconds() / 86400

    # ---- STEP 1: pair selection, TRAIN HALF ONLY --------------------------
    all_rows, sel = screen_pairs(lc.iloc[:split], syms, bpd, args.top_n, args.max_hl_days)
    n_coint = sum(1 for r in all_rows if r["adf_t"] < EG_CRIT["5%"])
    print(f"\nSTEP 1 -- Engle-Granger screen on TRAIN ONLY "
          f"({len(all_rows)} candidate pairs, EG 5% crit {EG_CRIT['5%']})")
    print(f"  pairs cointegrated at 5% on train: {n_coint}/{len(all_rows)} "
          f"({n_coint / len(all_rows):.1%})   selected (also half-life "
          f"<= {args.max_hl_days:.0f}d): {len(sel)}")
    # Reversion SPEED is the make-or-break number for a 10-day challenge.
    hls = np.array([r["hl_days"] for r in all_rows
                    if r["adf_t"] < EG_CRIT["5%"] and np.isfinite(r["hl_days"])])
    if len(hls):
        print(f"  half-life (days) of the {len(hls)} train-cointegrated pairs: "
              f"min {hls.min():.0f}  p25 {np.percentile(hls, 25):.0f}  "
              f"median {np.median(hls):.0f}  p75 {np.percentile(hls, 75):.0f}  "
              f"max {hls.max():.0f}   <=10d: {(hls <= 10).sum()}  "
              f"<=30d: {(hls <= 30).sum()}")
    print(f"  {'pair':<16}{'beta':>9}{'corr':>8}{'ADF_t':>9}{'HL_days':>10}"
          f"{'  |  OOS ADF_t (info only, NOT used to select)':<0}")
    for r in sel:
        la_o = lc[r["a"]].iloc[split:].to_numpy()
        lb_o = lc[r["b"]].iloc[split:].to_numpy()
        X = np.column_stack([lb_o, np.ones(len(lb_o))])
        c_o, *_ = np.linalg.lstsq(X, la_o, rcond=None)
        t_o = adf_tstat(la_o - c_o[1] - c_o[0] * lb_o, lags=1)
        print(f"  {r['a'] + '/' + r['b']:<16}{r['beta']:>9.3f}{r['corr']:>8.3f}"
              f"{r['adf_t']:>9.2f}{r['hl_days']:>10.1f}     {t_o:>8.2f}")
    if not sel:
        print("  NOTHING SELECTED -- no train-cointegrated pair with a tradeable half-life.")
        return

    # ---- STEP 2: parameter sweep on the FROZEN pair list ------------------
    lookbacks = [max(20, int(x * bpd)) for x in (5, 10, 20, 40)]   # calendar-day windows
    entries = [1.5, 2.0, 2.5, 3.0]
    exits = [0.0, 0.5, 1.0]
    stop_extra = 2.0

    print(f"\nSTEP 2 -- sweep on the FROZEN pair list "
          f"({len(lookbacks)}x{len(entries)}x{len(exits)} = "
          f"{len(lookbacks) * len(entries) * len(exits)} configs, "
          f"stop_z = entry_z + {stop_extra})")
    print("=" * 108)
    hdr = (f"{'L(bars/days)':<14}{'entry':>6}{'exit':>6}{'stop':>6} | "
           f"{'n_tr':>6}{'wr':>7}{'E_tr':>8}{'PF':>6} | "
           f"{'n_oos':>6}{'wr':>7}{'E_oos':>8}{'PF':>6} | "
           f"{'t/day':>6}{'hold_d':>7}{'cost_R':>7}{'gross_R':>8}{'rev%':>6}  gate")
    print(hdr)
    print("-" * len(hdr))

    # precompute z per (pair, lookback) once for the whole train+OOS series
    zcache: dict[tuple[str, str, int], tuple] = {}
    for r in sel:
        for L in lookbacks:
            zcache[(r["a"], r["b"], L)] = rolling_z(lc[r["a"]], lc[r["b"]], L)

    ts = closes.index
    survivors, table = [], []
    for L, ez, xz in itertools.product(lookbacks, entries, exits):
        sz = ez + stop_extra
        tr_res, oos_res, per_pair = [], [], []
        for r in sel:
            a, b = r["a"], r["b"]
            z, bt, sg = zcache[(a, b, L)]
            oa, ob = opens[a].to_numpy(), opens[b].to_numpy()
            # TRAIN: only bars < split. OOS: warm up the rolling window using
            # train data but only take trades whose ENTRY is at/after split.
            zt = z.copy(); zt[split:] = np.nan
            zo = z.copy(); zo[:split] = np.nan
            rt = backtest(a, b, oa, ob, ts, zt, bt, sg, ez, xz, sz,
                          max_hold, args.slip_mult, end_i=split - 1)
            ro = backtest(a, b, oa, ob, ts, zo, bt, sg, ez, xz, sz,
                          max_hold, args.slip_mult, end_i=len(ts) - 1)
            tr_res.append(rt); oos_res.append(ro)
            per_pair.append((f"{a}/{b}", stats(rt, span_tr), stats(ro, span_oos)))
        st = pool(tr_res, span_tr)
        so = pool(oos_res, span_oos)
        gate = "PASS" if (st["e"] > 0 and so["e"] > 0 and st["n"] >= 30 and so["n"] >= 30) else ""
        if gate:
            survivors.append((L, ez, xz, sz, st, so, per_pair))
        table.append((L, ez, xz, sz, st, so, per_pair))
        pf_t = f"{st['pf']:.2f}" if np.isfinite(st["pf"]) else "inf"
        pf_o = f"{so['pf']:.2f}" if np.isfinite(so["pf"]) else "inf"
        print(f"{L:>4}/{L / bpd:>4.0f}d      {ez:>6.1f}{xz:>6.1f}{sz:>6.1f} | "
              f"{st['n']:>6}{st['wr']:>7.1%}{st['e']:>+8.3f}{pf_t:>6} | "
              f"{so['n']:>6}{so['wr']:>7.1%}{so['e']:>+8.3f}{pf_o:>6} | "
              f"{so['tpd']:>6.2f}{so['hold']:>7.2f}{so['cost']:>7.3f}"
              f"{so['gross']:>+8.3f}{so['revert']:>6.0%}  {gate}", flush=True)

    # ---- STEP 3: verdict --------------------------------------------------
    print("\n" + "=" * 108)
    print("STEP 3 -- BOTH-HALVES GATE (pooled over the train-selected pairs; "
          "min 30 trades per half)")
    print("=" * 108)
    if not survivors:
        print("  NO CONFIGURATION IS POSITIVE IN BOTH HALVES. Stat-arb does not survive "
              "costs on this basket.")
    else:
        print(f"  {len(survivors)}/{len(table)} configurations positive in both halves:")
        for L, ez, xz, sz, st, so, per_pair in sorted(survivors, key=lambda x: -x[5]["e"]):
            print(f"    L={L}({L / bpd:.0f}d) entry={ez} exit={xz} stop={sz}  "
                  f"train E={st['e']:+.3f}R n={st['n']}  |  "
                  f"OOS E={so['e']:+.3f}R wr={so['wr']:.1%} n={so['n']} "
                  f"hold={so['hold']:.2f}d t/day={so['tpd']:.2f} "
                  f"reverted={so['revert']:.0%}")

    # cost anatomy on the best-gross config, to show where the money goes
    best = max(table, key=lambda x: x[5]["gross"])
    L, ez, xz, sz, st, so, per_pair = best
    print("\n" + "=" * 108)
    print(f"COST ANATOMY -- config with the highest OOS GROSS edge "
          f"(L={L}, entry={ez}, exit={xz}, stop={sz})")
    print("=" * 108)
    print(f"  OOS gross E (before costs) = {so['gross']:+.3f}R")
    print(f"  OOS cost per trade         = {so['cost']:.3f}R   "
          f"({so['cost'] / max(abs(so['gross']), 1e-9):.0%} of gross)")
    print(f"  OOS net E                  = {so['e']:+.3f}R")
    print(f"  OOS win rate {so['wr']:.1%}   median hold {so['hold']:.2f} days   "
          f"trades/day {so['tpd']:.2f}   pooled maxDD {so['maxdd']:.1f}R")

    print("\n  per-pair breakdown at this config (train | OOS):")
    print(f"  {'pair':<16}{'n_tr':>6}{'wr_tr':>7}{'E_tr':>9}   "
          f"{'n_oos':>6}{'wr_oos':>8}{'E_oos':>9}{'hold_d':>8}{'cost_R':>8}  both+")
    for name, a_tr, a_oos in sorted(per_pair, key=lambda x: -x[2]["e"]):
        both = "YES" if a_tr["e"] > 0 and a_oos["e"] > 0 else ""
        print(f"  {name:<16}{a_tr['n']:>6}{a_tr['wr']:>7.1%}{a_tr['e']:>+9.3f}   "
              f"{a_oos['n']:>6}{a_oos['wr']:>8.1%}{a_oos['e']:>+9.3f}"
              f"{a_oos['hold']:>8.2f}{a_oos['cost']:>8.3f}  {both}")

    # best config by OOS net, for the FTMO profile check
    bn = max(table, key=lambda x: x[5]["e"])
    L, ez, xz, sz, st, so, _ = bn
    print("\n" + "=" * 108)
    print("FTMO PROFILE CHECK -- best config by OOS NET expectancy")
    print("=" * 108)
    print(f"  L={L}({L / bpd:.0f}d) entry={ez} exit={xz} stop={sz}")
    for label, need, got, ok in [
        ("win rate > 65%", "65.0%", f"{so['wr']:.1%}", so["wr"] > 0.65),
        ("median hold < 5d", "5.00d", f"{so['hold']:.2f}d", so["hold"] < 5),
        ("OOS E_net > 0", "+0.000R", f"{so['e']:+.3f}R", so["e"] > 0),
        ("train E_net > 0", "+0.000R", f"{st['e']:+.3f}R", st["e"] > 0),
    ]:
        print(f"  {label:<20} need {need:>8}   got {got:>8}   "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
