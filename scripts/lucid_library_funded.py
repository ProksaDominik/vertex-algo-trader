"""EVERY strategy in the bot's library under LucidPro's funded and evaluation rules
(owner, 2026-10-01: "test out all the strategies").

Library: lab_registry.FAMILIES (16 families, 37 parameter sets) + orb_trend.
Markets: NQ, ES, YM, GC, CL (10.5-year 15m proxies, futures costs); timeframes
15m and 1h; exits 1R / 2R (stop from the strategy, Lucid's flat rule); risk
$250 / $500 / $1,000 per trade (whole micro contracts, Lucid's 40-unit cap).

Stage 1 (slow, parallel, checkpointed): signals -> trades per cell, saved to
data/futures/library_trades/<cell>.parquet.
Stage 2 (fast): every cell x exit x risk -> Lucid-day P&L paths -> the funded
replay (first payout $500 then $2,500) and the evaluation replay (confirmed
rules), ranked on 2019-22 starts, judged on 2023-26 starts.

Usage:
  .venv/Scripts/python.exe scripts/lucid_library_funded.py --probe     # time one cell
  .venv/Scripts/python.exe scripts/lucid_library_funded.py --stage1    # all cells (hours of CPU, parallel)
  .venv/Scripts/python.exe scripts/lucid_library_funded.py --stage2    # replays + report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import lucid_pass_research as lp  # noqa: E402
from metatraderbot import futures, lucid_rules  # noqa: E402
from metatraderbot.lab_registry import FAMILIES, _grid, instantiate  # noqa: E402

OUT = ROOT / "data" / "futures" / "library_trades"
MARKETS = ("NQ", "ES", "YM", "GC", "CL", "6E", "6B", "6J")   # + forex futures for the FX-style families
TFS = {"15m": 15, "1h": 60}
WINDOW = 300
H = 120
EVAL_COST = 185 + 120 * (1 / 0.41 - 1)
EVAL_WEEKS = (1 / 0.41) * 1.5 / 5


def cells() -> list[tuple]:
    out = []
    fams = dict(FAMILIES)
    fams["orb_trend"] = {}
    for fam, spec in fams.items():
        for params in _grid(fam, spec):
            for tf in TFS:
                if fam == "orb_trend" and tf != "15m":
                    continue
                for m in MARKETS:
                    out.append((fam, params, tf, m))
    return out


def cell_name(fam, params, tf, m) -> str:
    p = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    return f"{fam}__{p or 'default'}__{tf}__{m}"


_BARS: dict = {}


def bars(m: str, tf: str) -> tuple[pd.DataFrame, list]:
    key = (m, tf)
    if key not in _BARS:
        proxy, _, inverted = lp.MARKETS[m]
        df = lp.fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet").tz_convert("UTC")
        if inverted:                                   # USDJPY -> 6J, exactly as lp.load_grid does
            df = pd.DataFrame({"open": 1 / df["open"], "high": 1 / df["low"], "low": 1 / df["high"],
                               "close": 1 / df["close"]}, index=df.index)
        if tf == "1h":
            df = df.resample("1h", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        mins = TFS[tf]
        states = [lucid_rules.session_state((t + pd.Timedelta(minutes=mins)).to_pydatetime()) for t in df.index]
        _BARS[key] = (df, states)
    return _BARS[key]


def run_cell(args) -> str:
    fam, params, tf, m = args
    path = OUT / f"{cell_name(fam, params, tf, m)}.parquet"
    if path.exists():
        return "cached"
    try:
        df, states = bars(m, tf)
        contract = lp.MARKETS[m][1]
        tick = futures.spec(contract)["tick_size"]
        strat = instantiate(fam, {fam: dict(params)}, 1.0, contract)
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        n = len(df)
        # Lucid trading day of every bar: the flat rule closes everything at the day's
        # LAST bar -- the index proxies stop at 16:15 ET and never show the 16:40
        # flatten window, so without this trades silently ran overnight (2026-10-01).
        et = df.index.tz_convert(lp.fh.ET)
        mins = np.asarray(et.hour * 60 + et.minute)
        lday = (np.asarray(et.tz_localize(None).normalize().values.astype("datetime64[D]"))
                + (mins >= 1080).astype("timedelta64[D]"))
        last_of_day = np.r_[lday[1:] != lday[:-1], True]
        signals = []
        for i in range(WINDOW, n - 1):
            if states[i] != lucid_rules.Session.OPEN or last_of_day[i]:
                continue                        # the entry would land in the next trading day
            w = df.iloc[i - WINDOW + 1: i + 1]
            try:
                specs = strat.on_bar(len(w) - 1, w)
            except Exception:
                continue
            for s in specs or []:
                d = getattr(s, "direction", None)
                is_long = getattr(d, "value", str(d)).lower() == "long"
                dist = abs(float(s.entry_price) - float(s.stop_loss))
                if dist >= 2 * tick:
                    signals.append((i, is_long, dist))
                break
        rows = []
        for tp_r in (1.0, 2.0):
            busy = -1
            for i, is_long, dist in signals:
                if i <= busy:
                    continue
                e = i + 1
                entry = o[e]
                sl = entry - dist if is_long else entry + dist
                tp = entry + dist * tp_r if is_long else entry - dist * tp_r
                exit_px, j, why = None, e, ""
                for j in range(e, n):
                    if (l[j] <= sl) if is_long else (h[j] >= sl):
                        exit_px, why = sl, "sl"
                        break
                    if (h[j] >= tp) if is_long else (l[j] <= tp):
                        exit_px, why = tp, "tp"
                        break
                    if states[j] in (lucid_rules.Session.FLATTEN, lucid_rules.Session.CLOSED) or last_of_day[j]:
                        exit_px, why = c[j], "flat"
                        break
                if exit_px is None:
                    break
                busy = j
                rows.append({"tp_r": tp_r, "side": 1 if is_long else -1, "entry_ts": df.index[e],
                             "exit_ts": df.index[j], "entry": float(entry), "exit": float(exit_px),
                             "dist": float(dist), "why": why})
        pd.DataFrame(rows, columns=["tp_r", "side", "entry_ts", "exit_ts", "entry", "exit", "dist", "why"]).to_parquet(path)
        return f"{len(rows)} trades"
    except Exception as e:  # noqa: BLE001
        return f"error {e!r}"


# --- stage 2 --------------------------------------------------------------------

def lucid_pos(ts_utc: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    et = ts_utc.tz_convert(lp.fh.ET)
    mins = np.asarray(et.hour * 60 + et.minute)
    base = np.asarray(et.tz_localize(None).normalize().values.astype("datetime64[D]"))
    day = base + (mins >= 1080).astype("timedelta64[D]")
    slot = np.where(mins >= 1080, (mins - 1080) // 15, (mins + 360) // 15)
    return day, slot


def day_paths(g: lp.Grid, t: pd.DataFrame, minutes: int, risk_usd: float):
    """Trades -> per-Lucid-day cumulative $ paths (close / worst / best) on g's grid."""
    n = len(g.dates)
    C, W, B = np.zeros((n, lp.K)), np.zeros((n, lp.K)), np.zeros((n, lp.K))
    if t.empty:
        return C, W, B
    contract = g.contract
    s = futures.spec(contract)
    pv, tick = s["point_value"], s["tick_size"]
    cost1 = 2 * tick * pv + 2 * s["commission"]
    d_in, s_in = lucid_pos(pd.DatetimeIndex(t["entry_ts"]))
    d_out, s_out = lucid_pos(pd.DatetimeIndex(t["exit_ts"]) + pd.Timedelta(minutes=minutes - 15))
    di = np.searchsorted(g.dates, d_in)
    kk = np.arange(lp.K)
    for r, (a, b_, sd, ent, ex, dist, i1, i2) in enumerate(zip(s_in, s_out, t["side"].to_numpy(), t["entry"].to_numpy(),
                                                          t["exit"].to_numpy(), t["dist"].to_numpy(), di,
                                                          np.searchsorted(g.dates, d_out))):
        b_ = min(int(b_), lp.K - 1)                         # a 1h bar ending at the flat time: last slot
        if i1 >= n or g.dates[i1] != d_in[r] or i2 != i1 or not (0 <= a <= b_ < lp.K):
            continue                                        # outside the grid / crossed a day (should not happen)
        qty = int(risk_usd // (dist * pv + 1e-9))
        qty = min(qty, 40 if contract.startswith("M") else 4)
        if qty < 1:
            continue
        adv = np.where(sd > 0, g.L[i1, a:b_ + 1], g.H[i1, a:b_ + 1])
        fav = np.where(sd > 0, g.H[i1, a:b_ + 1], g.L[i1, a:b_ + 1])
        cl = sd * (g.C[i1, a:b_ + 1] - ent) * pv * qty - cost1 * qty
        wo = sd * (adv - ent) * pv * qty - cost1 * qty
        be = sd * (fav - ent) * pv * qty - cost1 * qty
        final = sd * (ex - ent) * pv * qty - cost1 * qty
        cl[-1] = final
        wo[-1] = min(wo[-1], final)
        be[-1] = max(min(be[-1], final if final > 0 else be[-1]), final)
        wo = np.minimum(wo, cl)
        C[i1, a:b_ + 1] += cl
        W[i1, a:b_ + 1] += wo
        B[i1, a:b_ + 1] += be
        C[i1, b_ + 1:] += final
        W[i1, b_ + 1:] += final
        B[i1, b_ + 1:] += final
    return C, W, B


def stage2() -> int:
    G = {m: lp.load_grid(m) for m in MARKETS}
    rows = []
    t0 = time.time()
    files = sorted(OUT.glob("*.parquet"))
    for k, f in enumerate(files):
        fam, params, tf, m = f.stem.split("__")
        g = G[m]
        trades = pd.read_parquet(f)
        yrs = pd.to_datetime(g.dates).year
        keep = yrs >= 2019
        starts = np.arange(0, keep.sum() - H)
        years = np.asarray(yrs[keep][starts])
        for tp_r in (1.0, 2.0):
            t = trades[trades.tp_r == tp_r]
            for risk in (250.0, 500.0, 1000.0):
                C, W, B = day_paths(g, t, TFS[tf], risk)
                C, W, B = C[keep], W[keep], B[keep]
                fres = lp.replay_funded(C, W, B, lp.FundedRules(want=(500.0, 2500.0, 2500.0, 2500.0, 2500.0),
                                                                max_days=H), starts, pause_when_met=True)
                eres = lp.replay(C, W, lp.Rules(max_days=H, pass_on="close", mll_on="equity", dll_on="balance",
                                                lock=100.0, **lp.PLANS["LucidPro 50K"]), starts)
                row = {"family": fam, "params": params, "tf": tf, "market": m, "tp_r": tp_r, "risk": risk,
                       "trades": int(len(t))}
                for tag, lo, hi in (("A", 2019, 2022), ("B", 2023, 2026)):
                    sel = (years >= lo) & (years <= hi)
                    fr, er = fres[sel], eres[sel]
                    got = fr.payouts >= 1
                    s = lp.summarise_funded(fr)
                    row[f"{tag}_p10"] = (got & (fr.first_day <= 10)).mean()
                    row[f"{tag}_payout"] = got.mean()
                    row[f"{tag}_blown"] = ((fr.status == 2) & ~got).mean()
                    row[f"{tag}_per_week"] = (s["avg_paid_net"] - EVAL_COST) / (EVAL_WEEKS + s["avg_days_alive"] / 5)
                    row[f"{tag}_eval_pass"] = lp.summarise(er)["pass_resolved"]
                rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(files)} cells replayed ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data" / "futures" / "library_funded.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 40)
    f = lambda v: f"{v:,.3f}"                                              # noqa: E731
    cols = ["family", "params", "tf", "market", "tp_r", "risk", "trades", "A_p10", "B_p10", "B_payout", "B_blown",
            "B_per_week", "A_eval_pass", "B_eval_pass"]
    print(f"\n{len(res)} strategy versions replayed")
    print("\nFUNDED -- top 15 ranked on 2019-22 ($500 payout within 10 days), judged on 2023-26:")
    print(res.sort_values("A_p10", ascending=False).head(15)[cols].to_string(index=False, float_format=f))
    print("\nFUNDED -- top 10 by 2019-22 $/week, judged on 2023-26:")
    print(res.sort_values("A_per_week", ascending=False).head(10)[cols].to_string(index=False, float_format=f))
    print("\nEVALUATION -- top 10 ranked on 2019-22 pass rate, judged on 2023-26:")
    print(res.sort_values("A_eval_pass", ascending=False).head(10)[cols].to_string(index=False, float_format=f))
    print("\nFor reference -- funded race: ~36% $500+ payout within 10 days, ~63% blown, ~$280/week;"
          " evaluation race: ~41% pass.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.probe:
        t0 = time.time()
        print(run_cell(("donchian", {"period": 20}, "15m", "NQ")), f"{time.time() - t0:.0f}s")
        return 0
    if a.stage1:
        todo = cells()
        lp.fh.keep_awake()
        print(f"{len(todo)} cells, {a.workers} workers", flush=True)
        t0 = time.time()
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(run_cell, todo), 1):
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} done ({time.time() - t0:.0f}s) last: {r}", flush=True)
        return 0
    if a.stage2:
        return stage2()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
