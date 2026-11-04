"""Live prediction lab: score candidate predictors on trading days they have never seen (2026-10-03).

Owner, 2026-10-03: "we need a predicting strategy". No predictor beat the race
on history in a way that held on unseen years (docs + notes/Research log), so
the near-misses are FROZEN here, as defined today, and scored on every new
trading day from LAB_START. Nothing trades on them. A predictor becomes worth
switching to only after >= 60 resolved race days on live data with a win rate
clearly above the deployed race's on the same days (z >= 2).

Each predictor answers, for a Lucid trading day: race long (+1), race short
(-1) or skip (0), at its entry time. The outcome is the race's own: 2 NQ,
+75.5 pts (the $3,000 target) before -50 pts (the $2,000 limit), adverse first
inside a 15-minute bar, out 16:00 ET. Every feature uses only bars before the
entry -- the lab recomputes them after the close from NinjaTrader's real bars,
which is equivalent to recording them live and survives a sleeping laptop.

Writes data/prediction_lab.json and the Obsidian note notes/Prediction lab.md.

Usage:
    .venv/Scripts/python.exe scripts/run_prediction_lab.py            # update once
    .venv/Scripts/python.exe scripts/run_prediction_lab.py --loop     # keep it updated (launcher)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import lucid_orderflow_search as of  # noqa: E402
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402
from metatraderbot import lucid_rules  # noqa: E402
from metatraderbot.nt_engine import NinjaTraderEngine  # noqa: E402

LAB_START = "2026-10-05"            # first Lucid trading day scored (frozen 2026-10-03)
PROMOTE_N, PROMOTE_Z = 60, 2.0
LEDGER = ROOT / "data" / "prediction_lab.json"
NOTE = ROOT.parents[1] / "notes" / "Prediction lab.md"        # the owner's Obsidian vault
BARS_DIR = ROOT / "data" / "futures"
TIMESFM_PY = Path.home() / ".venv-timesfm" / "Scripts" / "python.exe"
S, K, B = lp.S, lp.K, lp.RTH_LAST

PREDICTORS = {   # key -> (plain name, what it does)
    "deployed": ("Deployed race", "buy every day at 19:00 New York (the benchmark; since 7 Oct the bots follow "
                                  "Long-term trend: buy above the 200-day average, sell below)"),
    "first_hour": ("Follow the first hour", "at 10:30 New York, race in the direction NQ moved since 09:30"),
    "long_1030": ("Buy at 10:30", "buy every day at 10:30 New York (fair comparison for the line above)"),
    "events": ("Big-release days", "buy at 19:00 only before a Fed, CPI or jobs-report day; skip others"),
    "vwap_high": ("Above yesterday's VWAP", "buy at 19:00 only when price is high vs yesterday's average traded price"),
    "evening_buying": ("Evening buying", "buy at 19:00 only when 18:00-19:00 buying pressure is in its top third"),
    "treasury_buying": ("Treasury buying", "buy at 19:00 only when yesterday's Treasury buying pressure was in its top third"),
    "timesfm_15m": ("TimesFM 15-min", "race the side Google's TimesFM forecasts for 19:00->16:00"),
    "timesfm_daily": ("TimesFM daily", "race the side TimesFM forecasts for the next daily close"),
    # added 2026-10-04 (before the first scored day) -- lucid_vix_tick_search.py
    "breadth": ("Market breadth", "race the side NYSE stocks leaned yesterday (average NYSE TICK above / below 0)"),
    # added 2026-10-07 -- owner: "we can't work only with trades that predict up"; equal on average
    # 2019-26, better in 2022's bear market (41% vs 34% race wins)
    "trend200": ("Long-term trend", "race up when NQ closed above its 200-day average, down when below"),
    # added 2026-10-07 -- lucid_free_data_search.py: the only free-data rule better than always-buy in
    # BOTH halves (42.5% vs 38.5% on 2016-22, 45.2% vs 42.4% on 2023-26), but not significant -- watch only
    "skew_jump": ("SKEW jump", "buy at 19:00 only when Cboe's SKEW (price of crash protection) rose sharply "
                               "yesterday -- top third of daily changes"),
}
ENTRY_OF = {k: S(10, 30) if k in ("first_hour", "long_1030") else S(19) for k in PREDICTORS}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    handlers=[logging.FileHandler(ROOT / "logs" / "prediction_lab.log", encoding="utf-8"),
                              logging.StreamHandler()])
log = logging.getLogger("prediction_lab")


# --- data ----------------------------------------------------------------------

def front_quarter(root: str, today: date) -> str:
    """NinjaTrader name of the quarterly front contract (rolled a week before the month)."""
    for y in (today.year, today.year + 1):
        for m in (3, 6, 9, 12):
            if date(y, m, 1) - timedelta(days=7) > today:
                return f"{root} {date(y, m, 1):%b}".upper() + f"{y % 100:02d}"
    raise ValueError(today)


def fetch(engine: NinjaTraderEngine) -> dict[str, Path]:
    """Fresh real bars from NinjaTrader: NQ (5.5 years, for TimesFM's daily context) and ZN."""
    paths = {}
    b = engine.get_bars("NQ", "15m", 130_000).iloc[:-1]           # drop the forming stand-in
    paths["NQ"] = BARS_DIR / "lab_NQ_15m.parquet"
    b.to_parquet(paths["NQ"])
    d = engine._call("bars", instrument=front_quarter("ZN", date.today()), minutes=15, count=40_000)
    z = pd.DataFrame(d.get("bars") or [], columns=["t", "open", "high", "low", "close", "volume"])
    z["timestamp"] = pd.to_datetime(z["t"], unit="s", utc=True)
    paths["ZN"] = BARS_DIR / "lab_ZN_15m.parquet"
    z.drop(columns="t").set_index("timestamp").to_parquet(paths["ZN"])
    d = engine._call("bars", instrument="^TICK", minutes=15, count=3_000)
    t = pd.DataFrame(d.get("bars") or [], columns=["t", "open", "high", "low", "close", "volume"])
    t["timestamp"] = pd.to_datetime(t["t"], unit="s", utc=True)
    paths["TICK"] = BARS_DIR / "lab_TICK_15m.parquet"
    t.drop(columns="t").set_index("timestamp").to_parquet(paths["TICK"])
    return paths


def fetch_internet() -> dict[str, Path]:
    """Paper mode (owner 2026-10-15: no paid NinjaTrader data): the stored bars, with the internet's newest
    15-minute NQ and ZN bars (10 minutes delayed) added after them. NYSE TICK is not on the internet feed: its
    file stays as it was, so "Market breadth" has no new days to score."""
    from metatraderbot import free_prices
    paths = {}
    for m in ("NQ", "ZN"):
        p = BARS_DIR / f"lab_{m}_15m.parquet"
        new = free_prices.bars("15m", 6_000, m).iloc[:-1]
        if p.exists():
            old = pd.read_parquet(p)
            old.index = pd.DatetimeIndex(old.index).tz_convert("UTC")
            new = pd.concat([old, new[new.index > old.index[-1]]])
        new.to_parquet(p)
        paths[m] = p
    paths["TICK"] = BARS_DIR / "lab_TICK_15m.parquet"
    return paths


def align(G: dict, VV: dict, idx: pd.DatetimeIndex) -> None:
    """Put the other markets' grids on NQ's trading dates (as lucid_orderflow_search.main)."""
    for m in list(G):
        if m == "NQ":
            continue
        pos = pd.Series(np.arange(len(G[m].dates)), index=pd.to_datetime(G[m].dates)).reindex(idx).to_numpy()
        ok = ~np.isnan(pos)
        ti = np.where(ok, pos, 0).astype(int)
        for attr in ("O", "H", "L", "C"):
            arr = getattr(G[m], attr)[ti].copy()
            arr[~ok] = np.nan
            setattr(G[m], attr, arr)
        v = VV[m][ti].copy()
        v[~ok] = 0
        VV[m] = v


# --- the frozen predictors -------------------------------------------------------

def sides(G: dict, VV: dict, idx: pd.DatetimeIndex, tfm: dict) -> dict[str, np.ndarray]:
    g, V = G["NQ"], VV["NQ"]
    n = len(idx)
    a19, op, cl = S(19), lp.RTH_OPEN, B + 1
    out = {"deployed": np.ones(n, int), "long_1030": np.ones(n, int)}
    out["first_hour"] = np.sign(g.O[:, S(10, 30)] - g.O[:, op]).astype(int)
    ev = json.loads((BARS_DIR / "event_dates.json").read_text(encoding="utf-8"))
    evd = pd.DatetimeIndex([d + pd.offsets.BDay(0) for k in ("fomc", "cpi", "nfp") for d in pd.to_datetime(ev[k])])
    out["events"] = np.where(pd.Series(idx).isin(evd).to_numpy(), 1, 0)
    vw = pd.Series(of.vwap(g, V, op, cl), index=idx).shift(1)
    atr = g.daily["atr14"].reindex(idx)
    t = ss.past_terciles((pd.Series(g.O[:, a19], index=idx) - vw) / atr)
    out["vwap_high"] = np.where(t == 2, 1, 0)
    t = ss.past_terciles(pd.Series(of.pressure(g, V, 0, a19), index=idx))
    out["evening_buying"] = np.where(t == 2, 1, 0)
    if "ZN" in G:
        t = ss.past_terciles(pd.Series(of.pressure(G["ZN"], VV["ZN"], op, cl), index=idx).shift(1))
        out["treasury_buying"] = np.where(t == 2, 1, 0)
    else:
        out["treasury_buying"] = np.zeros(n, int)
    c200 = g.daily["close"].ffill()
    sma200 = c200.rolling(200, min_periods=160).mean()
    out["trend200"] = np.sign((c200.shift(1) - sma200.shift(1)).reindex(idx).fillna(0)).astype(int).to_numpy()
    for key, src in (("timesfm_15m", "m15"), ("timesfm_daily", "md")):
        out[key] = np.array([int(np.sign(tfm.get(str(d.date()), {}).get(src, 0.0))) for d in idx])
    return out


def breadth_sides(tick_path: Path, idx: pd.DatetimeIndex) -> np.ndarray:
    """Side = sign of YESTERDAY's average 15-min NYSE TICK close over 09:30-16:00 ET."""
    import lucid_vix_tick_search as vt
    df = pd.read_parquet(tick_path)
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    sess = vt.rth_15m_stats(df)["session"]
    sess.index = pd.DatetimeIndex(sess.index)
    return np.sign(np.nan_to_num(vt.known_before(sess, idx).to_numpy())).astype(int)


def skew_sides(idx: pd.DatetimeIndex) -> np.ndarray:
    """Buy only when yesterday's change of Cboe SKEW was in its top third (past-year thresholds).
    SKEW is a free Cboe download; without internet the last saved copy is used."""
    import fetch_free_data as ffd
    import lucid_vix_tick_search as vt
    try:
        skew = ffd.fetch_cboe_index("SKEW")
    except Exception as e:  # noqa: BLE001 -- offline: fall back to the saved table
        log.warning(f"Cboe SKEW download failed ({e}) -- using the saved copy")
        try:
            skew = pd.read_parquet(ffd.OUT / "cboe_indices.parquet")["SKEW"].dropna()
        except Exception:  # noqa: BLE001
            return np.zeros(len(idx), int)
    t = ss.past_terciles(vt.known_before(skew.diff(), idx))
    return np.where(t == 2, 1, 0)


def timesfm_forecasts(g: lp.Grid, idx: pd.DatetimeIndex, days: list[int], have: dict) -> dict:
    """TimesFM forecasts for the lab days that do not have one yet (its own environment)."""
    need = [i for i in days if str(idx[i].date()) not in have]
    if not need or not TIMESFM_PY.exists():
        return have
    a19 = S(19)
    flat = pd.Series(g.C.ravel()).ffill().to_numpy()
    closes = g.daily["close"].ffill().to_numpy()
    ctx15 = np.stack([flat[i * K + a19 - 512: i * K + a19] for i in need]).astype(np.float32)
    ctxd = np.stack([closes[i - 512: i] for i in need]).astype(np.float32)
    tmp_in, tmp_out = BARS_DIR / "lab_tfm_in.npz", BARS_DIR / "lab_tfm_out.npz"
    np.savez_compressed(tmp_in, ctx15=ctx15, ctxd=np.nan_to_num(ctxd, nan=float(np.nanmean(closes))))
    r = subprocess.run([str(TIMESFM_PY), str(ROOT / "scripts" / "lucid_timesfm_race.py"), "forecast",
                        "--inp", str(tmp_in), "--out", str(tmp_out)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=1800, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        log.warning(f"TimesFM forecast failed: {(r.stderr or "")[-400:]}")
        return have
    f = np.load(tmp_out)
    for j, i in enumerate(need):
        have[str(idx[i].date())] = {"m15": float(f["ctx15_med"][j]), "md": float(f["ctxd_med"][j])}
    return have


# --- scoring ---------------------------------------------------------------------

def update(engine: NinjaTraderEngine, start: str = LAB_START, ledger_path: Path = LEDGER) -> dict:
    led = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    led.setdefault("frozen", "2026-10-03")
    led.setdefault("start", start)
    led.setdefault("days", {})
    led.setdefault("timesfm", {})
    from metatraderbot import free_prices
    paths = fetch_internet() if engine is None or free_prices.in_use() else fetch(engine)
    for m in ("ZN",):
        lp.MARKETS.setdefault(m, ("nas100", "MNQ", False))
    tick_path = paths.pop("TICK")
    G = {m: lp.load_grid(m, path=p) for m, p in paths.items()}
    g = G["NQ"]
    idx = pd.to_datetime(g.dates)
    VV = {m: of.volume_grid(paths[m], G[m].dates) for m in G}
    align(G, VV, idx)
    lab_days = [i for i in range(len(idx)) if str(idx[i].date()) >= led["start"]
                and g.has[i, B] and g.has[i, S(19)]]
    led["timesfm"] = timesfm_forecasts(g, idx, lab_days, led["timesfm"])
    sd = sides(G, VV, idx, led["timesfm"])
    sd["breadth"] = breadth_sides(tick_path, idx)
    sd["skew_jump"] = skew_sides(idx)
    labels = {a: ss.race_labels(g, a) for a in set(ENTRY_OF.values())}
    for i in lab_days:
        row = {}
        for key in PREDICTORS:
            s_ = int(sd[key][i])
            lab = labels[ENTRY_OF[key]]
            y = lab[1][i] if s_ > 0 else (lab[-1][i] if s_ < 0 else np.nan)
            row[key] = {"side": s_, "result": "skip" if s_ == 0 else
                        ("open at 16:00" if np.isnan(y) else ("win" if y == 1 else "fail"))}
        led["days"][str(idx[i].date())] = row
    led["updated"] = datetime.now(timezone.utc).isoformat()
    led["summary"] = summarise(led)
    tmp = ledger_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(led, indent=1), encoding="utf-8")
    os.replace(tmp, ledger_path)
    return led


def summarise(led: dict) -> dict:
    days = led["days"]
    dep = {d: r["deployed"]["result"] for d, r in days.items()}
    out = {}
    for key in PREDICTORS:
        res = [r[key]["result"] for r in days.values()]
        w, f_ = res.count("win"), res.count("fail")
        n = w + f_
        # the deployed race on the SAME days this predictor raced and resolved
        both = [dep[d] for d, r in days.items() if r[key]["result"] in ("win", "fail") and dep[d] in ("win", "fail")]
        p0 = both.count("win") / len(both) if both else 0.4
        p = w / n if n else float("nan")
        z = (p - p0) / math.sqrt(max(p0 * (1 - p0), 1e-6) / n) if n else float("nan")
        out[key] = {"raced": len(res) - res.count("skip"), "resolved": n, "wins": w, "win_rate": p,
                    "deployed_same_days": p0, "z": z,
                    "status": ("PROMISING - tell the owner" if n >= PROMOTE_N and z >= PROMOTE_Z else
                               ("watching" if n < PROMOTE_N else "no edge so far"))}
    return out


def write_note(led: dict) -> None:
    s = led["summary"]
    n_days = len(led["days"])
    lines = ["# Prediction lab", "",
             f"*Updated {datetime.now():%Y-%m-%d %H:%M} by the lab (`scripts/run_prediction_lab.py`). "
             f"Predictors frozen {led['frozen']}; scored from {led['start']} on days they have never seen.*", "",
             "Nothing here trades. A predictor is worth switching to only after **60+ resolved races** with a win "
             "rate **clearly above the deployed race on the same days** (z ≥ 2). Until then every number below "
             "can be luck.", "",
             f"Trading days scored: **{n_days}**", "",
             "| Predictor | What it does | Raced | Won / resolved | Win rate | Deployed, same days | Status |",
             "|---|---|---|---|---|---|---|"]
    for key, (name, what) in PREDICTORS.items():
        r = s[key]
        wr = f"{r['win_rate']:.0%}" if r["resolved"] else "—"
        dep = f"{r['deployed_same_days']:.0%}" if r["resolved"] else "—"
        lines.append(f"| {name} | {what} | {r['raced']} | {r['wins']} / {r['resolved']} | {wr} | {dep} | "
                     f"{r['status']} |")
    lines += ["", "*Win = the +$3,000 target before the −$2,000 limit (2 NQ). 'Open at 16:00' days are not counted.*",
              "", "See [[Research log]] for how these predictors were found."]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def due(led: dict) -> bool:
    """A trading day has closed (16:15 ET) that the ledger does not have yet."""
    now = datetime.now(lucid_rules.ET)
    d = now.date() if now.hour * 60 + now.minute >= 16 * 60 + 15 else now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return str(d) >= led.get("start", LAB_START) and str(d) not in led.get("days", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--start", default=LAB_START, help="first day to score (testing only)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--no-note", action="store_true")
    args = ap.parse_args()
    try:                                     # never compete with the trading bots for the CPU
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:  # noqa: BLE001
        pass
    from metatraderbot import free_prices
    engine = None if free_prices.in_use() else \
        NinjaTraderEngine(account=os.environ.get("NT_ACCOUNT", "Sim101").strip(), timeout_s=120.0)
    last_try = 0.0
    while True:
        try:
            led = json.loads(args.ledger.read_text(encoding="utf-8")) if args.ledger.exists() else {}
            if not args.loop or (due(led) and time.monotonic() - last_try > 3 * 3600) or not led:
                last_try = time.monotonic()
                led = update(engine, args.start, args.ledger)
                if not args.no_note:
                    write_note(led)
                    try:                     # the trade brain: ledger + learning notes (scripts/trade_brain.py)
                        import trade_brain
                        mem = trade_brain.update()
                        log.info(f"trade brain: {len(mem['trades'])} trades in the ledger")
                    except Exception as e:  # noqa: BLE001 -- never let the notes stop the lab
                        log.warning(f"trade brain failed: {e!r}")
                log.info(f"scored {len(led['days'])} day(s): " + ", ".join(
                    f"{k} {v['wins']}/{v['resolved']}" for k, v in led["summary"].items()))
        except Exception as e:  # noqa: BLE001 -- NinjaTrader closed / no data yet: try again later
            log.warning(f"update failed: {e!r}")
        if not args.loop:
            return 0
        time.sleep(20 * 60)


if __name__ == "__main__":
    raise SystemExit(main())
