"""Pick the Lucid FUTURES funded roster from real futures history.

Why this exists
---------------
The MT5 funded roster (Donchian/MACD/Supertrend on 1d/4h, BTC, indices, gold)
cannot run on Lucid: positions must be flat by 4:45 PM ET every day and
Bitcoin is not an approved product. So the funded bot needs intraday legs,
chosen the same way the MT5 ones were -- screened on data, not picked by hand.

What it does
------------
1. Pulls 15m and 1h history for Lucid's 12 most liquid contracts from
   NinjaTrader (through the VertexBridge add-on) into data/futures/.
2. Runs every strategy config in the lab registry (same families and
   parameter grids the lab forward-tests) on every contract, on a rolling
   300-bar window -- exactly what the live bots feed the strategies.
3. Simulates each signal under LUCID'S rules: entry at the next bar's open,
   no entry within 15 minutes of the 4:45 PM ET cutoff, forced exit when the
   flatten window arrives, one position at a time per contract, costs of one
   tick plus Lucid's commission both ways. A bar that spans stop and target
   is scored as the stop (conservative).
4. Scores each (config, contract, target): trades, e_net in R, win rate,
   both chronological halves, and e_net with costs doubled.
5. Proposes a roster (data/futures/funded_roster.json) from the configs that
   pass every robustness bar, at most one leg per contract (the bridge allows
   one open bot ticket per contract) and at most two equity-index legs (they
   move together).

The proposal is a starting point for a human, not a verdict: 15 months of
intraday data is one regime, and searching ~1,800 cells finds some winners by
chance. The both-halves and double-cost bars are there to thin those out.

Resumable in short sessions: every finished cell is appended to
data/futures/roster_search_cells.jsonl straight away, a re-run skips cells
already there, and --max-minutes stops handing out new cells in time to shut
the machine down. The CSV and the roster proposal are rebuilt from the whole
checkpoint at the end of every session, so partial progress is never lost.

Usage:
    .venv/Scripts/python.exe scripts/futures_roster_search.py            # download + search
    .venv/Scripts/python.exe scripts/futures_roster_search.py --no-download
    .venv/Scripts/python.exe scripts/futures_roster_search.py --no-download --max-minutes 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot import futures, lucid_rules  # noqa: E402
from metatraderbot.lab_registry import FAMILIES, FUTURES_SYMBOLS, _grid, instantiate  # noqa: E402

DATA_DIR = ROOT / "data" / "futures"
TIMEFRAMES = {"15m": (15, 30_000), "1h": (60, 10_000)}   # minutes, bars to pull
TP_LEVELS = [1.0, 2.0]
WINDOW = 300                 # bars fed to a strategy, as in the live runners
MIN_TICKS_STOP = 2           # a stop tighter than this is not a tradeable signal

# Robustness bars for the roster proposal
MIN_TRADES = 50
MIN_E_NET = 0.05
MAX_LEGS = 6
MAX_INDEX_LEGS = 2
INDEX_ROOTS = {"MES", "MNQ", "MYM", "M2K"}


# --- data ----------------------------------------------------------------------

def download() -> None:
    from metatraderbot.nt_engine import NinjaTraderEngine
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eng = NinjaTraderEngine(timeout_s=180)
    for sym in FUTURES_SYMBOLS:
        for tf, (_mins, count) in TIMEFRAMES.items():
            bars = eng.get_bars(sym, tf, count).iloc[:-1]      # drop the forming stand-in
            bars.to_parquet(DATA_DIR / f"{sym}_{tf}.parquet")
            print(f"  {sym:>4} {tf:>3}: {len(bars):>6} bars  {bars.index[0]:%Y-%m-%d} -> {bars.index[-1]:%Y-%m-%d}",
                  flush=True)


@lru_cache(maxsize=8)
def load(sym: str, tf: str) -> tuple[pd.DataFrame, list]:
    df = pd.read_parquet(DATA_DIR / f"{sym}_{tf}.parquet")
    mins = TIMEFRAMES[tf][0]
    closes = [(t + pd.Timedelta(minutes=mins)).to_pydatetime() for t in df.index]
    states = [lucid_rules.session_state(c) for c in closes]
    return df, states


# --- one cell: (family, params, tf, symbol) -> results for every TP -----------

def run_cell(family: str, params: dict, tf: str, sym: str) -> list[dict]:
    df, states = load(sym, tf)
    spec = futures.spec(sym)
    tick, pv, comm = spec["tick_size"], spec["point_value"], spec["commission"]
    strat = instantiate(family, {family: dict(params)}, 1.0, sym)

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(df)

    # Signals first (independent of the target): bar index, direction, stop distance.
    signals = []
    for i in range(WINDOW, n - 1):
        if states[i] != lucid_rules.Session.OPEN:
            continue                    # entry would fall at/after the cutoff
        window = df.iloc[i - WINDOW + 1: i + 1]
        try:
            specs = strat.on_bar(len(window) - 1, window)
        except Exception:
            continue
        for s in specs or []:
            d = getattr(s, "direction", None)
            is_long = getattr(d, "value", str(d)).lower() == "long"
            dist = abs(float(s.entry_price) - float(s.stop_loss))
            if dist >= MIN_TICKS_STOP * tick:
                signals.append((i, is_long, dist))
            break

    out = []
    for tp_r in TP_LEVELS:
        trades = []                     # (entry_ts, r_net, r_gross, cost_r)
        busy_until = -1
        for i, is_long, dist in signals:
            if i <= busy_until:
                continue                # one position per contract, like the bot
            e = i + 1
            entry = o[e]
            sl = entry - dist if is_long else entry + dist
            tp = entry + dist * tp_r if is_long else entry - dist * tp_r
            exit_px, j = None, e
            for j in range(e, n):
                hit_sl = l[j] <= sl if is_long else h[j] >= sl
                hit_tp = h[j] >= tp if is_long else l[j] <= tp
                if hit_sl:
                    exit_px = sl        # also when both: conservative
                    break
                if hit_tp:
                    exit_px = tp
                    break
                if states[j] in (lucid_rules.Session.FLATTEN, lucid_rules.Session.CLOSED):
                    exit_px = c[j]      # Lucid flat rule
                    break
            if exit_px is None:
                break                   # ran off the end of the data
            busy_until = j
            gross = ((exit_px - entry) if is_long else (entry - exit_px)) / dist
            cost_r = (tick * pv + 2 * comm) / (dist * pv)
            trades.append((df.index[e], gross - cost_r, gross, cost_r))

        if not trades:
            out.append({"family": family, "params": json.dumps(params, sort_keys=True), "tf": tf,
                        "tp_r": tp_r, "symbol": sym, "n": 0})
            continue
        tdf = pd.DataFrame(trades, columns=["ts", "r", "gross", "cost"])
        half = len(tdf) // 2
        weeks = max(1.0, (tdf["ts"].iloc[-1] - tdf["ts"].iloc[0]).days / 7.0)
        sd = float(tdf["r"].std(ddof=1)) if len(tdf) > 1 else 0.0
        out.append({
            "family": family, "params": json.dumps(params, sort_keys=True), "tf": tf,
            "tp_r": tp_r, "symbol": sym,
            "n": len(tdf),
            "e_net": float(tdf["r"].mean()),
            "win": float((tdf["r"] > 0).mean()),
            "h1": float(tdf["r"].iloc[:half].mean()) if half else 0.0,
            "h2": float(tdf["r"].iloc[half:].mean()),
            "e_net_2x": float((tdf["r"] - tdf["cost"]).mean()),
            "avg_cost_r": float(tdf["cost"].mean()),
            "t": float(tdf["r"].mean() / (sd / len(tdf) ** 0.5)) if sd > 0 else 0.0,
            "per_week": len(tdf) / weeks,
            "first": str(tdf["ts"].iloc[0]), "last": str(tdf["ts"].iloc[-1]),
        })
    return out


def _cell_safe(args):
    try:
        return run_cell(*args)
    except Exception as e:  # noqa: BLE001 -- one broken cell must not stop the search
        fam, params, tf, sym = args
        return [{"family": fam, "params": json.dumps(params, sort_keys=True), "tf": tf,
                 "symbol": sym, "n": 0, "error": repr(e)}]


# --- roster proposal --------------------------------------------------------------

def propose(res: pd.DataFrame) -> list[dict]:
    ok = res[(res["n"] >= MIN_TRADES) & (res["e_net"] >= MIN_E_NET) & (res["h1"] > 0)
             & (res["h2"] > 0) & (res["e_net_2x"] > 0)].copy()
    ok = ok.sort_values("t", ascending=False)
    legs, used, n_index = [], set(), 0
    for _, r in ok.iterrows():
        if r["symbol"] in used:
            continue
        if r["symbol"] in INDEX_ROOTS:
            if n_index >= MAX_INDEX_LEGS:
                continue
            n_index += 1
        used.add(r["symbol"])
        legs.append({"symbol": r["symbol"], "family": r["family"], "params": json.loads(r["params"]),
                     "tf": r["tf"], "tp_r": float(r["tp_r"]),
                     "evidence": {k: round(float(r[k]), 4) for k in
                                  ("n", "e_net", "win", "h1", "h2", "e_net_2x", "t", "per_week")}})
        if len(legs) >= MAX_LEGS:
            break
    return legs


CHECKPOINT = DATA_DIR / "roster_search_cells.jsonl"


def cell_key(fam: str, params: dict, tf: str, sym: str) -> str:
    return f"{fam}|{json.dumps(params, sort_keys=True)}|{tf}|{sym}"


def load_checkpoint() -> tuple[set[str], list[dict]]:
    done, rows = set(), []
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue                 # a line cut off by a shutdown mid-write
            done.add(rec["key"])
            rows.extend(rec["rows"])
    return done, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop handing out new cells after this long (for short sessions)")
    args = ap.parse_args()

    if not args.no_download:
        print("[1/3] downloading history from NinjaTrader ...", flush=True)
        download()

    import time as _time
    started = _time.monotonic()
    done_keys, rows = load_checkpoint()
    cells = [(fam, params, tf, sym)
             for fam in FAMILIES for params in _grid(fam, FAMILIES[fam])
             for tf in TIMEFRAMES for sym in FUTURES_SYMBOLS]
    todo = [c for c in cells if cell_key(*c) not in done_keys]
    print(f"[2/3] {len(cells)} cells, {len(cells) - len(todo)} already done, "
          f"{len(todo)} to run on {args.workers} workers ...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(CHECKPOINT, "a", encoding="utf-8") as ck:
        pending = {}
        queue = list(todo)
        while queue or pending:
            out_of_time = args.max_minutes is not None and \
                (_time.monotonic() - started) / 60.0 >= args.max_minutes
            while queue and len(pending) < args.workers * 2 and not out_of_time:
                cell = queue.pop(0)
                pending[ex.submit(_cell_safe, cell)] = cell
            if not pending:
                break
            finished = next(as_completed(list(pending)))
            cell = pending.pop(finished)
            cell_rows = finished.result()
            rows.extend(cell_rows)
            ck.write(json.dumps({"key": cell_key(*cell), "rows": cell_rows}) + "\n")
            ck.flush()
            done += 1
            if done % 50 == 0:
                print(f"      {done}/{len(todo)} cells this session", flush=True)
        if queue:
            print(f"      time limit reached -- {len(queue)} cells left for the next session", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(DATA_DIR / "roster_search.csv", index=False)
    legs = propose(res)
    proposal = {"generated": datetime.now(timezone.utc).isoformat(),
                "method": "scripts/futures_roster_search.py",
                "bars": {tf: v[1] for tf, v in TIMEFRAMES.items()},
                "filters": {"min_trades": MIN_TRADES, "min_e_net": MIN_E_NET,
                            "both_halves_positive": True, "positive_at_2x_costs": True,
                            "max_legs": MAX_LEGS, "max_index_legs": MAX_INDEX_LEGS},
                "legs": legs}
    (DATA_DIR / "funded_roster.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    print("[3/3] done.")
    good = res[res["n"] >= MIN_TRADES].sort_values("t", ascending=False)
    cols = ["family", "params", "tf", "tp_r", "symbol", "n", "e_net", "win", "h1", "h2", "e_net_2x", "t", "per_week"]
    print("\nTop 25 by t-stat (n >= %d):" % MIN_TRADES)
    print(good[cols].head(25).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nProposed roster ({len(legs)} legs) -> {DATA_DIR / 'funded_roster.json'}")
    for leg in legs:
        ev = leg["evidence"]
        print(f"  {leg['symbol']:>4} {leg['family']:<16} {json.dumps(leg['params']):<40} {leg['tf']:>3} "
              f"TP {leg['tp_r']:.0f}R  n={ev['n']:.0f} e_net={ev['e_net']:+.3f} win={ev['win']:.0%} "
              f"h1={ev['h1']:+.3f} h2={ev['h2']:+.3f} 2x={ev['e_net_2x']:+.3f} ({ev['per_week']:.1f}/wk)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
