"""Post-pass survivability check for the top 4-day-FTMO candidates.

Passing a 4-day evaluation is not the end goal -- the config keeps running
on a funded account afterward, unchanged. A config that "passes" mostly via
extreme risk% x concurrency variance-harvesting can still be a bad long-run
bet even if its 4-day pass rate looks good, because the SAME aggression
that helps it hit +10% fast also raises long-run ruin probability (this
basket is only ~4.9 independent bets per the correlation-structure work in
this repo -- concurrency does not buy as much diversification as it looks
like it does).

This script, for a short list of candidate configs (strategy/timeframe
already fixed and pooled into a cache by ftmo_4day_search.py or
ftmo_4day_sltp_sweep.py):

  1. Reports the RAW per-trade OOS expectancy (mean R), win rate, PF and a
     per-trade Sharpe-like ratio (mean/std of R) -- is there an actual edge
     independent of position sizing, or is the 4-day pass rate pure
     variance harvesting on a ~flat/negative-edge stream?
  2. Runs the SAME real-replay engine (real bars, mark-to-market, all three
     FTMO rules) for a LONG horizon (250+ trading days) starting from many
     different dates, with the SAME risk% / concurrency the config used to
     pass -- i.e. "what happens if nothing changes after the pass." Reports
     % of 250-day windows that blow (hit -10% total or -5% in a day) at
     some point, and the average terminal equity for windows that don't.

No Monte Carlo -- same historical-replay methodology as the rest of this
project's FTMO work.

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_survivability.py
"""
from __future__ import annotations

import pickle
import statistics as stats
from pathlib import Path

ROOT = Path(r"C:\Users\proksa\OneDrive\Počítač\New folder\ruflo\ruflo\metatraderbot")

DAILY_LIM, TOTAL_LIM = 0.05, 0.10
LONG_HORIZON = 250

# Filled in from the results of ftmo_4day_search.py / ftmo_4day_sltp_sweep.py.
# Each candidate: (label, cache_file, cache_key, risk, conc, per_day)
# cache_key is either a roster string (search.py cache) or an (sl, tp) tuple
# (sltp_sweep.py cache) -- resolved against whichever cache file is named.
CANDIDATES: list[tuple[str, str, object, float, int, int]] = [
    ("RAW-PASS WINNER (risk 10%)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.10, 15, 6),
    ("DEFENSIBLE fail<=60% (risk 6%)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.06, 15, 6),
    ("DEFENSIBLE fail<=55% (risk 5%)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.05, 15, 6),
    ("DEFENSIBLE fail<=50% (risk 4%)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.04, 15, 6),
    ("RUNNER-UP: CONF+Supertrend@2R (risk 10%)", "cache_1d_trades.pkl",
     "CONF 1.0R + Supertrend@2R", 0.10, 15, 6),
    ("RUNNER-UP: CONF+Boll@4R (risk 6%)", "cache_1d_trades.pkl",
     "CONF 1.0R + Boll@4R", 0.06, 15, 6),
]


def trade_stats(trades: list[dict]) -> dict:
    rs = [t["final"] for t in trades]
    n = len(rs)
    if n == 0:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_loss = -sum(losses)
    mean = sum(rs) / n
    sd = stats.pstdev(rs) if n > 1 else 0.0
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "e_net": mean,
        "pf": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
        "sharpe_per_trade": (mean / sd) if sd > 0 else 0.0,
    }


def long_run_replay(by_day, days, i0, hz, risk, conc, per_day, cap_ccy=0):
    """Same portfolio mechanics as the pass/fail replay(), but runs the full
    horizon (or until the account blows), returning enough to judge
    survivability rather than a single pass/fail verdict."""
    eq, real, op = 1.0, 0.0, []
    peak, max_dd = 1.0, 0.0
    blown_at = None
    days_run = 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        days_run += 1
        d0 = eq
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            if cap_ccy:
                sym = tr["sym"]
                ccy = {sym[:3], sym[3:]}
                held = sum(1 for p, _ in op if ccy & {p["sym"][:3], p["sym"][3:]})
                if held >= cap_ccy:
                    continue
            op.append([tr, 0])
            opened += 1
        fl, still = 0.0, []
        for tr, age in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * risk
                still.append([tr, age])
            else:
                real += tr["final"] * risk
        op = still
        eq = 1.0 + real + fl
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            blown_at = j + 1
            break
    return {"final_eq": eq, "max_dd": max_dd, "blown": blown_at is not None,
            "blown_day": blown_at, "days_run": days_run}


def run_candidate(label, cache_file, key, risk, conc, per_day):
    with open(ROOT / "scratch" / cache_file, "rb") as f:
        cache = pickle.load(f)
    trades, by_day, days = cache[key]
    ts = trade_stats(trades)

    starts = list(range(0, max(len(days) - 1, 1), 4))
    results = [long_run_replay(by_day, days, i, LONG_HORIZON, risk, conc, per_day)
               for i in starts if i < len(days)]
    n = len(results)
    blown = sum(1 for r in results if r["blown"])
    survived = [r for r in results if not r["blown"]]
    avg_final_survived = (sum(r["final_eq"] for r in survived) / len(survived)
                           if survived else float("nan"))
    avg_final_all = sum(r["final_eq"] for r in results) / n if n else float("nan")
    avg_max_dd = sum(r["max_dd"] for r in results) / n if n else float("nan")

    print(f"\n[{label}]  risk={risk*100:.1f}%  conc={conc}  {per_day}/day  "
          f"(from {cache_file}:{key})")
    print(f"    per-trade: n={ts['n']}  win_rate={ts.get('win_rate', float('nan')):.1%}  "
          f"E_net={ts.get('e_net', float('nan')):+.3f}R  PF={ts.get('pf', float('nan')):.2f}  "
          f"per-trade Sharpe={ts.get('sharpe_per_trade', float('nan')):.3f}")
    print(f"    {LONG_HORIZON}-day continuation ({n} windows): "
          f"blown {blown/n:.1%}  survived-avg-final-equity {avg_final_survived:+.2f}x  "
          f"all-windows-avg-final-equity {avg_final_all:+.2f}x  avg-max-DD {avg_max_dd:.1%}")
    return {"label": label, "trade_stats": ts, "n_windows": n, "blown_rate": blown / n if n else 0,
            "avg_final_survived": avg_final_survived, "avg_final_all": avg_final_all,
            "avg_max_dd": avg_max_dd}


def main() -> int:
    print("=" * 108)
    print(f"POST-PASS SURVIVABILITY -- same config run for {LONG_HORIZON} trading days "
          f"after a simulated pass, real replay, no Monte Carlo")
    print("=" * 108)
    if not CANDIDATES:
        print("No candidates configured -- fill in CANDIDATES at the top of this script.")
        return 1
    for c in CANDIDATES:
        run_candidate(*c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
