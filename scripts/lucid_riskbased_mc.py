"""Fixed CONTRACTS vs fixed DOLLAR RISK per trade, through LucidPro Monte Carlo.

Fixed contracts let wide-opening-range days carry much bigger losses. Sizing
each trade as floor(risk$ / (stop distance x point value + commission)) keeps
every full stop-out near the same dollar amount -- the realistic way to cap a
day's damage (a flat "$600 daily stop" on the day's P&L is not: a single stop
at 4 MNQ already loses ~$680 in one fill, which is why that earlier test was
thrown out as a modelling artifact).

Contracts per trade are capped at 10 per leg (well inside Lucid's 40 micros),
and a trade whose single contract already exceeds the budget is skipped --
exactly what the live bots do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import lucid_montecarlo as mc  # noqa: E402
from metatraderbot import futures  # noqa: E402

MAX_PER_LEG = 10


def sized(trades: pd.DataFrame, risk_usd: dict[str, float]) -> pd.DataFrame:
    t = trades.copy()
    pv = t["leg"].map(futures.point_value)
    comm = 2 * t["leg"].map(futures.commission_per_side)
    per_contract = t["risk_pts"] * pv + comm
    n = np.floor(t["leg"].map(risk_usd) / per_contract).clip(0, MAX_PER_LEG)
    t["usd"] = t["usd"] * n
    t["contracts"] = n
    return t[n > 0]


def main() -> int:
    trades = pd.read_parquet(ROOT / "data" / "futures" / "stack_orb_trend_trades.parquet")
    for label, start in (("2016-2026", "2016-01-01"), ("2021-2026", "2021-01-01")):
        base = trades[pd.to_datetime(trades["date"]) >= pd.Timestamp(start)]
        cal = pd.DatetimeIndex(pd.bdate_range(start, "2026-08-14"))
        print(f"=== history {label} ===")
        print(f"{'sizing':<34} {'$/wk':>6} {'eval pass':>9} {'fail':>6} {'fails/pass':>10} {'days':>5} | "
              f"{'funded pay':>10} {'blow':>6} {'days':>5} | {'avg MNQ':>7} {'avg MGC':>7}")
        rows = [("fixed 4 MNQ + 2 MGC", base.assign(usd=base["usd"] * base["leg"].map({"MNQ": 4, "MGC": 2})), None)]
        for r in (250, 350, 450, 600):
            rows.append((f"risk ${r}/trade per leg", sized(base, {"MNQ": r, "MGC": r}), r))
        for r_mnq, r_mgc in ((600, 300), (450, 450), (700, 350)):
            rows.append((f"risk MNQ ${r_mnq} / MGC ${r_mgc}", sized(base, {"MNQ": r_mnq, "MGC": r_mgc}), None))
        for name, t, _ in rows:
            t = t.copy()
            t["leg"] = t["leg"]
            pnl, low = mc.daily_series(t.assign(leg="ALL"), {"ALL": 1}, cal)
            ev = mc.simulate(pnl, low, "eval", horizon=120)
            fu = mc.simulate(pnl, low, "funded", horizon=120)
            avg = t.groupby("leg")["contracts"].mean() if "contracts" in t else pd.Series({"MNQ": 4, "MGC": 2})
            print(f"{name:<34} {pnl.mean() * 5:>6.0f} {ev['p_success']:>9.0%} {ev['p_fail']:>6.0%} "
                  f"{ev['p_fail'] / max(ev['p_success'], 1e-9):>10.2f} {ev['median_days']:>5.0f} | "
                  f"{fu['p_success']:>10.0%} {fu['p_fail']:>6.0%} {fu['median_days']:>5.0f} | "
                  f"{avg.get('MNQ', 0):>7.1f} {avg.get('MGC', 0):>7.1f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
