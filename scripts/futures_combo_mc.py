"""ORB + trend, with afternoon continuation on the days ORB did not trade.

One position per contract (the bridge allows one bot ticket per contract), so
the afternoon setup (futures_second_edge.py H5) can only use days on which the
ORB leg did not trade. This builds that combined leg for MNQ and MGC, checks
it the same ways (halves, positive years, walk-forward over the afternoon
cells), and runs the LucidPro Monte Carlo for the combined stack.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402
import futures_robustness as fr  # noqa: E402
import futures_second_edge as se  # noqa: E402
import lucid_montecarlo as mc  # noqa: E402

M = fh.M
LEGS = {"MNQ": "nas100", "MGC": "xauusd"}
AFTERNOON_CELLS = list(product((M(12, 0), M(13, 0)), (0.3, 0.5, 0.7), ("open", "half")))
CENTRE = (M(13, 0), 0.5, "half")          # centre of the afternoon plateau


def main() -> int:
    fh.keep_awake()
    combined = {}
    for contract, proxy in LEGS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = fh.daily_frame(days)
        orb = pd.DataFrame(fr.orb_variant(days, dd, fh.Cost(contract), 3, M(12, 0), 2.0, "trend"))
        orb_days = set(orb["date"])
        free = [x for x in days if x.date not in orb_days]
        per_cell = {}
        for cell in AFTERNOON_CELLS:
            t = pd.DataFrame(se.h5_afternoon(free, dd, fh.Cost(contract), *cell))
            t["year"] = pd.to_datetime(t["date"]).dt.year
            per_cell[cell] = t
        # walk-forward over the afternoon cells, on ORB-free days
        wf_sum, wf_n, wf_years = 0.0, 0, []
        for y in range(2019, 2027):
            train = {c: t[t.year < y]["usd"].mean() for c, t in per_cell.items()}
            pick = max(train, key=train.get)
            oos = per_cell[pick][per_cell[pick].year == y]["usd"]
            wf_sum += oos.sum()
            wf_n += len(oos)
            wf_years.append(oos.sum() > 0)
        aft = per_cell[CENTRE].drop(columns="year")
        m_a = fh.metrics(aft)
        print(f"{contract} afternoon-on-ORB-free-days (centre cell): n={m_a['n']} ${m_a['usd_mean']:.2f}/trade "
              f"t={m_a['t']:.2f} halves ${m_a['h1']:.2f}/${m_a['h2']:.2f} years+ {m_a['years_pos']:.0%} | "
              f"walk-forward ${wf_sum:,.0f} over {wf_n} trades, positive years {np.mean(wf_years):.0%}")
        both = pd.concat([orb.assign(part="orb"), aft.assign(part="afternoon")], ignore_index=True)
        both["leg"] = contract
        m_b = fh.metrics(both)
        corr = pd.concat([orb.groupby("date")["usd"].sum(), aft.groupby("date")["usd"].sum()], axis=1).fillna(0).corr().iloc[0, 1]
        print(f"   combined leg: n={m_b['n']} ${m_b['usd_mean']:.2f}/trade t={m_b['t']:.2f} "
              f"halves ${m_b['h1']:.2f}/${m_b['h2']:.2f} years+ {m_b['years_pos']:.0%} "
              f"({m_b['per_week']:.2f}/wk; daily corr ORB vs afternoon {corr:+.2f})")
        combined[contract] = (orb.assign(leg=contract), both)

    cal = pd.DatetimeIndex(pd.bdate_range("2016-01-04", "2026-08-14"))
    print("\n=== LucidPro 50K Monte Carlo: ORB only vs ORB + afternoon ===")
    print(f"{'stack':<34} {'$/wk':>6} {'eval pass':>9} {'fails/pass':>10} {'days':>5} | {'funded pay':>10} {'blow':>6} {'days':>5}")
    for label, idx in (("ORB only", 0), ("ORB + afternoon", 1)):
        trades = pd.concat([combined[c][idx] for c in LEGS], ignore_index=True)
        for sizes in ({"MNQ": 4, "MGC": 2}, {"MNQ": 6, "MGC": 3}):
            pnl, low = mc.daily_series(trades, sizes, cal)
            ev = mc.simulate(pnl, low, "eval", horizon=120)
            fu = mc.simulate(pnl, low, "funded", horizon=120)
            name = f"{label} {sizes['MNQ']} MNQ + {sizes['MGC']} MGC"
            print(f"{name:<34} {pnl.mean() * 5:>6.0f} {ev['p_success']:>9.0%} "
                  f"{ev['p_fail'] / max(ev['p_success'], 1e-9):>10.2f} {ev['median_days']:>5.0f} | "
                  f"{fu['p_success']:>10.0%} {fu['p_fail']:>6.0%} {fu['median_days']:>5.0f}")
    pd.concat([combined[c][1] for c in LEGS], ignore_index=True).to_parquet(fh.OUT / "stack_orb_afternoon_trades.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
