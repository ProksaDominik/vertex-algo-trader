"""Walk-forward check of the ORB + trend-filter choice (backtest-expert step 5).

For each test year Y (2019-2026): pick the ORB cell (opening range 15-60 min x
last entry 11:00-13:00 x target 2R / end of day) with the best $/trade on
2016..Y-1 ONLY, then trade year Y with it. The out-of-sample years strung
together are what a trader re-selecting every January would actually have
earned. Compared with the fixed plateau-centre setting (45 min / 12:00 / 2R).
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import futures_hypotheses as fh  # noqa: E402
import futures_robustness as fr  # noqa: E402

CELLS = list(product((1, 2, 3, 4), (fh.M(11, 0), fh.M(12, 0), fh.M(13, 0)), (2.0, None)))
CENTRE = (3, fh.M(12, 0), 2.0)
MARKETS = {"MNQ": "nas100", "MGC": "xauusd"}


def main() -> int:
    for contract, proxy in MARKETS.items():
        days = fh.load_days(fh.bars_from_parquet(ROOT / "data" / f"{proxy}_15m_long.parquet"))
        dd = fh.daily_frame(days)
        trades = {}
        for cell in CELLS:
            t = pd.DataFrame(fr.orb_variant(days, dd, fh.Cost(contract), cell[0], cell[1], cell[2], "trend"))
            t["year"] = pd.to_datetime(t["date"]).dt.year
            trades[cell] = t
        rows = []
        for y in range(2019, 2027):
            train = {c: t[t.year < y]["usd"].mean() for c, t in trades.items()}
            pick = max(train, key=train.get)
            oos = trades[pick][trades[pick].year == y]["usd"]
            fixed = trades[CENTRE][trades[CENTRE].year == y]["usd"]
            rows.append({"year": y, "picked": f"{15 * pick[0]}m/{pick[1] // 60}:{pick[1] % 60:02d}/"
                                               f"{'2R' if pick[2] else 'EOD'}",
                         "train_$": train[pick], "oos_n": len(oos), "oos_$": oos.mean(), "oos_sum": oos.sum(),
                         "centre_$": fixed.mean(), "centre_sum": fixed.sum()})
        r = pd.DataFrame(rows)
        print(f"=== {contract}: walk-forward, $ per 1 micro ===")
        print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"  out-of-sample total ${r['oos_sum'].sum():,.0f} over {r['oos_n'].sum()} trades "
              f"(${r['oos_sum'].sum() / max(1, r['oos_n'].sum()):.2f}/trade), positive years "
              f"{(r['oos_sum'] > 0).mean():.0%} | fixed centre ${r['centre_sum'].sum():,.0f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
