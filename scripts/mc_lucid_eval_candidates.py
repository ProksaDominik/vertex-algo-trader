"""Monte Carlo of a LucidPro 50K evaluation for the few strategies with a small plus (owner, 2026-10-08).

The question: a small real edge traded SMALL can in theory pass more often than the race (just slower). So
each candidate's own trade list is run through the evaluation rules:
  target +$3,000 on the closed balance; loss limit $2,000 below the best end-of-day balance (locks at +$100);
  every trade sized to risk a fixed dollar amount in micro contracts (MNQ $2/pt, MES $5/pt; at least 1).
  (Optimistic: losses INSIDE a trade are not counted against the limit -- Lucid counts open equity.)
Two ways:
  history  -- an evaluation started at every 5th trade, running forward through the real trade order;
  shuffled -- 10,000 evaluations on trades drawn at random (with replacement) from the list.
Candidates: the 30-minute ORB breakout (research_three_setups, the "other side" stop, volume filter, 3R --
the best and most consistent version), on NQ / ES 5-min and NQ 15-min 10 years; the swing breakout with a
measured-move target (research_article_variants) on NQ 5-min. The race passes ~41% in 1-2 days.

Usage:  .venv/Scripts/python.exe scripts/mc_lucid_eval_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_article_variants as av  # noqa: E402
import research_three_setups as base  # noqa: E402

RNG = np.random.default_rng(7)


def evaluate(pnl: np.ndarray, days: np.ndarray, target=3000.0, mll=2000.0, lock=100.0, max_trades=3000):
    """(passed?, trades used, calendar days) for one evaluation over a sequence of trade results."""
    bal = peak_eod = 0.0
    cur_day = days[0] if len(days) else None
    for k, (x, d) in enumerate(zip(pnl[:max_trades], days[:max_trades])):
        if d != cur_day:
            peak_eod = max(peak_eod, bal)                      # end-of-day high water mark
            cur_day = d
        floor = min(peak_eod - mll, lock)
        bal += x
        if bal <= floor:
            return False, k + 1, (d - days[0]).days
        if bal >= target:
            return True, k + 1, (d - days[0]).days
    return None, len(pnl), None


def trades_usd(t: pd.DataFrame, risk_usd: float, usd_per_pt: float) -> np.ndarray:
    k = np.maximum(1, np.floor(risk_usd / (t["risk"].to_numpy() * usd_per_pt)))
    return t["R"].to_numpy() * t["risk"].to_numpy() * usd_per_pt * k


def main() -> int:
    cands = [
        ("ORB 30-min breakout, NQ 5-min 2021-26", base.run_orb(("NQ 5-min 2021-26", "opposite side", True, 3.0)), 2.0),
        ("ORB 30-min breakout, ES 5-min 2023-26", base.run_orb(("ES 5-min 2023-26", "opposite side", True, 3.0)), 5.0),
        ("ORB 30-min breakout, NQ 15-min 2016-26", base.run_orb(("NQ 15-min 2016-26", "opposite side", True, 3.0)), 2.0),
        ("swing breakout measured move, NQ 5-min 2021-26", av.swing_measured("NQ 5-min 2021-26", None), 2.0),
    ]
    rows = []
    for name, t, ppt in cands:
        t = t.sort_values("opened").reset_index(drop=True)
        days = pd.to_datetime(t["opened"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None).to_numpy()
        days = pd.DatetimeIndex(days)
        per_year = len(t) / ((days.max() - days.min()).days / 365.25)
        for risk in (100, 200, 400, 700, 1000):
            pnl = trades_usd(t, risk, ppt)
            hist = [evaluate(pnl[s:], days[s:]) for s in range(0, len(pnl) - 20, 5)]
            done = [h for h in hist if h[0] is not None]
            p_hist = np.mean([h[0] for h in done]) if done else np.nan
            d_hist = np.median([h[2] for h in done if h[0]]) if any(h[0] for h in done) else np.nan
            # shuffled: random trades, one trading day apart on average per the strategy's trade frequency
            sims = []
            for _ in range(10000):
                idx = RNG.integers(len(pnl), size=3000)
                gap = 252 / per_year * 365 / 252                   # calendar days between trades
                fake_days = pd.DatetimeIndex(np.datetime64("2020-01-01") +
                                             (np.cumsum(RNG.exponential(gap, size=3000))).astype("timedelta64[D]"))
                sims.append(evaluate(pnl[idx], fake_days))
            done_s = [s for s in sims if s[0] is not None]
            p_mc = np.mean([s[0] for s in done_s])
            d_mc = np.median([s[2] for s in done_s if s[0]]) if any(s[0] for s in done_s) else np.nan
            rows.append({"strategy": name, "trades/yr": per_year, "risk per trade": f"${risk}",
                         "$ per trade (avg)": pnl.mean(), "pass % (history)": p_hist, "median days (history)": d_hist,
                         "pass % (Monte Carlo)": p_mc, "median days (MC)": d_mc})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}" if abs(v) < 1 else f"{v:,.0f}"))
    print("\n(the race: ~41% of evaluations pass, most in 1-2 days; shown pass % ignore losses inside a trade, so the "
          "real figures are somewhat lower)")
    res.to_csv(ROOT / "data" / "mc_lucid_eval_candidates.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
