"""'If the chart is in this state, it will probably go up / down' -- tested literally (2026-10-07).

Owner: "if the graph is in a certain state we need a strategy that will predict it on the
certain parameters". Two direct versions, both chosen ONLY from 2016-2022 and judged on
2023-2026 (days they never saw), scored on the race (2 NQ, +75.5 / -50 pts, out 16:00 ET):

  1. Chart states: every day at 19:00 ET falls into one state made of
       trend      up / down / sideways (price vs its 20- and 50-day averages)
       volatility calm / wild (ATR vs the past year)
       yesterday  NQ's session up / down
       evening    16:00 -> 19:00 move up / down
     (3 x 2 x 2 x 2 = 24 states). For each state, 2016-2022 decides: race long, race short,
     whichever won more often (long when unsure).
  2. Look-alike charts (nearest neighbours): the 64 fifteen-minute bars before 19:00,
     normalised; for each day the 25 most similar earlier days (only days before it) vote
     long or short by which race won.

Usage:  .venv/Scripts/python.exe scripts/lucid_state_pattern.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lucid_pass_research as lp  # noqa: E402
import lucid_signal_search as ss  # noqa: E402

A19 = lp.S(19)


def win(side, L, S_, m):
    y = np.where(side > 0, L, np.where(side < 0, S_, np.nan))
    k = m & ~np.isnan(y)
    return float(np.nanmean(y[k])), int(k.sum())


def main() -> int:
    g = lp.load_grid("NQ")
    idx = pd.to_datetime(g.dates)
    yrs = np.asarray(idx.year)
    lab = ss.race_labels(g, A19)
    L, S_ = lab[1], lab[-1]
    d = g.daily.reindex(idx)
    c = d["close"].ffill()
    sma20, sma50 = c.rolling(20).mean().shift(1), c.rolling(50).mean().shift(1)
    pc = c.shift(1)
    trend = np.where((pc > sma20) & (sma20 > sma50), 1, np.where((pc < sma20) & (sma20 < sma50), -1, 0))
    atr = d["atr14"] / d["prev_close"]
    vol = (atr > atr.rolling(252, min_periods=120).median().shift(1)).astype(int).to_numpy()
    yday = (d["prev_rth_ret"] > 0).astype(int).to_numpy()
    eve = (pd.Series(g.O[:, A19], index=idx) > d["prev_close"]).astype(int).to_numpy()
    state = (trend + 1) * 8 + vol * 4 + yday * 2 + eve
    ok = ~np.isnan(atr.to_numpy()) & ~np.isnan(sma50.to_numpy())
    A, B = (yrs <= 2022) & ok, (yrs >= 2023) & ok
    names = {-1: "down-trend", 0: "sideways", 1: "up-trend"}
    rows, rule = [], {}
    for s_ in range(24):
        m = A & (state == s_)
        wl, nl = win(np.ones(len(idx)), L, S_, m)
        ws, _ = win(-np.ones(len(idx)), L, S_, m)
        side = -1 if (nl >= 20 and ws > wl + 0.05) else 1
        rule[s_] = side
        mb = B & (state == s_)
        bl, nb = win(np.ones(len(idx)), L, S_, mb)
        bs, _ = win(-np.ones(len(idx)), L, S_, mb)
        rows.append({"state": f"{names[s_ // 8 - 1]}, {'wild' if (s_ // 4) % 2 else 'calm'}, yesterday "
                              f"{'up' if (s_ // 2) % 2 else 'down'}, evening {'up' if s_ % 2 else 'down'}",
                     "2016-22 days": nl, "buy won": wl, "sell won": ws, "chosen": "SELL" if side < 0 else "buy",
                     "2023-26 days": nb, "2023-26 buy won": bl, "2023-26 sell won": bs})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    side = np.array([rule[s_] for s_ in state])
    print(f"\n1. chart-state rule: 2016-22 (where it learned) {win(side, L, S_, A)[0]:.1%} vs always buy "
          f"{win(np.ones(len(idx)), L, S_, A)[0]:.1%} | 2023-26 (never seen) {win(side, L, S_, B)[0]:.1%} vs "
          f"always buy {win(np.ones(len(idx)), L, S_, B)[0]:.1%}")
    # 2. look-alike charts
    flat = pd.Series(g.C.ravel()).ffill().to_numpy()
    n = len(idx)
    X = np.full((n, 64), np.nan)
    for i in range(n):
        j = i * lp.K + A19
        if j >= 64:
            w = flat[j - 64:j]
            if np.all(np.isfinite(w)) and w[-1] > 0:
                X[i] = (w / w[-1] - 1) * 100
    X = X - np.nanmean(X, axis=1, keepdims=True)
    good = ~np.isnan(X).any(1)
    winner = np.where(L == 1, 1, np.where(S_ == 1, -1, 0))
    pred = np.zeros(n, int)
    for i in np.flatnonzero(good & (yrs >= 2018)):
        past = np.flatnonzero(good[:i - 1] & (winner[:i - 1] != 0))
        if len(past) < 200:
            continue
        dist = np.sqrt(((X[past] - X[i]) ** 2).sum(1))
        votes = winner[past[np.argsort(dist)[:25]]]
        pred[i] = 1 if votes.sum() >= 0 else -1
    m18 = (yrs >= 2018) & (pred != 0)
    pa, pb = m18 & (yrs <= 2022), m18 & (yrs >= 2023)
    print(f"2. look-alike charts: 2018-22 {win(pred, L, S_, pa)[0]:.1%} vs always buy {win(np.ones(n), L, S_, pa)[0]:.1%}"
          f" | 2023-26 {win(pred, L, S_, pb)[0]:.1%} vs always buy {win(np.ones(n), L, S_, pb)[0]:.1%}"
          f" | went short on {np.mean(pred[m18] < 0):.0%} of days")
    print("\nfull evaluation replays (2019-22 / 2023-26):")
    for name, sd in (("always buy (current)", np.ones(n, int)), ("chart-state rule", side),
                     ("look-alike charts", np.where(pred == 0, 1, pred))):
        r = ss.replays(g, A19, sd)
        print(f"  {name:22s} eval pass {r['eval pass 19-22']:.1%} / {r['eval pass 23-26']:.1%} | funded $ to you "
              f"{r['funded $ to you 19-22']:,.0f} / {r['funded $ to you 23-26']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
