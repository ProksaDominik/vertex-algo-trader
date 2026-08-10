"""Is 60% pass-in-10-days possible for ANY strategy under FTMO's rules?

This stops searching and asks the structural question instead. Model equity as
a random walk with drift mu and daily volatility sigma, and impose FTMO's
three conditions exactly:

    +10%  target   (absorbing, good)
    -10%  overall  (absorbing, fatal)
    -5%   in ONE day (absorbing, fatal)  <-- the asymmetric one

The first two are symmetric barriers: with enough positive drift, P(hit +10%
first) rises above 50% smoothly. The third has NO upside counterpart. There is
no "make 5% in a day and you are through" rule. That asymmetry is what caps
the achievable pass rate, and it does so in a specific way:

    to reach +10% within 10 days you need roughly 1%/day of drift;
    to produce 1%/day of drift you need volatility;
    volatility large enough to do that makes a -5% day common.

This sweeps (mu, sigma) over the whole plane -- far beyond anything achievable
by a real strategy -- and reports the best pass rate available at each point.
If 60%/10-day requires a (mu, sigma) that also implies constant daily
breaches, then no strategy reaches it and further searching is pointless.

Usage:
    python scripts/theoretical_ceiling.py
"""
from __future__ import annotations

import numpy as np

N = 40_000
RNG = np.random.default_rng(20260805)
TARGET, TOTAL_LIM, DAILY_LIM = 0.10, 0.10, 0.05


def pass_rate(mu_daily, sigma_daily, days, daily_rule=True, n=N):
    """Fraction of paths hitting +10% before any fatal condition."""
    eq = np.zeros(n)
    st = np.zeros(n, np.int8)          # 0 live, 1 pass, 2 fail
    for _ in range(days):
        step = RNG.normal(mu_daily, sigma_daily, n)
        live = st == 0
        new = eq + step
        # daily rule: a single day worse than -5% is fatal
        if daily_rule:
            st[live & (step <= -DAILY_LIM)] = 2
        eq = np.where(live, new, eq)
        live = st == 0
        st[live & (eq <= -TOTAL_LIM)] = 2
        live = st == 0
        st[live & (eq >= TARGET)] = 1
    return (st == 1).mean()


print("=" * 96)
print("THEORETICAL CEILING — best possible 10-day pass rate vs (drift, vol)")
print("=" * 96)
print("Sweeping FAR beyond any real strategy. mu = daily drift, sigma = daily")
print("volatility, both as fractions of equity.\n")

sigmas = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
mus = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]

print(f"  {'mu\\sigma':>10}" + "".join(f"{s*100:>9.1f}%" for s in sigmas))
print("-" * 96)
best = (0.0, None)
for mu in mus:
    row = []
    for sg in sigmas:
        p = pass_rate(mu, sg, 10)
        row.append(p)
        if p > best[0]:
            best = (p, (mu, sg))
    print(f"  {mu*100:>9.1f}%" + "".join(f"{p:>9.1%}" for p in row))

print(f"\n  best 10-day pass anywhere on this grid: {best[0]:.1%} "
      f"at mu={best[1][0]*100:.1f}%/day, sigma={best[1][1]*100:.1f}%/day")

print("\n" + "=" * 96)
print("WHAT THE DAILY RULE COSTS — same grid, 5% daily rule REMOVED")
print("=" * 96)
print(f"  {'mu\\sigma':>10}" + "".join(f"{s*100:>9.1f}%" for s in sigmas))
print("-" * 96)
for mu in mus:
    row = [pass_rate(mu, sg, 10, daily_rule=False) for sg in sigmas]
    print(f"  {mu*100:>9.1f}%" + "".join(f"{p:>9.1%}" for p in row))

print("\n" + "=" * 96)
print("THE SQUEEZE, STATED DIRECTLY")
print("=" * 96)
for sg in (0.005, 0.01, 0.02, 0.03):
    # drift needed to average +10% in 10 days
    need_mu = TARGET / 10
    p_daily_breach = 1 - (1 - RNG.normal(0, 1, 1).size * 0)  # placeholder
    from math import erf, sqrt
    z = (-DAILY_LIM - need_mu) / sg
    p_breach = 0.5 * (1 + erf(z / sqrt(2)))
    p_survive10 = (1 - p_breach) ** 10
    print(f"  sigma {sg*100:>4.1f}%/day: to average +10% in 10 days needs "
          f"mu={need_mu*100:.1f}%/day")
    print(f"                  P(a -5% day) = {p_breach:.2%} -> "
          f"P(surviving 10 days) = {p_survive10:.1%}")

print("\n  The tension is structural: the drift required to finish in 10 days")
print("  demands volatility, and that volatility makes a -5% day likely. No")
print("  choice of strategy escapes it -- only a longer deadline does.")

print("\n" + "=" * 96)
print("SAME GRID, 30-DAY DEADLINE (drift requirement is 3x lower)")
print("=" * 96)
print(f"  {'mu\\sigma':>10}" + "".join(f"{s*100:>9.1f}%" for s in sigmas))
print("-" * 96)
best30 = (0.0, None)
for mu in mus:
    row = []
    for sg in sigmas:
        p = pass_rate(mu, sg, 30)
        row.append(p)
        if p > best30[0]:
            best30 = (p, (mu, sg))
    print(f"  {mu*100:>9.1f}%" + "".join(f"{p:>9.1%}" for p in row))
print(f"\n  best 30-day pass on this grid: {best30[0]:.1%} "
      f"at mu={best30[1][0]*100:.1f}%/day, sigma={best30[1][1]*100:.1f}%/day")
