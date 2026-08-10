"""What it actually takes to (a) pass FTMO in ONE month and (b) make large
weekly returns on the $200 book — priced honestly, at the measured edge.

Both goals are frequency-of-outcome questions, so both are answered as
probabilities rather than as a single "expected profit" number that hides the
failure branch.

FTMO is simulated against its REAL rule set, day by day, not as a smooth
compounding curve:
  - +10% of initial  -> PASS (evaluation ends)
  - -5% from that DAY's starting equity at any point -> FAIL (daily rule)
  - -10% from initial at any point -> FAIL (overall rule)
  - a 21-trading-day deadline: not passing inside it counts as NOT passed

The daily rule is what makes raising risk stop working: a bigger stake makes
the target reachable faster AND makes a single bad day fatal, so P(pass)
rises, peaks, then collapses. Finding that peak is the whole point.

The script also inverts the question: at each risk level, what per-trade
expectancy WOULD be needed to make passing-in-a-month more likely than not.
That number is the honest answer to "can I just turn the risk up?".

Usage:
    python scripts/aggressive_targets.py
"""
from __future__ import annotations

import numpy as np

N_SIMS = 30_000
TRADING_DAYS = 21
RNG = np.random.default_rng(23)

# Measured out-of-sample edges (backtest/canonical_majors.json, 10.5y).
SCENARIOS = {
    "RSI daily only  (E=+0.195R)": {"win": 0.612, "rr": 0.94, "per_day": 0.56},
    "Daily stack     (E=+0.060R)": {"win": 0.534, "rr": 1.00, "per_day": 3.11},
    "IF the edge is actually ZERO": {"win": 0.500, "rr": 1.00, "per_day": 3.11},
}

RISKS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]


def sim_ftmo(win: float, rr: float, per_day: float, risk: float,
             days: int = TRADING_DAYS, n: int = N_SIMS) -> dict:
    """Day-by-day FTMO simulation. Returns outcome probabilities."""
    init = 10_000.0
    eq = np.full(n, init)
    state = np.zeros(n, dtype=np.int8)  # 0 live, 1 pass, 2 fail daily, 3 fail total

    # Trades land unevenly; a Poisson day count reproduces the clustering that
    # makes the daily rule bite (several trades on one day, none the next).
    for _ in range(days):
        day_start = eq.copy()
        k = RNG.poisson(per_day, n)
        for t in range(int(k.max()) if k.max() > 0 else 0):
            live = state == 0
            acts = live & (k > t)
            if not acts.any():
                continue
            stake = eq[acts] * risk
            wins = RNG.random(acts.sum()) < win
            eq[acts] = eq[acts] + np.where(wins, stake * rr, -stake)

            idx = np.where(acts)[0]
            # Order matters: a blown account is a fail even if it later recovers.
            hit_daily = eq[idx] <= day_start[idx] * 0.95
            state[idx[hit_daily & (state[idx] == 0)]] = 2
            hit_total = eq[idx] <= init * 0.90
            state[idx[hit_total & (state[idx] == 0)]] = 3
            hit_pass = eq[idx] >= init * 1.10
            state[idx[hit_pass & (state[idx] == 0)]] = 1

    return {
        "pass": (state == 1).mean(),
        "fail_daily": (state == 2).mean(),
        "fail_total": (state == 3).mean(),
        "no_result": (state == 0).mean(),
        "median_eq": float(np.median(eq)),
    }


def required_edge(risk: float, per_day: float, days: int = TRADING_DAYS) -> float | None:
    """Smallest win-rate (at RR=1) making P(pass in a month) >= 50%."""
    for win in np.arange(0.50, 0.86, 0.01):
        r = sim_ftmo(float(win), 1.0, per_day, risk, days, n=6000)
        if r["pass"] >= 0.50:
            # E_net in R for RR=1: win*1 - (1-win)*1
            return float(win * 1.0 - (1 - win))
    return None


def sim_weekly(win: float, rr: float, per_day: float, risk: float,
               start: float = 200.0, n: int = N_SIMS) -> dict:
    """One WEEK (5 trading days) on the $200 book."""
    eq = np.full(n, start)
    n_trades = max(1, round(per_day * 5))
    for _ in range(n_trades):
        stake = eq * risk
        wins = RNG.random(n) < win
        eq = np.maximum(eq + np.where(wins, stake * rr, -stake), 0.0)
    ret = (eq - start) / start * 100
    return {
        "median": float(np.median(ret)),
        "mean": float(ret.mean()),
        "p5": float(np.percentile(ret, 5)),
        "p95": float(np.percentile(ret, 95)),
        "p_profit": float((ret > 0).mean()),
        "p_half": float((ret <= -50).mean()),
        "n_trades": n_trades,
    }


def main() -> int:
    print("=" * 100)
    print("GOAL 1 — PASS FTMO $10k WITHIN ONE MONTH (21 trading days)")
    print("Simulated against the real rules: +10% target, -5% daily, -10% overall")
    print("=" * 100)

    for label, e in SCENARIOS.items():
        print(f"\n### {label}   ~{e['per_day']:.2f} trades/day")
        print(f"{'risk/trade':>11}{'P(PASS in month)':>19}{'P(fail daily)':>15}"
              f"{'P(fail total)':>15}{'P(no result)':>14}")
        print("-" * 100)
        best = (0.0, None)
        for risk in RISKS:
            r = sim_ftmo(e["win"], e["rr"], e["per_day"], risk)
            if r["pass"] > best[0]:
                best = (r["pass"], risk)
            print(f"{risk*100:>10.1f}%{r['pass']:>18.1%}{r['fail_daily']:>15.1%}"
                  f"{r['fail_total']:>15.1%}{r['no_result']:>14.1%}")
        print(f"    >>> best achievable P(pass in a month) = {best[0]:.1%} "
              f"at {best[1]*100:.1f}% risk/trade")

    print("\n" + "=" * 100)
    print("THE INVERSE QUESTION — what edge would be NEEDED to pass in a month?")
    print("(minimum win rate at 1:1 reward:risk for P(pass) >= 50%)")
    print("=" * 100)
    print(f"{'risk/trade':>11}{'required E_net':>17}{'= win rate at 1:1':>21}{'vs measured':>16}")
    print("-" * 100)
    for risk in [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]:
        ed = required_edge(risk, 3.11)
        if ed is None:
            print(f"{risk*100:>10.1f}%{'unreachable':>17}{'even at 85% win':>21}{'':>16}")
        else:
            wr = (ed + 1) / 2
            print(f"{risk*100:>10.1f}%{ed:>+17.3f}R{wr:>20.1%}"
                  f"{'measured +0.060R':>16}")

    print("\n" + "=" * 100)
    print("GOAL 2 — BIG WEEKLY RETURNS ON THE $200 YOLO BOOK")
    print("One week = 5 trading days, daily stack (~3.1 trades/day)")
    print("=" * 100)
    e = SCENARIOS["Daily stack     (E=+0.060R)"]
    print(f"{'risk/trade':>11}{'median wk':>12}{'mean wk':>11}{'5th pct':>11}"
          f"{'95th pct':>11}{'P(up week)':>13}{'P(lose half)':>14}")
    print("-" * 100)
    for risk in [0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        r = sim_weekly(e["win"], e["rr"], e["per_day"], risk)
        print(f"{risk*100:>10.1f}%{r['median']:>11.1f}%{r['mean']:>10.1f}%"
              f"{r['p5']:>10.1f}%{r['p95']:>10.1f}%{r['p_profit']:>12.1%}"
              f"{r['p_half']:>13.1%}")

    print("\nSame, but if the edge is actually zero (the scenario this repo's")
    print("history keeps landing on):")
    print(f"{'risk/trade':>11}{'median wk':>12}{'mean wk':>11}{'5th pct':>11}"
          f"{'95th pct':>11}{'P(up week)':>13}{'P(lose half)':>14}")
    print("-" * 100)
    for risk in [0.05, 0.10, 0.20, 0.30, 0.40]:
        r = sim_weekly(0.50, 1.0, 3.11, risk)
        print(f"{risk*100:>10.1f}%{r['median']:>11.1f}%{r['mean']:>10.1f}%"
              f"{r['p5']:>10.1f}%{r['p95']:>10.1f}%{r['p_profit']:>12.1%}"
              f"{r['p_half']:>13.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
