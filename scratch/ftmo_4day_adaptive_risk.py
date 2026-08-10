"""Adaptive/path-dependent risk sizing -- the one untested lever left.

Every prior replay() in this investigation used a FIXED risk% for the
whole challenge. This asks a different question: does the SHAPE of the
risk path matter, not just its average level? Size each day's NEW trades
from the account's current distance to the nearest barrier (the +10%
target, or the -10% total drawdown -- the -5% daily limit resets every day
so it isn't informative at day-open sizing time):

    dist_to_target = (1 + target) - eq
    dist_to_total  = eq - (1 - TOTAL_LIM)
    nearest        = min(dist_to_target, dist_to_total)      [symmetric]
                     dist_to_total                            [loss-only]
                     dist_to_target                            [target-only]
    risk_today     = base_risk * clamp(nearest / band, floor, 1.0)

Trades keep the risk% they were opened with for their whole life (real
trading: you don't resize an open position mid-flight). Reuses the exact
same pooled CONFLUENCE 1.0R (RSI14+RSI7) trades from
scratch/cache_1d_trades.pkl -- no new backtesting, this only changes the
portfolio-replay sizing logic.

No Monte Carlo. Real daily bars already pooled, all three FTMO rules,
MIN_DAYS=4, walking every possible start date, horizon 4 AND 10 days.

Usage:
    .venv/Scripts/python.exe scratch/ftmo_4day_adaptive_risk.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\proksa\OneDrive\Počítač\New folder\ruflo\ruflo\metatraderbot")

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
ROSTER_KEY = "CONFLUENCE 1.0R (RSI14+RSI7)"


def flat_replay(by_day, days, i0, hz, risk, conc, per_day, target):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            op.append([tr, 0])
            opened += 1
        if opened:
            td += 1
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
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + target and td >= MIN_DAYS:
            return "pass"
    return "unresolved"


def adaptive_replay(by_day, days, i0, hz, base_risk, conc, per_day, target,
                     band, floor, mode):
    eq, real, op, td = 1.0, 0.0, [], 0
    for j in range(hz):
        i = i0 + j
        if i >= len(days):
            break
        d0 = eq
        dist_to_target = (1.0 + target) - eq
        dist_to_total = eq - (1.0 - TOTAL_LIM)
        if mode == "symmetric":
            nearest = min(dist_to_target, dist_to_total)
        elif mode == "loss_only":
            nearest = dist_to_total
        elif mode == "target_only":
            nearest = dist_to_target
        else:
            raise ValueError(mode)
        frac = max(floor, min(1.0, nearest / band))
        today_risk = base_risk * frac

        opened = 0
        for tr in by_day.get(days[i], []):
            if len(op) >= conc or opened >= per_day:
                break
            op.append([tr, 0, today_risk])
            opened += 1
        if opened:
            td += 1
        fl, still = 0.0, []
        for tr, age, r in op:
            age += 1
            if age < len(tr["path"]):
                fl += tr["path"][age] * r
                still.append([tr, age, r])
            else:
                real += tr["final"] * r
        op = still
        eq = 1.0 + real + fl
        if eq <= d0 - DAILY_LIM or eq <= 1.0 - TOTAL_LIM:
            return "fail"
        if eq >= 1.0 + target and td >= MIN_DAYS:
            return "pass"
    return "unresolved"


def pf(fn, *args):
    out = [fn(*a) for a in args[0]]
    n = len(out)
    if n == 0:
        return 0.0, 0.0, 0
    return out.count("pass") / n, out.count("fail") / n, n


def main() -> int:
    with open(ROOT / "scratch" / "cache_1d_trades.pkl", "rb") as f:
        by_roster = pickle.load(f)
    trades, by_day, days = by_roster[ROSTER_KEY]
    starts = list(range(0, len(days) - 1, 4))
    n_trades = len(trades)
    print("=" * 108)
    print(f"ADAPTIVE RISK SIZING -- {ROSTER_KEY}  (n={n_trades} pooled trades, "
          f"{len(starts)} start dates, conc=15, 6/day)")
    print("=" * 108)

    conc, per_day, target = 15, 6, 0.10
    base_risks = [0.05, 0.08, 0.10]
    variants = [
        ("FLAT (baseline)", None, None, None),
        ("GENTLE symmetric", "symmetric", 0.06, 0.5),
        ("AGGRESSIVE symmetric", "symmetric", 0.10, 0.15),
        ("LOSS-SIDE-ONLY taper", "loss_only", 0.06, 0.25),
        ("TARGET-ONLY taper", "target_only", 0.05, 0.30),
    ]

    all_results = []
    for horizon in (4, 10):
        print(f"\n{'-'*108}\nHORIZON = {horizon} days\n{'-'*108}")
        print(f"{'variant':<24}{'base_risk':>10}{'pass':>9}{'fail':>9}{'n':>7}")
        for vname, mode, band, floor in variants:
            for base_risk in base_risks:
                if mode is None:
                    args = [(by_day, days, i, horizon, base_risk, conc, per_day, target)
                            for i in starts]
                    p, f, n = pf(flat_replay, args)
                else:
                    args = [(by_day, days, i, horizon, base_risk, conc, per_day, target,
                              band, floor, mode) for i in starts]
                    p, f, n = pf(adaptive_replay, args)
                print(f"{vname:<24}{base_risk*100:>9.1f}%{p:>9.1%}{f:>9.1%}{n:>7}")
                all_results.append((horizon, vname, mode, band, floor, base_risk, p, f, n))

    print("\n" + "=" * 108)
    print("TOP 10 AT HORIZON=4 BY RAW PASS RATE:")
    h4 = sorted([r for r in all_results if r[0] == 4], key=lambda x: (-x[6], x[7]))[:10]
    for horizon, vname, mode, band, floor, base_risk, p, f, n in h4:
        print(f"  pass {p:>6.1%}  fail {f:>6.1%}  n={n:<6}  {vname:<24}  base_risk {base_risk*100:.1f}%"
              + (f"  band={band} floor={floor}" if mode else ""))

    print("\n" + "=" * 108)
    print("DEFENSIBLE (fail<=55%, horizon=4):")
    elig = [r for r in all_results if r[0] == 4 and r[7] <= 0.55]
    if elig:
        best = sorted(elig, key=lambda x: -x[6])[0]
        horizon, vname, mode, band, floor, base_risk, p, f, n = best
        print(f"  pass {p:.1%}  fail {f:.1%}  n={n}  {vname}  base_risk {base_risk*100:.1f}%"
              + (f"  band={band} floor={floor}" if mode else ""))

    # half-split validation: the horizon=4 raw-pass winner, the defensible
    # pick, AND anything that looks surprisingly good (>=35% pass) as an
    # explicit red-flag check against sample collapse / methodology bugs.
    print("\n" + "=" * 108)
    print("FIRST-HALF vs SECOND-HALF VALIDATION (horizon=4)")
    print("-" * 108)
    mid = len(days) // 2
    first_days, second_days = set(days[:mid]), set(days[mid:])
    starts_first = [i for i in range(0, len(days) - 1, 4) if days[i] in first_days]
    starts_second = [i for i in range(0, len(days) - 1, 4) if days[i] in second_days]

    to_check = {h4[0][:6]: h4[0]}
    if elig:
        to_check[best[:6]] = best
    flagged = [r for r in all_results if r[0] == 4 and r[6] >= 0.35]
    for r in flagged:
        to_check[r[:6]] = r
    for key, (horizon, vname, mode, band, floor, base_risk, p, f, n) in to_check.items():
        if mode is None:
            args1 = [(by_day, days, i, 4, base_risk, conc, per_day, target) for i in starts_first]
            args2 = [(by_day, days, i, 4, base_risk, conc, per_day, target) for i in starts_second]
            p1, f1, n1 = pf(flat_replay, args1)
            p2, f2, n2 = pf(flat_replay, args2)
        else:
            args1 = [(by_day, days, i, 4, base_risk, conc, per_day, target, band, floor, mode)
                     for i in starts_first]
            args2 = [(by_day, days, i, 4, base_risk, conc, per_day, target, band, floor, mode)
                     for i in starts_second]
            p1, f1, n1 = pf(adaptive_replay, args1)
            p2, f2, n2 = pf(adaptive_replay, args2)
        print(f"  [{vname}] base_risk {base_risk*100:.1f}%"
              + (f" band={band} floor={floor}" if mode else "") + f"  (full: pass {p:.1%} fail {f:.1%})")
        print(f"      1st half: pass {p1:>6.1%} fail {f1:>6.1%} (n={n1})   "
              f"2nd half: pass {p2:>6.1%} fail {f2:>6.1%} (n={n2})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
