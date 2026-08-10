"""10-day pass/fail for the winning 4-day-session configs, reusing the exact
same trade caches and replay() engine already validated for horizon=4 --
no new backtesting, just a different horizon over the same real trades.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

DAILY_LIM, TOTAL_LIM, MIN_DAYS = 0.05, 0.10, 4
TARGET = 0.10
HORIZONS = (4, 10)


def replay(by_day, days, i0, hz, risk, conc, per_day, target, cap_ccy=0):
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
            if cap_ccy:
                sym = tr["sym"]
                ccy = {sym[:3], sym[3:]}
                held = sum(1 for p, _ in op if ccy & {p["sym"][:3], p["sym"][3:]})
                if held >= cap_ccy:
                    continue
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


# (label, cache_file, key, risk, conc, per_day)
CANDIDATES = [
    ("CONFLUENCE 1.0R risk10/conc15 (4-day raw-pass winner)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.10, 15, 6),
    ("CONFLUENCE 1.0R risk6/conc15 (4-day defensible)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.06, 15, 6),
    ("CONFLUENCE 1.0R risk4/conc15 (4-day safest)", "cache_1d_trades.pkl",
     "CONFLUENCE 1.0R (RSI14+RSI7)", 0.04, 15, 6),
    ("sl=5 tp=1.0 risk5/conc15 (SL-tuned 4-day best)", "cache_sltp_trades.pkl",
     (5, 1.0), 0.05, 15, 6),
    ("CONF+Donchian20@2R risk10/conc15 (4-day best overall)", "cache_other_strategies.pkl",
     "CONF 1.0R + Donchian20@2R", 0.10, 15, 6),
    ("CONF+Donchian20@2R risk6/conc15 (4-day defensible)", "cache_other_strategies.pkl",
     "CONF 1.0R + Donchian20@2R", 0.06, 15, 6),
]

print("=" * 100)
print("10-DAY PASS/FAIL for the 4-day-session winning configs (same trades, same engine)")
print("=" * 100)
print(f"{'config':<52}{'4d pass':>9}{'4d fail':>9}{'10d pass':>10}{'10d fail':>10}{'n':>7}")
print("-" * 100)

_cache_mem = {}
for label, cache_file, key, risk, conc, per_day in CANDIDATES:
    if cache_file not in _cache_mem:
        with open(ROOT / cache_file, "rb") as f:
            _cache_mem[cache_file] = pickle.load(f)
    trades, by_day, days = _cache_mem[cache_file][key]
    starts = list(range(0, max(len(days) - 1, 1), 4))
    row = {}
    for hz in HORIZONS:
        out = [replay(by_day, days, i, hz, risk, conc, per_day, TARGET) for i in starts]
        n = len(out)
        row[hz] = (out.count("pass") / n, out.count("fail") / n, n)
    p4, f4, n4 = row[4]
    p10, f10, n10 = row[10]
    print(f"{label:<52}{p4:>9.1%}{f4:>9.1%}{p10:>10.1%}{f10:>10.1%}{n10:>7}")
