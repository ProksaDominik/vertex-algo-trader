import pickle
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\proksa\OneDrive\Počítač\New folder\ruflo\ruflo\metatraderbot")
sys.path.insert(0, str(ROOT / "scratch"))
from ftmo_4day_sltp_sweep import pass_fail  # noqa: E402

with open(ROOT / "scratch" / "cache_sltp_trades.pkl", "rb") as f:
    by_combo = pickle.load(f)

for key, (risk, conc, per_day) in [((5, 1.0), (0.05, 15, 6)), ((10, 1.0), (0.05, 15, 6))]:
    trades, by_day, days = by_combo[key]
    mid = len(days) // 2
    first_days, second_days = set(days[:mid]), set(days[mid:])
    starts_first = [i for i in range(0, len(days) - 1, 4) if days[i] in first_days]
    starts_second = [i for i in range(0, len(days) - 1, 4) if days[i] in second_days]
    starts_all = range(0, len(days) - 1, 4)
    pf_all = pass_fail(by_day, days, starts_all, risk, conc, per_day, 0.10)
    pf1 = pass_fail(by_day, days, starts_first, risk, conc, per_day, 0.10)
    pf2 = pass_fail(by_day, days, starts_second, risk, conc, per_day, 0.10)
    print(f"sl={key[0]} tp={key[1]} risk={risk*100:.0f}% conc={conc} {per_day}/day  n={len(trades)}")
    print(f"  full:  pass {pf_all[0]:.1%} fail {pf_all[1]:.1%} (n_starts={pf_all[2]})")
    print(f"  1st half: pass {pf1[0]:.1%} fail {pf1[1]:.1%} (n={pf1[2]})   "
          f"2nd half: pass {pf2[0]:.1%} fail {pf2[1]:.1%} (n={pf2[2]})")
