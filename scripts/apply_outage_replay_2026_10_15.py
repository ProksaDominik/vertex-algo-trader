"""One-off correction approved by the owner on 2026-10-15 ("make it look like nothing happened ... add those evals
with their outcome"; then "okay so now we go without NinjaTrader").

From 14 Oct 08:18 Prague NinjaTrader had no live market data (its free real-time data had run out). The trial
bot could not race (a leftover position, then a frozen price) and on 15 Oct 10:04-10:33 it booked 13 FAKE
failed evaluations (#64-#76) on the frozen price. This puts the bots' records where they would be had the data
never stopped: the fake evaluations go, and the races the bots would have run -- computed by the real bot code
on the real 1-minute NQ bars (scripts/soak_replay.py), one race a day at 19:00 New York -- go in, each marked
"replayed -- NinjaTrader had no live data". The records before the change are in data/backup_2026-10-15_fake_races.

Usage (bots stopped): .venv/Scripts/python.exe scripts/apply_outage_replay_2026_10_15.py <replay dir>
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTE = "replayed -- NinjaTrader had no live data (14-15 Oct 2026)"
REPLAY_FROM = {"trial": "2026-10-14T06:19", "funded": "2026-10-14T22:00"}


def apply(bot: str, src: Path, state_name: str, trades_name: str) -> None:
    assert (DATA / "backup_2026-10-15_fake_races" / state_name).exists(), "backup missing"
    new_trades = []
    for line in (src / "trades.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        if (t.get("opened_ts") or "") >= REPLAY_FROM[bot]:
            t["note"] = NOTE
        new_trades.append(json.dumps(t))
    state = json.loads((src / "state.json").read_text(encoding="utf-8"))
    for h in state.get("history") or []:
        if (h.get("resolved_ts") or h.get("end") or "") >= REPLAY_FROM[bot]:
            h["note"] = NOTE
    tmp = DATA / (state_name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(DATA / state_name)
    (DATA / trades_name).write_text("\n".join(new_trades) + "\n", encoding="utf-8")
    print(f"{bot}: records replaced from {src}")


if __name__ == "__main__":
    replay = Path(sys.argv[1])
    apply("trial", replay / "backfill_trial", "state_lucid_futures.json", "lucid_futures_trades.jsonl")
    apply("funded", replay / "backfill_funded", "state_lucid_futures_funded.json", "lucid_futures_funded_trades.jsonl")
    shutil.copytree(replay / "backfill_trial", DATA / "backup_2026-10-15_fake_races" / "replay_trial", dirs_exist_ok=True)
    shutil.copytree(replay / "backfill_funded", DATA / "backup_2026-10-15_fake_races" / "replay_funded", dirs_exist_ok=True)
