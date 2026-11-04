"""Several funded accounts at once (owner, 2026-10-09): one funded bot per account, each with its own
files, and the Funded page listing them all."""
from __future__ import annotations

import json
import sys

from test_funded_race import load_module
from test_trial_race_loop import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import dashboard  # noqa: E402


def test_the_first_account_keeps_its_files_and_others_get_their_own(monkeypatch):
    monkeypatch.setenv("NT_FUNDED_ACCOUNT", "DEMO9384168")
    monkeypatch.setattr(sys, "argv", ["run_lucid_futures_funded.py"])
    m = load_module("funded_primary")
    assert m.ACCOUNT == "DEMO9384168" and m.TAG == ""
    assert m.STATE_PATH.name == "state_lucid_futures_funded.json"
    assert m.TRADES_LOG_PATH.name == "lucid_futures_funded_trades.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_lucid_futures_funded.py", "--account", "LFE-50K 123"])
    m = load_module("funded_second")
    assert m.ACCOUNT == "LFE-50K 123" and m.TAG == "_LFE_50K_123"
    assert m.STATE_PATH.name == "state_lucid_futures_funded_LFE_50K_123.json"
    assert m.TRADES_LOG_PATH.name == "lucid_futures_funded_trades_LFE_50K_123.jsonl"
    assert m.BOOK == "Lucid-Funded LFE-50K 123"


def _state(account, **run):
    return {"account": account, "runs_total": 2, "runs_blown": 1,
            "run": {"realized_usd": 1046.5, "paid_usd": 0.0, "payouts": [], "tickets": {}, "cycle_pnl": 1046.5,
                    "cycle_best_day": 1046.5, "day_start_realized": 1046.5, **run},
            "history": [{"payout_list": [{"amount_usd": 500.0}]}]}


def test_the_funded_page_finds_every_account(tmp_path):
    (tmp_path / "state_lucid_futures_funded.json").write_text(json.dumps(_state("DEMO9384168")), encoding="utf-8")
    (tmp_path / "state_lucid_futures_funded_LFE1.json").write_text(
        json.dumps(_state("LFE1", tickets={"1": {}})), encoding="utf-8")
    (tmp_path / "state_lucid_futures_funded_roster.json").write_text("{}", encoding="utf-8")   # the old bot
    env = {"NT_FUNDED_ACCOUNT": "DEMO9384168", "LUCID_FUNDED_EXTRA_ACCOUNTS": "LFE1, LFE2"}
    insts = dashboard._funded_instances(tmp_path, env)
    assert [i["account"] for i in insts] == ["DEMO9384168", "LFE1", "LFE2"]
    assert insts[1]["trades_path"].name == "lucid_futures_funded_trades_LFE1.jsonl"
    rows = [dashboard._funded_overview(i) for i in insts]
    assert rows[0]["Balance"] == "$51,046" and rows[0]["Next payout needs"] == "$1,570"
    assert rows[0]["Payouts"] == 1 and rows[0]["Paid to you"] == "$450" and rows[0]["Blown"] == 1
    assert rows[1]["Status"] == "race running"
    assert rows[2]["Status"] == "not started yet" and rows[2]["Balance"] == "—"


def test_one_account_without_a_state_file_still_shows():
    insts = dashboard._funded_instances(ROOT / "tests" / "no_such_dir", {"NT_FUNDED_ACCOUNT": "Sim102"})
    assert [i["account"] for i in insts] == ["Sim102"]


def test_several_real_accounts_can_be_confirmed(monkeypatch):
    m = load_module("funded_live_lock")
    monkeypatch.setenv("LUCID_FUNDED_LIVE_CONFIRM", "LFE-1, LFE-2")
    assert m.live_confirmed("LFE-1") and m.live_confirmed("LFE-2") and not m.live_confirmed("LFE-3")
    monkeypatch.setenv("LUCID_FUNDED_LIVE_CONFIRM", "")
    assert not m.live_confirmed("LFE-1")
