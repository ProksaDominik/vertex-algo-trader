"""The dashboard notices NinjaTrader's own sign-in failing (2026-10-08/09: it retried every 30 s all night
and NinjaTrader locked the login)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dashboard  # noqa: E402


def _log(tmp_path, lines):
    d = tmp_path / "Documents" / "NinjaTrader 8" / "log"
    d.mkdir(parents=True)
    (d / "log.20261009.00000.en.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _at(minutes_ago):
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S:000")


def test_recent_sign_in_failures_are_counted(monkeypatch, tmp_path):
    msg = "|3|2|There was a problem authenticating account Google Simulation. Please try reconnecting."
    _log(tmp_path, [_at(40) + msg, _at(3) + msg, _at(2) + msg, _at(1) + msg, _at(1) + "|1|2|Simulation: Connected"])
    monkeypatch.setattr(dashboard.Path, "home", staticmethod(lambda: tmp_path))
    assert dashboard._nt_login_failures() == 3            # the one 40 minutes ago is too old
    assert dashboard._nt_login_failures(60) == 4


def test_a_quiet_log_has_no_failures(monkeypatch, tmp_path):
    _log(tmp_path, [_at(5) + "|2|4|Session Break (Version 8.1.8.3)"])
    monkeypatch.setattr(dashboard.Path, "home", staticmethod(lambda: tmp_path))
    assert dashboard._nt_login_failures() == 0
