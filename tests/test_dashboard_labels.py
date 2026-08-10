"""Regression: strategy_label() used a free substring search over the whole
lowercased config_name, so "TrendPullbackBOS__XAUUSD" (which contains
"...bos__...") matched the "bos_" keyword and got mislabeled as "BOS" --
indistinguishable on the dashboard from the old, since-removed
MarketOpenBOS strategy. Matching must be scoped to the leading
"Strategy__SYMBOL" segment (or the whole legacy underscore-delimited name,
when there's no "__"), not the full string.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import dashboard  # noqa: E402


def test_new_strategy_names_are_not_swallowed_by_old_keywords():
    assert dashboard.strategy_label("TrendPullbackBOS__XAUUSD") == "TrendPullbackBOS"
    assert dashboard.strategy_label("SRRetest__EURUSD") == "SRRetest"
    assert dashboard.strategy_label("EMACross__GBPUSD") == "EMACross"


def test_old_and_legacy_labels_still_resolve():
    assert dashboard.strategy_label("MarketOpenBOS__DE40") == "BOS"
    assert dashboard.strategy_label("Donchian__XAUUSD") == "Donchian"
    assert dashboard.strategy_label("euraud_donchian_30m_3r") == "Donchian"
