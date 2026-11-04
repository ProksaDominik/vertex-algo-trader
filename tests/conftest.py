"""Tests never use the owner's live settings for prices: paper mode on internet prices (VERTEX_PAPER=1 in .env
since 2026-10-15) would make them download from Yahoo. A test that wants paper mode sets it itself."""
import pytest


@pytest.fixture(autouse=True)
def _no_paper_mode_from_env(monkeypatch):
    monkeypatch.setenv("VERTEX_PAPER", "0")
    monkeypatch.delenv("VERTEX_PRICE_SOURCE", raising=False)
