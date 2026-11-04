"""Time-based strategies must signal on LIVE bars, not only in backtests.

Found 2026-09-29: ORB, VWAP reversion and time-of-day reversion read
bars["timestamp"], a column only metatraderbot.backtest creates. Live bars
(mt5_engine / nt_engine) carry the time in the INDEX, so on live data ORB
raised (swallowed by the lab) and the other two returned nothing -- none of the
three ever produced a live signal. Each is now fed the same bars both ways and
must produce the same signals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.lab_registry import instantiate  # noqa: E402


def _bars(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """15m bars with enough movement to trigger breakouts and stretches."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="15min", tz="UTC")
    close = 5000 + np.cumsum(rng.normal(0, 4, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.5, 6, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 6, n)
    vol = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": vol}, index=pd.Index(idx, name="timestamp"))


def _signals(family: str, params: dict, bars: pd.DataFrame) -> list[tuple]:
    strat = instantiate(family, {family: params}, 1.0, "MES")
    out = []
    for i in range(30, len(bars)):
        for s in strat.on_bar(i, bars) or []:
            out.append((i, s.direction.value, round(float(s.stop_loss), 6)))
    return out


@pytest.mark.parametrize("family,params", [
    ("orb", {"range_start_hour_utc": 13, "range_minutes": 30}),
    ("vwap", {}),
    ("tod_reversion", {}),
])
def test_live_bars_signal_exactly_like_backtest_bars(family, params):
    live = _bars()                                   # time in the index, like the engines
    backtest = live.reset_index()                    # time in a column, like Backtest
    live_sigs = _signals(family, params, live)
    bt_sigs = _signals(family, params, backtest)
    assert bt_sigs, f"{family}: the fixture produced no backtest signals to compare"
    assert live_sigs == bt_sigs
