"""Pure retune-step logic — bounds, floor, and max-retune cap."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metatraderbot.auto_retune import RetuneState, next_tp_r_multiple  # noqa: E402


def test_first_retune_shrinks_by_step_pct():
    state = RetuneState(original_tp_r=3.0, current_tp_r=3.0)
    new_tp, applied = next_tp_r_multiple(state, step_pct=0.20, floor_frac=0.30, max_retunes=5)
    assert applied is True
    assert new_tp == 3.0 * 0.80


def test_steps_shrink_further_each_time_it_compounds_down():
    state = RetuneState(original_tp_r=3.0, current_tp_r=3.0)
    first, _ = next_tp_r_multiple(state, step_pct=0.20, floor_frac=0.30, max_retunes=5)
    state.current_tp_r = first
    state.retune_count = 1
    second, _ = next_tp_r_multiple(state, step_pct=0.20, floor_frac=0.30, max_retunes=5)
    # Absolute drop from step 2 is smaller than step 1's, since it's 20% of a
    # smaller current value each time (compounding down, not linear).
    assert (first - second) < (3.0 - first)


def test_never_drops_below_floor_frac_of_original():
    state = RetuneState(original_tp_r=3.0, current_tp_r=1.0)  # already near the 0.9 floor
    new_tp, applied = next_tp_r_multiple(state, step_pct=0.50, floor_frac=0.30, max_retunes=5)
    assert applied is True
    assert new_tp == 3.0 * 0.30  # clamped to the floor, not 0.5


def test_no_further_change_once_floor_already_reached():
    floor = 3.0 * 0.30
    state = RetuneState(original_tp_r=3.0, current_tp_r=floor, retune_count=2)
    new_tp, applied = next_tp_r_multiple(state, step_pct=0.20, floor_frac=0.30, max_retunes=5)
    assert applied is False
    assert new_tp == floor


def test_stops_applying_once_max_retunes_reached():
    state = RetuneState(original_tp_r=3.0, current_tp_r=2.0, retune_count=5)
    new_tp, applied = next_tp_r_multiple(state, step_pct=0.20, floor_frac=0.30, max_retunes=5)
    assert applied is False
    assert new_tp == 2.0  # unchanged
