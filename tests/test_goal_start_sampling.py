"""Lock the `goal_start` sampling scheme.

Goal = a WELL-FRAMED frame (occupancy in range), start = a frame delta away that still
sees the subject, action = the immediate chunk_size steps toward the goal. The scheme
exists because the trajectory's terminal frame is a bad goal: these are random camera
motions, so the subject is framed mid-trajectory and the camera drifts past it (terminal
occupancy median 0, empty in 52% of all 42840 trajectories).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common.annotations import (
    DEFAULT_DELTA_RANGE,
    DEFAULT_GOAL_OCCUPANCY_RANGE,
    iter_goal_start_windows,
)

DATA = Path("data/trajectories/Abandoned-alley_9ee2b453__All-People-Are-Sisters_1795d425/data.json")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="v7 dataset not mounted")

CHUNK = 8


@pytest.fixture(scope="module")
def windows():
    """Scheme mechanics only — composition gates off, so one placement still yields
    enough windows to exercise delta / chunk / direction."""
    return list(iter_goal_start_windows(
        DATA, chunk_size=CHUNK, min_goal_body_in_frame=0.0,
        require_goal_center_on_screen=False,
    ))


@pytest.fixture(scope="module")
def framed_windows():
    """Default (production) gates: occupancy band + body-in-frame + centre on screen."""
    return list(iter_goal_start_windows(DATA, chunk_size=CHUNK))


def test_scheme_yields_windows(windows):
    assert len(windows) > 100


def test_every_goal_is_in_the_occupancy_band(windows):
    lo, hi = DEFAULT_GOAL_OCCUPANCY_RANGE
    for w in windows:
        assert lo <= float(w.goal_frame.raw["occupancy"]) <= hi


def test_composition_gate_keeps_only_well_framed_goals(framed_windows, windows):
    """Occupancy alone admits subjects hanging out of frame — these gates are what
    make a goal a photograph (median body_in_frame was 34 without them)."""
    assert 0 < len(framed_windows) < len(windows)
    for w in framed_windows:
        raw = w.goal_frame.raw
        assert float(raw["body_in_frame_ratio"]) >= 70.0
        assert 0.0 <= float(raw["object_center_x"]) <= 1024.0
        assert 0.0 <= float(raw["object_center_y"]) <= 768.0


def test_composition_gate_can_be_disabled(windows):
    loose_bodies = [float(w.goal_frame.raw["body_in_frame_ratio"]) for w in windows]
    assert min(loose_bodies) < 70.0          # the gate really is off in this fixture


def test_start_always_sees_the_subject(windows):
    """The policy must never be asked to act from a frame with no subject in it."""
    for w in windows:
        assert float(w.start.raw["occupancy"]) > 1.0


def test_delta_between_start_and_goal_is_in_range(windows):
    d_min, d_max = DEFAULT_DELTA_RANGE
    for w in windows:
        assert d_min <= abs(w.goal_frame.frame_idx - w.start_frame_idx) <= d_max


def test_chunk_is_contiguous_and_heads_toward_the_goal_without_overshooting(windows):
    for w in windows:
        assert len(w.keyframes) == CHUNK + 1
        assert w.frame_step == 1
        idxs = [k.frame_idx for k in w.keyframes]
        assert idxs == list(range(idxs[0], idxs[0] + w.direction * (CHUNK + 1), w.direction))
        # moving toward the goal, and stopping at or before it
        assert w.direction == (1 if w.goal_frame.frame_idx > w.start_frame_idx else -1)
        assert w.direction * (w.goal_frame.frame_idx - w.end_frame_idx) >= 0


def test_scheme_is_bidirectional(windows):
    directions = {w.direction for w in windows}
    assert directions == {1, -1}          # backward windows give dolly-out for free


def test_cap_is_deterministic_and_respected():
    kw = dict(chunk_size=CHUNK, max_per_pair=15, min_goal_body_in_frame=0.0,
              require_goal_center_on_screen=False)
    a = list(iter_goal_start_windows(DATA, **kw))
    b = list(iter_goal_start_windows(DATA, **kw))
    key = lambda ws: [(w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) for w in ws]
    assert key(a) == key(b)
    per_pair: dict[int, int] = {}
    for w in a:
        per_pair[w.pair_idx] = per_pair.get(w.pair_idx, 0) + 1
    assert max(per_pair.values()) <= 15


def test_delta_below_chunk_size_is_rejected():
    """delta < chunk_size would let the chunk run past the goal."""
    with pytest.raises(ValueError):
        next(iter_goal_start_windows(DATA, chunk_size=CHUNK, delta_range=(4, 32)))


def test_dataset_produces_9d_actions_and_a_subject_frame_goal():
    from src.common.dataset_base import BasePolicyDataset
    from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY

    ds = BasePolicyDataset(
        [str(DATA.parent)], chunk_size=CHUNK, sampling_scheme="goal_start",
        max_windows_per_pair=10,
    )
    assert len(ds) > 0
    s = ds[0]
    assert s.action_chunk.shape == (CHUNK, 9)
    bearing = s.goal_vec[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]
    assert 0.0 <= float(bearing) < 360.0


def test_bearing_balanced_weights_even_out_the_view_sectors():
    """The raw sector mix is heavily skewed (back ~31%, front ~4%, left ~1% measured
    over 400 placements), so the sampler weights rather than discards."""
    import numpy as np

    from src.common.dataset_base import BasePolicyDataset
    from src.common.facing import sector8

    ds = BasePolicyDataset(
        [str(DATA.parent)], chunk_size=CHUNK, sampling_scheme="goal_start",
        min_goal_body_in_frame=0.0, require_goal_center_on_screen=False,
        max_windows_per_pair=40,
    )
    w = ds.bearing_balanced_weights()
    assert w.shape == (len(ds),)
    assert w.mean() == pytest.approx(1.0, abs=1e-5)

    sectors = [sector8(float(b)) for b in ds.goal_bearings()]
    mass: dict[str, float] = {}
    for s, wt in zip(sectors, w):
        mass[s] = mass.get(s, 0.0) + float(wt)
    # every represented sector ends up with the same total weight
    assert max(mass.values()) == pytest.approx(min(mass.values()), rel=1e-5)
    assert np.isfinite(w).all()
