"""Lock the subject-frame goal: world azimuth -> per-asset bearing.

The sign convention is the load-bearing part — bearing 90 must mean the camera is on
the SUBJECT'S RIGHT (verified geometrically and against turntable renders), and an
asset missing from the facing map must yield NaN so the sample is dropped rather than
silently falling back to the ambiguous world angle.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.facing import (
    front_azimuth,
    load_facing_map,
    sector3,
    sector8,
    subject_bearing_deg,
    world_azimuth_deg,
)
from src.common.goal_space import (
    CYCLIC_GOAL_KEYS,
    DEFAULT_GOAL_KEYS,
    SUBJECT_BEARING_KEY,
    WORLD_AZIMUTH_KEY,
    goal_vector,
)

SOME_ASSET = "All-People-Are-Sisters_1795d425"


def test_goal_space_uses_subject_bearing_not_world_azimuth():
    assert SUBJECT_BEARING_KEY in DEFAULT_GOAL_KEYS
    assert WORLD_AZIMUTH_KEY not in DEFAULT_GOAL_KEYS
    assert SUBJECT_BEARING_KEY in CYCLIC_GOAL_KEYS
    assert len(DEFAULT_GOAL_KEYS) == 8


def test_facing_map_covers_the_library():
    fmap = load_facing_map()
    assert len(fmap) >= 100
    assert all("front_az" in v for v in fmap.values())


def test_bearing_convention_front_right_back_left():
    front = front_azimuth(SOME_ASSET)
    assert front is not None
    # bearing = front_az - az, so az == front_az is the front view.
    assert subject_bearing_deg(front, SOME_ASSET) == pytest.approx(0.0, abs=1e-6)
    assert sector8(subject_bearing_deg(front, SOME_ASSET)) == "front"
    # camera 90 deg BEFORE the front azimuth sees the subject's RIGHT
    assert sector8(subject_bearing_deg(front - 90.0, SOME_ASSET)) == "right"
    assert sector8(subject_bearing_deg(front + 180.0, SOME_ASSET)) == "back"
    assert sector8(subject_bearing_deg(front + 90.0, SOME_ASSET)) == "left"


def test_bearing_wraps_into_0_360():
    for az in (-720.0, -1.0, 0.0, 359.0, 721.0):
        b = subject_bearing_deg(az, SOME_ASSET)
        assert 0.0 <= b < 360.0


def test_world_azimuth_is_the_inverse_of_bearing():
    for az in (0.0, 37.5, 180.0, 359.9):
        b = subject_bearing_deg(az, SOME_ASSET)
        assert world_azimuth_deg(b, SOME_ASSET) == pytest.approx(az % 360.0, abs=1e-6)


def test_sector3_is_symmetric_so_a_mirrored_asset_cannot_break_it():
    for off in (10.0, 44.0, 80.0, 150.0):
        assert sector3(off) == sector3(-off % 360.0)


@pytest.mark.parametrize("bearing,expected", [
    (0.0, "front"), (30.0, "front"), (90.0, "side"),
    (135.1, "back"), (180.0, "back"), (270.0, "side"),
])
def test_sector3_bins(bearing, expected):
    assert sector3(bearing) == expected


def test_unmapped_asset_yields_nan_so_the_sample_is_dropped():
    view = {WORLD_AZIMUTH_KEY: 137.0, "occupancy": 40.0}
    i = DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)
    assert np.isnan(goal_vector(view, object_key="NOT_AN_ASSET")[i])
    assert np.isnan(goal_vector(view)[i])                      # no object_key at all
    got = goal_vector(view, object_key=SOME_ASSET)[i]
    assert np.isfinite(got)                                    # mapped asset works
    assert got == pytest.approx(subject_bearing_deg(137.0, SOME_ASSET), abs=1e-4)


def test_goal_vector_still_reads_the_other_keys():
    view = {WORLD_AZIMUTH_KEY: 10.0, "occupancy": 42.0, "score_body_in_frame_ratio": 88.0}
    vec = goal_vector(view, object_key=SOME_ASSET)
    assert vec[DEFAULT_GOAL_KEYS.index("occupancy")] == pytest.approx(42.0)
    assert vec[DEFAULT_GOAL_KEYS.index("body_in_frame_ratio")] == pytest.approx(88.0)
