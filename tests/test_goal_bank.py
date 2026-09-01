"""Goal bank: parsing, band masks, distances, and fidelity to the export it came from."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.goal_bank import DEFAULT_BANK_PATH, GoalBank, band_of

P = ("Move the camera to achieve this shot: a wide shot of the subject from the subject's front, "
     "at eye level, centered and mid in the frame, full body in frame, uncropped. (bearing 21°, "
     "occupancy 12%, elevation -6°, body_in_frame 100%, center 600/430 px, half_size 96/246 px, visible 1.00)")
Q = ("Move the camera to achieve this shot: a medium shot of the subject from the subject's front-left, "
     "at high angle, right third and mid in the frame, partially cut off, cropped at both the head and "
     "the feet. (bearing 297°, occupancy 55%, elevation -33°, body_in_frame 51%, center 713/384 px, "
     "half_size 280/384 px, visible 0.51)")


def synthetic():
    return GoalBank.from_prompts([P, Q, "not a prompt"], n_train=[3, 0, 1], n_val=[0, 2, 0])


def test_from_prompts_skips_non_matching_and_round_trips():
    b = synthetic()
    assert len(b) == 2 and b.meta["skipped"] == 1
    assert b.prompt(0) == P and b.prompt(1) == Q
    assert list(b.train_mask()) == [True, False]


def test_band_masks_follow_the_vocabulary():
    b = synthetic()
    assert list(b.band_mask("occupancy", 14.0)) == [True, False]           # wide shot
    assert list(b.band_mask(SUBJECT_BEARING_KEY, 300.0)) == [False, True]  # front-left
    assert list(b.band_mask("object_center_x", 900.0)) == [False, True]    # right third
    assert list(b.crop_mask("uncropped")) == [True, False]
    assert list(b.band_mask("bbox_x_offset", 5.0)) == [True, True]         # no band -> no constraint
    assert band_of("cam_to_obj_elevation_deg", -33.0) == "high angle"
    assert band_of("visible_frac", 0.5) is None


def test_distances_are_normalized_and_cyclic():
    b = synthetic()
    d = b.distances({SUBJECT_BEARING_KEY: 359.0}, [SUBJECT_BEARING_KEY])
    assert d[0] == pytest.approx(22.0 / 180.0) and d[1] == pytest.approx(62.0 / 180.0)
    d2 = b.distances({"occupancy": 12.0, "object_center_x": 600.0}, ["occupancy", "object_center_x"])
    assert d2[0] == pytest.approx(0.0)


def test_save_load_round_trip(tmp_path):
    b = synthetic()
    p = b.save(tmp_path / "bank.npz")
    c = GoalBank.load(p)
    assert len(c) == 2 and c.prompt(1) == Q and c.meta["skipped"] == 1
    assert np.array_equal(c.n_val, b.n_val)


@pytest.mark.skipif(not DEFAULT_BANK_PATH.exists(), reason="no shipped goal bank")
def test_shipped_bank_reproduces_the_export():
    b = GoalBank.load()
    assert len(b) > 40000
    tasks = Path("runs/lerobot_v5/meta/tasks.parquet")
    if not tasks.exists():
        pytest.skip("no exported dataset in this checkout")
    import pandas as pd
    df = pd.read_parquet(tasks)
    prompts = list(df["task"].astype(str)) if "task" in df.columns else list(df.index.astype(str))
    assert len(prompts) == len(b)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(b), 200, replace=False):
        assert b.prompt(int(i)) == prompts[int(i)]
    sup = b.support_table(b.train_mask())
    # the facts the completion design rests on
    assert sup["occupancy"]["extreme wide shot"] == 0
    assert sup["occupancy"]["close-up"] < 50
    assert sup["object_center_x"]["off-screen left"] == 0
