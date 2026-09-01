"""The serializer moved into the authoring package; nothing about it may have changed.

`goal_prompt` is the one function whose bytes the policy depends on. The exporter and every eval
script still import it from `src.data.lerobot_export`; that import must resolve to the SAME
function object, and the training-shape regex must accept every real training prompt and
invert it exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY
from src.goal_authoring import serialize
from src.goal_authoring.serialize import (
    FULL_PROFILE_KEYS, goal_prompt, is_training_shape, parse_training_prompt, profile_to_prompt,
)

TASKS = Path("runs/lerobot_v5/meta/tasks.parquet")


def test_exporter_import_is_the_same_object():
    from src.data import lerobot_export
    assert lerobot_export.goal_prompt is serialize.goal_prompt
    assert lerobot_export.crop_phrase is serialize.crop_phrase


def test_full_profile_prompt_has_the_training_shape():
    vals = {SUBJECT_BEARING_KEY: 57, "occupancy": 20, "cam_to_obj_elevation_deg": -8,
            "body_in_frame_ratio": 52, "object_center_x": 399, "object_center_y": 492,
            "bbox_x_offset": 140, "bbox_y_offset": 276,
            "top_cut_frac": 0.0, "bot_cut_frac": 0.48, "visible_frac": 0.52}
    p = profile_to_prompt(vals)
    assert is_training_shape(p), p
    back = parse_training_prompt(p)
    for k in DEFAULT_GOAL_KEYS:
        assert back[k] == vals[k]
    assert back["visible_frac"] == 0.52 and back["crop_label"] == "cropped below the waist"


def test_partial_prompt_is_not_the_training_shape():
    """The whole reason completion exists."""
    vals = {"occupancy": 88.0, SUBJECT_BEARING_KEY: 0.0}
    p = profile_to_prompt(vals, specified=set(vals))
    assert p.startswith(serialize.PROMPT_PREFIX) and not is_training_shape(p)


def test_full_keys_are_eight_plus_crop():
    assert FULL_PROFILE_KEYS == tuple(DEFAULT_GOAL_KEYS) + ("top_cut_frac", "bot_cut_frac", "visible_frac")


@pytest.mark.skipif(not TASKS.exists(), reason="no exported dataset in this checkout")
def test_every_exported_prompt_matches_and_round_trips():
    import pandas as pd
    from src.goal_authoring import vocab
    df = pd.read_parquet(TASKS)
    prompts = list(df["task"].astype(str)) if "task" in df.columns else list(df.index.astype(str))
    assert prompts
    bad_shape, bad_rt = [], []
    for p in prompts:
        row = parse_training_prompt(p)
        if row is None:
            bad_shape.append(p)
            continue
        t, b = vocab.CROP_SIDE[row.pop("crop_label")]
        row["top_cut_frac"], row["bot_cut_frac"] = t, b
        if profile_to_prompt(row) != p:
            bad_rt.append(p)
    assert not bad_shape, f"{len(bad_shape)} training prompts do not match TRAINING_PROMPT_RE, e.g. {bad_shape[0]}"
    assert not bad_rt, f"{len(bad_rt)} training prompts do not round-trip, e.g. {bad_rt[0]}"
