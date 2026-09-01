"""Completion: a partial goal always becomes a complete, internally consistent, training-shape one.

Rules: the person-box geometry must reproduce the requested occupancy on the CLIPPED box, honour
the requested crop word, and keep `body_in_frame == visible_frac` whenever the box is not
x-clipped (an exact identity in the data: corr 1.0, median |diff| 0). Bank: an NL-only request
must come back as a REAL training goal — byte-identical to one of the bank's prompts — with
measured numbers preserved and unsupported categories relaxed out loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH, SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.completion import (
    PERSON_ASPECT, BankCompleter, RuleCompleter, extent_from_visible_box, solve_person_box,
)
from src.goal_authoring.goal_bank import GoalBank
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.serialize import crop_phrase, is_training_shape

W, H = RENDER_WIDTH, RENDER_HEIGHT


# ------------------------------------------------------------------ person box ----
# 31.25% is the largest uncut box at aspect 2.4 (h = H/2 -> w = 160 px); anything bigger must be cut.
@pytest.mark.parametrize("occ", [10, 14, 25, 31])
def test_uncut_box_hits_occupancy_and_is_consistent(occ):
    g, notes = solve_person_box(occ, W / 2, H / 2)
    assert g["occupancy"] == pytest.approx(occ, abs=0.05)
    assert g["top_cut_frac"] == 0 and g["bot_cut_frac"] == 0 and g["visible_frac"] == 1.0
    assert g["body_in_frame_ratio"] == pytest.approx(100.0)
    assert g["bbox_y_offset"] / g["bbox_x_offset"] == pytest.approx(PERSON_ASPECT)
    assert crop_phrase(g["top_cut_frac"], g["bot_cut_frac"]) == "uncropped"


@pytest.mark.parametrize("occ", [40, 60])
def test_too_big_to_fit_uncut_becomes_cut_at_both_ends(occ):
    g, notes = solve_person_box(occ, W / 2, H / 2)
    assert g["occupancy"] == pytest.approx(occ, abs=0.05)
    assert crop_phrase(g["top_cut_frac"], g["bot_cut_frac"]) == "cropped at both the head and the feet"
    assert g["bbox_y_offset"] == pytest.approx(H / 2)          # fills the height
    assert any("does not fit" in n for n in notes)


@pytest.mark.parametrize("label", ["cropped at the legs", "cropped below the waist", "cropped above the head",
                                   "cropped at both the head and the feet"])
def test_requested_crop_word_is_reproduced(label):
    t, b = vocab.CROP_SIDE[label]
    g, _ = solve_person_box(30, W / 2, H / 2, t, b)
    assert crop_phrase(g["top_cut_frac"], g["bot_cut_frac"]) == label
    assert g["occupancy"] == pytest.approx(30, abs=0.05)
    assert g["visible_frac"] == pytest.approx(1.0 - g["top_cut_frac"] - g["bot_cut_frac"], abs=1e-6)
    if g["visible_box"][0] > 0 and g["visible_box"][2] < W:          # not x-clipped
        assert g["body_in_frame_ratio"] / 100.0 == pytest.approx(g["visible_frac"], abs=1e-6)


def test_bottom_cut_puts_the_visible_box_on_the_bottom_edge():
    g, _ = solve_person_box(20, W / 2, H / 2, 0.0, 0.2)
    assert g["visible_box"][3] == pytest.approx(H)
    assert g["object_center_y"] > H / 2


def test_placement_at_the_edge_is_nudged_inside_when_uncut():
    g, notes = solve_person_box(15, 0.95 * W, 0.9 * H)
    x0, y0, x1, y1 = g["full_box"]
    assert x0 >= -1e-6 and y0 >= -1e-6 and x1 <= W + 1e-6 and y1 <= H + 1e-6
    assert any("placement moved" in n for n in notes)


def test_extent_from_visible_box_matches_the_data_identity():
    # visible box not touching the sides, cut 30% below: body_in_frame == visible fraction
    g = extent_from_visible_box(500, 500, 100, 268, 0.0, 0.3)
    assert g["visible_frac"] == pytest.approx(0.7)
    assert g["body_in_frame_ratio"] == pytest.approx(70.0)
    # touching the left edge: the aspect supplies the hidden width, so body_in_frame drops
    g2 = extent_from_visible_box(60, 500, 60, 268, 0.0, 0.3)
    assert g2["body_in_frame_ratio"] < 70.0


# ------------------------------------------------------------------ rule completer ----
def _nl(cats):
    return GoalProfile.from_categories(cats)


def test_rule_completer_fills_everything_and_keeps_the_request():
    c = RuleCompleter().complete(_nl({"shot_size": "medium shot", "bearing": "front-right",
                                      "elevation": "high angle", "crop_side": "cropped at the legs"}))
    p = c.profile
    assert p.is_complete() and is_training_shape(c.prompt), c.prompt
    cats = p.categories()
    assert cats["shot_size"] == "medium shot" and cats["bearing"] == "front-right"
    assert cats["elevation"] == "high angle" and cats["crop_side"] == "cropped at the legs"
    assert c.source[SUBJECT_BEARING_KEY] == "given" and c.source["bbox_x_offset"] == "rule"
    assert c.method == "rules" and c.support is None
    assert any("widened" in w for w in c.warnings)          # 48% + legs cannot fit at aspect 2.4
    assert p.values["occupancy"] == pytest.approx(48.0, abs=0.1)


def test_rule_completer_defaults_are_explicit():
    c = RuleCompleter().complete(_nl({"shot_size": "wide shot"}))
    assert c.source[SUBJECT_BEARING_KEY] == "default" and c.profile.values[SUBJECT_BEARING_KEY] == 0.0
    assert c.source["cam_to_obj_elevation_deg"] == "default"
    assert c.profile.categories()["crop_side"] == "uncropped"


def test_rule_completer_body_framing_becomes_a_bottom_cut():
    c = RuleCompleter().complete(_nl({"shot_size": "medium-wide shot", "body_framing": "partially cut off"}))
    assert c.profile.values["body_in_frame_ratio"] == pytest.approx(45.0, abs=1.0)
    assert c.profile.categories()["crop_side"] == "cropped below the waist"
    assert any("assumed cut at the bottom" in w for w in c.warnings)


def test_rule_completer_image_path_keeps_measurements():
    measured = {"occupancy": 39.0, "object_center_x": 225.0, "object_center_y": 424.0,
                "bbox_x_offset": 225.0, "bbox_y_offset": 344.0, SUBJECT_BEARING_KEY: 73.0,
                "cam_to_obj_elevation_deg": -19.5, "top_cut_frac": 0.0, "bot_cut_frac": 1.0}   # side only
    gp = GoalProfile.from_full_profile(measured)
    c = RuleCompleter().complete(gp)
    for k in ("occupancy", "object_center_x", "bbox_y_offset", SUBJECT_BEARING_KEY):
        assert c.profile.values[k] == measured[k] and c.source[k] == "given"
    assert c.profile.categories()["crop_side"] == "cropped at the legs"       # side known, depth assumed
    assert 0 < c.profile.values["visible_frac"] < 1
    assert is_training_shape(c.prompt)


# ------------------------------------------------------------------ bank completer ----
P = ("Move the camera to achieve this shot: a wide shot of the subject from the subject's front, "
     "at eye level, centered and mid in the frame, full body in frame, uncropped. (bearing 21°, "
     "occupancy 12%, elevation -6°, body_in_frame 100%, center 600/430 px, half_size 96/246 px, visible 1.00)")
Q = ("Move the camera to achieve this shot: a medium shot of the subject from the subject's front-left, "
     "at high angle, right third and mid in the frame, partially cut off, cropped at both the head and "
     "the feet. (bearing 297°, occupancy 55%, elevation -33°, body_in_frame 51%, center 713/384 px, "
     "half_size 280/384 px, visible 0.51)")


def _bank():
    R = ("Move the camera to achieve this shot: a wide shot of the subject from the subject's back, at "
         "eye level, right third and lower in the frame, partially cut off, cropped below the waist. "
         "(bearing 178°, occupancy 13%, elevation -15°, body_in_frame 37%, center 645/612 px, "
         "half_size 159/156 px, visible 0.37)")
    S = ("Move the camera to achieve this shot: a wide shot of the subject from the subject's front, at "
         "eye level, centered and mid in the frame, mostly in frame, cropped at the legs. (bearing 10°, "
         "occupancy 15%, elevation -4°, body_in_frame 80%, center 520/400 px, half_size 100/250 px, "
         "visible 0.80)")
    return GoalBank.from_prompts([P, Q, R, S], n_train=[1, 1, 1, 1])


def test_bank_nl_request_returns_a_real_training_prompt():
    b = _bank()
    c = BankCompleter(b, min_support=1, k=5).complete(_nl({"shot_size": "wide shot", "bearing": "back"}))
    assert c.prompt == b.prompt(2)                       # the one back-view wide shot
    assert c.anchor == 2 and c.support == 1
    assert c.source["occupancy"] == "bank(anchored)" and c.source["visible_frac"] == "bank"


def test_bank_keeps_exact_numbers_and_fills_the_rest():
    b = _bank()
    measured = {"occupancy": 14.0, "object_center_x": 590.0, "object_center_y": 440.0,
                "bbox_x_offset": 90.0, "bbox_y_offset": 240.0, SUBJECT_BEARING_KEY: 18.0,
                "top_cut_frac": 0.0, "bot_cut_frac": 0.0, "visible_frac": 1.0}
    c = BankCompleter(b, min_support=1, k=5).complete(GoalProfile.from_full_profile(measured))
    assert c.anchor == 0                                      # the uncropped front wide shot
    assert c.profile.values["object_center_x"] == 590.0 and c.source["object_center_x"] == "given"
    assert c.profile.values["body_in_frame_ratio"] == 100.0 and c.source["body_in_frame_ratio"] == "bank"
    assert is_training_shape(c.prompt)


def test_bank_side_only_crop_lets_the_anchor_pick_the_depth():
    b = _bank()
    measured = {"occupancy": 13.0, "object_center_x": 640.0, "object_center_y": 600.0,
                "bbox_x_offset": 160.0, "bbox_y_offset": 160.0, SUBJECT_BEARING_KEY: 180.0,
                "top_cut_frac": 0.0, "bot_cut_frac": 1.0}                     # bottom cut, depth unknown
    c = BankCompleter(b, min_support=1, k=5).complete(GoalProfile.from_full_profile(measured))
    assert c.anchor == 2
    assert c.profile.categories()["crop_side"] == "cropped below the waist"
    assert c.profile.values["bot_cut_frac"] < 1.0 and c.source["bot_cut_frac"] == "bank(anchored)"


def test_bank_relaxes_unsupported_categories_with_a_warning():
    b = _bank()
    c = BankCompleter(b, min_support=1, k=5).complete(_nl({"shot_size": "close-up", "bearing": "back"}))
    assert c.warnings and "occupancy" in c.warnings[0]
    assert c.anchor == 2                                      # nearest by value among back views
    assert is_training_shape(c.prompt)


@pytest.mark.skipif(not __import__("src.goal_authoring.goal_bank", fromlist=["DEFAULT_BANK_PATH"]).DEFAULT_BANK_PATH.exists(),
                    reason="no shipped goal bank")
def test_shipped_bank_nl_requests_are_training_prompts():
    from pathlib import Path
    tasks = Path("runs/lerobot_v5/meta/tasks.parquet")
    if not tasks.exists():
        pytest.skip("no exported dataset in this checkout")
    import pandas as pd
    df = pd.read_parquet(tasks)
    exported = set(df["task"].astype(str)) if "task" in df.columns else set(df.index.astype(str))
    comp = BankCompleter()
    reqs = [{"shot_size": "wide shot", "bearing": "front"},
            {"shot_size": "medium shot", "bearing": "back-left", "elevation": "high angle"},
            {"shot_size": "close-up", "bearing": "front"},
            {"elevation": "low angle", "body_framing": "full body in frame", "bearing": "front"},
            {"shot_size": "medium-wide shot", "placement_x": "left third", "placement_y": "lower",
             "crop_side": "cropped at the legs"}]
    for r in reqs:
        c = comp.complete(_nl(r))
        assert c.prompt in exported, (r, c.prompt)
        assert c.support >= 1
