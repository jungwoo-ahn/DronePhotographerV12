"""Tests for the cinematography vocabulary (single source of truth) + GoalProfile.
Pure/deterministic — no torch, no data. Validates the profile<->categories<->NL round-trips that
both goal-authoring front-ends depend on."""
from __future__ import annotations

import pytest

from src.common.facing import SECTOR8, sector8
from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.goal_profile import GoalProfile, feasibility_project


def test_bearing_centroid_inverts_sector8():
    for label in SECTOR8:
        assert sector8(vocab.bearing_centroid(label)) == label


def test_all_category_centroids_reclassify_to_themselves():
    # centroid of a band must classify back into that same band (tables are self-consistent)
    for table, key, fn in [
        (vocab.SHOT_SIZE, "occupancy", lambda v: vocab._classify(v, vocab.SHOT_SIZE)),
        (vocab.BODY_FRAMING, "body", lambda v: vocab._classify(v, vocab.BODY_FRAMING)),
        (vocab.ELEVATION, "elev", lambda v: vocab._classify(v, vocab.ELEVATION)),
    ]:
        for label, (_lo, _hi, c) in table.items():
            assert fn(c) == label, f"{key}:{label} centroid {c} -> {fn(c)}"


def test_categories_profile_roundtrip_identity():
    # every category -> profile -> category returns the same category
    cats = {
        "shot_size": "medium shot",
        "body_framing": "mostly in frame",
        "elevation": "high angle",
        "bearing": "front-right",
        "placement_x": "right third",
        "placement_y": "upper",
    }
    vals, spec = vocab.categories_to_profile(cats)
    assert spec == frozenset({"occupancy", "body_in_frame_ratio", "cam_to_obj_elevation_deg",
                              SUBJECT_BEARING_KEY, "object_center_x", "object_center_y"})
    assert vocab.profile_to_categories(vals) == cats


def test_partial_goal_only_specifies_given_axes():
    vals, spec = vocab.categories_to_profile({"shot_size": "close-up"})
    assert spec == frozenset({"occupancy"})
    assert set(vals) == {"occupancy"}
    gp = GoalProfile(vals, spec)
    nl = gp.to_prompt()
    assert "close-up" in nl
    # unspecified axes are NOT asserted in the (partial) prompt
    assert "angle" not in nl and "third" not in nl
    assert not gp.is_complete() and "cam_to_obj_elevation_deg" in gp.missing()


def test_prompt_serializer_words_and_numbers():
    gp = GoalProfile.from_categories({
        "shot_size": "close-up", "bearing": "front", "elevation": "low angle",
        "placement_x": "left third", "placement_y": "lower", "body_framing": "full body in frame",
    })
    nl = gp.to_prompt()
    for w in ["close-up", "front", "low angle", "left third", "lower", "full body in frame"]:
        assert w in nl, f"missing '{w}' in: {nl}"
    assert "%" in nl and "°" in nl  # numbers included
    assert nl.startswith("Move the camera to achieve this shot:")   # the TRAINING serializer, not a second one
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert gp.to_nl() == nl                                       # deprecated alias, same string
    assert any(issubclass(r.category, DeprecationWarning) for r in rec)


def test_feasibility_clamps_and_wraps():
    out = feasibility_project({"occupancy": 250.0, "cam_to_obj_elevation_deg": -999.0,
                               SUBJECT_BEARING_KEY: 370.0})
    assert out["occupancy"] == 100.0
    assert out["cam_to_obj_elevation_deg"] == -90.0
    assert out[SUBJECT_BEARING_KEY] == pytest.approx(10.0)  # cyclic wrap, not clamp


def test_goalprofile_merge_overrides_on_specified_keys():
    ref = GoalProfile.from_categories({"shot_size": "wide shot", "bearing": "back"})
    tweak = GoalProfile.from_categories({"shot_size": "close-up"})   # "like ref but a close-up"
    merged = ref.merge(tweak)
    assert merged.categories()["shot_size"] == "close-up"           # tweak wins
    assert merged.categories()["bearing"] == "back"                 # ref preserved
    assert merged.specified == frozenset({"occupancy", SUBJECT_BEARING_KEY})


def test_from_full_profile_wraps_detection_output():
    # Module 2 hands a fully-computed profile; wrap it, all keys specified
    full = {"occupancy": 42.0, "body_in_frame_ratio": 88.0, SUBJECT_BEARING_KEY: 95.0,
            "cam_to_obj_elevation_deg": -30.0, "object_center_x": 700.0, "object_center_y": 120.0}
    gp = GoalProfile.from_full_profile(full)
    assert gp.specified == frozenset(full)
    assert not gp.is_partial() or SUBJECT_BEARING_KEY in gp.specified  # bearing present
    assert gp.categories()["shot_size"] == "medium shot"
