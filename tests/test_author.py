"""`author_goal` end to end (keyword classifier; no models, no GPU)."""
from __future__ import annotations

import pytest

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.author import author_goal
from src.goal_authoring.completion import RuleCompleter
from src.goal_authoring.goal_bank import DEFAULT_BANK_PATH
from src.goal_authoring.serialize import is_training_shape

_bank = pytest.mark.skipif(not DEFAULT_BANK_PATH.exists(), reason="no shipped goal bank")


def test_rules_path_needs_no_bank():
    g = author_goal("low angle full body from the front", method="rules", clamp_support=False)
    assert g.profile.is_complete() and is_training_shape(g.prompt)
    assert g.categories["elevation"] == "low angle" and g.categories["body_framing"] == "full body in frame"
    assert g.request.categories() == {"elevation": "low angle", "body_framing": "full body in frame",
                                      "bearing": "front"}
    rec = g.to_record()
    assert len(rec["goal_vec"]) == 8 and set(rec["goal_crop"]) == {"top_cut_frac", "bot_cut_frac", "visible_frac"}


def test_nothing_parsed_is_reported_not_silent():
    g = author_goal("please make it nice", method="rules", clamp_support=False)
    assert any("no cinematography attribute" in w for w in g.warnings)
    assert is_training_shape(g.prompt)


def test_requires_some_input():
    with pytest.raises(ValueError):
        author_goal()


@_bank
def test_bank_is_the_default_and_anchors_on_a_training_goal():
    g = author_goal("a close-up from the front")
    assert g.method == "bank" and g.support is not None and g.anchor is not None
    assert is_training_shape(g.prompt)
    assert g.categories["bearing"] == "front"
    # close-up has almost no training support: either relaxed with a warning, or a real
    # close-up row was found — both are legitimate, silence is not
    assert g.categories["shot_size"] in ("close-up", "medium close-up") or g.warnings


@_bank
def test_rules_are_clamped_to_the_trained_support_by_default():
    g = author_goal("a close-up from a low angle", method="rules")
    assert g.profile.values["occupancy"] <= 80.0
    assert g.profile.values["cam_to_obj_elevation_deg"] <= 20.0
    assert any("outside the trained support" in w for w in g.warnings)
    assert is_training_shape(g.prompt)


def test_nl_overrides_image_on_merge():
    """Simulate an image profile through a fake estimator; the words win on the keys they name."""
    from src.goal_authoring.goal_profile import GoalProfile
    measured = {"occupancy": 30.0, "object_center_x": 500.0, "object_center_y": 400.0,
                "bbox_x_offset": 150.0, "bbox_y_offset": 300.0, SUBJECT_BEARING_KEY: 170.0,
                "cam_to_obj_elevation_deg": -30.0, "top_cut_frac": 0.0, "bot_cut_frac": 0.0,
                "visible_frac": 1.0}
    est = lambda _img: GoalProfile.from_full_profile(measured)  # noqa: E731
    g = author_goal("but from the front", image="fake.jpg", estimator=est,
                    completer=RuleCompleter(), clamp_support=False)
    assert g.categories["bearing"] == "front"
    assert g.profile.values["occupancy"] == 30.0 and g.source["occupancy"] == "given"
    assert SUBJECT_BEARING_KEY not in g.request.exact       # overridden by words -> no longer a measurement
