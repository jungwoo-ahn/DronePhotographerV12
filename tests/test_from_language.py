"""Module 1 (natural language -> GoalProfile) tests. Deterministic — use the keyword classifier and
mocks (no model). Includes the serializer round-trip: profile -> NL -> categories recovers the goal."""
from __future__ import annotations

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.from_language import (
    keyword_classifier,
    language_to_goal,
    validate_categories,
)
from src.goal_authoring.goal_profile import GoalProfile


def test_keyword_classifier_curated_sentences():
    cats = keyword_classifier("a dramatic low-angle close-up from the side")
    assert cats["shot_size"] == "close-up"
    assert cats["elevation"] == "low angle"
    assert cats["bearing"] == "side"

    cats2 = keyword_classifier("wide establishing shot from behind, subject on the left")
    assert cats2["shot_size"] == "extreme wide shot"  # 'establishing shot' is an extreme wide
    assert cats2["bearing"] == "back"
    assert cats2["placement_x"] == "left third"


def test_language_to_goal_partial_and_grounded():
    gp = language_to_goal("just a close-up", keyword_classifier)
    assert gp.specified == frozenset({"occupancy"})   # nothing else stated
    assert gp.categories() == {"shot_size": "close-up"}
    assert "close-up" in gp.to_prompt()


def test_validate_drops_invalid_labels():
    assert validate_categories({"shot_size": "cinematic", "bearing": "front", "bogus": "x"}) == {
        "bearing": "front"
    }


def test_mock_classifier_flows_through_grounding():
    mock = lambda _t: {"shot_size": "medium shot", "elevation": "high angle", "bearing": "front-right"}
    gp = language_to_goal("whatever", mock)
    assert gp.values["occupancy"] == vocab.SHOT_SIZE["medium shot"][2]
    assert gp.values["cam_to_obj_elevation_deg"] == vocab.ELEVATION["high angle"][2]
    assert gp.values[SUBJECT_BEARING_KEY] == 45.0     # front-right centroid


def test_training_prompt_roundtrips_through_keyword_classifier():
    # profile -> TRAINING prompt -> keyword classifier -> categories should recover the goal.
    # This used to run through a second serializer that the parser had been written against.
    original = GoalProfile.from_categories({
        "shot_size": "close-up", "bearing": "front-right", "elevation": "high angle",
        "placement_x": "right third", "placement_y": "upper", "body_framing": "full body in frame",
        "crop_side": "cropped at the legs",
    })
    nl = original.to_prompt()
    recovered = language_to_goal(nl, keyword_classifier, project_feasible=False)
    assert recovered.categories() == original.categories()


def test_keyword_classifier_reads_a_real_training_prompt():
    tp = ("Move the camera to achieve this shot: a medium-wide shot of the subject from the subject's "
          "front-left, at eye level, left third and mid in the frame, mostly in frame, cropped above "
          "the head. (bearing 297°, occupancy 24%, elevation -5°, body_in_frame 68%, center 142/332 px, "
          "half_size 142/332 px, visible 0.71)")
    assert keyword_classifier(tp) == {
        "shot_size": "medium-wide shot", "elevation": "eye level", "bearing": "front-left",
        "placement_x": "left third", "placement_y": "mid", "body_framing": "mostly in frame",
        "crop_side": "cropped above the head"}


def test_gold_set_phrasings_that_used_to_miss():
    assert keyword_classifier("straight-on portrait, they're looking right at us")["bearing"] == "front"
    assert keyword_classifier("medium close-up, right profile")["bearing"] == "right"
    assert keyword_classifier("cut them off at the head and the feet")["crop_side"] == "cropped at both the head and the feet"
