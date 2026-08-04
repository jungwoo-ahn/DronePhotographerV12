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
    assert "close-up" in gp.to_nl()


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


def test_serializer_roundtrip_through_keyword_classifier():
    # profile -> NL (serializer) -> keyword classifier -> categories should recover the goal
    original = GoalProfile.from_categories({
        "shot_size": "close-up", "bearing": "front-right", "elevation": "high angle",
        "placement_x": "right third", "placement_y": "upper", "body_framing": "full body in frame",
    })
    nl = original.to_nl(numbers=True)
    recovered = language_to_goal(nl, keyword_classifier, project_feasible=False)
    assert recovered.categories() == original.categories()
