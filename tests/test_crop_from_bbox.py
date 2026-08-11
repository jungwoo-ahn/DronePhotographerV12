"""`crop_from_bbox` — the box edge detects the crop, keypoints only veto a false positive.

Pins the behaviour measured in the round-trip verification: on a shadowed floor the person
box runs to exactly `image_h` while the subject is fully visible, so the box-edge rule alone
recovered the uncropped class only 6/24. Ankle keypoints veto that. The veto must NOT be
able to invent a crop, only cancel one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.goal_authoring.from_reference import crop_from_bbox

H = 768
W = 1024


def kp(ankle_y=None, head_y=None, conf=0.9):
    """A 17-joint COCO-pose array with only the joints under test made confident."""
    a = np.zeros((17, 3), dtype=float)
    if head_y is not None:
        for i in (0, 1, 2):
            a[i] = (W / 2, head_y, conf)
    if ankle_y is not None:
        for i in (15, 16):
            a[i] = (W / 2, ankle_y, conf)
    return a


def side(d):
    t, b = d.get("top_cut_frac", 0.0) > 0, d.get("bot_cut_frac", 0.0) > 0
    return "both" if (t and b) else "top" if t else "bot" if b else "none"


def test_box_edge_detects_each_side():
    assert side(crop_from_bbox([0, 100, 400, 600], H)) == "none"
    assert side(crop_from_bbox([0, 0, 400, 600], H)) == "top"
    assert side(crop_from_bbox([0, 100, 400, H], H)) == "bot"
    assert side(crop_from_bbox([0, 0, 400, H], H)) == "both"


def test_visible_frac_only_when_nothing_is_cut():
    """A clipped photo cannot reveal the out-of-frame extent, so a fraction would be
    fabricated — and the prompt is the only channel the policy has."""
    assert crop_from_bbox([0, 100, 400, 600], H)["visible_frac"] == 1.0
    for bad in ([0, 0, 400, 600], [0, 100, 400, H], [0, 0, 400, H]):
        assert "visible_frac" not in crop_from_bbox(bad, H)


def test_confident_ankle_vetoes_a_shadow_false_positive():
    """The measured failure: box bottom at image_h, subject actually whole."""
    box = [0, 100, 400, H]
    assert side(crop_from_bbox(box, H, kp=None)) == "bot"
    assert side(crop_from_bbox(box, H, kp=kp(ankle_y=600))) == "none"


def test_veto_cannot_invent_a_crop():
    """Keypoints only cancel — a subject the box says is whole stays whole no matter
    where the joints land."""
    box = [0, 100, 400, 600]
    for a_y in (0.0, H, H / 2):
        for h_y in (0.0, H, H / 2):
            assert side(crop_from_bbox(box, H, kp=kp(ankle_y=a_y, head_y=h_y))) == "none"


def test_low_confidence_keypoints_do_not_veto():
    """An occluded joint is a guess; only a confident one may overrule the box."""
    box = [0, 100, 400, H]
    assert side(crop_from_bbox(box, H, kp=kp(ankle_y=600, conf=0.1))) == "bot"


def test_ankle_at_the_very_bottom_does_not_veto():
    """Feet touching the frame edge are consistent with being cut, so the veto must
    require the joint to sit meaningfully inside."""
    box = [0, 100, 400, H]
    assert side(crop_from_bbox(box, H, kp=kp(ankle_y=H - 1))) == "bot"


def test_degenerate_box_returns_nothing():
    assert crop_from_bbox([0, 300, 400, 300], H) == {}
    assert crop_from_bbox([0, 400, 400, 300], H) == {}


@pytest.mark.parametrize("margin", [0.0, 0.002, 0.02])
def test_head_in_frame_tracks_top_cut(margin):
    whole = crop_from_bbox([0, 100, 400, 600], H, margin_frac=margin)
    cut = crop_from_bbox([0, 0, 400, 600], H, margin_frac=margin)
    assert whole["head_in_frame"] == 1.0
    assert cut["head_in_frame"] == 0.0
