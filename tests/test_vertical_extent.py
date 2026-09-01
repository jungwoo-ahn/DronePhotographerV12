"""Out-of-frame extent from body proportions (Module 2), on synthetic keypoints."""
from __future__ import annotations

import numpy as np
import pytest

from src.goal_authoring.from_reference import (
    ANTHRO_LEVELS, crop_fractions_from_extent, crop_from_bbox, estimate_vertical_extent,
    profile_from_detection,
)

H, W = 768, 1024


def body(y_top: float, stature_px: float, conf: float = 0.9, drop=()):
    """COCO keypoints for an upright body whose vertex is at y_top and soles at y_top+stature."""
    kp = np.zeros((17, 3))
    for name, (joints, frac) in ANTHRO_LEVELS.items():
        if name in drop:
            continue
        for j in joints:
            kp[j] = (W / 2, y_top + stature_px * (1.0 - frac), conf)
    return kp


def test_recovers_vertex_and_soles_when_legs_are_cut():
    kp = body(y_top=100, stature_px=900)             # soles at 1000 > H: knees/ankles out of frame
    ext = estimate_vertical_extent(kp, H)
    assert ext is not None and ext["r2"] > 0.999
    assert ext["y_top"] == pytest.approx(100, abs=1) and ext["y_bottom"] == pytest.approx(1000, abs=1)


def test_needs_three_levels():
    kp = body(y_top=100, stature_px=900, drop=("knee", "ankle", "hip"))   # head + shoulder only
    assert estimate_vertical_extent(kp, H) is None
    kp2 = body(y_top=100, stature_px=900, conf=0.2)                        # nothing confident
    assert estimate_vertical_extent(kp2, H) is None


def test_upside_down_is_rejected():
    kp = body(y_top=100, stature_px=900)
    kp[:, 1] = H - kp[:, 1]
    assert estimate_vertical_extent(kp, H) is None


def test_fractions_follow_the_side_detection():
    kp = body(y_top=100, stature_px=900)
    ext = estimate_vertical_extent(kp, H)
    side = crop_from_bbox([300, 100, 700, H], H, kp=None)     # box says bottom is cut
    fr = crop_fractions_from_extent(side, ext, H)
    assert fr["top_cut_frac"] == 0.0
    assert fr["bot_cut_frac"] == pytest.approx((1000 - H) / 900, abs=0.01)
    assert fr["visible_frac"] == pytest.approx(1 - (1000 - H) / 900, abs=0.01)
    # a side the box says is NOT cut stays 0 even if the fit would put the vertex above the frame
    side2 = crop_from_bbox([300, 100, 700, 600], H, kp=None)
    assert crop_fractions_from_extent(side2, ext, H)["bot_cut_frac"] == 0.0


def test_profile_marks_measured_fractions_exact():
    kp = body(y_top=100, stature_px=900)
    gp = profile_from_detection((300, 100, 700, H), kp, W, H, None, None)
    assert {"top_cut_frac", "bot_cut_frac", "visible_frac"} <= gp.exact
    assert 0.6 < gp.values["visible_frac"] < 0.8
    assert "body_in_frame_ratio" not in gp.specified
    # without a usable fit the side stays a side (1.0 marker), not exact, and visible is absent
    gp2 = profile_from_detection((300, 100, 700, H), np.zeros((17, 3)), W, H, None, None)
    assert gp2.values["bot_cut_frac"] == 1.0 and "visible_frac" not in gp2.values
    assert "bot_cut_frac" not in gp2.exact
