"""Module 2 (reference image -> GoalProfile) unit tests for the pure pieces (no YOLO / no model):
exact geometric keys, body-in-frame estimate, and profile assembly incl. the elevation-unspecified
and bearing-confidence-gating rules."""
from __future__ import annotations

import numpy as np

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.from_reference import (
    GEOM_KEYS,
    estimate_body_in_frame,
    geometric_keys,
    profile_from_detection,
)

W, H = 1024, 768


def test_geometric_keys_exact_from_bbox():
    # a bbox covering the centre quarter (50% x 50%) -> occupancy 25%, centred
    g = geometric_keys((256, 192, 768, 576), W, H)
    assert g["occupancy"] == 25
    assert g["object_center_x"] == 512 and g["object_center_y"] == 384
    assert g["bbox_x_offset"] == 256 and g["bbox_y_offset"] == 192


def test_geometric_keys_offcenter():
    g = geometric_keys((0, 0, 300, 300), W, H)          # top-left
    assert g["object_center_x"] == 150 and g["object_center_y"] == 150


def test_body_in_frame_estimate():
    kp_full = np.zeros((17, 3)); kp_full[15, 2] = kp_full[16, 2] = 0.9   # ankles visible
    assert estimate_body_in_frame((300, 50, 700, 700), kp_full, H) == 100.0   # inside, ankles seen
    kp_noankle = np.zeros((17, 3))
    assert estimate_body_in_frame((300, 300, 700, 768), kp_noankle, H) == 55.0  # bottom cut, no ankles
    # uncertain -> None
    assert estimate_body_in_frame((300, 300, 700, 600), kp_noankle, H) is None


class _MockBearing:
    def __init__(self, deg, conf): self._d, self._c = deg, conf
    def bearing_deg(self, feats): return self._d, self._c


def test_profile_assembly_bearing_gated_and_elevation_unspecified():
    kp = np.zeros((17, 3)); kp[5:13, 2] = 0.9   # shoulders/hips visible so pose_features is well-defined
    bbox = (256, 192, 768, 576)

    # confident bearing -> included
    gp = profile_from_detection(bbox, kp, W, H, _MockBearing(90.0, 0.9))
    assert gp.values[SUBJECT_BEARING_KEY] == 90.0
    assert set(GEOM_KEYS).issubset(gp.specified)
    assert "cam_to_obj_elevation_deg" not in gp.specified   # elevation never guessed

    # low-confidence bearing -> dropped (unspecified)
    gp2 = profile_from_detection(bbox, kp, W, H, _MockBearing(90.0, 0.2))
    assert SUBJECT_BEARING_KEY not in gp2.specified

    # no bearing model at all -> geometric-only goal
    gp3 = profile_from_detection(bbox, kp, W, H, None)
    assert SUBJECT_BEARING_KEY not in gp3.specified
    assert "occupancy" in gp3.specified


def test_reference_profile_serializes_to_prompt():
    kp = np.zeros((17, 3)); kp[5:13, 2] = 0.9
    gp = profile_from_detection((256, 192, 768, 576), kp, W, H, _MockBearing(0.0, 0.9))
    nl = gp.to_nl()
    assert "shot" in nl and "front" in nl        # bearing 0 -> "front"
    assert "angle" not in nl                      # elevation absent
