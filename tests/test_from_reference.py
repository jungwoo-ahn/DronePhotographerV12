"""Module 2 (reference image -> GoalProfile) unit tests for the pure pieces (no YOLO / no model):
exact geometric keys, body-in-frame estimate, and profile assembly incl. the elevation-unspecified
and bearing-confidence-gating rules."""
from __future__ import annotations

import numpy as np

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.from_reference import (
    GEOM_KEYS,
    crop_from_bbox,
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


def test_visible_bbox_convention_for_cutoff_subject():
    # a subject whose legs extend below the frame (y2=1200 > H=768): object_center/offset must come
    # from the VISIBLE (clipped) box, not the full box — so a reference image and a training goal agree.
    g = geometric_keys((400, 200, 600, 1200), W, H)
    assert g["object_center_y"] == (200 + 768) // 2      # visible centre 484, NOT full centre 700
    assert g["bbox_y_offset"] == (768 - 200) // 2         # visible half-height, NOT full


def test_crop_side_from_bbox():
    """The SIDE is observable from a clipped box; the FRACTION is not."""
    inside = crop_from_bbox((300, 50, 700, 700), H)          # clear of both edges
    assert inside["head_in_frame"] == 1.0
    assert inside["top_cut_frac"] == 0.0 and inside["bot_cut_frac"] == 0.0
    assert inside["visible_frac"] == 1.0                     # only knowable when nothing is cut

    bottom = crop_from_bbox((300, 300, 700, 768), H)         # touches the bottom edge
    assert bottom["head_in_frame"] == 1.0 and bottom["bot_cut_frac"] > 0
    assert "visible_frac" not in bottom                      # extent past the edge is unobservable

    top = crop_from_bbox((300, 0, 700, 600), H)              # touches the top edge
    assert top["head_in_frame"] == 0.0 and top["top_cut_frac"] > 0
    assert "visible_frac" not in top


def test_crop_label_matches_the_training_serializer():
    """An authored goal and a training goal describing the same framing must read the same."""
    from src.data.lerobot_export import crop_phrase
    from src.goal_authoring import vocab
    for label, (t, b) in vocab.CROP_SIDE.items():
        assert crop_phrase(t, b) == label


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
