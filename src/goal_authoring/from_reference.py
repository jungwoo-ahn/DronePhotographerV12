"""Module 2: reference image -> GoalProfile (composition transfer).

The geometric framing keys are EXACT — a detected person bbox fed through the SAME shot-profile
computation used in training (`compute_v5_scores`): occupancy, subject placement, bbox offsets. The
semantic keys come from the SAME body-pose keypoints (one YOLO-pose model supplies box + keypoints):
subject bearing via a fitted regressor (~85-91% front/side/back, MAE ~18deg) and camera elevation via
a second regressor (MAE ~8deg vs 13.7 for predict-median). Both beat a VLM by a wide margin — the VLM
was at chance on elevation and 68%/32% on bearing.

Caveat on elevation: the v7 data is 68% high-angle / 3.5% low-angle, so low-angle references are
extrapolation; treat the estimate as a prior the user can override in words.
"""
from __future__ import annotations

import os

import numpy as np

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.pose_bearing import BearingModel, ElevationModel, elev_features, pose_features
from src.scoring.bbox_control import compute_v5_scores

DEFAULT_POSE_MODEL = "yolo11l-pose.pt"
DEFAULT_BEARING_MODEL = "assets/models/bearing_pose_rf.joblib"
DEFAULT_ELEVATION_MODEL = "assets/models/elevation_pose_rf.joblib"

# keys the detected 2D bbox pins exactly (same computation as the training profile)
GEOM_KEYS = ("occupancy", "object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset")


def geometric_keys(bbox, image_w: int, image_h: int) -> dict[str, float]:
    """Exact 2D framing keys from a person bbox, via the training shot-profile computation.
    (az/el are irrelevant to these keys; body_in_frame is handled separately since a reference image
    doesn't reveal the out-of-frame extent.)"""
    sc = compute_v5_scores(int(image_w), int(image_h), tuple(float(v) for v in bbox), 0.0, 0.0)
    return {k: float(sc[k]) for k in GEOM_KEYS}


def estimate_body_in_frame(bbox, kp, image_h: int) -> float | None:
    """Rough body-in-frame % from vertical framing + ankle-keypoint visibility. None if uncertain
    (-> left unspecified rather than guessed)."""
    y0, y1 = float(bbox[1]), float(bbox[3])
    m = 0.02 * image_h
    top_cut, bot_cut = y0 <= m, y1 >= image_h - m
    ankle_conf = float(max(kp[15, 2], kp[16, 2])) if kp is not None else 0.0
    if not top_cut and not bot_cut and ankle_conf > 0.5:
        return 100.0                      # whole body inside the frame
    if bot_cut and ankle_conf < 0.3:
        return 55.0                       # legs cut off at the bottom
    if top_cut and not bot_cut:
        return 60.0                       # head/top cropped
    return None                           # too uncertain -> unspecified


def profile_from_detection(bbox, kp, image_w: int, image_h: int, bearing: BearingModel | None,
                           elevation: "ElevationModel | None" = None,
                           *, bearing_conf_min: float = 0.35) -> GoalProfile:
    """Assemble a GoalProfile from one detection. `bearing`/`elevation` may be None (then unspecified).

    Elevation comes from pose keypoints, NOT a VLM (which was at chance): vertical foreshortening
    carries camera pitch, CV MAE ~8deg. Caveat: the v7 data it was fitted on is 68% high-angle /
    3.5% low-angle, so low-angle references are extrapolation."""
    vals = geometric_keys(bbox, image_w, image_h)
    spec = set(GEOM_KEYS)
    if bearing is not None and kp is not None:
        bdeg, conf = bearing.bearing_deg(pose_features(kp, bbox))
        if conf >= bearing_conf_min:
            vals[SUBJECT_BEARING_KEY] = bdeg
            spec.add(SUBJECT_BEARING_KEY)
    if elevation is not None and kp is not None:
        vals["cam_to_obj_elevation_deg"] = elevation.elevation_deg(
            elev_features(kp, bbox, image_w, image_h))
        spec.add("cam_to_obj_elevation_deg")
    bif = estimate_body_in_frame(bbox, kp, image_h)
    if bif is not None:
        vals["body_in_frame_ratio"] = bif
        spec.add("body_in_frame_ratio")
    return GoalProfile(vals, frozenset(spec))


class ReferenceEstimator:
    """reference image -> GoalProfile. Loads YOLO-pose (person box + keypoints) + the bearing model."""

    def __init__(self, pose_model: str = DEFAULT_POSE_MODEL,
                 bearing_model: str = DEFAULT_BEARING_MODEL,
                 elevation_model: str | None = DEFAULT_ELEVATION_MODEL, device=None):
        import cv2
        if not hasattr(cv2, "imshow"):
            cv2.imshow = lambda *a, **k: None
        from ultralytics import YOLO
        self.yolo = YOLO(pose_model)
        if device is None:
            # the cluster GPUs are often all taken; falling back keeps goal extraction working
            # (slower) instead of dying mid-scan with an unhelpful CUDA error
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.bearing = BearingModel.load(bearing_model)
        self.elevation = (ElevationModel.load(elevation_model)
                          if elevation_model and os.path.exists(elevation_model) else None)

    def detect_main_subject(self, image):
        """Return (bbox, keypoints, W, H) for the most prominent person, or None."""
        res = self.yolo.predict(image, verbose=False, device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        j = int(np.argmax(areas * confs))                 # prominence = area x confidence
        kp = res.keypoints.data.cpu().numpy()[j] if res.keypoints is not None else None
        H, W = res.orig_shape
        return boxes[j], kp, int(W), int(H)

    def __call__(self, image) -> GoalProfile:
        det = self.detect_main_subject(image)
        if det is None:
            return GoalProfile({}, frozenset())            # no subject -> empty goal
        bbox, kp, W, H = det
        return profile_from_detection(bbox, kp, W, H, self.bearing, self.elevation)
