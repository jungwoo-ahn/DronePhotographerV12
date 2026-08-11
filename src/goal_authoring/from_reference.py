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
# crop keys the bbox pins by SIDE (see `crop_from_bbox` for why not by fraction)
CROP_KEYS = ("head_in_frame", "top_cut_frac", "bot_cut_frac", "visible_frac")


def geometric_keys(bbox, image_w: int, image_h: int) -> dict[str, float]:
    """Exact 2D framing keys from a person bbox, via the training shot-profile computation.
    (az/el are irrelevant to these keys; body_in_frame is handled separately since a reference image
    doesn't reveal the out-of-frame extent.)"""
    sc = compute_v5_scores(int(image_w), int(image_h), tuple(float(v) for v in bbox), 0.0, 0.0)
    return {k: float(sc[k]) for k in GEOM_KEYS}


# COCO-pose joint indices used by the false-positive veto in `crop_from_bbox`.
_L_ANKLE, _R_ANKLE = 15, 16
_HEAD_KP = (0, 1, 2)          # nose, left eye, right eye
# Tuned on 96 hand-labelled frames (24 per crop class) against the training labels; see the
# docstring. conf 0.7 / inset 0.03 was the best of a 4x4 sweep.
_KP_CONF_MIN = 0.7
_KP_INSET = 0.03


def crop_from_bbox(bbox, image_h: int, *, margin_frac: float = 0.002, kp=None) -> dict[str, float]:
    """Which END of the subject the frame cuts, from the bbox alone.

    The training side computes the same quantities in
    `src.common.annotations._apply_crop_extent`, but from the UNCLIPPED projected bbox, so it
    knows both the side and the fraction. A detector's box on a real photograph is already
    clipped to the image: the subject's true extent beyond the edge is simply not observable.

    So this returns the side (`top_cut_frac` / `bot_cut_frac` as booleans-in-float form) and
    reports `visible_frac` ONLY when nothing is cut, where it is exactly 1.0. Inventing a
    fraction for a cut box would put a fabricated number straight into the prompt — and the
    prompt is the only channel the policy has.

    Replaces the old `estimate_body_in_frame`, which detected `top_cut`/`bot_cut` internally and
    then threw the side away into one of three magic area numbers (100/55/60) — the exact
    ambiguity that let a beheaded subject and a chest-up portrait score identically.

    `kp` (YOLO-pose keypoints) vetoes a false positive, and only that — the box edge is still
    what DETECTS a crop. Measured on 96 frames labelled by the training side, 24 per class:
    the box-edge rule alone recovered `bot` 23/24, `top` 21/24, `both` 20/24 but `none` only
    6/24. Cause, from the failing frames: on a shadowed floor the person box runs to exactly
    y1 = image_h while the projected mesh ends 12-119 px higher, so a fully visible subject
    reads as bottom-cropped. A shadow has no ankles. Sweeping `margin_frac` cannot fix it
    (0.002 -> 0.05 moved the total 73% -> 68%), which is why the margin default dropped to
    0.002 and the veto carries the rest.

    The veto costs `both` accuracy (20/24 -> 14/24) because YOLO-pose still emits a confident
    ankle for a subject whose feet are out of frame. That trade is worth it in the real
    distribution — goal frames run bot 42.8% / none 19.6% / both 19.1% / top 18.4%, so
    prevalence-weighted accuracy goes 77.9% -> 87.0%. Pass `kp=None` for the box-only rule.
    """
    y0, y1 = float(bbox[1]), float(bbox[3])
    span = y1 - y0
    if span <= 0:
        return {}
    m = margin_frac * image_h
    top_cut, bot_cut = y0 <= m, y1 >= image_h - m
    if kp is not None and len(kp) > _R_ANKLE:
        ankles = [kp[i] for i in (_L_ANKLE, _R_ANKLE) if float(kp[i][2]) >= _KP_CONF_MIN]
        if ankles and max(float(a[1]) for a in ankles) < image_h * (1.0 - _KP_INSET):
            bot_cut = False                       # feet are visible; the box caught a shadow
        head = [kp[i] for i in _HEAD_KP if float(kp[i][2]) >= _KP_CONF_MIN]
        if head and min(float(h[1]) for h in head) > image_h * _KP_INSET:
            top_cut = False
    out = {
        "head_in_frame": 0.0 if top_cut else 1.0,
        "top_cut_frac": 1.0 if top_cut else 0.0,
        "bot_cut_frac": 1.0 if bot_cut else 0.0,
    }
    if not top_cut and not bot_cut:
        out["visible_frac"] = 1.0
    return out


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
    # Crop side, not an area ratio. `body_in_frame_ratio` is left UNSPECIFIED on purpose: it is
    # an area fraction of the subject's full extent, and that extent is unobservable in a
    # clipped photograph. `goal_prompt(specified=...)` drops the clause rather than guess.
    crop = crop_from_bbox(bbox, image_h, kp=kp)
    vals.update(crop)
    spec.update(crop)
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
