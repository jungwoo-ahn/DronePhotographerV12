"""Subject bearing (front/side/back view) from a single image's body-pose keypoints.

YOLO-pose detects the whole person reliably (unlike face-only detection). The anatomical left/right
shoulder x-ordering flips between a front and a back view (mirroring); facial-keypoint confidence
separates front from back; shoulder alignment collapses in profile. A classifier over these features
predicts the subject-relative bearing sector (benchmarked ~87% sector3 / ~72% sector8, MAE ~19° on
held-out renders — far above VLM's 68%/32%). Feature extraction here is shared by training
(`scripts/train_bearing_model.py`) and inference (`from_reference.py`); NO ultralytics import so the
package stays light — callers pass in the (17,3) COCO keypoints + person box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_FEATURE_NAMES = (
    "shoulder_dx", "hip_dx", "ear_dx", "eye_dx", "nose_dx", "shoulder_w",
    "c_nose", "c_leye", "c_reye", "c_lear", "c_rear", "c_lsh", "c_rsh", "c_lhip", "c_rhip",
    "eye_conf_asym", "ear_conf_asym",
)
POSE_FEATURE_DIM = len(POSE_FEATURE_NAMES)


def pose_features(kp: np.ndarray, box: np.ndarray) -> list[float]:
    """(17,3) COCO keypoints [x,y,conf] + person xyxy box -> orientation feature vector.
    All positional features are normalized by torso height and are translation-invariant."""
    kp = np.asarray(kp, dtype=np.float64)
    xy, c = kp[:, :2], kp[:, 2]
    lsh, rsh, lhip, rhip = xy[5], xy[6], xy[11], xy[12]
    sh_mid = 0.5 * (lsh + rsh)
    scale = float(np.linalg.norm(sh_mid - 0.5 * (lhip + rhip)))
    if scale < 5.0:                                   # profile / missing hips -> fall back to box height
        scale = float(box[3] - box[1]) * 0.3
    scale = max(scale, 1e-6)

    def dxn(a, b):  # signed, normalized L-R x gap (sign flips front<->back)
        return float((a[0] - b[0]) / scale)

    return [
        dxn(lsh, rsh), dxn(lhip, rhip), dxn(xy[3], xy[4]), dxn(xy[1], xy[2]),
        float((xy[0][0] - sh_mid[0]) / scale),
        float(np.linalg.norm(lsh - rsh) / scale),
        float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]),
        float(c[5]), float(c[6]), float(c[11]), float(c[12]),
        float(c[1] - c[2]), float(c[3] - c[4]),
    ]


class BearingModel:
    """A fitted sklearn regressor over `pose_features` predicting the CONTINUOUS subject bearing as
    (sin, cos). Regression (vs sector classification) gives a finer angle (benchmark MAE ~19°) and a
    natural confidence: an ensemble that disagrees averages toward the origin, so the predicted
    (sin,cos) magnitude shrinks -> we read it as confidence in [0,1]."""

    def __init__(self, reg):
        self.reg = reg

    def bearing_deg(self, features: list[float]) -> tuple[float, float]:
        """Return (subject_bearing_deg in [0,360), confidence in [0,1])."""
        import math
        v = self.reg.predict(np.asarray(features, dtype=np.float64).reshape(1, -1))[0]
        s, c = float(v[0]), float(v[1])
        conf = float(min(1.0, math.hypot(s, c)))
        deg = math.degrees(math.atan2(s, c)) % 360.0
        return deg, conf

    def predict(self, features: list[float]) -> tuple[str, float]:
        """Return (sector8 label, confidence)."""
        from src.common.facing import sector8
        deg, conf = self.bearing_deg(features)
        return sector8(deg), conf

    def save(self, path: str | Path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.reg, path)

    @classmethod
    def load(cls, path: str | Path) -> "BearingModel":
        import joblib
        return cls(joblib.load(path))
