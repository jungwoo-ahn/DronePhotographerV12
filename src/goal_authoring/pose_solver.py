"""Goal profile -> camera pose, exactly, for a given placement.

A goal names a shot; a pose realises it. The map from pose to profile is
`src/scoring/projection.py:score_pose` — the stage-3 scorer itself, which reproduces the
stored scores on 1312/1312 frames — so this module INVERTS that map rather than a proxy of it.
It needs the two Blender-only inputs `score_pose` takes (the subject's world mesh vertices and
the camera frame bounds), dumped once per placement by `scripts/dump_placement_geometry.py`.

The profile pins five geometric degrees of freedom and the camera has exactly five (position on
the viewing sphere, distance, and where the optical axis points, roll being zero throughout the
data), so the inversion is well posed:

    az, el   <- the goal's bearing (through the TARGET object's facing) and elevation: exact,
                no search — they are closed-form functions of the pose.
    r        <- apparent size
    yaw, pitch offsets <- where the subject sits in the frame

The last three are found by least squares on the profile's own geometry decode
(`reward.profile_to_geometry`: size and aim in radians), with an occupancy term that carries
the width when a goal is cut at both ends and `size`/`aim_y` no longer discriminate.

Transferring a goal to a DIFFERENT subject is where the bearing convention earns its keep: the
goal stores `subject_bearing_deg`, and the world azimuth that realises it is
`world_azimuth_deg(bearing, target_object)`. The same authored goal therefore means "seen from
the subject's front-right" in every scene, not "seen from world azimuth 57°".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.common.facing import world_azimuth_deg
from src.common.goal_space import SUBJECT_BEARING_KEY
from src.common.reward import CameraIntrinsics, _geometry_distance, profile_to_geometry
from src.scoring.bbox_control import compute_v5_scores
from src.scoring.projection import cam_to_subject_angles, project_verts_to_bbox

# Occupancy residual weight, in radians per unit of area fraction, used ONLY when the goal is
# cut vertically. `size` and `aim_y` are read off the VISIBLE box, so as soon as one end is
# clipped they stop carrying the subject's true extent: with the top cut they depend on the
# bottom edge alone, which leaves a one-parameter family of (distance, tilt) that produces the
# identical visible box — measured, the solver lands on r = 4.6 for a goal made at r = 3.5 with
# residual 1e-13. Width is what still moves, and occupancy carries it.
#
# For an uncut goal the term is left OUT on purpose: occupancy is stored as an INTEGER percent,
# so at a wide shot one quantisation step is a large relative change, and it would drag the
# distance away from the sub-pixel-accurate size and aim terms.
W_OCC = 0.5


def is_vertically_cut(goal: Mapping[str, float], height: float) -> bool:
    """Does the goal's box run off the top or the bottom of the frame?

    Read from the crop fractions when they are there, and from the visible box otherwise —
    a goal parsed back from a training prompt carries `visible`, but one perturbed by hand
    may not.
    """
    if "top_cut_frac" in goal or "bot_cut_frac" in goal:
        return (float(goal.get("top_cut_frac", 0.0)) > 0.02
                or float(goal.get("bot_cut_frac", 0.0)) > 0.02)
    cy, hy = float(goal.get("object_center_y", 0.0)), float(goal.get("bbox_y_offset", 0.0))
    return (cy - hy) <= 1.0 or (cy + hy) >= height - 1.0


def _unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector from the CAMERA towards the subject for a viewing direction (az, el)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _basis_roll_free(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right, up) for a roll-free camera looking along `forward` — the data has zero roll
    (`scripts/check_roll.py`), and `apply_action_9d` re-projects upright, so a solved pose
    with roll would not be reachable by the policy either."""
    f = forward / np.linalg.norm(forward)
    z = np.array([0.0, 0.0, 1.0])
    up = z - np.dot(z, f) * f
    n = np.linalg.norm(up)
    if n < 1e-8:                       # looking straight up/down: pick any consistent roll
        up = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), f) * f
        n = np.linalg.norm(up)
    up = up / n
    return np.cross(f, up), up


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of `v` about a unit `axis`."""
    a = axis / np.linalg.norm(axis)
    return (v * math.cos(ang) + np.cross(a, v) * math.sin(ang)
            + a * np.dot(a, v) * (1.0 - math.cos(ang)))


@dataclass
class PlacementGeometry:
    """Everything needed to score a pose in one placement, exactly."""
    placement: str
    verts: np.ndarray                 # (N, 3) world-space subject mesh vertices
    frame_bounds: tuple[float, float, float, float]
    width: int
    height: int
    subject_center: np.ndarray
    subject_foot: np.ndarray
    subject_height: float

    @classmethod
    def load(cls, path: str | Path) -> "PlacementGeometry":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                placement=str(z["placement"]),
                verts=np.asarray(z["verts"], dtype=np.float64),
                frame_bounds=tuple(float(v) for v in z["frame_bounds"]),
                width=int(z["resolution"][0]), height=int(z["resolution"][1]),
                subject_center=np.asarray(z["subject_center"], dtype=np.float64),
                subject_foot=np.asarray(z["subject_foot"], dtype=np.float64),
                subject_height=float(z["subject_height"]),
            )

    @property
    def object_key(self) -> str:
        return self.placement.split("__", 1)[1] if "__" in self.placement else self.placement

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_render(self.width, self.height)

    # ---- forward model ---------------------------------------------------------------
    def _scores_float(self, bbox, az: float, el: float) -> dict[str, float]:
        """`compute_v5_scores` without the integer rounding.

        The scorer emits INTEGERS, which makes the residual piecewise constant: a numerical
        Jacobian steps by ~1e-8 and sees exactly zero, so least squares reports success without
        having moved (measured: it returned the initial distance on every case). The optimiser
        therefore runs on this float twin and everything reported to the caller still comes
        from the integer scorer, so the numbers stay the ones the training pipeline would have
        written. The two agree to the rounding, which `test_float_twin_matches_the_scorer` pins.
        """
        W, H = float(self.width), float(self.height)
        x0, y0, x1, y1 = (float(v) for v in bbox)
        fw, fh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        full_area = fw * fh
        cx0, cx1 = max(0.0, min(x0, W)), max(0.0, min(x1, W))
        cy0, cy1 = max(0.0, min(y0, H)), max(0.0, min(y1, H))
        cw, ch = max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)
        clipped = cw * ch
        extreme = fw > 4.0 * W or fh > 4.0 * H or full_area > 16.0 * W * H
        denom = (min(fw, 4.0 * W) * min(fh, 4.0 * H)) if extreme else full_area
        return {
            "occupancy": min(100.0, max(0.0, 100.0 * clipped / (W * H))),
            "body_in_frame_ratio": (min(100.0, max(0.0, 100.0 * clipped / denom)) if denom > 0 else 0.0),
            "cam_to_obj_azimuth_deg": float(az) % 360.0,
            "cam_to_obj_elevation_deg": max(-90.0, min(90.0, float(el))),
            "object_center_x": 0.5 * (cx0 + cx1),
            "object_center_y": 0.5 * (cy0 + cy1),
            "bbox_x_offset": 0.5 * cw,
            "bbox_y_offset": 0.5 * ch,
        }

    def profile_float(self, position, forward, up) -> dict[str, float] | None:
        """The float twin of `profile` — what the solver optimises on."""
        bbox = self.full_bbox(position, forward, up)
        if bbox is None:
            return None
        az, el = cam_to_subject_angles(position, self.subject_center)
        return self._scores_float(bbox, az, el)

    def full_bbox(self, position, forward, up) -> list[float] | None:
        return project_verts_to_bbox(self.verts, position, forward, up,
                                     self.width, self.height, self.frame_bounds)

    def profile(self, position, forward, up) -> dict[str, float] | None:
        """The 8-key V5 profile a pose achieves, plus the crop keys — the stage-3 scorer."""
        bbox = self.full_bbox(position, forward, up)
        if bbox is None:
            return None
        az, el = cam_to_subject_angles(position, self.subject_center)
        out = {k: float(v) for k, v in
               compute_v5_scores(self.width, self.height, bbox, az, el).items()}
        y0, y1 = float(bbox[1]), float(bbox[3])
        span = y1 - y0
        if span > 0:
            out["top_cut_frac"] = max(0.0, -y0) / span
            out["bot_cut_frac"] = max(0.0, y1 - self.height) / span
            out["visible_frac"] = (min(y1, self.height) - max(y0, 0.0)) / span
            out["head_in_frame"] = 1.0 if y0 >= 0.0 else 0.0
        bearing = None
        if self.object_key:
            from src.common.facing import subject_bearing_deg
            bearing = subject_bearing_deg(az, self.object_key)
        if bearing is not None:
            out[SUBJECT_BEARING_KEY] = float(bearing)
        out["_bbox_full"] = bbox
        return out

    # ---- pose parameterisation -------------------------------------------------------
    def pose(self, az_deg: float, el_deg: float, r: float,
             yaw_off: float = 0.0, pitch_off: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Camera pose from (viewing direction, distance, aim offsets in radians)."""
        u = _unit(az_deg, el_deg)
        position = self.subject_center - r * u
        f = _rotate(u, np.array([0.0, 0.0, 1.0]), yaw_off)       # pan
        right, _ = _basis_roll_free(f)
        f = _rotate(f, right, pitch_off)                          # tilt
        right, up = _basis_roll_free(f)
        return position, f / np.linalg.norm(f), up

    # ---- inversion -------------------------------------------------------------------
    def solve(self, goal: Mapping[str, float], *,
              r_bounds: tuple[float, float] = (0.4, 60.0),
              max_offset_deg: float = 35.0) -> dict:
        """Pose that realises `goal` in this placement. Returns the pose, the profile it
        actually achieves, and the residual — the caller is expected to look at the residual
        rather than assume the goal was reachable."""
        from scipy.optimize import least_squares

        intr = self.intrinsics
        g_geo = profile_to_geometry(goal, intr)
        occ_goal = float(goal.get("occupancy", 0.0)) / 100.0

        use_occ = is_vertically_cut(goal, self.height)
        el = float(goal.get("cam_to_obj_elevation_deg", 0.0))
        az = goal.get("cam_to_obj_azimuth_deg")
        if az is None:
            b = goal.get(SUBJECT_BEARING_KEY)
            if b is None:
                raise ValueError("goal has neither cam_to_obj_azimuth_deg nor subject_bearing_deg")
            az = world_azimuth_deg(float(b), self.object_key)
            if az is None:
                raise ValueError(f"no facing entry for {self.object_key!r}; bearing is unusable here")
        az = float(az)
        # An authored goal carries the SUBJECT-relative bearing, not a world azimuth, so
        # `profile_to_geometry` would read the missing key as 0 deg and the reported distance
        # would be dominated by a view difference that does not exist (measured: a median
        # 1.50 rad of pure artefact). The direction the goal asks for in THIS placement is the
        # one just derived, so state it.
        g_geo = profile_to_geometry({**dict(goal), "cam_to_obj_azimuth_deg": az}, intr)

        # Start from the distance the apparent size implies for a subject of this height, the
        # closed-form answer when nothing is clipped.
        size = max(float(g_geo["size"]), 1e-3)
        r0 = min(max((self.subject_height / 2.0) / math.tan(size), r_bounds[0]), r_bounds[1])

        def residual(p):
            r, yaw, pitch = p
            pos, f, up = self.pose(az, el, r, yaw, pitch)
            prof = self.profile_float(pos, f, up)
            if prof is None:                       # subject behind the camera
                return np.full(4 if use_occ else 3, 10.0)
            a = profile_to_geometry(prof, intr)
            out = [a["size"] - g_geo["size"],
                   a["aim_x"] - g_geo["aim_x"],
                   a["aim_y"] - g_geo["aim_y"]]
            if use_occ:
                out.append(W_OCC * (prof["occupancy"] / 100.0 - occ_goal))
            return np.array(out)

        m = math.radians(max_offset_deg)
        # Seed from a coarse sweep over distance. The closed form above assumes nothing is
        # clipped; for a goal cut at both ends it can be far off, and least squares from a bad
        # start settles into a local minimum where occupancy is met by moving the aim instead.
        seeds = sorted({min(max(r0 * s, r_bounds[0]), r_bounds[1])
                        for s in (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0)})
        seeds = sorted(seeds, key=lambda rr: float(np.linalg.norm(residual([rr, 0.0, 0.0]))))
        best = None
        for seed in seeds[:3]:
            cand = least_squares(residual, [seed, 0.0, 0.0],
                                 bounds=([r_bounds[0], -m, -m], [r_bounds[1], m, m]),
                                 xtol=1e-12, ftol=1e-12, max_nfev=300)
            if best is None or float(np.linalg.norm(cand.fun)) < float(np.linalg.norm(best.fun)):
                best = cand
        r, yaw, pitch = best.x
        pos, f, up = self.pose(az, el, r, yaw, pitch)
        prof = self.profile(pos, f, up)
        return {
            "position": pos, "forward": f, "up": up,
            "az": az, "el": el, "r": float(r),
            "yaw_off_deg": math.degrees(yaw), "pitch_off_deg": math.degrees(pitch),
            "profile": prof,
            "residual": float(np.linalg.norm(best.fun)),
            "occupancy_term": bool(use_occ),
            # A goal cut at BOTH ends leaves the tilt free: within the range that keeps the
            # box overflowing, every tilt yields the identical visible box, so the profile is
            # reproduced exactly while the pose is not the one that made it. Said out loud
            # because it is a property of the goal, not a solver failure.
            "pose_underdetermined": bool(float(goal.get("top_cut_frac", 0.0)) > 0.02
                                         and float(goal.get("bot_cut_frac", 0.0)) > 0.02),
            "geom_distance": (float(_geometry_distance(profile_to_geometry(prof, intr), g_geo))
                              if prof else float("inf")),
            "converged": bool(best.success),
        }


def nearest_recorded_distance(position, data: Mapping) -> float:
    """Distance from a solved camera position to the closest camera the DATA ever used here.

    Every recorded pose passed v7's `is_camera_valid` ray-casts (not inside geometry, not
    through a wall); a solved pose has passed nothing. This does not prove validity — it is a
    cheap flag for "this shot asks the camera to stand somewhere the sampler never stood",
    which is exactly where a render comes back showing a wall.
    """
    pts = [np.asarray(fr["pos"], dtype=np.float64)
           for pair in data.get("accepted_pairs", []) for fr in pair.get("trajectory_32f", [])]
    if not pts:
        return float("nan")
    return float(np.min(np.linalg.norm(np.stack(pts) - np.asarray(position, dtype=np.float64), axis=1)))


__all__ = ["PlacementGeometry", "nearest_recorded_distance", "is_vertically_cut", "W_OCC"]
