"""Pose solver: the forward model is the stage-3 scorer, and the inversion round-trips.

Synthetic geometry (a box of vertices standing on the ground) so the test needs no Blender and
no data: take a pose, score it, feed that profile back as a goal, and require the solver to
recover the pose. That is the property the experiment rests on — if a goal that CAME from a
pose does not solve back to it, no conclusion about an authored goal is possible.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.pose_solver import PlacementGeometry, nearest_recorded_distance

W, H = 1024, 768
FB = (-6.4 / 24.0, 6.4 / 24.0, -4.8 / 24.0, 4.8 / 24.0)     # 24 mm lens, 12.8x9.6 mm sensor


def geom(height=1.7, radius=0.25, n=24):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], 1)
    verts = np.concatenate([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, height)],
                            np.c_[ring, np.full(n, height / 2)]])
    return PlacementGeometry("scene__obj", verts, FB, W, H,
                             np.array([0.0, 0.0, height / 2]), np.zeros(3), height)


@pytest.mark.parametrize("az,el,r,yaw,pitch", [
    (0.0, -10.0, 6.0, 0.0, 0.0),
    (137.0, -25.0, 3.5, 0.10, -0.06),
    (300.0, 5.0, 9.0, -0.12, 0.08),
    (57.0, -40.0, 2.2, 0.05, 0.05),
])
def test_solve_reproduces_the_profile_that_made_the_goal(az, el, r, yaw, pitch):
    """Every key back, on a goal that came from a real pose — the property the whole
    experiment rests on: if a goal built FROM a pose does not solve back to that shot, an
    authored goal says nothing."""
    g = geom()
    pos, f, up = g.pose(az, el, r, yaw, pitch)
    goal = g.profile(pos, f, up)
    assert goal is not None
    out = g.solve(goal)
    assert out["converged"] and out["geom_distance"] < 0.01          # radians
    for k in ("occupancy", "body_in_frame_ratio", "object_center_x", "object_center_y",
              "bbox_x_offset", "bbox_y_offset"):
        assert abs(out["profile"][k] - goal[k]) <= 2.0, (k, out["profile"][k], goal[k])


@pytest.mark.parametrize("az,el,r", [(0.0, -10.0, 6.0), (137.0, -25.0, 3.5), (300.0, 5.0, 9.0)])
def test_an_uncut_or_single_cut_goal_pins_the_camera(az, el, r):
    """Where the goal is NOT cut at both ends the pose itself comes back, to the quantisation
    of the integer profile (1 px, 1 %) — hence a tolerance proportional to the distance."""
    g = geom()
    pos, f, up = g.pose(az, el, r, 0.08, -0.05)
    goal = g.profile(pos, f, up)
    out = g.solve(goal)
    assert not out["pose_underdetermined"]
    assert np.linalg.norm(out["position"] - pos) < 0.02 * r, out


def test_a_goal_cut_at_both_ends_does_not_pin_the_camera():
    """Reported, not hidden: every tilt that keeps the box overflowing gives the same visible
    box, so the profile is reproduced exactly while the pose need not be the original."""
    g = geom()
    pos, f, up = g.pose(57.0, -40.0, 2.2, 0.05, 0.05)
    goal = g.profile(pos, f, up)
    out = g.solve(goal)
    assert out["pose_underdetermined"] and out["occupancy_term"]
    assert out["geom_distance"] < 0.01
    assert abs(out["profile"]["occupancy"] - goal["occupancy"]) <= 2.0


def test_float_twin_matches_the_scorer():
    """The solver optimises on `profile_float`; it must be the same function up to rounding."""
    g = geom()
    for az, el, r, yaw, pitch in [(0, -10, 6, 0, 0), (137, -25, 3.5, 0.1, -0.06), (90, -5, 1.6, 0, 0)]:
        pos, f, up = g.pose(az, el, r, yaw, pitch)
        exact, cont = g.profile(pos, f, up), g.profile_float(pos, f, up)
        for k in ("occupancy", "body_in_frame_ratio", "object_center_x", "object_center_y",
                  "bbox_x_offset", "bbox_y_offset"):
            assert abs(exact[k] - cont[k]) <= 0.5001, (az, el, r, k, exact[k], cont[k])


def test_forward_model_is_the_stage3_scorer():
    """`profile` must BE `compute_v5_scores` of the projected bbox, not a lookalike."""
    from src.scoring.bbox_control import compute_v5_scores
    from src.scoring.projection import cam_to_subject_angles
    g = geom()
    pos, f, up = g.pose(30.0, -12.0, 5.0)
    bbox = g.full_bbox(pos, f, up)
    az, el = cam_to_subject_angles(pos, g.subject_center)
    want = compute_v5_scores(W, H, bbox, az, el)
    got = g.profile(pos, f, up)
    for k, v in want.items():
        assert got[k] == float(v), k


def test_elevation_and_azimuth_are_exact_not_optimised():
    g = geom()
    pos, f, up = g.pose(210.0, -33.0, 4.0, 0.2, -0.1)
    prof = g.profile(pos, f, up)
    assert prof["cam_to_obj_azimuth_deg"] == pytest.approx(210.0, abs=0.6)   # scorer rounds to int
    assert prof["cam_to_obj_elevation_deg"] == pytest.approx(-33.0, abs=0.6)


def test_roll_is_zero():
    g = geom()
    _, f, up = g.pose(45.0, -20.0, 5.0, 0.15, 0.1)
    right = np.cross(f, up)
    assert abs(float(right[2])) < 1e-9          # the camera's right axis stays horizontal
    assert float(np.dot(f, up)) == pytest.approx(0.0, abs=1e-9)


def test_a_cut_goal_still_pins_the_distance_through_occupancy():
    """Cut at both ends, `size` and `aim_y` saturate; occupancy is what is left."""
    g = geom()
    pos, f, up = g.pose(90.0, -5.0, 1.6)                     # close enough to overflow the frame
    goal = g.profile(pos, f, up)
    assert goal["top_cut_frac"] > 0.02 and goal["bot_cut_frac"] > 0.02
    out = g.solve(goal)
    assert out["occupancy_term"], "a frame-filling goal must use the occupancy residual"
    assert abs(out["profile"]["occupancy"] - goal["occupancy"]) <= 2.0
    # Distance is only weakly identified here — that is a property of the GOAL, not a defect:
    # once the subject overflows the frame, occupancy alone carries the distance and it is
    # stored to the nearest percent.
    assert abs(out["r"] - 1.6) < 0.35


def test_the_occupancy_term_tracks_vertical_clipping():
    """On for a cut goal (the visible box stops carrying the size), off for an uncut one
    (where the integer occupancy would only add quantisation noise)."""
    from src.goal_authoring.pose_solver import is_vertically_cut
    g = geom()
    far, _, _ = g.pose(20.0, -10.0, 6.0)
    assert not g.solve(g.profile(*g.pose(20.0, -10.0, 6.0)))["occupancy_term"]
    assert g.solve(g.profile(*g.pose(20.0, -10.0, 1.8)))["occupancy_term"]
    assert is_vertically_cut({"object_center_y": 366.0, "bbox_y_offset": 366.0}, 768)
    assert not is_vertically_cut({"object_center_y": 384.0, "bbox_y_offset": 200.0}, 768)


def test_nearest_recorded_distance():
    data = {"accepted_pairs": [{"trajectory_32f": [{"pos": [0, 0, 0]}, {"pos": [3, 4, 0]}]}]}
    assert nearest_recorded_distance([0, 0, 1], data) == pytest.approx(1.0)
    assert math.isnan(nearest_recorded_distance([0, 0, 0], {}))
