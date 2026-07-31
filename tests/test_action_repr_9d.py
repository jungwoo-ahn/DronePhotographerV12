"""Lock the 9D Cosmos-native action rep: [Δtranslation(3), rot6d(6)].

Covers the two things Cosmos does NOT do for us (verified in the framework source):
the Blender->OpenCV camera-frame conversion at encode, and the roll-free
re-projection at decode.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.action_repr import (
    ACTION_DIM,
    apply_action_9d,
    decode_action_9d,
    encode_action_9d,
)
from src.utils.rotation_utils import (
    BLENDER_TO_OPENCV,
    camera_rotation_opencv_from_forward_up,
    forward_up_from_camera_rotation_opencv,
    make_camera_rotation_from_forward_up,
    matrix_from_rot6d,
    project_forward_up_upright,
    roll_angle_deg,
    rot6d_from_matrix,
    rotation_quality,
)


def _level_pose(azimuth_deg: float, elevation_deg: float, position=(0.0, 0.0, 0.0)):
    """A roll-free camera pose aiming at (azimuth, elevation), world up = +Z."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd, up = project_forward_up_upright(forward)
    return np.asarray(position, dtype=np.float64), fwd, up


POSE_PAIRS = [
    ((0.0, 0.0, (0.0, 0.0, 1.5)), (35.0, -12.0, (1.0, -0.5, 1.7))),
    ((120.0, 20.0, (-2.0, 3.0, 2.0)), (95.0, 5.0, (-1.5, 2.0, 1.2))),
    ((-70.0, -30.0, (4.0, 1.0, 3.0)), (-70.0, -30.0, (4.0, 1.0, 3.0))),   # identity
]


@pytest.mark.parametrize("a,b", POSE_PAIRS)
def test_encode_apply_round_trip_is_exact(a, b):
    p0, f0, u0 = _level_pose(*a)
    p1, f1, u1 = _level_pose(*b)
    action = encode_action_9d(p0, f0, u0, p1, f1, u1)
    assert action.shape == (ACTION_DIM,)
    rp, rf, ru = apply_action_9d(p0, f0, u0, action)
    assert np.allclose(rp, p1, atol=1e-5)
    assert np.allclose(rf, f1, atol=1e-6)
    assert np.allclose(ru, u1, atol=1e-6)


def test_identity_action_is_a_no_op():
    p0, f0, u0 = _level_pose(42.0, -8.0, (1.0, 2.0, 3.0))
    action = encode_action_9d(p0, f0, u0, p0, f0, u0)
    assert np.allclose(action[:3], 0.0, atol=1e-6)
    assert np.allclose(matrix_from_rot6d(action[3:]), np.eye(3), atol=1e-6)


def test_rot6d_is_the_first_two_columns_cosmos_convention():
    _, f, u = _level_pose(31.0, 17.0)
    rot = camera_rotation_opencv_from_forward_up(f, u)
    six = rot6d_from_matrix(rot)
    assert np.allclose(six[:3], rot[:, 0], atol=1e-6)
    assert np.allclose(six[3:], rot[:, 1], atol=1e-6)
    assert np.allclose(matrix_from_rot6d(six), rot, atol=1e-6)


def test_matrix_from_rot6d_orthonormalizes_noisy_model_output():
    _, f, u = _level_pose(10.0, -5.0)
    six = rot6d_from_matrix(camera_rotation_opencv_from_forward_up(f, u))
    noisy = six + np.random.default_rng(0).normal(scale=0.05, size=6).astype(np.float32)
    rot = matrix_from_rot6d(noisy)                       # normalize_matrix=True default
    det_err, orth_err = rotation_quality(rot)
    assert det_err < 1e-6 and orth_err < 1e-6


def test_opencv_frame_conversion_matches_blender_times_flip():
    _, f, u = _level_pose(-23.0, 9.0)
    assert np.allclose(
        camera_rotation_opencv_from_forward_up(f, u),
        make_camera_rotation_from_forward_up(f, u) @ BLENDER_TO_OPENCV,
        atol=1e-6,
    )
    # OpenCV columns are [right, down, forward]
    rot = camera_rotation_opencv_from_forward_up(f, u)
    assert np.allclose(rot[:, 2], f, atol=1e-6)
    assert np.allclose(rot[:, 1], -u, atol=1e-6)
    rf, ru = forward_up_from_camera_rotation_opencv(rot)
    assert np.allclose(rf, f, atol=1e-6) and np.allclose(ru, u, atol=1e-6)


def test_decode_rejects_wrong_dim():
    with pytest.raises(ValueError):
        decode_action_9d(np.zeros(5, dtype=np.float32))


def test_upright_projection_removes_roll_but_keeps_aim():
    _, f, u = _level_pose(60.0, -15.0)
    # bank the camera about its own forward axis -> real roll
    c, s = np.cos(np.radians(25.0)), np.sin(np.radians(25.0))
    axis = f / np.linalg.norm(f)
    rolled = (u * c + np.cross(axis, u) * s + axis * np.dot(axis, u) * (1 - c))
    assert abs(roll_angle_deg(f, rolled)) > 20.0
    pf, pu = project_forward_up_upright(f, rolled)
    assert np.allclose(pf, f, atol=1e-6)                 # aim preserved exactly
    assert abs(roll_angle_deg(pf, pu)) < 1e-4            # roll removed


def test_apply_upright_false_preserves_model_roll_for_logging():
    """upright=False must NOT silently fix roll — that path is how we measure it."""
    p0, f0, u0 = _level_pose(0.0, 0.0, (0.0, 0.0, 1.0))
    rot_cur = camera_rotation_opencv_from_forward_up(f0, u0)
    # a pure roll about the camera's forward (OpenCV +Z) axis
    ang = np.radians(20.0)
    delta = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                      [np.sin(ang), np.cos(ang), 0.0],
                      [0.0, 0.0, 1.0]])
    action = np.concatenate([np.zeros(3), rot6d_from_matrix(delta)]).astype(np.float32)
    _, f_raw, u_raw = apply_action_9d(p0, f0, u0, action, upright=False)
    assert abs(roll_angle_deg(f_raw, u_raw)) > 15.0      # roll is visible
    _, f_up, u_up = apply_action_9d(p0, f0, u0, action, upright=True)
    assert abs(roll_angle_deg(f_up, u_up)) < 1e-4        # and removed when asked
    assert np.allclose(f_raw, f_up, atol=1e-6)           # same aim either way


def test_chunk_of_actions_composes_back_onto_the_endpoint():
    poses = [_level_pose(az, el, (0.1 * i, -0.2 * i, 1.5 + 0.05 * i))
             for i, (az, el) in enumerate([(0, 0), (12, -4), (25, -9), (30, -14)])]
    pos, fwd, up = poses[0]
    for prev, nxt in zip(poses, poses[1:]):
        action = encode_action_9d(*prev, *nxt)
        pos, fwd, up = apply_action_9d(pos, fwd, up, action)
    assert np.allclose(pos, poses[-1][0], atol=1e-5)
    assert np.allclose(fwd, poses[-1][1], atol=1e-6)
