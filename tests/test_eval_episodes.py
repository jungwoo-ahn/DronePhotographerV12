"""The shared episode manifest — the artefact that makes two repos' numbers comparable.

The manifest exists because the closed-loop score is dominated by how far the start is
(`corr(d_start, improvement) = 0.66` over 138 rollouts), so two policies measured on two episode
sets cannot be put in one table however identical the protocol. What has to hold, then, is that
replaying the manifest gives back *the same shots* — and that the `d_start` written in the file is
the checksum which proves it, in this repo and in the baselines' port alike.

These reconstruct episodes the way `closed_loop_eval.episodes_from_manifest` does, without
importing that module: it pulls in the whole cosmos framework at import time, which the analysis
venv does not have. The logic under test is the manifest's, not the caller's.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.common.annotations import iter_goal_start_windows
from src.common.dataset_base import _window_object
from src.common.facing import sector8
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector
from src.common.reward import _geometry_distance, pose_to_geometry

MANIFEST = Path("configs/eval_episodes_v1.json")
_needs = pytest.mark.skipif(not MANIFEST.exists(), reason="no shared manifest in this checkout")

KEY_RE = re.compile(r"^(?P<placement>.+):p(?P<pair>\d{2}):s(?P<start>\d{2}):g(?P<goal>\d{2})$")


def _doc():
    return json.loads(MANIFEST.read_text())


def _d_start(window) -> float:
    gv = window.goal_frame
    a = pose_to_geometry(window.start.camera_position, window.start.camera_forward,
                         window.start.camera_up, subject_center=gv.subject_center,
                         subject_height=gv.subject_height)
    g = pose_to_geometry(gv.camera_position, gv.camera_forward, gv.camera_up,
                         subject_center=gv.subject_center, subject_height=gv.subject_height)
    return float(_geometry_distance(a, g))


@_needs
def test_manifest_shape():
    d = _doc()
    for k in ("_contract", "_verify", "version", "source_root", "val_scenes", "chunk_size",
              "max_per_pair", "seed", "sector_mix", "episodes"):
        assert k in d, k
    eps = d["episodes"]
    assert len(eps) == d["n_episodes"] == len({e["key"] for e in eps})
    # The window pool depends on BOTH of these, so they belong to the manifest and not to
    # whoever runs it — the baselines' harness enumerates with max_per_pair=0 and would
    # otherwise be selecting from a different pool under the same placement names.
    assert d["chunk_size"] == 8 and d["max_per_pair"] == 2


@_needs
def test_every_scene_is_represented_and_sectors_are_balanced():
    """An alphabetical first-N collapses into one room; that is what this replaces."""
    d = _doc()
    val = set(json.loads(Path(d["val_scenes"]).read_text())["scenes"])
    scenes = {e["scene"] for e in d["episodes"]}
    assert scenes <= val
    assert len(scenes) >= 6, f"only {len(scenes)} of the 8 held-out scenes appear"
    counts = list(d["sector_mix"].values())
    assert len(d["sector_mix"]) == 8
    assert max(counts) - min(counts) <= 2, d["sector_mix"]


@_needs
def test_keys_parse_and_agree_with_their_fields():
    for e in _doc()["episodes"]:
        m = KEY_RE.match(e["key"])
        assert m, e["key"]
        assert m["placement"] == e["placement"]
        assert int(m["pair"]) == e["pair_idx"]
        assert int(m["start"]) == e["start_frame_idx"]
        assert int(m["goal"]) == e["goal_frame_idx"]


@_needs
def test_every_episode_is_self_consistent():
    for e in _doc()["episodes"]:
        assert e["delta"] == abs(e["goal_frame_idx"] - e["start_frame_idx"])
        assert e["n_chunks"] == int(np.ceil(e["delta"] / 8))
        assert sector8(e["bearing"]) == e["sector"]
        assert len(e["goal_vec"]) == len(DEFAULT_GOAL_KEYS)
        assert e["goal_keys"] == list(DEFAULT_GOAL_KEYS)
        assert e["d_start"] > 0.0
        for pose in (e["start_pose"], e["goal_pose"]):
            assert len(pose["position"]) == 3 and len(pose["forward"]) == 3


@_needs
def test_d_start_is_reproducible_from_the_materialised_poses():
    """The cross-repo checksum: a repo that never runs `iter_goal_start_windows` can still
    verify it is rolling out the same episode, from the poses alone."""
    for e in _doc()["episodes"][:20]:
        a = pose_to_geometry(e["start_pose"]["position"], e["start_pose"]["forward"],
                             e["start_pose"]["up"], subject_center=e["subject_center"],
                             subject_height=e["subject_height"])
        g = pose_to_geometry(e["goal_pose"]["position"], e["goal_pose"]["forward"],
                             e["goal_pose"]["up"], subject_center=e["subject_center"],
                             subject_height=e["subject_height"])
        assert float(_geometry_distance(a, g)) == pytest.approx(e["d_start"], abs=1e-9)


@_needs
def test_replay_finds_the_same_window_in_the_data():
    """The other half: enumerate the pool the manifest declares and find the named window.
    A miss means the manifest and the checkout disagree about the data, which is exactly the
    failure the replay must raise rather than paper over."""
    d = _doc()
    checked = 0
    for e in d["episodes"]:
        path = Path(e["data_json"])
        if not path.exists():
            continue
        want = (e["pair_idx"], e["start_frame_idx"], e["goal_frame_idx"])
        found = next((w for w in iter_goal_start_windows(
            path, chunk_size=d["chunk_size"], max_per_pair=d["max_per_pair"])
            if (w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) == want), None)
        assert found is not None, f"{e['key']}: window {want} is not in the declared pool"
        assert _d_start(found) == pytest.approx(e["d_start"], abs=1e-9)
        g = goal_vector(found.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(found))
        assert np.isfinite(g).all()
        assert g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)] == pytest.approx(e["bearing"])
        assert list(np.asarray(g, dtype=float)) == pytest.approx(e["goal_vec"])
        checked += 1
        if checked >= 12:
            break
    if checked == 0:
        pytest.skip("trajectory tree not present in this checkout")


@_needs
def test_the_manifest_is_not_the_thing_that_drifts():
    """`iter_goal_start_windows` uses an RNG only to SUB-SELECT when max_per_pair > 0, seeded by
    placement name and pair index — so two enumerations of the same pool agree. Pin it, because
    the manifest's whole promise rests on it."""
    d = _doc()
    e = next((x for x in d["episodes"] if Path(x["data_json"]).exists()), None)
    if e is None:
        pytest.skip("trajectory tree not present in this checkout")
    key = lambda ws: [(w.pair_idx, w.start_frame_idx, w.goal_frame.frame_idx) for w in ws]
    a = list(iter_goal_start_windows(Path(e["data_json"]), chunk_size=d["chunk_size"],
                                     max_per_pair=d["max_per_pair"]))
    b = list(iter_goal_start_windows(Path(e["data_json"]), chunk_size=d["chunk_size"],
                                     max_per_pair=d["max_per_pair"]))
    assert key(a) == key(b) and a
