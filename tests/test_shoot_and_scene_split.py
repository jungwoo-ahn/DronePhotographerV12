"""The shoot channel and the scene-level val split.

Both defences were written against bugs that a smoke export actually produced, not against
hypotheticals:

* `shoot_column` first used a SIGNED `goal_idx - start_idx`. Goals are drawn with
  `delta = abs(g - s)`, so a goal can sit *before* the start; the signed difference then
  went negative and latched the whole chunk to 1. Measured: shoot=1 on 62 % of chunks
  against a true arrival rate of 30 %.
* the split used to be the last 5 % of `episode_index` — placement-disjoint but
  scene-complete on both sides (88/88 scene overlap), so it measured a new camera path in
  an already-known room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.common.dataset_base import shoot_column


@dataclass
class _Frame:
    frame_idx: int


@dataclass
class _Window:
    """Just the fields `shoot_column` reads."""
    keyframes: list[_Frame]
    goal_frame: _Frame
    chunk_size: int = 8
    start: _Frame | None = None
    intermediate: list[_Frame] | None = None
    end: _Frame | None = None


def _walk(start: int, goal: int, chunk: int = 8) -> _Window:
    """Reproduce `iter_goal_start_windows`'s walk: step toward the goal, clamp there."""
    d = 1 if goal > start else -1
    idx = [start + d * k for k in range(chunk + 1)]
    idx = [min(i, goal) if d > 0 else max(i, goal) for i in idx]
    fr = [_Frame(i) for i in idx]
    return _Window(keyframes=fr, goal_frame=_Frame(goal), chunk_size=chunk)


def test_latched_forward():
    s = shoot_column(_walk(0, 3))
    assert list(s) == [0, 0, 0, 1, 1, 1, 1, 1]


def test_backward_goal_is_not_all_ones():
    """The measured bug: goal BEFORE start made the signed difference negative."""
    s = shoot_column(_walk(20, 17))
    assert list(s) == [0, 0, 0, 1, 1, 1, 1, 1]
    assert s.sum() < len(s), "a backward window must not latch the whole chunk"


def test_goal_beyond_the_chunk_never_fires():
    assert shoot_column(_walk(0, 25)).sum() == 0


def test_goal_at_start_fires_immediately():
    assert list(shoot_column(_walk(7, 7))) == [1] * 8


@pytest.mark.parametrize("start,goal", [(0, 1), (0, 8), (31, 24), (12, 12), (5, 40)])
def test_always_latched(start, goal):
    """Non-decreasing: it is a STATE, not an event spike."""
    s = shoot_column(_walk(start, goal))
    assert np.all(np.diff(s) >= 0)
    assert set(np.unique(s)) <= {0.0, 1.0}


def test_arrival_is_read_from_the_walk_not_the_indices():
    """Frame indices alone are ambiguous under the clamp; the keyframes are not."""
    w = _walk(10, 12)
    # frames: 10, 11, 12, 12, 12, ... — the clamp repeats the goal
    assert [f.frame_idx for f in w.keyframes] == [10, 11, 12, 12, 12, 12, 12, 12, 12]
    assert list(shoot_column(w)) == [0, 0, 1, 1, 1, 1, 1, 1]


# --------------------------------------------------------------------------- split ----

MANIFEST = Path("configs/val_scenes.json")


@pytest.mark.skipif(not MANIFEST.exists(), reason="no val-scene manifest in this checkout")
def test_manifest_shape():
    m = json.loads(MANIFEST.read_text())
    assert m["level"] == "scene"
    assert len(m["scenes"]) == len(set(m["scenes"])) == 8
    assert set(m["placements_per_scene"]) == set(m["scenes"])
    assert m["val_placements"] == sum(m["placements_per_scene"].values())
    # Not the 8 smallest: a val set of runts would be its own distribution.
    assert max(m["placements_per_scene"].values()) >= 90


# A real dataset root is needed: the base class reads meta/info.json in __init__, so a
# fake path dies there before either guard is reached. v4 is the pre-provenance export,
# which is exactly the case the second guard is for.
LEGACY_ROOT = Path("runs/lerobot_v4")
_needs_legacy = pytest.mark.skipif(
    not (LEGACY_ROOT / "meta" / "info.json").exists(),
    reason="no pre-provenance dataset in this checkout")


@_needs_legacy
def test_val_ratio_is_refused():
    """The old knob must fail loudly, not silently do something different."""
    from src.data.cosmos_camera_dataset import CameraPoseLeRobotDataset
    with pytest.raises(ValueError, match="val_ratio"):
        CameraPoseLeRobotDataset(root=str(LEGACY_ROOT), val_ratio=0.05)


@_needs_legacy
def test_missing_scene_column_is_fatal():
    """No silent fall back to the tail slice: the run would work and the number would
    quietly mean something else, which is how a retired gate survived for weeks."""
    from src.data.cosmos_camera_dataset import CameraPoseLeRobotDataset
    with pytest.raises(ValueError, match="scene"):
        CameraPoseLeRobotDataset(root=str(LEGACY_ROOT))


def test_manifest_default_is_absolute():
    """The training launcher `cd`s into the cosmos-framework root before importing this,
    so a relative default resolves inside the vendored checkout and dies. The 100-iter
    smoke caught exactly that: FileNotFoundError 'configs/val_scenes.json'."""
    from src.data.cosmos_camera_dataset import DEFAULT_VAL_SCENES
    p = Path(DEFAULT_VAL_SCENES)
    assert p.is_absolute(), DEFAULT_VAL_SCENES
    assert p.exists(), DEFAULT_VAL_SCENES


def test_relative_manifest_resolves_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    from src.data import cosmos_camera_dataset as m
    monkeypatch.chdir(tmp_path)                      # anywhere but the repo
    rel = Path("configs/val_scenes.json")
    resolved = rel if rel.is_absolute() else m.V12_ROOT / rel
    assert resolved.exists(), resolved
