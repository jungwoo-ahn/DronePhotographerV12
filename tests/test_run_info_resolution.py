"""`write_run_info` — the placement transform, and the two render knobs added for goal rendering.

The transform itself is what this adapter exists to get right (a second copy of it is a second
chance to render a scene whose subject sits somewhere else); these pin the parts that changed:
a non-square resolution, and the render options a full-resolution render needs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.run_info import write_run_info

SHARED = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")
ROOT = Path("data/trajectories/v7_stage2_renders_lookat075")
_needs = pytest.mark.skipif(not ROOT.is_dir(), reason="no trajectory data in this checkout")


def _any_placement():
    for d in sorted(ROOT.iterdir()):
        if (d / "data.json").exists():
            return d.name, d / "data.json"
    pytest.skip("no placement with data.json")


@_needs
def test_square_resolution_is_the_default_shape(tmp_path):
    name, data = _any_placement()
    ri = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED)).read_text())
    assert ri["options"]["resolution"] == [256, 256]


@_needs
def test_a_pair_is_kept_as_width_height(tmp_path):
    """The goal profile's pixel keys are defined on the 1024x768 frame; a square render would
    change the field of view and make the same goal a different shot."""
    name, data = _any_placement()
    ri = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED,
                                        resolution=(1024, 768))).read_text())
    assert ri["options"]["resolution"] == [1024, 768]


@_needs
def test_extra_options_merge_without_displacing_the_transform(tmp_path):
    name, data = _any_placement()
    p = write_run_info(name, data, tmp_path, shared_root=SHARED, resolution=(1024, 768),
                       extra_options={"render_device": "GPU", "samples": 32})
    ri = json.loads(Path(p).read_text())
    assert ri["options"]["render_device"] == "GPU" and ri["options"]["samples"] == 32
    assert ri["options"]["resolution"] == [1024, 768]
    # the transform keys BlenderDrone reconstructs the placement from must all survive
    for k in ("input_scene", "input_object", "scene_scale", "scale", "rotation_xyz_rad"):
        assert k in ri, k
    assert len(ri["options"]["object_position"]) == 3


@_needs
def test_extra_options_cannot_silently_override_the_object_position(tmp_path):
    """A caller passing `object_position` would move the subject; the merge order puts
    extra_options last, so this test states what that costs and keeps it visible."""
    name, data = _any_placement()
    base = json.loads(Path(write_run_info(name, data, tmp_path, shared_root=SHARED)).read_text())
    p = write_run_info(name, data, tmp_path, shared_root=SHARED,
                       extra_options={"samples": 8})
    assert json.loads(Path(p).read_text())["options"]["object_position"] == base["options"]["object_position"]
