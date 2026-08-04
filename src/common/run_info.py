"""Translate a v7 `data.json` into the run_info the Blender pose renderer expects.

Shared by every rollout script. This lives in one place on purpose: it reconstructs
the *placement transform* (scene scale, object scale, object rotation, object
position) that the training frames were rendered with, and any script that gets it
wrong renders a scene where the subject sits somewhere else — every distance-to-goal
then measures the wrong thing while looking perfectly healthy. A second copy of this
function is a second chance for that to drift.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_run_info(
    placement: str,
    data_path: str | Path,
    out_dir: Path,
    *,
    shared_root: str | Path,
    resolution: int = 256,
    v6_dir: str = "data/vlm_object_placing_v6_260428_061326",
) -> str:
    """Write the run_info JSON for `placement` and return its path.

    `BlenderDrone.from_run_info` wants `input_scene` / `input_object` /
    `options.object_position` / `scene_scale` / `rotation_xyz_rad` / `scale`; v7
    stores `scene_file` / `object_file` / `subject_foot` and keeps the placement
    transform in the v6 placement JSON, so the transform is read back from there.
    Falls back to the v7 `subject_foot` with identity transform only when the v6
    file is missing.
    """
    shared = Path(shared_root)
    data = json.loads(Path(data_path).read_text())
    v6_path = shared / v6_dir / f"{placement}.json"

    scene_scale, scale, rotation = 1.0, 1.0, [0.0, 0.0, 0.0]
    position = data.get("subject_foot", [0.0, 0.0, 0.0])
    if v6_path.exists():
        v6 = json.loads(v6_path.read_text())
        chosen = v6.get("placements", [{}])[int(data.get("placement_idx", 0))]
        scene_scale = float(v6.get("scene_scale", 1.0))
        scale = float(chosen.get("scale", 1.0))
        rotation = [float(v) for v in chosen.get("rotation", [0.0, 0.0, 0.0])]
        position = [float(v) for v in chosen.get("position", position)]

    run_info = {
        "input_scene": str(shared / data["scene_file"]),
        "input_object": str(shared / data["object_file"]),
        "scene_scale": scene_scale,
        "scale": scale,
        "rotation_xyz_rad": rotation,
        "options": {"object_position": position,
                    "resolution": [resolution, resolution]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{placement}.run_info.json"
    path.write_text(json.dumps(run_info, indent=1))
    return str(path)


__all__ = ["write_run_info"]
