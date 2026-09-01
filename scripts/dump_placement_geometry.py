"""Dump a placement's exact subject geometry so a goal can be SOLVED without Blender.

    blender/blender -b -P scripts/dump_placement_geometry.py -- \
        --run-info runs/goal_render/_run_info/<placement>.run_info.json \
        --data data/trajectories/.../<placement>/data.json \
        --out runs/goal_render/_geom/<placement>.npz

`src/scoring/projection.py:score_pose` reproduces the stored stage-3 scores exactly (1312/1312
frames) given the subject's world-space mesh vertices and the camera frame bounds — but those
two are Blender-only, which is why every downstream script has had to approximate. Extracting
them ONCE per placement makes the pose->profile map exact, so `pose_solver` can invert it and
the achieved profile of a solved pose is known before anything is rendered.

Approximating instead is not good enough here: a cylinder proxy fitted per placement leaves a
~25 px mean bbox error (elliptic + z offsets: ~20 px) on the 1024x768 frame, which is several
percent of occupancy and can move a placement band across its boundary.

Runs inside Blender (needs `bpy`), and builds the scene through the SAME `BlenderDrone` the
renderer uses, so the placement transform cannot drift from what gets rendered. Subject objects
are the names v7 recorded in `data.json:subject_names`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHARED = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from src.drones.blender_drone import BlenderDrone  # noqa: E402


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-info", required=True)
    ap.add_argument("--data", required=True, help="the placement's v7 data.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-verts", type=int, default=50000,
                    help="subsample cap. v7 used 5000 for sampling; the bbox is an extreme "
                         "statistic, so a denser sample only helps and costs nothing here")
    cut = argv.index("--") + 1 if "--" in argv else len(argv)
    return ap.parse_args(argv[cut:])


def camera_frame_bounds(scene, cam_obj):
    """(min_x, max_x, min_y, max_y) of the camera frame normalized to z=1 — verbatim the rule
    `v7_sample_pairs_smoke._get_camera_frame_bounds` used to score the training data."""
    frame = cam_obj.data.view_frame(scene=scene)
    view_z = abs(frame[0].z) or 1.0
    xs = [f.x / view_z for f in frame]
    ys = [f.y / view_z for f in frame]
    return float(min(xs)), float(max(xs)), float(min(ys)), float(max(ys))


def main(argv):
    args = parse_args(argv)
    data = json.loads(Path(args.data).read_text())
    drone = BlenderDrone.from_run_info(args.run_info)

    names = list(data.get("subject_names") or [])
    chunks = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            continue
        mw = np.array(obj.matrix_world, dtype=np.float64)
        local = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
        if local.size == 0:
            continue
        homog = np.concatenate([local, np.ones((local.shape[0], 1))], axis=1)
        chunks.append((homog @ mw.T)[:, :3])
    if not chunks:
        raise RuntimeError(f"no mesh vertices for subject_names={names!r} in {args.run_info}")
    verts = np.concatenate(chunks, axis=0)
    if verts.shape[0] > args.max_verts:
        verts = verts[np.linspace(0, verts.shape[0] - 1, args.max_verts).astype(np.int64)]

    fb = camera_frame_bounds(drone.scene, drone.camera)
    res = drone.scene.render.resolution_x, drone.scene.render.resolution_y
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        verts=verts.astype(np.float32),
        frame_bounds=np.array(fb, dtype=np.float64),
        resolution=np.array(res, dtype=np.int64),
        subject_center=np.array(data["subject_center"], dtype=np.float64),
        subject_foot=np.array(data["subject_foot"], dtype=np.float64),
        subject_height=np.float64(data["subject_height"]),
        placement=np.array(data.get("placement", Path(args.data).parent.name)),
    )
    print(f"[geom] {out} verts={verts.shape[0]} frame_bounds={fb} res={res}")


if __name__ == "__main__":
    main(sys.argv)
