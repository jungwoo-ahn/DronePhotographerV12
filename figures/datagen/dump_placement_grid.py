"""Re-run the placement worker's floor-candidate search on one scene and log every cell, for the data-gen figure.

Runs inside Blender with the scene open:
    blender -b <scene.blend> -t 16 -P figures/datagen/dump_placement_grid.py -- \
        --object_file data/objects/<obj> --scene_scale 2.0 --out <dir>

It calls the real `mode_discover_candidates` of DronePhotographer/scripts/vlm_place_worker.py (same object import,
metric normalisation and scene-scale override as the placement run). `find_flat_candidates` is wrapped: before the
real call, the same cell grid is rebuilt and every cell is classified the way the function does it
(out of view / uneven / not flat / not visible / scored, with its score). Then a top-down orthographic view and the
scene-camera view are rendered. The VLM interest region of the original run was not saved, so the interest term
is computed without it (region score from no region).

Writes <out>/cells.json, <out>/topdown.png (+ its world extent in cells.json), <out>/scene_cam.png.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

OLD = Path("/home/nas_main/jungwooahn/projects/DronePhotographer")
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("worker", OLD / "scripts/vlm_place_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
from src.blender import surfaces as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = Path(argv[argv.index("--out") + 1])
OUT.mkdir(parents=True, exist_ok=True)
cmd = {"object_file": argv[argv.index("--object_file") + 1],
       "scene_scale": float(argv[argv.index("--scene_scale") + 1]),
       "top_k": 3, "min_candidate_distance": 0.8, "mode": "discover_candidates"}
REC = {}


def classify(scene, depsgraph, bounds_min, bounds_max, obj_size, ignore_names, camera_target=None,
             scene_camera_pos=None, visible_centroid=None, visible_hits=None, scene_meshes=None,
             interest_region=None, **_):
    """The cell loop of surfaces.find_flat_candidates, recording each cell's fate instead of dropping it."""
    height_tol, slope_cos = S.HEIGHT_TOL, math.cos(math.radians(S.MAX_SLOPE_DEG))
    obj_radius = max(obj_size.x, obj_size.y) * 0.5
    x_min, x_max = bounds_min.x + obj_radius, bounds_max.x - obj_radius
    y_min, y_max = bounds_min.y + obj_radius, bounds_max.y - obj_radius
    if x_min >= x_max or y_min >= y_max:
        x_min, x_max, y_min, y_max = bounds_min.x, bounds_max.x, bounds_min.y, bounds_max.y
    area = max(0.01, (bounds_max.x - bounds_min.x) * (bounds_max.y - bounds_min.y))
    step = max(max(S.CELL_SIZE * 0.5, obj_radius * 0.5), math.sqrt(area / S.MAX_GRID_SAMPLES))
    H = bounds_max.z - bounds_min.z
    zcfg = [(bounds_min.z + 0.3, -1), (bounds_min.z + 0.5, 1), (bounds_min.z + 0.9, 1), (bounds_min.z + 1.2, 1),
            (bounds_min.z + H * 0.5, -1), (bounds_min.z - 0.5, 1), (bounds_max.z + max(5.0, obj_size.z * 2.0), -1)]
    cells = {}
    for zi, (z0, d) in enumerate(zcfg):
        for x, y in S.iter_grid(x_min, x_max, y_min, y_max, step):
            hit, loc, n, _, obj, _ = scene.ray_cast(depsgraph, Vector((x, y, z0)), Vector((0, 0, d)))
            if not hit or obj is None or obj.name in ignore_names or abs(n.z) < slope_cos:
                continue
            k = (int(math.floor((loc.x - bounds_min.x) / S.CELL_SIZE)),
                 int(math.floor((loc.y - bounds_min.y) / S.CELL_SIZE)), zi)
            c = cells.setdefault(k, {"count": 0, "sum": Vector((0, 0, 0)), "h": []})
            c["count"] += 1
            c["sum"] += loc
            c["h"].append(loc.z)
    xs = [h.x for h in visible_hits] if visible_hits else []
    ys = [h.y for h in visible_hits] if visible_hits else []
    vis_bbox = None
    if xs:
        m = obj_radius + 1.0
        vis_bbox = (min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
    ref = visible_centroid or camera_target or (bounds_min + bounds_max) / 2
    ref_xy = Vector((ref.x, ref.y))
    diag = math.hypot(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y)
    vdiag = math.hypot(vis_bbox[1] - vis_bbox[0], vis_bbox[3] - vis_bbox[2]) if vis_bbox else diag
    B = max(1, min(5, round(vdiag / 3.0)))
    ground = S._identify_ground_mesh_names(scene_meshes or [], max(0.01, H), vis_bbox)
    out = []
    for c in cells.values():
        ctr = c["sum"] / c["count"]
        rec = {"x": ctr.x, "y": ctr.y, "z": ctr.z}
        if vis_bbox and not (vis_bbox[0] <= ctr.x <= vis_bbox[1] and vis_bbox[2] <= ctr.y <= vis_bbox[3]):
            rec["status"] = "out_of_view"
        elif max(c["h"]) - min(c["h"]) > height_tol:
            rec["status"] = "uneven"
        elif not S.area_is_flat(scene, depsgraph, ctr, obj_size, ctr.z + 0.3, slope_cos, ignore_names,
                                height_tol=height_tol):
            rec["status"] = "not_flat"
        elif scene_camera_pos is not None and not S.is_visible_from_camera(
                scene, depsgraph, scene_camera_pos, Vector((ctr.x, ctr.y, ctr.z + 0.4)),
                ignore_names=ignore_names, tolerance=0.5):
            rec["status"] = "not_visible"
        else:
            spread = max(c["h"]) - min(c["h"])
            region = S._compute_region_score(Vector((ctr.x, ctr.y)), interest_region, vdiag, B)
            variety = S._compute_variety_score(scene, depsgraph, ctr, ground, ignore_names, n_dirs=16)
            rec["status"] = "scored"
            rec["score"] = (0.15 * (1 - min(1, (Vector((ctr.x, ctr.y)) - ref_xy).length / (diag * 0.5 + 0.01)))
                            + 0.15 * (1 - min(1, spread / height_tol))
                            + 0.25 * (1 - (ctr.z - bounds_min.z) / max(0.01, H))
                            + 0.45 * (0.6 * region + 0.4 * variety))
        out.append(rec)
    return out, vis_bbox, step


_orig = worker.find_flat_candidates


def wrapped(*a, **kw):
    if "cells" not in REC:
        names = ["scene", "depsgraph", "bounds_min", "bounds_max", "obj_size"]
        args = dict(zip(names, a))
        args.update(kw)
        cells, vb, step = classify(**args)
        REC.update(cells=cells, vis_bbox=vb, cell_size=S.CELL_SIZE, step=step,
                   ignore=sorted(kw.get("ignore_names", [])),
                   scene_cam=list(kw["scene_camera_pos"]) if kw.get("scene_camera_pos") is not None else None)
    res = _orig(*a, **kw)
    REC.setdefault("top_k", res)
    return res


worker.find_flat_candidates = wrapped
worker.mode_discover_candidates(worker.args_from_command(cmd))

# ---- renders: hide the subject; top-down orthographic over the cells, and the scene camera view
scn = bpy.context.scene
for o in scn.objects:
    if o.name in REC.get("ignore", []):
        o.hide_render = True
scn.render.engine = "CYCLES"
scn.cycles.device = "CPU"
scn.cycles.samples = 48
scn.cycles.use_denoising = True
scn.render.image_settings.file_format = "PNG"

pts = [c for c in REC["cells"]]
xs, ys = [c["x"] for c in pts], [c["y"] for c in pts]
x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
side = max(x1 - x0, y1 - y0)
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
cam_data = bpy.data.cameras.new("topdown")
cam_data.type = "ORTHO"
cam_data.ortho_scale = side
cam = bpy.data.objects.new("topdown", cam_data)
scn.collection.objects.link(cam)
cam.location = (cx, cy, max(c["z"] for c in pts) + 30)
cam.rotation_euler = (0, 0, 0)
cam_data.clip_end = 500
orig_cam = scn.camera
scn.camera = cam
scn.render.resolution_x = scn.render.resolution_y = 900
scn.render.filepath = str(OUT / "topdown.png")
bpy.ops.render.render(write_still=True)
REC["topdown_extent"] = [cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2]

if orig_cam is not None:
    scn.camera = orig_cam
    scn.render.resolution_x, scn.render.resolution_y = 768, 576
    scn.render.filepath = str(OUT / "scene_cam.png")
    bpy.ops.render.render(write_still=True)
    REC["scene_cam_matrix"] = [list(r) for r in orig_cam.matrix_world]
    REC["scene_cam_lens"] = orig_cam.data.lens
    REC["scene_cam_sensor"] = orig_cam.data.sensor_width

json.dump(REC, open(OUT / "cells.json", "w"), default=lambda v: list(v) if hasattr(v, "__iter__") else str(v))
print("DUMPED", len(REC["cells"]), "cells to", OUT)
