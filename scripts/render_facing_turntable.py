"""Render each distinct asset ISOLATED on a clean turntable, to recover per-asset subject
FACING on a de-noised substrate (replaces YuNet-over-noisy-scene-frames in facing_auto.py).

WHY this is correct (verified 2026-07-30):
  * The v7 object orientation is reproduced EXACTLY for our (upright) assets: import the same
    .blend -> parent -> move bottom-center to origin -> rotation_z=0. `auto_fix_orientation`
    only rotates aspect>2 (lying-flat) imports; for standing characters (aspect<1) it is a
    no-op, so both repo loaders (render_object.place_imported_object and src.blender.objects)
    give the identical final orientation. We reuse the shared, proven loader
    (src.blender.objects, same as scripts/render_object_canonical_views.py) so textures /
    orientation match the data pipeline.
  * `cam_to_obj_azimuth_deg = atan2(dy, dx)` (dy/dx = subject_center - cam_pos, world frame,
    +Z up) is the EXACT stored convention (src/scoring/projection.py:cam_to_subject_angles).
    Azimuth is translation-invariant, so placing the object at the origin (dropping the scene
    translation) preserves the data-frame facing. The camera orbits at K azimuths EVENLY
    SPACED IN THIS CONVENTION, so a front azimuth picked on these renders drops straight into
    the facing map with ZERO frame conversion.

Lighting: uniform world fill + a camera-tracking key SUN, so whichever side faces the camera
(the side we judge) is lit at every azimuth.

Runs INSIDE the shared Blender (needs its syslibs on LD_LIBRARY_PATH) -- launch via
scripts/run_facing_turntable.sh (login smoke) or scripts/sbatch_facing_turntable.sh (full).

Writes runs/facing_turntable/<obj>/az_<deg>.png + runs/facing_turntable/index.json
(obj -> {object_file, center, bounding_radius, cam_distance, views:[{az, cam_pos, path}]}).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
os.chdir(V12)
# Rendering is a shared-repo concern (v12 has no renderer); reuse its proven Blender loader.
sys.path.insert(0, SHARED)
from src.blender.objects import (  # noqa: E402
    auto_fix_orientation,
    fix_missing_textures,
    import_object,
    parent_and_center,
)
from src.blender.bbox import get_world_bbox  # noqa: E402
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=DEFAULT_TRAJ_ROOT)
ap.add_argument("--out", default="runs/facing_turntable")
ap.add_argument("--views", type=int, default=12, help="azimuth samples (every 360/K deg, data convention)")
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--cam-elev-deg", type=float, default=10.0, help="camera elevation above subject center")
ap.add_argument("--margin", type=float, default=1.25, help="framing headroom on the bounding sphere")
ap.add_argument("--samples", type=int, default=16)
ap.add_argument("--engine", default="CYCLES", help="CYCLES (proven headless here) or EEVEE (faster)")
ap.add_argument("--gpu", action="store_true", help="enable Cycles GPU (OPTIX/CUDA)")
ap.add_argument("--max-assets", type=int, default=0, help="0 = all")
ap.add_argument("--only", default="", help="comma-separated obj-name substrings (smoke)")
ap.add_argument("--skip-existing", action="store_true", help="skip objects that already have all views (resume)")
args = ap.parse_args(argv)


def az_of(cam_pos: Vector, center: Vector) -> float:
    """Stored cam_to_obj_azimuth_deg convention: atan2(dy, dx), dy/dx = center - cam."""
    d = center - cam_pos
    return math.degrees(math.atan2(d.y, d.x)) % 360.0


def look_at(cam_pos: Vector, target: Vector, world_up=Vector((0.0, 0.0, 1.0))) -> Matrix:
    """4x4 world matrix for a Blender camera at cam_pos looking at target (verbatim convention
    of render_object.look_at_matrix: local +X right, +Y up, -Z forward)."""
    fwd = (target - cam_pos).normalized()
    right = fwd.cross(world_up)
    if right.length < 1e-6:
        right = fwd.cross(Vector((0.0, 1.0, 0.0)))
    right.normalize()
    up = right.cross(fwd).normalized()
    rot = Matrix((
        (right.x, up.x, -fwd.x),
        (right.y, up.y, -fwd.y),
        (right.z, up.z, -fwd.z),
    )).to_4x4()
    return Matrix.Translation(cam_pos) @ rot


def enable_cycles_gpu() -> bool:
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        bpy.ops.preferences.addon_enable(module="cycles")
        prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception as exc:  # noqa: BLE001
            print(f"[tt] gpu backend {backend} unavailable: {exc}")
            continue
        gpus = [d for d in prefs.devices if d.type == backend]
        if gpus:
            for d in prefs.devices:
                d.use = (d.type == backend)
            bpy.context.scene.cycles.device = "GPU"
            print(f"[tt] Cycles GPU: {backend} x{len(gpus)}")
            return True
    print("[tt] no Cycles GPU found -> CPU")
    return False


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Uniform world fill (even illumination at every azimuth).
    if scene.world is None:
        scene.world = bpy.data.worlds.new("FacingWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.50, 0.52, 0.55, 1.0)
        bg.inputs[1].default_value = 0.7

    # Camera-tracking key sun (re-aimed per view).
    sun_data = bpy.data.lights.new("KeySun", "SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("KeySun", sun_data)
    scene.collection.objects.link(sun)

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = 10000.0
    cam = bpy.data.objects.new("Cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    r = scene.render
    r.resolution_x = r.resolution_y = args.res
    r.resolution_percentage = 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"

    engine = "BLENDER_EEVEE_NEXT" if args.engine.upper() == "EEVEE" else "CYCLES"
    r.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.max_bounces = 4
        try:
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = "OPENIMAGEDENOISE"  # works on CPU and GPU
        except Exception:  # noqa: BLE001
            pass
        if args.gpu:
            enable_cycles_gpu()
    else:
        try:
            scene.eevee.taa_render_samples = args.samples
        except Exception:  # noqa: BLE001
            pass
    print(f"[tt] engine={engine} res={args.res} samples={args.samples} views={args.views}")
    return scene, cam, sun


def frame_distance(bs_radius: float, cam) -> float:
    half_fov = math.atan((cam.data.sensor_width * 0.5) / cam.data.lens)
    return bs_radius / math.tan(half_fov) * args.margin


def clear_imports(keep_names: set):
    for obj in list(bpy.data.objects):
        if obj.name not in keep_names:
            bpy.data.objects.remove(obj, do_unlink=True)
    # Purge orphaned datablocks so 102 imports don't balloon memory.
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                 bpy.data.armatures, bpy.data.lights, bpy.data.cameras):
        for blk in list(coll):
            if blk.users == 0:
                try:
                    coll.remove(blk)
                except Exception:  # noqa: BLE001
                    pass


def build_obj_list():
    root = args.data_root
    by_obj = {}
    for d in sorted(os.listdir(root)):
        dp = os.path.join(root, d)
        jp = os.path.join(dp, "data.json")
        if not os.path.isdir(dp) or not os.path.exists(jp):
            continue
        obj = d.split("__", 1)[1] if "__" in d else d
        if obj in by_obj:
            continue
        try:
            of = json.load(open(jp)).get("object_file")
        except Exception:  # noqa: BLE001
            continue
        if of and os.path.exists(of):
            by_obj[obj] = of
    items = sorted(by_obj.items())
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        items = [it for it in items if any(s in it[0] for s in subs)]
    if args.max_assets > 0:
        items = items[:args.max_assets]
    return items


def safe_name(obj: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in obj)[:90]


def main():
    scene, cam, sun = setup_scene()
    keep = {cam.name, sun.name}
    items = build_obj_list()
    print(f"[tt] {len(items)} objects x {args.views} views = {len(items) * args.views} renders")

    index = {}
    for i, (obj, of) in enumerate(items):
        odir = os.path.join(args.out, safe_name(obj))
        if args.skip_existing and os.path.isdir(odir):
            pngs = sorted(f for f in os.listdir(odir) if f.endswith(".png"))
            if len(pngs) >= args.views:
                # Reconstruct the index entry from disk so a RESUMED run still indexes assets a
                # previous (possibly time-killed) job rendered but never wrote to index.json
                # (index is written only at the end). az parsed from the "az_<deg>.png" filename.
                index[obj] = {
                    "object_file": of,
                    "views": [{"az": float(os.path.splitext(f)[0].split("_")[-1]),
                               "path": os.path.relpath(os.path.join(odir, f), V12)}
                              for f in pngs],
                    "skipped": True,
                }
                print(f"[tt] {i + 1}/{len(items)} skip (exists) {obj[:44]}")
                continue
        os.makedirs(odir, exist_ok=True)
        try:
            imported = import_object(Path(of))
            if not imported:
                raise RuntimeError("import_object returned nothing")
            # Some blends ship their mesh with hide_render=True -> imports & bboxes fine but renders
            # EMPTY (blank gray). render_object.py force-shows via set_hidden; do the same here.
            for o in imported:
                o.hide_render = False
                o.hide_viewport = False
            root = parent_and_center(imported)
            auto_fix_orientation(root, imported)
            fix_missing_textures()
            bpy.context.view_layer.update()

            bbox = get_world_bbox(imported)
            if bbox is None:
                raise RuntimeError("no world bbox")
            mn, mx = bbox
            center = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, (mn.z + mx.z) / 2))
            bs = (mx - mn).length / 2.0  # bounding-sphere radius
            dist = frame_distance(bs, cam)
            el = math.radians(args.cam_elev_deg)
            horiz, vert = dist * math.cos(el), dist * math.sin(el)

            views = []
            for k in range(args.views):
                az = k * (360.0 / args.views)           # data-convention target azimuth
                psi = math.radians(az + 180.0)          # camera position angle (cam is opposite)
                cam_pos = Vector((
                    center.x + horiz * math.cos(psi),
                    center.y + horiz * math.sin(psi),
                    center.z + vert,
                ))
                cam.matrix_world = look_at(cam_pos, center)
                sun.rotation_euler = (center - cam_pos).normalized().to_track_quat("-Z", "Y").to_euler()
                got = az_of(cam_pos, center)
                if abs(((got - az + 180) % 360) - 180) > 0.6:
                    raise AssertionError(f"azimuth mismatch: want {az:.1f} got {got:.1f}")
                png = os.path.join(odir, f"az_{int(round(az)):03d}.png")
                scene.render.filepath = os.path.abspath(png)
                bpy.ops.render.render(write_still=True)
                views.append({"az": round(az, 1),
                              "cam_pos": [round(v, 3) for v in cam_pos],
                              "path": os.path.relpath(png, V12)})

            index[obj] = {
                "object_file": of,
                "center": [round(v, 3) for v in center],
                "bounding_radius": round(bs, 3),
                "cam_distance": round(dist, 3),
                "cam_elev_deg": args.cam_elev_deg,
                "views": views,
            }
            print(f"[tt] {i + 1}/{len(items)} {obj[:44]:44s} bs={bs:.2f} D={dist:.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tt] {i + 1}/{len(items)} FAIL {obj[:44]}: {exc}")
            index[obj] = {"object_file": of, "error": str(exc)}
        finally:
            clear_imports(keep)

    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "index.json")
    if args.skip_existing and os.path.exists(outp):
        merged = json.load(open(outp))
        merged.update(index)
        index = merged
    json.dump(index, open(outp, "w"), indent=1)
    ok = sum(1 for v in index.values() if "error" not in v)
    print(f"[tt] DONE {ok}/{len(index)} objects rendered -> {outp}")


if __name__ == "__main__":
    main()
