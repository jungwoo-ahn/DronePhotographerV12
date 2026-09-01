"""Does an authored goal actually LOOK like what was asked?

Nothing here involves the policy. A goal names a shot; `pose_solver` inverts the stage-3 scorer
to the camera pose that realises it in a given placement, and Blender renders that pose. So the
picture answers a question the policy cannot be blamed for: is the goal itself the shot the user
asked for, and does the same goal mean the same composition in a scene it was not authored in?

Two experiments, one driver:

  A. NATURAL LANGUAGE.  request -> author_goal -> solve -> render, the same request in several
     different scenes and subjects. The comparison is against the WORDS: re-read the render with
     Module 2 and check the categories come back.

  B. REFERENCE IMAGE.   a photo -> Module 2 -> goal -> solve -> render in a DIFFERENT scene with
     a DIFFERENT subject. The comparison is against the reference: same shot size, same view of
     the subject, same placement in frame, with everything else changed.

Both run with both completers (`bank` and `rules`), because the whole question of what to do
with the keys the user did not state is exactly what a picture can settle.

Stages (each writes its own artefacts, so a preempted run resumes):
    geom    Blender, once per placement: dump the subject's mesh vertices + frame bounds
    solve   CPU: author the goals and invert the scorer for the pose
    render  Blender: render every solved pose (grouped by placement, persistent server)
    measure CPU: re-read every render with Module 2

    python scripts/render_authored_goals.py --stage all --targets 6 --out runs/goal_render

venv: .venv-analysis (Blender is a subprocess). See scripts/sbatch_render_goals.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import hashlib
import subprocess
import sys
import time
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np  # noqa: E402

from src.common.annotations import (_apply_crop_extent, _apply_visible_geometry,  # noqa: E402
                                    is_goal_frame)
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, DEFAULT_TRAJ_ROOT  # noqa: E402
from src.common.facing import front_azimuth, sector3, sector8  # noqa: E402
from src.common.goal_space import (DEFAULT_GOAL_KEYS, RENDER_HEIGHT,  # noqa: E402
                                   RENDER_WIDTH, SUBJECT_BEARING_KEY)
from src.common.run_info import write_run_info  # noqa: E402
from src.goal_authoring.goal_bank import band_of  # noqa: E402
from src.goal_authoring.serialize import crop_phrase  # noqa: E402

# Requests spanning the vocabulary, including three the trained data barely supports
# (close-up, extreme wide, low angle) — those are the ones where a render says most.
NL_REQUESTS = [
    "a wide shot of the subject from the front",
    "medium shot at eye level, facing the camera, centered",
    "a medium-wide shot from the subject's back-left",
    "shoot them from above, three-quarter front-right, full body",
    "a close-up from the front",
    "wide establishing shot from behind, subject small in the lower left",
    "profile shot from the left, medium framing",
    "low angle full body from the front",
    "put them on the right third, wide, from the back",
    "an intimate portrait, tight on the face",
    "high angle medium shot from the side, subject in the upper part of the frame",
    "full body from the front-left, subject on the left third",
    "waist-up from the right, eye level",
    "extreme wide landscape shot, tiny subject, centered",
]

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="all", choices=("geom", "solve", "render", "measure", "all"))
ap.add_argument("--out", default="runs/goal_render")
ap.add_argument("--root", default=DEFAULT_TRAJ_ROOT)
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--targets", type=int, default=6, help="target placements (scene+subject)")
ap.add_argument("--visible-checks", type=int, default=5,
                help="goal frames sampled per candidate target to test that its subject is "
                     "really there. All of them must pass: one lucky detection is how a "
                     "snowman and a subject behind a wall both got in.")
ap.add_argument("--min-visible-iou", type=float, default=0.6,
                help="a frame counts as showing its subject only if a detector finds it where "
                     "the projected mesh says it is. The scorer cannot see occlusion, so "
                     "without this a 'goal frame' can be a picture of the wall in front of "
                     "the subject")
ap.add_argument("--min-goal-frames", type=int, default=50,
                help="a target must offer a real range of shots of its own, or an authored goal "
                     "is being asked of a placement whose camera anchors barely see the subject")
ap.add_argument("--nl-placements", type=int, default=2, help="scenes each NL request is rendered in")
ap.add_argument("--refs", type=int, default=8, help="reference images for the transfer experiment")
ap.add_argument("--ref-targets", type=int, default=2, help="target placements per reference")
ap.add_argument("--methods", default="bank,rules")
ap.add_argument("--samples", type=int, default=64,
                help="Cycles samples. v7 rendered the training frames at 32, which is visibly "
                     "noisy in the dark interiors; these frames are meant to be looked at.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--blender", default=f"{SHARED}/blender/blender")
ap.add_argument("--exclude", default="",
                help="comma-separated placement names to drop from the candidate list. One of the "
                     "held-out scenes segfaults Cycles on this worker (rc=-11) through both the "
                     "persistent server and one-process-per-frame, while loading fine for the "
                     "geometry dump — excluding it lets its goals be shot elsewhere instead of "
                     "coming back as holes in the gallery.")
ap.add_argument("--max-renders", type=int, default=0,
                help="stop after this many frames (0 = all). For the login-node smoke that "
                     "checks the pipeline before a full sweep goes to the queue.")
args = ap.parse_args()

OUT = Path(args.out).resolve()      # Blender subprocesses run with cwd=SHARED
GEOM, RUNINFO, FRAMES = OUT / "_geom", OUT / "_run_info", OUT / "frames"
for d in (GEOM, RUNINFO, FRAMES):
    d.mkdir(parents=True, exist_ok=True)
PLAN, SOLVED, MEASURED = OUT / "plan.json", OUT / "solved.json", OUT / "measured.json"
METHODS = [m.strip() for m in args.methods.split(",") if m.strip()]
RENDER_OPTS = {"render_device": "GPU", "samples": args.samples, "blender_threads": 8,
               "persistent_data": True}


def val_scene_set() -> frozenset[str]:
    return frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix, iy = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


# Torso joints. A person has shoulders and hips; the vocabulary these goals are written in
# assumes one ("bearing: front = we see their face"), and the held-out scenes contain assets
# that are not people at all — a snowman, which YOLO reports as a person from its silhouette
# and which has no front to be seen from.
_TORSO_JOINTS = (5, 6, 11, 12)


def subject_really_visible(record: dict, est, image_path: str, min_iou: float = 0.45,
                           require_body: bool = True) -> float:
    """How much of the subject the frame claims is in view is actually THERE, 0..1.

    The stage-3 profile comes from projecting the subject's mesh, which knows nothing about
    what stands in front of it. A goal frame can therefore be a perfectly ordinary "wide shot,
    subject at right, cropped above the head" whose picture is a wall, or a reed bed with a
    coat sleeve behind it — and every downstream measurement of that frame is of the reeds.
    Comparing the CLIPPED projected box with what a detector finds is the cheapest test that
    sees occlusion at all; below `min_iou` the frame is not a composition anyone could copy.
    """
    det = est.detect_main_subject(image_path)
    if det is None:
        return 0.0
    bbox, kp, _W, _H = det
    if require_body:
        if kp is None or sum(float(kp[i][2]) >= 0.5 for i in _TORSO_JOINTS) < 3:
            return 0.0
    x0, y0, x1, y1 = (float(v) for v in record["bbox_xyxy_full"])
    proj = (max(0.0, x0), max(0.0, y0), min(RENDER_WIDTH, x1), min(RENDER_HEIGHT, y1))
    return _iou(proj, [float(v) for v in bbox])


def gt_profile(record: dict) -> dict:
    """The exporter's own view of a render record: visible geometry + crop fractions."""
    raw = dict(record["scores"])
    _apply_visible_geometry(raw, int(RENDER_WIDTH), int(RENDER_HEIGHT))
    _apply_crop_extent(raw, record["bbox_xyxy_full"], float(RENDER_HEIGHT))
    return raw


# --------------------------------------------------------------------------- plan
def pick_placements() -> list[dict]:
    """Target placements from the HELD-OUT scenes, one subject per scene, spread over scenes.

    Held out because a goal is supposed to be scene-agnostic; asking it to be realised in a room
    the pipeline was not tuned on is the honest version of the question.
    """
    from src.goal_authoring.from_reference import ReferenceEstimator
    est = ReferenceEstimator()
    val = val_scene_set()
    dirs = [d for d in sorted(os.listdir(args.root))
            if d.split("__")[0] in val and (Path(args.root) / d / "data.json").exists()]
    rng = random.Random(args.seed)
    rng.shuffle(dirs)
    picked, seen_scene, seen_obj = [], set(), set()
    for dn in dirs:
        scene = dn.split("__")[0]
        obj = dn.split("__", 1)[1] if "__" in dn else dn
        if scene in seen_scene or obj in seen_obj or obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        if front_azimuth(obj) is None:
            continue
        doc = json.loads((Path(args.root) / dn / "data.json").read_text())
        frames = [r for pair in doc.get("render_records", []) for r in pair if is_goal_frame(r)]
        if len(frames) < args.min_goal_frames:    # a placement with few usable shots is no target
            continue
        picked.append({"placement": dn, "scene": scene, "object": obj,
                       "data": str((Path(args.root) / dn / "data.json").resolve()),
                       "n_goal_frames": len(frames)})
        seen_scene.add(scene)
        seen_obj.add(obj)
        if len(picked) >= args.targets:
            break
    return picked


def pick_references(placements: list[dict]) -> list[dict]:
    """Reference frames from held-out placements, spread over (shot size, view sector).

    Deliberately drawn from placements that are NOT targets, so a transfer is never to the
    scene or the subject the composition came from.
    """
    val = val_scene_set()
    used = {p["placement"] for p in placements}
    used_obj = {p["object"] for p in placements}
    dirs = [d for d in sorted(os.listdir(args.root))
            if d.split("__")[0] in val and d not in used
            and (Path(args.root) / d / "data.json").exists()]
    from src.goal_authoring.from_reference import ReferenceEstimator
    est = ReferenceEstimator()
    rng = random.Random(args.seed + 1)
    rng.shuffle(dirs)
    out, cells = [], set()
    for dn in dirs:
        obj = dn.split("__", 1)[1] if "__" in dn else dn
        if obj in used_obj or obj in DEFAULT_EXCLUDE_OBJECTS or front_azimuth(obj) is None:
            continue
        doc = json.loads((Path(args.root) / dn / "data.json").read_text())
        cands = [r for pair in doc.get("render_records", []) for r in pair
                 if is_goal_frame(r) and r.get("bbox_xyxy_full")]
        rng.shuffle(cands)
        for r in cands:
            raw = gt_profile(r)
            bearing = (front_azimuth(obj) - float(raw["cam_to_obj_azimuth_deg"])) % 360.0
            cell = (band_of("occupancy", raw["occupancy"]), sector3(bearing))
            if cell in cells:
                continue
            iou = subject_really_visible(r, est, str(Path(args.root) / dn / r["path_rel"]),
                                         min_iou=args.min_visible_iou)
            if iou < args.min_visible_iou:
                continue        # occluded: the picture is not the composition the profile claims
            out.append({
                "visible_iou": float(iou),
                "image": str(Path(args.root) / dn / r["path_rel"]),
                "placement": dn, "object": obj, "cell": list(cell),
                "gt": {k: float(v) for k, v in raw.items()},
                "gt_bearing": bearing,
                "gt_crop": crop_phrase(raw["top_cut_frac"], raw["bot_cut_frac"]),
            })
            cells.add(cell)
            used_obj.add(obj)
            break
        if len(out) >= args.refs:
            break
    return out


def stage_geom(placements: list[dict]) -> None:
    for i, p in enumerate(placements):
        npz = GEOM / f"{p['placement']}.npz"
        ri = write_run_info(p["placement"], p["data"], RUNINFO, shared_root=SHARED,
                            resolution=(int(RENDER_WIDTH), int(RENDER_HEIGHT)),
                            extra_options=RENDER_OPTS)
        p["run_info"] = ri
        p["geom"] = str(npz)
        if npz.exists():
            print(f"[geom] {i+1}/{len(placements)} cached {p['placement'][:50]}", flush=True)
            continue
        t0 = time.time()
        cmd = [args.blender, "-b", "-P", f"{V12}/scripts/dump_placement_geometry.py", "--",
               "--run-info", ri, "--data", p["data"], "--out", str(npz)]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{SHARED}/blender/syslibs/lib:" + env.get("LD_LIBRARY_PATH", "")
        r = subprocess.run(cmd, cwd=SHARED, capture_output=True, text=True, timeout=900, env=env)
        if not npz.exists():
            raise RuntimeError(f"geometry dump failed for {p['placement']}:\n{r.stderr[-1500:]}")
        print(f"[geom] {i+1}/{len(placements)} {p['placement'][:50]} ({time.time()-t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- solve
def stage_solve(plan: dict) -> dict:
    from src.goal_authoring.author import author_goal
    from src.goal_authoring.completion import make_completer
    from src.goal_authoring.from_language import keyword_classifier
    from src.goal_authoring.pose_solver import PlacementGeometry, nearest_recorded_distance

    geoms = {p["placement"]: PlacementGeometry.load(p["geom"]) for p in plan["placements"]}
    datas = {p["placement"]: json.loads(Path(p["data"]).read_text()) for p in plan["placements"]}
    completers = {m: make_completer(m) for m in METHODS}
    jobs: list[dict] = []

    def solve_into(goal_vals: dict, placement: str, tag: str, extra: dict) -> dict | None:
        g = geoms[placement]
        try:
            s = g.solve(goal_vals)
        except Exception as exc:                       # no facing entry, unreachable, ...
            return {"tag": tag, "placement": placement, "error": f"{type(exc).__name__}: {exc}", **extra}
        if s["profile"] is None:
            return {"tag": tag, "placement": placement, "error": "subject behind the camera", **extra}
        ach = s["profile"]
        # The pose goes in the FILENAME. Frames are skipped when they already exist so a
        # preempted sweep resumes, but the placement a goal is shot in is chosen, not fixed —
        # so the same tag can legitimately mean a different pose on a later run, and a plain
        # tag-named file would be silently reused for a shot it is not a picture of.
        digest = hashlib.sha1(
            np.round(np.concatenate([s["position"], s["forward"], s["up"]]), 4).tobytes()
        ).hexdigest()[:8]
        rec = {
            "tag": tag, "pose_id": digest,
            "placement": placement, "object": placement.split("__", 1)[1],
            "requested": {k: float(goal_vals[k]) for k in goal_vals if not k.startswith("_")},
            "achieved": {k: float(v) for k, v in ach.items() if k != "_bbox_full"},
            "pose": {"position": s["position"].tolist(), "forward": s["forward"].tolist(),
                     "up": s["up"].tolist()},
            "r": s["r"], "az": s["az"], "el": s["el"],
            "geom_distance": s["geom_distance"], "residual": s["residual"],
            "pose_underdetermined": s["pose_underdetermined"],
            "nearest_recorded_camera_m": nearest_recorded_distance(s["position"], datas[placement]),
            "frame": str(FRAMES / f"{tag}_{digest}.jpg"),
            **extra,
        }
        return rec

    def choose(goal_vals: dict, candidates: list[str], n: int, tag_fmt, extra: dict) -> list[dict]:
        """Solve the goal in every candidate placement and keep the `n` where the camera it
        asks for is closest to one the v7 sampler actually used.

        This is choosing a LOCATION for a fixed shot, which is what a photographer does — the
        goal is not altered. It matters because a goal is scene-agnostic while a scene is not
        goal-agnostic: the first sweep put a camera in a basement on a sight line that runs
        through a wall, and the render is a photograph of brickwork. The sampler's recorded
        cameras all passed its `is_camera_valid` ray-casts, so proximity to one is the
        cheapest evidence available here that the subject can be seen from there.
        """
        tried = []
        for pl in candidates:
            rec = solve_into(goal_vals, pl, "tmp", extra)
            if rec and "error" not in rec:
                tried.append(rec)
        tried.sort(key=lambda r: r["nearest_recorded_camera_m"])
        out = []
        for k, rec in enumerate(tried[:n]):
            rec["tag"] = tag_fmt(k)
            rec["frame"] = str(FRAMES / f"{rec['tag']}_{rec['pose_id']}.jpg")
            rec["placement_rank"] = k
            rec["placements_considered"] = len(tried)
            out.append(rec)
        return out

    # ---- A: natural language ----
    dropped = {x.strip() for x in args.exclude.split(",") if x.strip()}
    tp = [p["placement"] for p in plan["placements"]
          if not any(d in p["placement"] for d in dropped)]
    if dropped:
        print(f"[solve] excluded {len(plan['placements']) - len(tp)} placement(s); "
              f"{len(tp)} candidates remain", flush=True)
    for qi, text in enumerate(NL_REQUESTS):
        for m in METHODS:
            g = author_goal(text, classifier=keyword_classifier, completer=completers[m])
            jobs += choose(dict(g.profile.values), tp, args.nl_placements,
                           lambda k, qi=qi, m=m: f"nl{qi:02d}_{m}_{k}",
                           {"kind": "nl", "request": text, "method": m,
                            "prompt": g.prompt, "categories": g.categories,
                            "parsed": g.request.categories(),
                            "warnings": g.warnings, "support": g.support})
        print(f"[solve] nl {qi+1}/{len(NL_REQUESTS)}", flush=True)

    # ---- B: reference image -> another scene/subject ----
    if plan["references"]:
        from src.goal_authoring.from_reference import ReferenceEstimator
        est = ReferenceEstimator()
        for ri, ref in enumerate(plan["references"]):
            gp = est(ref["image"])
            ref["detected"] = bool(gp.specified)
            if not gp.specified:
                print(f"[solve] ref {ri}: no subject detected", flush=True)
                continue
            ref["measured"] = {k: float(v) for k, v in gp.values.items()}
            ref["measured_categories"] = gp.categories()
            for m in METHODS:
                from src.goal_authoring.author import complete_request
                done = complete_request(gp, completers[m])
                targets = [p["placement"] for p in plan["placements"]
                           if p["placement"] in tp and p["placement"] != ref["placement"]
                           and p["object"] != ref["object"]
                           and p["scene"] != ref["placement"].split("__")[0]]
                jobs += choose(dict(done.profile.values), targets, args.ref_targets,
                               lambda k, ri=ri, m=m: f"ref{ri:02d}_{m}_{k}",
                               {"kind": "ref", "method": m, "ref_index": ri,
                                "ref_image": ref["image"], "ref_object": ref["object"],
                                "prompt": done.prompt,
                                "categories": done.profile.categories(),
                                "warnings": done.warnings, "support": done.support})
            print(f"[solve] ref {ri+1}/{len(plan['references'])}", flush=True)

    ok = [j for j in jobs if "error" not in j]
    print(f"[solve] {len(ok)} poses solved, {len(jobs)-len(ok)} failed")
    doc = {"args": vars(args), "placements": plan["placements"], "references": plan["references"],
           "jobs": jobs}
    SOLVED.write_text(json.dumps(doc, indent=1))
    return doc


# --------------------------------------------------------------------------- render
def stage_render(doc: dict) -> None:
    from src.common.blender_env import PersistentBlenderRenderer, SubprocessBlenderRenderer

    os.environ["LD_LIBRARY_PATH"] = (f"{SHARED}/blender/syslibs/lib:"
                                     + os.environ.get("LD_LIBRARY_PATH", ""))
    run_info = {p["placement"]: p["run_info"] for p in doc["placements"]}
    todo = [j for j in doc["jobs"] if "error" not in j and not Path(j["frame"]).exists()]
    todo.sort(key=lambda j: j["placement"])          # one scene load per placement, not per frame
    if args.max_renders:
        todo = todo[: args.max_renders]
    print(f"[render] {len(todo)} frames over {len({j['placement'] for j in todo})} placements",
          flush=True)
    # One bad scene must not take the sweep down with it. The persistent server segfaulted on
    # a scene load after 64 of 85 frames ("render server died (rc=-11): could not get a list of
    # mounted file-systems"), losing two whole placements that had nothing wrong with them; on
    # the retry it died on the same scene twice more, so that scene is reproducibly fatal to the
    # SERVER — not to Blender, since the geometry dump loads the same file through the same
    # `BlenderDrone` without trouble. So a scene that kills the server twice falls back to the
    # one-process-per-frame renderer (slower, and the path the evals have always used) rather
    # than being dropped.
    fast = PersistentBlenderRenderer(repo_root=SHARED)
    slow: SubprocessBlenderRenderer | None = None
    fallback: set[str] = set()
    t0, done = time.time(), 0
    try:
        for i, j in enumerate(todo):
            pl = j["placement"]
            for attempt in (1, 2, 3):
                use_slow = pl in fallback or attempt == 3
                try:
                    if use_slow:
                        if slow is None:
                            slow = SubprocessBlenderRenderer(repo_root=SHARED, timeout_s=900.0)
                        img = slow.render(run_info[pl], np.array(j["pose"]["position"]),
                                          np.array(j["pose"]["forward"]), np.array(j["pose"]["up"]))
                    else:
                        img = fast.render(run_info[pl], np.array(j["pose"]["position"]),
                                          np.array(j["pose"]["forward"]), np.array(j["pose"]["up"]))
                    img.save(j["frame"], quality=92)
                    done += 1
                    break
                except Exception as exc:                      # noqa: BLE001
                    print(f"[render] {j['tag']} attempt {attempt} "
                          f"({'subprocess' if use_slow else 'server'}) failed: "
                          f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)
                    if use_slow:
                        if attempt == 3:
                            j["error"] = f"render failed: {type(exc).__name__}"
                        continue
                    try:
                        fast.close()
                    except Exception:                          # noqa: BLE001
                        pass
                    fast = PersistentBlenderRenderer(repo_root=SHARED)
                    if attempt == 2 and pl not in fallback:
                        fallback.add(pl)
                        print(f"[render] {pl[:50]}: server died twice — falling back to "
                              "one process per frame for this scene", flush=True)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[render] {i+1}/{len(todo)} ({done} rendered, {time.time()-t0:.0f}s)",
                      flush=True)
                SOLVED.write_text(json.dumps(doc, indent=1))
    finally:
        fast.close()
        if slow is not None:
            slow.close()
    SOLVED.write_text(json.dumps(doc, indent=1))
    print(f"[render] done ({time.time()-t0:.0f}s)")


# --------------------------------------------------------------------------- measure
def stage_measure(doc: dict) -> None:
    """Re-read every render the way a user's own photo would be read.

    Two yardsticks, and they answer different questions. The POSE profile is what the shot
    geometrically is (the scorer on the solved pose, exact). The MODULE 2 read is what a
    detector recovers from the pixels, which is the only thing available for a real photo and
    carries the detector's own error — so a disagreement here is not necessarily a bad render.
    """
    from src.goal_authoring.from_reference import ReferenceEstimator
    est = ReferenceEstimator()
    n = 0
    for j in doc["jobs"]:
        if "error" in j or not Path(j["frame"]).exists():
            continue
        gp = est(j["frame"])
        j["read_detected"] = bool(gp.specified)
        if gp.specified:
            j["read"] = {k: float(v) for k, v in gp.values.items()}
            j["read_categories"] = gp.categories()
        n += 1
        if n % 20 == 0:
            print(f"[measure] {n}", flush=True)
    MEASURED.write_text(json.dumps(doc, indent=1))
    print(f"[measure] {n} renders read -> {MEASURED}")


def main() -> int:
    if PLAN.exists() and args.stage != "geom":
        plan = json.loads(PLAN.read_text())
    else:
        placements = pick_placements()
        plan = {"placements": placements, "references": pick_references(placements)}
        print(f"[plan] {len(placements)} target placements, {len(plan['references'])} references")
        for p in placements:
            print(f"   target  {p['placement'][:64]}  ({p['n_goal_frames']} goal frames)")
        for r in plan["references"]:
            print(f"   ref     {Path(r['image']).parent.parent.name[:44]}  {r['cell']}")
    if args.stage in ("geom", "all"):
        stage_geom(plan["placements"])
        PLAN.write_text(json.dumps(plan, indent=1))
    if args.stage in ("solve", "all"):
        doc = stage_solve(plan)
    else:
        doc = json.loads(SOLVED.read_text()) if SOLVED.exists() else None
    if args.stage in ("render", "all") and doc:
        stage_render(doc)
    if args.stage in ("measure", "all") and doc:
        stage_measure(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
