"""Stage 1 of the true recon: extract goals from reference images with Module 2.

Kept separate from the rollout so Module 2's stack (ultralytics/sklearn/its own torch) never shares
an interpreter with the cosmos-framework environment. Picks (reference image, target placement) pairs
where the target is a DIFFERENT scene AND a different subject — re-shooting the same composition
somewhere else is the whole point. venv: .venv-analysis."""
import argparse, json, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.annotations import (DEFAULT_GOAL_OCCUPANCY_RANGE,
                                    DEFAULT_MIN_GOAL_VISIBLE_FRAC,
                                    _apply_crop_extent, _is_well_framed)
from src.common.facing import front_azimuth
from src.common.goal_space import (DEFAULT_GOAL_KEYS, RENDER_HEIGHT, RENDER_WIDTH,
                                   SUBJECT_BEARING_KEY)
from src.goal_authoring.from_reference import ReferenceEstimator


def _framed_raw(record: dict) -> dict:
    """scores + the crop keys the gate reads (see plan_rerender_yaw for the same helper)."""
    raw = dict(record.get("scores") or {})
    bbox = record.get("bbox_xyxy_full")
    if bbox:
        _apply_crop_extent(raw, bbox, float(RENDER_HEIGHT))
    return raw


ap = argparse.ArgumentParser()
ap.add_argument("--cases", type=int, default=6)
ap.add_argument("--root", default="data/trajectories/v7_stage2_renders_lookat075")
ap.add_argument("--out", default="runs/recon_ref/goals.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start-occ", type=float, nargs=2, default=(18.0, 40.0),
                help="start-pose occupancy window: big enough for the detector to read the subject "
                     "(~10%% was unreadable), small enough to leave the shot to be earned")
ap.add_argument("--goal-occ", type=float, nargs=2, default=DEFAULT_GOAL_OCCUPANCY_RANGE,
                help="training goal band; imported, not retyped — the literal (20, 80) here "
                     "outlived the constant twice")
ap.add_argument("--min-visible", type=float, default=DEFAULT_MIN_GOAL_VISIBLE_FRAC,
                help="training composition gate: fraction of the subject's VERTICAL extent in "
                     "frame. Replaces --min-body 70, which named a constant that no longer "
                     "exists and picked head-cropped references (72.7%% of accepted goals) "
                     "while rejecting every chest-up one")
ap.add_argument("--require-face", type=int, default=1,
                help="reference must show the face — what makes it a shot worth asking for")
ap.add_argument("--min-occ-gap", type=float, default=12.0,
                help="reject starts already at the requested size — nothing to measure")
args = ap.parse_args()

est = ReferenceEstimator()
root = args.root
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
random.seed(args.seed); random.shuffle(dirs)

usable = []
for dn in dirs:
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if front_azimuth(obj) is None:
        continue
    p = os.path.join(root, dn, "data.json")
    if not os.path.exists(p):
        continue
    try:
        doc = json.load(open(p))
    except Exception:
        continue
    frames = [r for pair in doc.get("render_records", []) for r in pair
              if (s := r.get("scores")) and r.get("in_frame")
              and args.goal_occ[0] <= s["occupancy"] <= args.goal_occ[1]
              and _is_well_framed(_framed_raw(r), args.min_visible)]
    if not frames or not doc.get("accepted_pairs"):
        continue
    usable.append((dn, obj, frames, doc))
    if len(usable) >= 12 * args.cases:
        break
print(f"usable placements: {len(usable)}", flush=True)

cases = []
used_ref = set()
for i, (rdn, robj, rframes, _rdoc) in enumerate(usable):
    if len(cases) >= args.cases or robj in used_ref:
        continue
    rscene = rdn.split("__", 1)[0]
    # spread the goals over the shot vocabulary instead of taking whatever comes first:
    # at most 2 cases per (shot size, front/side/back) cell
    from src.common.facing import sector3 as _s3
    used_tgt = {c["tgt_object"] for c in cases}
    tgt = next(((dn, obj, doc) for (dn, obj, _f, doc) in usable
                if obj != robj and dn.split("__", 1)[0] != rscene and obj not in used_tgt), None)
    if tgt is None:
        continue
    ref_rec = random.choice(rframes)
    ref_img = os.path.join(root, rdn, ref_rec["path_rel"])
    if args.require_face:
        det = est.detect_main_subject(ref_img)
        if det is None:
            continue
        kp = det[1]
        # nose or both eyes clearly visible => we are looking at the person, not their back
        if kp is None or not (float(kp[0, 2]) > 0.5 or (float(kp[1, 2]) > 0.5 and float(kp[2, 2]) > 0.5)):
            continue
    gp = est(ref_img)
    if "occupancy" not in gp.specified or SUBJECT_BEARING_KEY not in gp.specified:
        print(f"  skip {robj[:24]}: no usable goal", flush=True)
        continue
    cats = gp.categories()
    cell = (cats.get("shot_size"), _s3(gp.values[SUBJECT_BEARING_KEY]))
    if sum(1 for c in cases if c.get("cell") == list(cell)) >= 2:
        continue
    gvec = [float(gp.values.get(k, 0.0)) for k in DEFAULT_GOAL_KEYS]
    tdoc = tgt[2]
    start = None
    for pi, pair in enumerate(tdoc.get("render_records", [])):
        for fi, r in enumerate(pair):
            sc = r.get("scores")
            if not sc or not r.get("in_frame"):
                continue
            if not (args.start_occ[0] <= sc["occupancy"] <= args.start_occ[1]
                    and _is_well_framed(_framed_raw(r), 0.40)):
                continue
            if abs(sc["occupancy"] - gp.values.get("occupancy", 0.0)) >= args.min_occ_gap:
                fr = tdoc["accepted_pairs"][pi]["trajectory_32f"][fi]
                start = {"pair": pi, "frame": fi, "pos": fr["pos"],
                         "forward": fr["forward"], "up": fr["up"], "scores": sc}
                break
        if start:
            break
    if start is None:
        print(f"  skip target {tgt[1][:24]}: no visible start", flush=True)
        continue

    cases.append({
        "cell": list(cell),
        "start_pose": start,
        "ref_image": ref_img, "ref_object": robj, "ref_dir": os.path.join(root, rdn),
        "ref_scores": ref_rec.get("scores"),
        "tgt_dir": os.path.join(root, tgt[0]), "tgt_object": tgt[1],
        "goal_vec": gvec,
        # WHICH keys are real. Without this the consumer cannot tell a requested 0 from an
        # unspecified one, and `goal_prompt` would assert e.g. "centre 0/0 px" — the top-left
        # corner — for an axis the user never constrained.
        "goal_specified": sorted(gp.specified),
        "goal_crop": {k: float(gp.values[k]) for k in
                      ("top_cut_frac", "bot_cut_frac", "visible_frac") if k in gp.specified},
        "goal": {"occupancy": round(float(gp.values.get("occupancy", 0)), 1),
                 "bearing": round(float(gp.values.get(SUBJECT_BEARING_KEY, 0)), 1),
                 "elevation": round(float(gp.values.get("cam_to_obj_elevation_deg", 0)), 1),
                 "categories": gp.categories()},
    })
    used_ref.add(robj)
    print(f"  [{len(cases)-1}] {robj[:24]} -> {tgt[1][:24]} | {gp.categories()}", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"cases": cases, "goal_keys": list(DEFAULT_GOAL_KEYS)}, open(args.out, "w"), indent=1)
print(f"wrote {args.out} ({len(cases)} cases)")
