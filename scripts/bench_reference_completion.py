"""Module 2 on held-out renders: what the image pins, what the body-extent fit adds, and what
each completer guesses for the rest — all against the training-side ground truth.

Frames come from the 8 held-out scenes (never in the bank's train rows). GT is the exporter's
own view of the frame: `scores` + the crop fractions from the signed `bbox_xyxy_full`
(`annotations._apply_crop_extent`) and the visible-bbox geometry.

    python scripts/bench_reference_completion.py --n 300        # <=1000 renders: fine on login

Reports: detection rate; geometry (occupancy / placement words, centre MAE); bearing & elevation;
crop SIDE (box rule) vs crop WORD (5-class, with the extent fit) vs the completed word; visible /
body_in_frame MAE for extent-fit vs bank vs rules. venv: .venv-analysis (GPU if free).
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.annotations import _apply_crop_extent, _apply_visible_geometry, is_goal_frame
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, DEFAULT_TRAJ_ROOT
from src.common.facing import front_azimuth, sector3, sector8
from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH, SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.author import complete_request
from src.goal_authoring.completion import BankCompleter, RuleCompleter
from src.goal_authoring.from_reference import (ReferenceEstimator, crop_from_bbox,
                                               estimate_vertical_extent, profile_from_detection)
from src.goal_authoring.goal_bank import band_of
from src.goal_authoring.serialize import crop_phrase

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--per-placement", type=int, default=2)
ap.add_argument("--root", default=DEFAULT_TRAJ_ROOT)
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--out", default="runs/eval/reference_bench.json")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

val = set(json.load(open(args.val_scenes))["scenes"])
dirs = [d for d in sorted(os.listdir(args.root)) if d.split("__")[0] in val
        and os.path.isdir(os.path.join(args.root, d))]
random.seed(args.seed); random.shuffle(dirs)
print(f"held-out placements: {len(dirs)}", flush=True)

samples = []
for dn in dirs:
    if len(samples) >= args.n:
        break
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if obj in DEFAULT_EXCLUDE_OBJECTS or front_azimuth(obj) is None:
        continue
    try:
        doc = json.load(open(os.path.join(args.root, dn, "data.json")))
    except Exception:
        continue
    cand = [r for pair in doc.get("render_records", []) for r in pair
            if is_goal_frame(r) and r.get("bbox_xyxy_full")]
    random.shuffle(cand)
    for r in cand[: args.per_placement]:
        raw = dict(r["scores"])
        _apply_visible_geometry(raw, int(RENDER_WIDTH), int(RENDER_HEIGHT))
        _apply_crop_extent(raw, r["bbox_xyxy_full"], float(RENDER_HEIGHT))
        raw[SUBJECT_BEARING_KEY] = (front_azimuth(obj) - float(raw["cam_to_obj_azimuth_deg"])) % 360.0
        samples.append({"image": os.path.join(args.root, dn, r["path_rel"]), "gt": raw, "obj": obj})
print(f"samples: {len(samples)}", flush=True)

est = ReferenceEstimator()
bank_c, rule_c = BankCompleter(), RuleCompleter()

rows = []
t0 = time.time()
for i, s in enumerate(samples):
    gt = s["gt"]
    det = est.detect_main_subject(s["image"])
    rec = {"image": s["image"], "detected": det is not None, "gt_occ": float(gt["occupancy"]),
           "gt_bearing": float(gt[SUBJECT_BEARING_KEY]),
           "gt_crop": crop_phrase(gt["top_cut_frac"], gt["bot_cut_frac"]),
           "gt_visible": float(gt["visible_frac"]), "gt_bif": float(gt["body_in_frame_ratio"])}
    if det is not None:
        bbox, kp, W, H = det
        side = crop_from_bbox(bbox, H, kp=kp)
        ext = estimate_vertical_extent(kp, H) if kp is not None else None
        gp = profile_from_detection(bbox, kp, W, H, est.bearing, est.elevation)
        v = gp.values
        rec.update({
            "occ": v["occupancy"],
            "cx": v["object_center_x"], "cy": v["object_center_y"],
            "gt_cx": float(gt["object_center_x"]), "gt_cy": float(gt["object_center_y"]),
            "bearing": v.get(SUBJECT_BEARING_KEY),
            "elev": v.get("cam_to_obj_elevation_deg"), "gt_elev": float(gt["cam_to_obj_elevation_deg"]),
            "side_word": crop_phrase(side.get("top_cut_frac", 0.0), side.get("bot_cut_frac", 0.0)),
            "extent_ok": bool({"top_cut_frac", "bot_cut_frac", "visible_frac"} <= gp.exact),
            "extent_r2": (ext["r2"] if ext else None),
            "extent_visible": (v.get("visible_frac") if {"top_cut_frac", "bot_cut_frac", "visible_frac"} <= gp.exact else None),
            "extent_word": crop_phrase(v["top_cut_frac"], v["bot_cut_frac"]) if {"top_cut_frac", "bot_cut_frac", "visible_frac"} <= gp.exact else None,
        })
        for name, comp in (("bank", bank_c), ("rules", rule_c)):
            done = complete_request(gp, comp)
            cv = done.profile.values
            rec[f"{name}_word"] = crop_phrase(cv["top_cut_frac"], cv["bot_cut_frac"])
            rec[f"{name}_visible"] = float(cv["visible_frac"])
            rec[f"{name}_bif"] = float(cv["body_in_frame_ratio"])
            rec[f"{name}_prompt"] = done.prompt
    rows.append(rec)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)

det = [r for r in rows if r["detected"]]
n = len(det)
print(f"\n===== Module 2 on {len(rows)} held-out goal frames: detected {n} ({100*n/max(1,len(rows)):.0f}%) =====")
def pct(xs): return f"{100*np.mean(xs):.1f}%" if len(xs) else "n/a"
def mae(a, b): return f"{np.mean(np.abs(np.array(a)-np.array(b))):.3f}"
print("GEOMETRY  occupancy MAE", mae([r["occ"] for r in det], [r["gt_occ"] for r in det]),
      " shot-size word", pct([band_of("occupancy", r["occ"]) == band_of("occupancy", r["gt_occ"]) for r in det]),
      " centre MAE px", mae([r["cx"] for r in det], [r["gt_cx"] for r in det]), "/", mae([r["cy"] for r in det], [r["gt_cy"] for r in det]),
      " placement words", pct([band_of("object_center_x", r["cx"]) == band_of("object_center_x", r["gt_cx"]) for r in det]),
      "/", pct([band_of("object_center_y", r["cy"]) == band_of("object_center_y", r["gt_cy"]) for r in det]))
b = [r for r in det if r["bearing"] is not None]
cyc = lambda d: abs(((d + 180) % 360) - 180)
print(f"BEARING   given {len(b)}/{n}  sector3", pct([sector3(r["bearing"]) == sector3(r["gt_bearing"]) for r in b]),
      " sector8", pct([sector8(r["bearing"]) == sector8(r["gt_bearing"]) for r in b]),
      " MAE deg", f"{np.mean([cyc(r['bearing']-r['gt_bearing']) for r in b]):.1f}" if b else "n/a")
e = [r for r in det if r["elev"] is not None]
print(f"ELEVATION MAE deg", mae([r["elev"] for r in e], [r["gt_elev"] for r in e]),
      " word", pct([band_of("cam_to_obj_elevation_deg", r["elev"]) == band_of("cam_to_obj_elevation_deg", r["gt_elev"]) for r in e]))
side4 = lambda w: "none" if w == "uncropped" else ("top" if "above" in w else ("both" if "both" in w else "bot"))
print("CROP      side (4-class, box rule)", pct([side4(r["side_word"]) == side4(r["gt_crop"]) for r in det]))
ex = [r for r in det if r["extent_ok"]]
print(f"          extent fit available {len(ex)}/{n} ({100*len(ex)/max(1,n):.0f}%)  ->  5-class word", pct([r["extent_word"] == r["gt_crop"] for r in ex]),
      " visible MAE", mae([r["extent_visible"] for r in ex], [r["gt_visible"] for r in ex]) if ex else "n/a")
for name in ("bank", "rules"):
    print(f"          completed [{name}]  5-class word", pct([r[f"{name}_word"] == r["gt_crop"] for r in det]),
          " visible MAE", mae([r[f"{name}_visible"] for r in det], [r["gt_visible"] for r in det]),
          " body_in_frame MAE", mae([r[f"{name}_bif"] for r in det], [r["gt_bif"] for r in det]))
    nx = [r for r in det if not r["extent_ok"]]
    if nx:
        print(f"              (no extent fit, n={len(nx)})  word", pct([r[f"{name}_word"] == r["gt_crop"] for r in nx]),
              " visible MAE", mae([r[f"{name}_visible"] for r in nx], [r["gt_visible"] for r in nx]))
from collections import Counter
print("GT crop mix:", dict(Counter(r["gt_crop"] for r in det)))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"n": len(rows), "rows": rows}, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
