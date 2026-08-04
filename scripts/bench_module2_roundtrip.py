"""Killer validation for Module 2: run the reference-image estimator on our OWN renders (where the
full GT profile is known) and check the recovered goal matches. Measures both raw error and, more
importantly, CINEMATOGRAPHY-CATEGORY agreement (shot size / placement / bearing sector) — since the
goal ultimately conditions on the category, small raw gaps that stay in-band don't matter.
Also surfaces the YOLO-bbox vs training-mesh-bbox convention gap on occupancy/placement.
venv: .venv-analysis (GPU)."""
import json, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.facing import front_azimuth, sector3, sector8
from src.goal_authoring import vocab
from src.goal_authoring.from_reference import ReferenceEstimator
from src.scoring.bbox_control import compute_v5_scores

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ROOT = "data/trajectories"
est = ReferenceEstimator()

dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
random.seed(1); random.shuffle(dirs)
occ_err, cx_err, cy_err = [], [], []
shot_ok = place_ok = place_y_ok = 0; place_n = 0
b_s3 = b_s8 = 0; b_n = 0; b_err = []
n = 0
for dn in dirs:
    if n >= N: break
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if front_azimuth(obj) is None: continue
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: continue
    W, H = d.get("render_width", 1024), d.get("render_height", 768)
    cand = [(r, s) for pair in d.get("render_records", []) for r in pair
            if (s := r.get("scores")) and r.get("in_frame") and 30 <= s["occupancy"] <= 92 and s["body_in_frame_ratio"] >= 45]
    if not cand: continue
    r, s = random.choice(cand)
    bb = r.get("bbox_xyxy_full")
    if bb is None: continue
    # GT under the VISIBLE-bbox convention (same as the training goal after _apply_visible_geometry)
    gt = compute_v5_scores(int(W), int(H), [float(v) for v in bb],
                           float(s["cam_to_obj_azimuth_deg"]), float(s["cam_to_obj_elevation_deg"]))
    gt["occupancy"] = s["occupancy"]                 # occupancy keeps the stored clipped ground truth
    gp = est(os.path.join(ROOT, dn, r["path_rel"]))
    if "occupancy" not in gp.specified: continue     # no person detected
    n += 1
    # geometry
    occ_err.append(abs(gp.values["occupancy"] - gt["occupancy"]))
    cx_err.append(abs(gp.values["object_center_x"] - gt["object_center_x"]))
    cy_err.append(abs(gp.values["object_center_y"] - gt["object_center_y"]))
    shot_ok += (vocab._classify(gp.values["occupancy"], vocab.SHOT_SIZE) == vocab._classify(gt["occupancy"], vocab.SHOT_SIZE))
    place_n += 1
    place_ok += (vocab._classify(gp.values["object_center_x"]/W, vocab.PLACE_X) == vocab._classify(gt["object_center_x"]/W, vocab.PLACE_X))
    place_y_ok += (vocab._classify(gp.values["object_center_y"]/H, vocab.PLACE_Y) == vocab._classify(gt["object_center_y"]/H, vocab.PLACE_Y))
    # bearing
    if "subject_bearing_deg" in gp.specified:
        gt_b = (front_azimuth(obj) - s["cam_to_obj_azimuth_deg"]) % 360
        pb = gp.values["subject_bearing_deg"]; b_n += 1
        b_err.append(abs(((pb - gt_b + 180) % 360) - 180))
        b_s3 += (sector3(pb) == sector3(gt_b)); b_s8 += (sector8(pb) == sector8(gt_b))

print(f"\n===== Module 2 GT round-trip (n={n} renders with a detected subject) =====")
print(f"GEOMETRY (YOLO bbox vs training mesh-bbox GT):")
print(f"  occupancy: MAE={np.mean(occ_err):.1f}%   shot-size CATEGORY match={100*shot_ok/n:.0f}%")
print(f"  object_center: MAE=({np.mean(cx_err):.0f},{np.mean(cy_err):.0f})px   placement CATEGORY match x={100*place_ok/place_n:.0f}% y={100*place_y_ok/place_n:.0f}%")
print(f"BEARING (n={b_n} with confident bearing):")
if b_n:
    print(f"  sector3 acc={100*b_s3/b_n:.0f}%   sector8 acc={100*b_s8/b_n:.0f}%   MAE={np.mean(b_err):.0f}deg")
print("INTERPRET: high category-match => Module 2's goal matches training's goal space (usable). "
      "Large occupancy MAE but high shot-size match => raw scale differs but the conditioning category is right.")
