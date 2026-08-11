"""Fit + save the subject-bearing regressor used by Module 2 (reference image -> goal).
Extracts YOLO-pose keypoint features on our GT renders (GT bearing = front_az - cam_azimuth from the
verified facing map), caches them, reports 5-fold CV, then fits a compact RandomForest on (sin,cos)
and saves it. venv: .venv-analysis."""
import argparse, json, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.annotations import is_goal_frame
from src.common.facing import front_azimuth, sector3, sector8
from src.goal_authoring.pose_bearing import BearingModel, pose_features
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--pose-model", default="yolo11l-pose.pt")
ap.add_argument("--out", default="assets/models/bearing_pose_rf.joblib")
ap.add_argument("--trees", type=int, default=120)
ap.add_argument("--max-depth", type=int, default=14)      # cap so the committed artifact stays small
ap.add_argument("--cache", default="runs/bearing_feats.npz")
ap.add_argument("--refresh", action="store_true")
args = ap.parse_args()

ROOT = DEFAULT_TRAJ_ROOT


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def extract():
    import cv2
    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda *a, **k: None
    from ultralytics import YOLO
    yolo = YOLO(args.pose_model)
    dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    random.seed(0); random.shuffle(dirs)
    X, y, n = [], [], 0
    for dn in dirs:
        obj = dn.split("__", 1)[1] if "__" in dn else dn
        if front_azimuth(obj) is None:
            continue
        try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
        except Exception: continue
        cand = [(r, s) for pair in d.get("render_records", []) for r in pair
                if is_goal_frame(r)]
        if not cand: continue
        r, s = random.choice(cand); n += 1
        gt = (front_azimuth(obj) - s["cam_to_obj_azimuth_deg"]) % 360
        bb = r.get("bbox_xyxy_full")
        if bb is None: continue
        res = yolo.predict(os.path.join(ROOT, dn, r["path_rel"]), verbose=False, device=0)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None: continue
        boxes = res.boxes.xyxy.cpu().numpy(); j = int(np.argmax([iou(b, bb) for b in boxes]))
        if iou(boxes[j], bb) < 0.2: continue
        X.append(pose_features(res.keypoints.data.cpu().numpy()[j], boxes[j])); y.append(gt)
        if n % 500 == 0: print(f"  scanned {n}, kept {len(X)}", flush=True)
    return np.array(X), np.array(y)


if os.path.exists(args.cache) and not args.refresh:
    dat = np.load(args.cache); X, y = dat["X"], dat["y"]
    print(f"loaded cached features: {X.shape}", flush=True)
else:
    X, y = extract()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, X=X, y=y)
    print(f"cached features -> {args.cache}  ({X.shape})", flush=True)

y3 = np.array([sector3(v) for v in y]); y8 = np.array([sector8(v) for v in y])
sc = np.c_[np.sin(np.deg2rad(y)), np.cos(np.deg2rad(y))]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

def make(): return RandomForestRegressor(args.trees, max_depth=args.max_depth, random_state=0, n_jobs=-1)

pred = cross_val_predict(make(), X, sc, cv=5)
ang = np.rad2deg(np.arctan2(pred[:, 0], pred[:, 1])) % 360
mae = float(np.mean([abs(((a - b + 180) % 360) - 180) for a, b in zip(ang, y)]))
acc3 = float(np.mean([sector3(a) == b for a, b in zip(ang, y3)]))
acc8 = float(np.mean([sector8(a) == b for a, b in zip(ang, y8)]))
print(f"5-fold CV (trees={args.trees}, depth={args.max_depth}): "
      f"sector3={100*acc3:.0f}%  sector8={100*acc8:.0f}%  MAE={mae:.0f}deg", flush=True)

BearingModel(make().fit(X, sc)).save(args.out)
print(f"saved {args.out}  ({os.path.getsize(args.out)//1024} KB)")
