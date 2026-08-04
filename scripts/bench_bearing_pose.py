"""Can body-pose keypoints beat the 68% VLM bearing baseline? YOLO-pose detects the whole person
(high recall, unlike face-only YuNet's 18%). Anatomical left/right shoulder x-ordering flips between
front and back (mirroring); facial-keypoint confidence separates front from back; shoulder alignment
=> profile. Fit a classifier on these features vs GT bearing (cross-validated). venv: .venv-analysis."""
import json, math, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
import cv2
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *a, **k: None
from ultralytics import YOLO

from src.common.facing import front_azimuth, sector3, sector8

ROOT = "data/trajectories"
model = YOLO("yolo11l-pose.pt")

# ---- collect GT samples (one frame per placement, subject visible, GT bearing known) ----
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
random.seed(0); random.shuffle(dirs)
items = []
for dn in dirs:
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if front_azimuth(obj) is None:
        continue
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: continue
    cand = [(r, s) for pair in d.get("render_records", []) for r in pair
            if (s := r.get("scores")) and r.get("in_frame") and 30 <= s["occupancy"] <= 92 and s["body_in_frame_ratio"] >= 45]
    if not cand: continue
    r, s = random.choice(cand)
    gt = (front_azimuth(obj) - s["cam_to_obj_azimuth_deg"]) % 360
    items.append((os.path.join(ROOT, dn, r["path_rel"]), r.get("bbox_xyxy_full"), gt))
print(f"GT samples: {len(items)}", flush=True)

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)

def features(kp, box):
    # kp: (17,3) [x,y,conf]; build orientation features, normalized by body scale, translation-invariant
    xy, c = kp[:, :2], kp[:, 2]
    lsh, rsh, lhip, rhip = xy[5], xy[6], xy[11], xy[12]
    sh_mid = 0.5 * (lsh + rsh)
    scale = np.linalg.norm(sh_mid - 0.5 * (lhip + rhip)) + 1e-6  # shoulder->hip (torso height)
    if scale < 5:
        scale = (box[3] - box[1]) * 0.3 + 1e-6
    def dxn(a, b): return float((a[0] - b[0]) / scale)          # signed normalized x-gap
    feats = [
        dxn(lsh, rsh), dxn(lhip, rhip), dxn(xy[3], xy[4]), dxn(xy[1], xy[2]),  # L-R x order (front/back sign)
        float((xy[0][0] - sh_mid[0]) / scale),                                 # nose vs shoulder-mid x
        float(np.linalg.norm(lsh - rsh) / scale),                              # shoulder width (profile->small)
        float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]),       # nose/eyes/ears conf (front/back)
        float(c[5]), float(c[6]), float(c[11]), float(c[12]),                  # shoulder/hip conf
        float(c[1] - c[2]), float(c[3] - c[4]),                                # eye/ear L-R conf asym (yaw)
    ]
    return feats

X, y = [], []
det = 0
for i, (ip, bb, gt) in enumerate(items):
    res = model.predict(ip, verbose=False, device=0)[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0 or bb is None:
        continue
    boxes = res.boxes.xyxy.cpu().numpy()
    ious = [iou(b, bb) for b in boxes]
    j = int(np.argmax(ious))
    if ious[j] < 0.2:
        continue
    kp = res.keypoints.data.cpu().numpy()[j]
    X.append(features(kp, boxes[j])); y.append(gt); det += 1
    if (i + 1) % 500 == 0: print(f"  {i+1}/{len(items)} (det {det})", flush=True)

X = np.array(X); y = np.array(y)
print(f"\ndetections used: {len(X)}/{len(items)} ({100*len(X)/len(items):.0f}%)", flush=True)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict

y3 = np.array([sector3(v) for v in y]); y8 = np.array([sector8(v) for v in y])
p3 = cross_val_predict(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y3, cv=5)
p8 = cross_val_predict(RandomForestClassifier(300, random_state=0, n_jobs=-1), X, y8, cv=5)
acc3 = float(np.mean(p3 == y3)); acc8 = float(np.mean(p8 == y8))
# circular angle regression (sin/cos) -> MAE
sc = np.c_[np.sin(np.deg2rad(y)), np.cos(np.deg2rad(y))]
pred = cross_val_predict(RandomForestRegressor(300, random_state=0, n_jobs=-1), X, sc, cv=5)
ang = np.rad2deg(np.arctan2(pred[:, 0], pred[:, 1])) % 360
mae = float(np.mean([abs(((a - b + 180) % 360) - 180) for a, b in zip(ang, y)]))

print("\n===== YOLO-pose + RandomForest (5-fold CV) bearing =====")
print(f"sector3(front/side/back) acc={100*acc3:.0f}%   sector8 acc={100*acc8:.0f}%   circular MAE={mae:.0f}°")
print(f"(VLM baseline: s3=68%, s8=32%, MAE=45°;  chance: s3~33%, s8~12%)")
