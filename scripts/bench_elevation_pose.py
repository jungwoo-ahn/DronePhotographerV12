"""Can body-pose keypoints recover CAMERA ELEVATION, where the VLM was at chance (34-36%)?

Camera pitch shows up in keypoint geometry: vertical foreshortening (head->hip vs hip->ankle ratio
compresses as the camera tilts), head size relative to body length, and where the body sits in frame.
Same trick that solved bearing. GT elevation = cam_to_obj_elevation_deg on our renders.
venv: .venv-analysis (GPU)."""
import json, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
import cv2
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *a, **k: None
from ultralytics import YOLO

from src.goal_authoring import vocab
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ROOT = DEFAULT_TRAJ_ROOT
CACHE = "runs/elevation_feats.npz"
REFRESH = "--refresh" in sys.argv


def elev_features(kp, box, W, H):
    """Vertical-geometry features that encode camera pitch (translation-invariant, scale-normalized)."""
    xy, c = kp[:, :2], kp[:, 2]
    nose, leye, reye = xy[0], xy[1], xy[2]
    lsh, rsh, lhip, rhip = xy[5], xy[6], xy[11], xy[12]
    lkn, rkn, lank, rank = xy[13], xy[14], xy[15], xy[16]
    sh, hip = 0.5 * (lsh + rsh), 0.5 * (lhip + rhip)
    kn, ank = 0.5 * (lkn + rkn), 0.5 * (lank + rank)
    bh = max(float(box[3] - box[1]), 1e-6)          # body pixel height = scale
    bw = max(float(box[2] - box[0]), 1e-6)

    def vy(a, b): return float((b[1] - a[1]) / bh)   # signed vertical span, normalized

    torso = vy(sh, hip); thigh = vy(hip, kn); shin = vy(kn, ank)
    head = vy(nose, sh)
    seg = [head, torso, thigh, shin]
    tot = sum(abs(s) for s in seg) + 1e-6
    eye_sep = float(np.linalg.norm(leye - reye)) / bh          # head apparent size
    sh_w = float(np.linalg.norm(lsh - rsh)) / bh
    hip_w = float(np.linalg.norm(lhip - rhip)) / bh
    return [
        head, torso, thigh, shin,                               # raw vertical spans
        head / tot, torso / tot, thigh / tot, shin / tot,        # RATIOS: foreshortening signature
        (torso / (thigh + shin + 1e-6)),                         # upper vs lower body compression
        eye_sep, sh_w, hip_w, sh_w / (hip_w + 1e-6),
        eye_sep / (abs(torso) + 1e-6),                           # head size vs torso (pitch cue)
        bh / H, bw / W, bh / (bw + 1e-6),                        # apparent size / aspect
        float((box[1] + box[3]) * 0.5 / H),                      # vertical position in frame
        float(box[3] / H), float(box[1] / H),                    # feet / head frame position
        float(c[0]), float(c[15]), float(c[16]),                 # nose / ankle confidence
        float(np.mean(c[5:13])),
    ]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def extract():
    yolo = YOLO("yolo11l-pose.pt")
    dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    random.seed(0); random.shuffle(dirs)
    X, y, n = [], [], 0
    for dn in dirs:
        try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
        except Exception: continue
        W, H = d.get("render_width", 1024), d.get("render_height", 768)
        cand = [(r, s) for pair in d.get("render_records", []) for r in pair
                if (s := r.get("scores")) and r.get("in_frame") and 25 <= s["occupancy"] <= 92]
        if not cand: continue
        # up to 2 frames per placement, spread in elevation, to cover the band range
        random.shuffle(cand)
        for r, s in cand[:2]:
            n += 1
            bb = r.get("bbox_xyxy_full")
            if bb is None: continue
            res = yolo.predict(os.path.join(ROOT, dn, r["path_rel"]), verbose=False, device=0)[0]
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None: continue
            boxes = res.boxes.xyxy.cpu().numpy(); j = int(np.argmax([iou(b, bb) for b in boxes]))
            if iou(boxes[j], bb) < 0.2: continue
            X.append(elev_features(res.keypoints.data.cpu().numpy()[j], boxes[j], W, H))
            y.append(float(s["cam_to_obj_elevation_deg"]))
        if n % 600 == 0: print(f"  scanned {n}, kept {len(X)}", flush=True)
        if len(X) > 4000: break
    return np.array(X), np.array(y)


if os.path.exists(CACHE) and not REFRESH:
    dat = np.load(CACHE); X, y = dat["X"], dat["y"]
    print(f"loaded cache {X.shape}", flush=True)
else:
    X, y = extract()
    os.makedirs("runs", exist_ok=True); np.savez(CACHE, X=X, y=y)
    print(f"cached -> {CACHE} {X.shape}", flush=True)

bands = np.array([vocab._classify(v, vocab.ELEVATION) for v in y])
print(f"samples={len(X)}  band distribution: "
      f"{ {b: int((bands==b).sum()) for b in set(bands)} }", flush=True)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

pred = cross_val_predict(RandomForestRegressor(200, random_state=0, n_jobs=-1), X, y, cv=5)
mae = float(np.mean(np.abs(pred - y)))
pb = np.array([vocab._classify(v, vocab.ELEVATION) for v in pred])
acc = float(np.mean(pb == bands))
# majority-class baseline
maj = max(set(bands), key=lambda b: (bands == b).sum())
base = float(np.mean(bands == maj))

print("\n===== ELEVATION from pose keypoints (5-fold CV) =====")
print(f"regression MAE = {mae:.1f} deg   (GT range {y.min():.0f}..{y.max():.0f}, std {y.std():.1f})")
print(f"band(high/eye/low) acc = {100*acc:.0f}%   majority-baseline = {100*base:.0f}%   VLM was ~34-36% (chance 33%)")
print("INTERPRET: acc >> baseline and MAE << std => pose DOES encode camera pitch; elevation becomes usable.")
