"""Can single-image bearing beat the 68% VLM baseline using YuNet face landmarks (no model install)?
Face detected => subject faces the camera hemisphere; nose offset between the eyes => yaw (front vs
3/4 vs profile); no face => back hemisphere. Calibrate a simple rule vs GT and report sector3/sector8.
venv: .venv-analysis (cv2 CPU)."""
import json, math, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np, cv2

from src.common.facing import front_azimuth, sector3, sector8

ROOT = "data/trajectories"
det = cv2.FaceDetectorYN.create("assets/models/yunet_2023mar.onnx", "", (320, 320), 0.5, 0.3, 50)

dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
random.seed(0); random.shuffle(dirs)
rows = []
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
    # detect face in the head region
    im = cv2.imread(os.path.join(ROOT, dn, r["path_rel"]))
    if im is None: continue
    H, W = im.shape[:2]; bb = r.get("bbox_xyxy_full")
    if bb:
        x0, y0, x1, y1 = [int(v) for v in bb]; bw, bh = x1 - x0, y1 - y0
        x0 = max(0, int(x0 - bw * 0.2)); x1 = min(W, int(x1 + bw * 0.2))
        y0 = max(0, int(y0 - bh * 0.1)); y1 = min(H, int(y0 + bh * 0.55))
        if x1 - x0 > 24 and y1 - y0 > 24: im = im[y0:y1, x0:x1]
    h, w = im.shape[:2]
    sc = 512 / max(h, w)
    if sc > 1: im = cv2.resize(im, (int(w * sc), int(h * sc))); h, w = im.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(im)
    if faces is None or not len(faces):
        rows.append((gt, False, 0.0)); continue
    f = faces[np.argmax(faces[:, 14])]
    re_x, le_x, nose_x = f[4], f[6], f[8]
    eye_sep = abs(le_x - re_x) + 1e-6
    r_off = float((nose_x - 0.5 * (re_x + le_x)) / eye_sep)  # nose offset between eyes (yaw proxy)
    rows.append((gt, True, r_off))

n = len(rows); det_rate = sum(1 for _, d, _ in rows if d) / n
print(f"samples={n}  face-detect rate={100*det_rate:.0f}%", flush=True)
# detection vs GT hemisphere
det_front = [d for gt, d, _ in rows if sector3(gt) != "back"]
det_back = [d for gt, d, _ in rows if sector3(gt) == "back"]
print(f"detect rate | GT front/side={100*np.mean(det_front):.0f}%  GT back={100*np.mean(det_back):.0f}%  "
      f"(want high vs low = face separates front-hemisphere from back)")

# calibrate a rule: no-face->back(180); face-> front(0) if |r|<t else side; sign of r -> which side.
def predict(detected, r_off, t, sgn):
    if not detected: return 180.0
    if abs(r_off) < t: return 0.0
    return 90.0 if (sgn * r_off) > 0 else 270.0

best = None
for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
    for sgn in (+1, -1):
        s3 = np.mean([sector3(predict(d, r, t, sgn)) == sector3(gt) for gt, d, r in rows])
        if best is None or s3 > best[0]: best = (s3, t, sgn)
s3_acc, t, sgn = best
s8_acc = np.mean([sector8(predict(d, r, t, sgn)) == sector8(gt) for gt, d, r in rows])
print(f"\n===== YuNet-landmark bearing (best rule: |nose|<{t}, sign {sgn:+d}) =====")
print(f"sector3(front/side/back) acc={100*s3_acc:.0f}%   sector8 acc={100*s8_acc:.0f}%")
print(f"(VLM baseline was s3=68%, s8=32%)")
