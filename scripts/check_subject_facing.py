"""Is subject canonical facing SHARED across assets (=> one global offset from world-azimuth to
subject-relative) or per-asset? For several DISTINCT objects, find the cam->subject azimuth where a
FRONTAL face detector peaks (= that subject's front), and check cross-object consistency.
Self-contained; reads data symlink + Haar cascades in assets/. venv: .venv-analysis."""
import os, sys, json, math, random
from src.common.dataset_base import DEFAULT_TRAJ_ROOT
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np, cv2

ROOT = DEFAULT_TRAJ_ROOT
fc = cv2.CascadeClassifier("assets/haarcascades/haarcascade_frontalface_default.xml")
pc = cv2.CascadeClassifier("assets/haarcascades/haarcascade_profileface.xml")

# one placement per DISTINCT object
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
by_obj = {}
for d in sorted(dirs):
    obj = d.split("__", 1)[1] if "__" in d else d
    by_obj.setdefault(obj, d)
random.seed(0)
objs = random.sample(list(by_obj.items()), min(12, len(by_obj)))

def detect(imgpath, bbox):
    im = cv2.imread(imgpath)
    if im is None: return 0, 0
    H, W = im.shape[:2]
    if bbox:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 24 or y1 - y0 < 24: return 0, 0
        im = im[y0:y1, x0:x1]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # upscale small crops to help the detector
    if max(g.shape) < 200:
        s = 200 / max(g.shape); g = cv2.resize(g, None, fx=s, fy=s)
    fr = fc.detectMultiScale(g, 1.1, 4, minSize=(24, 24))
    gp = pc.detectMultiScale(g, 1.1, 4, minSize=(24, 24))
    gpf = pc.detectMultiScale(cv2.flip(g, 1), 1.1, 4, minSize=(24, 24))  # profile is orientation-specific
    return (1 if len(fr) else 0), (1 if (len(gp) or len(gpf)) else 0)

print(f"testing {len(objs)} distinct objects\n")
NSEC = 8
front_az = []   # per-object front azimuth (deg), None if undetermined
for obj, dn in objs:
    p = os.path.join(ROOT, dn, "data.json")
    try: d = json.load(open(p))
    except Exception: continue
    fr_by = np.zeros(NSEC); pr_by = np.zeros(NSEC); n_by = np.zeros(NSEC)
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or s["occupancy"] < 35 or not r.get("in_frame"): continue
            bb = r.get("bbox_xyxy_full")
            f, pf = detect(os.path.join(ROOT, dn, r["path_rel"]), bb)
            sec = int((s["cam_to_obj_azimuth_deg"] % 360) // (360 / NSEC))
            fr_by[sec] += f; pr_by[sec] += pf; n_by[sec] += 1
    tot_f = fr_by.sum()
    if tot_f < 3:
        front_az.append(None)
        print(f"{obj[:34]:34s} frontal-hits={int(tot_f)} (too few) -> undetermined")
        continue
    # front sector = argmax frontal rate (with coverage)
    rate = fr_by / np.maximum(n_by, 1)
    peak = int(np.argmax(np.where(n_by >= 2, rate, -1)))
    az = peak * (360 / NSEC) + (360 / NSEC) / 2
    front_az.append(az)
    print(f"{obj[:34]:34s} frontal-hits={int(tot_f):3d}  front≈{az:5.0f}°  "
          f"frontal-rate/sector={[round(x,2) for x in rate.tolist()]}")

vals = [a for a in front_az if a is not None]
print(f"\n=== consistency of 'front azimuth' across {len(vals)} objects ===")
if len(vals) >= 3:
    r = np.deg2rad(vals)
    C, S = np.cos(r).mean(), np.sin(r).mean()
    circ_mean = math.degrees(math.atan2(S, C)) % 360
    R = math.hypot(C, S)                          # 1=perfectly aligned, 0=uniform
    circ_std = math.degrees(math.sqrt(max(0, -2*math.log(max(R,1e-9)))))
    print(f"front azimuths: {[round(a) for a in vals]}")
    print(f"circular mean={circ_mean:.0f}°  concentration R={R:.2f}  circular std≈{circ_std:.0f}°")
    if R > 0.7:
        print(f"=> SHARED facing (R>0.7): world-azimuth -> subject-relative is ~a GLOBAL offset ({circ_mean:.0f}°). EASY.")
    else:
        print("=> facing VARIES per asset (R low): need a PER-ASSET facing map (or a VLM/model pass).")
else:
    print("too few determined; face detector may be unreliable on these renders -> consider a VLM pass")
print("\nDONE")
