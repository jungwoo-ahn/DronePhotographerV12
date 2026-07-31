"""Auto facing (first pass) on the CLEAN isolated turntable renders — replaces facing_auto.py's
noisy scene-frame pass. For each asset, run YuNet on its K turntable views (the object is centered
and large, so detection is far more reliable than on scene frames), smooth face-confidence over the
azimuth circle; the peak azimuth = camera at the subject's FRONT => facing = peak + 180.

Output schema matches facing_auto_full.json (front_az / facing_world_deg / conf / ...) so it is a
drop-in for the verify UI and downstream. Assets with no detectable face (snowman, etc.) fall to
`too_few_faces` and are picked by a human in the verify gallery.

Self-contained; .venv-analysis (cv2 CPU). Reads runs/facing_turntable/index.json.
Writes runs/facing_turntable_auto.json.
"""
import argparse
import json
import os

os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="runs/facing_turntable/index.json")
ap.add_argument("--out", default="runs/facing_turntable_auto.json")
args = ap.parse_args()

det = cv2.FaceDetectorYN.create("assets/models/yunet_2023mar.onnx", "", (320, 320), 0.4, 0.3, 50)


def face_conf(imgpath):
    """Max YuNet face score on the head region (isolated render => object is centered)."""
    im = cv2.imread(imgpath)
    if im is None:
        return 0.0
    H, W = im.shape[:2]
    # Head region: top ~55%, central 70% of the frame (character is centered & upright).
    crop = im[int(0.03 * H):int(0.55 * H), int(0.15 * W):int(0.85 * W)]
    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return 0.0
    s = 512 / max(h, w)
    if s > 1:
        crop = cv2.resize(crop, (int(w * s), int(h * s)))
        h, w = crop.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(crop)
    return float(faces[:, 14].max()) if faces is not None and len(faces) else 0.0


def circ_smooth(azs, vals, kappa=6.0, step=5):
    azs = np.deg2rad(np.array(azs))
    vals = np.array(vals)
    grid = np.deg2rad(np.arange(0, 360, step))
    curve = []
    for g in grid:
        w = np.exp(kappa * np.cos(g - azs))
        curve.append((w * vals).sum() / (w.sum() + 1e-9))
    return np.arange(0, 360, step), np.array(curve)


def cdiff(a, b):
    return abs(((a - b + 180) % 360) - 180)


idx = json.load(open(args.index))
results = {}
for obj, info in idx.items():
    if "error" in info or not info.get("views"):
        results[obj] = {"status": "render_error"}
        print(f"  {obj[:34]:34s} render error", flush=True)
        continue
    azs = [v["az"] for v in info["views"]]
    confs = [face_conf(v["path"]) for v in info["views"]]
    ndet = sum(c > 0.6 for c in confs)
    if ndet < 2:
        results[obj] = {"n_views": len(azs), "n_detect": ndet, "status": "too_few_faces"}
        print(f"  {obj[:34]:34s} faces={ndet} -> too few (human pick)", flush=True)
        continue
    grid, curve = circ_smooth(azs, confs)
    pk = int(np.argmax(curve))
    front = float(grid[pk])
    facing = (front + 180) % 360
    mn = int(np.argmin(curve))
    back = float(grid[mn])
    contrast = float((curve.max() - curve.min()) / (curve.max() + 1e-9))
    sep = cdiff(front, back)  # face-min should sit ~180 from the face-peak for a clean front
    conf = "OK" if (contrast > 0.45 and abs(sep - 180) < 55 and ndet >= 3) else "VERIFY"
    results[obj] = {
        "n_views": len(azs), "n_detect": ndet, "front_az": round(front),
        "facing_world_deg": round(facing), "contrast": round(contrast, 2),
        "face_min_az": round(back), "front_back_sep": round(sep), "conf": conf,
    }
    print(f"  {obj[:34]:34s} faces={ndet:2d} front={front:4.0f} facing≈{facing:4.0f}° "
          f"contrast={contrast:.2f} sep={sep:3.0f} {conf}", flush=True)

lab = [r for r in results.values() if "facing_world_deg" in r]
ok = [r for r in lab if r.get("conf") == "OK"]
print(f"\n=== summary === labeled {len(lab)}/{len(idx)};  OK {len(ok)};  "
      f"VERIFY {len(lab) - len(ok)};  no-face/err {len(idx) - len(lab)}")
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(results, open(args.out, "w"), indent=2)
print(f"wrote {args.out}")
