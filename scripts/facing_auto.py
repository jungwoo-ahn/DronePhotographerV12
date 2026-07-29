"""Automated per-asset facing (first pass): aggregate FACE-detection confidence over the full orbit.
For each asset, run YuNet on every usable frame (known cam azimuth A); smooth face-confidence over the
azimuth circle; the PEAK azimuth = camera at the subject's FRONT => facing = peak_az + 180. Robust
because it aggregates hundreds of frames (per-frame noise averages out) with a purpose-built detector.
Emits a confidence per asset so the low-confidence tail can be human-verified.
Self-contained; .venv-analysis (cv2 CPU). Writes runs/facing_auto.json."""
import os, json, math, random, argparse
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--max-assets", type=int, default=10)
ap.add_argument("--max-frames", type=int, default=250)     # per asset cap
ap.add_argument("--out", default="runs/facing_auto.json")
args = ap.parse_args()

ROOT = "data/trajectories"
det = cv2.FaceDetectorYN.create("assets/models/yunet_2023mar.onnx", "", (320,320), 0.4, 0.3, 50)

dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
by_obj = {}
for d in sorted(dirs):
    by_obj.setdefault(d.split("__",1)[1] if "__" in d else d, d)
random.seed(0)
objs = random.sample(list(by_obj.items()), min(args.max_assets, len(by_obj)))

def face_conf(imgpath, bbox):
    im = cv2.imread(imgpath)
    if im is None: return 0.0
    H, W = im.shape[:2]
    if bbox:
        x0,y0,x1,y1 = [int(v) for v in bbox]
        bw,bh = x1-x0, y1-y0
        # focus on the HEAD region (top ~55% of the body bbox) so the face is larger for the detector
        x0=max(0,int(x0-bw*0.2)); x1=min(W,int(x1+bw*0.2))
        y0=max(0,int(y0-bh*0.1)); y1=min(H,int(y0+bh*0.55))
        if x1-x0<24 or y1-y0<24: return 0.0
        im = im[y0:y1, x0:x1]
    h,w = im.shape[:2]
    s = 512/max(h,w)                       # upscale small crops
    if s>1: im = cv2.resize(im,(int(w*s),int(h*s))); h,w = im.shape[:2]
    det.setInputSize((w,h))
    _, faces = det.detect(im)
    return float(faces[:,14].max()) if faces is not None and len(faces) else 0.0

def circ_smooth(azs, vals, kappa=6.0, step=5):
    azs = np.deg2rad(np.array(azs)); vals = np.array(vals)
    grid = np.deg2rad(np.arange(0,360,step))
    curve = []
    for g in grid:
        w = np.exp(kappa*np.cos(g-azs)); curve.append((w*vals).sum()/ (w.sum()+1e-9))
    return np.arange(0,360,step), np.array(curve)

def cdiff(a,b): return abs(((a-b+180)%360)-180)

results = {}
for obj, dn in objs:
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: continue
    frames = []
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or not r.get("in_frame") or s["occupancy"] < 20: continue
            frames.append((s["cam_to_obj_azimuth_deg"], os.path.join(ROOT,dn,r["path_rel"]), r.get("bbox_xyxy_full")))
    if len(frames) > args.max_frames:
        idx = np.linspace(0,len(frames)-1,args.max_frames).astype(int); frames = [frames[i] for i in idx]
    azs = [A for A,_,_ in frames]
    confs = [face_conf(ip,bb) for _,ip,bb in frames]
    ndet = sum(c>0.6 for c in confs)
    if ndet < 5:
        results[obj] = {"n_frames":len(frames), "n_detect":ndet, "status":"too_few_faces"}
        print(f"  {obj[:30]:30s} n={len(frames)} faces={ndet} -> too few", flush=True); continue
    grid, curve = circ_smooth(azs, confs)
    pk = int(np.argmax(curve)); front_az = float(grid[pk]); facing = (front_az+180)%360
    mn = int(np.argmin(curve)); back_az = float(grid[mn])
    contrast = float((curve.max()-curve.min())/(curve.max()+1e-9))
    sep = cdiff(front_az, back_az)                     # face-min should be ~180 from face-peak
    conf = "OK" if (contrast>0.4 and abs(sep-180)<50 and ndet>=15) else "VERIFY"
    results[obj] = {"n_frames":len(frames), "n_detect":ndet, "front_az":round(front_az),
                    "facing_world_deg":round(facing), "contrast":round(contrast,2),
                    "face_min_az":round(back_az), "front_back_sep":round(sep), "conf":conf}
    print(f"  {obj[:30]:30s} faces={ndet:3d} front={front_az:4.0f} facing≈{facing:4.0f}° contrast={contrast:.2f} sep={sep:3.0f} {conf}", flush=True)

lab = [r for r in results.values() if "facing_world_deg" in r]
ok = [r for r in lab if r["conf"]=="OK"]
print(f"\n=== summary ===")
print(f"labeled {len(lab)}/{len(objs)};  auto-confident (OK): {len(ok)}/{len(lab)};  need human-verify: {len(lab)-len(ok)} + {len(objs)-len(lab)} no-face")
print("INTERPRET: high contrast + face-min ~180 from face-peak => clean unimodal front peak => facing reliable.")
os.makedirs("runs", exist_ok=True); json.dump(results, open(args.out,"w"), indent=2)
print(f"wrote {args.out}")
