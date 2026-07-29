"""Recover per-asset canonical facing (world azimuth) via a VLM, so world cam->subject azimuth can
be converted to SUBJECT-RELATIVE view (front/side/back) for the goal descriptor.

Method (self-validating): for each asset, take several frames at DIFFERENT camera azimuths A; ask
Qwen2.5-VL which side of the person faces the camera (8-way view V). Each frame implies the asset's
facing world-bearing:  phi = (A + 180 - offset(V)) mod 360.  If the VLM+convention are right, all
frames of ONE asset agree on phi (low circular std = confidence). Sign of offset is calibrated
globally by whichever gives better within-asset agreement.

Runs on the shared venv (torch/transformers). Reads data symlink + HF cache. Writes runs/facing_map.json.
"""
import os, sys, json, math, random, argparse
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--max-assets", type=int, default=8)      # pilot default; set high for full 102
ap.add_argument("--frames-per-asset", type=int, default=5)
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--out", default="runs/facing_map.json")
args = ap.parse_args()

ROOT = "data/trajectories"
LABELS = ["FRONT","FRONT_RIGHT","RIGHT","BACK_RIGHT","BACK","BACK_LEFT","LEFT","FRONT_LEFT"]
OFFSET = {L: i*45 for i, L in enumerate(LABELS)}          # convention; sign calibrated below

# ---- select assets + diverse frames ----
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
by_obj = {}
for d in sorted(dirs):
    obj = d.split("__", 1)[1] if "__" in d else d
    by_obj.setdefault(obj, d)
random.seed(0)
objs = random.sample(list(by_obj.items()), min(args.max_assets, len(by_obj)))

def pick_frames(dn, k):
    p = os.path.join(ROOT, dn, "data.json")
    try: d = json.load(open(p))
    except Exception: return []
    cand = []
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or not r.get("in_frame"): continue
            if not (30 <= s["occupancy"] <= 88 and s["body_in_frame_ratio"] >= 45): continue
            cand.append((s["cam_to_obj_azimuth_deg"], os.path.join(ROOT, dn, r["path_rel"]), r.get("bbox_xyxy_full")))
    if len(cand) < 2: return []
    # spread across azimuth sectors
    cand.sort(key=lambda c: c[0])
    idx = np.linspace(0, len(cand)-1, min(k, len(cand))).astype(int)
    return [cand[i] for i in idx]

tasks = [(obj, dn, pick_frames(dn, args.frames_per_asset)) for obj, dn in objs]
tasks = [(o,dn,fr) for o,dn,fr in tasks if fr]
print(f"assets with usable frames: {len(tasks)}/{len(objs)}  (frames/asset ~{args.frames_per_asset})", flush=True)

# ---- load VLM ----
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
print("loading VLM:", args.model, flush=True)
proc = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

PROMPT = ("This is a rendered 3D human character. From THIS camera view, which side of the person "
          "faces the camera? Answer with EXACTLY one label:\n"
          "FRONT (see the face/chest), BACK (back of head/back), LEFT (person's left side in profile), "
          "RIGHT (person's right side in profile), FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT "
          "(three-quarter views). Reply with only the label.")

def crop(imgpath, bbox, pad=0.35):
    im = Image.open(imgpath).convert("RGB"); W, H = im.size
    if bbox:
        x0,y0,x1,y1 = bbox; bw,bh = x1-x0, y1-y0
        x0 = max(0, x0-bw*pad); x1 = min(W, x1+bw*pad); y0 = max(0, y0-bh*pad); y1 = min(H, y1+bh*pad)
        if x1-x0 > 20 and y1-y0 > 20: im = im.crop((int(x0),int(y0),int(x1),int(y1)))
    return im

@torch.no_grad()
def classify(im):
    msgs = [{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().upper()
    for L in LABELS:                       # match longest first
        pass
    for L in sorted(LABELS, key=len, reverse=True):
        if L in ans.replace(" ", "_"): return L
    return None

# ---- run + aggregate ----
def circ_mean_std(degs):
    r = np.deg2rad(degs); C,S = np.cos(r).mean(), np.sin(r).mean(); R = math.hypot(C,S)
    mean = math.degrees(math.atan2(S,C)) % 360
    std = math.degrees(math.sqrt(max(0,-2*math.log(max(R,1e-9)))))
    return mean, std, R

results = {}
for obj, dn, frames in tasks:
    rows = []
    for A, imgpath, bbox in frames:
        V = classify(crop(imgpath, bbox))
        if V: rows.append((A, V))
    if len(rows) < 2:
        results[obj] = {"n":len(rows), "status":"too_few"}; print(f"  {obj[:30]:30s} too few", flush=True); continue
    # phi under + and - sign conventions
    phi_p = [(A + 180 - OFFSET[V]) % 360 for A,V in rows]
    phi_m = [(A + 180 + OFFSET[V]) % 360 for A,V in rows]
    mp,sp,_ = circ_mean_std(phi_p); mm,sm,_ = circ_mean_std(phi_m)
    sign = "+" if sp <= sm else "-"
    mean, std = (mp,sp) if sign=="+" else (mm,sm)
    results[obj] = {"n":len(rows), "facing_world_deg":round(mean,1), "within_std_deg":round(std,1),
                    "sign":sign, "views":[f"{int(A)}:{V}" for A,V in rows]}
    flag = "OK" if std < 35 else "NOISY"
    print(f"  {obj[:30]:30s} facing≈{mean:5.0f}° std={std:4.0f}° [{sign}] {flag}  {results[obj]['views']}", flush=True)

# global summary: which sign won, how many confident
signs = [r["sign"] for r in results.values() if "sign" in r]
stds  = [r["within_std_deg"] for r in results.values() if "within_std_deg" in r]
print(f"\n=== summary ===")
print(f"assets labeled: {len([r for r in results.values() if 'facing_world_deg' in r])}/{len(tasks)}")
if signs: print(f"sign vote: + {signs.count('+')} / - {signs.count('-')}  (should be consistent if convention is real)")
if stds:  print(f"within-asset std: median={np.median(stds):.0f}°  confident(<35°)={sum(s<35 for s in stds)}/{len(stds)}")
print("INTERPRET: low within-asset std across DIFFERENT camera azimuths => VLM views + geometry are consistent => facing recoverable.")
os.makedirs("runs", exist_ok=True)
json.dump(results, open(args.out,"w"), indent=2)
print(f"wrote {args.out}")
