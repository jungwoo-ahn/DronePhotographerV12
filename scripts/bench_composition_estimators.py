"""Benchmark single-image estimators for Module 2's hard keys (subject_bearing, camera elevation)
against ground truth on our OWN renders. GT bearing = (front_az[obj] - cam_azimuth) % 360 from the
verified facing map; GT elevation = cam_to_obj_elevation_deg from scores. Pick the estimator with
evidence before building Module 2 around it.  venv: shared (torch/transformers) + HF cache."""
import argparse, json, math, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
from PIL import Image

from src.common.facing import front_azimuth, sector3, sector8
from src.goal_authoring import vocab

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--estimator", default="vlm", choices=["vlm"])
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--full-frame", action="store_true", help="pass the whole frame (keep scene context for elevation)")
args = ap.parse_args()

ROOT = "data/trajectories"

# ---- sample GT renders (object with a facing entry, subject clearly visible, varied azimuth) ----
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
random.seed(0); random.shuffle(dirs)
samples = []
for dn in dirs:
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if front_azimuth(obj) is None:
        continue
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: continue
    cand = []
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or not r.get("in_frame") or not (30 <= s["occupancy"] <= 92) or s["body_in_frame_ratio"] < 45:
                continue
            cand.append((r, s))
    if not cand: continue
    r, s = random.choice(cand)
    front = front_azimuth(obj)
    gt_bearing = (front - s["cam_to_obj_azimuth_deg"]) % 360
    samples.append({"img": os.path.join(ROOT, dn, r["path_rel"]), "bbox": r.get("bbox_xyxy_full"),
                    "gt_bearing": gt_bearing, "gt_elev": s["cam_to_obj_elevation_deg"], "obj": obj})
    if len(samples) >= args.n: break
print(f"GT samples: {len(samples)} (distinct objects, subject-visible)", flush=True)

def crop(ip, bb, pad=0.15, size=384):
    im = Image.open(ip).convert("RGB"); W, H = im.size
    if bb:
        x0,y0,x1,y1 = bb; bw,bh = x1-x0, y1-y0
        x0=max(0,x0-bw*pad); x1=min(W,x1+bw*pad); y0=max(0,y0-bh*pad); y1=min(H,y1+bh*pad)
        if x1-x0>20 and y1-y0>20: im = im.crop((int(x0),int(y0),int(x1),int(y1)))
    im.thumbnail((size,size)); return im

# ---- VLM estimator: single image -> (bearing sector8, elevation band) ----
import re, torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
print("loading", args.model, flush=True)
proc = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

PROMPT = ('This is a rendered 3D human. Two questions.\n'
          '1) VIEW: which side of the person faces the camera? Choose one: front, front-right, right, '
          'back-right, back, back-left, left, front-left.\n'
          '2) CAMERA HEIGHT relative to the person: high (camera above, looking down), eye (eye level), '
          'or low (camera below, looking up).\n'
          'Answer: "VIEW=<label> HEIGHT=<high|eye|low>".')

_VIEW_ORDER = ("front-right","front-left","back-right","back-left","front","back","right","left")  # longest first

@torch.no_grad()
def vlm_estimate(im, debug=False):
    msgs=[{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":PROMPT}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inp=proc(text=[text],images=[im],return_tensors="pt").to("cuda")
    out=model.generate(**inp,max_new_tokens=40,do_sample=False)
    ans=proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].lower()
    if debug: print("   RAW:", ans.replace("\n"," ")[:120], flush=True)
    v=next((w for w in _VIEW_ORDER if w in ans.replace("_","-")), None)
    if "above" in ans or "looking down" in ans or "high" in ans: h="high"
    elif "below" in ans or "looking up" in ans or "low" in ans: h="low"
    elif "eye" in ans: h="eye"
    else: h=None
    return v, h

def cdiff(a,b): return abs(((a-b+180)%360)-180)
ELEV_BAND = {"high":"high angle","eye":"eye level","low":"low angle"}

bearing_err=[]; s3_ok=0; s8_ok=0; b_n=0
elev_ok=0; e_n=0
for i,smp in enumerate(samples):
    im=crop(smp["img"], None if args.full_frame else smp["bbox"], pad=0.15, size=512 if args.full_frame else 384)
    v,h=vlm_estimate(im, debug=(i<3))
    if v is not None:
        b_n+=1; pred=vocab.bearing_centroid(v)
        bearing_err.append(cdiff(pred,smp["gt_bearing"]))
        s3_ok+= (sector3(pred)==sector3(smp["gt_bearing"]))
        s8_ok+= (sector8(pred)==sector8(smp["gt_bearing"]))
    if h is not None:
        e_n+=1
        elev_ok+= (ELEV_BAND[h]==vocab._classify(smp["gt_elev"],vocab.ELEVATION))
    if (i+1)%30==0: print(f"  {i+1}/{len(samples)}",flush=True)

print("\n===== RESULTS (single-image VLM estimator vs GT) =====")
if b_n:
    print(f"BEARING (n={b_n}): sector3(front/side/back) acc={100*s3_ok/b_n:.0f}%  "
          f"sector8 acc={100*s8_ok/b_n:.0f}%  circular MAE={np.mean(bearing_err):.0f}°  "
          f"(chance: s3~33%, s8~12%)")
if e_n:
    print(f"ELEVATION (n={e_n}): band(high/eye/low) acc={100*elev_ok/e_n:.0f}%  (chance ~33%)")
print("INTERPRET: high bearing acc => VLM enough; low => need a pose/head-orientation model. "
      "Elevation band acc tells whether VLM suffices for the camera-height key.")
