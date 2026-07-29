"""Per-asset facing via MULTI-FRAME comparison (VLMs are far better at 'which of these is X' than at
absolute per-image classification). Show the VLM several views of one asset at known azimuths; ask
which is FRONT and which is BACK. front/back ~180 apart => confident; facing = front_az + 180.
Shared venv (torch/transformers). Writes runs/facing_map*.json."""
import os, sys, json, math, random, argparse
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--max-assets", type=int, default=8)
ap.add_argument("--views", type=int, default=8)         # frames shown per asset (spread over azimuth)
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--out", default="runs/facing_map.json")
args = ap.parse_args()

ROOT = "data/trajectories"
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
by_obj = {}
for d in sorted(dirs):
    by_obj.setdefault(d.split("__",1)[1] if "__" in d else d, d)
random.seed(0)
objs = random.sample(list(by_obj.items()), min(args.max_assets, len(by_obj)))

def pick_views(dn, k):
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: return []
    cand = []
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or not r.get("in_frame"): continue
            if not (30 <= s["occupancy"] <= 90 and s["body_in_frame_ratio"] >= 45): continue
            cand.append((s["cam_to_obj_azimuth_deg"], os.path.join(ROOT, dn, r["path_rel"]), r.get("bbox_xyxy_full")))
    if len(cand) < 3: return []
    # one frame per azimuth sector, spread around the circle
    buckets = {}
    for A, ip, bb in cand:
        b = int((A % 360) // (360/k))
        if b not in buckets: buckets[b] = (A, ip, bb)
    return list(buckets.values())

def crop(ip, bb, pad=0.3, size=256):
    im = Image.open(ip).convert("RGB"); W,H = im.size
    if bb:
        x0,y0,x1,y1 = bb; bw,bh = x1-x0, y1-y0
        x0=max(0,x0-bw*pad); x1=min(W,x1+bw*pad); y0=max(0,y0-bh*pad); y1=min(H,y1+bh*pad)
        if x1-x0>20 and y1-y0>20: im = im.crop((int(x0),int(y0),int(x1),int(y1)))
    im.thumbnail((size,size)); return im

tasks = [(o,dn,pick_views(dn,args.views)) for o,dn in objs]
tasks = [(o,dn,v) for o,dn,v in tasks if len(v)>=3]
print(f"assets usable: {len(tasks)}/{len(objs)}", flush=True)

import torch, re
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
print("loading", args.model, flush=True)
proc = AutoProcessor.from_pretrained(args.model)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

@torch.no_grad()
def ask_front_back(imgs):
    n = len(imgs)
    content = []
    for i, im in enumerate(imgs, 1):
        content += [{"type":"text","text":f"View {i}:"}, {"type":"image","image":im}]
    content.append({"type":"text","text":
        f"These are {n} views of the SAME 3D human character from different camera angles. "
        f"Identify which view shows the person's FRONT (face and chest toward the camera) and which "
        f"shows the BACK (back of the head/body toward the camera). "
        f"Answer EXACTLY as: FRONT=<n> BACK=<n>"})
    msgs = [{"role":"user","content":content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=16, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    f = re.search(r"FRONT\s*=\s*<?(\d+)", ans, re.I); b = re.search(r"BACK\s*=\s*<?(\d+)", ans, re.I)
    fi = int(f.group(1))-1 if f else None; bi = int(b.group(1))-1 if b else None
    return fi, bi, ans.strip().replace("\n"," ")

def cdiff(a,b): return abs(((a-b+180)%360)-180)

results = {}
for obj, dn, views in tasks:
    azs = [A for A,_,_ in views]
    imgs = [crop(ip,bb) for _,ip,bb in views]
    fi, bi, raw = ask_front_back(imgs)
    if fi is None or not (0<=fi<len(views)):
        results[obj] = {"status":"parse_fail","raw":raw}; print(f"  {obj[:28]:28s} parse_fail: {raw}", flush=True); continue
    front_az = azs[fi]; facing = (front_az + 180) % 360
    back_az = azs[bi] if (bi is not None and 0<=bi<len(views)) else None
    opp = cdiff(front_az, back_az) if back_az is not None else None   # should be ~180
    conf = "OK" if (opp is not None and abs(opp-180)<45) else "CHECK"
    results[obj] = {"n_views":len(views), "front_az":round(front_az), "back_az":(round(back_az) if back_az else None),
                    "facing_world_deg":round(facing), "front_back_sep":(round(opp) if opp else None), "conf":conf,
                    "azimuths":[round(a) for a in azs]}
    print(f"  {obj[:28]:28s} front_az={front_az:4.0f} back_az={str(back_az):>4} sep={str(round(opp) if opp else None):>4} facing≈{facing:4.0f}° {conf}", flush=True)

labeled = [r for r in results.values() if "facing_world_deg" in r]
ok = [r for r in labeled if r["conf"]=="OK"]
print(f"\n=== summary ===")
print(f"labeled {len(labeled)}/{len(tasks)}; front/back ~180deg-consistent (conf OK): {len(ok)}/{len(labeled)}")
print("INTERPRET: high conf-OK rate => multi-frame front/back picking is reliable => scale to all 102.")
os.makedirs("runs", exist_ok=True); json.dump(results, open(args.out,"w"), indent=2)
print(f"wrote {args.out}")
