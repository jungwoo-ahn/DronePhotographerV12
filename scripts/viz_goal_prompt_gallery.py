

# Category words come from src/goal_authoring/vocab.py, the single source of truth. The local
# copies these replaced had drifted: elevation cut at -25/+10 instead of -20/+15, the label
# "medium close-up shot" instead of "medium close-up", and an invented "mostly out of frame"
# band. The same goal therefore read differently depending on which script rendered it.
from src.goal_authoring import vocab
from src.goal_authoring.vocab import _classify


def shot_size(occ):
    return _classify(float(occ), vocab.SHOT_SIZE)


def elevation_word(el):
    return _classify(float(el), vocab.ELEVATION)


def body_word(b):
    return _classify(float(b), vocab.BODY_FRAMING)

"""Validate the DRAFT cinematography goal-descriptor against real renders.
For a shot-size x angle STRATIFIED sample of frames, show: rendered image + the draft NL prompt
(cinematography words + concrete numbers) + the raw 8-key profile. Lets us eyeball whether the
serialized goal actually matches the picture before committing. Self-contained; reads data symlink.

DRAFT serializer = the thing under review (edit + re-run to iterate the descriptor).
"""
import os, sys, json, random, base64, io
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ROOT = DEFAULT_TRAJ_ROOT
OUT = "runs/goal_prompt_gallery.html"

# ---------- DRAFT cinematography serializer (profile -> words + numbers) ----------

def azimuth_sector(az):  # 0..360, camera bearing around subject (convention TBD — image validates)
    az %= 360
    secs = ["front","front-right","right","back-right","back","back-left","left","front-left"]
    return secs[int(((az+22.5)%360)//45)]


def placement_x(x, W):
    if x < 0 or x > W: return "off-screen " + ("left" if x < 0 else "right")
    t = x / W
    return "left third" if t < 0.38 else ("right third" if t > 0.62 else "centered")

def placement_y(y, H):
    if y < 0 or y > H: return "off-screen " + ("top" if y < 0 else "bottom")
    t = y / H
    return "upper" if t < 0.38 else ("lower" if t > 0.62 else "mid")


def serialize(s, W, H):
    occ, body = s["occupancy"], s["body_in_frame_ratio"]
    az, el = s["cam_to_obj_azimuth_deg"], s["cam_to_obj_elevation_deg"]
    cx, cy = s["object_center_x"], s["object_center_y"]
    sent = (f"A {shot_size(occ)} of the subject, viewed from the {azimuth_sector(az)} "
            f"at {elevation_word(el)}; subject {placement_x(cx,W)}, {placement_y(cy,H)} of frame; "
            f"{body_word(body)}.")
    nums = (f"occupancy {occ}% · azimuth {az}° · elevation {el}° · "
            f"center ({cx},{cy})/{W}x{H} · body_in_frame {body}%")
    return sent, nums

# ---------- stratified sample ----------
dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
random.seed(1); random.shuffle(dirs)
# bins: 6 shot-size (occupancy) x 4 azimuth quadrants ; aim ~2 per cell -> ~48, cap 42
from collections import defaultdict
cells = defaultdict(list)
def occ_bin(o): return min(5, int(o//17))
def az_bin(a): return int((a%360)//90)
scanned = 0
for dn in dirs:
    if scanned > 220: break
    p = os.path.join(ROOT, dn, "data.json")
    if not os.path.exists(p): continue
    try: d = json.load(open(p))
    except Exception: continue
    scanned += 1
    W, H = d.get("render_width",1024), d.get("render_height",768)
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores");
            if not s or not r.get("in_frame", True): continue
            img = os.path.join(ROOT, dn, r.get("path_rel",""))
            if not os.path.exists(img): continue
            key = (occ_bin(s["occupancy"]), az_bin(s["cam_to_obj_azimuth_deg"]))
            if len(cells[key]) < 2:
                cells[key].append((img, s, W, H, dn))
picks = [c for cell in cells.values() for c in cell][:42]
random.shuffle(picks)
print(f"scanned {scanned} placements; gallery frames: {len(picks)} across {len(cells)} shot-size×azimuth cells")

# ---------- HTML ----------
def b64(img, w=360):
    im = Image.open(img).convert("RGB")
    im = im.resize((w, int(im.height*w/im.width)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()

cards = []
for img, s, W, H, dn in picks:
    sent, nums = serialize(s, W, H)
    cards.append(f"""<div class=card>
<img src="data:image/jpeg;base64,{b64(img)}">
<div class=prompt>{sent}</div>
<div class=nums>{nums}</div>
<div class=src>{dn[:52]}</div></div>""")

html = f"""<title>v12 goal-prompt validation</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:24px}}
h1{{font-size:18px;font-weight:600}} p.sub{{color:#9aa0a6;font-size:13px;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;margin-top:18px}}
.card{{background:#1e2126;border:1px solid #2a2e35;border-radius:10px;overflow:hidden}}
.card img{{width:100%;display:block}}
.prompt{{padding:10px 12px;font-size:14px;line-height:1.4}}
.nums{{padding:0 12px 8px;font-size:12px;color:#8ab4f8;font-variant-numeric:tabular-nums}}
.src{{padding:0 12px 10px;font-size:11px;color:#5f6368}}
</style>
<h1>v12 goal-descriptor validation — does the prompt match the picture?</h1>
<p class=sub>Draft cinematography serializer (words + numbers) from the 8-key V5 profile, on a shot-size×azimuth stratified sample. Eyeball each: is the description faithful? Flag mismatches (esp. azimuth sector convention & elevation high/low sign).</p>
<div class=grid>{''.join(cards)}</div>"""
os.makedirs("runs", exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB)")
