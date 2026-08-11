

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

"""Facing-aware goal-prompt gallery for (almost) EVERY object — one representative scene frame per asset
(occupancy in [20,80], well-composed) + its prompt whose view sector is SUBJECT-RELATIVE
(azimuth - front_az via runs/facing_map_final.json). Lets us eyeball prompt quality across the whole
library. Self-contained HTML (base64). .venv-analysis (PIL). Writes runs/prompt_gallery_all.html.
"""
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict

os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

FACING = json.load(open("runs/facing_map_final.json"))
ROOT = DEFAULT_TRAJ_ROOT
OUT = "runs/prompt_gallery_all.html"
EXCLUDE = {"rp_posedplus_00068_18_100k",   # ~100x scale bug
           "Girls-Hugs_5d7050d8"}          # two people hugging -> ambiguous subject/facing



def sector8(b):
    return SECT8[int(((b + 22.5) % 360) // 45)]


def sector3(b):
    a = abs(((b % 360) + 180) % 360 - 180)
    return "front" if a < 45 else ("back" if a > 135 else "side")



def placement_x(x, W):
    if x < 0 or x > W:
        return "off-screen " + ("left" if x < 0 else "right")
    t = x / W
    return "left third" if t < 0.38 else ("right third" if t > 0.62 else "centered")


def b64(path, w=300):
    im = Image.open(path).convert("RGB")
    im = im.resize((w, int(im.height * w / im.width)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=76)
    return base64.b64encode(buf.getvalue()).decode()


# object -> all its placement dirs
t0 = time.time()
dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
by_obj = defaultdict(list)
for d in dirs:
    obj = d.split("__", 1)[1] if "__" in d else d
    if obj in FACING and obj not in EXCLUDE:
        by_obj[obj].append(d)
print(f"objects={len(by_obj)}  (listdir {time.time()-t0:.0f}s)", flush=True)


def best_frame(dn):
    """Best-composed frame in a placement: in-frame, occ in [20,80], occ closest to 50."""
    p = os.path.join(ROOT, dn, "data.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:  # noqa: BLE001
        return None
    W, H = d.get("render_width", 1024), d.get("render_height", 768)
    best = None
    for rec in d.get("render_records", []):
        for r in rec:
            s = r.get("scores")
            if not s or not r.get("in_frame", True) or not (20 <= s["occupancy"] <= 80):
                continue
            score = abs(s["occupancy"] - 50)
            if best is None or score < best[0]:
                img = os.path.join(ROOT, dn, r.get("path_rel", ""))
                if os.path.exists(img):
                    best = (score, img, s, W, H)
    return best


cards = []
missed = []
for i, (obj, placements) in enumerate(sorted(by_obj.items())):
    fr = None
    for dn in placements[:5]:              # try up to 5 placements to find a good frame
        fr = best_frame(dn)
        if fr:
            break
    if not fr:
        missed.append(obj)
        continue
    _, img, s, W, H = fr
    az, el = s["cam_to_obj_azimuth_deg"], s["cam_to_obj_elevation_deg"]
    fa = FACING[obj]["front_az"]
    br = (fa - az) % 360   # subject-frame bearing (front_az - az): left/right = the SUBJECT's own
    sent = (f"A {shot_size(s['occupancy'])} of the subject from the subject's <b>{sector3(br)}</b> "
            f"({sector8(br)}), {elevation_word(el)}; subject {placement_x(s['object_center_x'], W)}.")
    cards.append(
        f'<div class=card><img loading=lazy src="data:image/jpeg;base64,{b64(img)}">'
        f'<div class=prompt>{sent}</div>'
        f'<div class=meta>bearing {int(br)}° = front {fa}° − az {int(az)}° · occ {s["occupancy"]}% · el {int(el)}°</div>'
        f'<div class=nm>{obj[:34]}</div></div>')
    if (i + 1) % 20 == 0:
        print(f"...{i+1}/{len(by_obj)} objs, {len(cards)} cards, {time.time()-t0:.0f}s", flush=True)

html = f"""<title>v12 goal prompt — all objects</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:22px;line-height:1.4}}
h1{{font-size:18px;margin:0 0 3px}} p.sub{{color:#9aa0a6;font-size:13px;margin:0 0 14px;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:#1e2126;border:1px solid #2a2e35;border-radius:10px;overflow:hidden}}
.card img{{width:100%;display:block}}
.prompt{{padding:9px 11px 3px;font-size:13.5px}} .prompt b{{color:#7fd88f}}
.meta{{padding:0 11px 4px;font-size:11px;color:#8ab4f8;font-variant-numeric:tabular-nums}}
.nm{{padding:0 11px 9px;font-size:11px;color:#5f6368}}
</style>
<h1>v12 goal prompt — (almost) every object · facing-aware</h1>
<p class=sub>One representative scene frame per asset (occupancy 20–80, well-composed). View sector =
<b>subject-frame</b> bearing = front_az − azimuth (facing map) — left/right are the SUBJECT's own
(camera on the subject's right = right-side view). {len(cards)} objects{(' · missing: '+', '.join(m[:18] for m in missed)) if missed else ''}.
<br><span style="color:#5f6368">Handedness caveat: 8-way left/right assumes non-mirrored assets; the 3-way front/side/back (bold) is always safe.</span></p>
<div class=grid>{''.join(cards)}</div>"""
os.makedirs("runs", exist_ok=True)
open(OUT, "w").write(html)
print(f"\nwrote {OUT} ({os.path.getsize(OUT)//1024} KB) — {len(cards)} objects, {len(missed)} missed, {time.time()-t0:.0f}s", flush=True)
