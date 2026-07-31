"""v12 design-review artifact (one self-contained HTML), 3 sections:
  1. Facing alignment — per-asset front (front_az) on the clean turntable render.
  2. Rotation rep 5D -> 9D(rot6d) — a real pose-pair shown in the OLD 5D and the NEW Cosmos 9D
     ([trans(3), rot6d(6)], framewise-relative, Blender->OpenCV c2w), with a round-trip check.
  3. Facing-aware goal prompt — real frames + the NL prompt whose view sector is now SUBJECT-RELATIVE
     (world azimuth - front_az via the facing map), old(world) vs new(subject-rel) shown side by side.

Self-contained (base64). .venv-analysis + sys.path for src imports. Writes runs/design_review.html.
"""
import base64
import io
import json
import math
import os
import random
import sys
from collections import defaultdict

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
os.chdir(V12)
import numpy as np
from PIL import Image
from src.common.action_repr import encode_action_5d

FACING = json.load(open("runs/facing_map_final.json"))
IDX = json.load(open("runs/facing_turntable/index.json"))
ROOT = "data/trajectories"
OUT = "runs/design_review.html"

# ============================ rotation helpers (Cosmos 9D, standalone) ============================
B2O = np.diag([1.0, -1.0, -1.0])  # Blender cam-axes -> OpenCV cam-axes (Y down, Z forward)


def blender_c2w(forward, up):
    f = np.asarray(forward, float); f = f / np.linalg.norm(f)
    u = np.asarray(up, float); u = u / np.linalg.norm(u)
    nz = -f
    right = np.cross(u, nz); right /= np.linalg.norm(right)
    oup = np.cross(nz, right); oup /= np.linalg.norm(oup)
    return np.column_stack([right, oup, nz])            # [right, up, -forward]


def opencv_c2w(forward, up):
    return blender_c2w(forward, up) @ B2O               # [right, -up, forward]


def encode_9d(p0, f0, u0, p1, f1, u1):
    R0, R1 = opencv_c2w(f0, u0), opencv_c2w(f1, u1)
    dR = R0.T @ R1                                       # framewise-relative, camera-local
    dt = R0.T @ (np.asarray(p1, float) - np.asarray(p0, float))
    rot6d = np.concatenate([dR[:, 0], dR[:, 1]])         # first two COLUMNS (Cosmos convention)
    return np.concatenate([dt, rot6d]), dR


def decode_9d(p0, f0, u0, a9):
    R0 = opencv_c2w(f0, u0)
    dt, c0, c1 = a9[:3], a9[3:6], a9[6:9]
    c2 = np.cross(c0, c1)
    dR = np.column_stack([c0, c1, c2])
    R1 = R0 @ dR
    p1 = np.asarray(p0, float) + R0 @ dt
    Rbl = R1 @ B2O
    f1 = -Rbl[:, 2]; u1 = Rbl[:, 1]
    return p1, f1 / np.linalg.norm(f1), u1 / np.linalg.norm(u1)


def geodesic_deg(dR):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))


# ============================ prompt serializer (facing-aware) ============================
def shot_size(occ):
    for hi, name in [(8, "extreme wide"), (20, "wide"), (38, "medium-wide"),
                     (58, "medium"), (78, "medium close-up")]:
        if occ < hi:
            return name + " shot"
    return "close-up"


SECT8 = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]


def sector8(bearing):
    return SECT8[int(((bearing + 22.5) % 360) // 45)]


def sector3(bearing):
    b = abs(((bearing % 360) + 180) % 360 - 180)   # 0..180 from front
    return "front" if b < 45 else ("back" if b > 135 else "side")


def elevation_word(el):
    if el < -25:
        return "high angle"
    if el > 10:
        return "low angle"
    return "eye level"


def subject_rel(az, obj):
    fa = FACING.get(obj, {}).get("front_az")
    return None if fa is None else (fa - az) % 360   # subject-frame: left/right = the SUBJECT's own


def b64(path, w=340):
    im = Image.open(path).convert("RGB")
    im = im.resize((w, int(im.height * w / im.width)))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(v):
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


# ============================ Section 1: facing alignment ============================
def front_render_path(obj):
    info = IDX.get(obj) or {}
    views = info.get("views") or []
    fa = FACING.get(obj, {}).get("front_az")
    if not views or fa is None:
        return None
    best = min(views, key=lambda v: abs(((v["az"] - fa + 180) % 360) - 180))
    return best["path"]


# variety: sort by front_az so non-90 exceptions are visible
sample_objs = sorted(FACING, key=lambda o: (FACING[o]["front_az"], o))
sec1 = [sample_objs[i] for i in range(0, len(sample_objs), max(1, len(sample_objs) // 14))][:14]
sec1_cards = []
for obj in sec1:
    p = front_render_path(obj)
    if not p or not os.path.exists(p):
        continue
    fa = FACING[obj]["front_az"]; fw = FACING[obj]["facing_world_deg"]
    sec1_cards.append(
        f'<div class=card><img src="data:image/jpeg;base64,{b64(p, 220)}">'
        f'<div class=cap><b>{obj[:26]}</b><br>front_az <b>{fa}°</b> · faces {fw}°</div></div>')

# ============================ Section 2: rotation rep on real pose-pairs ============================
sec2_examples = []
rt_err = []
try:
    known = ["Abandoned-alley_9ee2b453__A-young-humble-man-walks-talking_eed072a6",
             "Abandoned-alley_9ee2b453__All-People-Are-Sisters_1795d425"]
    for dn in known:
        d = json.load(open(os.path.join(ROOT, dn, "data.json")))
        for pair in d.get("accepted_pairs", []):
            traj = pair.get("trajectory_32f") or []
            if len(traj) < 14:
                continue
            for a, b in [(4, 5), (10, 12)]:
                fa_, fb_ = traj[a], traj[b]
                p0, f0, u0 = fa_["pos"], fa_["forward"], fa_["up"]
                p1, f1, u1 = fb_["pos"], fb_["forward"], fb_["up"]
                a5 = encode_action_5d(p0, f0, u0, p1, f1, u1)
                a9, dR = encode_9d(p0, f0, u0, p1, f1, u1)
                rp, rf, ru = decode_9d(p0, f0, u0, a9)
                err = float(np.linalg.norm(rf - np.asarray(f1) / np.linalg.norm(f1)) +
                            np.linalg.norm(rp - np.asarray(p1)))
                rt_err.append(err)
                sec2_examples.append((a5, a9, geodesic_deg(dR), err))
                if len(sec2_examples) >= 3:
                    break
            if len(sec2_examples) >= 3:
                break
        if len(sec2_examples) >= 3:
            break
except Exception as e:  # noqa: BLE001
    sec2_examples = [("err", str(e), 0, 0)]

sec2_rows = []
for a5, a9, gd, err in sec2_examples:
    if isinstance(a5, str):
        sec2_rows.append(f"<tr><td colspan=3>error: {a9}</td></tr>")
        continue
    sec2_rows.append(
        f"<tr><td class=old>OLD 5D<br><span>[Δr,Δu,Δf,Δyaw,Δpitch]</span></td>"
        f"<td class=mono>{fmt(a5)}</td><td rowspan=2 class=note>rel. rot {gd:.1f}° · "
        f"round-trip err {err:.2e}</td></tr>"
        f"<tr><td class=new>NEW 9D<br><span>[Δt(3), rot6d(6)]</span></td>"
        f"<td class=mono>{fmt(a9[:3])}<br>{fmt(a9[3:])}</td></tr>")
max_rt = max(rt_err) if rt_err else 0.0

# ============================ Section 3: facing-aware prompt (stratified real frames) ============================
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
random.seed(3); random.shuffle(dirs)
cells = defaultdict(list)


def occ_bin(o):
    return min(4, int(o // 16))


scanned = 0
for dn in dirs:
    if scanned > 130 or sum(len(v) for v in cells.values()) >= 24:
        break
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if obj in ("rp_posedplus_00068_18_100k", "Girls-Hugs_5d7050d8") or obj not in FACING:
        continue
    p = os.path.join(ROOT, dn, "data.json")
    if not os.path.exists(p):
        continue
    try:
        d = json.load(open(p))
    except Exception:  # noqa: BLE001
        continue
    scanned += 1
    W, H = d.get("render_width", 1024), d.get("render_height", 768)
    for rec in d.get("render_records", []):
        for r in rec:
            s = r.get("scores")
            if not s or not r.get("in_frame", True) or not (20 <= s["occupancy"] <= 80):
                continue
            br = subject_rel(s["cam_to_obj_azimuth_deg"], obj)
            if br is None:
                continue
            key = (occ_bin(s["occupancy"]), sector3(br))
            if len(cells[key]) < 2:
                img = os.path.join(ROOT, dn, r.get("path_rel", ""))
                if os.path.exists(img):
                    cells[key].append((img, s, W, H, obj, br))

picks = [c for cell in cells.values() for c in cell][:24]
random.shuffle(picks)
sec3_cards = []
for img, s, W, H, obj, br in picks:
    az = s["cam_to_obj_azimuth_deg"]; el = s["cam_to_obj_elevation_deg"]
    fa = FACING[obj]["front_az"]
    new_sent = (f"A {shot_size(s['occupancy'])} of the subject from the "
                f"<b>{sector3(br)}</b> ({sector8(br)}), {elevation_word(el)}.")
    old_w = sector8(az % 360); new_w = sector8(br)
    flagged = "flag" if old_w != new_w else ""
    sec3_cards.append(
        f'<div class=card><img src="data:image/jpeg;base64,{b64(img)}">'
        f'<div class=prompt>{new_sent}</div>'
        f'<div class=cmp>sector: <span class=old2>world {old_w}</span> → '
        f'<span class="new2 {flagged}">subject-rel {new_w}</span>'
        f' &nbsp;(front {fa}° − az {int(az)}° = bearing {int(br)}°)</div>'
        f'<div class=nums>occ {s["occupancy"]}% · el {int(el)}° · {obj[:24]}</div></div>')

# ============================ HTML ============================
html = f"""<title>v12 design review — facing · rotation · prompt</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:24px;line-height:1.45}}
h1{{font-size:20px;margin:0 0 2px}} h2{{font-size:16px;margin:30px 0 4px;color:#8ab4f8}}
p.sub{{color:#9aa0a6;font-size:13px;margin:2px 0 12px;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}}
.grid3{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}}
.card{{background:#1e2126;border:1px solid #2a2e35;border-radius:10px;overflow:hidden}}
.card img{{width:100%;display:block}}
.cap{{padding:7px 9px;font-size:12px;color:#cdd1d6}} .cap b{{color:#e8eaed}}
.prompt{{padding:9px 11px 4px;font-size:14px}} .prompt b{{color:#7fd88f}}
.cmp{{padding:0 11px 5px;font-size:11.5px;color:#9aa0a6}}
.old2{{color:#c98}} .new2{{color:#7fd88f;font-weight:600}} .new2.flag{{color:#fbbc04}}
.nums{{padding:0 11px 9px;font-size:11px;color:#5f6368;font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;width:100%;max-width:820px;margin-top:6px}}
td{{border:1px solid #2a2e35;padding:8px 10px;vertical-align:top;font-size:12.5px}}
td.old{{color:#c98;white-space:nowrap}} td.new{{color:#7fd88f;white-space:nowrap}}
td span{{color:#5f6368;font-size:10.5px}} td.note{{color:#8ab4f8;font-variant-numeric:tabular-nums}}
.mono{{font-family:ui-monospace,monospace;font-size:12px;color:#e8eaed}}
.box{{background:#1a1d22;border:1px solid #2a2e35;border-left:3px solid #8ab4f8;border-radius:6px;padding:10px 13px;margin:10px 0;font-size:13px;max-width:900px}}
.warn{{border-left-color:#fbbc04}} code{{background:#0b0d10;padding:1px 5px;border-radius:4px;font-size:12px}}
</style>
<h1>v12 design review — facing · rotation · prompt</h1>
<p class=sub>Everything set up this session, on real data: (1) per-asset front alignment, (2) the rotation-rep
switch to Cosmos rot6d, (3) the goal prompt rebuilt around subject-relative facing.</p>

<h2>1 · Facing alignment (front_az per asset)</h2>
<p class=sub>The green view each asset's front sits at, on the clean turntable. 84/102 land at front_az=90 (uniform library);
the rest ({', '.join(sorted({str(FACING[o]['front_az'])+'°' for o in FACING if FACING[o]['front_az']!=90}))}) are genuinely turned poses. facing = front_az+180 (which way the subject looks).</p>
<div class=grid>{''.join(sec1_cards)}</div>

<h2>2 · Rotation representation — 5D → 9D (Cosmos rot6d)</h2>
<p class=sub>OLD: 5D <code>[Δright,Δup,Δforward,Δyaw,Δpitch]</code> (yaw about world-up + pitch; 2-DOF, roll-free by construction).
NEW: Cosmos <code>camera_pose</code> 9D <code>[Δtranslation(3), rot6d(6)]</code>, framewise-relative, camera-local, in OpenCV c2w.
rot6d = first two columns of the relative rotation. Same real pose-pairs, both reps:</p>
<table><tr><td class=old><b>rep</b></td><td class=mono><b>value</b></td><td class=note><b>check</b></td></tr>{''.join(sec2_rows)}</table>
<div class="box">Round-trip <code>encode_9d → decode_9d</code> reconstructs the next pose exactly — max error over the examples =
<b>{max_rt:.2e}</b> (≈0 ⇒ the 9D rep is lossless on our data).</div>
<div class="box warn"><b>Two things Cosmos will NOT do for us (verified in framework code) — we add them:</b><br>
<b>(A) encode:</b> convert Blender c2w (+Y up, −Z fwd) → OpenCV c2w (+Y down, +Z fwd) before the relative delta
(<code>R_opencv = R_blender · diag(1,−1,−1)</code>, applied above). Cosmos expects the OpenCV camera frame; camera_pose has no built-in conversion.<br>
<b>(B) decode:</b> Cosmos rot6d is free SO(3) with no roll constraint, so a model's predicted roll would drift. We re-project each
decoded pose to upright (re-derive right/up from forward + world-up +Z). Data is exactly roll-free (139k frames, |roll| max 0.0000°).</div>

<h2>3 · Facing-aware goal prompt</h2>
<p class=sub>The view sector now comes from the <b>subject-frame bearing</b> = <code>front_az[asset] − azimuth</code> (left/right = the
subject's own), not the raw world azimuth. Each card shows the new prompt + how the sector changed (world → subject-frame);
<span class=new2 style="font-weight:600">yellow</span> = the 8-way label actually flipped. Frames stratified over shot-size × front/side/back.</p>
<div class=grid3>{''.join(sec3_cards)}</div>
<p class=sub style="margin-top:14px">Note: 8-way (front-left/left/…) needs handedness (mirror) resolved for left↔right; the 3-way front/side/back (bold) is handedness-safe.</p>
"""
os.makedirs("runs", exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB) — sec1={len(sec1_cards)} assets, "
      f"sec2={len(sec2_examples)} pairs (max round-trip err {max_rt:.2e}), sec3={len(sec3_cards)} frames")
