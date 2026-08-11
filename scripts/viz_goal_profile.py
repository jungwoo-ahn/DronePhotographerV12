"""Goal-profile inspector — see a goal profile as a CAMERA IN SPACE and as a FRAME.

The goal reaches Cosmos only as a sentence, and a sentence is hard to check. But the
profile is not vague: the 8 keys plus the crop fractions pin the camera down to 5 geometric
DOF, so a goal can be *drawn* rather than read. This builds one self-contained HTML page
that draws it three ways at once, live:

  1. 3D  — the subject at the origin of its OWN frame (+X = where it looks), the camera on
           the viewing sphere at (bearing, elevation, distance), with its frustum and its
           optical axis. The bearing ring is labelled with the eight `sector8` words, so a
           goal's view sector is something you see rather than compute.
  2. Frame — the 1024x768 frame with the visible bbox, the FULL unclipped projection
           (dashed, the part the crop fractions describe), the crop bands, thirds grid.
  3. Text — the exact training prompt from `src.data.lerobot_export.goal_prompt`, plus the
           `vocab` category word for every axis.

Every number and word comes from the repo's own modules — the vocabulary tables, ranges and
intrinsics are exported into the page as data, not retyped, because hand-copied vocab tables
have drifted here before (see docs/v4_session_changes.md section 5). The prompt is computed
BOTH in Python (embedded) and in the page's JS, and the page shows a PROMPT PARITY badge if
the two ever disagree.

WHY the distance is recoverable: `bbox_y_offset` is the subject's half-height in pixels under
the VISIBLE-bbox convention, and `visible_frac` says what share of the subject that is, so the
full projected half-height is `bbox_y_offset / visible_frac` and the range follows from the
render intrinsics: r = (subject_height/2) * focal_px_y / half_height_full. Distances are in
SUBJECT HEIGHTS, which keeps the picture scene-agnostic exactly like the profile itself.

Usage
-----
    python scripts/viz_goal_profile.py                      # inspector only (no data needed)
    python scripts/viz_goal_profile.py --sample 12          # + gallery of real dataset goals
    python scripts/viz_goal_profile.py --nl "low-angle close-up from the side"
    python scripts/viz_goal_profile.py --profile '{"occupancy": 45, "subject_bearing_deg": 300}'

With `--sample`, each card carries the real render AND the frame's true camera pose, so the
page can draw the reconstruction and the ground truth in the same 3D scene and print the
residual. That is the tool checking itself: if a goal profile really does determine the shot,
the two cameras coincide.

Needs numpy + Pillow only (system python3 is enough; `--sample` reads the data symlink).
Writes runs/goal_profile_viz.html.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
from pathlib import Path

V12 = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np  # noqa: E402

from src.common.annotations import iter_windows, is_goal_frame, list_annotation_files  # noqa: E402
from src.common.dataset_base import DEFAULT_TRAJ_ROOT  # noqa: E402
from src.common.facing import SECTOR8, load_facing_map  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS,
    DEFAULT_V5_RANGES,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    SUBJECT_BEARING_KEY,
    goal_vector,
)
from src.common.reward import CameraIntrinsics, pose_to_geometry  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402
from src.goal_authoring import vocab  # noqa: E402

OUT_DEFAULT = "runs/goal_profile_viz.html"
INTR = CameraIntrinsics.from_render(int(RENDER_WIDTH), int(RENDER_HEIGHT))


# --------------------------------------------------------------------------- #
# config exported to the page (vocab.py stays the single source of truth)
# --------------------------------------------------------------------------- #
def _table(t: dict[str, tuple[float, float, float]]) -> list[list]:
    """{label: (lo, hi, centroid)} -> [[label, lo, hi, centroid], ...], order preserved.

    `vocab._classify` walks the table in insertion order and returns the first band that
    contains the value, so the ORDER is part of the semantics — a dict would round-trip
    through JSON fine, but a list makes that explicit."""
    return [[k, lo, hi, c] for k, (lo, hi, c) in t.items()]


def page_config() -> dict:
    return {
        "render_w": RENDER_WIDTH,
        "render_h": RENDER_HEIGHT,
        "fx": INTR.focal_px_x,
        "fy": INTR.focal_px_y,
        "goal_keys": list(DEFAULT_GOAL_KEYS),
        "bearing_key": SUBJECT_BEARING_KEY,
        "ranges": {k: list(v) for k, v in DEFAULT_V5_RANGES.items()},
        "sector8": list(SECTOR8),
        "axis_key": dict(vocab.AXIS_KEY),
        "tables": {
            "SHOT_SIZE": _table(vocab.SHOT_SIZE),
            "BODY_FRAMING": _table(vocab.BODY_FRAMING),
            "ELEVATION": _table(vocab.ELEVATION),
            "PLACE_X": _table(vocab.PLACE_X),
            "PLACE_Y": _table(vocab.PLACE_Y),
        },
        "crop_side": {k: list(v) for k, v in vocab.CROP_SIDE.items()},
    }


# --------------------------------------------------------------------------- #
# presets — a small spread of authored goals, grounded through vocab centroids
# --------------------------------------------------------------------------- #
PRESET_CATEGORIES: list[tuple[str, dict[str, str]]] = [
    ("close-up, front, eye level", {
        "shot_size": "close-up", "bearing": "front", "elevation": "eye level",
        "placement_x": "centered", "placement_y": "mid", "body_framing": "tightly cropped"}),
    ("medium, front-right, low angle", {
        "shot_size": "medium shot", "bearing": "front-right", "elevation": "low angle",
        "placement_x": "centered", "placement_y": "mid", "body_framing": "partially cut off"}),
    ("wide, from behind, high angle", {
        "shot_size": "wide shot", "bearing": "back", "elevation": "high angle",
        "placement_x": "left third", "placement_y": "lower", "body_framing": "full body in frame"}),
    ("medium-wide, left profile, thirds", {
        "shot_size": "medium-wide shot", "bearing": "left", "elevation": "eye level",
        "placement_x": "right third", "placement_y": "upper", "body_framing": "mostly in frame"}),
    ("extreme wide, back-left, eye level", {
        "shot_size": "extreme wide shot", "bearing": "back-left", "elevation": "eye level",
        "placement_x": "centered", "placement_y": "mid", "body_framing": "full body in frame"}),
]


def preset_goals() -> list[dict]:
    out = []
    for name, cats in PRESET_CATEGORIES:
        vals, _spec = vocab.categories_to_profile(cats)
        g = {k: float(vals.get(k, 0.0)) for k in DEFAULT_GOAL_KEYS}
        half_h, half_w = _half_extents_from_occupancy(g["occupancy"])
        g, crop = _fit_authored_box(g, half_w, half_h)
        out.append({"name": name, "goal": g, "crop": crop})
    return out


def _half_extents_from_occupancy(occ: float) -> tuple[float, float]:
    """Half-height / half-width in px for an authored goal that names only a shot size.

    occupancy is a silhouette-area ratio; a bbox is not recoverable from it in general. This
    inverts the crude version — subject box area = occ% of the frame at aspect 2.4:1 — purely
    so an authored goal opens at a plausible distance. Real dataset goals never take this path.
    """
    area = max(1e-4, float(occ) / 100.0) * RENDER_WIDTH * RENDER_HEIGHT
    aspect = 2.4
    half_w = 0.5 * math.sqrt(area / aspect)
    return half_w * aspect, half_w


def _fit_authored_box(g: dict, half_w_full: float, half_h_full: float) -> tuple[dict, dict]:
    """Make an authored goal internally consistent under the VISIBLE-bbox convention.

    A category pair like ("close-up", "centered/mid") names where the subject sits and how big
    it is, but at occupancy 88 the subject does not FIT — so the box the goal keys describe is
    not the box the shot size implies. Place the full projection at the requested centre, cut it
    at the frame edges, and report the surviving box as `object_center_*` / `bbox_*_offset` with
    the cut recorded in the crop fractions. Otherwise a "close-up" preset would claim a 1290 px
    subject inside a 768 px frame with `uncropped` in its own prompt.
    """
    W, H = RENDER_WIDTH, RENDER_HEIGHT
    cy = float(g["object_center_y"])
    y0, y1 = cy - half_h_full, cy + half_h_full
    span = max(y1 - y0, 1e-6)
    crop = {"top": max(0.0, -y0) / span, "bot": max(0.0, y1 - H) / span}
    vy0, vy1 = max(y0, 0.0), min(y1, H)
    g = dict(g)
    g["object_center_y"] = 0.5 * (vy0 + vy1)
    g["bbox_y_offset"] = max(0.5 * (vy1 - vy0), 1.0)
    cx = float(g["object_center_x"])
    vx0, vx1 = max(cx - half_w_full, 0.0), min(cx + half_w_full, W)
    g["object_center_x"] = 0.5 * (vx0 + vx1)
    g["bbox_x_offset"] = max(0.5 * (vx1 - vx0), 1.0)
    return g, crop


# --------------------------------------------------------------------------- #
# real dataset goals
# --------------------------------------------------------------------------- #
def _b64_jpeg(path: str, width: int = 420, quality: int = 72) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _subject_frame_pose(view, bearing_deg: float) -> dict:
    """The frame's TRUE camera pose expressed in the subject frame the page draws.

    Subject frame: origin at `subject_center`, +X = the direction the subject looks, +Z = world
    up, one unit = one subject height. The world->subject rotation is a yaw by -theta where
    theta is the subject's world-frame look azimuth; that is recovered from the pose itself
    (`bearing = look_azimuth - camera_position_azimuth`, see src/common/facing.py) rather than
    from the facing map a second time, so the two can never disagree here.
    """
    cam = np.asarray(view.camera_position, dtype=np.float64)
    ctr = np.asarray(view.subject_center, dtype=np.float64)
    h = float(view.subject_height) or 1.0
    d = cam - ctr
    pos_az = math.degrees(math.atan2(d[1], d[0]))          # camera's azimuth seen from subject
    theta = math.radians(pos_az + float(bearing_deg))      # subject's world look azimuth
    c, s = math.cos(-theta), math.sin(-theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_subj(v, translate=False):
        v = np.asarray(v, dtype=np.float64)
        return (rz @ ((v - ctr) / h if translate else v)).tolist()

    return {
        "pos": to_subj(cam, translate=True),
        "forward": to_subj(view.camera_forward),
        "up": to_subj(view.camera_up),
        "r": float(np.linalg.norm(d) / h),
    }


def sample_real_goals(n: int, root: str, seed: int, max_placements: int = 1600) -> list[dict]:
    """Pick `n` frames that pass the real training goal gate, with their render + true pose.

    Balanced across the eight view sectors rather than taken in scan order. The pool is not
    uniform in bearing, so a plain random draw returns a gallery that is mostly one side of the
    subject — which is precisely the axis a viewer wants to see vary.
    """
    files = list_annotation_files([root])
    if not files:
        print(f"no data.json under {root} — skipping the gallery", file=sys.stderr)
        return []
    rng = random.Random(seed)
    rng.shuffle(files)
    facing = load_facing_map()
    per_sector = max(1, math.ceil(n / len(SECTOR8)))
    buckets: dict[str, list[dict]] = {s: [] for s in SECTOR8}
    total = lambda: sum(len(v) for v in buckets.values())

    for path in files[:max_placements]:
        if total() >= n:
            break
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        obj = doc.get("object") or doc.get("placement", "").split("__")[-1]
        if obj not in facing:            # no facing entry -> no bearing -> not a usable goal
            continue
        records = doc.get("render_records") or []
        # Enriched views (visible-bbox geometry + crop fractions) via the public iterator; a
        # chunk of 1 makes every frame the start of its own window, so this enumerates all 32.
        try:
            wins = list(iter_windows(path, chunk_size=1, stride=1))
        except (KeyError, OSError, ValueError):
            continue
        if not wins:
            continue
        views = [w.start for w in wins] + [wins[-1].end]
        rng.shuffle(views)
        for view in views:
            recs = records[view.pair_idx] if view.pair_idx < len(records) else []
            rec = next((r for r in recs if int(r.get("frame_idx", -1)) == view.frame_idx), None)
            if rec is None or not is_goal_frame(rec):
                continue
            vec = goal_vector(view.raw, DEFAULT_GOAL_KEYS, object_key=obj)
            if not np.isfinite(vec).all():
                continue
            g = {k: float(vec[i]) for i, k in enumerate(DEFAULT_GOAL_KEYS)}
            sector = sector8(g[SUBJECT_BEARING_KEY])
            if len(buckets[sector]) >= per_sector:
                continue                # sector already full — keep looking on this placement
            # `vis` rides along rather than being recomputed as 1 - top - bot in the page: the
            # identity holds exactly in real arithmetic, but the prompt prints it to 2 decimals
            # and a float difference in the last bit could flip that digit and raise a parity
            # alarm that means nothing.
            crop = {"top": float(view.raw.get("top_cut_frac", 0.0)),
                    "bot": float(view.raw.get("bot_cut_frac", 0.0)),
                    "vis": float(view.raw.get("visible_frac", 1.0))}
            img = _b64_jpeg(view.image)
            if img is None:
                continue
            buckets[sector].append({
                "name": view.object.replace("-", " "),
                "sub": f"{sector} · {vocab.crop_label(crop['top'], crop['bot'])}",
                "sector": sector,
                "goal": g,
                "crop": crop,
                "image": img,
                # Python's own prompt — the page recomputes it and flags any disagreement.
                "prompt": goal_prompt(vec, DEFAULT_GOAL_KEYS,
                                      crop={**crop, "top_cut_frac": crop["top"],
                                            "bot_cut_frac": crop["bot"],
                                            "visible_frac": view.raw.get("visible_frac")}),
                "truth": _subject_frame_pose(view, g[SUBJECT_BEARING_KEY]),
                "pose_geom": {k: math.degrees(v) for k, v in pose_to_geometry(
                    view.camera_position, view.camera_forward, view.camera_up,
                    view.subject_center, view.subject_height).items()},
            })
            break                       # one frame per placement keeps the gallery diverse

    out = [e for s in SECTOR8 for e in buckets[s]]
    # Print the mix, the way the export does: a gallery that quietly collapsed onto one side
    # would otherwise look like a deliberate sample.
    crops: dict[str, int] = {}
    for e in out:
        k = vocab.crop_label(e["crop"]["top"], e["crop"]["bot"])
        crops[k] = crops.get(k, 0) + 1
    print("  sectors: " + ", ".join(f"{s} {len(buckets[s])}" for s in SECTOR8))
    print("  crops:   " + ", ".join(f"{k} {v}" for k, v in sorted(crops.items())))
    return out


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def build_html(cfg: dict, goals: list[dict], initial: dict) -> str:
    payload = json.dumps({"cfg": cfg, "goals": goals, "initial": initial}, allow_nan=False)
    return HTML_TEMPLATE.replace("/*__PAYLOAD__*/", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goal profile inspector — DronePhotographer v12</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2b313d; --ink:#e6e9ef;
  --dim:#98a1b3; --accent:#6ea8fe; --accent2:#ffb454; --good:#5ad19b; --bad:#ff7a7a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif}
header{padding:18px 22px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:17px;font-weight:650;letter-spacing:.2px}
header p{margin:0;color:var(--dim);font-size:12.5px;max-width:86ch}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:#cbd5e6}
.wrap{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr) 300px;gap:14px;padding:14px 22px 26px}
@media(max-width:1400px){.wrap{grid-template-columns:minmax(0,1fr) 300px}}
@media(max-width:1000px){.wrap{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card h2{margin:0;padding:9px 12px;font-size:12px;font-weight:600;letter-spacing:.6px;
  text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--line);
  display:flex;justify-content:space-between;align-items:center;gap:8px}
.card .body{padding:12px}
canvas{display:block;width:100%;border-radius:6px;background:#0b0d11;touch-action:none}
.readout{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px;margin-top:10px}
.readout b{color:var(--dim);font-weight:500}
.readout span{font-family:ui-monospace,Menlo,monospace}
.ctl{margin-bottom:9px}
.ctl label{display:flex;justify-content:space-between;font-size:11.5px;color:var(--dim);gap:8px}
.ctl label .w{color:var(--accent);font-weight:600;text-align:right}
.ctl input[type=range]{width:100%;margin:3px 0 0;accent-color:var(--accent);height:16px}
.ctl .v{font-family:ui-monospace,Menlo,monospace;color:var(--ink);font-size:11.5px}
.sec{border-top:1px solid var(--line);margin:12px -12px 10px;padding:10px 12px 0}
.sec>span{font-size:10.5px;letter-spacing:.7px;text-transform:uppercase;color:var(--dim)}
button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:6px;
  padding:5px 9px;font-size:11.5px;cursor:pointer;font-family:inherit}
button:hover{border-color:var(--accent);color:#fff}
button.on{background:#243049;border-color:var(--accent)}
.row{display:flex;flex-wrap:wrap;gap:6px}
.prompt{background:#0b0d11;border:1px solid var(--line);border-radius:7px;padding:10px 11px;
  font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.65;color:#d7e2f5;white-space:pre-wrap}
.cats{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}
.tag{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:2px 9px;font-size:11px}
.tag b{color:var(--dim);font-weight:500}
.badge{font-size:10.5px;padding:2px 7px;border-radius:20px;border:1px solid var(--line)}
.badge.ok{color:var(--good);border-color:#255c45}
.badge.bad{color:var(--bad);border-color:#6b2f2f}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.gal{background:var(--panel2);border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.gal:hover{border-color:var(--accent)}
.gal.on{border-color:var(--accent2)}
.gal img{width:100%;display:block;aspect-ratio:4/3;object-fit:cover}
.gal div{padding:6px 8px;font-size:10.5px;color:var(--dim);line-height:1.35;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hint{color:var(--dim);font-size:11px;margin-top:8px}
.err{color:var(--bad)}
</style></head><body>
<header>
  <h1>Goal profile inspector <span style="color:var(--dim);font-weight:400">· DronePhotographer v12</span></h1>
  <p>A goal reaches the policy only as a sentence. But the 8 profile keys plus the crop fractions
     pin the camera to 5 geometric DOF, so the same goal can be drawn: where the camera stands
     around the subject, and what the frame looks like from there. Distances are in <b>subject
     heights</b>. Drag the 3D view to orbit, scroll to zoom.</p>
</header>
<div class="wrap">

  <section class="card">
    <h2>Camera in the subject frame <span id="dist3d" class="badge"></span></h2>
    <div class="body">
      <canvas id="c3d" width="1180" height="720"></canvas>
      <div class="row" style="margin-top:10px">
        <button id="tFrustum" class="on">frustum</button>
        <button id="tAxis" class="on">optical axis</button>
        <button id="tRing" class="on">bearing ring</button>
        <button id="tTruth" class="on">true pose</button>
        <button id="tReset">reset view</button>
      </div>
      <div class="readout" id="geom3d"></div>
    </div>
  </section>

  <section class="card">
    <h2>The frame <span style="font-weight:400;text-transform:none;letter-spacing:0">1024 × 768</span></h2>
    <div class="body">
      <canvas id="c2d" width="1024" height="768"></canvas>
      <div class="readout" id="geom2d"></div>
      <div class="hint">Solid blue box = the <b>visible</b> bbox the goal keys describe; the dot is
        <code>object_center_x/y</code>. Dashed orange = the full unclipped projection, and the
        orange cross is where the subject's real centre lands — the point the camera aims at.
        The shaded bands are what the frame cuts.</div>
    </div>
  </section>

  <section class="card">
    <h2>Goal <span id="parity" class="badge"></span></h2>
    <div class="body" id="controls"></div>
  </section>

  <section class="card" style="grid-column:1/-1">
    <h2>Prompt — exactly what conditions Cosmos</h2>
    <div class="body">
      <div class="prompt" id="prompt"></div>
      <div class="cats" id="cats"></div>
      <div class="hint" id="parityMsg"></div>
    </div>
  </section>

  <section class="card" id="gallerySec" style="grid-column:1/-1;display:none">
    <h2>Real goals from the dataset <span style="font-weight:400;text-transform:none;letter-spacing:0"
      id="galCount"></span></h2>
    <div class="body">
      <div class="gallery" id="gallery"></div>
      <div class="hint">Every frame here passes <code>is_goal_frame</code> — the real training
        gate. Click one to load its profile; the 3D view then also draws its <b>true</b> camera
        pose, so the reconstruction can be checked against the pose it came from.</div>
    </div>
  </section>
</div>

<script>
const DATA = /*__PAYLOAD__*/;
const C = DATA.cfg, GOALS = DATA.goals;
const D2R = Math.PI/180, R2D = 180/Math.PI;
const KEYS = C.goal_keys, BKEY = C.bearing_key;

/* ------------------------------------------------------------------ vocab */
/* Ported from src/goal_authoring/vocab.py, but driven entirely by the tables
   exported above — the bands live in one place, in Python. */
function classify(v, table){
  for(const [label,lo,hi] of table) if(v>=lo && v<hi) return label;
  return v<0 ? table[0][0] : table[table.length-1][0];
}
const sector8 = b => C.sector8[Math.floor((((b+22.5)%360)+360)%360/45)];
function cropPhrase(top,bot){
  const t=top>0.02, b=bot>0.02;
  if(t&&b) return "cropped at both the head and the feet";
  if(t) return "cropped above the head";
  if(b) return bot>0.35 ? "cropped below the waist" : "cropped at the legs";
  return "uncropped";
}
/* Port of src.data.lerobot_export.goal_prompt (full-goal branch: every key specified). */
function goalPrompt(g, crop){
  const T=C.tables, W=C.render_w, H=C.render_h;
  const cl=[`a ${classify(g.occupancy,T.SHOT_SIZE)} of the subject from the subject's ${sector8(g[BKEY])}`];
  cl.push(`at ${classify(g.cam_to_obj_elevation_deg,T.ELEVATION)}`);
  cl.push(`${classify(g.object_center_x/W,T.PLACE_X)} and ${classify(g.object_center_y/H,T.PLACE_Y)} in the frame`);
  cl.push(classify(g.body_in_frame_ratio,T.BODY_FRAMING));
  const vis = crop.vis!=null ? crop.vis : Math.max(0, 1-crop.top-crop.bot);
  if(crop.has) cl.push(cropPhrase(crop.top,crop.bot));
  const n=[`bearing ${Math.round(g[BKEY])}°`, `occupancy ${Math.round(g.occupancy)}%`,
           `elevation ${Math.round(g.cam_to_obj_elevation_deg)}°`,
           `body_in_frame ${Math.round(g.body_in_frame_ratio)}%`,
           `center ${Math.round(g.object_center_x)}/${Math.round(g.object_center_y)} px`,
           `half_size ${Math.round(g.bbox_x_offset)}/${Math.round(g.bbox_y_offset)} px`];
  if(crop.has) n.push(`visible ${vis.toFixed(2)}`);
  return `Move the camera to achieve this shot: ${cl.join(", ")}. (${n.join(", ")})`;
}
function categories(g,crop){
  const T=C.tables;
  const o={"shot size":classify(g.occupancy,T.SHOT_SIZE),
           "bearing":sector8(g[BKEY]),
           "elevation":classify(g.cam_to_obj_elevation_deg,T.ELEVATION),
           "placement x":classify(g.object_center_x/C.render_w,T.PLACE_X),
           "placement y":classify(g.object_center_y/C.render_h,T.PLACE_Y),
           "body framing":classify(g.body_in_frame_ratio,T.BODY_FRAMING)};
  if(crop.has) o["crop"]=cropPhrase(crop.top,crop.bot);
  return o;
}

/* ------------------------------------------------------------ vector math */
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const len=a=>Math.hypot(a[0],a[1],a[2]);
const norm=a=>{const l=len(a)||1;return [a[0]/l,a[1]/l,a[2]/l];};
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));

/* -------------------------------------------------- profile -> camera pose */
/* Pan-tilt (roll-free) frame, matching make_camera_basis_from_forward_up:
   right = forward x up, so a camera looking +X with up +Z has right = -Y. */
function panTilt(theta, phi){
  const cp=Math.cos(phi), sp=Math.sin(phi), ct=Math.cos(theta), st=Math.sin(theta);
  return {F:[cp*ct,cp*st,sp], U:[-sp*ct,-sp*st,cp], R:[st,-ct,0]};
}
/* Find the roll-free orientation whose aim offsets are exactly (ax, ay).
   Near the solution d(aim_x)/d(theta) = cos(elev) and d(aim_y)/d(phi) = 1, which is what the
   two step sizes below are; a dozen iterations drive the residual under 1e-10 rad. */
function aimFrame(e, ax, ay){
  let theta=Math.atan2(e[1],e[0]), phi=Math.asin(clamp(e[2],-1,1)), fr;
  for(let i=0;i<16;i++){
    phi=clamp(phi,-1.5533,1.5533);            // keep cos(phi) away from 0
    fr=panTilt(theta,phi);
    const df=dot(e,fr.F);
    const cax=Math.atan2(dot(e,fr.R),df), cay=Math.atan2(-dot(e,fr.U),df);
    const dax=ax-cax, day=ay-cay;
    if(Math.abs(dax)<1e-11 && Math.abs(day)<1e-11) break;
    theta+=dax/Math.max(Math.cos(phi),0.2); phi+=day;
  }
  return panTilt(theta,clamp(phi,-1.5533,1.5533));
}
/* The reconstruction. Subject frame: origin = subject_center, +X = where the subject looks,
   +Z = up, 1 unit = 1 subject height. */
function reconstruct(g, crop){
  const W=C.render_w,H=C.render_h,fx=C.fx,fy=C.fy;
  const vis=clamp(crop.vis!=null?crop.vis:(1-crop.top-crop.bot),0.02,1);
  const spanFull=2*g.bbox_y_offset/vis;
  const oyFull=Math.max(spanFull/2,1e-3);                 // visible half-height -> full
  const r=(0.5*fy)/oyFull;                                // subject heights
  const b=g[BKEY]*D2R, el=g.cam_to_obj_elevation_deg*D2R;
  const cam=[r*Math.cos(el)*Math.cos(b), -r*Math.cos(el)*Math.sin(b), -r*Math.sin(el)];
  // The aim is where the subject's CENTRE projects, and under the visible-bbox convention
  // `object_center_y` is the centre of the CLIPPED box — a median 116 px away from it on real
  // cropped goals. Undo the clip with the crop fractions: aiming from the recovered full centre
  // brings the reconstructed optical axis to 0.84 deg of the true pose instead of 2.96 deg.
  // (Horizontally there is no crop fraction to undo with, so object_center_x is used as-is;
  // it is only biased for a subject cut by the LEFT or RIGHT edge.)
  const y0f=(g.object_center_y-g.bbox_y_offset)-crop.top*spanFull;
  const cyFull=y0f+spanFull/2, y1f=y0f+spanFull;
  const ax=Math.atan((g.object_center_x-W/2)/fx), ay=Math.atan((cyFull-H/2)/fy);
  const fr=aimFrame(norm(mul(cam,-1)), ax, ay);
  // What profile_to_geometry() would decode from the VISIBLE bbox — equal to r only when
  // nothing is cropped, because that decode reads the clipped half-height as if it were full.
  const rDecode=(0.5*fy)/Math.max(g.bbox_y_offset,1e-3);
  return {r,cam,ax,ay,oyFull,rDecode,vis,spanFull,y0f,y1f,cyFull,...fr};
}

/* --------------------------------------------------------------- 3D scene */
const c3d=document.getElementById("c3d"), g3=c3d.getContext("2d");
const view={yaw:-2.25,pitch:0.30,dist:9.5};
const SHOW={frustum:1,axis:1,ring:1,truth:1};
const COL={cam:"#6ea8fe",truth:"#ffb454",subj:"#8f9bb3",grid:"#232833",ink:"#c6cfdd",dim:"#7d8798"};

/* Orbit around a point pushed a third of the way toward the goal camera, so the subject and the
   camera both sit inside the view instead of the camera hugging one edge. */
function viewBasis(){
  const t=STATE.rec?mul(STATE.rec.cam,0.32):[0,0,0];
  const cp=Math.cos(view.pitch), off=[view.dist*cp*Math.cos(view.yaw),
    view.dist*cp*Math.sin(view.yaw), view.dist*Math.sin(view.pitch)];
  const eye=add(t,off);
  const F=norm(mul(off,-1)), R=norm(cross(F,[0,0,1])), U=cross(R,F);
  return {eye,F,R,U};
}
function project(p,vb,w,h){
  const d=sub(p,vb.eye), z=dot(d,vb.F);
  if(z<=0.05) return null;
  const f=h/(2*Math.tan(28*D2R));
  return {x:w/2+f*dot(d,vb.R)/z, y:h/2-f*dot(d,vb.U)/z, z};
}
/* Every primitive carries its own depth and the whole scene is painter-sorted, so a wire
   behind the subject really is drawn behind it. */
function scene(rec, truth){
  const P=[];
  const line=(a,b,color,width,dash)=>P.push({t:"l",a,b,color,width:width||1,dash,
    z:(len(sub(a,VB.eye))+len(sub(b,VB.eye)))/2});
  const poly=(pts,fill,stroke)=>P.push({t:"p",pts,fill,stroke,
    z:pts.reduce((s,p)=>s+len(sub(p,VB.eye)),0)/pts.length});
  const text=(p,s,color,size,align)=>P.push({t:"t",p,s,color,size:size||11,align:align||"center",
    z:len(sub(p,VB.eye))});

  const FEET=-0.5;
  // ground grid, sized to the shot so a far wide goal still lands on it
  const rho0=Math.max(1.5,Math.hypot(rec.cam[0],rec.cam[1]));
  const ext=Math.max(3,rho0*1.35), step=ext/4;
  for(let i=-4;i<=4;i++){
    line([i*step,-ext,FEET],[i*step,ext,FEET],COL.grid,1);
    line([-ext,i*step,FEET],[ext,i*step,FEET],COL.grid,1);
  }
  // subject: boxes in subject-height units (feet at -0.5, head top at ~+0.47)
  const box=(cx,cy,cz,sx,sy,sz,shade)=>{
    const p=[[cx-sx,cy-sy,cz-sz],[cx+sx,cy-sy,cz-sz],[cx+sx,cy+sy,cz-sz],[cx-sx,cy+sy,cz-sz],
             [cx-sx,cy-sy,cz+sz],[cx+sx,cy-sy,cz+sz],[cx+sx,cy+sy,cz+sz],[cx-sx,cy+sy,cz+sz]];
    const faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[3,0,4,7]];
    const tint=[.55,1,.8,.7,.95,.75];
    faces.forEach((f,i)=>poly(f.map(k=>p[k]),shade(tint[i]),null));
  };
  const sh=t=>`rgba(${Math.round(126*t)},${Math.round(139*t)},${Math.round(163*t)},0.95)`;
  box(0,-0.075,-0.28,0.045,0.035,0.22,sh);   // legs
  box(0,0.075,-0.28,0.045,0.035,0.22,sh);
  box(0,0,0.09,0.055,0.10,0.17,sh);          // torso
  box(0,-0.135,0.08,0.035,0.03,0.15,sh);     // arms
  box(0,0.135,0.08,0.035,0.03,0.15,sh);
  box(0,0,0.29,0.035,0.035,0.035,sh);        // neck
  box(0,0,0.395,0.06,0.06,0.07,sh);          // head
  box(0.075,0,0.385,0.02,0.018,0.018,()=> "#d8dee9");   // nose -> +X is the facing direction
  // facing arrow on the ground
  line([0,0,FEET],[1.05,0,FEET],"#d8dee9",2);
  line([1.05,0,FEET],[0.9,0.09,FEET],"#d8dee9",2);
  line([1.05,0,FEET],[0.9,-0.09,FEET],"#d8dee9",2);
  text([1.28,0,FEET],"looks this way",COL.dim,11);

  const rho=Math.hypot(rec.cam[0],rec.cam[1]);
  if(SHOW.ring && rho>0.05){
    // bearing ring at the camera's horizontal radius + the eight sector8 words
    let prev=null;
    for(let i=0;i<=72;i++){
      const a=i/72*2*Math.PI, p=[rho*Math.cos(a),rho*Math.sin(a),FEET];
      if(prev) line(prev,p,"#394152",1); prev=p;
    }
    C.sector8.forEach((w,i)=>{
      const b=i*45*D2R, p=[rho*Math.cos(b),-rho*Math.sin(b),FEET];
      line([p[0]*0.96,p[1]*0.96,FEET],[p[0]*1.04,p[1]*1.04,FEET],"#55617a",1);
      text([p[0]*1.22,p[1]*1.22,FEET],w,i===rec.bearingIdx?COL.cam:COL.dim,
           i===rec.bearingIdx?12.5:10.5);
    });
    // Bearing arc from the facing direction to the camera, swept the SHORT way — it is an angle
    // between two directions, and a 303 deg sweep drawn the long way is indistinguishable from
    // the ring itself.
    const sweep=(rec.bearing<=180?-rec.bearing:360-rec.bearing)*D2R, ar=rho*0.34; let pv=null;
    for(let i=0;i<=48;i++){
      const a=sweep*i/48, p=[ar*Math.cos(a),ar*Math.sin(a),FEET+0.004];
      if(pv) line(pv,p,"#3f6ea8",1.5); pv=p;
    }
    const mid=sweep/2;
    text([ar*1.18*Math.cos(mid),ar*1.18*Math.sin(mid),FEET],`${rec.bearing.toFixed(0)}°`,COL.cam,11);
    // drop line + elevation wedge
    line([rec.cam[0],rec.cam[1],FEET],rec.cam,COL.cam,1,[3,4]);
    line([0,0,FEET],[rec.cam[0],rec.cam[1],FEET],COL.cam,1,[3,4]);
  }
  drawCam(rec.cam,rec.F,rec.U,rec.R,COL.cam,line,poly,text,rec);
  if(truth && SHOW.truth){
    const tf=norm(truth.forward), tu=norm(truth.up), tr=norm(cross(tf,tu));
    drawCam(truth.pos,tf,tu,tr,COL.truth,line,poly,text,null);
    text(add(truth.pos,mul(tu,-0.26)),"true pose",COL.truth,11);
  }
  // line of sight to the subject centre (dead-centre aim), for contrast with the optical axis
  line(rec.cam,[0,0,0],"#4b5568",1);
  return P;
}
function drawCam(pos,F,U,R,color,line,poly,text,rec){
  const hx=Math.atan(C.render_w/2/C.fx), hy=Math.atan(C.render_h/2/C.fy);
  const d=Math.max(0.35,len(pos)*0.30);
  const corner=(sx,sy)=>add(pos,mul(norm(add(add(mul(F,1),mul(R,sx*Math.tan(hx))),
                                             mul(U,sy*Math.tan(hy)))),d));
  const c=[corner(1,1),corner(-1,1),corner(-1,-1),corner(1,-1)];
  if(SHOW.frustum){
    c.forEach(p=>line(pos,p,color,1.2));
    for(let i=0;i<4;i++) line(c[i],c[(i+1)%4],color,1.2);
    // a gable over the top edge, so the camera's ROLL is readable at a glance
    const apex=add(mul(add(c[0],c[1]),0.5),mul(U,0.11*d));
    line(c[0],apex,color,1.2); line(c[1],apex,color,1.2);
  }
  // camera body
  const bx=(a,b_)=>line(a,b_,color,1.6);
  const s=0.055*Math.max(1,len(pos)*0.35);
  const p=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
    .map(v=>add(pos,add(add(mul(R,v[0]*s),mul(U,v[1]*s)),mul(F,v[2]*s))));
  [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(e=>bx(p[e[0]],p[e[1]]));
  if(SHOW.axis){
    // the optical axis, extended past the subject plane: where it lands vs the subject centre
    // IS the aim offset that object_center_x/y encode.
    const t=Math.max(0.1,len(pos));
    line(pos,add(pos,mul(F,t*1.25)),color,1,[5,5]);
    line(add(pos,mul(F,t)),[0,0,0],"#59657d",1,[2,4]);
  }
  if(rec) text(add(pos,mul(U,0.28)),`${rec.r.toFixed(2)} h`,color,11.5);
}
function paint(){
  const w=c3d.width,h=c3d.height;
  g3.setTransform(1,0,0,1,0,0); g3.clearRect(0,0,w,h);
  g3.fillStyle="#0b0d11"; g3.fillRect(0,0,w,h);
  VB=viewBasis();
  const P=scene(STATE.rec,STATE.truth).sort((a,b)=>b.z-a.z);
  for(const it of P){
    if(it.t==="l"){
      const a=project(it.a,VB,w,h), b=project(it.b,VB,w,h); if(!a||!b) continue;
      g3.strokeStyle=it.color; g3.lineWidth=it.width; g3.setLineDash(it.dash||[]);
      g3.beginPath(); g3.moveTo(a.x,a.y); g3.lineTo(b.x,b.y); g3.stroke(); g3.setLineDash([]);
    } else if(it.t==="p"){
      const pts=it.pts.map(p=>project(p,VB,w,h)); if(pts.some(p=>!p)) continue;
      g3.beginPath(); pts.forEach((p,i)=>i?g3.lineTo(p.x,p.y):g3.moveTo(p.x,p.y)); g3.closePath();
      if(it.fill){g3.fillStyle=it.fill; g3.fill();}
      if(it.stroke){g3.strokeStyle=it.stroke; g3.lineWidth=1; g3.stroke();}
    } else {
      const p=project(it.p,VB,w,h); if(!p) continue;
      g3.fillStyle=it.color; g3.font=`${it.size}px ui-sans-serif,sans-serif`;
      g3.textAlign=it.align; g3.textBaseline="middle"; g3.fillText(it.s,p.x,p.y);
    }
  }
}
// Placeholder only — paint() recomputes it before anything reads it. Calling viewBasis() here
// would touch STATE before its declaration a few blocks down (temporal dead zone) and the whole
// page would fail to boot.
let VB={eye:[0,0,0],F:[1,0,0],R:[0,1,0],U:[0,0,1]};
(function orbit(){
  let drag=null;
  c3d.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];c3d.setPointerCapture(e.pointerId);});
  c3d.addEventListener("pointermove",e=>{
    if(!drag) return;
    view.yaw-=(e.clientX-drag[0])*0.008;
    view.pitch=clamp(view.pitch+(e.clientY-drag[1])*0.006,-1.35,1.45);
    drag=[e.clientX,e.clientY]; paint();
  });
  addEventListener("pointerup",()=>drag=null);
  c3d.addEventListener("wheel",e=>{e.preventDefault();
    view.dist=clamp(view.dist*Math.exp(e.deltaY*0.0012),1.2,80); paint();},{passive:false});
})();

/* --------------------------------------------------------------- 2D frame */
const c2d=document.getElementById("c2d"), g2=c2d.getContext("2d");
/* The frame is drawn as a WINDOW inside a larger canvas: whatever the frame cuts off has to be
   somewhere, and putting it in the margin is what makes "cropped above the head" a picture
   instead of a phrase. Image pixels map through tx/ty; nothing is drawn in raw image coords. */
function paint2d(){
  const W=C.render_w,H=C.render_h,g=STATE.goal,crop=STATE.crop;
  const rec=STATE.rec||reconstruct(g,crop);
  // Fit the frame AND whatever spills out of it. A fixed margin is enough for a mild crop and
  // useless for a close-up whose subject is twice the frame's height — and that is exactly the
  // case worth looking at.
  const bx0=Math.min(0,g.object_center_x-g.bbox_x_offset)-20;
  const bx1=Math.max(W,g.object_center_x+g.bbox_x_offset)+20;
  const by0=Math.min(0,rec.y0f)-20, by1=Math.max(H,rec.y1f)+20;
  const S2=Math.min(0.74, 1024/(bx1-bx0), 768/(by1-by0));
  const OX2=(1024-(bx1-bx0)*S2)/2-bx0*S2, OY2=(768-(by1-by0)*S2)/2-by0*S2;
  const tx=x=>OX2+S2*x, ty=y=>OY2+S2*y, ts=v=>S2*v;
  g2.setTransform(1,0,0,1,0,0);
  g2.fillStyle="#080a0d"; g2.fillRect(0,0,1024,768);
  const fx0=tx(0),fy0=ty(0),fw=ts(W),fh=ts(H);
  if(STATE.image){ g2.globalAlpha=0.6; g2.drawImage(STATE.image,fx0,fy0,fw,fh); g2.globalAlpha=1; }
  else { g2.fillStyle="#11151c"; g2.fillRect(fx0,fy0,fw,fh); }

  const cx=g.object_center_x, cy=g.object_center_y, ox=g.bbox_x_offset, oy=g.bbox_y_offset;
  const y0f=rec.y0f, y1f=rec.y1f, bh=y1f-y0f, hw=ox;
  // The regions the frame cuts away, laid down FIRST so the silhouette still reads through them.
  g2.fillStyle="rgba(255,122,122,0.17)";
  if(crop.top>0.001) g2.fillRect(tx(cx-hw)-6,ty(y0f),ts(2*hw)+12,ts(-y0f));
  if(crop.bot>0.001) g2.fillRect(tx(cx-hw)-6,ty(H),ts(2*hw)+12,ts(y1f-H));
  // subject silhouette, drawn across the FULL projection so the cut ends land in the margin
  const U=u=>tx(cx+u*hw), V=v=>ty(y0f+v*bh);
  // A real render underneath is the better evidence, so the silhouette thins out to an outline
  // rather than covering it. Without an image it fills, because then it is all there is.
  g2.fillStyle=STATE.image?"rgba(110,168,254,0.05)":"rgba(110,168,254,0.20)";
  g2.strokeStyle="rgba(110,168,254,0.65)"; g2.lineWidth=2;
  g2.beginPath(); g2.ellipse(U(0),V(0.065),ts(hw*0.32),ts(bh*0.065),0,0,7); g2.fill(); g2.stroke();
  const BODY=[[-0.22,0.135],[-1,0.22],[-0.86,0.32],[-0.62,0.55],[-0.66,1],[-0.12,1],
              [-0.06,0.62],[0.06,0.62],[0.12,1],[0.66,1],[0.62,0.55],[0.86,0.32],[1,0.22],[0.22,0.135]];
  g2.beginPath(); BODY.forEach((p,i)=>i?g2.lineTo(U(p[0]),V(p[1])):g2.moveTo(U(p[0]),V(p[1])));
  g2.closePath(); g2.fill(); g2.stroke();

  // thirds, inside the frame only
  g2.strokeStyle="rgba(255,255,255,0.16)"; g2.lineWidth=1.5;
  [1,2].forEach(i=>{g2.beginPath();
    g2.moveTo(tx(W*i/3),fy0);g2.lineTo(tx(W*i/3),fy0+fh);
    g2.moveTo(fx0,ty(H*i/3));g2.lineTo(fx0+fw,ty(H*i/3));g2.stroke();});
  // full bbox (dashed) then visible bbox (solid)
  g2.setLineDash([9,7]); g2.strokeStyle="rgba(255,180,84,0.9)"; g2.lineWidth=2;
  g2.strokeRect(tx(cx-ox),ty(y0f),ts(2*ox),ts(bh)); g2.setLineDash([]);
  g2.strokeStyle="#6ea8fe"; g2.lineWidth=3;
  g2.strokeRect(tx(cx-ox),ty(cy-oy),ts(2*ox),ts(2*oy));
  // the frame edge itself — the thing doing the cutting
  g2.strokeStyle="#e6e9ef"; g2.lineWidth=2.5; g2.strokeRect(fx0,fy0,fw,fh);
  // object_center marker
  g2.strokeStyle="rgba(110,168,254,0.4)"; g2.lineWidth=1.5; g2.setLineDash([4,6]);
  g2.beginPath();g2.moveTo(tx(cx),fy0);g2.lineTo(tx(cx),fy0+fh);
  g2.moveTo(fx0,ty(cy));g2.lineTo(fx0+fw,ty(cy));g2.stroke(); g2.setLineDash([]);
  g2.fillStyle="#6ea8fe"; g2.beginPath(); g2.arc(tx(cx),ty(cy),6,0,7); g2.fill();
  // Where the subject's TRUE centre projects. It is what the camera aims at, and on a cropped
  // shot it is not object_center_y — that key names the centre of the clipped box.
  if(Math.abs(rec.cyFull-cy)>2){
    g2.strokeStyle="#ffb454"; g2.lineWidth=2.5; g2.beginPath();
    g2.moveTo(tx(cx)-11,ty(rec.cyFull)); g2.lineTo(tx(cx)+11,ty(rec.cyFull));
    g2.moveTo(tx(cx),ty(rec.cyFull)-11); g2.lineTo(tx(cx),ty(rec.cyFull)+11); g2.stroke();
    g2.setLineDash([3,5]); g2.strokeStyle="rgba(255,180,84,0.65)"; g2.lineWidth=2;
    g2.beginPath(); g2.moveTo(tx(cx),ty(cy)); g2.lineTo(tx(cx),ty(rec.cyFull)); g2.stroke();
    g2.setLineDash([]);
  }
  g2.font="600 21px ui-sans-serif,sans-serif"; g2.textBaseline="middle"; g2.textAlign="left";
  g2.fillStyle="#e6e9ef";
  g2.fillText(`${classify(g.occupancy,C.tables.SHOT_SIZE)} · ${sector8(g[BKEY])} · `
             +`${classify(g.cam_to_obj_elevation_deg,C.tables.ELEVATION)}`,14,22);
  g2.font="15px ui-sans-serif,sans-serif"; g2.fillStyle="#7d8798"; g2.textAlign="right";
  g2.fillText(cropPhrase(crop.top,crop.bot),1010,22);
}

/* ------------------------------------------------------------------ state */
const STATE={goal:{},crop:{top:0,bot:0,has:true},rec:null,truth:null,image:null,
             pyPrompt:null,active:-1};
function setGoal(g,crop,opts){
  KEYS.forEach(k=>STATE.goal[k]=+g[k]||0);
  STATE.crop={top:+(crop&&crop.top||0),bot:+(crop&&crop.bot||0),has:true,
              vis:(crop&&crop.vis!=null)?+crop.vis:null};
  STATE.truth=(opts&&opts.truth)||null;
  STATE.pyPrompt=(opts&&opts.prompt)||null;
  STATE.image=null;
  if(opts&&opts.image){const im=new Image();im.onload=()=>{STATE.image=im;paint2d();};im.src=opts.image;}
  syncControls(); render(); frameView(); paint();
}
function render(){
  const g=STATE.goal,crop=STATE.crop;
  const rec=reconstruct(g,crop);
  rec.bearing=((g[BKEY]%360)+360)%360;
  rec.bearingIdx=Math.floor((((rec.bearing+22.5)%360)+360)%360/45);
  STATE.rec=rec;
  paint(); paint2d();

  const el=g.cam_to_obj_elevation_deg;
  document.getElementById("dist3d").textContent=`${rec.r.toFixed(2)} subject heights`;
  const rows=[
    ["bearing", `${rec.bearing.toFixed(1)}° · ${sector8(rec.bearing)}`],
    ["elevation", `${el.toFixed(1)}° · camera ${el<0?"above":(el>0?"below":"level with")} subject`],
    ["range", `${rec.r.toFixed(2)} h  (${(rec.r*1.7).toFixed(1)} m for a 1.7 m subject)`],
    ["aim offset", `${(rec.ax*R2D).toFixed(2)}° x · ${(rec.ay*R2D).toFixed(2)}° y`],
    ["half-height", `${g.bbox_y_offset.toFixed(0)} px visible → ${rec.oyFull.toFixed(0)} px full`],
  ];
  // A goal is only shootable if the camera can stand somewhere. Elevation is measured from the
  // subject's CENTRE, so a steep "low angle" close in puts the camera under the floor — the
  // vocabulary allows it, the world does not.
  if(rec.cam[2]<-0.5) rows.push(["reachable?",
    `camera sits ${(-0.5-rec.cam[2]).toFixed(2)} h BELOW the ground plane — not shootable at `
    +`this range without lowering the elevation or backing off`]);
  if(Math.abs(rec.cyFull-g.object_center_y)>2) rows.push(["subject centre",
    `projects at y ${rec.cyFull.toFixed(0)} px, ${Math.abs(rec.cyFull-g.object_center_y).toFixed(0)} px `
    +`${rec.cyFull>g.object_center_y?"below":"above"} object_center_y — that key names the `
    +`CLIPPED box's centre, and the camera aims at the subject's`]);
  if(Math.abs(rec.rDecode-rec.r)>0.02) rows.push(["profile_to_geometry",
    `decodes ${rec.rDecode.toFixed(2)} h — it reads the visible half-height as if it were the `
    +`full one, so a cropped goal decodes ${(100*(1-rec.rDecode/rec.r)).toFixed(0)}% too near`]);
  if(STATE.truth){
    const d=len(sub(rec.cam,STATE.truth.pos));
    const ang=Math.acos(clamp(dot(norm(rec.F),norm(STATE.truth.forward)),-1,1))*R2D;
    rows.push(["vs true pose",
      `position ${d.toFixed(3)} h · optical axis ${ang.toFixed(2)}° · true range ${STATE.truth.r.toFixed(2)} h`]);
  }
  document.getElementById("geom3d").innerHTML=
    rows.map(([a,b])=>`<b>${a}</b><span>${b}</span>`).join("");

  const vis=rec.vis;
  document.getElementById("geom2d").innerHTML=[
    ["visible bbox", `x ${(g.object_center_x-g.bbox_x_offset).toFixed(0)}–`
      +`${(g.object_center_x+g.bbox_x_offset).toFixed(0)}, y `
      +`${(g.object_center_y-g.bbox_y_offset).toFixed(0)}–${(g.object_center_y+g.bbox_y_offset).toFixed(0)} px`],
    ["crop", `${cropPhrase(crop.top,crop.bot)} · visible ${vis.toFixed(2)} `
      +`(top ${crop.top.toFixed(2)} / bottom ${crop.bot.toFixed(2)})`],
    ["occupancy", `${g.occupancy.toFixed(0)}% of frame area · body in frame ${g.body_in_frame_ratio.toFixed(0)}%`],
  ].map(([a,b])=>`<b>${a}</b><span>${b}</span>`).join("");

  const jsPrompt=goalPrompt(g,crop);
  document.getElementById("prompt").textContent=jsPrompt;
  document.getElementById("cats").innerHTML=Object.entries(categories(g,crop))
    .map(([k,v])=>`<span class="tag"><b>${k}</b> ${v}</span>`).join("");
  const badge=document.getElementById("parity"), msg=document.getElementById("parityMsg");
  if(STATE.pyPrompt){
    const ok=STATE.pyPrompt===jsPrompt;
    badge.className="badge "+(ok?"ok":"bad");
    badge.textContent=ok?"prompt parity ✓":"prompt parity ✗";
    msg.className=ok?"hint":"hint err";
    msg.textContent=ok
      ? "Identical to the string src.data.lerobot_export.goal_prompt produced for this frame in Python."
      : "MISMATCH vs Python:\n"+STATE.pyPrompt;
  } else { badge.className="badge"; badge.textContent="authored"; msg.className="hint";
    msg.textContent="Authored goal — load a dataset goal below to check this page's serializer "
      +"against the Python one."; }
}

/* --------------------------------------------------------------- controls */
const SLIDERS=[
  ["occupancy","occupancy %",0,100,1,g=>classify(g.occupancy,C.tables.SHOT_SIZE)],
  [BKEY,"subject bearing °",0,359,1,g=>sector8(g[BKEY])],
  ["cam_to_obj_elevation_deg","elevation ° (− = camera above)",-90,90,1,
    g=>classify(g.cam_to_obj_elevation_deg,C.tables.ELEVATION)],
  ["object_center_x","object_center_x px",-200,1224,1,
    g=>classify(g.object_center_x/C.render_w,C.tables.PLACE_X)],
  ["object_center_y","object_center_y px",-200,968,1,
    g=>classify(g.object_center_y/C.render_h,C.tables.PLACE_Y)],
  ["bbox_x_offset","bbox_x_offset px (half width)",5,1200,1,null],
  ["bbox_y_offset","bbox_y_offset px (half height)",5,1200,1,null],
  ["body_in_frame_ratio","body_in_frame %",0,100,1,g=>classify(g.body_in_frame_ratio,C.tables.BODY_FRAMING)],
];
function buildControls(){
  const host=document.getElementById("controls"); host.innerHTML="";
  SLIDERS.forEach(([key,label,lo,hi,step,word])=>{
    const d=document.createElement("div"); d.className="ctl";
    d.innerHTML=`<label><span>${label}</span><span class="w" data-w="${key}"></span></label>`
      +`<input type="range" min="${lo}" max="${hi}" step="${step}" data-k="${key}">`
      +`<div class="v" data-v="${key}"></div>`;
    host.appendChild(d);
    d.querySelector("input").addEventListener("input",e=>{
      STATE.goal[key]=+e.target.value; STATE.pyPrompt=null; STATE.active=-1;
      document.querySelectorAll(".gal").forEach(x=>x.classList.remove("on"));
      syncControls(); render();
    });
  });
  const sec=document.createElement("div"); sec.className="sec";
  sec.innerHTML=`<span>crop — which end the frame cuts</span>`;
  host.appendChild(sec);
  [["top","top cut fraction"],["bot","bottom cut fraction"]].forEach(([k,label])=>{
    const d=document.createElement("div"); d.className="ctl";
    d.innerHTML=`<label><span>${label}</span><span class="w" data-w="crop_${k}"></span></label>`
      +`<input type="range" min="0" max="0.85" step="0.01" data-k="crop_${k}">`
      +`<div class="v" data-v="crop_${k}"></div>`;
    host.appendChild(d);
    d.querySelector("input").addEventListener("input",e=>{
      STATE.crop[k]=Math.min(+e.target.value,0.97-STATE.crop[k==="top"?"bot":"top"]);
      STATE.crop.vis=null;            // hand-edited crop: recompute visible_frac from the pair
      STATE.pyPrompt=null; syncControls(); render();
    });
  });
  const p=document.createElement("div"); p.className="sec";
  p.innerHTML=`<span>presets</span><div class="row" style="margin-top:8px" id="presets"></div>`;
  host.appendChild(p);
  PRESETS.forEach((pr,i)=>{
    const b=document.createElement("button"); b.textContent=pr.name;
    b.onclick=()=>setGoal(pr.goal,pr.crop,{});
    document.getElementById("presets").appendChild(b);
  });
}
function syncControls(){
  const g=STATE.goal;
  SLIDERS.forEach(([key,,,,,word])=>{
    const inp=document.querySelector(`input[data-k="${key}"]`); if(inp) inp.value=g[key];
    const v=document.querySelector(`[data-v="${key}"]`); if(v) v.textContent=(+g[key]).toFixed(0);
    const w=document.querySelector(`[data-w="${key}"]`); if(w) w.textContent=word?word(g):"";
  });
  ["top","bot"].forEach(k=>{
    const inp=document.querySelector(`input[data-k="crop_${k}"]`); if(inp) inp.value=STATE.crop[k];
    const v=document.querySelector(`[data-v="crop_${k}"]`); if(v) v.textContent=STATE.crop[k].toFixed(2);
  });
  const w=document.querySelector(`[data-w="crop_top"]`);
  if(w) w.textContent=cropPhrase(STATE.crop.top,STATE.crop.bot);
}

/* --------------------------------------------------------------- gallery */
function buildGallery(){
  if(!GOALS.length) return;
  document.getElementById("gallerySec").style.display="";
  document.getElementById("galCount").textContent=`${GOALS.length} frames`;
  const host=document.getElementById("gallery");
  GOALS.forEach((s,i)=>{
    const d=document.createElement("div"); d.className="gal";
    d.innerHTML=`<img src="${s.image}" loading="lazy"><div>${s.name}</div>`;
    d.onclick=()=>{
      document.querySelectorAll(".gal").forEach(x=>x.classList.remove("on"));
      d.classList.add("on"); STATE.active=i;
      setGoal(s.goal,s.crop,{truth:s.truth,prompt:s.prompt,image:s.image});
    };
    host.appendChild(d);
  });
}

/* ------------------------------------------------------------------- boot */
const PRESETS=DATA.initial.presets;
["frustum","axis","ring","truth"].forEach(k=>{
  const b=document.getElementById("t"+k[0].toUpperCase()+k.slice(1));
  b.onclick=()=>{SHOW[k]=!SHOW[k]; b.classList.toggle("on",!!SHOW[k]); paint();};
});
/* Frame the scene FROM the goal: viewing from ~115 deg around the goal camera keeps the camera
   and the subject at similar depth and well apart. A fixed yaw puts the goal camera behind the
   subject for whole bands of bearings, where it is small and overlapping and reads as wrong. */
function frameView(){
  view.yaw=Math.atan2(STATE.rec.cam[1],STATE.rec.cam[0])+2.0;
  view.pitch=0.42;
  view.dist=clamp(STATE.rec.r*1.9,3.4,60);
}
document.getElementById("tReset").onclick=()=>{frameView(); paint();};
buildControls(); buildGallery();
setGoal(DATA.initial.goal, DATA.initial.crop, DATA.initial.opts||{});
frameView(); paint();
</script></body></html>
"""


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=0,
                    help="number of real dataset goals to embed in the gallery (0 = none)")
    ap.add_argument("--root", default=DEFAULT_TRAJ_ROOT,
                    help="trajectory root; name the SUBDIRECTORY, the loader globs recursively")
    ap.add_argument("--nl", default=None,
                    help="seed the inspector from a natural-language request (keyword classifier)")
    ap.add_argument("--profile", default=None,
                    help="seed from a JSON dict of raw profile values")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    presets = preset_goals()
    initial = {"goal": presets[1]["goal"], "crop": presets[1]["crop"], "presets": presets, "opts": {}}

    if args.nl:
        from src.goal_authoring.from_language import keyword_classifier, language_to_goal
        gp = language_to_goal(args.nl, keyword_classifier)
        g = dict(presets[1]["goal"])
        g.update({k: float(v) for k, v in gp.values.items() if k in DEFAULT_GOAL_KEYS})
        half_h, half_w = _half_extents_from_occupancy(g["occupancy"])
        initial["goal"], initial["crop"] = _fit_authored_box(g, half_w, half_h)
        unspecified = [k for k in DEFAULT_GOAL_KEYS if k not in gp.specified]
        print(f'NL "{args.nl}" -> {gp.categories()}')
        if unspecified:
            print(f"  unconstrained (shown at the preset default): {', '.join(unspecified)}")
    elif args.profile:
        g = dict(presets[1]["goal"])
        g.update({k: float(v) for k, v in json.loads(args.profile).items() if k in DEFAULT_GOAL_KEYS})
        initial["goal"] = g

    goals = sample_real_goals(args.sample, args.root, args.seed) if args.sample > 0 else []
    if goals:
        print(f"sampled {len(goals)} real goal frames")

    html = build_html(page_config(), goals, initial)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
