"""Human-verification gallery for the isolated TURNTABLE facing pass (clean-substrate replacement
for verify_facing_html.py). One card per asset: the K-view turntable contact sheet (each thumbnail
labeled with its data-convention azimuth = cam_to_obj_azimuth_deg), with the AUTO-estimated FRONT
(facing_from_turntable.py) highlighted green. Sorted tail-first (no-face -> VERIFY -> OK) so review
effort goes where it's needed. The user checks: is the highlighted view the subject's front? If not,
note the correct azimuth. Faceless assets (snowman) have no auto guess -> pick the front by eye.

Self-contained; .venv-analysis. Reads runs/facing_turntable/index.json + runs/facing_turntable_auto.json.
Writes runs/facing_turntable_verify.html.
"""
import base64
import io
import json
import os

os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image

IDX = json.load(open("runs/facing_turntable/index.json"))
AUTO = (json.load(open("runs/facing_turntable_auto.json"))
        if os.path.exists("runs/facing_turntable_auto.json") else {})
OUT = "runs/facing_turntable_verify.html"


def cdiff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def thumb(ip, w=150):
    im = Image.open(ip).convert("RGB")
    im.thumbnail((w, int(w * im.height / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=74)
    return base64.b64encode(buf.getvalue()).decode()


ORDER = {"render_error": 0, "too_few_faces": 0, None: 0, "VERIFY": 1, "OK": 2}


def sortkey(kv):
    a = AUTO.get(kv[0], {})
    return ORDER.get(a.get("conf", a.get("status")), 3)


rows = sorted(IDX.items(), key=sortkey)

cards = []
n_ok = n_verify = n_manual = 0
for obj, info in rows:
    a = AUTO.get(obj, {})
    front = a.get("front_az")
    facing = a.get("facing_world_deg")
    conf = a.get("conf", a.get("status", "?"))
    badge = {"OK": "#34a853", "VERIFY": "#fbbc04",
             "too_few_faces": "#ea4335", "render_error": "#9333ea"}.get(conf, "#ea4335")
    if conf == "OK":
        n_ok += 1
    elif conf == "VERIFY":
        n_verify += 1
    else:
        n_manual += 1

    views = info.get("views", [])
    if not views:
        cards.append(f"""<div class=card>
<div class=hd><span class=nm>{obj[:46]}</span>
<span class=badge style="background:#9333ea">render_error</span></div>
<div class=meta><b>no render</b>: {info.get('error', 'unknown')}</div></div>""")
        continue

    step = 360.0 / len(views)
    cells = []
    for v in views:
        A = v["az"]
        is_front = (front is not None and cdiff(A, front) < step / 2 + 1)
        bd = "border:3px solid #34a853" if is_front else "border:3px solid #2a2e35"
        tag = "<div class=ft>FRONT?</div>" if is_front else ""
        cells.append(f'<div class=v style="{bd}">'
                     f'<img src="data:image/jpeg;base64,{thumb(v["path"])}">{tag}'
                     f'<div class=az>az {int(A)}°</div></div>')
    if facing is not None:
        meta = (f"auto facing ≈ <b>{facing}°</b> · front cam az <b>{front}°</b> · "
                f"faces {a.get('n_detect', 0)}/{len(views)} · contrast {a.get('contrast', '–')} · "
                f"sep {a.get('front_back_sep', '–')}")
    else:
        meta = (f"<b>no auto face</b> ({a.get('status', '?')}) — "
                f"pick the subject's front by eye (carrot / nose / gaze / chest)")
    cards.append(f"""<div class=card>
<div class=hd><span class=nm>{obj[:46]}</span>
<span class=badge style="background:{badge}">{conf}</span></div>
<div class=meta>{meta}</div>
<div class=sheet>{''.join(cells)}</div></div>""")

html = f"""<title>v12 facing verification (turntable)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:22px}}
h1{{font-size:18px;margin:0 0 4px}} p.sub{{color:#9aa0a6;font-size:13px;margin:0 0 14px}}
.tot{{color:#8ab4f8}}
.card{{background:#1e2126;border:1px solid #2a2e35;border-radius:10px;padding:12px;margin:14px 0}}
.hd{{display:flex;justify-content:space-between;align-items:center}}
.nm{{font-weight:600;font-size:14px}}
.badge{{color:#111;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px}}
.meta{{color:#8ab4f8;font-size:12px;margin:5px 0 9px;font-variant-numeric:tabular-nums}}
.sheet{{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}}
.v{{position:relative;border-radius:6px;flex:0 0 auto}}
.v img{{display:block;border-radius:4px}}
.az{{position:absolute;bottom:2px;left:2px;background:#000a;color:#fff;font-size:10px;padding:1px 4px;border-radius:3px}}
.ft{{position:absolute;top:2px;right:2px;background:#34a853;color:#fff;font-size:10px;font-weight:700;padding:1px 4px;border-radius:3px}}
</style>
<h1>v12 facing verification — isolated turntable · is the green view the subject's FRONT?</h1>
<p class=sub>Clean solo renders (no scene). Each thumbnail labeled with its <b>camera azimuth</b>
(= stored cam_to_obj_azimuth_deg). Green = auto-estimated front. Sorted worst-first
(<span class=tot>{n_manual} no-face/err → {n_verify} VERIFY → {n_ok} OK</span>, {len(rows)} total).
If the green view isn't the front, note the correct azimuth. Red/error cards need a manual front pick.</p>
{''.join(cards)}"""
os.makedirs("runs", exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB) — {len(cards)} assets "
      f"({n_ok} OK / {n_verify} VERIFY / {n_manual} manual)")
