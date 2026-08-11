"""Human-verification UI for the automated facing pass. One card per asset: an 8-view orbit contact
sheet (each thumbnail labeled with its camera azimuth), with the AUTO-estimated FRONT highlighted.
Sorted tail-first (no-face -> VERIFY -> OK) so review effort goes where it's needed. The user scans:
is the highlighted view actually the person's front? If not, note the correct azimuth.
Self-contained; .venv-analysis. Reads runs/facing_auto_full.json + data. Writes runs/facing_verify.html."""
import os, json, base64, io
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ROOT = DEFAULT_TRAJ_ROOT
AUTO = json.load(open("runs/facing_auto_full.json"))
OUT  = "runs/facing_verify.html"

# map obj -> a placement dir
dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
by_obj = {}
for d in sorted(dirs):
    by_obj.setdefault(d.split("__",1)[1] if "__" in d else d, d)

def cdiff(a,b): return abs(((a-b+180)%360)-180)

def orbit_views(dn, k=8):
    try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
    except Exception: return []
    buckets = {}
    for pair in d.get("render_records", []):
        for r in pair:
            s = r.get("scores")
            if not s or not r.get("in_frame"): continue
            A = s["cam_to_obj_azimuth_deg"]; b = int((A % 360)//(360/k))
            # prefer a well-sized frame per sector
            if b not in buckets or s["occupancy"] > buckets[b][2]:
                buckets[b] = (A, os.path.join(ROOT,dn,r["path_rel"]), s["occupancy"])
    return sorted(buckets.values())

def thumb(ip, w=170):
    im = Image.open(ip).convert("RGB"); im.thumbnail((w,int(w*im.height/im.width)))
    buf = io.BytesIO(); im.save(buf,"JPEG",quality=72); return base64.b64encode(buf.getvalue()).decode()

order = {"too_few_faces":0, None:0, "VERIFY":1, "OK":2}
rows = sorted(AUTO.items(), key=lambda kv: order.get(kv[1].get("conf", kv[1].get("status")), 3))

cards = []
for obj, r in rows:
    dn = by_obj.get(obj)
    if not dn: continue
    views = orbit_views(dn)
    if not views: continue
    facing = r.get("facing_world_deg"); front = r.get("front_az")
    conf = r.get("conf", r.get("status","?"))
    badge = {"OK":"#34a853","VERIFY":"#fbbc04","too_few_faces":"#ea4335"}.get(conf,"#ea4335")
    cells = []
    for A, ip, occ in views:
        is_front = (front is not None and cdiff(A, front) < 23)
        bd = "border:3px solid #34a853" if is_front else "border:3px solid transparent"
        tag = "<div class=ft>FRONT?</div>" if is_front else ""
        cells.append(f'<div class=v style="{bd}"><img src="data:image/jpeg;base64,{thumb(ip)}">{tag}<div class=az>az {int(A)}°</div></div>')
    meta = (f"auto facing ≈ <b>{facing}°</b> · front cam az {front}° · contrast {r.get('contrast','–')} · "
            f"faces {r.get('n_detect','0')} · sep {r.get('front_back_sep','–')}") if facing is not None else \
           f"<b>no reliable face</b> ({r.get('n_detect',0)} detections) — needs manual"
    cards.append(f"""<div class=card>
<div class=hd><span class=nm>{obj[:46]}</span>
<span class=badge style="background:{badge}">{conf}</span></div>
<div class=meta>{meta}</div>
<div class=sheet>{''.join(cells)}</div></div>""")

html = f"""<title>v12 facing verification</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:22px}}
h1{{font-size:18px}} p.sub{{color:#9aa0a6;font-size:13px}}
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
<h1>v12 facing verification — is the green view the person's FRONT?</h1>
<p class=sub>Automated (YuNet over the full orbit). Sorted worst-first (red no-face → yellow VERIFY → green OK). For each asset: does the green-highlighted view show the face/front? If wrong, note the correct <b>camera azimuth</b> (labels on each thumbnail). Red cards need a manual front pick.</p>
{''.join(cards)}"""
os.makedirs("runs", exist_ok=True); open(OUT,"w").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB) — {len(cards)} assets")
