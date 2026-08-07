"""Render the true-recon result as a TRAJECTORY: reference, then every chunk the policy executed.

The earlier version showed only start and end, which cannot tell "went straight there" from
"wandered and came back". Here each case is the full strip — reference | start | after chunk 1 | ...
— with the distance-to-goal under each frame, so the whole rollout is visible. venv: .venv-analysis."""
import base64, io, json, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default="runs/recon_ref")
_ap.add_argument("--out", default="runs/recon_report.html")
_ap.add_argument("--checkpoint-label", default="iter 6000")
_a = _ap.parse_args()
ROWS = json.load(open(f"{_a.dir}/measured.json"))
RECON = {c["ref_image"]: c for c in json.load(open(f"{_a.dir}/recon.json"))["results"]}
OUT = _a.out


def b64(path, w=300):
    im = Image.open(path).convert("RGB"); im.thumbnail((w, w * 3))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=76)
    return base64.b64encode(buf.getvalue()).decode()


def caption_of(rec):
    p = rec.get("prompt", "")
    if not p:
        return ""
    try:
        return json.loads(p).get("ai_caption", p) if p.strip().startswith("{") else p
    except Exception:
        return p


def sparkline(vals, w=260, h=34):
    """Distance-to-goal over the rollout — the shape says whether it converged or drifted."""
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys); rng = (hi - lo) or 1
    step = w / max(max(xs), 1)
    d = " ".join(f"{'M' if k == 0 else 'L'}{x*step:.1f},{h - (y-lo)/rng*(h-6) - 3:.1f}"
                 for k, (x, y) in enumerate(pts))
    col = "#34a853" if ys[-1] < ys[0] else "#ea4335"
    return (f'<svg width="{w}" height="{h}" class="spark"><path d="{d}" fill="none" '
            f'stroke="{col}" stroke-width="2"/></svg>')


cards = []
n_ok = imp_o = imp_b = imp_e = 0
for r in ROWS:
    traj = r.get("trajectory") or []
    steps = r.get("step_frames") or []
    g = r["goal"]
    s, a = r.get("start"), r.get("achieved")
    if s and a:
        n_ok += 1
        imp_o += a["d_occ"] < s["d_occ"]
        if s["d_bear"] is not None and a["d_bear"] is not None: imp_b += a["d_bear"] < s["d_bear"]
        if s["d_elev"] is not None and a["d_elev"] is not None: imp_e += a["d_elev"] < s["d_elev"]

    # reference frame first, then the rollout strip
    cells = [f'<div class="fr ref"><div class="lbl">REFERENCE<br>goal read from here</div>'
             f'<img src="data:image/jpeg;base64,{b64(r["reference"])}"></div>']
    for i, (p, m) in enumerate(zip(steps, traj)):
        if not os.path.exists(p):
            continue
        lab = "start" if i == 0 else f"chunk {i}"
        if m is None:
            info = '<div class="mi">subject not readable</div>'
        else:
            info = (f'<div class="mi">occ {m["occupancy"]:.0f}% '
                    f'<span class="dd">Δ{m["d_occ"]:.0f}</span></div>')
        cells.append(f'<div class="fr"><div class="lbl">{lab}</div>'
                     f'<img src="data:image/jpeg;base64,{b64(p)}">{info}</div>')

    occ_curve = sparkline([m["d_occ"] if m else None for m in traj])
    bear_curve = sparkline([m["d_bear"] if m else None for m in traj])
    if s and a:
        summary = (f'<table class="cmp"><tr><th>distance to requested shot</th><th>start</th>'
                   f'<th>end</th><th>over the rollout</th></tr>'
                   f'<tr><td>|Δ occupancy| (%)</td><td>{s["d_occ"]:.0f}</td>'
                   f'<td class="{"ok" if a["d_occ"]<s["d_occ"] else "no"}">{a["d_occ"]:.0f}</td>'
                   f'<td>{occ_curve}</td></tr>'
                   f'<tr><td>|Δ bearing| (°)</td><td>{s["d_bear"] if s["d_bear"] is not None else "—"}</td>'
                   f'<td class="{"ok" if (a["d_bear"] is not None and s["d_bear"] is not None and a["d_bear"]<s["d_bear"]) else "no"}">'
                   f'{a["d_bear"] if a["d_bear"] is not None else "—"}</td><td>{bear_curve}</td></tr>'
                   f'<tr><td>|Δ elevation| (°)</td><td>{s["d_elev"] if s["d_elev"] is not None else "—"}</td>'
                   f'<td class="{"ok" if (a["d_elev"] is not None and s["d_elev"] is not None and a["d_elev"]<s["d_elev"]) else "no"}">'
                   f'{a["d_elev"] if a["d_elev"] is not None else "—"}</td><td></td></tr></table>')
    else:
        summary = '<div class="warn">the detector could not read the subject in these frames</div>'

    cards.append(f"""<div class="card">
<div class="hd">{r['ref_object'][:32]} <span class="arrow">→ re-shot in</span> {r['target'][:32]}
<span class="req">requested: {g['occupancy']:.0f}% · bearing {g['bearing']:.0f}° · elev {g['elevation']:.0f}°</span></div>
<div class="goal">{caption_of(RECON.get(r['reference'], {}))[:230]}</div>
<div class="strip">{''.join(cells)}</div>
{summary}</div>""")

html = f"""<title>v12 — reference composition, re-shot by the policy (full rollout)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f1115;color:#e8eaed;margin:0;padding:24px;max-width:1500px;margin:auto}}
h1{{font-size:21px}} .sub{{color:#9aa0a6;font-size:13px;line-height:1.55}}
.card{{background:#181b21;border:1px solid #2a2e35;border-radius:10px;padding:14px;margin:16px 0}}
.hd{{font-weight:600;font-size:14px}} .arrow{{color:#8ab4f8;font-weight:400}}
.req{{float:right;color:#8ab4f8;font-size:12px;font-weight:400}}
.goal{{color:#9aa0a6;font-size:12px;margin:6px 0 10px;line-height:1.4}}
.strip{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}}
.fr img{{width:100%;border-radius:5px;display:block}}
.fr.ref img{{outline:2px solid #8ab4f8}}
.lbl{{font-size:11px;color:#9aa0a6;margin-bottom:4px}}
.mi{{font-size:11px;color:#c9d1d9;margin-top:3px}} .dd{{color:#8ab4f8}}
.spark{{display:block}}
table.cmp{{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}}
table.cmp th{{text-align:left;color:#7a828c;font-weight:500}} table.cmp td{{padding:2px 6px}}
td.ok{{color:#34a853;font-weight:600}} td.no{{color:#ea4335;font-weight:600}}
.warn{{color:#fbbc04;font-size:12px;margin-top:8px}}
.metrics{{display:flex;gap:14px;margin:12px 0}}
.metric{{background:#1a1d23;border:1px solid #2a2e35;border-radius:10px;padding:10px 16px}}
.metric b{{font-size:20px;color:#8ab4f8}} .metric span{{display:block;color:#9aa0a6;font-size:12px}}
</style>
<h1>The reference composition, re-shot by the policy — full rollout</h1>
<p class="sub">The goal is read off the reference image by <b>Module 2 (pixels only)</b>, serialized into the prompt the policy was trained on, and handed to the trained Cosmos 3 camera policy, which drives a camera in a <b>different scene with a different subject</b>. Every frame after the reference is a real render from that rollout — one per executed action chunk — and every number is Module 2 re-reading that frame, so nothing consults the camera pose. This replaces the earlier retrieval-based "recon".</p>
<div class="metrics">
<div class="metric"><b>{imp_o}/{n_ok}</b><span>ended closer in shot size</span></div>
<div class="metric"><b>{imp_b}/{n_ok}</b><span>ended closer in bearing</span></div>
<div class="metric"><b>{imp_e}/{n_ok}</b><span>ended closer in elevation</span></div>
<div class="metric"><b>{len(ROWS)}</b><span>cases · distinct scenes &amp; subjects</span></div>
</div>
<p class="sub">Checkpoint {_a.checkpoint_label}.  The sparkline is distance-to-goal across the rollout: green ends below where it started, red ends above.</p>
{''.join(cards)}
"""
open(OUT, "w").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB, {len(cards)} cases)")
