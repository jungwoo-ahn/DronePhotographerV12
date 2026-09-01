"""Self-contained HTML for the authored-goal renders — the pictures, side by side with the ask.

Two galleries, matching the two experiments in `scripts/render_authored_goals.py`:

  1. NATURAL LANGUAGE. One row per request: what the words were parsed into, then the same goal
     realised in several scenes, by both completers. Under each frame, the categories Module 2
     recovers from the pixels, marked against what was asked.
  2. REFERENCE IMAGE. The reference beside the same composition re-shot in other scenes with
     other subjects.

Everything shown is measured: the ACHIEVED column is the stage-3 scorer on the solved pose (so
it is what the shot geometrically is), the READ column is Module 2 on the render (what a
detector recovers from pixels, the only thing available for a real photo).

    python scripts/build_goal_render_report.py --in runs/goal_render/measured.json
venv: .venv-analysis
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
os.chdir(V12)

from src.common.facing import sector3, sector8  # noqa: E402
from src.goal_authoring.goal_bank import band_of  # noqa: E402
from src.goal_authoring.serialize import crop_phrase  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="runs/goal_render/measured.json")
ap.add_argument("--out", default="runs/goal_render_report.html")
ap.add_argument("--width", type=int, default=430)
args = ap.parse_args()

AXES = ("shot_size", "bearing", "elevation", "placement_x", "placement_y", "crop_side")


def b64(path: str, width: int) -> str | None:
    from PIL import Image
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    im.thumbnail((width, width * 3))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def words(profile: dict, bearing_key="subject_bearing_deg") -> dict:
    """Category words for a profile — the same tables the prompt is built from."""
    out = {}
    if "occupancy" in profile:
        out["shot_size"] = band_of("occupancy", profile["occupancy"])
    if bearing_key in profile:
        out["bearing"] = sector8(profile[bearing_key])
    if "cam_to_obj_elevation_deg" in profile:
        out["elevation"] = band_of("cam_to_obj_elevation_deg", profile["cam_to_obj_elevation_deg"])
    if "object_center_x" in profile:
        out["placement_x"] = band_of("object_center_x", profile["object_center_x"])
    if "object_center_y" in profile:
        out["placement_y"] = band_of("object_center_y", profile["object_center_y"])
    if "top_cut_frac" in profile or "bot_cut_frac" in profile:
        out["crop_side"] = crop_phrase(profile.get("top_cut_frac", 0.0), profile.get("bot_cut_frac", 0.0))
    return out


def chips(d: dict, ref: dict | None = None) -> str:
    out = []
    for k, v in d.items():
        cls = "chip"
        if ref is not None and k in ref:
            cls += " ok" if ref[k] == v else " no"
        out.append(f'<span class="{cls}">{html.escape(str(v))}</span>')
    return f'<span class="chips">{"".join(out)}</span>'


def line(label: str, body: str) -> str:
    return f'<div class="line"><span class="k">{label}</span>{body}</div>'



def frame_card(j: dict, asked: dict, width: int) -> str:
    img = b64(j["frame"], width) if Path(j["frame"]).exists() else None
    if img is None:
        return (f'<div class="frame"><div class="missing">no frame — '
                f'{html.escape(j.get("error", "not rendered"))}</div></div>')
    ach = words(j["achieved"])
    read = words(j.get("read", {})) if j.get("read_detected") else None
    notes = []
    if j.get("pose_underdetermined"):
        notes.append("cut at both ends — this goal does not pin the camera")
    if j.get("nearest_recorded_camera_m", 0) > 4.0:
        notes.append(f'camera stands {j["nearest_recorded_camera_m"]:.1f} m from anywhere the '
                     "sampler ever put one")
    notes += [w for w in j.get("warnings", [])[:2]]
    scene = j["placement"].split("__")[0].replace("-", " ")[:26]
    return f"""<div class="frame">
  <img src="data:image/jpeg;base64,{img}" alt="rendered frame">
  <div class="cap">
    <div class="slate"><span class="m">{html.escape(j["method"])}</span>
      <span>{html.escape(j["object"][:22])}</span><span>·</span><span>{html.escape(scene)}</span></div>
    {line("achieved", chips(ach, asked))}
    {line("read back", chips(read, ach) if read else '<span class="chips"><span class="chip no">no subject detected</span></span>')}
    {''.join(f'<span class="note">{html.escape(n)}</span>' for n in notes)}
  </div>
</div>"""


def main() -> int:
    doc = json.loads(Path(args.inp).read_text())
    jobs = [j for j in doc["jobs"] if "error" not in j]
    refs = {i: r for i, r in enumerate(doc["references"])}

    # ---------------- experiment A ----------------
    by_request: dict[str, list] = defaultdict(list)
    for j in jobs:
        if j["kind"] == "nl":
            by_request[j["request"]].append(j)
    nl_html = []
    for req, group in by_request.items():
        group.sort(key=lambda j: (j["method"], j["tag"]))
        asked = {k: v for k, v in group[0]["categories"].items() if k in AXES}
        parsed = group[0].get("parsed", {})
        cards = "".join(frame_card(j, asked, args.width) for j in group)
        unstated = [a for a in AXES if a not in parsed]
        nl_html.append(f"""<div class="sheet">
  <div class="rail">
    <p class="ask">{html.escape(req)}</p>
    <div class="k">stated</div>{chips(parsed)}
    <div class="k">left open</div>
    <span class="chips">{''.join(f'<span class="chip">{html.escape(a)}</span>' for a in unstated) or '<span class="chip">nothing</span>'}</span>
    <div class="prompt">{html.escape(group[0]["prompt"])}</div>
  </div>
  <div class="strip">{cards}</div>
</div>""")

    # ---------------- experiment B ----------------
    by_ref: dict[int, list] = defaultdict(list)
    for j in jobs:
        if j["kind"] == "ref":
            by_ref[j["ref_index"]].append(j)
    ref_html = []
    for ri, group in sorted(by_ref.items()):
        r = refs[ri]
        group.sort(key=lambda j: (j["method"], j["tag"]))
        gt = dict(r["gt"])
        gt["subject_bearing_deg"] = r["gt_bearing"]
        asked = {k: v for k, v in words(gt).items() if k in AXES}
        img = b64(r["image"], args.width)
        meas = r.get("measured_categories", {})
        cards = "".join(frame_card(j, asked, args.width) for j in group)
        scene = r["placement"].split("__")[0].replace("-", " ")[:26]
        ref_html.append(f"""<div class="sheet">
  <div class="rail">
    <p class="ask">the same composition, somewhere else</p>
    <div class="k">source shot</div>{chips(asked)}
    <div class="k">module 2 reads</div>{chips({k: v for k, v in meas.items() if k in AXES}, asked)}
    <div class="prompt">{html.escape(group[0]["prompt"]) if group else ''}</div>
  </div>
  <div class="strip">
    <div class="frame source">
      <img src="data:image/jpeg;base64,{img}" alt="reference photograph">
      <div class="cap"><div class="slate"><span class="m">source</span>
        <span>{html.escape(r["object"][:22])}</span><span>·</span><span>{html.escape(scene)}</span></div>
        {line("shot", chips(asked))}
      </div>
    </div>
    {cards}
  </div>
</div>""")

    # ---------------- headline numbers ----------------
    def agree(js, key):
        n = ok = 0
        for j in js:
            a = words(j["achieved"])
            r = words(j.get("read", {})) if j.get("read_detected") else None
            if not r or key not in a or key not in r:
                continue
            n += 1
            ok += a[key] == r[key]
        return ok, n

    nl_jobs = [j for j in jobs if j["kind"] == "nl"]
    ref_jobs = [j for j in jobs if j["kind"] == "ref"]
    det = sum(1 for j in jobs if j.get("read_detected"))
    rows = []
    for ax in AXES:
        o1, n1 = agree(nl_jobs, ax)
        o2, n2 = agree(ref_jobs, ax)
        rows.append(f"<tr><td>{ax.replace('_', ' ')}</td><td class='num'>{o1}/{n1}</td>"
                    f"<td class='num'>{o2}/{n2}</td></tr>")

    style = """
:root{
  --paper:#F4F3F0; --surface:#FFFFFF; --sunk:#EDEBE6; --ink:#191B1F; --muted:#6C7178;
  --line:#DFDDD8; --accent:#2C5F63; --ok:#2C6A45; --ok-bg:#E4EFE7; --no:#A63A31; --no-bg:#F6E4E1;
  --flag:#7A5B12; --flag-bg:#F4EBD5;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#131519; --surface:#1A1D23; --sunk:#15181D; --ink:#E6E8EC; --muted:#8F959E;
    --line:#2A2E36; --accent:#79B0AD; --ok:#7BD196; --ok-bg:#16301F; --no:#EF9A93; --no-bg:#361D1B;
    --flag:#DBBB6A; --flag-bg:#2B2413;
  }
}
:root[data-theme="dark"]{
  --paper:#131519; --surface:#1A1D23; --sunk:#15181D; --ink:#E6E8EC; --muted:#8F959E;
  --line:#2A2E36; --accent:#79B0AD; --ok:#7BD196; --ok-bg:#16301F; --no:#EF9A93; --no-bg:#361D1B;
  --flag:#DBBB6A; --flag-bg:#2B2413;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:0;
  font-family:"Source Sans 3",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1560px;margin:0 auto;padding:44px 28px 80px}
h1,h2,h3,.eyebrow,.ask{font-family:Archivo,"Source Sans 3",system-ui,sans-serif}
h1{font-size:clamp(28px,3.4vw,42px);line-height:1.08;letter-spacing:-.022em;font-weight:700;
  margin:0 0 14px;max-width:19ch;text-wrap:balance}
h2{font-size:21px;letter-spacing:-.012em;font-weight:650;margin:0 0 4px}
.eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
  font-weight:640;margin:0 0 10px}
.lede{color:var(--muted);max-width:70ch;margin:0 0 6px}
.lede b{color:var(--ink);font-weight:600}
.rule{height:1px;background:var(--line);margin:38px 0 26px}
section{margin-top:52px}
.sechead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:8px}
.sechead .count{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}

/* --- summary table --- */
table{border-collapse:collapse;font-size:14px;margin:16px 0 0;font-variant-numeric:tabular-nums}
th{font-family:Archivo,sans-serif;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);font-weight:640;text-align:left;padding:0 22px 7px 0;border-bottom:1px solid var(--line)}
td{padding:6px 22px 6px 0;border-bottom:1px solid var(--line)}
td:first-child{color:var(--muted)}
td.num{font-family:"IBM Plex Mono",monospace;font-size:13px}

/* --- contact sheet --- */
.sheet{display:grid;grid-template-columns:260px 1fr;gap:26px;padding:20px 0 24px;
  border-top:1px solid var(--line)}
.sheet:last-child{border-bottom:1px solid var(--line)}
.rail{position:sticky;top:20px;align-self:start}
.ask{font-size:17px;font-weight:640;letter-spacing:-.01em;line-height:1.3;margin:0 0 10px;
  text-wrap:balance}
.ask::before{content:"\201C"} .ask::after{content:"\201D"}
.rail .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  font-family:Archivo,sans-serif;font-weight:640;margin:12px 0 4px}
.prompt{font-family:"IBM Plex Mono",monospace;font-size:10.5px;line-height:1.5;color:var(--muted);
  background:var(--sunk);border:1px solid var(--line);border-radius:6px;padding:8px 9px;
  margin-top:12px;max-height:190px;overflow:auto}
.strip{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.frame{flex:1 1 320px;max-width:440px;background:var(--surface);border:1px solid var(--line);
  border-radius:9px;overflow:hidden}
.frame.source{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.frame img{width:100%;display:block;background:var(--sunk)}
.cap{padding:9px 11px 11px}
.slate{display:flex;align-items:baseline;gap:8px;font-size:11px;color:var(--muted);
  font-family:"IBM Plex Mono",monospace;margin-bottom:7px}
.slate .m{font-family:Archivo,sans-serif;font-size:11px;font-weight:650;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink)}
.frame.source .slate .m{color:var(--accent)}
.line{display:flex;gap:6px;align-items:flex-start;margin:3px 0}
.line .k{flex:0 0 62px;font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-family:Archivo,sans-serif;font-weight:640;padding-top:3px}
.chips{display:flex;gap:4px;flex-wrap:wrap}
.chip{font-size:11px;padding:1px 7px;border-radius:4px;background:var(--sunk);
  border:1px solid var(--line);color:var(--ink)}
.chip.ok{background:var(--ok-bg);border-color:transparent;color:var(--ok)}
.chip.no{background:var(--no-bg);border-color:transparent;color:var(--no)}
.note{display:block;font-size:10.5px;color:var(--flag);background:var(--flag-bg);
  border-radius:4px;padding:3px 7px;margin-top:5px}
.missing{padding:26px;font-size:12px;color:var(--no)}
@media (max-width:900px){
  .sheet{grid-template-columns:1fr;gap:14px}
  .rail{position:static}
}
"""
    head = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
            'family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&'
            'family=Source+Sans+3:wght@400;600&display=swap">')
    out = f"""<title>Authored Shot Sheets</title>
{head}
<style>{style}</style>
<div class="wrap">
<p class="eyebrow">goal authoring · held-out scenes</p>
<h1>Does an authored goal look like the shot that was asked for?</h1>
<p class="lede">No policy here. A goal is a shot profile; the solver inverts the stage-3 scorer to
the camera pose that realises it in a given placement, and Blender renders that pose. These frames
grade the <b>goal</b> — the words, and the reference photograph — not the policy that would have to
fly there. Every target scene is one of the eight held out from training.</p>
<p class="lede">A goal is scene-agnostic; a scene is not goal-agnostic. Each goal is solved in every
candidate placement and shot where the camera it asks for stands closest to one the data generator
had already ray-cast as valid — choosing a location for a fixed shot, not altering the shot. Without
that step a basement placement put the camera on a sight line through a wall.</p>
<p class="lede"><b>achieved</b> is the scorer on the solved pose: what the shot geometrically is.
<b>read back</b> is Module 2 run on the render — the only measurement a real photograph allows.
Green means it matches what was asked (top line) or what was achieved (bottom line).</p>
<table>
<tr><th>axis</th><th>natural language</th><th>reference transfer</th></tr>
{''.join(rows)}
</table>
<p class="lede" style="margin-top:10px">A person was detected in {det} of {len(jobs)} renders. For
scale: on 80 sampled goal frames from the held-out scenes <b>the training data itself</b> fails that
test 30&nbsp;% of the time — the profile is computed by projecting the subject's mesh, which cannot
see the wall in front of it.</p>

<section>
<div class="sechead"><h2>Natural language → goal → frame</h2>
<span class="count">{len(nl_html)} requests · {len(nl_jobs)} frames</span></div>
<p class="lede">Each request is realised in different scenes with different subjects, under both
completers. The words state a few axes; everything else has to be filled in — <b>bank</b> anchors
on a real training goal, <b>rules</b> derives it from person-box geometry.</p>
{''.join(nl_html)}
</section>

<section>
<div class="sechead"><h2>Reference photograph → goal → another scene</h2>
<span class="count">{len(ref_html)} references · {len(ref_jobs)} frames</span></div>
<p class="lede">The outlined frame is the source. The rest are the same composition solved and
rendered somewhere else entirely: different room, different person.</p>
{''.join(ref_html)}
</section>
</div>
"""
    Path(args.out).write_text(out)
    print(f"wrote {args.out} ({os.path.getsize(args.out)//1024} KB, "
          f"{len(nl_html)} requests, {len(ref_html)} references)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
