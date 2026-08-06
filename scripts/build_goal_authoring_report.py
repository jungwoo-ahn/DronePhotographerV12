"""Build a self-contained HTML report exercising both goal-authoring modules broadly.

Module 1 (NL -> goal): a large diverse prompt set (parsed -> categories -> grounded profile -> prompt)
  + a round-trip accuracy metric (known profile -> serialize -> parse -> recover categories).
Module 2 (ref image -> goal): run on many renders; per card show the reference with an OVERLAY of what
  was extracted, the recovered goal vs the GT categories (with match marks), and a COMPOSITION-TRANSFER
  "recon" — the nearest-composition render from a DIFFERENT scene/subject (what this goal transfers to).
venv: .venv-analysis (GPU for YOLO). Writes runs/goal_authoring_report.html."""
import base64, io, json, os, random, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common.facing import front_azimuth, sector3, sector8
from src.common.goal_space import DEFAULT_V5_RANGES, SUBJECT_BEARING_KEY, CYCLIC_GOAL_KEYS
from src.goal_authoring import vocab
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.from_language import keyword_classifier, language_to_goal
from src.goal_authoring.from_reference import ReferenceEstimator
from src.scoring.bbox_control import compute_v5_scores

ROOT = "data/trajectories"
OUT = "runs/goal_authoring_report.html"

# ============================ MODULE 1: NL -> goal ============================
NL_PROMPTS = [
    "a dramatic low-angle close-up of the subject from the side",
    "wide establishing shot from behind, subject small in the lower left",
    "medium shot at eye level, facing the camera, centered",
    "shoot them from above, three-quarter front-right, full body",
    "an intimate portrait, tight on the face",
    "back view, subject in the upper right",
    "extreme wide landscape shot, tiny subject",
    "a heroic low angle from below, looking up at them",
    "profile shot from the left, medium framing",
    "high-angle bird's-eye looking straight down, centered",
    "close-up from the front, eye level",
    "full-length shot, subject on the right third",
    "over-the-shoulder from behind, upper left",
    "a headshot facing the camera",
    "medium-wide shot from the right side, lower frame",
    "waist shot, three-quarter front-left",
    "rear view, full body, centered",
    "tight close-up, high angle looking down",
    "wide shot, subject facing away, bottom of frame",
    "eye-level medium shot from the front-right",
    "just a close-up",
    "from behind",
    "shoot from below, on the left",
    "portrait from the side, upper third",
    "establishing shot, subject in the distance, lower right",
    "extreme close-up on the face, front",
    "low angle full body from the front",
    "high angle, subject small, centered",
    "medium close-up, right profile",
    "the back of the subject, from above",
]


def chips(cats):
    return "".join(f'<span class="chip">{k}: <b>{v}</b></span>' for k, v in cats.items())


def module1_cards():
    cards = []
    for p in NL_PROMPTS:
        gp = language_to_goal(p, keyword_classifier)
        cards.append(f"""<div class="card m1">
<div class="nl">“{p}”</div>
<div class="chips">{chips(gp.categories()) or '<span class=chip>(none parsed)</span>'}</div>
<div class="prompt">→ {gp.to_nl()}</div></div>""")
    return "\n".join(cards)


def module1_roundtrip(profiles):
    """known profile -> serialize NL -> parse -> did we recover the same categories?"""
    ok = tot = 0
    per_axis_ok = {}; per_axis_tot = {}
    for prof in profiles:
        gp0 = GoalProfile.from_full_profile(prof)
        cats0 = gp0.categories()
        cats1 = language_to_goal(gp0.to_nl(), keyword_classifier, project_feasible=False).categories()
        for ax, lb in cats0.items():
            per_axis_tot[ax] = per_axis_tot.get(ax, 0) + 1
            if cats1.get(ax) == lb:
                per_axis_ok[ax] = per_axis_ok.get(ax, 0) + 1
                ok += 1
            tot += 1
    overall = 100 * ok / max(tot, 1)
    axes = {ax: 100 * per_axis_ok.get(ax, 0) / per_axis_tot[ax] for ax in per_axis_tot}
    return overall, axes


# ============================ MODULE 2: ref image -> goal ============================
def visible_gt(bb, W, H, az, el):
    return compute_v5_scores(int(W), int(H), [float(v) for v in bb], float(az), float(el))


def prof_distance(a, b, keys):
    """normalized distance over shared keys (cyclic for bearing)."""
    d = 0.0; n = 0
    for k in keys:
        if k not in a or k not in b:
            continue
        lo, hi = DEFAULT_V5_RANGES.get(k, (0, 1)); rng = (hi - lo) or 1
        if k in CYCLIC_GOAL_KEYS:
            diff = abs(((a[k] - b[k] + 180) % 360) - 180) / 180.0
        else:
            diff = abs(a[k] - b[k]) / rng
        d += diff * diff; n += 1
    return (d / n) ** 0.5 if n else 9.9


def gt_categories(prof, W, H):
    cats = {}
    cats["shot_size"] = vocab._classify(prof["occupancy"], vocab.SHOT_SIZE)
    cats["placement_x"] = vocab._classify(prof["object_center_x"] / W, vocab.PLACE_X)
    cats["placement_y"] = vocab._classify(prof["object_center_y"] / H, vocab.PLACE_Y)
    if "cam_to_obj_elevation_deg" in prof:
        cats["elevation"] = vocab._classify(prof["cam_to_obj_elevation_deg"], vocab.ELEVATION)
    return cats


def b64(im, w=300, q=72):
    im = im.convert("RGB"); im.thumbnail((w, w * 3))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def overlay(imgpath, det, gp):
    im = Image.open(imgpath).convert("RGB"); dr = ImageDraw.Draw(im, "RGBA")
    if det is not None:
        b = det[0]
        dr.rectangle([b[0], b[1], b[2], b[3]], outline=(52, 168, 83, 255), width=4)
    label = " · ".join(v for v in gp.categories().values())
    dr.rectangle([0, 0, im.width, 26], fill=(0, 0, 0, 150))
    dr.text((6, 6), label[:70], fill=(255, 255, 255, 255))
    return im


def main():
    est = ReferenceEstimator()
    dirs = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    random.seed(7); random.shuffle(dirs)

    # ---- build a composition INDEX (GT visible profiles) for transfer recon + metrics ----
    index = []      # (profile, categories, object, imgpath, W, H)
    prof_pool = []  # profiles for module-1 round-trip
    scanned = 0
    for dn in dirs:
        if len(index) > 600: break
        obj = dn.split("__", 1)[1] if "__" in dn else dn
        if front_azimuth(obj) is None: continue
        try: d = json.load(open(os.path.join(ROOT, dn, "data.json")))
        except Exception: continue
        scanned += 1
        W, H = d.get("render_width", 1024), d.get("render_height", 768)
        for pair in d.get("render_records", []):
            for r in pair:
                s = r.get("scores"); bb = r.get("bbox_xyxy_full")
                if not s or not r.get("in_frame") or not bb: continue
                if not (30 <= s["occupancy"] <= 92 and s["body_in_frame_ratio"] >= 45): continue
                gt = visible_gt(bb, W, H, s["cam_to_obj_azimuth_deg"], s["cam_to_obj_elevation_deg"])
                gt["occupancy"] = s["occupancy"]
                gt[SUBJECT_BEARING_KEY] = (front_azimuth(obj) - s["cam_to_obj_azimuth_deg"]) % 360
                gt["cam_to_obj_elevation_deg"] = float(s["cam_to_obj_elevation_deg"])
                index.append((gt, gt_categories(gt, W, H), obj, os.path.join(ROOT, dn, r["path_rel"]), W, H))
                prof_pool.append({k: gt[k] for k in ("occupancy", "object_center_x", "object_center_y",
                                                     "cam_to_obj_elevation_deg", SUBJECT_BEARING_KEY)})
        if scanned > 350: break
    print(f"index: {len(index)} renders; module-1 round-trip pool: {len(prof_pool)}", flush=True)

    # ---- module 1 ----
    m1_html = module1_cards()
    gold = json.load(open("runs/nl_paraphrase_results.json")) if os.path.exists("runs/nl_paraphrase_results.json") else {}
    rt_overall, rt_axes = module1_roundtrip(random.sample(prof_pool, min(200, len(prof_pool))))

    # ---- module 2: queries (stratified) + metrics + cards ----
    tkeys = ("occupancy", "object_center_x", "object_center_y", SUBJECT_BEARING_KEY)
    # metrics over a large random query sample
    metric_idx = random.sample(range(len(index)), min(150, len(index)))
    bear_ok = px_ok = py_ok = shot_ok = elev_ok = m = bn = 0
    elev_err = []
    for i in metric_idx:
        gt, gcats, obj, ip, W, H = index[i]
        gp = est(ip)
        if "occupancy" not in gp.specified: continue
        m += 1
        rc = gp.categories()
        shot_ok += (rc.get("shot_size") == gcats["shot_size"])
        px_ok += (rc.get("placement_x") == gcats["placement_x"])
        py_ok += (rc.get("placement_y") == gcats["placement_y"])
        if SUBJECT_BEARING_KEY in gp.specified:
            bn += 1; bear_ok += (sector3(gp.values[SUBJECT_BEARING_KEY]) == sector3(gt[SUBJECT_BEARING_KEY]))
        if "cam_to_obj_elevation_deg" in gp.specified:
            elev_ok += (rc.get("elevation") == gcats.get("elevation"))
            elev_err.append(abs(gp.values["cam_to_obj_elevation_deg"] - gt["cam_to_obj_elevation_deg"]))
    metrics = dict(n=m, bearing=100*bear_ok/max(bn,1), placement_x=100*px_ok/max(m,1),
                   placement_y=100*py_ok/max(m,1), shot=100*shot_ok/max(m,1),
                   elev=100*elev_ok/max(m,1), elev_mae=(float(np.mean(elev_err)) if elev_err else float("nan")))
    print(f"module2 metrics: {metrics}", flush=True)

    # cards: round-robin over (shot_size, sector3) cells for diversity, up to 30
    from collections import defaultdict
    from src.goal_authoring.from_reference import profile_from_detection
    cells = defaultdict(list)
    for i in metric_idx:
        gt, gcats, obj, ip, W, H = index[i]
        cells[(gcats["shot_size"], sector3(gt[SUBJECT_BEARING_KEY]))].append(i)
    card_ids, rnd = [], 0
    while len(card_ids) < 30 and any(len(v) > rnd for v in cells.values()):
        for v in cells.values():
            if len(v) > rnd:
                card_ids.append(v[rnd])
            if len(card_ids) >= 30:
                break
        rnd += 1
    m2_cards = []
    for i in card_ids:
        gt, gcats, obj, ip, W, H = index[i]
        det = est.detect_main_subject(ip)            # single YOLO call per card
        if det is None: continue
        bbox, kp, Wd, Hd = det
        gp = profile_from_detection(bbox, kp, Wd, Hd, est.bearing)
        if "occupancy" not in gp.specified: continue
        rc = gp.categories()
        # recovered vs GT chips with match marks
        rows = []
        for ax, glab in gcats.items():
            rlab = rc.get(ax, "—"); mark = "✓" if rlab == glab else "✗"
            cls = "ok" if rlab == glab else "no"
            rows.append(f'<tr><td>{ax}</td><td class="{cls}">{rlab} {mark}</td><td class="gt">{glab}</td></tr>')
        if SUBJECT_BEARING_KEY in gp.specified:
            gb, rb = sector3(gt[SUBJECT_BEARING_KEY]), sector3(gp.values[SUBJECT_BEARING_KEY])
            mk = "✓" if gb == rb else "✗"; cls = "ok" if gb == rb else "no"
            rows.append(f'<tr><td>bearing</td><td class="{cls}">{sector8(gp.values[SUBJECT_BEARING_KEY])} {mk}</td><td class="gt">{sector8(gt[SUBJECT_BEARING_KEY])}</td></tr>')
        # transfer recon: nearest composition from a DIFFERENT subject
        q = {k: gp.values[k] for k in tkeys if k in gp.values}
        best = min((e for e in index if e[2] != obj), key=lambda e: prof_distance(q, e[0], tkeys), default=None)
        recon_img = f'<img src="data:image/jpeg;base64,{b64(Image.open(best[3]),260)}"><div class="cap">{best[2][:26]}</div>' if best else ""
        m2_cards.append(f"""<div class="card m2">
<div class="pair">
  <div class="col"><div class="lbl">reference (overlay)</div><img src="data:image/jpeg;base64,{b64(overlay(ip, det, gp),260)}"></div>
  <div class="col"><div class="lbl">recon: same composition, other subject</div>{recon_img}</div>
</div>
<div class="prompt">→ {gp.to_nl()}</div>
<table class="cmp"><tr><th></th><th>recovered</th><th>GT</th></tr>{''.join(rows)}</table></div>""")

    # ---- assemble HTML ----
    ax_html = " · ".join(f"{a} {v:.0f}%" for a, v in sorted(rt_axes.items()))
    _names = {"keyword": "keyword rules (no model)", "llm": "LLM only", "hybrid": "hybrid (deployed default)"}
    clf_rows = "".join(
        f'<tr><td>{_names.get(k,k)}</td><td class="{"ok" if k=="hybrid" else ""}">{v["recall"]:.1f}%</td>'
        f'<td>{v["exact"]:.1f}%</td><td>{v["hallucinated"]:.1f}%</td></tr>'
        for k, v in gold.items()) or '<tr><td colspan=4>run scripts/bench_nl_paraphrase.py --eval-llm</td></tr>' 
    html = f"""<title>DronePhotographer v12 — goal authoring report</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0f1115;color:#e8eaed;margin:0;padding:24px;max-width:1400px;margin:auto}}
h1{{font-size:22px}} h2{{font-size:18px;margin-top:32px;border-bottom:1px solid #2a2e35;padding-bottom:6px}}
.sub{{color:#9aa0a6;font-size:13px}} .metrics{{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}}
.metric{{background:#1a1d23;border:1px solid #2a2e35;border-radius:10px;padding:10px 16px}}
.metric b{{font-size:22px;color:#8ab4f8}} .metric span{{display:block;color:#9aa0a6;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;margin-top:14px}}
.card{{background:#181b21;border:1px solid #2a2e35;border-radius:10px;padding:12px}}
.card.m1 .nl{{font-size:14px;font-weight:600}} .chips{{margin:8px 0}}
.chip{{display:inline-block;background:#222733;border-radius:12px;padding:2px 9px;margin:2px;font-size:11px;color:#c9d1d9}}
.prompt{{color:#9aa0a6;font-size:12px;margin-top:6px;line-height:1.4}}
.pair{{display:flex;gap:8px}} .col{{flex:1}} .col img{{width:100%;border-radius:6px;display:block}}
.lbl{{font-size:10px;color:#7a828c;margin-bottom:3px}} .cap{{font-size:10px;color:#7a828c}}
table.cmp{{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}}
table.cmp th{{text-align:left;color:#7a828c;font-weight:500}} table.cmp td{{padding:1px 4px}}
td.ok{{color:#34a853}} td.no{{color:#ea4335}} td.gt{{color:#8ab4f8}}
</style>
<h1>DronePhotographer v12 — goal authoring modules</h1>
<p class="sub">Turning a user's <b>natural language</b> or a <b>reference image</b> into the structured shot-profile goal that conditions the policy. Both target the same cinematography vocabulary; elevation is left to the user for reference images (not single-image-recoverable).</p>

<h2>1 · Natural language → goal &nbsp;<span class="sub">({len(NL_PROMPTS)} prompts)</span></h2>
<p class="sub">Scored on a <b>hand-labelled gold set of 40 user-style requests</b> (assets/nl_gold_set.json) — written the way a photographer talks, labelled with the intended attributes. This replaces an earlier round-trip metric that was <i>circular</i> (it fed our serializer's own phrasing back to the parser that was written for it) and therefore inflated: the same keyword parser scores {rt_overall:.0f}% on round-trip but {gold.get('keyword',{}).get('recall',0):.0f}% on real phrasings.</p>
<table class="cmp big"><tr><th>classifier</th><th>attribute recall</th><th>all attributes correct</th><th>hallucinated attributes ↓</th></tr>
{clf_rows}
</table>
<p class="sub">A <b>hallucinated</b> attribute is one the request never stated — in a partial-goal design that silently over-constrains the shot, so it is worse than a miss. <b>hybrid</b> = deterministic keyword rules take precedence, the LLM fills only what they miss, and every LLM attribute must quote supporting words that are checked against the request (evidence grounding — this cut hallucination from 25% to 15%).</p>
<div class="grid">{m1_html}</div>

<h2>2 · Reference image → goal &nbsp;<span class="sub">(YOLO-pose framing + bearing; GT from our renders)</span></h2>
<div class="metrics">
<div class="metric"><b>{metrics['bearing']:.0f}%</b><span>bearing (front/side/back)</span></div>
<div class="metric"><b>{metrics['placement_x']:.0f}%</b><span>placement-x category</span></div>
<div class="metric"><b>{metrics['placement_y']:.0f}%</b><span>placement-y category</span></div>
<div class="metric"><b>{metrics['shot']:.0f}%</b><span>shot-size category</span></div>
<div class="metric"><b>{metrics['elev_mae']:.0f}°</b><span>elevation MAE ({metrics['elev']:.0f}% band)</span></div>
<div class="metric"><span>n renders</span><b style="font-size:15px">{metrics['n']}</b></div>
</div>
<p class="sub"><b>Elevation update:</b> a VLM was at chance on camera height (34–36% vs 33%), so it was originally left unspecified. The same body-pose keypoints that give bearing also encode camera pitch through <i>vertical foreshortening</i> — a regressor on them reaches MAE ≈8° (vs 13.7° for predict-median, corr 0.76), so elevation is now recovered too. Caveat: the source data is 68% high-angle / 3.5% low-angle, so low-angle references are extrapolation — and that same imbalance means a “heroic low angle” goal has almost no training support downstream.</p>
<p class="sub">Each card: the reference with an overlay of what was extracted · the recovered goal prompt · recovered-vs-GT categories (✓/✗) · a “recon” = the nearest-composition render from a <b>different scene/subject</b> — what this goal transfers to.</p>
<div class="grid">{''.join(m2_cards)}</div>
"""
    os.makedirs("runs", exist_ok=True); open(OUT, "w").write(html)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)//1024} KB, {len(m2_cards)} ref cards)")


if __name__ == "__main__":
    main()
