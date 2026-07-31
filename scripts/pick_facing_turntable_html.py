"""INTERACTIVE click-to-pick facing gallery. Same turntable contact sheets, but each thumbnail is
clickable: click the view that shows the subject's FRONT. Selections are pre-filled from the auto
pass, so you only click to CHANGE a wrong one or to set a no-face asset. A sticky bar tracks progress
and a 'Copy JSON' button (+ always-visible textarea fallback) gives the full {asset: front_az} map to
paste back to Claude in one shot.

Pure self-contained static HTML (no Artifact runtime capabilities needed) — works in the claude.ai
Artifact sandbox. .venv-analysis. Reads runs/facing_turntable/index.json + facing_turntable_auto.json.
Writes runs/facing_pick.html.
"""
import base64
import io
import json
import os
from html import escape

os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from PIL import Image

IDX = json.load(open("runs/facing_turntable/index.json"))
AUTO = (json.load(open("runs/facing_turntable_auto.json"))
        if os.path.exists("runs/facing_turntable_auto.json") else {})
OUT = "runs/facing_pick.html"

ORDER = {"render_error": 0, "too_few_faces": 0, None: 0, "VERIFY": 1, "OK": 2}
BADGE = {"OK": "#34a853", "VERIFY": "#fbbc04", "too_few_faces": "#ea4335", "render_error": "#9333ea"}


def thumb(ip, w=132):
    im = Image.open(ip).convert("RGB")
    im.thumbnail((w, int(w * im.height / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def sortkey(kv):
    a = AUTO.get(kv[0], {})
    return ORDER.get(a.get("conf", a.get("status")), 3)


rows = sorted(IDX.items(), key=sortkey)

auto_front = {}   # asset -> auto front_az (fine, 5deg) or None
cards = []
for obj, info in rows:
    a = AUTO.get(obj, {})
    conf = a.get("conf", a.get("status", "?"))
    auto_front[obj] = a.get("front_az")  # None for no-face / error
    views = info.get("views", [])
    cells = []
    for v in sorted(views, key=lambda x: x["az"]):
        az = int(round(v["az"]))
        cells.append(
            f'<div class="v" data-az="{az}">'
            f'<img loading="lazy" src="data:image/jpeg;base64,{thumb(v["path"])}">'
            f'<div class="az">{az}°</div><div class="chk">FRONT ✓</div></div>'
        )
    autotxt = (f"auto {a.get('front_az')}° · faces {a.get('n_detect', 0)}/{len(views)} · "
               f"contrast {a.get('contrast', '–')}") if a.get("front_az") is not None \
        else f"no auto ({a.get('status', '?')}) — pick front by eye"
    cards.append(
        f'<div class="card" data-asset="{escape(obj, quote=True)}" data-conf="{conf}">'
        f'<div class="hd"><span class="nm">{escape(obj[:52])}</span>'
        f'<span class="badge" style="background:{BADGE.get(conf, "#ea4335")}">{conf}</span>'
        f'<span class="pick">—</span></div>'
        f'<div class="meta">{autotxt}</div>'
        f'<div class="sheet">{"".join(cells)}</div></div>'
    )

AUTO_JS = json.dumps(auto_front)

html = f"""<title>v12 facing — click to pick front</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{font-family:system-ui,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:0 22px 60px}}
.bar{{position:sticky;top:0;z-index:20;background:#0f1114ee;backdrop-filter:blur(6px);
 border-bottom:1px solid #2a2e35;padding:12px 0;margin:0 -22px 8px;padding-left:22px;padding-right:22px}}
.bar h1{{font-size:15px;margin:0 0 8px}}
.row{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
button{{font:inherit;font-size:13px;font-weight:600;border:1px solid #3a4150;background:#222732;color:#e8eaed;
 border-radius:8px;padding:6px 12px;cursor:pointer}}
button:hover{{background:#2c333f}}
button.primary{{background:#1a73e8;border-color:#1a73e8;color:#fff}}
button.on{{background:#34506e;border-color:#5a86c0}}
.stat{{font-size:12.5px;color:#9aa0a6;font-variant-numeric:tabular-nums}}
.stat b{{color:#8ab4f8}} .stat .u{{color:#f6a}} .stat .c{{color:#fbbc04}}
textarea{{width:100%;height:44px;margin-top:8px;background:#0b0d10;color:#7fd88f;border:1px solid #2a2e35;
 border-radius:6px;font-family:ui-monospace,monospace;font-size:11px;padding:6px;resize:vertical}}
.card{{background:#1e2126;border:1px solid #2a2e35;border-radius:10px;padding:12px;margin:12px 0}}
.card.unset{{border-color:#ea4335}} .card.changed{{border-color:#34a853}}
.hd{{display:flex;justify-content:space-between;align-items:center;gap:10px}}
.nm{{font-weight:600;font-size:14px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.badge{{color:#111;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px}}
.pick{{font-size:13px;font-weight:700;color:#9aa0a6;min-width:120px;text-align:right;font-variant-numeric:tabular-nums}}
.pick.set{{color:#34a853}} .pick.unset{{color:#ff6a6a}} .pick.changed{{color:#7fd88f}}
.meta{{color:#8ab4f8;font-size:12px;margin:5px 0 9px;font-variant-numeric:tabular-nums}}
.sheet{{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}}
.v{{position:relative;border-radius:6px;flex:0 0 auto;cursor:pointer;border:3px solid #2a2e35}}
.v:hover{{border-color:#5a86c0}}
.v img{{display:block;border-radius:4px}}
.v .az{{position:absolute;bottom:2px;left:2px;background:#000a;color:#fff;font-size:10px;padding:1px 4px;border-radius:3px}}
.v .chk{{display:none;position:absolute;top:2px;right:2px;background:#34a853;color:#fff;font-size:10px;font-weight:700;padding:1px 5px;border-radius:3px}}
.v.chosen{{border-color:#34a853}} .v.chosen .chk{{display:block}}
</style>
<div class="bar">
  <h1>v12 facing — click the view that shows each subject's FRONT</h1>
  <div class="row">
    <button class="primary" id="copy">Copy JSON</button>
    <span class="stat" id="stat"></span>
    <span style="flex:1"></span>
    <button class="filt on" data-f="all">All</button>
    <button class="filt" data-f="review">Needs review</button>
    <button class="filt" data-f="unset">Unset only</button>
  </div>
  <textarea id="out" readonly onclick="this.select()"></textarea>
</div>
{''.join(cards)}
<script>
const AUTO = {AUTO_JS};              // asset -> auto front_az (fine) or null
const sel = {{}};                     // asset -> chosen rendered az (multiple of 30) or null
const cards = [...document.querySelectorAll('.card')];
const out = document.getElementById('out');
const stat = document.getElementById('stat');

function cdiff(a,b){{return Math.abs(((a-b+180)%360)-180);}}
function nearestAz(card, target){{
  let best=null, bd=1e9;
  card.querySelectorAll('.v').forEach(v=>{{const az=+v.dataset.az, d=cdiff(az,target); if(d<bd){{bd=d;best=az;}}}});
  return best;
}}
function setChosen(card, az){{
  const asset=card.dataset.asset;
  sel[asset]=az;
  card.querySelectorAll('.v').forEach(v=>v.classList.toggle('chosen', (+v.dataset.az)===az));
  const auto=AUTO[asset];
  const changed = (auto==null) ? (az!=null) : (az!==nearestAz(card, auto));
  card.classList.toggle('changed', changed);
  card.classList.toggle('unset', az==null);
  const p=card.querySelector('.pick');
  p.className='pick '+(az==null?'unset':(changed?'changed':'set'));
  p.textContent = az==null ? 'UNSET ✋' : (az+'°'+(changed?' ✎':' ✓'));
  refresh();
}}
function refresh(){{
  let unset=0, changed=0;
  for(const c of cards){{if(sel[c.dataset.asset]==null)unset++; if(c.classList.contains('changed'))changed++;}}
  stat.innerHTML=`<b>${{cards.length}}</b> assets · <span class="u">unset: ${{unset}}</span> · <span class="c">changed: ${{changed}}</span>`;
  const clean={{}}; for(const k in sel) clean[k]=sel[k];
  out.value=JSON.stringify(clean);
}}
// init from auto (highlight nearest rendered view; no-face -> unset)
cards.forEach(card=>{{
  const a=AUTO[card.dataset.asset];
  setChosen(card, a==null?null:nearestAz(card, a));
}});
// click a thumbnail
document.addEventListener('click', e=>{{
  const v=e.target.closest('.v'); if(!v) return;
  setChosen(v.closest('.card'), +v.dataset.az);
}});
// copy
document.getElementById('copy').onclick=async()=>{{
  const b=document.getElementById('copy');
  try{{await navigator.clipboard.writeText(out.value); b.textContent='Copied ✓';}}
  catch(e){{out.style.display='block'; out.select();
    try{{document.execCommand('copy'); b.textContent='Copied ✓';}}catch(_){{b.textContent='Select textarea & copy';}}}}
  setTimeout(()=>b.textContent='Copy JSON', 1600);
}};
// filter
document.querySelectorAll('.filt').forEach(btn=>btn.onclick=()=>{{
  document.querySelectorAll('.filt').forEach(b=>b.classList.remove('on')); btn.classList.add('on');
  const f=btn.dataset.f;
  cards.forEach(c=>{{
    const conf=c.dataset.conf, unset=sel[c.dataset.asset]==null;
    let show=true;
    if(f==='review') show=(conf!=='OK');
    else if(f==='unset') show=unset;
    c.style.display=show?'':'none';
  }});
}});
</script>"""
os.makedirs("runs", exist_ok=True)
open(OUT, "w").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB) — {len(cards)} assets")
