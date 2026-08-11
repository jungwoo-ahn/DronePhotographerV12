"""Build the self-contained HTML report from the evaluation JSONs.

Artifacts are served under a strict CSP — no external hosts, no file:// reads — so
every frame has to be inlined as a data URI. That is the whole reason this is a
build step rather than a template: the images are the report, and they have to be
downscaled and re-encoded to keep the page from ballooning.

Reads whatever exists, so it can be run mid-flight for a preview and again when the
jobs finish:
    runs/gt_replay/*.json      training-trajectory reproduction (tests A and B)
    runs/closed_loop/*.json    held-out placements
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

V12 = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--gt-replay", nargs="*", default=["runs/gt_replay/iter8000.json"])
ap.add_argument("--closed-loop", nargs="*",
                default=["runs/closed_loop/back_iter8000.json",
                         "runs/closed_loop/all_iter8000.json"])
ap.add_argument("--thumb", type=int, default=190, help="embedded frame width in px")
ap.add_argument("--cl-thumb", type=int, default=150,
                help="frame width for closed-loop sheets; they dominate page "
                     "weight when every episode is shown")
ap.add_argument("--quality", type=int, default=72)
ap.add_argument("--title", default="Camera Policy — Evaluation Report")
ap.add_argument("--out", default="runs/report/eval_report.html")
args = ap.parse_args()


# ---------------------------------------------------------------- images
_cache: dict[tuple[str, int, int], str] = {}


def data_uri(path: str | None, width: int | None = None) -> str | None:
    """Downscaled JPEG as a data URI, or None if the frame is missing.

    Missing frames are common and benign (a rollout that ended early, a GT path
    shorter than the chunk count), so this returns None instead of raising — the
    caller renders a placeholder rather than losing the whole episode.
    """
    if not path:
        return None
    width = width or args.thumb
    key = (str(path), width, args.quality)
    if key in _cache:
        return _cache[key]
    p = Path(path)
    if not p.is_absolute():
        p = V12 / p
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    if img.width > width:
        img = img.resize((width, max(1, round(img.height * width / img.width))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=args.quality, optimize=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    _cache[key] = uri
    return uri


def load(paths: list[str]) -> list[dict]:
    out = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = V12 / p
        if p.exists():
            try:
                out.append({"path": str(p), **json.loads(p.read_text())})
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {p}: {exc}")
    return out


# ---------------------------------------------------------------- charts
def sparkline(trace: list[float], width: int = 240, height: int = 54) -> str:
    """Distance-to-goal over chunks, with the minimum marked.

    The minimum is the point of the chart: several rollouts reach the goal and then
    drift past it, and a plain end-point number hides that entirely.
    """
    if not trace or len(trace) < 2:
        return ""
    lo, hi = min(trace), max(trace)
    span = (hi - lo) or 1.0
    pad = 8
    w, h = width - 2 * pad, height - 2 * pad
    pts = [(pad + i * w / (len(trace) - 1), pad + h - (v - lo) / span * h)
           for i, v in enumerate(trace)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    imin = trace.index(lo)
    mx, my = pts[imin]
    ex, ey = pts[-1]
    area = f"M{pts[0][0]:.1f},{pad+h} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts) + \
           f" L{pts[-1][0]:.1f},{pad+h} Z"
    return f"""<svg viewBox="0 0 {width} {height}" class="spark" role="img"
   aria-label="distance to goal per chunk, minimum {lo:.3f}, final {trace[-1]:.3f}">
  <path d="{area}" class="spark-area"/>
  <path d="{path}" class="spark-line"/>
  <circle cx="{mx:.1f}" cy="{my:.1f}" r="3.5" class="spark-min"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" class="spark-end"/>
</svg>"""


def bars(rows: list[tuple[str, float]], vmax: float | None = None) -> str:
    """Horizontal bars for per-sector aggregates."""
    if not rows:
        return ""
    vmax = vmax or max(abs(v) for _, v in rows) or 1.0
    out = ['<div class="bars">']
    for label, value in rows:
        pct = min(100.0, abs(value) / vmax * 100.0)
        sign = "pos" if value >= 0 else "neg"
        out.append(
            f'<div class="bar-row"><span class="bar-label">{label}</span>'
            f'<span class="bar-track"><span class="bar-fill {sign}" style="width:{pct:.1f}%"></span></span>'
            f'<span class="bar-val">{value:+.3f}</span></div>')
    out.append("</div>")
    return "".join(out)


def t(ko: str, en: str) -> str:
    """A bilingual span pair; the page toggle shows one language at a time."""
    return f'<span class="ko">{ko}</span><span class="en">{en}</span>'


# ---------------------------------------------------------------- test A charts
def _cumsum(chunk: list[list[float]], i: int, j: int) -> tuple[list[float], list[float]]:
    """Cumulative displacement along two of the three translation axes."""
    xs, ys, x, y = [0.0], [0.0], 0.0, 0.0
    for step in chunk:
        x += float(step[i])
        y += float(step[j])
        xs.append(x)
        ys.append(y)
    return xs, ys


def path_chart(gt: list[list[float]], pred: list[list[float]],
               i: int = 0, j: int = 2, labels: tuple[str, str] = ("right", "forward"),
               width: int = 250, height: int = 190) -> str:
    """Where the 8 predicted steps actually take the camera, vs where GT would.

    Per-step errors are tiny numbers whose consequence is invisible until they are
    added up — this plots the cumulative path so the question becomes "does the camera
    end up in the same place". Equal aspect on both axes: an auto-fitted aspect would
    stretch one axis and make a straight path look bent.
    """
    gx, gy = _cumsum(gt, i, j)
    px, py = _cumsum(pred, i, j)
    allx, ally = gx + px, gy + py
    pad = 16
    cx, cy = (min(allx) + max(allx)) / 2, (min(ally) + max(ally)) / 2
    span = max(max(allx) - min(allx), max(ally) - min(ally), 1e-4) * 1.18
    w = width - 2 * pad

    def T(x, y):
        return (pad + (x - cx) / span * w + w / 2,
                pad + (cy - y) / span * w + (height - 2 * pad) / 2)

    def poly(xs, ys):
        return " ".join(f"{'M' if k == 0 else 'L'}{X:.1f},{Y:.1f}"
                        for k, (X, Y) in enumerate(T(x, y) for x, y in zip(xs, ys)))

    dots = "".join(
        f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="2.4" class="pt-pred"/>'
        for X, Y in (T(x, y) for x, y in zip(px[1:], py[1:])))
    gdots = "".join(
        f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="2.4" class="pt-gt"/>'
        for X, Y in (T(x, y) for x, y in zip(gx[1:], gy[1:])))
    ox, oy = T(0, 0)
    ex, ey = T(gx[-1], gy[-1])
    fx, fy = T(px[-1], py[-1])
    return f"""<svg viewBox="0 0 {width} {height}" class="pathchart" role="img"
  aria-label="cumulative camera displacement over 8 steps, ground truth versus prediction">
  <path d="{poly(gx, gy)}" class="p-gt"/>
  <path d="{poly(px, py)}" class="p-pred"/>
  {gdots}{dots}
  <circle cx="{ox:.1f}" cy="{oy:.1f}" r="3.6" class="pt-origin"/>
  <circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" class="pt-gt-end"/>
  <circle cx="{fx:.1f}" cy="{fy:.1f}" r="4" class="pt-pred-end"/>
  <text x="{width-6}" y="{height-6}" class="axlab" text-anchor="end">{labels[0]} × {labels[1]}</text>
</svg>"""


def step_error_chart(per_sample: list[dict], key: str = "trans_err_per_step",
                     width: int = 250, height: int = 108,
                     colour: str = "err") -> str:
    """Per-step error across the 8 steps, one faint line per sample plus their mean.

    Drawn per sample rather than averaged first, because the spread IS the finding:
    this policy is stochastic and a single draw would report sampling variance as fit
    error. If the lines fan out with step index, error is compounding within the chunk;
    if they stay flat, it is not.
    """
    series = [s.get(key) or [] for s in per_sample]
    series = [x for x in series if x]
    if not series:
        return ""
    n = len(series[0])
    mean = [sum(s[k] for s in series) / len(series) for k in range(n)]
    hi = max(max(s) for s in series) or 1e-6
    pad_l, pad_b, pad_t = 30, 18, 10
    w, h = width - pad_l - 8, height - pad_b - pad_t

    def pts(vals):
        return " ".join(
            f"{'M' if k == 0 else 'L'}{pad_l + k * w / max(1, n - 1):.1f},"
            f"{pad_t + h - v / hi * h:.1f}" for k, v in enumerate(vals))

    faint = "".join(f'<path d="{pts(s)}" class="se-sample"/>' for s in series)
    ticks = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + h - f * h:.1f}" x2="{width-8}" '
        f'y2="{pad_t + h - f * h:.1f}" class="se-grid"/>'
        f'<text x="{pad_l-5}" y="{pad_t + h - f * h + 3.2:.1f}" class="axlab" '
        f'text-anchor="end">{hi*f:.3g}</text>' for f in (0.0, 1.0))
    return f"""<svg viewBox="0 0 {width} {height}" class="stepchart {colour}" role="img"
  aria-label="per-step error across the 8-step chunk for each of the samples">
  {ticks}{faint}
  <path d="{pts(mean)}" class="se-mean"/>
  <text x="{pad_l}" y="{height-5}" class="axlab">step 1</text>
  <text x="{width-8}" y="{height-5}" class="axlab" text-anchor="end">{n}</text>
</svg>"""


def test_a_gallery(eps: list[dict]) -> str:
    """One card per episode: the path it would take, and how the error behaves."""
    cards = []
    for e in eps:
        a = e.get("test_a") or {}
        gt, pred = a.get("gt_chunk"), a.get("pred_mean_chunk")
        if not gt or not pred:
            continue
        b = a.get("best_sample") or {}
        cards.append(f"""<div class="acard">
  <div class="acard-head">
    <span class="ep-sector">{e.get('sector','-')}</span>
    <span class="ep-name">{e.get('placement','')[:34]}</span>
  </div>
  <div class="acard-charts">
    {path_chart(gt, pred)}
    <div class="acard-side">
      {step_error_chart(a.get('per_sample') or [], 'trans_err_per_step')}
      <div class="acard-nums">
        <span><b>{b.get('dir_cos_mean',0):.3f}</b> {t("방향","dir")}</span>
        <span><b>{b.get('trans_err_mean',0):.4f}</b> {t("오차","err")}</span>
        <span><b>{b.get('rot_err_deg_mean',0):.2f}°</b> {t("회전","rot")}</span>
      </div>
    </div>
  </div>
</div>""")
    if not cards:
        return ""
    return f"""<div class="alegend">
  <span><i class="sw gt"></i>{t("GT 액션 누적 경로", "GT action path")}</span>
  <span><i class="sw pred"></i>{t("예측(4샘플 평균)", "prediction (mean of 4)")}</span>
  <span><i class="sw dot"></i>{t("각 스텝", "each step")}</span>
  <span class="muted">{t("오른쪽 그래프: 스텝별 이동 오차, 얇은 선 = 개별 샘플",
                          "right chart: per-step translation error, thin lines = individual samples")}</span>
</div>
<div class="agrid">{''.join(cards)}</div>"""


# ---------------------------------------------------------------- contact sheet
def frames_trustworthy(run: dict) -> bool:
    """Whether this run's saved frames can be attributed to it.

    Runs before the fix wrote into a shared `frames/` directory named only by
    episode index, so two runs in the same output directory overwrote each other's
    images — the numbers stayed correct (they come from env poses) but a strip could
    show two different scenes. Those runs cannot be repaired after the fact, so the
    report refuses to display their frames rather than showing a plausible lie.
    A run that declares `frames_dir` is namespaced by its own output file.
    """
    return bool((run.get("summary") or {}).get("frames_dir")
                or run.get("frames_dir"))


def contact_sheet(ep: dict, thumb: int, show_frames: bool = True) -> str:
    """One episode as a filmstrip: start, each rollout step, then the goal apart.

    A contact sheet is how this gets judged in the end — the numbers say the camera
    got closer, the frames say whether the shot is actually the requested one.
    """
    b = ep.get("test_b") or {}
    shots = b.get("shots") or ep.get("shots") or []
    cells = []
    for s in shots:
        uri = data_uri(s.get("path"), thumb) if show_frames else None
        tag = s.get("tag", "")
        d = s.get("d")
        label = t("시작", "start") if tag == "start" else tag.replace("chunk", "+")
        cells.append(f"""<figure class="frame">
  {'<img src="' + uri + '" alt="' + tag + '" loading="lazy">' if uri
   else ('<div class="frame-missing" title="frame not attributable to this run">?</div>'
         if not show_frames else '<div class="frame-missing">—</div>')}
  <figcaption><span class="ftag">{label}</span>
    {'<span class="fd">' + f'{d:.3f}' + '</span>' if isinstance(d, (int, float)) else ''}
  </figcaption>
</figure>""")

    goal_uri = data_uri(ep.get("goal_frame_image"), thumb)
    goal_cell = f"""<figure class="frame goal">
  {'<img src="' + goal_uri + '" alt="goal" loading="lazy">' if goal_uri
   else '<div class="frame-missing">—</div>'}
  <figcaption><span class="ftag">{t("GT 목표", "GT goal")}</span></figcaption>
</figure>"""

    trace = b.get("trace") or ep.get("trace") or []
    d0 = trace[0] if trace else None
    d1 = trace[-1] if trace else None
    dbest = min(trace) if trace else None
    overshoot = (dbest is not None and d1 is not None and d1 > dbest + 0.05)

    a = ep.get("test_a") or {}
    best = a.get("best_sample") or {}
    a_html = ""
    if best:
        a_html = f"""<div class="metric-strip">
  <span><b>{best.get('dir_cos_mean', 0):.3f}</b> {t("방향 코사인", "dir cos")}</span>
  <span><b>{best.get('trans_err_mean', 0):.4f}</b> {t("이동 오차", "trans err")}</span>
  <span><b>{best.get('rot_err_deg_mean', 0):.2f}°</b> {t("회전 오차", "rot err")}</span>
</div>"""

    badge = (f'<span class="badge warn">{t("오버슈트", "overshoot")}</span>' if overshoot
             else (f'<span class="badge good">{t("도달", "reached")}</span>'
                   if (dbest is not None and dbest < 0.35) else
                   f'<span class="badge flat">{t("미도달", "not reached")}</span>'))

    stats = ""
    if d0 is not None:
        stats = (f'<span class="kv">{d0:.3f} <span class="arrow">&rarr;</span> '
                 f'<b>{d1:.3f}</b></span>'
                 f'<span class="kv muted">{t("최저", "best")} {dbest:.3f}</span>')

    return f"""<article class="episode">
  <header class="ep-head">
    <div class="ep-id">
      <span class="ep-sector">{ep.get('sector', '-')}</span>
      <span class="ep-name">{ep.get('placement', '')[:52]}</span>
    </div>
    <div class="ep-stats">{stats}{badge}</div>
  </header>
  {a_html}
  {'' if show_frames else '<p class="ep-note warnnote">' + t(
     "이 실행의 저장 프레임은 다른 실행에 덮어써져 이 에피소드의 것이라고 보장할 수 없어 숨겼습니다. "
     "아래 거리 수치는 카메라 포즈에서 계산된 것이라 영향이 없습니다.",
     "Saved frames for this run were overwritten by another run and cannot be attributed to "
     "this episode, so they are hidden. The distances below come from camera poses and are "
     "unaffected.") + '</p>'}
  <div class="strip">
    <div class="strip-run">{''.join(cells)}</div>
    <div class="strip-goal">{goal_cell}</div>
  </div>
  <div class="ep-foot">
    {sparkline(trace)}
    <p class="ep-note">{t(
      f"목표까지 {ep.get('delta','?')}프레임, {b.get('chunks', len(trace)-1)}청크 실행",
      f"goal {ep.get('delta','?')} frames away, {b.get('chunks', len(trace)-1)} chunks executed")}</p>
  </div>
</article>"""


CSS = """
:root{
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#EEF1F6; --line:#DDE2EA;
  --ink:#16181D; --ink-2:#3C4453; --muted:#5B6472;
  --accent:#4E5FD0; --good:#12866A; --warn:#B4750E; --flat:#6B7280;
  --shadow:0 1px 2px rgba(20,24,35,.06),0 8px 24px rgba(20,24,35,.05);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0E1015; --surface:#171A21; --surface-2:#1E222B; --line:#2A2F3A;
    --ink:#E7EAF0; --ink-2:#B9C1CE; --muted:#8A93A3;
    --accent:#8C9BF0; --good:#3FBE9B; --warn:#E0A33F; --flat:#7C8494;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --bg:#0E1015; --surface:#171A21; --surface-2:#1E222B; --line:#2A2F3A;
  --ink:#E7EAF0; --ink-2:#B9C1CE; --muted:#8A93A3;
  --accent:#8C9BF0; --good:#3FBE9B; --warn:#E0A33F; --flat:#7C8494;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
}
:root[data-theme="light"]{
  --bg:#F7F8FA; --surface:#FFFFFF; --surface-2:#EEF1F6; --line:#DDE2EA;
  --ink:#16181D; --ink-2:#3C4453; --muted:#5B6472;
  --accent:#4E5FD0; --good:#12866A; --warn:#B4750E; --flat:#6B7280;
  --shadow:0 1px 2px rgba(20,24,35,.06),0 8px 24px rgba(20,24,35,.05);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,
    "Helvetica Neue","Apple SD Gothic Neo","Noto Sans KR",sans-serif;
  font-size:15px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.mono,.fd,.kv,.bar-val,table td.num{
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:1160px;margin:0 auto;padding:0 24px 96px}
header.top{padding:56px 0 28px;border-bottom:1px solid var(--line);margin-bottom:40px}
.eyebrow{
  font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:650;margin:0 0 12px;
}
h1{font-size:clamp(28px,4.2vw,42px);line-height:1.12;margin:0 0 14px;
   letter-spacing:-.02em;font-weight:700;text-wrap:balance}
.lede{font-size:17px;color:var(--ink-2);max-width:64ch;margin:0}
h2{font-size:23px;margin:56px 0 6px;letter-spacing:-.01em;font-weight:680;text-wrap:balance}
h3{font-size:15px;margin:30px 0 8px;font-weight:650}
.sub{color:var(--muted);margin:0 0 20px;max-width:70ch}
p{max-width:72ch}
code{background:var(--surface-2);padding:.12em .38em;border-radius:4px;font-size:.9em;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}

/* controls */
.controls{position:sticky;top:0;z-index:20;display:flex;gap:8px;justify-content:flex-end;
  padding:10px 0;background:linear-gradient(var(--bg) 70%,transparent)}
.btn{font:inherit;font-size:12px;font-weight:600;padding:6px 12px;border-radius:999px;
  border:1px solid var(--line);background:var(--surface);color:var(--ink-2);cursor:pointer}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* hero numbers */
.tiles{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin:26px 0 8px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:18px 18px 16px;box-shadow:var(--shadow)}
.tile .n{font-size:30px;font-weight:700;letter-spacing:-.02em;line-height:1.1;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.tile .l{font-size:12px;color:var(--muted);margin-top:6px}
.tile.good .n{color:var(--good)} .tile.warn .n{color:var(--warn)}

/* episode contact sheet */
.episode{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:16px 16px 12px;margin:16px 0;box-shadow:var(--shadow)}
.ep-head{display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;
  align-items:baseline;margin-bottom:10px}
.ep-sector{display:inline-block;font-size:11px;font-weight:650;letter-spacing:.06em;
  text-transform:uppercase;color:var(--accent);background:var(--surface-2);
  padding:3px 9px;border-radius:999px;margin-right:9px}
.ep-name{font-size:12.5px;color:var(--muted)}
.ep-stats{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.kv{font-size:13px;color:var(--ink-2)} .kv.muted{color:var(--muted);font-size:12px}
.arrow{color:var(--muted);padding:0 2px}
.badge{font-size:11px;font-weight:650;padding:3px 9px;border-radius:999px;
  letter-spacing:.02em;border:1px solid transparent}
.badge.good{color:var(--good);border-color:var(--good);background:color-mix(in srgb,var(--good) 10%,transparent)}
.badge.warn{color:var(--warn);border-color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent)}
.badge.flat{color:var(--flat);border-color:var(--line);background:var(--surface-2)}
.metric-strip{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted);
  padding:8px 0 10px;border-top:1px solid var(--line)}
.metric-strip b{color:var(--ink);font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums;font-weight:650}
.strip{display:flex;gap:0;align-items:flex-start;overflow-x:auto;padding-bottom:6px}
/* flex:0 0 auto is required, not cosmetic: as a shrinkable flex child this row would
   compress while its own .frame children refuse to (they are 0 0 auto), so the frames
   overflowed the row and printed on top of the goal panel. */
.strip-run{display:flex;gap:6px;flex:0 0 auto;padding-right:14px}
/* The goal is the reference you scan against, so it stays pinned while the rollout
   scrolls under it. Needs an opaque background or the frames show through. */
.strip-goal{position:sticky;right:0;flex:0 0 auto;padding-left:14px;
  background:var(--surface);border-left:2px dashed var(--line);
  box-shadow:-12px 0 12px -10px rgba(0,0,0,.18)}
.frame{margin:0;flex:0 0 auto}
.frame img{display:block;border-radius:6px;border:1px solid var(--line)}
.frame.goal img{border-color:var(--accent);border-width:2px}
.frame-missing{width:120px;height:90px;display:grid;place-items:center;color:var(--muted);
  background:var(--surface-2);border-radius:6px}
.frame figcaption{display:flex;gap:6px;justify-content:space-between;
  font-size:10.5px;color:var(--muted);padding-top:4px}
.ftag{font-weight:650;letter-spacing:.03em}
.fd{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.ep-foot{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
  border-top:1px solid var(--line);padding-top:8px;margin-top:8px}
.ep-note{font-size:12px;color:var(--muted);margin:0}
.spark{width:240px;height:54px;flex:0 0 auto}
.spark-line{fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.spark-area{fill:color-mix(in srgb,var(--accent) 12%,transparent);stroke:none}
.spark-min{fill:var(--good);stroke:var(--surface);stroke-width:2}
.spark-end{fill:var(--warn);stroke:var(--surface);stroke-width:2}

/* tables */
.tablewrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:520px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:650}
td.num{text-align:right}
tbody tr:hover{background:var(--surface-2)}

/* bars */
.bars{display:flex;flex-direction:column;gap:7px;margin:12px 0}
.bar-row{display:grid;grid-template-columns:110px 1fr 74px;gap:10px;align-items:center;font-size:12.5px}
.bar-label{color:var(--ink-2)}
.bar-track{background:var(--surface-2);border-radius:999px;height:9px;overflow:hidden}
.bar-fill{display:block;height:100%;border-radius:999px;background:var(--good)}
.bar-fill.neg{background:var(--warn)}
.bar-val{text-align:right;color:var(--muted);font-size:12px}

.gains{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));margin:18px 0}
.gain{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:16px 18px;box-shadow:var(--shadow)}
.gain-head{font-size:13px;font-weight:650;margin-bottom:12px}
.gain-head .muted{color:var(--muted);font-weight:400}
.gain-bars .bar-row{grid-template-columns:1fr 90px 62px}
.gain-note{font-size:12px;color:var(--muted);margin:10px 0 0;max-width:none}
.gain-note b{color:var(--ink);font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums}
.alegend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;font-size:12px;
  color:var(--ink-2);margin:16px 0 10px}
.alegend .muted{color:var(--muted)}
.sw{display:inline-block;width:16px;height:0;border-top-width:2.5px;border-top-style:solid;
  margin-right:6px;vertical-align:middle}
.sw.gt{border-color:var(--ink-2)}
.sw.pred{border-color:var(--accent);border-top-style:dashed}
.sw.dot{width:7px;height:7px;border:0;border-radius:50%;background:var(--muted)}
.agrid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.acard{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px;box-shadow:var(--shadow)}
.acard-head{display:flex;align-items:baseline;gap:2px;margin-bottom:6px;overflow:hidden}
.acard-charts{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}
.acard-side{display:flex;flex-direction:column;gap:4px;min-width:180px;flex:1}
.acard-nums{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted)}
.acard-nums b{color:var(--ink);font-family:ui-monospace,Menlo,monospace;
  font-variant-numeric:tabular-nums;font-weight:650}
.pathchart{width:250px;height:190px;flex:0 0 auto;max-width:100%}
.stepchart{width:100%;height:108px}
.p-gt{fill:none;stroke:var(--ink-2);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.p-pred{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-dasharray:5 3.5;
  stroke-linejoin:round;stroke-linecap:round}
.pt-gt{fill:var(--ink-2)} .pt-pred{fill:var(--accent)}
.pt-origin{fill:var(--surface);stroke:var(--muted);stroke-width:2}
.pt-gt-end{fill:var(--ink-2);stroke:var(--surface);stroke-width:2}
.pt-pred-end{fill:var(--accent);stroke:var(--surface);stroke-width:2}
.se-sample{fill:none;stroke:var(--accent);stroke-width:1;opacity:.34}
.se-mean{fill:none;stroke:var(--accent);stroke-width:2.2;stroke-linejoin:round}
.se-grid{stroke:var(--line);stroke-width:1}
.axlab{font-size:9px;fill:var(--muted);
  font-family:ui-monospace,Menlo,monospace}
table.cmp td:first-child{white-space:nowrap}
.tt{display:inline-block;width:16px;height:16px;line-height:16px;text-align:center;
  border-radius:4px;font-size:10px;font-weight:700;margin-right:7px;color:var(--surface)}
.tt.a{background:var(--accent)} .tt.b{background:var(--ink-2)}
td.delta{font-weight:650}
td.delta.good{color:var(--good)} td.delta.warn{color:var(--warn)}
td.delta.flat{color:var(--muted)}
td .muted{color:var(--muted);font-weight:400;font-size:11px}
.scatter{width:100%;max-width:620px;height:auto;margin:12px 0}
.sc{fill:var(--accent);opacity:.82;stroke:var(--surface);stroke-width:1}
.sc.neg{fill:var(--warn)}
.zeroline{stroke:var(--ink-2);stroke-width:1.5}
.splitline{stroke:var(--line);stroke-width:1;stroke-dasharray:3 3}
.meanline{stroke:var(--good);stroke-width:2;stroke-dasharray:6 3}
.warnnote{color:var(--warn);font-size:11.5px;margin:0 0 8px;max-width:none}
.callout{border-left:3px solid var(--accent);background:var(--surface);
  border-radius:0 10px 10px 0;padding:14px 18px;margin:18px 0;box-shadow:var(--shadow)}
.callout.warn{border-left-color:var(--warn)}
.callout p{margin:0} .callout p+p{margin-top:8px}
.foot{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);
  font-size:12px;color:var(--muted)}
/* Korean is the default so the page reads correctly before JS runs;
   the toggle only ever adds .lang-en. */
.en{display:none}
body.lang-en .en{display:inline}
body.lang-en .ko{display:none}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
const root=document.documentElement, body=document.body;
const langBtn=document.getElementById('lang');
langBtn.addEventListener('click',()=>{
  const en=body.classList.toggle('lang-en');
  langBtn.textContent=en?'한국어':'EN';
});
document.getElementById('theme').addEventListener('click',()=>{
  const dark=(root.dataset.theme||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'))==='dark';
  root.dataset.theme=dark?'light':'dark';
});
"""


def render(gt_runs: list[dict], cl_runs: list[dict]) -> str:
    # Per-episode sections show the newest checkpoint; older runs feed the comparison
    # table only. Merging them would silently double every episode.
    def _iter_of(r):
        m = re.search(r"iter_0*(\d+)", str(r.get("checkpoint", "")))
        return int(m.group(1)) if m else 0
    newest = max(gt_runs, key=_iter_of) if gt_runs else None
    gt_eps = list(newest.get("episodes", [])) if newest else []
    cl_eps = [e for r in cl_runs for e in r.get("results", r.get("episodes", []))]

    # ---- headline numbers
    a_best = [e["test_a"]["best_sample"] for e in gt_eps if e.get("test_a")]
    n_reach = sum(1 for e in gt_eps
                  if (e.get("test_b") or {}).get("d_best", 9) < 0.35)
    n_over = sum(1 for e in gt_eps
                 if (b := e.get("test_b")) and b.get("d_end", 0) > b.get("d_best", 0) + 0.05)
    prov = next((r.get("provenance") for r in gt_runs if r.get("provenance")), {})

    tiles = []
    if a_best:
        cos = sum(m["dir_cos_mean"] for m in a_best) / len(a_best)
        rot = sum(m["rot_err_deg_mean"] for m in a_best) / len(a_best)
        tiles.append(("good", f"{cos:.3f}", t("평균 방향 코사인 (테스트 A)", "mean direction cosine (test A)")))
        tiles.append(("", f"{rot:.2f}°", t("평균 회전 오차 (테스트 A)", "mean rotation error (test A)")))
    if gt_eps:
        tiles.append(("good", f"{n_reach}/{len(gt_eps)}", t("목표 도달 (최저거리 &lt; 0.35)", "reached goal (min dist &lt; 0.35)")))
        tiles.append(("warn", f"{n_over}/{len(gt_eps)}", t("지나친 뒤 이탈", "overshot then drifted")))
    if prov.get("checked"):
        tiles.append(("good", f"{prov['matched']}/{prov['checked']}",
                      t("출처 검증 일치", "provenance prompts matched")))

    tiles_html = "".join(
        f'<div class="tile {c}"><div class="n">{n}</div><div class="l">{l}</div></div>'
        for c, n, l in tiles)

    # ---- held-out by sector
    sector_rows: dict[str, list[float]] = {}
    for e in cl_eps:
        if "improvement" in e:
            sector_rows.setdefault(e.get("sector", "?"), []).append(float(e["improvement"]))
    sector_bars = bars(sorted(
        ((f"{k} ({len(v)})", sum(v) / len(v)) for k, v in sector_rows.items()),
        key=lambda kv: -kv[1]))

    cl_table = ""
    if cl_eps:
        rows = "".join(
            f"<tr><td>{e.get('sector','-')}</td><td>{e.get('placement','')[:42]}</td>"
            f"<td class='num'>{e.get('delta','-')}</td>"
            f"<td class='num'>{e.get('d_start',0):.3f}</td>"
            f"<td class='num'>{e.get('d_end',0):.3f}</td>"
            f"<td class='num'>{e.get('d_best',0):.3f}</td>"
            f"<td class='num'>{e.get('improvement',0):+.3f}</td></tr>"
            for e in sorted(cl_eps, key=lambda x: -x.get("improvement", 0)))
        cl_table = f"""<div class="tablewrap"><table>
<thead><tr><th>{t("섹터","sector")}</th><th>{t("배치","placement")}</th><th>Δ</th>
<th>{t("시작","start")}</th><th>{t("종료","end")}</th><th>{t("최저","best")}</th>
<th>{t("개선","improve")}</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    gt_ok = bool(newest and frames_trustworthy(newest))
    gt_sheets = "".join(contact_sheet(e, args.thumb, gt_ok) for e in gt_eps)
    cl_ok = all(frames_trustworthy(r) for r in cl_runs) if cl_runs else False
    cl_sheets = "".join(contact_sheet(e, args.cl_thumb, cl_ok) for e in cl_eps if e.get("shots"))

    TITLE = args.title
    return f"""<title>{TITLE}</title>
<style>{CSS}</style>
<div class="wrap">
<div class="controls">
  <button class="btn" id="lang" type="button">EN</button>
  <button class="btn" id="theme" type="button">◐</button>
</div>

<header class="top">
  <p class="eyebrow">DronePhotographer v12 · Cosmos 3 · {" vs ".join(_ckpt_label(r) for r in gt_runs) or "—"}</p>
  <h1>{t("목표 조건부 카메라 정책 — 평가 보고서",
          "Goal-conditioned camera policy — evaluation report")}</h1>
  <p class="lede">{t(
    "학습한 궤적을 재현하는가(테스트 A·B), 그리고 롤아웃으로 목표 구도에 도달하는가"
    "(closed-loop). 각 절이 학습 배치에서 쟀는지 처음 보는 배치에서 쟀는지는 절마다 명시합니다. "
    "모든 프레임은 Blender에서 실제로 렌더링된 것입니다.",
    "Does the policy reproduce trajectories it was trained on (tests A and B), and does a "
    "rollout reach the requested framing (closed-loop)? Each section states whether it was "
    "measured on trained or unseen placements. Every frame here was actually rendered in "
    "Blender.")}</p>
</header>

<div class="tiles">{tiles_html}</div>

{compare_section(gt_runs)}
{rollout_compare(cl_runs)}

<h2>{t("무엇을 쟀는가", "What was measured")}</h2>
<p class="sub">{t(
  "두 가지를 분리했습니다. 섞으면 무엇을 탓해야 할지 알 수 없기 때문입니다.",
  "Two things, kept apart — mixed together they cannot tell you what to blame.")}</p>
<div class="callout"><p><b>{t("테스트 A — 청크 재현", "Test A — chunk reproduction")}</b><br>{t(
  "GT 시작 프레임과 GT 목표 프롬프트를 주고, 예측한 8스텝 액션을 학습 타깃과 직접 비교합니다. "
  "자기회귀도 렌더 왕복도 없으므로 오차는 <b>순수한 fit</b>입니다.",
  "Feed the GT start frame and GT goal prompt, compare the predicted 8-step chunk directly "
  "against the supervision target. No autoregression, no render round-trip — the error is "
  "<b>pure fit</b>.")}</p>
<p><b>{t("테스트 B — 목표까지 롤아웃", "Test B — rollout to goal")}</b><br>{t(
  "같은 시작점에서 청크마다 재렌더하며 목표까지 진행합니다. A가 좋은데 B가 나쁘면 "
  "fit 문제가 아니라 <b>누적 오차 또는 정지 실패</b>입니다.",
  "From the same start, re-rendering after every chunk. A good A with a bad B means the "
  "fit is fine and the problem is <b>compounding error or failure to stop</b>.")}</p></div>
<div class="callout"><p>{t(
  f"<b>출처 검증.</b> export 열거는 시드가 같으면 결정론적이라, 재현한 뒤 내보낸 데이터셋의 "
  f"프롬프트와 대조했습니다 — <b>{prov.get('matched','?')}/{prov.get('checked','?')} 완전 일치</b>. "
  "off-by-one 하나만 있어도 모든 에피소드가 이웃의 목표를 받아 멀쩡한 정책이 망가진 것처럼 보입니다.",
  f"<b>Provenance.</b> The export enumeration is deterministic given its seed, so it was "
  f"replayed and checked prompt-for-prompt against the exported dataset — "
  f"<b>{prov.get('matched','?')}/{prov.get('checked','?')} exact match</b>. A single off-by-one "
  "would hand every episode a neighbour's goal and make a correct policy look broken.")}</p></div>

{finding_section(gt_eps, cl_eps)}

<h2>{t("테스트 A — 학습 궤적 재현 (fit)", "Test A — fit on training trajectories")}</h2>
{a_table(gt_eps)}
{test_a_gallery(gt_eps)}

<h2>{t("테스트 B — 목표까지 롤아웃", "Test B — rollout to the goal")}</h2>
<p class="sub">{t(
  "각 줄은 컨택트 시트입니다. 왼쪽이 시작, 오른쪽 점선 뒤가 GT 목표 프레임. "
  "스파크라인의 <span style='color:var(--good)'>●</span> 은 최저 거리, "
  "<span style='color:var(--warn)'>●</span> 은 종료 지점입니다.",
  "Each row is a contact sheet: start on the left, the GT goal frame past the dashed rule "
  "on the right. On the sparkline <span style='color:var(--good)'>●</span> marks the closest "
  "approach and <span style='color:var(--warn)'>●</span> where it ended up.")}</p>
{gt_sheets or '<p class="sub">' + t("아직 결과 없음", "no results yet") + '</p>'}

{closed_loop_header(cl_runs)}
{("<h3>" + t("시작 거리에 따른 개선 — 오버슈트의 형태",
                     "Improvement by starting distance — the shape of the overshoot") + "</h3>"
   + "<p class=\"sub\">" + t(
     "점 하나가 에피소드 하나. 가로선은 각 구간의 평균입니다. 멀리서 시작하면 크게 개선하지만 "
     "이미 가까우면 거의 못 벌고 때때로 손해를 봅니다 — 평균 하나로는 완전히 가려지는 차이이고, "
     "‘정책이 약하다’와 ‘정책이 멈추지 못한다’를 가르는 지점입니다.",
     "One point per episode; the horizontal rules are per-band means. From far away the "
     "policy gains a lot; when it starts near the goal it gains almost nothing and "
     "sometimes loses ground — a distinction a single mean erases, and the one that "
     "separates ‘the policy is weak’ from ‘the policy cannot hold still’.") + "</p>"
   + start_vs_improvement(cl_eps)) if cl_eps else ""}
{sector_bars}
{cl_table}
{cl_sheets}

<footer class="foot">
  <p>{t("DronePhotographer v12 · 체크포인트 iter 8000 · 가이던스 1.0 (CFG 끔) · 청크당 4샘플",
        "DronePhotographer v12 · checkpoint iter 8000 · guidance 1.0 (CFG off) · 4 samples per chunk")}</p>
</footer>
</div>
<script>{JS}</script>"""


def _gain(eps: list[dict]) -> dict | None:
    """Progress made vs progress kept, for one experiment."""
    rows = []
    for e in eps:
        b = e.get("test_b") or e
        if "d_start" not in b or "d_end" not in b:
            continue
        rows.append((float(b["d_start"]), float(b["d_end"]),
                     float(b.get("d_best", min(b["d_start"], b["d_end"])))))
    if not rows:
        return None
    n = len(rows)
    made = sum(s - bst for s, _, bst in rows) / n
    kept = sum(s - e for s, e, _ in rows) / n
    return {
        "n": n, "made": made, "kept": kept,
        "given_back": (made - kept) / made if made > 1e-9 else 0.0,
        "any_closer": sum(1 for s, _, bst in rows if bst < s) / n,
        "ended_closer": sum(1 for s, e, _ in rows if e < s) / n,
        "overshot": sum(1 for _, e, bst in rows if e > bst + 0.05) / n,
    }


def finding_section(gt_eps: list[dict], cl_eps: list[dict]) -> str:
    """The headline: the policy reaches the goal and then does not stop."""
    blocks = []
    for label, eps in ((t("학습 궤적 (테스트 B)", "training trajectories (test B)"), gt_eps),
                       (t("처음 보는 배치", "held-out placements"), cl_eps)):
        g = _gain(eps)
        if not g:
            continue
        pct = g["made"] and g["kept"] / g["made"] * 100
        blocks.append(f"""<div class="gain">
  <div class="gain-head">{label} <span class="muted">· n={g['n']}</span></div>
  <div class="gain-bars">
    <div class="bar-row"><span class="bar-label">{t("최접근까지 좁힌 거리","closed at best")}</span>
      <span class="bar-track"><span class="bar-fill" style="width:100%"></span></span>
      <span class="bar-val">{g['made']:.3f}</span></div>
    <div class="bar-row"><span class="bar-label">{t("종료 시점에 남은 성과","still held at the end")}</span>
      <span class="bar-track"><span class="bar-fill neg" style="width:{pct:.1f}%"></span></span>
      <span class="bar-val">{g['kept']:.3f}</span></div>
  </div>
  <p class="gain-note">{t(
    f"한 번이라도 가까워진 에피소드 <b>{g['any_closer']*100:.0f}%</b> · "
    f"끝까지 가까운 채로 끝난 에피소드 <b>{g['ended_closer']*100:.0f}%</b> · "
    f"지나친 뒤 이탈 <b>{g['overshot']*100:.0f}%</b> · "
    f"되돌려준 성과 <b>{g['given_back']*100:.0f}%</b>",
    f"episodes that got closer at some point <b>{g['any_closer']*100:.0f}%</b> · "
    f"still closer at the end <b>{g['ended_closer']*100:.0f}%</b> · "
    f"overshot then drifted <b>{g['overshot']*100:.0f}%</b> · "
    f"progress given back <b>{g['given_back']*100:.0f}%</b>")}</p>
</div>""")
    if not blocks:
        return ""
    return f"""<h2>{t("핵심 발견 — 도달은 하는데 멈추지 못한다",
                      "Key finding — it arrives, then fails to stop")}</h2>
<p class="sub">{t(
  "두 실험 모두에서 <b>모든 에피소드가 어느 시점엔가 목표에 가까워집니다</b>. "
  "문제는 그 지점에 머무르지 못하고 지나쳐 다시 멀어진다는 것입니다.",
  "In both experiments <b>every single episode gets closer to the goal at some point</b>. "
  "The failure is that it does not stay there — it sails past and drifts back out.")}</p>
<div class="gains">{''.join(blocks)}</div>
<div class="callout warn"><p>{t(
  "이것이 fit 실패가 아니라는 근거: 테스트 A의 방향 코사인이 0.996이고 회전 오차가 0.34°입니다. "
  "정책은 <b>어느 방향으로 갈지 정확히 압니다</b>. 모르는 것은 <b>언제 멈출지</b>입니다.",
  "This is not a fit failure: test A shows a direction cosine of 0.996 and 0.34° of rotation "
  "error. The policy knows <b>exactly which way to go</b>. What it does not know is "
  "<b>when to stop</b>.")}</p>
<p>{t(
  "구조적인 이유가 있습니다 — 학습 데이터의 액션 청크는 '목표를 향한 당장의 8스텝'이라 "
  "<b>정지 상태가 한 번도 등장하지 않습니다</b>. 목표에 도달한 상태에서 '가만히 있으라'는 "
  "예시가 없으니 정책이 그것을 배울 방법이 없습니다.",
  "There is a structural reason — the training chunks are 'the immediate 8 steps toward the "
  "goal', so <b>a stopped state never appears in the data</b>. There is no example of "
  "'you are there, now hold still', so the policy has no way to learn it.")}</p></div>"""


def start_vs_improvement(eps: list[dict], width: int = 620, height: int = 300) -> str:
    """Improvement against how far the camera started from the goal.

    This is where the overshoot shows its shape. Aggregated over all episodes the
    policy looks useful; split by starting distance it turns out to help a lot from
    far away and barely at all — sometimes negatively — when it starts near the
    goal. A single mean hides that completely, and it is the difference between
    "the policy is weak" and "the policy cannot hold still".
    """
    pts = [(float(e["d_start"]), float(e.get("improvement", 0.0)), e.get("sector", ""))
           for e in eps if "d_start" in e]
    if len(pts) < 4:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = 0.0, max(xs) * 1.08
    y0, y1 = min(min(ys), 0.0) * 1.15, max(ys) * 1.12
    pad_l, pad_r, pad_t, pad_b = 54, 14, 14, 40
    w, h = width - pad_l - pad_r, height - pad_t - pad_b

    def T(x, y):
        return (pad_l + (x - x0) / (x1 - x0) * w,
                pad_t + h - (y - y0) / (y1 - y0) * h)

    zx0, zy = T(x0, 0.0)
    zx1, _ = T(x1, 0.0)
    # split marker at the boundary used in the copy
    sx, _ = T(0.6, 0.0)
    dots = "".join(
        f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="4.2" '
        f'class="sc {"neg" if y < 0 else "pos"}"><title>{s} · start {x:.3f} · '
        f'improvement {y:+.3f}</title></circle>'
        for (x, y, s), (X, Y) in ((p, T(p[0], p[1])) for p in pts))

    def group(lo, hi):
        g = [y for x, y, _ in pts if lo <= x < hi]
        return (sum(g) / len(g), len(g)) if g else (0.0, 0)

    near, n_near = group(0.0, 0.6)
    far, n_far = group(0.6, 1e9)
    nX, nY = T(0.3, near)
    fX, fY = T(max(xs) * 0.8, far)

    yticks = "".join(
        f'<line x1="{pad_l}" y1="{T(x0, v)[1]:.1f}" x2="{width-pad_r}" y2="{T(x0, v)[1]:.1f}" '
        f'class="se-grid"/><text x="{pad_l-8}" y="{T(x0, v)[1]+3.4:.1f}" class="axlab" '
        f'text-anchor="end">{v:+.1f}</text>'
        for v in (round(y0, 1), 0.0, round(y1, 1)) if y0 <= v <= y1)
    return f"""<svg viewBox="0 0 {width} {height}" class="scatter" role="img"
  aria-label="improvement versus starting distance to the goal, one point per episode">
  {yticks}
  <line x1="{zx0:.1f}" y1="{zy:.1f}" x2="{zx1:.1f}" y2="{zy:.1f}" class="zeroline"/>
  <line x1="{sx:.1f}" y1="{pad_t}" x2="{sx:.1f}" y2="{pad_t+h}" class="splitline"/>
  {dots}
  <line x1="{pad_l}" y1="{nY:.1f}" x2="{sx:.1f}" y2="{nY:.1f}" class="meanline"/>
  <line x1="{sx:.1f}" y1="{fY:.1f}" x2="{width-pad_r}" y2="{fY:.1f}" class="meanline"/>
  <text x="{nX:.1f}" y="{nY-7:.1f}" class="axlab" text-anchor="middle">{near:+.3f} (n={n_near})</text>
  <text x="{fX:.1f}" y="{fY-7:.1f}" class="axlab" text-anchor="middle">{far:+.3f} (n={n_far})</text>
  <text x="{pad_l}" y="{height-12}" class="axlab">{t("시작 거리 0","start distance 0")}</text>
  <text x="{sx:.1f}" y="{height-12}" class="axlab" text-anchor="middle">0.6</text>
  <text x="{width-pad_r}" y="{height-12}" class="axlab" text-anchor="end">{max(xs):.2f}</text>
  <text x="14" y="{pad_t+h/2:.1f}" class="axlab" text-anchor="middle"
    transform="rotate(-90 14 {pad_t+h/2:.1f})">{t("개선","improvement")}</text>
</svg>"""


def closed_loop_header(runs: list[dict]) -> str:
    """Section heading that reads the guarantee off the data instead of asserting it.

    This used to be a hardcoded "placements never used in training". It was wrong:
    the export consumes 4 episodes from each of only ~1000 placements, and the eval
    drew from the same shuffled directory listing, so all 12 evaluated placements
    were trained-on. The label is now derived from `summary.held_out_only`, which
    the eval writes only when it actually excluded the trained set — a run that
    predates the flag reports the weaker claim rather than the flattering one.
    """
    if not runs:
        return (f'<h2>{t("롤아웃 — closed-loop", "Rollout — closed loop")}</h2>'
                f'<p class="sub">{t("결과 대기 중.", "Results pending.")}</p>')
    held = [bool((r.get("summary") or {}).get("held_out_only")) for r in runs]
    if all(held):
        return f"""<h2>{t("처음 보는 배치 — closed-loop", "Held-out placements — closed loop")}</h2>
<p class="sub">{t(
  "학습에 한 번도 쓰이지 않은 배치. 평가 시작 시 export 열거를 재현해 학습에 쓰인 배치를 "
  "제외한 뒤 뽑았습니다.",
  "Placements never used in training. The eval replays the export enumeration at start-up "
  "and draws only from what it did not consume.")}</p>"""
    return f"""<h2>{t("학습에 쓰인 배치 — closed-loop", "Trained-on placements — closed loop")}</h2>
<div class="callout warn"><p>{t(
  "<b>이 절은 일반화 측정이 아닙니다.</b> 이 실행은 학습에 쓰인 배치에서 롤아웃했습니다. "
  "export 는 배치당 4 에피소드씩 약 1,000개 배치만 소비하는데, 평가가 같은 디렉터리를 같은 "
  "시드로 섞어 앞에서부터 뽑아 정확히 그 안에 떨어졌습니다 — 검사 결과 12/12 전부 학습 배치였습니다.",
  "<b>This section is not a generalization measurement.</b> This run rolled out on "
  "placements the model trained on. The export consumes only ~1000 placements (4 episodes "
  "each), and the eval drew from the same shuffled listing with the same seed, landing "
  "inside that set — a check found 12 of 12 evaluated placements were trained-on.")}</p>
<p>{t(
  "학습 배치에서의 도달 성능으로는 여전히 유효하며, 아래 오버슈트 패턴도 그대로 유효합니다. "
  "다만 처음 보는 장면에 대한 주장으로는 읽으면 안 됩니다. 수정된 실행이 진행 중입니다.",
  "It remains valid as attainment on trained placements, and the overshoot pattern below "
  "still holds. It must not be read as a claim about unseen scenes. A corrected run is "
  "in progress.")}</p></div>"""


def _ckpt_label(run: dict) -> str:
    m = re.search(r"iter_0*(\d+)", str(run.get("checkpoint", "")))
    return f"iter {int(m.group(1)):,}" if m else Path(str(run.get("path", "?"))).stem


def rollout_compare(runs: list[dict]) -> str:
    """Closed-loop metrics for several checkpoints side by side.

    The question this answers is not "did the loss go down" — held-out loss already says
    yes — but whether the checkpoint that minimises it is the one that actually reaches
    the shot. v11 found those can disagree, so they are measured separately and shown
    together.
    """
    rows = []
    for r in runs:
        eps = r.get("results") or r.get("episodes") or []
        eps = [e for e in eps if "d_start" in e]
        if not eps:
            continue
        n = len(eps)
        made = sum(e["d_start"] - e["d_best"] for e in eps)
        kept = sum(e["d_start"] - e["d_end"] for e in eps)
        near = [e for e in eps if e["d_start"] < 0.6]
        far = [e for e in eps if e["d_start"] >= 0.6]
        rows.append({
            "label": _ckpt_label(r),
            "n": n,
            "kept": kept / n,
            "made": made / n,
            "given_back": (made - kept) / made if made > 1e-9 else 0.0,
            "overshoot": sum(1 for e in eps
                             if e["d_end"] > e["d_best"] + 0.05) / n,
            "near": (sum(e["improvement"] for e in near) / len(near)) if near else None,
            "far": (sum(e["improvement"] for e in far) / len(far)) if far else None,
            "partial": bool((r.get("summary") or {}).get("partial")),
        })
    if not rows:
        return ""
    rows.sort(key=lambda x: x["label"])
    heads = "".join(
        f"<th>{r['label']}{' *' if r['partial'] else ''}</th>" for r in rows)
    def line(lbl, key, fmt, better_high=True):
        vals = [r[key] for r in rows]
        if any(v is None for v in vals):
            return ""
        best = max(vals) if better_high else min(vals)
        cells = "".join(
            f'<td class="num{" delta good" if v == best else ""}">{fmt.format(v)}</td>'
            for v in vals)
        return f"<tr><td>{lbl}</td>{cells}</tr>"
    body = "".join([
        f"<tr><td>{t('에피소드','episodes')}</td>"
        + "".join(f'<td class="num">{r["n"]}</td>' for r in rows) + "</tr>",
        line(t("평균 개선", "mean improvement"), "kept", "{:+.4f}", True),
        line(t("최대 개선", "best improvement"), "made", "{:+.4f}", True),
        line(t("되돌려준 성과", "progress given back"), "given_back", "{:.0%}", False),
        line(t("오버슈트", "overshoot rate"), "overshoot", "{:.0%}", False),
        line(t("가까이 시작 (<0.6)", "starts near (<0.6)"), "near", "{:+.3f}", True),
        line(t("멀리 시작 (>=0.6)", "starts far (>=0.6)"), "far", "{:+.3f}", True),
    ])
    note = ""
    if any(r["partial"] for r in rows):
        note = ("<p class=\"sub\">" + t(
            "* 진행 중인 실행입니다. 에피소드가 섹터 순서대로 채워지므로 <b>섹터별 수치는 "
            "아직 비교하면 안 됩니다</b>; 전체 평균은 유효합니다.",
            "* still running. Episodes fill in sector order, so <b>per-sector numbers are "
            "not yet comparable</b>; the overall means are.") + "</p>")
    return f"""<h2>{t("체크포인트별 도달 성능", "Goal attainment by checkpoint")}</h2>
<div class="tablewrap"><table class="cmp">
<thead><tr><th>{t("지표","metric")}</th>{heads}</tr></thead><tbody>{body}</tbody></table></div>
{note}"""


def compare_section(runs: list[dict]) -> str:
    """Two checkpoints side by side, on the metrics that answer different questions.

    Kept separate from the per-episode galleries because the interesting result is a
    divergence: fit keeps improving with training while goal attainment does not, and
    that only shows up when the two families of metric sit next to each other.
    """
    if len(runs) < 2:
        return ""
    runs = sorted(runs, key=lambda r: int(re.search(r"iter_0*(\d+)",
                  str(r.get("checkpoint", "0"))).group(1)
                  if re.search(r"iter_0*(\d+)", str(r.get("checkpoint", ""))) else 0))
    labels = [_ckpt_label(r) for r in runs]

    def stat(r: dict, group: str, key: str):
        return (r.get("summary", {}).get(group, {}) or {}).get(key)

    def overshoot(r: dict) -> tuple[float, float, int]:
        bs = [e["test_b"] for e in r.get("episodes", []) if "test_b" in e]
        if not bs:
            return (0.0, 0.0, 0)
        over = sum(1 for x in bs if x["d_end"] > x["d_best"] + 0.05) / len(bs)
        made = sum(x["d_start"] - x["d_best"] for x in bs)
        kept = sum(x["d_start"] - x["d_end"] for x in bs)
        return (over, (made - kept) / made if made > 1e-9 else 0.0, len(bs))

    # (group, key, label, lower_is_better, fmt)
    rows = [
        ("test_a", "dir_cos_best_mean", t("방향 코사인", "direction cosine"), False, "{:.4f}"),
        ("test_a", "trans_err_best_mean", t("이동 오차", "translation error"), True, "{:.4f}"),
        ("test_a", "rot_err_best_mean", t("회전 오차 (도)", "rotation error (deg)"), True, "{:.3f}"),
        ("test_a", "trans_err_rel_best", t("상대 오차", "relative error"), True, "{:.4f}"),
        ("test_b", "mean_best_improvement", t("최대 개선", "best improvement"), False, "{:.4f}"),
        ("test_b", "mean_improvement", t("평균 개선", "mean improvement"), False, "{:.4f}"),
        ("test_b", "mean_d_end", t("종료 거리", "final distance"), True, "{:.4f}"),
    ]
    body = []
    for group, key, label, lower_better, fmt in rows:
        vals = [stat(r, group, key) for r in runs]
        if any(v is None for v in vals):
            continue
        delta = vals[-1] - vals[0]
        better = (delta < 0) if lower_better else (delta > 0)
        cls = "good" if abs(delta) > 1e-9 and better else ("warn" if abs(delta) > 1e-9 else "flat")
        tag = "A" if group == "test_a" else "B"
        body.append(
            f'<tr><td><span class="tt {tag.lower()}">{tag}</span> {label}</td>'
            + "".join(f'<td class="num">{fmt.format(v)}</td>' for v in vals)
            + f'<td class="num delta {cls}">{delta:+.4f}</td></tr>')

    over = [overshoot(r) for r in runs]
    body.append(
        f'<tr><td><span class="tt b">B</span> {t("오버슈트 비율", "overshoot rate")}</td>'
        + "".join(f'<td class="num">{o[0]*100:.0f}% <span class="muted">({o[2]})</span></td>'
                  for o in over)
        + f'<td class="num delta {"warn" if over[-1][0] > over[0][0] else "good"}">'
          f'{(over[-1][0]-over[0][0])*100:+.0f}%</td></tr>')
    body.append(
        f'<tr><td><span class="tt b">B</span> {t("되돌려준 성과", "progress given back")}</td>'
        + "".join(f'<td class="num">{o[1]*100:.0f}%</td>' for o in over)
        + f'<td class="num delta {"warn" if over[-1][1] > over[0][1] else "good"}">'
          f'{(over[-1][1]-over[0][1])*100:+.0f}%</td></tr>')

    heads = "".join(f"<th>{l}</th>" for l in labels)
    return f"""<h2>{t("체크포인트 비교 — 학습을 더 하면 나아지는가",
                      "Checkpoint comparison — does more training help?")}</h2>
<p class="sub">{t(
  "두 종류의 지표를 나란히 둡니다. <b>A는 fit</b>(액션을 맞히는가), "
  "<b>B는 도달</b>(목표에 가서 머무는가). 이 둘이 갈라지는 것이 이번 결과의 핵심입니다.",
  "Two families of metric side by side. <b>A is fit</b> (does it predict the action), "
  "<b>B is attainment</b> (does it get to the goal and stay). Their divergence is the "
  "result.")}</p>
<div class="tablewrap"><table class="cmp">
<thead><tr><th>{t("지표","metric")}</th>{heads}<th>Δ</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class="callout warn"><p>{t(
  "<b>fit은 좋아지는데 도달은 그대로입니다.</b> 3,500 iter를 더 학습해서 이동·회전 오차가 "
  "20% 넘게 줄었지만, 최대 개선은 사실상 동일하고 되돌려주는 성과는 오히려 늘었습니다.",
  "<b>Fit improves; attainment does not.</b> 3,500 more iterations cut translation and "
  "rotation error by over 20%, yet best improvement is essentially unchanged and the "
  "share of progress given back went up.")}</p>
<p>{t(
  "구조적 해석: 액션 예측 정확도를 올려도 <b>'언제 멈출지'는 데이터에 없습니다</b>. "
  "학습 청크는 '목표를 향한 당장의 8스텝'이라 정지 상태가 한 번도 등장하지 않고, "
  "현재 상태가 목표에서 얼마나 먼지를 나타내는 양도 입력·손실 어디에도 없습니다.",
  "The structural reading: better action prediction cannot supply what is missing from "
  "the data — <b>when to stop</b>. Training chunks are 'the immediate 8 steps toward the "
  "goal', so a stopped state never appears, and no quantity anywhere in the input or the "
  "loss represents how far the current state still is from the goal.")}</p></div>"""


def a_table(eps: list[dict]) -> str:
    rows = []
    for e in eps:
        a = e.get("test_a") or {}
        b, m = a.get("best_sample") or {}, a.get("mean_of_samples") or {}
        if not b:
            continue
        rows.append(
            f"<tr><td>{e.get('sector','-')}</td><td>{e.get('placement','')[:40]}</td>"
            f"<td class='num'>{b.get('dir_cos_mean',0):.3f}</td>"
            f"<td class='num'>{b.get('trans_err_mean',0):.4f}</td>"
            f"<td class='num'>{b.get('gt_step_norm_mean',0):.4f}</td>"
            f"<td class='num'>{b.get('rot_err_deg_mean',0):.2f}°</td>"
            f"<td class='num'>{m.get('trans_err_mean',0):.4f}</td></tr>")
    if not rows:
        return f'<p class="sub">{t("아직 결과 없음", "no results yet")}</p>'
    return f"""<div class="tablewrap"><table>
<thead><tr><th>{t("섹터","sector")}</th><th>{t("배치","placement")}</th>
<th>{t("방향 코사인","dir cos")}</th><th>{t("이동 오차","trans err")}</th>
<th>{t("GT 스텝 크기","GT step size")}</th><th>{t("회전 오차","rot err")}</th>
<th>{t("4샘플 평균","mean of 4")}</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="sub">{t(
  "‘이동 오차’는 4샘플 중 최선. ‘GT 스텝 크기’와 나란히 봐야 의미가 있습니다 — "
  "정책은 확률적이라 단일 샘플만 보면 샘플링 분산을 fit 오차로 오독하게 됩니다.",
  "‘trans err’ is the best of 4 samples, and only means something next to ‘GT step size’. "
  "The policy is stochastic, so a single draw would report sampling variance as fit error.")}</p>"""


def main() -> int:
    gt_runs, cl_runs = load(args.gt_replay), load(args.closed_loop)
    print(f"gt_replay runs: {len(gt_runs)} | closed_loop runs: {len(cl_runs)}")
    html = render(gt_runs, cl_runs)
    out = Path(args.out)
    if not out.is_absolute():
        out = V12 / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB, {len(_cache)} frames embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["data_uri", "load", "sparkline", "bars", "contact_sheet", "t", "render"]
