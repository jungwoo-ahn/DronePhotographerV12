"""Qualitative rollouts: 3-D camera paths of Ours vs baselines against the GT goal pose, and their last frames.

Episodes come from benchmark v2 (runs/bench_v2, v7 render, seed 0). They were picked for a readable scene and
a clear gap: Ours succeeds (d <= 0.10, subject visible) while pi0.5, GR00T, text-to-action and diffusion policy
all miss. Image-to-action is left out on purpose: it is given the GT goal image, which no deployment has.

The 3-D view is a close-up of where the episode is decided: a virtual camera behind and above the GT goal
camera, looking at the subject, fitted to the last steps of every path. Drop lines give each end camera's height.

    .venv-analysis/bin/python figures/qualitative/make_fig_rollouts.py [--root <main checkout>] [--rows i,j,..]
writes fig_rollouts.{pdf,svg,png} next to this file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overview"))
from make_fig2 import (BLUE, BLUE_D, GREEN, GRID, INK, MUTED, PANEL, Canvas, Proj,  # noqa: E402
                       draw_frustum, load)

POLICIES = [  # key in scores.json, rollout file stem, label, colour, path width
    ("ours", "ours_v6_45k", "Ours", GREEN, 1.9),
    ("pi05", "pi05", "π0.5", "#C4553C", 1.0),
    ("groot", "groot", "GR00T", "#D99A26", 1.0),
    ("t2a", "t2a_caption_45k", "Text→action", "#8A6BAA", 1.0),
    ("dp", "dp", "Diffusion policy", "#7C8792", 1.0),
]
EPISODES = [  # key, title, view (azimuth offset from the side-on direction, elevation)
    ("Chill-Camping_150d7263__young-guy-keeps-_ad9f69fc-9bc7-461f-95c3-1026c06b6797:p07:s02:g24",
     "lakeside camp", (-20, 55)),
    ("Nature-Snowy-Mountain-Village-Retreat_f884ef3e__Guy-in-yellow-jacket_660f924e:p08:s17:g01",
     "snowy village", (-20, 55)),
    ("Desert-Roadside-Repair-Garage_a1a2b2ac__young-guy-keeps-_ad9f69fc-9bc7-461f-95c3-1026c06b6797:p07:s29:g13",
     "desert roadside", (-20, 55)),
    ("office-buildingwarehouse-creative-headquarters__Young-Tall-Guy-Talking-by-Phone_7e126859:p07:s15:g02",
     "office atrium", (-20, 55)),
    ("Chill-Camping_150d7263__Hahaha-woman-on-chair_50c9329e:p05:s10:g25",
     "lakeside reeds", (-20, 55)),
]
DEFAULT_ROWS = [0, 1, 2, 4]
TAIL = 4            # steps of each path the close-up is fitted to
RH, PW3, FX0, FW, GAP = 1.10, 1.92, 2.04, 0.80, 0.045      # row height, 3-D panel width, frame strip
W = 7.16


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nas_main/jungwooahn/projects/DronePhotographerV12")
    ap.add_argument("--rows", default=None, help="comma-separated indices into EPISODES")
    ap.add_argument("--out", default="fig_rollouts")
    return ap.parse_args()


def episode(bench, scores, rollouts, key):
    out = {}
    for pk, stem, *_ in POLICIES:
        i, e = next((i, e) for i, e in enumerate(rollouts[pk]) if e["key"] == key)
        out[pk] = dict(ep=e, idx=i, d=scores[pk][key]["d_end"], succ=scores[pk][key]["succ"],
                       final=bench / f"frames_rollout_{stem}/ep{i:03d}_chunk{e['executed_chunks']}.jpg")
    return out


def pose_pts(e):
    return np.array([p["position"] for p in e["step_poses"]])


def row(cv, y0, rh, data, title, view, first):
    ref = data["ours"]["ep"]
    subj = np.array(ref["subject_center"]); hgt = ref["subject_height"]
    foot = subj - [0, 0, hgt / 2]
    goal = ref["goal_pose"]
    gpos = np.asarray(goal["position"])
    paths = [pose_pts(data[pk]["ep"]) for pk, *_ in POLICIES]
    ends = np.array([P[-1] for P in paths])

    # ---- close-up 3-D: a camera behind and above the GT goal camera, looking at the subject
    cv.box(0.04, y0, PW3, rh, fc=PANEL, ec="none", r=0.08, z=0.5)
    rect = (0.06, y0 + 0.03, PW3 - 0.04, rh - 0.30)
    clip = Rectangle(rect[:2], rect[2], rect[3], transform=cv.ax.transData)
    n_lines, n_patches, n_coll = len(cv.ax.lines), len(cv.ax.patches), len(cv.ax.collections)
    # look across the paths: perpendicular to start->goal, from the side away from the subject,
    # so the whole route spans the panel's width instead of running into the screen
    s0 = np.asarray(ref["step_poses"][0]["position"])
    run = (gpos - s0)[:2]
    side = np.array([-run[1], run[0]])
    if np.dot(side, (subj - (s0 + gpos) / 2)[:2]) > 0:
        side = -side
    az = np.degrees(np.arctan2(side[1], side[0])) + view[0]
    # fit every path end to end, the GT camera and the subject
    fit_pts = np.vstack(paths + [gpos[None], foot[None], (foot + [0, 0, hgt])[None]])
    span = max(np.ptp(fit_pts, axis=0).max(), 0.3)
    center = fit_pts.mean(0)
    pr = Proj(center, az_deg=az, el_deg=view[1], dist=span * 40, fov_deg=3)     # near-orthographic close-up
    pr.fit(fit_pts, (rect[0] + 0.05, rect[1] + 0.08, rect[2] - 0.10, rect[3] - 0.12), margin=0.14)

    # ground grid, its spacing picked for the close-up's scale
    step = 0.25 if span < 1.5 else 0.5 if span < 4 else 1.0
    lo, hi = fit_pts[:, :2].min(0) - span * 1.5, fit_pts[:, :2].max(0) + span * 1.5
    for v in np.arange(np.floor(lo[0] / step) * step, hi[0], step):
        q = pr(np.c_[[v, v], [lo[1], hi[1]], [foot[2]] * 2])
        cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1).set_clip_path(clip)
    for v in np.arange(np.floor(lo[1] / step) * step, hi[1], step):
        q = pr(np.c_[[lo[0], hi[0]], [v, v], [foot[2]] * 2])
        cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1).set_clip_path(clip)

    fs = float(np.clip(span * 0.075, 0.08, 0.5))   # frustum size (m), scaled to what the view shows

    def drop(p, col, z):
        a, b = pr(p)[0], pr(np.r_[p[:2], foot[2]])[0]
        cv.line([a[0], b[0]], [a[1], b[1]], color=col, lw=0.45, ls=(0, (1.2, 1.2)), z=z, alpha=0.8).set_clip_path(clip)
        e = Ellipse(b, 0.045, 0.018, fc=col, ec="none", alpha=0.5, zorder=z)
        cv.ax.add_patch(e)
        e.set_clip_path(clip)

    # GT goal camera: drop line, look-at ray, frustum
    drop(gpos, BLUE, 2.5)
    gq, sq = pr(gpos)[0], pr(subj)[0]
    cv.line([gq[0], sq[0]], [gq[1], sq[1]], color=BLUE, lw=0.5, ls=(0, (2, 2)), z=2.6).set_clip_path(clip)
    # the subject is usually outside the close-up: point to it where its look-at ray leaves the panel
    u = (sq - gq) / (np.linalg.norm(sq - gq) + 1e-9)
    inside = (rect[0] + 0.12 <= sq[0] <= rect[0] + rect[2] - 0.12) and (rect[1] + 0.08 <= sq[1] <= rect[1] + rect[3] - 0.10)
    ph = pr(foot + [0, 0, hgt])[0][1] - pr(foot)[0][1]
    if inside and ph < 0.45 * rect[3]:
        cv.person(pr(foot)[0], pr(foot + [0, 0, hgt])[0], z=6)
    elif inside:
        cv.ax.scatter(*sq, s=10, color=BLUE_D, zorder=10, lw=0)
        cv.text(sq[0], sq[1] + 0.10, "subject", size=5.0, color=BLUE_D, z=10,
                bbox=dict(fc=PANEL, ec="none", pad=0.6, alpha=0.9))
    else:
        ts = [((rect[0] + 0.14 if u[0] < 0 else rect[0] + rect[2] - 0.14) - gq[0]) / u[0] if abs(u[0]) > 1e-6 else 1e9,
              ((rect[1] + 0.10 if u[1] < 0 else rect[1] + rect[3] - 0.10) - gq[1]) / u[1] if abs(u[1]) > 1e-6 else 1e9]
        e = gq + u * min(t_ for t_ in ts if t_ > 0) * 0.78     # a little inside, clear of the scale bar
        cv.text(e[0], e[1], "subject →" if u[0] >= 0 else "← subject", size=5.0, color=BLUE_D, z=10,
                ha="right" if u[0] >= 0 else "left", va="center",
                bbox=dict(fc=PANEL, ec="none", pad=0.6, alpha=0.9))
    # scale bar
    p0 = center.copy(); p0[2] = foot[2]
    a0, a1 = pr(p0)[0], pr(p0 + [step * 2, 0, 0])[0]
    sx0 = rect[0] + rect[2] - 0.10 - np.linalg.norm(a1 - a0)
    sy = rect[1] + rect[3] - 0.05
    cv.line([sx0, rect[0] + rect[2] - 0.10], [sy] * 2, color=INK, lw=0.8, z=10)
    cv.text(sx0 - 0.04, sy, f"{step * 2:g} m", size=4.8, ha="right", va="center", z=10)

    # paths (clipped where they leave the close-up) and end cameras; baselines first so Ours sits on top
    for (pk, _, lab, col, lw), P in reversed(list(zip(POLICIES, paths))):
        z = 7 if pk == "ours" else 4
        q = pr(P)
        cv.line(q[:, 0], q[:, 1], color=col, lw=lw, z=z, alpha=0.95).set_clip_path(clip)
        drop(P[-1], col, 2.4)
        draw_frustum(cv, pr, data[pk]["ep"]["step_poses"][-1], fs, ec=col, fc="white",
                     lw=1.0 if pk == "ours" else 0.65, z=z + 0.2)
    start = ref["step_poses"][0]
    As, _ = draw_frustum(cv, pr, start, fs, ec=INK, fc="#F1F1F3", lw=0.7, z=8)
    drop(np.asarray(start["position"]), INK, 2.4)
    cv.text(As[0], As[1] + 0.09, "start", size=5.0, z=10, bbox=dict(fc=PANEL, ec="none", pad=0.5, alpha=0.85))
    Ag, _ = draw_frustum(cv, pr, goal, fs * 1.08, ec=BLUE, fc="none", lw=1.2, z=9)
    cv.ax.scatter(*Ag, s=34, marker="*", color=BLUE, zorder=10, lw=0.4, edgecolors="white")
    for art in cv.ax.lines[n_lines:] + cv.ax.patches[n_patches:] + cv.ax.collections[n_coll:]:
        art.set_clip_path(clip)                   # everything 3-D stays inside the close-up
    cv.text(Ag[0], Ag[1] - 0.13, "GT", size=5.2, color=BLUE_D, z=10, weight="semibold",
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))

    cv.text(0.12, y0 + rh - 0.10, title, size=6.0, weight="semibold", ha="left", z=10)
    cv.text(0.12, y0 + rh - 0.21, f"start d = {ref['d_start']:.2f}", size=5.0, color=MUTED, ha="left", z=10)

    # ---- last frames: GT goal, then each policy
    fh = FW * 0.75
    fy = y0 + (rh - fh) / 2
    cols = [("goal", "GT goal", BLUE, None)] + [(pk, lab, col, data[pk]) for pk, _, lab, col, _ in POLICIES]
    for j, (pk, lab, col, dd) in enumerate(cols):
        x = FX0 + j * (FW + GAP)
        img = load(ref["goal_frame_image"] if pk == "goal" else dd["final"], 600)
        cv.image(img, x, fy, FW, fh, ec=col, lw=1.6 if pk in ("goal", "ours") else 1.0, shadow=True)
        if first:
            cv.text(x + FW / 2, fy + fh + 0.09, lab, size=6.0, color=col if pk != "dp" else MUTED,
                    weight="semibold" if pk in ("goal", "ours") else "normal")
        if dd is not None:
            cv.text(x + FW - 0.03, fy + 0.03, f"d {dd['d']:.2f}", size=5.0, ha="right", va="bottom",
                    color="white", family="IBM Plex Mono", z=7,
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.3",
                              fc=(0.14, 0.52, 0.35, 0.9) if dd["succ"] else (0, 0, 0, 0.5), ec="none"))


def main():
    a = args()
    root = Path(a.root)
    bench = root / "runs/bench_v2"
    scores = json.load(open(bench / "scores.json"))
    rollouts = {pk: json.load(open(bench / f"rollout_{stem}.json"))["episodes"] for pk, stem, *_ in POLICIES}
    rows = [int(r) for r in a.rows.split(",")] if a.rows else DEFAULT_ROWS
    H = 0.06 + len(rows) * RH + (len(rows) - 1) * 0.08 + 0.26
    cv = Canvas()
    cv.fig.set_size_inches(W, H)
    cv.ax.set_xlim(0, W)
    cv.ax.set_ylim(0, H)
    for r, ei in enumerate(rows):
        key, title, view = EPISODES[ei]
        y0 = H - 0.06 - (r + 1) * RH - r * 0.08
        row(cv, y0, RH, episode(bench, scores, rollouts, key), title, view, first=(r == 0))
    lx = 0.10
    for pk, _, lab, col, lw in POLICIES:
        cv.line([lx, lx + 0.16], [0.08, 0.08], color=col, lw=lw)
        cv.text(lx + 0.20, 0.08, lab, size=5.4, ha="left")
        lx += 0.30 + 0.055 * len(lab)
    cv.ax.scatter([lx + 0.05], [0.08], s=22, marker="*", color=BLUE, lw=0)
    cv.text(lx + 0.12, 0.08, "GT goal camera", size=5.4, ha="left")
    cv.text(W - 0.06, 0.08, "d: distance to the GT goal (success d ≤ 0.10 + subject visible, green)", size=5.2,
            color=MUTED, ha="right")
    for ext in (("png",) if a.rows else ("pdf", "svg", "png")):
        cv.fig.savefig(HERE / f"{a.out}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", HERE / f"{a.out}.png")


if __name__ == "__main__":
    main()
