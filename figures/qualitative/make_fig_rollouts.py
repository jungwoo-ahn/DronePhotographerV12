"""Qualitative rollouts: 3-D camera paths of Ours vs baselines against the GT goal pose, and their last frames.

Episodes come from benchmark v2 (runs/bench_v2, v7 render, seed 0). They were picked for a readable scene and
a clear gap: Ours succeeds (d <= 0.10, subject visible) while pi0.5, GR00T, text-to-action and diffusion policy
all miss. Image-to-action is left out on purpose: it is given the GT goal image, which no deployment has.

    .venv-analysis/bin/python figures/qualitative/make_fig_rollouts.py [--root <main checkout>]
writes fig_rollouts.{pdf,svg,png} next to this file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overview"))
from make_fig2 import (BLUE, BLUE_D, FAINT, GREEN, GREEN_D, GRID, INK, MUTED, PANEL, Canvas, Proj,  # noqa: E402
                       draw_frustum, load)

W, H = 7.16, 2.62
GOAL_LAB = (0.0, -0.12)
RH, PW3, FX0, FW, GAP = 1.10, 1.92, 2.04, 0.80, 0.045      # row height, 3-D panel width, frame strip

POLICIES = [  # key in scores.json, rollout file stem, label, colour, path width
    ("ours", "ours_v6_45k", "Ours", GREEN, 1.9),
    ("pi05", "pi05", "π0.5", "#C4553C", 1.0),
    ("groot", "groot", "GR00T", "#D99A26", 1.0),
    ("t2a", "t2a_caption_45k", "Text→action", "#8A6BAA", 1.0),
    ("dp", "dp", "Diffusion policy", "#7C8792", 1.0),
]
EPISODES = [  # key, 3-D view (azimuth, elevation, eye distance), title
    ("Chill-Camping_150d7263__young-guy-keeps-_ad9f69fc-9bc7-461f-95c3-1026c06b6797:p07:s02:g24",
     (-100, 60, 30), "lakeside camp", None),
    ("Nature-Snowy-Mountain-Village-Retreat_f884ef3e__Guy-in-yellow-jacket_660f924e:p08:s17:g01",
     (-60, 50, 30), "snowy village", None),
]
END_FS = 0.32       # final-camera frustum size (m)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nas_main/jungwooahn/projects/DronePhotographerV12")
    ap.add_argument("--views", default=None, help="az,el,dist;az,el,dist to try other 3-D views")
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


def zoom_inset(cv, pr, goal, data, paths, box, view):
    """A second, close 3-D camera on the goal region: each path's tail and its final camera, vs the GT camera."""
    x, y, w, h = box
    gpos = np.asarray(goal["position"])
    zp = Proj(gpos, az_deg=view[0], el_deg=view[1], dist=8, fov_deg=30)
    tails = [P[-ZOOM_STEPS:] for P in paths]
    zp.fit(np.vstack(tails + [gpos[None]]), (x, y, w, h), margin=0.14)
    span = np.ptp(np.vstack(tails + [gpos[None]]), axis=0).max()
    fs = max(span * 0.18, 0.08)     # frustum size in metres, relative to what the zoom shows

    cv.box(x, y, w, h, fc="white", ec=FAINT, lw=0.6, r=0.05, z=11, shadow=True)
    clip = Rectangle((x, y), w, h, transform=cv.ax.transData)
    arts = []
    for (pk, _, lab, col, lw), P, (_, dd) in reversed(list(zip(POLICIES, tails, [(p[0], data[p[0]]) for p in POLICIES]))):
        q = zp(P)
        zz = 13 if pk == "ours" else 12
        arts.append(cv.line(q[:, 0], q[:, 1], color=col, lw=lw * 0.9, z=zz))
        draw_frustum(cv, zp, dd["ep"]["step_poses"][-1], fs, ec=col, fc="white", lw=0.8 if pk == "ours" else 0.55,
                     z=zz + 0.1)
    Ag, _ = draw_frustum(cv, zp, goal, fs * 1.05, ec=BLUE, fc="none", lw=1.0, z=13.5)
    cv.ax.scatter(*Ag, s=34, marker="*", color=BLUE, zorder=14, lw=0.4, edgecolors="white")
    for art in cv.ax.lines[-40:] + cv.ax.patches[-60:] + cv.ax.collections[-10:]:
        if art.get_zorder() >= 12:
            art.set_clip_path(clip)
    cv.text(x + 0.04, y + h - 0.04, "at the goal", size=4.8, color=MUTED, ha="left", va="top", z=15)
    # mark where the zoom looks
    gq = pr(gpos)[0]
    cv.ax.add_patch(Circle(gq, 0.06, fc="none", ec=INK, lw=0.5, zorder=10.5))
    cv.line([gq[0] + 0.06, x], [gq[1], y + h / 2], color=INK, lw=0.4, ls=(0, (1, 1.2)), z=10.4)


def row(cv, y0, rh, data, view, title, first, inset):
    ref = data["ours"]["ep"]
    subj = np.array(ref["subject_center"]); hgt = ref["subject_height"]
    foot = subj - [0, 0, hgt / 2]
    goal = ref["goal_pose"]
    start = ref["step_poses"][0]

    # ---- 3-D paths
    cv.box(0.04, y0, PW3, rh, fc=PANEL, ec="none", r=0.08, z=0.5)
    rect = (0.08, y0 + 0.04, PW3 - 0.12, rh - 0.30)
    pr = Proj(foot + [0, 0, 0.8], az_deg=view[0], el_deg=view[1], dist=view[2], fov_deg=30)
    paths = [pose_pts(data[pk]["ep"]) for pk, *_ in POLICIES]
    pts = np.vstack(paths + [np.asarray(goal["position"])[None], foot[None], (foot + [0, 0, hgt])[None]])
    pr.fit(pts, rect, margin=0.16)
    clip = Rectangle(rect[:2], rect[2], rect[3], transform=cv.ax.transData)

    c = (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2
    R = np.abs(pts[:, :2] - c).max() + 1.5
    for v in np.arange(np.floor(-R), np.ceil(R) + 0.01, 1.0):
        for a, b in (((v, -R), (v, R)), ((-R, v), (R, v))):
            q = pr(np.c_[np.linspace(a[0], b[0], 2) + c[0], np.linspace(a[1], b[1], 2) + c[1], [foot[2]] * 2])
            cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1).set_clip_path(clip)

    # subject
    t = np.linspace(0, 2 * np.pi, 48)
    sh = pr(np.c_[foot[0] + 0.4 * np.cos(t), foot[1] + 0.4 * np.sin(t), np.full(48, foot[2])])
    cv.ax.add_patch(Polygon(sh, closed=True, fc="#CDD1D8", ec="none", zorder=2))
    cv.person(pr(foot)[0], pr(foot + [0, 0, hgt])[0], z=6)

    # paths and end cameras; baselines first so Ours sits on top
    for (pk, _, lab, col, lw), P in reversed(list(zip(POLICIES, paths))):
        q = pr(P)
        z = 7 if pk == "ours" else 4
        cv.line(q[:, 0], q[:, 1], color=col, lw=lw, z=z)
        draw_frustum(cv, pr, data[pk]["ep"]["step_poses"][-1], END_FS, ec=col, fc="white",
                     lw=0.8 if pk == "ours" else 0.55, z=z + 0.2)
    A, _ = draw_frustum(cv, pr, start, 0.40, ec=INK, fc="#F1F1F3", lw=0.6, z=8)
    cv.text(A[0] - 0.06, A[1] + 0.13, "start", size=5.2, z=10)
    Ag, _ = draw_frustum(cv, pr, goal, 0.45, ec=BLUE, fc="none", lw=0.9, z=9)
    cv.ax.scatter(*Ag, s=30, marker="*", color=BLUE, zorder=10, lw=0)
    cv.text(Ag[0] + GOAL_LAB[0], Ag[1] + GOAL_LAB[1], "GT goal", size=5.2, color=BLUE_D, z=10)
    cv.text(0.12, y0 + rh - 0.10, title, size=6.0, weight="semibold", ha="left", z=10)
    cv.text(0.12, y0 + rh - 0.21, f"start d = {ref['d_start']:.2f}", size=5.0, color=MUTED, ha="left", z=10)

    # ---- last frames: GT goal, then each policy
    fx0, fw, gap = FX0, FW, GAP
    fh = fw * 0.75
    fy = y0 + (rh - fh) / 2
    cols = [("goal", "GT goal", BLUE, None)] + [(pk, lab, col, data[pk]) for pk, _, lab, col, _ in POLICIES]
    for j, (pk, lab, col, dd) in enumerate(cols):
        x = fx0 + j * (fw + gap)
        img = load(ref["goal_frame_image"] if pk == "goal" else dd["final"], 600)
        cv.image(img, x, fy, fw, fh, ec=col, lw=1.6 if pk in ("goal", "ours") else 1.0, shadow=True)
        if first:
            cv.text(x + fw / 2, fy + fh + 0.09, lab, size=6.0, color=col if pk != "dp" else MUTED,
                    weight="semibold" if pk in ("goal", "ours") else "normal")
        if dd is not None:
            ok = dd["succ"]
            cv.text(x + fw - 0.03, fy + 0.03, f"d {dd['d']:.2f}", size=5.0, ha="right", va="bottom",
                    color="white", family="IBM Plex Mono", z=7,
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.3",
                              fc=(0.14, 0.52, 0.35, 0.9) if ok else (0, 0, 0, 0.5), ec="none"))
    return pr


def main():
    a = args()
    root = Path(a.root)
    bench = root / "runs/bench_v2"
    scores = json.load(open(bench / "scores.json"))
    rollouts = {pk: json.load(open(bench / f"rollout_{stem}.json"))["episodes"] for pk, stem, *_ in POLICIES}
    views = [tuple(map(float, v.split(","))) for v in a.views.split(";")] if a.views else [e[1] for e in EPISODES]
    cv = Canvas()
    cv.fig.set_size_inches(W, H)
    cv.ax.set_xlim(0, W)
    cv.ax.set_ylim(0, H)
    rh = RH
    for r, ((key, _, title, inset), view) in enumerate(zip(EPISODES, views)):
        y0 = H - 0.06 - (r + 1) * rh - r * 0.08
        row(cv, y0, rh, episode(bench, scores, rollouts, key), view, title, first=(r == 0), inset=inset)
    # legend for the 3-D paths
    lx = 0.10
    for pk, _, lab, col, lw in POLICIES:
        cv.line([lx, lx + 0.16], [0.07, 0.07], color=col, lw=lw)
        cv.text(lx + 0.20, 0.07, lab, size=5.4, ha="left")
        lx += 0.30 + 0.055 * len(lab)
    cv.ax.scatter([lx + 0.05], [0.07], s=22, marker="*", color=BLUE, lw=0)
    cv.text(lx + 0.12, 0.07, "GT goal camera", size=5.4, ha="left")
    cv.text(W - 0.06, 0.07, "d: distance to the GT goal (success d ≤ 0.10 + subject visible, green)", size=5.2,
            color=MUTED, ha="right")
    for ext in (("png",) if a.views else ("pdf", "svg", "png")):
        cv.fig.savefig(HERE / f"{a.out}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", HERE / f"{a.out}.png")


if __name__ == "__main__":
    main()
