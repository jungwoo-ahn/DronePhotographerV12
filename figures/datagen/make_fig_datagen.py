"""Data-generation pipeline figure: assets -> VLM placement -> camera trajectories -> render & hindsight label ->
windows and action chunks, with the count that survives each stage.

Numbers are counted from the data (2026-10-14):
  placements: 9,889 scene x subject runs (97 x 102), 8,356 with >= 1 accepted spot, 7,885 kept (91 scenes)
  trajectories: 393,085 sampled pairs, 318,121 rejected, 74,964 accepted (12 per placement max)
  windows: lerobot_v5 = 149,999 episodes (135,096 train / 14,903 val), 9 keyframes each
Worked examples come from placement Chill-Camping__Knit-and-Denim: trajectory 0 for the goal filter (it zooms in
from 1 % occupancy), trajectory 4 for the windows (as in Fig. 2).

    .venv-analysis/bin/python figures/datagen/make_fig_datagen.py [--root <main checkout>]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overview"))
from make_fig2 import (BLUE, BLUE_D, BLUE_L, FAINT, GREEN, GREEN_D, GREEN_L, GRID, INK, MUTED, ORANGE,  # noqa: E402
                       PANEL, Canvas, action_chunk, draw_action_tokens, load)

W, H = 7.16, 4.90
KEY = "Chill-Camping_150d7263__Knit-and-Denim_82dd220b"
PAIR_D, PAIR_E, G_IDX = 0, 4, 4
WINDOWS = [(26, 18), (18, 10), (10, 4)]
WIN = ["#A6D9BE", "#5BB58B", GREEN]
CACHE = HERE / "_cache"
REJECT = [  # stage-1 rejections over all 7,885 placements
    ("subject off frame", 87359 + 64170),
    ("camera blocked", 87510 + 47940),
    ("arc midpoint blocked", 16892),
    ("endpoints < 15° apart", 9570),
    ("degenerate look-at", 4680),
]
FUNNEL = [("9,889", "scene × subject\nruns"), ("7,885", "placements\n(91 scenes)"), ("393k", "sampled\ncamera pairs"),
          ("75k", "trajectories\n(19 % kept)"), ("2.4M", "rendered\nframes"), ("150k", "training\nwindows")]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nas_main/jungwooahn/projects/DronePhotographerV12")
    ap.add_argument("--b", default="schematic", choices=["schematic", "real"],
                    help="panel (b): schematic grid, or the real re-run grid projected into the scene camera")
    return ap.parse_args()


def traj_dir(root):
    return root / "data/trajectories/v7_stage2_renders_lookat075"


# ---------------------------------------------------------------- cached thumbnails
def readable(im):
    """A frame that shows the scene: not mostly sky or shadow."""
    g = np.asarray(im.convert("L"), dtype=float)
    return 70 <= g.mean() <= 185 and g.std() >= 42


def scene_thumbs(root, n):
    """One frame per scene that shows the place (subject small, image readable), cached as small JPEGs."""
    out = CACHE / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("*.jpg"))
    if len(have) >= n:
        return have[:n]
    T = traj_dir(root)
    seen = {}
    for d in sorted(os.listdir(T)):
        seen.setdefault(d.split("__")[0], d)
    paths = []
    for sc in np.random.default_rng(1).permutation(sorted(seen)):
        try:
            dj = json.load(open(T / seen[sc] / "data.json"))
        except Exception:
            continue
        cands = sorted(((r["scores"]["occupancy"], r["path_rel"]) for recs in dj.get("render_records", [])
                        for r in recs if r.get("in_frame") and 3 <= r["scores"]["occupancy"] <= 12))
        for _, rel in cands[::5][:8]:
            im = load(T / seen[sc] / rel, 360)
            if readable(im):
                f = out / f"{sc}.jpg"
                im.save(f, quality=88)
                paths.append(f)
                break
        if len(paths) >= n:
            break
    return paths


def subject_cutouts(root, n):
    """Front views from the facing turntables, background keyed out to transparency."""
    out = CACHE / "subjects"
    out.mkdir(parents=True, exist_ok=True)
    have = sorted(out.glob("*.png"))
    if len(have) >= n:
        return have[:n]
    fmap = json.load(open(root / "runs/facing_map_final.json"))
    tt = root / "runs/facing_turntable"
    objs = sorted(o for o in os.listdir(tt) if (tt / o).is_dir())
    paths = []
    for o in np.random.default_rng(3).permutation(objs):
        az = int(round(fmap.get(o, {}).get("front_az", 90) / 30) * 30) % 360
        im = np.asarray(Image.open(tt / o / f"az_{az:03d}.png").convert("RGB")).astype(float)
        a = (np.abs(im - im[2, 2]).sum(-1) > 22).astype(np.uint8) * 255
        ys, xs = np.nonzero(a)
        if len(xs) < 50:
            continue
        rgba = Image.fromarray(np.dstack([im.astype(np.uint8), a])).crop(
            (xs.min() - 4, ys.min() - 4, xs.max() + 4, ys.max() + 4))
        f = out / f"{o}.png"
        rgba.save(f)
        paths.append(f)
        if len(paths) >= n:
            break
    return paths


def cutout(cv, path, x, y, h, z=5):
    im = Image.open(path)
    w = h * im.width / im.height
    cv.ax.imshow(np.asarray(im), extent=(x - w / 2, x + w / 2, y, y + h), zorder=z, interpolation="lanczos")


def badge(cv, x, y, n, col=INK):
    cv.ax.add_patch(Circle((x, y), 0.052, fc=col, ec="none", zorder=7))
    cv.text(x, y - 0.002, str(n), size=5.2, color="white", weight="bold", z=8)


# ---------------------------------------------------------------- panels
def panel_assets(cv, root):
    X0, Y0, PW, PH = 0.04, 2.42, 2.36, 2.44
    cv.panel(X0, Y0, PW, PH, "a", "Assets")
    cv.text(X0 + PW - 0.12, Y0 + PH - 0.16, "BlenderKit", size=5.0, color=MUTED, ha="right")
    cols, tw = 6, 0.352
    th = tw * 0.75
    top = Y0 + PH - 0.34
    for i, f in enumerate(scene_thumbs(root, 18)):
        r, c = divmod(i, cols)
        cv.image(load(f), X0 + 0.10 + c * (tw + 0.016), top - (r + 1) * th - r * 0.016, tw, th, z=4)
    cv.text(X0 + 0.10, top - 3 * th - 2 * 0.016 - 0.06, "91 indoor & outdoor scenes", size=5.6, ha="left",
            va="top", weight="semibold")
    subs = subject_cutouts(root, 22)
    for i, f in enumerate(subs[:11]):
        cutout(cv, f, X0 + 0.20 + i * 0.196, Y0 + 0.62, 0.40, z=4)
    for i, f in enumerate(subs[11:22]):
        cutout(cv, f, X0 + 0.30 + i * 0.196, Y0 + 0.26, 0.40, z=5)
    cv.text(X0 + 0.10, Y0 + 0.12, "102 human subjects · front calibrated on a 12-view turntable", size=5.3,
            ha="left", weight="semibold")


def panel_placement(cv, root):
    X0, Y0, PW, PH = 2.46, 2.42, 2.30, 2.44
    cv.panel(X0, Y0, PW, PH, "b", "VLM-guided placement")
    T = traj_dir(root) / KEY
    dj = json.load(open(T / "data.json"))

    # 1: interest region on a readable wide view
    wide = None
    for recs in dj["render_records"]:
        for r in recs:
            if r["in_frame"] and 4 <= r["scores"]["occupancy"] <= 9 and readable(load(T / r["path_rel"], 200)):
                wide = r
                break
        if wide:
            break
    x, w = X0 + 0.10, 0.96
    h = w * 0.75
    y = Y0 + PH - 0.36 - h
    cv.image(load(T / wide["path_rel"], 500), x, y, w, h, shadow=True)
    bx = wide["bbox_xyxy_full"]
    cx, cy = x + (bx[0] + bx[2]) / 2 / 1024 * w, y + h - (bx[1] + bx[3]) / 2 / 768 * h
    bw, bh = 0.40 * w, 0.42 * h
    rx = np.clip(cx - bw / 2, x + 0.01, x + w - bw - 0.01)
    ry = np.clip(cy - bh / 2, y + 0.01, y + h - bh - 0.01)
    cv.ax.add_patch(Rectangle((rx, ry), bw, bh, fc="none", ec=ORANGE, lw=1.1, ls=(0, (2.5, 1.5)), zorder=6))
    badge(cv, x + 0.07, y + h - 0.07, 1, ORANGE)
    cv.text(x, y - 0.05, "VLM marks where a\nperson belongs", size=5.1, ha="left", va="top", weight="semibold",
            linespacing=1.15)
    cv.text(x, y - 0.27, "Gemini 2.5 Flash, per scene", size=4.7, ha="left", va="top", color=MUTED)

    # 2: floor candidates (schematic top-down grid scored by ray casts)
    n, cs = 7, 0.112
    gx0 = X0 + PW - 0.10 - n * cs
    gy0 = y + h - n * cs
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:n, 0:n]
    score = np.exp(-((xx - 4.3) ** 2 + (yy - 3.6) ** 2) / 6.0) + 0.25 * rng.random((n, n))
    blocked = rng.random((n, n)) < 0.18
    for i in range(n):
        for j in range(n):
            s_ = min(score[i, j], 1.0)
            fc = "#E6E8EC" if blocked[i, j] else (0.95 - 0.60 * s_, 0.97 - 0.30 * s_, 0.97 - 0.55 * s_)
            cv.ax.add_patch(Rectangle((gx0 + j * cs, gy0 + i * cs), cs * 0.9, cs * 0.9, fc=fc, ec="none", zorder=4))
            if blocked[i, j]:
                cv.line([gx0 + j * cs + 0.02, gx0 + j * cs + cs * 0.9 - 0.02],
                        [gy0 + i * cs + 0.02, gy0 + i * cs + cs * 0.9 - 0.02], color=FAINT, lw=0.4, z=4.1)
    sc = np.where(blocked, -1, score)
    chosen = []
    for i, j in zip(*np.unravel_index(np.argsort(-sc, axis=None), sc.shape)):
        if all(abs(i - a) + abs(j - b) >= 2 for a, b in chosen):
            chosen.append((i, j))
        if len(chosen) == 3:
            break
    for i, j in chosen:
        cv.ax.add_patch(Circle((gx0 + j * cs + cs * 0.45, gy0 + i * cs + cs * 0.45), 0.032, fc=GREEN, ec="white",
                               lw=0.6, zorder=5))
    badge(cv, gx0 + 0.06, gy0 + n * cs - 0.06, 2)
    cv.text(gx0, gy0 - 0.05, "ray-cast floor cells", size=5.1, ha="left", va="top", weight="semibold")
    cv.text(gx0, gy0 - 0.17, "slope ≤ 12°, flat, visible;\nkeep top-3, spread out", size=4.7, ha="left", va="top",
            color=MUTED, linespacing=1.15)

    # 3: orbit views judged by a VLM
    vw = 0.46
    vh = vw * 0.75
    vy = Y0 + 0.30
    for k, (p, f) in enumerate([(4, 4), (9, 20), (1, 12)]):
        r = dj["render_records"][p][f]
        cv.image(load(T / r["path_rel"], 300), X0 + 0.10 + k * (vw + 0.04), vy, vw, vh, shadow=True)
    badge(cv, X0 + 0.17, vy + vh - 0.07, 3, GREEN)
    jx = X0 + 0.10 + 3 * (vw + 0.04)
    jw = X0 + PW - 0.10 - jx
    cv.box(jx, vy, jw, vh, fc="white", ec=GREEN, lw=0.7, r=0.04, shadow=True)
    cv.text(jx + jw / 2, vy + vh - 0.06, "VLM judge", size=5.0, weight="semibold", color=GREEN_D)
    cv.text(jx + jw / 2, vy + vh / 2 - 0.04, "grounded · visible\nscale · context\nkeep if ≥ 7/10", size=4.2,
            linespacing=1.1)
    cv.text(X0 + 0.10, vy - 0.06, "place, render up to 6 orbit views; else move & retry", size=4.9, ha="left",
            va="top", weight="semibold")


GRID_RUN = "runs/placement_grid/Chill-Camping__Knit-and-Denim"   # sbatch_placement_grid.sh output
VIEW_SCORES = [3, 2, 1, 8, 4]   # the placement run's VLM view scores for this spot (data/vlm_object_placing_v6_*)


def panel_placement_real(cv, root):
    """Panel (b) from a real re-run of the floor search, projected into the scene camera's own view."""
    X0, Y0, PW, PH = 2.46, 2.42, 2.30, 2.44
    cv.panel(X0, Y0, PW, PH, "b", "VLM-guided placement")
    G = root / GRID_RUN
    r = json.load(open(G / "cells.json"))
    im = Image.open(G / "scene_cam.png").convert("RGB")
    Wi, Hi = im.size
    Mi = np.linalg.inv(np.array(r["scene_cam_matrix"]))
    k = r["scene_cam_lens"] / r["scene_cam_sensor"]

    def proj(p):
        q = Mi @ np.r_[p, 1.0]
        if q[2] >= -0.1:
            return None
        u, v = Wi / 2 + k * q[0] / -q[2] * Wi, Hi / 2 - k * q[1] / -q[2] * Wi
        return (u, v) if 0 <= u < Wi and 0 <= v < Hi else None

    w = 1.90
    h = w * Hi / Wi
    x, y = X0 + 0.10, Y0 + PH - 0.34 - h
    cv.image(im, x, y, w, h, shadow=True)
    clip = Rectangle((x, y), w, h, transform=cv.ax.transData)
    sc = [c for c in r["cells"] if c["status"] == "scored"]
    s = np.array([c["score"] for c in sc])
    lo, hi = np.percentile(s, [5, 95])
    for c, v in zip(sc, s):
        q = proj((c["x"], c["y"], c["z"]))
        if q is None:
            continue
        t = float(np.clip((v - lo) / (hi - lo + 1e-9), 0, 1))
        cv.ax.scatter([x + q[0] / Wi * w], [y + h - q[1] / Hi * h], s=1.6 + 3.5 * t,
                      color=(0.55 - 0.45 * t, 0.85 - 0.25 * t, 0.55 - 0.35 * t), zorder=5, lw=0).set_clip_path(clip)
    for i, t in enumerate(r["top_k"]):
        q = proj(t["position"])
        if q is not None:
            cv.ax.add_patch(Circle((x + q[0] / Wi * w, y + h - q[1] / Hi * h), 0.035, fc="none", ec="white", lw=1.0,
                                   zorder=6))
    badge(cv, x + 0.07, y + h - 0.07, 2)
    n = {}
    for c in r["cells"]:
        n[c["status"]] = n.get(c["status"], 0) + 1
    cv.text(x + w - 0.04, y + h - 0.05, "ray-cast floor cells,\nseen from the scene camera", size=4.5,
            color="white", ha="right", va="top", linespacing=1.1)
    cv.text(x, y - 0.05, f"{n.get('scored', 0):,} flat & visible (green, darker = higher score); dropped: "
            f"{n.get('out_of_view', 0) / 1000:.1f}k outside the view,\n{n.get('not_visible', 0):,} hidden, "
            f"{n.get('not_flat', 0) + n.get('uneven', 0):,} not flat · rings: top-3 kept (re-run without the VLM region)",
            size=4.2, ha="left", va="top", color=MUTED, linespacing=1.2)

    # 3: VLM judge with the real per-view scores of the accepted spot
    T = traj_dir(root) / KEY
    dj = json.load(open(T / "data.json"))
    vw = 0.40
    vh = vw * 0.75
    vy = Y0 + 0.10
    for i, (p, f) in enumerate([(4, 4), (9, 20), (1, 12)]):
        cv.image(load(T / dj["render_records"][p][f]["path_rel"], 300), X0 + 0.10 + i * (vw + 0.035), vy, vw, vh,
                 shadow=True)
    badge(cv, X0 + 0.16, vy + vh - 0.06, 3, GREEN)
    jx = X0 + 0.10 + 3 * (vw + 0.035)
    jw = X0 + PW - 0.10 - jx
    cv.box(jx, vy, jw, vh, fc="white", ec=GREEN, lw=0.7, r=0.04, shadow=True)
    cv.text(jx + jw / 2, vy + vh - 0.06, "VLM judge", size=4.9, weight="semibold", color=GREEN_D)
    cw = (jw - 0.08) / len(VIEW_SCORES)
    for i, v in enumerate(VIEW_SCORES):
        ok = v >= 7
        cx = jx + 0.04 + (i + 0.5) * cw
        cv.ax.add_patch(Rectangle((cx - cw * 0.42, vy + 0.10), cw * 0.84, 0.09, fc=GREEN if ok else "#E6E8EC",
                                  ec="none", zorder=6))
        cv.text(cx, vy + 0.145, str(v), size=4.6, color="white" if ok else INK, z=7, weight="bold" if ok else "normal")
    cv.text(jx + jw / 2, vy + 0.05, "best 8 ≥ 7 → keep", size=4.2, color=GREEN_D)


def panel_trajectory(cv, root):
    X0, Y0, PW, PH = 4.82, 2.42, 2.30, 2.44
    cv.panel(X0, Y0, PW, PH, "c", "Camera trajectories")
    dj = json.load(open(traj_dir(root) / KEY / "data.json"))
    el = dj["accepted_pairs"][PAIR_E]["ellipse"]
    a, b, thn = float(el["a"]), float(el["b"]), np.radians(float(el["theta_near_deg"]))
    # the whole ellipse in its own plane: centre O, major axis along O -> C_far (u), minor axis v
    ox, oy = X0 + 1.02, Y0 + 1.50
    s = 0.86 / a
    t = np.linspace(0, 2 * np.pi, 240)

    def P(th):
        r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
        return ox + s * r * np.cos(th), oy + s * r * np.sin(th)

    ex, ey = P(t)
    cv.line(ex, ey, color="#B9BEC6", lw=0.6, ls=(0, (2.2, 1.8)), z=2)
    # semi-axes
    cv.line([ox, ox + s * a], [oy, oy], color=MUTED, lw=0.5, z=2.5)
    cv.line([ox, ox], [oy, oy + s * b], color=MUTED, lw=0.5, z=2.5)
    cv.text(ox + s * a * 0.45, oy - 0.05, r"$a=\|C_{far}-O\|$", size=5.2, color=MUTED, va="top")
    cv.text(ox - 0.03, oy + s * b * 0.5, r"$b$", size=5.6, color=MUTED, ha="right")
    # sampled arc C_far (theta = 0) -> C_near, 32 frames at equal delta-theta
    tt = np.linspace(0, thn, 32)
    px, py = P(tt)
    for k in range(0, 32, 4):
        cv.line([ox, px[k]], [oy, py[k]], color=GRID, lw=0.4, z=1.5)
    cv.line(px, py, color=INK, lw=1.1, z=4)
    cv.ax.scatter(px, py, s=3.5, color=INK, zorder=4.5, lw=0)
    for k in (0, 31):
        d = np.array([ox - px[k], oy - py[k]])
        d /= np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        tip = np.array([px[k], py[k]])
        cv.ax.add_patch(Polygon([tip, tip - d * 0.085 + nrm * 0.042, tip - d * 0.085 - nrm * 0.042], closed=True,
                                fc="white", ec=INK, lw=0.7, zorder=5))
    cv.text(px[0] + 0.03, py[0] - 0.07, r"$C_{far}$ = vertex" "\n(frame 0)", size=5.0, ha="center", va="top",
            linespacing=1.1)
    cv.text(px[31] + 0.10, py[31] + 0.05, r"$C_{near}$ (frame 31)", size=5.0, ha="left", va="bottom")
    cv.text(ox + s * a * 0.70, oy + s * b * 0.40, r"$\Delta\theta$ equal", size=4.6, color=MUTED)
    cv.ax.add_patch(Circle((ox, oy), 0.030, fc=BLUE, ec="none", zorder=6))
    cv.text(ox - 0.04, oy - 0.05, "O = centre\n(look-at, 0.75 × height)", size=4.6, color=BLUE_D, ha="right",
            va="top", linespacing=1.1)
    cv.text(X0 + 0.10, Y0 + PH - 0.34, "ellipse centred at O with C_far as its vertex; b is solved so it passes\n"
            "through C_near. Endpoints: azimuth ~ U[0°, 360°), distance ~ logU[0.8, 8] m.", size=4.6, ha="left",
            va="top", linespacing=1.2)
    # rejection bars
    by = Y0 + 0.86
    cv.text(X0 + 0.10, by, "rejected (of 393k pairs)", size=5.0, ha="left", va="center", weight="semibold")
    cv.text(X0 + PW - 0.10, by, "kept 75k · 19 %", size=5.0, ha="right", va="center", color=GREEN_D,
            weight="semibold")
    mx = max(c for _, c in REJECT)
    for k, (lab, c) in enumerate(REJECT):
        yb = by - 0.14 - k * 0.115
        wbar = 0.82 * c / mx
        cv.ax.add_patch(Rectangle((X0 + 0.10, yb - 0.035), wbar, 0.07, fc="#D5D9E0", ec="none", zorder=4))
        cv.text(X0 + 0.13 + wbar, yb, f"{c / 1000:.0f}k  {lab}", size=4.5, ha="left")


def panel_render(cv, root):
    X0, Y0, PW, PH = 0.04, 0.58, 3.66, 1.78
    cv.panel(X0, Y0, PW, PH, "d", "Render & hindsight label")
    T = traj_dir(root) / KEY
    recs = json.load(open(T / "data.json"))["render_records"][PAIR_D]
    frames = list(range(0, 32, 4)) + [31]
    gap = 0.03
    fw = (PW - 0.20 - 8 * gap) / 9
    fh = fw * 0.75
    fy = Y0 + PH - 0.62 - fh
    for k, f in enumerate(frames):
        r = recs[f]
        x = X0 + 0.10 + k * (fw + gap)
        bx = r["bbox_xyxy_full"]
        x0b, x1b = max(bx[0], 0), min(bx[2], 1023)
        y0b, y1b = max(bx[1], 0), min(bx[3], 767)
        occ = r["scores"]["occupancy"]
        vis = (y1b - y0b) / max(bx[3] - bx[1], 1)
        ok = 10 <= occ <= 80 and vis >= 0.35 and r["in_frame"]
        cv.image(load(T / r["path_rel"], 300), x, fy, fw, fh, z=4, ec=GREEN if ok else None, lw=0.9)
        cv.ax.add_patch(Rectangle((x + x0b / 1024 * fw, fy + fh - y1b / 768 * fh), (x1b - x0b) / 1024 * fw,
                                  (y1b - y0b) / 768 * fh, fc="none", ec=BLUE, lw=0.6, ls=(0, (2, 1)), zorder=5))
        cv.text(x + fw / 2, fy + fh + 0.05, f"t = {f}", size=4.6, color=MUTED, va="bottom")
        cv.text(x + fw / 2, fy - 0.05, f"occ {occ}%", size=4.6, va="top", family="IBM Plex Mono")
        cv.text(x + fw / 2, fy - 0.17, "goal-eligible" if ok else "too small", size=4.4, va="top",
                color=GREEN_D if ok else MUTED, weight="semibold" if ok else "normal")
    cv.text(X0 + 0.10, Y0 + 0.50, "Each frame is rendered (Cycles, 1024×768) and the subject mesh is projected into it,\n"
            "so its box, bearing, elevation and occupancy are exact — no detector, no annotator. A frame can serve\n"
            "as a goal if occupancy is 10–80 % and ≥ 35 % of the body is visible.", size=4.8, ha="left", va="top",
            linespacing=1.3)


def panel_windows(cv, root):
    X0, Y0, PW, PH = 3.76, 0.58, 3.36, 1.78
    cv.panel(X0, Y0, PW, PH, "e", "Windows → action chunks")
    traj = json.load(open(traj_dir(root) / KEY / "data.json"))["accepted_pairs"][PAIR_E]["trajectory_32f"]
    tx0, tx1, ty = X0 + 0.16, X0 + PW - 0.14, Y0 + PH - 0.66
    xs = np.linspace(tx0, tx1, 32)
    cv.line([tx0, tx1], [ty, ty], color=FAINT, lw=0.6, z=2)
    for k, x in enumerate(xs):
        cv.line([x, x], [ty - 0.022, ty + 0.022], color=FAINT, lw=0.5, z=2)
        if k % 4 == 0 or k == 31:
            cv.text(x, ty - 0.05, str(k), size=4.4, color=MUTED, va="top")
    cv.text(tx1, ty - 0.15, "frame", size=4.4, color=MUTED, ha="right", va="top")
    for w, (s, e) in enumerate(WINDOWS):
        x0, x1 = xs[e], xs[s]
        hgt = 0.10 + 0.05 * w
        arc = np.linspace(0, np.pi, 40)
        cv.line((x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(arc), ty + hgt * np.sin(arc), color=WIN[w], lw=1.0, z=3)
        cv.ax.scatter([x1], [ty], s=9, color=WIN[w], zorder=5, lw=0)
        cv.text((x0 + x1) / 2, ty + hgt + 0.03, f"$w_{w + 1}$", size=5.6, color=WIN[w] if w else "#3E9A70",
                va="bottom")
    cv.ax.scatter([xs[G_IDX]], [ty], s=36, marker="*", color=BLUE, zorder=6, lw=0)
    cv.text(xs[G_IDX] - 0.05, ty + 0.05, "goal", size=4.8, color=BLUE_D, ha="right", va="bottom",
            weight="semibold")
    cv.text(X0 + 0.16, Y0 + PH - 0.32, "pick a goal frame g, then starts s with |g − s| ≤ 32; each walks 8 frames toward g",
            size=4.7, ha="left")

    # w3 expanded: 9 keyframes -> 8 action tokens
    A, fr = action_chunk(traj, *WINDOWS[-1])
    ky = Y0 + 0.62
    kx = np.linspace(X0 + 0.22, X0 + 1.52, 9)
    for k, (x, f) in enumerate(zip(kx, fr)):
        at = f == G_IDX
        cv.ax.add_patch(Circle((x, ky), 0.048, fc=BLUE_L if at else "white", ec=BLUE if at else INK, lw=0.6, zorder=5))
        cv.text(x, ky, str(f), size=4.5, z=6)
        if k:
            cv.arrow((kx[k - 1] + 0.05, ky), (x - 0.05, ky), color=GREEN, lw=0.5, ms=2.4)
    cv.text(X0 + 0.16, ky + 0.13, "$w_3$: 9 keyframes (clamped at g)", size=4.9, ha="left", color=GREEN_D,
            weight="semibold")
    cv.text(X0 + 0.16, ky - 0.11, r"$a_k$: pose change $k\to k{+}1$ in the camera frame," "\n"
            r"$(\Delta t, R_{6D})$ + shoot, latched once g is reached", size=4.6, ha="left", va="top",
            linespacing=1.25)
    tx = X0 + 1.92
    draw_action_tokens(cv, A, tx, ky + 0.19, c=0.112, h=0.032)
    cv.text(tx + 0.45, ky - 0.24, "8 × 10-D action chunk", size=4.8, va="top", color=MUTED)
    cv.text(X0 + 0.16, Y0 + 0.10, "≤ 20 windows per trajectory (25 % near goal) · 8 view sectors balanced · 8 val scenes held out",
            size=4.4, ha="left", color=MUTED)


def funnel(cv):
    x0, x1, y, h = 0.10, W - 0.10, 0.06, 0.40
    n = len(FUNNEL)
    w = (x1 - x0) / n
    for k, (num, lab) in enumerate(FUNNEL):
        x = x0 + k * w
        last = k == n - 1
        pts = [(x, y), (x + w - 0.06, y), (x + w + 0.02, y + h / 2), (x + w - 0.06, y + h), (x, y + h)]
        if k:
            pts.append((x + 0.08, y + h / 2))
        cv.ax.add_patch(Polygon(pts, closed=True, fc=GREEN_L if last else "#ECEEF1", ec="none", zorder=1))
        cv.text(x + 0.16, y + h / 2, num, size=7.5, weight="bold", ha="left", color=GREEN_D if last else INK)
        cv.text(x + 0.18 + 0.105 * len(num), y + h / 2, lab, size=4.8, ha="left", color=MUTED, linespacing=1.1)


def main():
    a = args()
    root = Path(a.root)
    cv = Canvas()
    cv.fig.set_size_inches(W, H)
    cv.ax.set_xlim(0, W)
    cv.ax.set_ylim(0, H)
    panel_assets(cv, root)
    (panel_placement_real if a.b == "real" else panel_placement)(cv, root)
    panel_trajectory(cv, root)
    panel_render(cv, root)
    panel_windows(cv, root)
    funnel(cv)
    stem = "fig_datagen" + ("_realgrid" if a.b == "real" else "")
    for ext in ("pdf", "svg", "png"):
        cv.fig.savefig(HERE / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", HERE / f"{stem}.png")


if __name__ == "__main__":
    main()
