"""Fig. 2 — method overview: (a) self-labelling data engine, (b) world camera action model, (c) deployment.

Every number and frame in the figure comes from the repo's data:
  (a) camera flights of placement Chill-Camping__Knit-and-Denim (data.json, 12 pairs x 32 frames);
      the walk s=26 -> g=4 of pair 04 is cut into the three training windows 26->18, 18->10, 10->4.
  (b) Cosmos3-Nano MoT layout from configs/policy/action_policy_camera_nano.py and the framework configs;
      the action tokens are the real 10-D labels of window 10->4 (encode_action_9d + shoot_column).
  (c) the closed-loop rollout of Ours v6@12k on episode Chill-Camping__All-People-Are-Sisters p00:s31:g02
      (runs/gif/rollout_ours_v6_12k.json), which declared shoot at step 7 of chunk 4.

    .venv-analysis/bin/python figures/overview/make_fig2.py [--root <main checkout>] [--view=az,el,r,dist]
writes fig2_overview.{pdf,svg,png} next to this file.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle, Ellipse
from matplotlib.transforms import Affine2D
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.common.action_repr import encode_action_9d  # noqa: E402

for f in (HERE / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "IBM Plex Sans",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.45,
})

# ---------------------------------------------------------------- palette
INK = "#1F2126"
MUTED = "#61656D"
FAINT = "#C3C6CC"
GRID = "#E2E5EA"
PANEL = "#F6F7F9"
ORANGE, ORANGE_L = "#D06A1E", "#FCEBDC"                    # intent / judgement
BLUE, BLUE_L, BLUE_D = "#2E5FBF", "#E4ECFA", "#1C3D80"     # shot profile p (text)
GREEN, GREEN_L, GREEN_D = "#22855A", "#E0F2E8", "#165A3C"  # action / trained
VIS, VIS_L = "#6A6EA0", "#E9E9F5"                          # vision latents
FROZEN_L = "#ECEDF0"
WIN = ["#A6D9BE", "#5BB58B", GREEN]                        # the three windows of one walk, light -> dark
SHADOW = (0, 0, 0, 0.07)

W, H = 7.16, 4.95   # inches: full two-column width

KEY = "Chill-Camping_150d7263__Knit-and-Denim_82dd220b"     # panels (a) and (b)
PAIR, S_IDX, G_IDX = 4, 26, 4
WINDOWS = [(26, 18), (18, 10), (10, 4)]                     # one goal, three windows
WIN_LAB = [(0.11, 0.07), (0.14, 0.03), (0.17, 0.05)]        # label offsets of w1..w3 in panel (a)
KEY_C = "Chill-Camping_150d7263__All-People-Are-Sisters_1795d425_p00_s31_g02"   # panel (c) rollout
OAZ, OEL, ODIST = -10, 48, 30                               # orbit overview of panel (a)
PINS = [(2, 6, None), (3, 15, None), (1, 8, None)]  # (flight, frame, offset or None = outward)
OWIN_LAB = [(0.11, 0.06), (0.13, 0.03), (0.15, 0.04)]
OSTART_DY, OEND_DY = -0.02, -0.34                            # start/end thumbnails, relative to their cameras
AZ, EL, RV, DIST = 35, 28, 4.5, 18                      # 3-D view of panel (a)


def args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nas_main/jungwooahn/projects/DronePhotographerV12")
    ap.add_argument("--view", default=None, help="az,el,radius,dist to try another 3-D view of panel (a)")
    ap.add_argument("--out", default="fig2_overview")
    ap.add_argument("--a-style", default="orbits", choices=["orbits", "behind"],
                    help="panel (a): all orbits from above, or the view from behind the cameras")
    ap.add_argument("--orbit-view", default=None, help="az,el,dist for the orbit overview")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- primitives
class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def box(self, x, y, w, h, fc="white", ec=INK, lw=0.6, r=0.05, ls="-", z=2, hatch=None, shadow=False):
        if shadow:
            self.ax.add_patch(FancyBboxPatch((x + 0.018, y - 0.022), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                             fc=SHADOW, ec="none", zorder=z - 0.05))
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, ls=ls, zorder=z, hatch=hatch)
        self.ax.add_patch(p)
        return p

    def text(self, x, y, s, size=6.5, color=INK, ha="center", va="center", weight="normal", z=6,
             style="normal", family=None, **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z,
                            style=style, family=family or plt.rcParams["font.family"], **kw)

    def arrow(self, p0, p1, color=INK, lw=0.7, ms=5, ls="-", z=3, cs="arc3"):
        a = FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={ms * 0.09},head_width={ms * 0.05}",
                            mutation_scale=10, color=color, lw=lw, ls=ls, zorder=z, connectionstyle=cs,
                            shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="round")
        self.ax.add_patch(a)

    def line(self, xs, ys, color=INK, lw=0.6, ls="-", z=3, alpha=1):
        return self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                            solid_capstyle="round", dash_capstyle="butt")[0]

    def image(self, img, x, y, w, h=None, ec=None, lw=0.8, z=4, shadow=False):
        if h is None:
            h = w * img.height / img.width
        if shadow:
            self.ax.add_patch(Rectangle((x + 0.018, y - 0.022), w, h, fc=SHADOW, ec="none", zorder=z - 0.05))
        self.ax.imshow(np.asarray(img), extent=(x, x + w, y, y + h), zorder=z, interpolation="lanczos")
        if ec:
            self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.1))
        return h

    def warp(self, img, quad, z=8):
        """Paste an image onto a projected quad (BL, BR, TR, TL), clipped to it."""
        bl, br, tr, tl = quad
        tf = Affine2D.from_values(br[0] - bl[0], br[1] - bl[1], tl[0] - bl[0], tl[1] - bl[1], bl[0], bl[1])
        im = self.ax.imshow(np.asarray(img), extent=(0, 1, 0, 1), zorder=z, interpolation="lanczos",
                            transform=tf + self.ax.transData)
        im.set_clip_path(Polygon(quad, closed=True, transform=self.ax.transData))

    def tokens(self, x, y, n, fc, ec, s=0.062, gap=0.016, hatch=None, z=5):
        for i in range(n):
            self.ax.add_patch(FancyBboxPatch((x + i * (s + gap), y), s, s, boxstyle="round,pad=0,rounding_size=0.012",
                                             fc=fc, ec=ec, lw=0.5, zorder=z, hatch=hatch))

    def trapezoid(self, x, y, w, h, wide_left=True, fc=VIS_L, ec=VIS, label="", z=4):
        d = h * 0.22
        if wide_left:
            pts = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)]
        else:
            pts = [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.6, zorder=z, joinstyle="round"))
        if label:
            self.text(x + w / 2, y + h / 2, label, size=7, color=VIS, z=z + 1)

    def shutter(self, x, y, r=0.035, on=True, z=9):
        """The shoot symbol: a filled aperture when shoot = 1, an empty ring when 0."""
        self.ax.add_patch(Circle((x, y), r, fc=INK if on else "white", ec=INK, lw=0.5, zorder=z))
        if on:
            self.ax.add_patch(Circle((x, y), r * 0.45, fc="white", ec="none", zorder=z + 0.1))

    def panel(self, x, y, w, h, letter, title):
        self.box(x, y, w, h, fc=PANEL, ec="none", r=0.09, z=0.5)
        bx, by = x + 0.16, y + h - 0.16
        self.ax.add_patch(Circle((bx, by), 0.085, fc=INK, ec="none", zorder=6))
        self.text(bx, by - 0.003, letter, size=7, color="white", weight="bold")
        self.text(bx + 0.14, by, title, size=7.8, weight="semibold", ha="left")

    def person(self, foot, top, color=INK, z=7):
        """A standing-person glyph between two screen points (foot, top of head)."""
        foot, top = np.asarray(foot), np.asarray(top)
        h = top[1] - foot[1]
        cx = (foot[0] + top[0]) / 2
        w = h * 0.30
        hr = h * 0.10
        self.ax.add_patch(Circle((top[0], top[1] - hr), hr, fc=color, ec="none", zorder=z))
        torso_top, hip = top[1] - 2.25 * hr, foot[1] + 0.47 * h
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, hip), w, torso_top - hip,
                                         boxstyle=f"round,pad=0,rounding_size={w * 0.35}",
                                         fc=color, ec="none", zorder=z))
        for dx in (-0.22, 0.22):
            self.line([cx + dx * w, cx + dx * w * 1.15], [hip + 0.02 * h, foot[1] + 0.02 * h], color=color,
                      lw=h * 72 * 0.11, z=z)


def load(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im.thumbnail((size, size), Image.LANCZOS)
    return im


def action_chunk(traj, s, g):
    """The real 10-D label of window s -> g: 9-D camera-local pose deltas + latched shoot."""
    step = -1 if g < s else 1
    fr = [max(s + step * k, g) if step < 0 else min(s + step * k, g) for k in range(9)]
    out = np.zeros((8, 10), np.float32)
    for i in range(8):
        a, b = traj[fr[i]], traj[fr[i + 1]]
        out[i, :9] = encode_action_9d(*[np.asarray(v, np.float32) for v in
                                        (a["pos"], a["forward"], a["up"], b["pos"], b["forward"], b["up"])])
    k = fr.index(g)                    # arrival: shoot is 1 from the goal keyframe on (shoot_column)
    out[:, 9] = (np.arange(8) >= k).astype(np.float32)
    return out, fr


# ---------------------------------------------------------------- (a) data engine, 3-D
class Proj:
    """Perspective view of the placement, fitted into a panel rectangle."""

    def __init__(self, center, az_deg, el_deg, dist, fov_deg=30):
        az, el = np.radians(az_deg), np.radians(el_deg)
        self.c = np.asarray(center, float)
        self.eye = self.c + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        f = self.c - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.k = 1 / np.tan(np.radians(fov_deg) / 2)
        self.s, self.o = 1.0, np.zeros(2)

    def raw(self, P):
        q = (np.atleast_2d(P) - self.eye) @ self.R.T
        return self.k * q[:, :2] / q[:, 2:3]

    def fit(self, pts, rect, margin=0.0):
        q = self.raw(pts)
        lo, hi = q.min(0), q.max(0)
        x, y, w, h = rect
        self.s = min((w - 2 * margin) / (hi - lo)[0], (h - 2 * margin) / (hi - lo)[1])
        self.o = np.array([x + w / 2, y + h / 2]) - self.s * (lo + hi) / 2

    def __call__(self, P):
        return self.raw(P) * self.s + self.o


def frustum(pose, size):
    p = np.asarray(pose["pos"] if "pos" in pose else pose["position"])   # trajectory data / rollout logs
    f = np.asarray(pose["forward"]); f /= np.linalg.norm(f)
    u = np.asarray(pose["up"]); u /= np.linalg.norm(u)
    r = np.cross(f, u)
    hw, hh, d = 0.5 * size * 1.333, 0.5 * size, size * 1.1   # 4:3 image
    corners = [p + d * f + sx * hw * r + sy * hh * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return p, corners


def draw_frustum(cv, pr, pose, size, ec, fc, lw=0.5, z=5, img=None):
    apex, cs = frustum(pose, size)
    A = pr(apex)[0]
    C = pr(np.array(cs))
    for c in C:
        cv.line([A[0], c[0]], [A[1], c[1]], color=ec, lw=lw * 0.8, z=z)
    cv.ax.add_patch(Polygon(C, closed=True, fc=fc, ec=ec, lw=lw, zorder=z + 0.05, joinstyle="round"))
    if img is not None:
        cv.warp(img, C, z=z + 0.1)
        cv.ax.add_patch(Polygon(C, closed=True, fc="none", ec=ec, lw=lw, zorder=z + 0.2, joinstyle="round"))
    return A, C


def panel_a_orbits(cv, root):
    """Overview: all 12 orbits around the subject, one walk highlighted, real frames pinned to other flights."""
    X0, Y0, PW, PH = 0.04, 1.93, 3.46, 2.99
    cv.panel(X0, Y0, PW, PH, "a", "Self-labelling data engine")
    D = root / "data/trajectories/v7_stage2_renders_lookat075" / KEY
    d = json.load(open(D / "data.json"))
    foot = np.array(d["subject_foot"]); hgt = d["subject_height"]
    pairs = d["accepted_pairs"]
    T = [np.array([f["pos"] for f in p["trajectory_32f"]]) for p in pairs]
    traj = pairs[PAIR]["trajectory_32f"]
    im_g = load(D / f"renders/pair_{PAIR:02d}_frame_{G_IDX:02d}.jpg", 700)

    rect = (X0 + 0.04, Y0 + 0.92, PW - 0.08, PH - 1.20)
    pr = Proj(foot + [0, 0, 0.6], az_deg=OAZ, el_deg=OEL, dist=ODIST, fov_deg=30)
    pr.fit(np.vstack(T + [foot + [0, 0, hgt]]), rect, margin=0.16)
    R = np.hypot(*(np.vstack(T)[:, :2] - foot[:2]).T).max() + 0.4
    t96 = np.linspace(0, 2 * np.pi, 96)

    # ground: soft disc, 2 m grid, range rings
    for rr, a in ((R, 0.55), (R * 0.66, 0.6), (R * 0.33, 0.7)):
        g = pr(np.c_[foot[0] + rr * np.cos(t96), foot[1] + rr * np.sin(t96), np.zeros_like(t96)])
        cv.ax.add_patch(Polygon(g, closed=True, fc="white", ec="none", alpha=a, zorder=0.8))
    for v in np.arange(-np.floor(R / 2) * 2, R, 2.0):
        for a_, b_ in (((v, -R), (v, R)), ((-R, v), (R, v))):
            seg = np.linspace(a_, b_, 120)
            m = np.hypot(seg[:, 0], seg[:, 1]) <= R
            if m.sum() > 1:
                q = pr(np.c_[seg[m] + foot[:2], np.zeros(m.sum())])
                cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1)

    # every flight: ground shadow + path; endpoints as dots
    for i, P in enumerate(T):
        sh_ = pr(np.c_[P[:, :2], np.zeros(len(P))])
        cv.line(sh_[:, 0], sh_[:, 1], color="#CBD0D7", lw=1.1, z=1.2, alpha=0.55)
        if i == PAIR:
            continue
        q = pr(P)
        cv.line(q[:, 0], q[:, 1], color="#9AA0AA", lw=0.65, z=2)
        cv.ax.scatter(*q[[0, -1]].T, s=2.5, color="#9AA0AA", zorder=2.1, lw=0)

    # subject
    sh = pr(np.c_[foot[0] + 0.45 * np.cos(t96), foot[1] + 0.45 * np.sin(t96), np.zeros_like(t96)])
    cv.ax.add_patch(Polygon(sh, closed=True, fc="#CDD1D8", ec="none", zorder=3))
    cv.person(pr(foot)[0], pr(foot + [0, 0, hgt * 1.6])[0], z=7)     # glyph drawn 1.6x for legibility
    hd = pr(foot + [0, 0, hgt * 1.6])[0]
    cv.text(hd[0], hd[1] + 0.08, "subject", size=5.6, color=MUTED)

    # real frames from other flights, pinned to their cameras
    sub2 = pr(foot + [0, 0, hgt / 2])[0]
    for p, f, off in PINS:
        cam = pr(T[p][f])[0]
        draw_frustum(cv, pr, pairs[p]["trajectory_32f"][f], 0.45, ec=INK, fc="white", lw=0.5, z=5)
        tw, th_ = 0.40, 0.30
        if off is None:
            u_ = (cam - sub2) / (np.linalg.norm(cam - sub2) + 1e-9)
            off = 0.32 * u_
        cxy = cam + np.array(off)
        cxy[1] = min(cxy[1], rect[1] + rect[3] - th_ / 2 - 0.02)          # keep clear of the panel title
        cxy[0] = np.clip(cxy[0], rect[0] + tw / 2 + 0.04, rect[0] + rect[2] - tw / 2 - 0.04)
        cv.line([cam[0], cxy[0]], [cam[1], cxy[1]], color=INK, lw=0.4, ls=(0, (1, 1.2)), z=4)
        cv.image(load(D / f"renders/pair_{p:02d}_frame_{f:02d}.jpg", 300), cxy[0] - tw / 2, cxy[1] - th_ / 2,
                 tw, th_, ec="white", lw=1.0, z=9, shadow=True)

    # the highlighted walk: one goal, three windows
    P = T[PAIR]
    q = pr(P)
    cv.line(q[:, 0], q[:, 1], color=INK, lw=0.8, z=5)
    for w, (s, e) in enumerate(WINDOWS):
        qw = pr(P[e:s + 1][::-1])
        cv.line(qw[:-1, 0], qw[:-1, 1], color=WIN[w], lw=1.3, z=6 + w * 0.1)
        cv.arrow(qw[-2], qw[-1], color=WIN[w], lw=1.3, ms=3.6, z=6 + w * 0.1)
        m = qw[len(qw) // 2] + np.array(OWIN_LAB[w])
        cv.text(*m, f"$w_{w + 1}$", size=6.5, color="#3E9A70" if w == 0 else WIN[w], z=10)
    look = pr(np.array(d["subject_center"]))[0]
    As, Cs = draw_frustum(cv, pr, traj[S_IDX], 0.55, ec=INK, fc="#F1F1F3", lw=0.8, z=8)
    Ag, Cg = draw_frustum(cv, pr, traj[G_IDX], 0.55, ec=BLUE, fc=BLUE_L, lw=1.0, z=8.5)
    cv.line([Ag[0], look[0]], [Ag[1], look[1]], color=BLUE, lw=0.6, ls=(0, (2, 2)), z=4)

    # the walk's start and end frames, pinned beside the path
    tw, th_ = 0.50, 0.375
    xr = rect[0] + rect[2] - tw / 2 - 0.06
    for A_, f_, yc_, ec_, lab in ((As, S_IDX, As[1] + OSTART_DY, INK, r"start $o_{26}$"),
                                  (Ag, G_IDX, Ag[1] + OEND_DY, BLUE, r"end $o_g$ ($g=4$)")):
        cv.line([A_[0] + 0.04, xr - tw / 2], [A_[1], yc_], color=ec_, lw=0.45, ls=(0, (1, 1.2)), z=4)
        cv.image(load(D / f"renders/pair_{PAIR:02d}_frame_{f_:02d}.jpg", 400), xr - tw / 2, yc_ - th_ / 2, tw, th_,
                 ec=ec_, lw=1.0, z=9.5, shadow=True)
        cv.text(xr - tw / 2, yc_ + th_ / 2 + 0.03, lab, size=5.6, ha="left", va="bottom",
                color=BLUE_D if ec_ == BLUE else INK, weight="semibold", z=10)
    ye = Ag[1] + OEND_DY
    cv.shutter(xr + tw / 2 - 0.05, ye - th_ / 2 + 0.05, r=0.032, z=10)
    # the projected subject box on the end frame is where its label comes from
    bx = d["render_records"][PAIR][G_IDX]["bbox_xyxy_full"]
    sx, sy = tw / 1024, th_ / 768
    x0e, y0e = xr - tw / 2, ye - th_ / 2
    cv.ax.add_patch(Rectangle((x0e + bx[0] * sx, y0e + th_ - bx[3] * sy), (min(bx[2], 1022) - bx[0]) * sx,
                              (bx[3] - bx[1]) * sy, fc="none", ec=BLUE, lw=0.7, ls=(0, (2, 1.2)), zorder=9.7))
    end_bottom = y0e

    # bottom row: training pair and scale on the left; the end frame's hindsight label on the right
    yg, th = Y0 + 0.12, 0.50
    LW = 0.92
    LX = xr + tw / 2 - LW
    cv.box(LX, yg, LW, th, fc="white", ec=BLUE, lw=0.8, r=0.05, shadow=True)
    cv.text(LX + 0.06, yg + th - 0.08, r"label $p(o_g)$", size=5.6, color=BLUE_D, ha="left", weight="semibold")
    cv.text(LX + 0.06, yg + th / 2 - 0.07, "bearing   222°\nelevation  −1°\noccupancy  18%\ncentre  895,351",
            size=4.5, color=INK, ha="left", family="IBM Plex Mono", linespacing=1.2)
    cv.arrow((xr, end_bottom - 0.01), (xr, yg + th + 0.01), color=BLUE, lw=0.7, ms=3.2)
    cv.text(xr - 0.05, (end_bottom + yg + th) / 2, "from the mesh", size=5.0, color=BLUE_D, ha="right")
    QX = X0 + 0.14
    cv.text(QX, yg + 0.62, "no reward, no human label:\nevery frame a flight reaches is a goal",
            size=5.2, color=MUTED, ha="left", va="top", linespacing=1.2)
    cv.text(QX, yg + 0.25, r"$(o_s,\ p(o_g))\ \mapsto\ (a_{s:s+8},\ o_{s+1:s+8})$", size=7.0, ha="left",
            va="bottom")
    cv.text(QX, yg, "12 orbits × 32 frames per placement\n91 scenes × 102 subjects → 150k windows",
            size=5.0, color=MUTED, ha="left", va="bottom", linespacing=1.2)
    return dict(D=D, traj=traj)


def panel_a_behind(cv, root):
    X0, Y0, PW, PH = 0.04, 1.93, 3.46, 2.99
    cv.panel(X0, Y0, PW, PH, "a", "Self-labelling data engine")
    D = root / "data/trajectories/v7_stage2_renders_lookat075" / KEY
    d = json.load(open(D / "data.json"))
    foot = np.array(d["subject_foot"]); hgt = d["subject_height"]
    pairs = d["accepted_pairs"]
    T = [np.array([f["pos"] for f in p["trajectory_32f"]]) for p in pairs]
    traj = pairs[PAIR]["trajectory_32f"]
    im_s = load(D / f"renders/pair_{PAIR:02d}_frame_{S_IDX:02d}.jpg", 500)
    im_g = load(D / f"renders/pair_{PAIR:02d}_frame_{G_IDX:02d}.jpg", 700)

    rect = (X0 + 0.04, Y0 + 0.42, 2.42, 2.28)
    pr = Proj(foot + [1.0, 1.6, 0.8], az_deg=AZ, el_deg=EL, dist=DIST, fov_deg=30)
    around = np.c_[foot[0] + 1.2 * np.cos(np.linspace(0, 2 * np.pi, 16)),
                   foot[1] + 1.2 * np.sin(np.linspace(0, 2 * np.pi, 16)), np.zeros(16)]
    pr.fit(np.vstack([T[PAIR], around, foot + [0, 0, hgt]]), rect, margin=0.26)
    clip = FancyBboxPatch(rect[:2], rect[2], rect[3], boxstyle="round,pad=0,rounding_size=0.08",
                          transform=cv.ax.transData)

    # ground: a soft disc with a fading grid
    t64 = np.linspace(0, 2 * np.pi, 96)
    for rr, a in ((RV, 0.55), (RV * 0.72, 0.6), (RV * 0.45, 0.7)):
        g = pr(np.c_[foot[0] + rr * np.cos(t64), foot[1] + rr * np.sin(t64), np.zeros_like(t64)])
        pg = Polygon(g, closed=True, fc="white", ec="none", alpha=a, zorder=0.8)
        cv.ax.add_patch(pg)
        pg.set_clip_path(clip)
    for v in np.arange(-RV, RV + 0.01, 1.0):
        for a_, b_ in (((v, -RV), (v, RV)), ((-RV, v), (RV, v))):
            seg = np.linspace(a_, b_, 80)
            m = np.hypot(seg[:, 0], seg[:, 1]) <= RV
            if m.sum() < 2:
                continue
            q = pr(np.c_[seg[m] + foot[:2], np.zeros(m.sum())])
            cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1).set_clip_path(clip)

    # the other eleven flights, faded
    for i, P in enumerate(T):
        if i != PAIR:
            q = pr(P)
            cv.line(q[:, 0], q[:, 1], color=FAINT, lw=0.55, z=2, alpha=0.8).set_clip_path(clip)

    # highlighted flight: ground shadow, drop lines, path, small frusta
    P = T[PAIR]
    shd = pr(np.c_[P[:, :2], np.zeros(len(P))])
    cv.line(shd[:, 0], shd[:, 1], color="#BFC4CC", lw=1.6, z=1.5, alpha=0.6)
    for f in (S_IDX, G_IDX):
        top_, bot_ = pr(P[f])[0], pr(np.r_[P[f][:2], 0])[0]
        cv.line([top_[0], bot_[0]], [top_[1], bot_[1]], color="#AEB3BB", lw=0.5, ls=(0, (1.5, 1.5)), z=1.6)
        cv.ax.add_patch(Ellipse(bot_, 0.05, 0.02, fc="#AEB3BB", ec="none", zorder=1.6))
    q = pr(P)
    cv.line(q[:, 0], q[:, 1], color=INK, lw=0.7, z=5)
    for f in range(0, 32, 3):
        if f > S_IDX + 1:
            draw_frustum(cv, pr, traj[f], 0.26, ec="#8A8E96", fc="white", lw=0.4, z=5)

    # subject
    sh = pr(np.c_[foot[0] + 0.40 * np.cos(t64), foot[1] + 0.40 * np.sin(t64), np.zeros_like(t64)])
    cv.ax.add_patch(Polygon(sh, closed=True, fc="#CDD1D8", ec="none", zorder=3))
    cv.person(pr(foot)[0], pr(foot + [0, 0, hgt])[0], z=7)
    sub = pr(foot + [0, 0, hgt * 0.75])[0]
    cv.text(sub[0] - 0.16, sub[1], "subject", size=5.8, color=MUTED, ha="right")

    # one goal, three windows
    look = pr(np.array(d["subject_center"]))[0]
    for w, (s, e) in enumerate(WINDOWS):
        qw = pr(P[e:s + 1][::-1])
        cv.line(qw[:-1, 0], qw[:-1, 1], color=WIN[w], lw=2.4, z=6 + w * 0.1)
        cv.arrow(qw[-2], qw[-1], color=WIN[w], lw=2.4, ms=5.5, z=6 + w * 0.1)
        m = qw[len(qw) // 2] + np.array(WIN_LAB[w])
        cv.text(*m, f"$w_{w + 1}$", size=7.2, color="#3E9A70" if w == 0 else WIN[w])

    # start and goal cameras carry their rendered frames
    As, Cs = draw_frustum(cv, pr, traj[S_IDX], 0.62, ec=INK, fc="white", lw=0.8, z=8, img=im_s)
    Ag, Cg = draw_frustum(cv, pr, traj[G_IDX], 0.62, ec=BLUE, fc="white", lw=1.0, z=8.5, img=im_g)
    cv.line([Ag[0], look[0]], [Ag[1], look[1]], color=BLUE, lw=0.6, ls=(0, (2, 2)), z=4)
    cv.text(Cs[:, 0].min() - 0.04, Cs[:, 1].max() - 0.02, r"$o_{26}$", size=8, ha="right")
    cv.text(Cg[:, 0].min() - 0.04, Cg[:, 1].mean(), r"$o_g$", size=8, color=BLUE_D, ha="right")
    sx_, sy_ = Cg[:, 0].max() + 0.09, Cg[:, 1].min() - 0.06
    cv.shutter(sx_, sy_, r=0.04)
    cv.text(sx_ + 0.06, sy_, "shoot = 1", size=5.6, ha="left")

    # goal close-up: projected subject box -> hindsight label
    TX, TW = X0 + 2.50, 0.88
    th = TW * 0.75
    yg = Y0 + PH - 0.42 - th
    cv.image(im_g, TX, yg, TW, th, ec=BLUE, lw=1.0, shadow=True)
    cv.text(TX, yg + th + 0.05, r"goal $o_g$", size=6, ha="left", va="bottom", color=BLUE_D, weight="semibold")
    bx = d["render_records"][PAIR][G_IDX]["bbox_xyxy_full"]
    sx, sy = TW / 1024, th / 768
    x0b, x1b = TX + bx[0] * sx, TX + min(bx[2], 1022) * sx
    y0b, y1b = yg + th - bx[3] * sy, yg + th - bx[1] * sy
    cv.ax.add_patch(Rectangle((x0b, y0b), x1b - x0b, y1b - y0b, fc="none", ec=BLUE, lw=0.8,
                              ls=(0, (2.5, 1.5)), zorder=6))
    cv.ax.scatter([(x0b + x1b) / 2], [(y0b + y1b) / 2], s=6, marker="+", color=BLUE, lw=0.8, zorder=6)

    py = yg - 0.80
    cv.box(TX, py, TW, 0.62, fc="white", ec=BLUE, lw=0.8, r=0.05, shadow=True)
    cv.text(TX + 0.06, py + 0.51, r"label $p(o_g)$", size=6, color=BLUE_D, ha="left", weight="semibold")
    cv.text(TX + 0.06, py + 0.23, "bearing   222°\nelevation  −1°\noccupancy  18%\ncentre  895,351",
            size=4.8, color=INK, ha="left", family="IBM Plex Mono", linespacing=1.3)
    cv.arrow((TX + TW / 2, yg - 0.01), (TX + TW / 2, py + 0.625), color=BLUE, lw=0.7, ms=3.2)
    cv.text(TX + TW / 2, py - 0.06, "read off the mesh:\nno reward, no human", size=5.0,
            color=MUTED, va="top", linespacing=1.15)

    cv.text(X0 + 0.12, Y0 + 0.30, "one goal → three 8-step windows · 12 flights × 32 frames per placement"
            "\n91 scenes × 102 subjects → 150k training windows", size=5.4, color=MUTED, ha="left", va="bottom",
            linespacing=1.25, bbox=dict(fc=PANEL, ec="none", pad=1.5, alpha=0.9), z=9)
    cv.text(X0 + 0.12, Y0 + 0.08, r"$(o_s,\ p(o_g))\ \mapsto\ (a_{s:s+8},\ o_{s+1:s+8})$",
            size=7.2, color=INK, ha="left", va="bottom")
    return dict(D=D, traj=traj)


# ---------------------------------------------------------------- (b) world action model
def draw_action_tokens(cv, A, x, y, c=0.074, h=0.034):
    """8 action tokens as columns of their 10 real values: dt (3), rot6d (6), shoot (1)."""
    top = y
    for i in range(8):
        cx = x + i * c
        for r in range(10):
            v = A[i, r]
            if r < 3:
                fc = plt.cm.PRGn(0.5 + 0.42 * np.clip(v / 0.06, -1, 1))
            elif r < 9:
                fc = plt.cm.Greys(0.07 + 0.40 * abs(v))
            else:
                fc = INK if v > 0.5 else "white"
            yy = top - (r + 1) * h - (0.02 if r >= 3 else 0) - (0.02 if r == 9 else 0)
            cv.ax.add_patch(Rectangle((cx, yy), c * 0.84, h, fc=fc, ec="white" if r < 9 else INK,
                                      lw=0.3 if r < 9 else 0.45, zorder=6))
        gx, gy = cx + c * 0.42, top + 0.07
        if A[i, 9] > 0.5:
            cv.shutter(gx, gy, r=0.022)
        else:
            v = np.array([A[i, 0], -A[i, 1]])
            v = 0.027 * v / (np.linalg.norm(v) + 1e-9)
            cv.arrow((gx - v[0], gy - v[1]), (gx + v[0], gy + v[1]), color=GREEN_D, lw=0.6, ms=2.4, z=7)
    lx = x + 8 * c + 0.03
    cv.text(lx, top - 1.5 * h, r"$\Delta t$", size=5.6, ha="left", color=GREEN_D)
    cv.text(lx, top - 6 * h - 0.02, r"$R_{6D}$", size=5.6, ha="left", color=MUTED)
    cv.text(lx, top - 9.5 * h - 0.04, "shoot", size=5.0, ha="left")
    return top - 10 * h - 0.04


def panel_b(cv, assets):
    X0, Y0, PW, PH = 3.58, 1.93, 3.54, 2.99
    cv.panel(X0, Y0, PW, PH, "b", "World camera action model")
    X0 += 0.06
    D, traj = assets["D"], assets["traj"]
    s, g = WINDOWS[-1]
    A, fr = action_chunk(traj, s, g)
    yT, yV0, yV1, yA = Y0 + 2.46, Y0 + 1.94, Y0 + 1.42, Y0 + 0.90
    xin, xe, xt, xm, wm = X0, X0 + 0.55, X0 + 0.92, X0 + 1.45, 0.88
    xo = xm + wm + 0.07
    rng = np.random.default_rng(0)

    # inputs
    cv.box(xin + 0.02, yT - 0.14, 0.40, 0.28, fc=BLUE_L, ec=BLUE, lw=0.7, r=0.05, shadow=True)
    cv.text(xin + 0.22, yT, r"$p$", size=8.5, color=BLUE_D)
    cv.text(xin + 0.22, yT - 0.19, "profile prompt", size=5.0, color=MUTED, va="top")
    th = 0.30
    cv.image(load(D / f"renders/pair_{PAIR:02d}_frame_{s:02d}.jpg", 400), xin + 0.02, yV0 - th / 2, 0.40, th,
             ec=INK, lw=0.5, shadow=True)
    cv.text(xin + 0.22, yV0 - th / 2 - 0.04, r"$o_s$", size=7, va="top")
    nz = (rng.normal(0.5, 0.20, (24, 32)).clip(0, 1) * 255).astype(np.uint8)
    cv.image(Image.fromarray(nz).convert("RGB"), xin + 0.02, yV1 - th / 2, 0.40, th, ec=MUTED, lw=0.5, shadow=True)
    cv.text(xin + 0.22, yV1 - th / 2 - 0.04, r"$\epsilon\sim\mathcal{N}(0,I)$", size=6, va="top")
    noise = rng.normal(0, 1, (10, 8))
    cv.ax.add_patch(Rectangle((xin + 0.02 + 0.018, yA - 0.15 - 0.022), 0.40, 0.30, fc=SHADOW, ec="none", zorder=3.9))
    cv.ax.imshow(noise, extent=(xin + 0.02, xin + 0.42, yA - 0.15, yA + 0.15), cmap="Greens", vmin=-3, vmax=3,
                 zorder=4, aspect="auto", interpolation="nearest")
    cv.ax.add_patch(Rectangle((xin + 0.02, yA - 0.15), 0.40, 0.30, fc="none", ec=GREEN, lw=0.6, zorder=4.1))
    cv.text(xin + 0.22, yA - 0.19, r"$a_\tau\in\mathbb{R}^{8\times10}$", size=6, va="top")

    # encoders
    cv.box(xe, yT - 0.11, 0.26, 0.22, fc="white", ec=BLUE, lw=0.6, r=0.05)
    cv.text(xe + 0.13, yT, "tok", size=6, color=BLUE_D)
    cv.trapezoid(xe, yV0 - 0.15, 0.26, 0.30, wide_left=True, label=r"$\mathcal{E}$")
    cv.ax.add_patch(Circle((xe + 0.13, yV1), 0.08, fc="white", ec=VIS, lw=0.6, zorder=4))
    cv.text(xe + 0.13, yV1, r"$\tau$", size=7, color=VIS)
    cv.box(xe, yA - 0.11, 0.26, 0.22, fc="white", ec=GREEN, lw=0.6, r=0.05)
    cv.text(xe + 0.13, yA, r"$W_a$", size=7, color=GREEN_D)
    for y, c in ((yT, BLUE), (yV0, INK), (yV1, MUTED), (yA, GREEN)):
        cv.arrow((xin + 0.44, y), (xe - 0.01, y), color=c, lw=0.55, ms=3.5)
    cv.arrow((xe + 0.13, yV0 - 0.12), (xe + 0.13, yV1 + 0.085), color=VIS, lw=0.55, ms=3.2)
    cv.text(xe + 0.17, (yV0 + yV1) / 2 - 0.04, r"$o_{s+1:s+8}$", size=5.6, color=VIS, ha="left")

    # token columns
    cv.tokens(xt, yT - 0.031, 4, BLUE_L, BLUE)
    cv.text(xt + 0.32, yT, "…", size=6, color=BLUE, ha="left")

    def tile(x, y, fc, hatch=None):
        cv.ax.add_patch(FancyBboxPatch((x, y - 0.11), 0.22, 0.22, boxstyle="round,pad=0,rounding_size=0.02",
                                       fc=fc, ec=VIS, lw=0.55, zorder=5, hatch=hatch))
        for k in range(1, 4):
            cv.line([x + k * 0.055] * 2, [y - 0.11, y + 0.11], color=VIS, lw=0.2, z=5.1, alpha=0.5)
            cv.line([x, x + 0.22], [y - 0.11 + k * 0.055] * 2, color=VIS, lw=0.2, z=5.1, alpha=0.5)

    tile(xt, yV0, VIS_L)
    cv.text(xt + 0.27, yV0 + 0.04, r"$z^0$", size=7, ha="left", color=VIS)
    cv.text(xt + 0.27, yV0 - 0.09, "clean", size=4.9, ha="left", color=MUTED)
    tile(xt, yV1 + 0.04, "white", hatch="//////")
    tile(xt + 0.05, yV1 - 0.03, "white", hatch="//////")
    cv.text(xt + 0.31, yV1 + 0.04, r"$z^{1:2}_\tau$", size=7, ha="left", color=VIS)
    cv.text(xt + 0.31, yV1 - 0.09, "noisy", size=4.9, ha="left", color=MUTED)
    cv.tokens(xt, yA - 0.031, 4, GREEN_L, GREEN, hatch="//////")
    cv.text(xt + 0.32, yA, "×8", size=5.5, color=GREEN, ha="left")
    for y, c in ((yT, BLUE), (yV0, VIS), (yV1, VIS), (yA, GREEN)):
        cv.arrow((xe + 0.27, y), (xt - 0.02, y), color=c, lw=0.55, ms=3.5)
        cv.arrow((xt + 0.47 if y != yT else xt + 0.40, y), (xm - 0.005, y), color=c, lw=0.55, ms=3.5)

    # ---- Mixture-of-Transformers: two experts, one joint attention
    top, bot = yT + 0.24, yA - 0.30
    cv.box(xm, bot, wm, top - bot, fc="white", ec=INK, lw=0.8, r=0.07, z=2, shadow=True)
    cx = xm + wm / 2
    cv.box(xm + 0.06, yT - 0.15, wm - 0.12, 0.31, fc=FROZEN_L, ec=FAINT, lw=0.6, r=0.05, z=3)
    cv.text(cx, yT + 0.055, "VLM expert", size=5.9, weight="semibold")
    cv.text(cx, yT - 0.075, "Qwen3-VL-8B · frozen", size=4.6, color=MUTED)
    ja_t, ja_b = yT - 0.24, yT - 0.44
    cv.box(xm + 0.06, ja_b, wm - 0.12, ja_t - ja_b, fc="#EEF0FA", ec=VIS, lw=0.6, r=0.04, z=3)
    cv.text(cx, (ja_t + ja_b) / 2, r"joint attn · $T\Vert V\Vert A$", size=4.9)
    cv.arrow((cx - 0.18, yT - 0.155), (cx - 0.18, ja_t + 0.005), color=BLUE, lw=0.8, ms=3.5)
    cv.text(cx - 0.15, (yT - 0.155 + ja_t) / 2, r"$K,V$", size=5.4, color=BLUE_D, ha="left")
    wt, wb = ja_b - 0.08, bot + 0.07
    for k in (2, 1):
        cv.box(xm + 0.06 + 0.025 * k, wb - 0.025 * k, wm - 0.12, wt - wb, fc=GREEN_L, ec=GREEN, lw=0.4, r=0.05,
               z=2.6 - 0.1 * k)
    cv.box(xm + 0.06, wb, wm - 0.12, wt - wb, fc=GREEN_L, ec=GREEN, lw=0.7, r=0.05, z=3)
    cv.arrow((cx + 0.18, wt + 0.005), (cx + 0.18, ja_b - 0.005), color=GREEN, lw=0.8, ms=3.5)
    cv.arrow((cx + 0.06, ja_b - 0.005), (cx + 0.06, wt + 0.005), color=GREEN, lw=0.8, ms=3.5)
    cv.text(cx + 0.02, (wt + ja_b) / 2, r"$Q,K,V$", size=4.8, color=GREEN_D, ha="right")
    cv.text(cx, wt - 0.10, "World-action expert", size=5.2, weight="semibold", color=GREEN_D)
    cv.text(cx, wt - 0.22, "trained", size=4.8, color=GREEN_D)
    bx_, bw = xm + 0.16, wm - 0.32
    for yy, lab in ((yV1 + 0.08, r"$W^{G}_{qkv}$"), (yV1 - 0.20, r"$\mathrm{FFN}^{G}$")):
        cv.box(bx_, yy - 0.09, bw, 0.18, fc="white", ec=GREEN, lw=0.5, r=0.04, z=3.5)
        cv.text(bx_ + bw / 2, yy, lab, size=5.2, color=GREEN_D, z=6)
    cv.arrow((bx_ + bw / 2, yV1 - 0.01), (bx_ + bw / 2, yV1 - 0.11 + 0.001), color=GREEN, lw=0.5, ms=2.6, z=4)
    cv.text(bx_ + bw / 2, yA - 0.04, r"$+\ \mathrm{emb}(\tau)$", size=5.0, color=MUTED)
    cv.text(xm + wm - 0.10, wb + 0.03, "×36", size=5.4, ha="right", va="bottom", color=GREEN_D, weight="semibold")

    # ---- outputs
    cv.arrow((xm + wm, yV1), (xo + 0.04, yV1), color=VIS, lw=0.6, ms=3.5)
    cv.text(xo - 0.03, yV1 + 0.14, r"$\hat v_z$", size=7, color=VIS, ha="left")
    cv.trapezoid(xo + 0.05, yV1 - 0.14, 0.20, 0.28, wide_left=False, label=r"$\mathcal{D}$")
    fx, fw, fh = xo + 0.33, 0.36, 0.27
    f1 = load(D / f"renders/pair_{PAIR:02d}_frame_{fr[4]:02d}.jpg", 400)
    f2 = load(D / f"renders/pair_{PAIR:02d}_frame_{fr[8]:02d}.jpg", 400)
    cv.image(f1, fx, yV1 + 0.02, fw, fh, ec=VIS, lw=0.5, shadow=True)
    cv.image(f2, fx + 0.08, yV1 - 0.19, fw, fh, ec=VIS, lw=0.5, z=4.5, shadow=True)
    cv.text(fx + 0.22, yV1 + 0.37, r"$\hat o_{s+4},\ \hat o_{s+8}$", size=6.0, color=VIS)
    cv.arrow((xo + 0.26, yV1), (fx - 0.01, yV1), color=VIS, lw=0.55, ms=3)

    cv.arrow((xm + wm, yA), (xo + 0.04, yA), color=GREEN, lw=0.6, ms=3.5)
    cv.text(xo - 0.03, yA + 0.14, r"$\hat v_a$", size=7, color=GREEN_D, ha="left")
    ax0 = xo + 0.10
    btm = draw_action_tokens(cv, A, ax0, yA + 0.16)
    cv.text(ax0 + 0.30, btm - 0.04, r"$\hat a$: 8 tokens × 10-D", size=5.2, color=INK, va="top")
    cv.text(ax0 + 0.30, btm - 0.16, "6 moves, then shoot", size=5.0, color=MUTED, va="top")

    cv.text(xo + 0.46, Y0 + 0.24, r"$\mathcal{L}=\|\hat v_z-v_z\|^2+\|\hat v_a-v_a\|^2$", size=6.0)
    cv.text(xo + 0.46, Y0 + 0.09, "rectified flow, shared τ", size=5.0, color=MUTED)

    # attention mask (rows = queries, cols = keys)
    mx, my, ms_ = X0 + 0.04, Y0 + 0.08, 0.34
    n = [3, 3, 2]
    N, c = sum(n), ms_ / sum(n)
    edges = np.cumsum([0] + n)
    cols = [BLUE, VIS, GREEN]
    for i in range(N):
        gi = np.searchsorted(edges, i, side="right") - 1
        for j in range(N):
            on = (j <= i and j < n[0]) if gi == 0 else True
            cv.ax.add_patch(Rectangle((mx + j * c, my + ms_ - (i + 1) * c), c, c, fc=cols[gi] if on else "white",
                                      ec="white", lw=0.35, zorder=5, alpha=0.8 if on else 1))
    for k, lab in enumerate("TVA"):
        cv.text(mx + (edges[k] + n[k] / 2) * c, my + ms_ + 0.04, lab, size=4.8, color=cols[k], va="bottom")
    cv.text(mx + ms_ + 0.08, my + ms_ / 2,
            "attention mask\nno KV cache across chunks",
            size=5.0, color=MUTED, ha="left", va="center", linespacing=1.3)


# ---------------------------------------------------------------- (c) deployment
def panel_c(cv, root):
    X0, Y0, PW, PH = 0.04, 0.04, 7.08, 1.81
    cv.panel(X0, Y0, PW, PH, "c", "Deployment: intent → shot profile → closed-loop motion")
    ep = next(e for e in json.load(open(root / "runs/gif/rollout_ours_v6_12k.json"))["episodes"]
              if e["key"].replace(":", "_") == KEY_C)
    yc = Y0 + 0.86
    x = X0 + 0.12

    cv.box(x, yc - 0.36, 1.02, 0.72, fc=ORANGE_L, ec=ORANGE, lw=0.7, r=0.06, shadow=True)
    cv.text(x + 0.07, yc + 0.25, "user intent", size=5.8, color=ORANGE, ha="left", weight="semibold")
    cv.text(x + 0.07, yc - 0.01, "“A wide shot from her\nback-left, full body,\ncentred, eye level.”",
            size=5.5, ha="left", style="italic", linespacing=1.2)
    cv.text(x + 0.07, yc - 0.27, "or a reference photo", size=5.0, color=MUTED, ha="left")
    xl = x + 1.14
    cv.box(xl, yc - 0.24, 0.60, 0.48, fc="white", ec=ORANGE, lw=0.7, r=0.06, shadow=True)
    cv.text(xl + 0.30, yc + 0.07, "LLM / VLM", size=6.0, weight="semibold", color=ORANGE)
    cv.text(xl + 0.30, yc - 0.10, "off-the-shelf", size=4.9, color=MUTED)
    cv.arrow((x + 1.02, yc), (xl - 0.01, yc), color=ORANGE, lw=0.7, ms=4)

    xp = xl + 0.74
    cv.box(xp, yc - 0.48, 1.22, 0.96, fc=BLUE_L, ec=BLUE, lw=0.9, r=0.07, shadow=True)
    cv.text(xp + 0.07, yc + 0.36, r"shot profile $p$", size=6.6, color=BLUE_D, ha="left", weight="semibold")
    cv.text(xp + 0.07, yc + 0.15, "wide · back-left · eye level\ncentred · full body", size=5.0, color=BLUE_D,
            ha="left", linespacing=1.25)
    cv.text(xp + 0.07, yc - 0.22, "bearing    214°\nelevation   −7°\noccupancy   10%\ncentre  468,364",
            size=4.9, ha="left", family="IBM Plex Mono", linespacing=1.3)
    cv.arrow((xl + 0.60, yc), (xp - 0.01, yc), color=ORANGE, lw=0.7, ms=4)

    xw = xp + 1.36
    cv.box(xw, yc - 0.30, 0.60, 0.60, fc=GREEN_L, ec=GREEN, lw=0.8, r=0.07, shadow=True)
    cv.text(xw + 0.30, yc + 0.09, "world camera\naction model", size=5.5, weight="semibold", color=GREEN_D,
            linespacing=1.0)
    cv.text(xw + 0.30, yc - 0.17, "panel (b)", size=4.8, color=MUTED)
    cv.arrow((xp + 1.22, yc), (xw - 0.01, yc), color=BLUE, lw=0.8, ms=4)

    F = root / "runs/gif/frames" / KEY_C / "ours"
    ends = np.cumsum([0] + ep["chunk_steps"]) * 3
    tr = ep["trace"]
    frames = [(ends[0], r"start $o_0$", f"d {tr[0]:.2f}"), (ends[2], "after chunk 2", f"{tr[2]:.2f}"),
              (ends[3], "after chunk 3", f"{tr[3]:.2f}"), (ends[4], "chunk 4 · shoot", f"{tr[4]:.3f}")]
    fw, fh, gap = 0.66, 0.495, 0.06
    xs0 = xw + 0.76
    yf = yc - fh / 2 + 0.08
    for k, (i, lab, dd) in enumerate(frames):
        xx = xs0 + k * (fw + gap)
        last = k == len(frames) - 1
        cv.image(load(F / f"f{i:04d}.jpg", 600), xx, yf, fw, fh, ec=GREEN if last else None, lw=1.4, shadow=True)
        cv.text(xx, yf + fh + 0.05, lab, size=5.5, ha="left", va="bottom", color=GREEN_D if last else INK,
                weight="semibold" if last else "normal")
        cv.text(xx + fw - 0.03, yf + 0.03, dd, size=4.9, ha="right", va="bottom", color="white",
                family="IBM Plex Mono", z=7,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.3", fc=(0, 0, 0, 0.45), ec="none"))
        if k:
            cv.arrow((xx - gap + 0.008, yf + fh / 2), (xx - 0.008, yf + fh / 2), color=GREEN, lw=0.6, ms=3)
    cv.arrow((xw + 0.60, yc), (xs0 - 0.01, yf + fh / 2), color=GREEN, lw=0.8, ms=4)
    cv.text(xw + 0.68, yc + 0.17, r"$\hat a_{1:8}$", size=6.3, color=GREEN_D)

    sh = np.array(ep["shoot"][3])
    xl4 = xs0 + 3 * (fw + gap)
    c4 = fw / 8
    yy = yf - 0.15
    for i, v in enumerate(sh):
        cv.ax.add_patch(Rectangle((xl4 + i * c4 + 0.005, yy), c4 - 0.01, 0.065, fc=INK if v >= 0.5 else "white",
                                  ec=INK, lw=0.4, zorder=6))
    k = int(np.argmax(sh >= 0.5))
    cv.line([xl4 + k * c4, xl4 + k * c4], [yy - 0.03, yy + 0.095], color=GREEN, lw=0.9, z=7)
    cv.text(xl4 - 0.04, yy + 0.032, "shoot", size=4.9, ha="right")
    cv.text(xl4 + fw, yy - 0.05, f"execute {k} steps, capture", size=4.8, ha="right", va="top", color=GREEN_D)

    yb = Y0 + 0.17
    xr = xs0 + 2 * (fw + gap) + fw / 2
    cv.line([xr, xr], [yf - 0.02, yb], color=MUTED, lw=0.6)
    cv.line([xr, xw + 0.30], [yb, yb], color=MUTED, lw=0.6)
    cv.arrow((xw + 0.30, yb), (xw + 0.30, yc - 0.31), color=MUTED, lw=0.6, ms=3.5)
    cv.text((xr + xw + 0.30) / 2, yb + 0.05, "re-render $o_t$, next chunk", size=5.2, color=MUTED, va="bottom")
    cv.text(X0 + PW - 0.12, Y0 + PH - 0.16, "the goal image is never shown to the policy",
            size=5.0, color=MUTED, ha="right", style="italic")


def build(a):
    """The figure for parsed arguments `a` (see args()); returns the matplotlib Figure."""
    global AZ, EL, RV, DIST, OAZ, OEL, ODIST
    root = Path(a.root)
    if a.view:
        AZ, EL, RV, DIST = map(float, a.view.split(","))
    if a.orbit_view:
        OAZ, OEL, ODIST = map(float, a.orbit_view.split(","))
    cv = Canvas()
    assets = (panel_a_orbits if a.a_style == "orbits" else panel_a_behind)(cv, root)
    panel_b(cv, assets)
    panel_c(cv, root)
    return cv


def main():
    a = args()
    cv = build(a)
    for ext in (("png",) if (a.view or a.orbit_view) else ("pdf", "svg", "png")):
        cv.fig.savefig(HERE / f"{a.out}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", HERE / f"{a.out}.png")


if __name__ == "__main__":
    main()
