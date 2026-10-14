"""Fig. 2 — method overview: (a) self-labelling data engine, (b) world action model, (c) deployment.

Every number and frame in the figure comes from the repo's data:
  (a) camera flights of placement Chill-Camping__Knit-and-Denim (data.json, 12 pairs x 32 frames),
      window s=26 -> g=4 of pair 04 (the GIF benchmark episode p04:s26:g04).
  (b) Cosmos3-Nano MoT layout from configs/policy/action_policy_camera_nano.py and the framework configs.
  (c) the closed-loop rollout of Ours v6@12k on that episode (runs/gif/frames/.../ours).

    .venv-analysis/bin/python figures/overview/make_fig2.py [--root <main checkout>]
writes fig2_overview.{pdf,svg,png} next to this file.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle
from PIL import Image

HERE = Path(__file__).resolve().parent
for f in (HERE / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "IBM Plex Sans",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "hatch.linewidth": 0.5,
})

# ---------------------------------------------------------------- palette
INK = "#1C1D21"
MUTED = "#5F636B"
FAINT = "#B4B8BF"
GRID = "#E4E6E9"
ORANGE, ORANGE_L = "#D46F1F", "#FBEADB"                  # intent / judgement
BLUE, BLUE_L, BLUE_D = "#2F62C0", "#E3EBFA", "#1D3F82"   # shot profile p (text)
GREEN, GREEN_L, GREEN_D = "#24865C", "#DFF1E7", "#17583C"  # action / trainable
VIS, VIS_L = "#6B6F9E", "#E7E7F3"                        # vision latents
FROZEN_L = "#EEEFF1"

W, H = 7.16, 4.95   # inches: full two-column width

KEY = "Chill-Camping_150d7263__Knit-and-Denim_82dd220b"
PAIR, S_IDX, G_IDX = 4, 26, 4
AZ, EL, RV, DIST = -10, 38, 4.5, 18                      # 3-D view of panel (a): azimuth, elevation, ground radius (m), eye distance
LAB_S, LAB_G, LAB_A = (0.05, 0.17), (0.0, -0.20), (-0.22, -0.06)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/nas_main/jungwooahn/projects/DronePhotographerV12")
    ap.add_argument("--view", default=None, help="az,el,radius,dist to try another 3-D view of panel (a)")
    ap.add_argument("--out", default="fig2_overview")
    return ap.parse_args()


# ---------------------------------------------------------------- primitives
class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def box(self, x, y, w, h, fc="white", ec=INK, lw=0.6, r=0.05, ls="-", z=2, hatch=None):
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
                            shrinkA=0, shrinkB=0)
        self.ax.add_patch(a)

    def line(self, xs, ys, color=INK, lw=0.6, ls="-", z=3, alpha=1):
        return self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                            solid_capstyle="round", dash_capstyle="butt")[0]

    def image(self, img, x, y, w, h=None, ec=None, lw=0.8, z=4):
        if h is None:
            h = w * img.height / img.width
        self.ax.imshow(np.asarray(img), extent=(x, x + w, y, y + h), zorder=z, interpolation="lanczos")
        if ec:
            self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.1))
        return h

    def tokens(self, x, y, n, fc, ec, s=0.062, gap=0.014, hatch=None, z=5):
        for i in range(n):
            self.ax.add_patch(Rectangle((x + i * (s + gap), y), s, s, fc=fc, ec=ec, lw=0.45, zorder=z, hatch=hatch))
        return n * (s + gap) - gap

    def trapezoid(self, x, y, w, h, wide_left=True, fc=VIS_L, ec=VIS, label="", z=4):
        d = h * 0.22
        if wide_left:
            pts = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)]
        else:
            pts = [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.6, zorder=z))
        if label:
            self.text(x + w / 2, y + h / 2, label, size=7, color=VIS, z=z + 1)

    def panel_label(self, x, y, letter, title):
        self.text(x, y, f"({letter})", size=8, weight="bold", ha="left", va="baseline")
        self.text(x + 0.22, y, title, size=8, weight="semibold", ha="left", va="baseline")


def load(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im.thumbnail((size, size), Image.LANCZOS)
    return im


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
    p = np.asarray(pose["pos"])
    f = np.asarray(pose["forward"]); f /= np.linalg.norm(f)
    u = np.asarray(pose["up"]); u /= np.linalg.norm(u)
    r = np.cross(f, u)
    hw, hh, d = 0.5 * size * 1.333, 0.5 * size, size * 1.1   # 4:3 image, 24 mm lens on a 12.8 mm sensor
    corners = [p + d * f + sx * hw * r + sy * hh * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return p, corners


def draw_frustum(cv, pr, pose, size, ec, fc, lw=0.5, z=5):
    apex, cs = frustum(pose, size)
    A = pr(apex)[0]
    C = pr(np.array(cs))
    for c in C:
        cv.line([A[0], c[0]], [A[1], c[1]], color=ec, lw=lw * 0.8, z=z)
    cv.ax.add_patch(Polygon(C, closed=True, fc=fc, ec=ec, lw=lw, zorder=z + 0.05))
    return A, C


def panel_a(cv, root):
    X0, Y0, PW, PH = 0.06, 1.98, 3.40, 2.80
    cv.panel_label(X0, Y0 + PH + 0.04, "a", "Self-labelling data engine")
    D = root / "data/trajectories/v7_stage2_renders_lookat075" / KEY
    d = json.load(open(D / "data.json"))
    foot = np.array(d["subject_foot"]); hgt = d["subject_height"]
    pairs = d["accepted_pairs"]
    T = [np.array([f["pos"] for f in p["trajectory_32f"]]) for p in pairs]
    traj = pairs[PAIR]["trajectory_32f"]

    # 3-D viewport on the left; thumbnails and the label card on the right
    rect = (X0, Y0 + 0.36, 2.22, 2.26)
    pr = Proj(foot + [1.0, 1.6, 0.8], az_deg=AZ, el_deg=EL, dist=DIST, fov_deg=30)
    ring = np.c_[foot[0] + RV * np.cos(np.linspace(0, 2 * np.pi, 64)),
                 foot[1] + RV * np.sin(np.linspace(0, 2 * np.pi, 64)), np.zeros(64)]
    # fit the subject and the highlighted flight; everything else is clipped to the viewport
    around = np.c_[foot[0] + 1.2 * np.cos(np.linspace(0, 2 * np.pi, 16)),
                   foot[1] + 1.2 * np.sin(np.linspace(0, 2 * np.pi, 16)), np.zeros(16)]
    pr.fit(np.vstack([T[PAIR], around, foot + [0, 0, hgt]]), rect, margin=0.30)
    clip = Rectangle(rect[:2], rect[2], rect[3], transform=cv.ax.transData)

    # ground grid, clipped to a disc around the subject
    for v in np.arange(-RV, RV + 0.01, 1.0):
        for a, b in (((v, -RV), (v, RV)), ((-RV, v), (RV, v))):
            seg = np.linspace(a, b, 80)
            m = np.hypot(seg[:, 0], seg[:, 1]) <= RV
            if m.sum() < 2:
                continue
            q = pr(np.c_[seg[m] + foot[:2], np.zeros(m.sum())])
            cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.35, z=1).set_clip_path(clip)
    q = pr(ring)
    cv.line(q[:, 0], q[:, 1], color=GRID, lw=0.5, z=1).set_clip_path(clip)

    # the other eleven flights
    for i, P in enumerate(T):
        if i != PAIR:
            q = pr(P)
            cv.line(q[:, 0], q[:, 1], color=FAINT, lw=0.55, z=2).set_clip_path(clip)

    # subject: body + head over a ground shadow
    th_ = np.linspace(0, 2 * np.pi, 40)
    sh = pr(np.c_[foot[0] + 0.35 * np.cos(th_), foot[1] + 0.35 * np.sin(th_), np.zeros(40)])
    cv.ax.add_patch(Polygon(sh, closed=True, fc="#D5D8DD", ec="none", zorder=3))
    b0, b1 = pr(foot + [0, 0, 0.05])[0], pr(foot + [0, 0, hgt * 0.80])[0]
    cv.line([b0[0], b1[0]], [b0[1], b1[1]], color=INK, lw=3.4, z=7)
    hd = pr(foot + [0, 0, hgt * 0.93])[0]
    cv.ax.add_patch(Circle(hd, 0.048, fc=INK, ec="none", zorder=7))
    sub = pr(foot + [0, 0, hgt])[0]
    cv.text(sub[0], sub[1] + 0.10, "subject", size=5.8, color=MUTED)

    # highlighted flight with frusta every 3rd frame
    P = T[PAIR]
    q = pr(P)
    cv.line(q[:, 0], q[:, 1], color=INK, lw=0.7, z=5)
    for f in range(0, 32, 3):
        if not (S_IDX - 9 <= f <= S_IDX + 1) and abs(f - G_IDX) > 1:
            draw_frustum(cv, pr, traj[f], 0.26, ec="#80848C", fc="white", lw=0.4, z=5)

    # the action window: 8 steps from s toward g (this one walks toward the far end)
    win = P[S_IDX - 8:S_IDX + 1][::-1]
    qw = pr(win)
    cv.line(qw[:-1, 0], qw[:-1, 1], color=GREEN, lw=2.0, z=6)
    cv.arrow(qw[-2], qw[-1], color=GREEN, lw=2.0, ms=6, z=6)
    for k in range(1, 8):
        cv.ax.scatter(*qw[k], s=4, color="white", edgecolors=GREEN, linewidths=0.6, zorder=6.5)

    # start and goal frusta, and the goal's look-at ray
    As, _ = draw_frustum(cv, pr, traj[S_IDX], 0.42, ec=INK, fc="#F1F1F3", lw=0.8, z=8)
    Ag, _ = draw_frustum(cv, pr, traj[G_IDX], 0.42, ec=BLUE, fc=BLUE_L, lw=0.9, z=8)
    look = pr(np.array(d["subject_center"]))[0]
    cv.line([Ag[0], look[0]], [Ag[1], look[1]], color=BLUE, lw=0.6, ls=(0, (2, 2)), z=4)
    cv.text(As[0] + LAB_S[0], As[1] + LAB_S[1], r"$o_s$", size=8)
    cv.text(Ag[0] + LAB_G[0], Ag[1] + LAB_G[1], r"$o_g$", size=8, color=BLUE_D)
    cv.text(qw[4][0] + LAB_A[0], qw[4][1] + LAB_A[1], r"$a_{s:s+8}$", size=7.5, color=GREEN_D)
    cv.text(rect[0] + 0.02, rect[1] + rect[3] - 0.02, "12 random flights × 32 frames\nper placement (one in black)",
            size=5.6, color=MUTED, ha="left", va="top", linespacing=1.15,
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))

    # thumbnails: o_s, then o_g with its projected box, then the hindsight label
    TX, TW = X0 + 2.30, 0.98
    th = TW * 0.75
    im_s = load(D / f"renders/pair_{PAIR:02d}_frame_{S_IDX:02d}.jpg", 600)
    im_g = load(D / f"renders/pair_{PAIR:02d}_frame_{G_IDX:02d}.jpg", 600)
    ys = Y0 + PH - 0.22 - th
    cv.image(im_s, TX, ys, TW, th, ec=INK, lw=0.8)
    cv.text(TX, ys + th + 0.05, r"start $o_s$ (frame 26)", size=5.8, ha="left", va="bottom")
    yg = ys - 0.24 - th
    cv.image(im_g, TX, yg, TW, th, ec=BLUE, lw=1.0)
    cv.text(TX, yg + th + 0.05, r"goal $o_g$ (frame 4)", size=5.8, ha="left", va="bottom", color=BLUE_D)
    bx = d["render_records"][PAIR][G_IDX]["bbox_xyxy_full"]
    sx, sy = TW / 1024, th / 768
    x0b, x1b = TX + bx[0] * sx, TX + min(bx[2], 1022) * sx
    y0b, y1b = yg + th - bx[3] * sy, yg + th - bx[1] * sy
    cv.ax.add_patch(Rectangle((x0b, y0b), x1b - x0b, y1b - y0b, fc="none", ec=BLUE, lw=0.8,
                              ls=(0, (2.5, 1.5)), zorder=6))
    cv.ax.scatter([(x0b + x1b) / 2], [(y0b + y1b) / 2], s=6, marker="+", color=BLUE, lw=0.8, zorder=6)
    cv.line([As[0] + 0.06, TX - 0.03], [As[1], ys + th / 2], color=INK, lw=0.4, ls=(0, (1, 1.5)), z=3)
    cv.line([Ag[0] + 0.07, TX - 0.03], [Ag[1], yg + th / 2], color=BLUE, lw=0.4, ls=(0, (1, 1.5)), z=3)

    py = yg - 0.66
    cv.box(TX, py, TW, 0.56, fc=BLUE_L, ec=BLUE, lw=0.7, r=0.04)
    cv.text(TX + 0.05, py + 0.46, r"hindsight label $p(o_g)$", size=5.8, color=BLUE_D, ha="left", weight="semibold")
    cv.text(TX + 0.05, py + 0.20, "bearing 222° elev −1°\nocc 18%  body 100%\nctr 895,351  vis 1.0",
            size=4.7, color=INK, ha="left", family="IBM Plex Mono", linespacing=1.3)
    cv.arrow((TX + TW / 2, yg - 0.005), (TX + TW / 2, py + 0.565), color=BLUE, lw=0.6, ms=3)
    cv.text(TX + TW / 2, py - 0.05, "from the projected mesh:\nno reward, no human label", size=5.0,
            color=MUTED, va="top", linespacing=1.15)

    cv.text(X0, Y0 + 0.20, "91 scenes × 102 subjects → 7,885 placements → 150k windows",
            size=5.6, color=INK, ha="left", va="bottom")
    cv.text(X0, Y0 + 0.01, r"$(o_s,\ p(o_g))\ \mapsto\ (a_{s:s+8},\ o_{s+1:s+8}),\quad |g-s|\leq 32$",
            size=7.0, color=INK, ha="left", va="bottom")
    return dict(im_s=im_s, D=D)


# ---------------------------------------------------------------- (b) world action model
def panel_b(cv, assets):
    X0, Y0, PH = 3.62, 1.98, 2.80
    cv.panel_label(X0, Y0 + PH + 0.04, "b", "World action model")
    D = assets["D"]
    yT, yV0, yV1, yA = Y0 + 2.50, Y0 + 1.98, Y0 + 1.46, Y0 + 0.94
    xin, xe, xt, xm, wm = X0, X0 + 0.56, X0 + 0.94, X0 + 1.50, 0.80
    xo = xm + wm + 0.06
    rng = np.random.default_rng(0)

    # inputs
    cv.box(xin + 0.02, yT - 0.14, 0.40, 0.28, fc=BLUE_L, ec=BLUE, lw=0.7, r=0.04)
    cv.text(xin + 0.22, yT, r"$p$", size=8, color=BLUE_D)
    cv.text(xin + 0.22, yT - 0.20, "JSON prompt", size=5.0, color=MUTED, va="top")
    th = 0.30
    cv.image(assets["im_s"], xin + 0.02, yV0 - th / 2, 0.40, th, ec=INK, lw=0.6)
    cv.text(xin + 0.22, yV0 - th / 2 - 0.05, r"$o_s$", size=7, va="top")
    nz = (rng.normal(0.5, 0.22, (24, 32, 3)).clip(0, 1) * 255).astype(np.uint8)
    cv.image(Image.fromarray(nz), xin + 0.02, yV1 - th / 2, 0.40, th, ec=MUTED, lw=0.5)
    cv.text(xin + 0.22, yV1 - th / 2 - 0.05, r"$\epsilon\sim\mathcal{N}(0,I)$", size=6, va="top")
    act = rng.normal(0, 1, (8, 10))
    cv.ax.imshow(act.T, extent=(xin + 0.02, xin + 0.42, yA - 0.13, yA + 0.13), cmap="Greens",
                 zorder=4, aspect="auto", interpolation="nearest")
    cv.ax.add_patch(Rectangle((xin + 0.02, yA - 0.13), 0.40, 0.26, fc="none", ec=GREEN, lw=0.6, zorder=4.1))
    cv.text(xin + 0.22, yA - 0.18, r"$a_\tau\in\mathbb{R}^{8\times10}$", size=6, va="top")

    # encoders: tokenizer, VAE encoder E, noising node, action projection W_a
    cv.box(xe, yT - 0.11, 0.26, 0.22, fc="white", ec=BLUE, lw=0.6, r=0.03)
    cv.text(xe + 0.13, yT, "tok", size=6, color=BLUE_D)
    cv.trapezoid(xe, yV0 - 0.15, 0.26, 0.30, wide_left=True, label=r"$\mathcal{E}$")
    cv.ax.add_patch(Circle((xe + 0.13, yV1), 0.08, fc="white", ec=VIS, lw=0.6, zorder=4))
    cv.text(xe + 0.13, yV1, r"$\tau$", size=7, color=VIS)
    cv.box(xe, yA - 0.11, 0.26, 0.22, fc="white", ec=GREEN, lw=0.6, r=0.03)
    cv.text(xe + 0.13, yA, r"$W_a$", size=7, color=GREEN_D)
    for y, c in ((yT, BLUE), (yV0, INK), (yV1, MUTED), (yA, GREEN)):
        cv.arrow((xin + 0.43, y), (xe - 0.01, y), color=c, lw=0.5, ms=3.5)
    # E also encodes the future frames, which are noised to time tau
    cv.arrow((xe + 0.13, yV0 - 0.12), (xe + 0.13, yV1 + 0.085), color=VIS, lw=0.5, ms=3.2)
    cv.text(xe + 0.17, (yV0 + yV1) / 2 - 0.04, r"$o_{s+1:s+8}$", size=5.6, color=VIS, ha="left")

    # token columns
    cv.tokens(xt, yT - 0.031, 4, BLUE_L, BLUE)
    cv.text(xt + 0.31, yT, "…", size=6, color=BLUE, ha="left")

    def tile(x, y, fc, hatch=None):
        cv.ax.add_patch(Rectangle((x, y - 0.11), 0.22, 0.22, fc=fc, ec=VIS, lw=0.5, zorder=5, hatch=hatch))
        for k in range(1, 4):
            cv.line([x + k * 0.055] * 2, [y - 0.11, y + 0.11], color=VIS, lw=0.2, z=5.1, alpha=0.6)
            cv.line([x, x + 0.22], [y - 0.11 + k * 0.055] * 2, color=VIS, lw=0.2, z=5.1, alpha=0.6)

    tile(xt, yV0, VIS_L)
    cv.text(xt + 0.27, yV0 + 0.04, r"$z^0$", size=7, ha="left", color=VIS)
    cv.text(xt + 0.27, yV0 - 0.09, "clean", size=4.9, ha="left", color=MUTED)
    tile(xt, yV1 + 0.04, "white", hatch="//////")
    tile(xt + 0.05, yV1 - 0.03, "white", hatch="//////")
    cv.text(xt + 0.31, yV1 + 0.04, r"$z^{1:2}_\tau$", size=7, ha="left", color=VIS)
    cv.text(xt + 0.31, yV1 - 0.09, "noisy", size=4.9, ha="left", color=MUTED)
    cv.tokens(xt, yA - 0.031, 4, GREEN_L, GREEN, hatch="//////")
    cv.text(xt + 0.31, yA, "…×8", size=5.5, color=GREEN, ha="left")
    for y, c in ((yT, BLUE), (yV0, VIS), (yV1, VIS), (yA, GREEN)):
        cv.arrow((xe + 0.27, y), (xt - 0.02, y), color=c, lw=0.5, ms=3.5)
        cv.arrow((xt + 0.47 if y != yT else xt + 0.40, y), (xm - 0.005, y), color=c, lw=0.5, ms=3.5)
    cv.text(xt + 0.24, yT + 0.17, "1 packed sequence", size=4.9, color=MUTED)

    # transformer: a frozen text tower and a trained generation tower sharing one attention
    top, bot = yT + 0.24, yA - 0.26
    cv.box(xm, bot, wm, top - bot, fc="white", ec=INK, lw=0.8, r=0.06, z=2)
    cv.box(xm + 0.06, yT - 0.15, wm - 0.12, 0.30, fc=FROZEN_L, ec=FAINT, lw=0.6, r=0.04, z=3)
    cv.text(xm + wm / 2, yT + 0.04, "text tower", size=5.8, weight="semibold")
    cv.text(xm + wm / 2, yT - 0.08, "Qwen3-VL-8B · frozen", size=4.6, color=MUTED)
    gt, gb = yV0 + 0.20, yA - 0.17
    cv.box(xm + 0.06, gb, wm - 0.12, gt - gb, fc=GREEN_L, ec=GREEN, lw=0.7, r=0.04, z=3)
    cv.text(xm + wm / 2, gt - 0.08, "generation\ntower", size=5.8, weight="semibold", color=GREEN_D,
            linespacing=1.0, va="top")
    cv.text(xm + wm / 2, gt - 0.36, "trained", size=4.8, color=GREEN_D, va="top")
    cv.arrow((xm + wm / 2, yT - 0.16), (xm + wm / 2, gt + 0.005), color=BLUE, lw=0.9, ms=4.5)
    cv.text(xm + wm / 2 + 0.05, (yT - 0.16 + gt) / 2, r"$K,V$", size=6.3, color=BLUE_D, ha="left")
    cy = (yV1 + yA) / 2
    cv.text(xm + wm / 2, cy + 0.04, r"$\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$", size=5.6,
            color=GREEN_D)
    cv.text(xm + wm / 2, cy - 0.15, "×36 layers", size=4.8, color=MUTED)

    # outputs: predicted future frames and the action chunk
    cv.arrow((xm + wm, yV1), (xo + 0.05, yV1), color=VIS, lw=0.6, ms=3.5)
    cv.text(xo - 0.02, yV1 + 0.14, r"$\hat v_z$", size=7, color=VIS, ha="left")
    cv.trapezoid(xo + 0.06, yV1 - 0.15, 0.22, 0.30, wide_left=False, label=r"$\mathcal{D}$")
    fx, fw, fh = xo + 0.36, 0.34, 0.255
    f22 = load(D / f"renders/pair_{PAIR:02d}_frame_22.jpg", 400)
    f18 = load(D / f"renders/pair_{PAIR:02d}_frame_18.jpg", 400)
    cv.image(f22, fx, yV1 + 0.0, fw, fh, ec=VIS, lw=0.6)
    cv.image(f18, fx + 0.08, yV1 - 0.20, fw, fh, ec=VIS, lw=0.6, z=4.5)
    cv.text(fx + 0.21, yV1 + 0.34, r"$\hat o_{s+4},\ \hat o_{s+8}$", size=6.2, color=VIS)
    cv.arrow((xo + 0.29, yV1), (fx - 0.01, yV1), color=VIS, lw=0.5, ms=3)

    cv.arrow((xm + wm, yA), (xo + 0.09, yA), color=GREEN, lw=0.6, ms=3.5)
    cv.text(xo - 0.02, yA + 0.14, r"$\hat v_a$", size=7, color=GREEN_D, ha="left")
    gx = xo + 0.10
    cv.box(gx, yA - 0.13, 0.66, 0.26, fc="white", ec=GREEN, lw=0.6, r=0.03)
    for k in range(8):
        cx = gx + 0.06 + k * 0.077
        cv.arrow((cx - 0.022, yA - 0.045 + 0.006 * k), (cx + 0.026, yA - 0.02 + 0.006 * k), color=GREEN,
                 lw=0.6, ms=2.6)
    cv.text(gx + 0.33, yA + 0.065, r"$\Delta t\!\in\!\mathbb{R}^3,\ \Delta R_{6D},\ \mathrm{shoot}$",
            size=4.7, color=GREEN_D)
    cv.text(gx + 0.33, yA - 0.17, "8-step action chunk", size=5.0, color=MUTED, va="top")

    cv.text(xo + 0.42, Y0 + 0.30, r"$\mathcal{L}=\|\hat v_z-v_z\|^2+\|\hat v_a-v_a\|^2$", size=6.0)
    cv.text(xo + 0.42, Y0 + 0.13, "rectified flow, one shared τ", size=5.0, color=MUTED)

    # attention mask (rows = queries, cols = keys)
    mx, my, ms_ = X0 + 0.04, Y0 + 0.02, 0.38
    n = [3, 3, 2]   # text, vision, action (schematic proportions)
    N, c = sum(n), ms_ / sum(n)
    edges = np.cumsum([0] + n)
    cols = [BLUE, VIS, GREEN]
    for i in range(N):
        gi = np.searchsorted(edges, i, side="right") - 1
        for j in range(N):
            on = (j <= i) if gi == 0 else True
            on = on and not (gi == 0 and j >= n[0])
            cv.ax.add_patch(Rectangle((mx + j * c, my + ms_ - (i + 1) * c), c, c, fc=cols[gi] if on else "white",
                                      ec="white", lw=0.3, zorder=5, alpha=0.85 if on else 1))
    cv.ax.add_patch(Rectangle((mx, my), ms_, ms_, fc="none", ec=MUTED, lw=0.5, zorder=6))
    for k, lab in enumerate("TVA"):
        cv.text(mx + (edges[k] + n[k] / 2) * c, my + ms_ + 0.04, lab, size=4.8, color=cols[k], va="bottom")
    cv.text(mx + ms_ + 0.07, my + ms_ + 0.02,
            "attention: text causal,\nvision + action see all.\nNo KV cache across chunks:\neach chunk re-reads $o_t$.",
            size=5.0, color=MUTED, ha="left", va="top", linespacing=1.2)


# ---------------------------------------------------------------- (c) deployment
def panel_c(cv, root):
    X0, Y0, PH = 0.06, 0.08, 1.66
    cv.panel_label(X0, Y0 + PH, "c", "Deployment: intent → shot profile → closed-loop motion")
    yc = Y0 + 0.92

    cv.box(X0, yc - 0.40, 1.14, 0.80, fc=ORANGE_L, ec=ORANGE, lw=0.7, r=0.05)
    cv.text(X0 + 0.07, yc + 0.29, "user intent", size=5.8, color=ORANGE, ha="left", weight="semibold")
    cv.text(X0 + 0.07, yc + 0.01, "“Shoot her from behind\non the left, full body,\nin the right third.”",
            size=5.6, ha="left", style="italic", linespacing=1.2)
    cv.text(X0 + 0.07, yc - 0.30, "or a reference photo", size=5.2, color=MUTED, ha="left")

    xl = X0 + 1.28
    cv.box(xl, yc - 0.26, 0.66, 0.52, fc="white", ec=ORANGE, lw=0.7, r=0.05)
    cv.text(xl + 0.33, yc + 0.08, "LLM / VLM", size=6.2, weight="semibold", color=ORANGE)
    cv.text(xl + 0.33, yc - 0.10, "off-the-shelf", size=5.0, color=MUTED)
    cv.arrow((X0 + 1.14, yc), (xl - 0.01, yc), color=ORANGE, lw=0.7, ms=4)

    xp = xl + 0.80
    cv.box(xp, yc - 0.52, 1.32, 1.04, fc=BLUE_L, ec=BLUE, lw=0.9, r=0.06)
    cv.text(xp + 0.07, yc + 0.40, r"shot profile $p$", size=6.6, color=BLUE_D, ha="left", weight="semibold")
    cv.text(xp + 0.07, yc + 0.17, "wide · from her back-left · eye level\nright third · full body · uncropped",
            size=4.9, color=BLUE_D, ha="left", linespacing=1.25)
    cv.text(xp + 0.07, yc - 0.23, "bearing 222°  elevation −1°\noccupancy 18%  body 100%\ncentre 895/351\nhalf-size 123/280",
            size=4.9, ha="left", family="IBM Plex Mono", linespacing=1.3)
    cv.arrow((xl + 0.66, yc), (xp - 0.01, yc), color=ORANGE, lw=0.7, ms=4)

    xw = xp + 1.46
    cv.box(xw, yc - 0.30, 0.62, 0.60, fc=GREEN_L, ec=GREEN, lw=0.8, r=0.06)
    cv.text(xw + 0.31, yc + 0.10, "world action\nmodel", size=5.9, weight="semibold", color=GREEN_D, linespacing=1.0)
    cv.text(xw + 0.31, yc - 0.17, "UniPC, 30 steps", size=4.6, color=MUTED)
    cv.arrow((xp + 1.32, yc), (xw - 0.01, yc), color=BLUE, lw=0.8, ms=4)

    F = root / "runs/gif/frames" / f"{KEY}_p{PAIR:02d}_s{S_IDX}_g{G_IDX:02d}" / "ours"
    frames = [(0, r"start $o_0$", "d 0.37"), (24, "after chunk 1", "0.23"),
              (48, "after chunk 2", "0.10"), (72, "chunk 3 · shoot", "0.035")]
    fw, fh, gap = 0.62, 0.465, 0.07
    xs0 = xw + 0.80
    yf = yc - fh / 2 + 0.06
    for k, (i, lab, dd) in enumerate(frames):
        x = xs0 + k * (fw + gap)
        last = k == len(frames) - 1
        cv.image(load(F / f"f{i:04d}.jpg", 500), x, yf, fw, fh, ec=GREEN if last else INK, lw=1.2 if last else 0.5)
        cv.text(x, yf + fh + 0.05, lab, size=5.5, ha="left", va="bottom", color=GREEN_D if last else INK,
                weight="semibold" if last else "normal")
        cv.text(x + fw, yf - 0.04, dd, size=4.9, ha="right", va="top", color=MUTED, family="IBM Plex Mono")
        if k:
            cv.arrow((x - gap + 0.008, yc + 0.06), (x - 0.008, yc + 0.06), color=GREEN, lw=0.6, ms=3)
    cv.arrow((xw + 0.62, yc + 0.06), (xs0 - 0.01, yc + 0.06), color=GREEN, lw=0.8, ms=4)
    cv.text(xw + 0.71, yc + 0.20, r"$a_{1:8}$", size=6.3, color=GREEN_D)

    # closed loop back into the model
    yb = yc - 0.60
    xr = xs0 + 2 * (fw + gap) + fw / 2
    cv.line([xr, xr], [yf - 0.15, yb], color=MUTED, lw=0.6)
    cv.line([xr, xw + 0.31], [yb, yb], color=MUTED, lw=0.6)
    cv.arrow((xw + 0.31, yb), (xw + 0.31, yc - 0.31), color=MUTED, lw=0.6, ms=3.5)
    cv.text((xr + xw + 0.31) / 2, yb - 0.05, "execute ≤ 8 steps, re-render $o_t$, repeat;  stop when shoot ≥ 0.5",
            size=5.3, color=MUTED, va="top")
    cv.text(xs0 + 4 * fw + 3 * gap, yb - 0.05, "goal image never\nshown to the policy",
            size=4.8, color=FAINT, ha="right", va="top", style="italic", linespacing=1.1)


def main():
    global AZ, EL, RV, DIST
    a = args()
    root = Path(a.root)
    if a.view:
        AZ, EL, RV, DIST = map(float, a.view.split(","))
    cv = Canvas()
    cv.line([3.54, 3.54], [2.05, 4.85], color=GRID, lw=0.6, z=0)
    cv.line([0.06, 7.10], [1.86, 1.86], color=GRID, lw=0.6, z=0)
    assets = panel_a(cv, root)
    panel_b(cv, assets)
    panel_c(cv, root)
    for ext in (("png",) if a.view else ("pdf", "svg", "png")):
        cv.fig.savefig(HERE / f"{a.out}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", HERE / f"{a.out}.png")


if __name__ == "__main__":
    main()
