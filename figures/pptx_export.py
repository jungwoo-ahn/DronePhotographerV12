"""Draw a matplotlib Figure onto a PowerPoint slide as native, editable objects.

A renderer that subclasses RendererAgg (for text metrics) but, instead of rasterising, writes every draw call
as a slide object:
  paths (patches, lines, arrows, markers, collections) -> freeform shapes (custGeom) with their fill, alpha,
                                                          stroke width, dashes and hatch
  text                                                   -> text boxes; mathtext ($o_s$, \\Delta t, ...) becomes
                                                          runs with italics and sub/superscripts
  images                                                 -> pictures (JPEG when opaque, PNG otherwise)
Rectangular and polygonal clip paths are applied to the geometry (shapely), since slide shapes do not clip.

    from pptx_export import add_figure
    add_figure(slide, fig, x=0.4, y=0.9, w=12.5, h=6.4)   # inches; keeps the figure's aspect ratio
"""
import io
import re

import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.path import Path as MPath
from PIL import Image
from pptx.oxml import parse_xml
from pptx.util import Emu
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box as sbox

EMU_IN = 914400
NS = ('xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"')
SANS, MONO, MATH = "Arial", "Courier New", "Times New Roman"   # fonts every Office install has


def _hex(rgb):
    return "%02X%02X%02X" % tuple(int(round(max(0, min(1, c)) * 255)) for c in rgb[:3])


def _clr(rgb, alpha=1.0):
    a = "" if alpha >= 0.999 else f'<a:alpha val="{int(round(alpha * 100000))}"/>'
    return f'<a:srgbClr val="{_hex(rgb)}">{a}</a:srgbClr>'


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- mathtext -> runs
_SYM = {"Delta": "Δ", "theta": "θ", "tau": "τ", "epsilon": "ε", "sim": "~", "mapsto": "↦", "to": "→",
        "rightarrow": "→", "Vert": "‖", "|": "‖", "leq": "≤", "geq": "≥", "in": "∈", "top": "⊤", "cdot": "·",
        "quad": "  ", "qquad": "    ", ",": " ", ";": " ", " ": " ", "!": "", "times": "×", "pm": "±",
        "alpha": "α", "beta": "β", "gamma": "γ", "lambda": "λ", "mu": "μ", "sigma": "σ", "pi": "π", "phi": "φ",
        "left": "", "right": "", "ldots": "…", "dots": "…", "star": "⋆", "circ": "∘", "Vert ": "‖"}
_CAL = {"E": "ℰ", "D": "𝒟", "N": "𝒩", "L": "ℒ", "P": "𝒫"}
_BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ"}


def _group(s, i):
    """Return (content, next index) of a {...} group or a single token at s[i]."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        return s[i + 1:j - 1], j
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _math_runs(s, base=0, upright=False):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in "_^":
            g, i = _group(s, i + 1)
            out += _math_runs(g, base=(-25000 if c == "_" else 30000), upright=upright)
            continue
        if c == "{":
            g, i = _group(s, i)
            out += _math_runs(g, base, upright)
            continue
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            cmd, i = m.group(1), i + len(m.group(0))
            if cmd in ("mathrm", "text", "mathbf", "operatorname"):
                g, i = _group(s, i)
                out += _math_runs(g, base, upright=True)
            elif cmd == "mathcal":
                g, i = _group(s, i)
                out.append((_CAL.get(g, g), base, False))
            elif cmd == "mathbb":
                g, i = _group(s, i)
                out.append((_BB.get(g, g), base, False))
            elif cmd == "hat":
                g, i = _group(s, i)
                out.append((g + "̂", base, not upright))
            elif cmd == "sqrt":
                g, i = _group(s, i)
                out.append(("√", base, False))
                out += _math_runs(g, base, upright)
            elif cmd == "frac":
                a, i = _group(s, i)
                b, i = _group(s, i)
                out += _math_runs(a, base, upright) + [("/", base, False)] + _math_runs(b, base, upright)
            else:
                out.append((_SYM.get(cmd, cmd), base, False))
            continue
        out.append((c, base, c.isalpha() and not upright))
        i += 1
    return out


def text_runs(s, ismath):
    """[(text, baseline, italic, is_math)] for a matplotlib string (math segments between $...$)."""
    if not ismath:
        return [(s, 0, False, False)]
    parts, runs = s.split("$"), []
    for k, p in enumerate(parts):
        if not p:
            continue
        if k % 2 == 0:
            runs.append((p, 0, False, False))
        else:
            for t, b, it in _math_runs(p):
                if runs and runs[-1][1:] == (b, it, True):
                    runs[-1] = (runs[-1][0] + t, b, it, True)
                else:
                    runs.append((t, b, it, True))
    return runs


# ---------------------------------------------------------------- renderer
class SlideRenderer(RendererAgg):
    """Draw calls in display pixels (origin bottom-left) -> slide objects in EMU."""

    def __init__(self, fig, slide, x, y, scale):
        super().__init__(*fig.canvas.get_width_height(physical=True), fig.dpi)
        # RendererAgg binds these to its C++ renderer as instance attributes; drop them so ours are used
        for name in ("draw_image", "draw_markers", "draw_path_collection", "draw_quad_mesh",
                     "draw_gouraud_triangles"):
            self.__dict__.pop(name, None)
        self.slide, self.ox, self.oy, self.k = slide, x, y, scale   # inches, inches, slide-in per fig-in
        self.Hpx = self.height
        self.dpi_ = fig.dpi
        self.spTree = slide.shapes._spTree
        self.nid = max([int(e.get("id")) for e in self.spTree.iter() if e.tag.endswith("}cNvPr")] + [1]) + 1
        self.n = {"shape": 0, "text": 0, "image": 0}

    # route the C++ fast paths back through draw_path
    draw_markers = RendererBase.draw_markers
    draw_path_collection = RendererBase.draw_path_collection
    draw_quad_mesh = RendererBase.draw_quad_mesh

    def option_image_nocomposite(self):
        return True

    def option_scale_image(self):
        return False

    # ---- coordinates
    def emu(self, px, py):
        """display px (y up) -> slide EMU (y down)"""
        xi = self.ox + px / self.dpi_ * self.k
        yi = self.oy + (self.Hpx - py) / self.dpi_ * self.k
        return xi * EMU_IN, yi * EMU_IN

    def _id(self):
        self.nid += 1
        return self.nid

    def _clip_geom(self, gc):
        g = None
        r = gc.get_clip_rectangle()
        if r is not None:
            g = sbox(r.x0, r.y0, r.x1, r.y1)
        cp, aff = gc.get_clip_path()
        if cp is not None:
            polys = [Polygon(p) for p in cp.to_polygons(aff) if len(p) >= 3]
            if polys:
                pg = polys[0].buffer(0)
                for p in polys[1:]:
                    pg = pg.union(p.buffer(0))
                g = pg if g is None else g.intersection(pg)
        return g

    # ---- paths
    def draw_path(self, gc, path, transform, rgbFace=None):
        segs = list(path.iter_segments(transform, remove_nans=True, simplify=False, curves=True))
        if not segs:
            return
        alpha = gc.get_alpha() if gc.get_forced_alpha() else 1.0
        fill = None
        if rgbFace is not None:
            fa = (rgbFace[3] if len(rgbFace) > 3 else 1.0) * (gc.get_alpha() if gc.get_forced_alpha() else 1.0)
            if fa > 0.003:
                fill = (rgbFace, fa)
        lw = gc.get_linewidth()
        sc = gc.get_rgb()
        sa = (sc[3] if len(sc) > 3 else 1.0) * alpha
        stroke = (sc, sa, lw) if lw > 0 and sa > 0.003 else None
        hatch = gc.get_hatch()
        if fill is None and stroke is None and not hatch:
            return

        clip = self._clip_geom(gc)
        pts = np.concatenate([v.reshape(-1, 2) for v, _ in segs])
        if clip is not None:
            bb = sbox(*pts.min(0), *pts.max(0))
            if not clip.intersects(bb) and bb.area > 0:
                return
            if not clip.buffer(1e-6).contains(bb):
                self._draw_clipped(gc, path, transform, fill, stroke, hatch, clip)
                return
        self._emit(segs, fill, stroke, hatch, gc)

    def _draw_clipped(self, gc, path, transform, fill, stroke, hatch, clip):
        polys = path.to_polygons(transform, closed_only=False)
        for p in polys:
            if len(p) < 2:
                continue
            closed = fill is not None or hatch
            if closed and len(p) >= 3:
                g = Polygon(p).buffer(0).intersection(clip)
                parts = [g] if isinstance(g, Polygon) else [q for q in getattr(g, "geoms", []) if isinstance(q, Polygon)]
                for q in parts:
                    if q.is_empty:
                        continue
                    c = np.asarray(q.exterior.coords)
                    segs = [(c[0], MPath.MOVETO)] + [(v, MPath.LINETO) for v in c[1:]] + [(c[0], MPath.CLOSEPOLY)]
                    self._emit(segs, fill, stroke, hatch, gc)
            else:
                g = LineString(p).intersection(clip)
                lines = [g] if isinstance(g, LineString) else [q for q in getattr(g, "geoms", []) if isinstance(q, LineString)]
                for q in lines:
                    c = np.asarray(q.coords)
                    if len(c) < 2:
                        continue
                    segs = [(c[0], MPath.MOVETO)] + [(v, MPath.LINETO) for v in c[1:]]
                    self._emit(segs, None, stroke, None, gc)

    def _emit(self, segs, fill, stroke, hatch, gc):
        allpts = []
        for v, code in segs:
            v = np.asarray(v, float).reshape(-1, 2)
            allpts.append(np.array([self.emu(*q) for q in v]))
        P = np.concatenate(allpts)
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        w, h = max(int(x1 - x0), 1), max(int(y1 - y0), 1)
        cmds = []
        for (v, code), E in zip(segs, allpts):
            pts = "".join(f'<a:pt x="{int(px - x0)}" y="{int(py - y0)}"/>' for px, py in E)
            if code == MPath.MOVETO:
                cmds.append(f"<a:moveTo>{pts}</a:moveTo>")
            elif code == MPath.LINETO:
                cmds.append(f"<a:lnTo>{pts}</a:lnTo>")
            elif code == MPath.CURVE3:
                cmds.append(f"<a:quadBezTo>{pts}</a:quadBezTo>")
            elif code == MPath.CURVE4:
                cmds.append(f"<a:cubicBezTo>{pts}</a:cubicBezTo>")
            elif code == MPath.CLOSEPOLY:
                cmds.append("<a:close/>")
        has_fill = fill is not None or bool(hatch)
        path_attr = "" if has_fill else ' fill="none"'
        if hatch:
            hc = gc.get_hatch_color()
            prst = {"/": "wdUpDiag", "\\": "wdDnDiag", "|": "ltVert", "-": "ltHorz", "+": "smGrid", "x": "smCheck",
                    ".": "pct20"}.get(hatch.strip()[0], "wdUpDiag")
            bg = _clr(fill[0], fill[1]) if fill else _clr((1, 1, 1), 0.0)
            fill_xml = f'<a:pattFill prst="{prst}"><a:fgClr>{_clr(hc)}</a:fgClr><a:bgClr>{bg}</a:bgClr></a:pattFill>'
        elif fill:
            fill_xml = f"<a:solidFill>{_clr(*fill)}</a:solidFill>"
        else:
            fill_xml = "<a:noFill/>"
        if stroke:
            sc, sa, lw = stroke
            wemu = int(lw * self.k * 12700)
            cap = {"round": "rnd", "projecting": "sq", "butt": "flat"}.get(gc.get_capstyle(), "flat")
            join = {"round": "<a:round/>", "bevel": "<a:bevel/>"}.get(gc.get_joinstyle(), '<a:miter lim="800000"/>')
            dash = ""
            off, seq = gc.get_dashes()
            if seq is not None and len(seq) >= 2 and lw > 0:
                ds = "".join(f'<a:ds d="{int(seq[i] / lw * 100000)}" sp="{int(seq[i + 1] / lw * 100000)}"/>'
                             for i in range(0, len(seq) - 1, 2))
                dash = f"<a:custDash>{ds}</a:custDash>"
            ln = f'<a:ln w="{wemu}" cap="{cap}"><a:solidFill>{_clr(sc, sa)}</a:solidFill>{dash}{join}</a:ln>'
        else:
            ln = "<a:ln><a:noFill/></a:ln>"
        sid = self._id()
        self.n["shape"] += 1
        xml = (f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
               f'<p:spPr><a:xfrm><a:off x="{int(x0)}" y="{int(y0)}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
               f'<a:custGeom><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/><a:rect l="0" t="0" r="r" b="b"/>'
               f'<a:pathLst><a:path w="{w}" h="{h}"{path_attr}>{"".join(cmds)}</a:path></a:pathLst></a:custGeom>'
               f'{fill_xml}{ln}</p:spPr></p:sp>')
        self.spTree.append(parse_xml(xml))

    # ---- images
    def draw_image(self, gc, x, y, im, transform=None):
        arr = np.asarray(im)[::-1]                  # rows arrive bottom-first
        h, w = arr.shape[:2]
        x0, x1, y0, y1 = x, x + w, y, y + h
        clip = self._clip_geom(gc)
        if clip is not None:
            cx0, cy0, cx1, cy1 = clip.bounds
            ix0, ix1 = int(max(0, cx0 - x)), int(min(w, cx1 - x))
            iy0, iy1 = int(max(0, cy0 - y)), int(min(h, cy1 - y))
            if ix1 <= ix0 or iy1 <= iy0:
                return
            arr = arr[h - iy1:h - iy0, ix0:ix1]       # rows run top -> bottom
            x0, x1, y0, y1 = x + ix0, x + ix1, y + iy0, y + iy1
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        if arr.shape[2] == 4 and arr[..., 3].min() < 255:
            img.save(buf, "PNG", optimize=True)
        else:
            img.convert("RGB").save(buf, "JPEG", quality=92)
        buf.seek(0)
        L, T = self.emu(x0, y1)
        R, B = self.emu(x1, y0)
        self.slide.shapes.add_picture(buf, Emu(int(L)), Emu(int(T)), Emu(int(R - L)), Emu(int(B - T)))
        self.n["image"] += 1

    # ---- text
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if not s.strip():
            return
        rgb = gc.get_rgb()
        alpha = (rgb[3] if len(rgb) > 3 else 1.0) * (gc.get_alpha() if gc.get_forced_alpha() else 1.0)
        size = prop.get_size_in_points() * self.k
        wpx, hpx, dpx = self.get_text_width_height_descent(s, prop, ismath)
        ha = mtext.get_horizontalalignment() if mtext is not None else "left"
        fam = prop.get_name()
        font = MONO if "Mono" in fam else SANS
        weight = prop.get_weight()
        bold = (weight if isinstance(weight, (int, float)) else
                {"semibold": 600, "demibold": 600, "bold": 700, "heavy": 800, "black": 900}.get(weight, 400)) >= 600
        italic = prop.get_style() == "italic"
        # box: generous width so a font with other metrics does not wrap; anchored by the text's own alignment
        wi = wpx / self.dpi_ * self.k
        bw = wi * 1.35 + 0.1
        bx_px = {"left": x, "center": x + wpx / 2, "right": x + wpx}.get(ha, x)
        L, Tb = self.emu(bx_px, self.Hpx - y)      # Text.draw has already flipped y to top-down
        left = {"left": L, "center": L - bw * EMU_IN / 2, "right": L - bw * EMU_IN}.get(ha, L)
        asc = 0.905 * size / 72 * EMU_IN                       # first baseline sits one ascent below the top
        top = Tb - asc
        hgt = 1.25 * size / 72 * EMU_IN
        algn = {"left": "l", "center": "ctr", "right": "r"}.get(ha, "l")
        rot = ""
        if abs(angle) > 0.1:
            rot = f' rot="{int(-angle * 60000)}"'
        runs = []
        for t, base, it, is_math in text_runs(s, ismath):
            attrs = f'lang="en-US" sz="{max(100, int(round(size * 100)))}"'
            if bold:
                attrs += ' b="1"'
            if italic or it:
                attrs += ' i="1"'
            if base:
                attrs += f' baseline="{base}"'
            face = MATH if is_math else font
            runs.append(f'<a:r><a:rPr {attrs} dirty="0"><a:solidFill>{_clr(rgb, alpha)}</a:solidFill>'
                        f'<a:latin typeface="{face}"/><a:cs typeface="{face}"/><a:sym typeface="Cambria Math"/>'
                        f'</a:rPr><a:t>{_esc(t)}</a:t></a:r>')
        sid = self._id()
        self.n["text"] += 1
        xml = (f'<p:sp {NS}><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/>'
               f'</p:nvSpPr><p:spPr><a:xfrm{rot}><a:off x="{int(left)}" y="{int(top)}"/>'
               f'<a:ext cx="{int(bw * EMU_IN)}" cy="{int(hgt)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
               f'</a:prstGeom><a:noFill/></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" '
               f'bIns="0" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/><a:p><a:pPr algn="{algn}"/>'
               f'{"".join(runs)}</a:p></p:txBody></p:sp>')
        self.spTree.append(parse_xml(xml))


def add_figure(slide, fig, x, y, w, h, dpi=300):
    """Draw `fig` into the slide box (inches), centred, keeping its aspect ratio. Returns counts of objects."""
    fig.set_dpi(dpi)
    fw, fh = fig.get_size_inches()
    k = min(w / fw, h / fh)
    ox, oy = x + (w - fw * k) / 2, y + (h - fh * k) / 2
    r = SlideRenderer(fig, slide, ox, oy, k)
    fig.draw(r)
    return r.n
