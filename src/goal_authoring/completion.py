"""Completion — turn a PARTIAL goal into the FULL profile the policy was trained on.

The policy has only ever seen full prompts (`serialize.TRAINING_PROMPT_RE`): all eight keys, a
crop phrase and a `visible` number. Authoring produces partial profiles by design — "a close-up
from the front" says nothing about elevation, placement or crop — so something has to supply
the rest, and dropping the clauses is not an option: that is a prompt shape the policy has never
been shown. Two completers, same interface, deliberately different in what they lean on:

* `BankCompleter` — DATA-DRIVEN. Filters the goal bank (every goal the policy trained on) to the
  requested categories, ranks the survivors by distance to the requested numbers, and anchors on
  the medoid: a REAL training goal. Unspecified keys, and specified-by-category keys, take the
  anchor's values (so an NL-only request yields a prompt that is byte-identical to a training
  prompt); measured (`exact`) numbers are kept. A category with too little support is relaxed
  to nearest-by-value and reported.

* `RuleCompleter` — DATASET-INDEPENDENT. Models the subject as one upright box of aspect
  `PERSON_ASPECT` placed in the frame: occupancy fixes its size, placement / crop fix where it
  sits and where the frame cuts it, and every remaining key (visible centre and half-sizes,
  body_in_frame, crop fractions, visible fraction) follows from the geometry. Semantic keys
  that geometry cannot supply (bearing, elevation) get explicit defaults. Nothing here reads
  the bank, so it works without one and its assumptions are all in this file.

Both return a `Completed` carrying per-key provenance and warnings, so a consumer can tell a
requested value from a filled-in one — the distinction the prompt itself erases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH, SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.goal_bank import BAND_KEYS, NUMERIC_KEYS, GoalBank, band_of
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.serialize import FULL_PROFILE_KEYS, crop_phrase

# Full projected bbox of a standing person, height over width. Anthropometric (pedestrian
# boxes run ~2.4-2.5 tall-to-wide); the export's own uncropped goals have a median of 2.4 —
# checked as a sanity comparison, not fitted.
PERSON_ASPECT = 2.4

# What the rule completer assumes when a key is simply not mentioned. Explicit and boring on
# purpose: frame centre, a conventional full-figure size, seen from the front at eye level.
RULE_DEFAULTS: dict[str, float] = {
    "occupancy": 25.0,
    "object_center_x": RENDER_WIDTH / 2.0,
    "object_center_y": RENDER_HEIGHT / 2.0,
    SUBJECT_BEARING_KEY: 0.0,
    "cam_to_obj_elevation_deg": 0.0,
}

BOX_KEYS = ("object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset")
CROP_FRACTION_KEYS = ("top_cut_frac", "bot_cut_frac")

# `crop_from_bbox` reports a cut side it cannot size as exactly 1.0 ("cut, fraction unknown").
# Anything at or above this is a side-only statement, not a fraction.
_SIDE_ONLY = 1.0


@dataclass
class Completed:
    profile: GoalProfile                 # complete: every FULL_PROFILE_KEYS present
    source: dict[str, str]               # key -> given | default | rule | bank | bank(anchored)
    warnings: list[str] = field(default_factory=list)
    support: int | None = None           # bank: training goals matching the (relaxed) request
    anchor: int | None = None            # bank: row index of the training goal anchored on
    method: str = ""

    @property
    def prompt(self) -> str:
        return self.profile.to_prompt()


# --------------------------------------------------------------------------- #
# the person box (shared geometry)
# --------------------------------------------------------------------------- #
def _layout(w: float, h: float, cx: float, cy: float, top: float, bot: float,
            W: float, H: float) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Full box (half sizes w, h) placed by the cut request; returns (full, visible) xyxy."""
    if top > 0.0:
        y0 = -top * 2.0 * h
    elif bot > 0.0:
        y0 = H + bot * 2.0 * h - 2.0 * h
    else:
        y0 = cy - h
    y1 = y0 + 2.0 * h
    x0, x1 = cx - w, cx + w
    vis = (max(x0, 0.0), max(y0, 0.0), max(min(x1, W), max(x0, 0.0)), max(min(y1, H), max(y0, 0.0)))
    return (x0, y0, x1, y1), vis


def solve_person_box(occupancy: float, cx: float, cy: float, top_cut: float = 0.0, bot_cut: float = 0.0,
                     *, aspect: float = PERSON_ASPECT, keep_crop: bool = False,
                     W: float = RENDER_WIDTH, H: float = RENDER_HEIGHT) -> tuple[dict[str, float], list[str]]:
    """Place one upright box so its CLIPPED area is `occupancy` % of the frame.

    `occupancy` is defined on the clipped box (`src/scoring/bbox_control.py`), so the box is
    sized for its visible area, not its full one — otherwise a "medium shot" whose subject
    overflows the frame would come out a size smaller than the number it carries. Vertical
    placement follows the cut request (`top_cut` / `bot_cut` as fractions of the full height,
    the same quantities `annotations._apply_crop_extent` measures); an uncut box sits at
    (cx, cy) and is nudged inside the frame if it would otherwise poke out.

    When the requested size cannot fit at this aspect with the requested cut, one of two
    things gives. With `keep_crop=False` (the cut was a default, not a request) the other end
    is cut too — what a photographer does with a big subject. With `keep_crop=True` the crop
    WORD is honoured and the aspect gives instead: the box is widened until the requested cut
    fills the frame height, which is how the data contains "medium shot, cropped at the legs"
    (the projected bbox of a person runs from 1.8 to 2.9 tall-to-wide). Either way the change
    is reported. Horizontal overflow is allowed (a fifth of training goals are x-clipped) and
    reduces body_in_frame, as it does in the data.

    Returns the derived keys plus `full_box` / `visible_box`, and a list of adjustments made.
    """
    notes: list[str] = []
    occ = min(max(float(occupancy), 0.1), 100.0)
    target = occ / 100.0 * W * H
    top, bot = max(0.0, float(top_cut)), max(0.0, float(bot_cut))
    if top + bot > 0.9:
        s = 0.9 / (top + bot)
        top, bot = top * s, bot * s
        notes.append(f"crop fractions summed past 0.9; scaled to top {top:.2f} / bottom {bot:.2f}")

    h_fixed: float | None = None
    if top > 0.0 and bot > 0.0:
        h_fixed = H / (2.0 * (1.0 - top - bot))
    else:
        # Would the aspect box overflow the frame vertically once the requested cut is applied?
        # (x unclipped estimate; the exact solve below handles clipping.)
        w0 = math.sqrt(target / (4.0 * aspect * (1.0 - top - bot)))
        if 2.0 * aspect * w0 * (1.0 - top - bot) > H + 1e-9:
            if keep_crop:
                # honour the word: the visible height becomes the frame height, width does the rest
                h_fixed = H / (2.0 * (1.0 - top - bot))
                w_eff = target / (2.0 * H)
                notes.append(f"occupancy {occ:.0f}% with this crop does not fit at aspect {aspect}; "
                             f"box widened to aspect {h_fixed / max(w_eff, 1e-9):.2f} to keep the crop")
            else:
                w_needed = min(target / (2.0 * H), W / 2.0)     # visible height is H, so width does the work
                h_needed = aspect * w_needed
                outside = max(0.0, 1.0 - H / (2.0 * h_needed))
                if top > 0.0:
                    bot = max(0.0, outside - top)
                elif bot > 0.0:
                    top = max(0.0, outside - bot)
                else:
                    top = bot = outside / 2.0
                notes.append(f"occupancy {occ:.0f}% does not fit the frame uncut at aspect {aspect}; "
                             f"cut top {top:.2f} / bottom {bot:.2f} of the subject")
                if top > 0.0 and bot > 0.0:
                    h_fixed = H / (2.0 * (1.0 - top - bot))

    hf = (lambda w: h_fixed) if h_fixed is not None else (lambda w: aspect * w)  # noqa: E731

    def area(w: float, cxx: float, cyy: float) -> float:
        _, v = _layout(w, hf(w), cxx, cyy, top, bot, W, H)
        return (v[2] - v[0]) * (v[3] - v[1])

    def solve_w(cxx: float, cyy: float) -> float:
        lo, hi = 0.5, 4.0 * W
        if area(hi, cxx, cyy) < target - 1e-6:
            notes.append(f"occupancy {occ:.0f}% is not reachable with this placement; capped")
            return hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if area(mid, cxx, cyy) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if top == 0.0 and bot == 0.0 and h_fixed is None:
        # Uncut means uncut. The box that reaches the occupancy while fully inside the frame
        # is known in closed form (no clipping), so place THAT box and keep it inside: a
        # shifted centre is not a crop. (The overflow case was turned into a cut above.)
        w = math.sqrt(target / (4.0 * aspect))
        h = aspect * w
        ncx = min(max(cx, w), W - w) if 2.0 * w <= W else cx
        ncy = min(max(cy, h), H - h) if 2.0 * h <= H else cy
        if (ncx, ncy) != (cx, cy):
            notes.append(f"placement moved from ({cx:.0f},{cy:.0f}) to ({ncx:.0f},{ncy:.0f}) px so the "
                         "uncut subject stays inside the frame")
            cx, cy = ncx, ncy
    else:
        w = solve_w(cx, cy)
        h = hf(w)

    full, vis = _layout(w, h, cx, cy, top, bot, W, H)
    vis_w, vis_h = vis[2] - vis[0], vis[3] - vis[1]
    span = full[3] - full[1]
    out = {
        "occupancy": 100.0 * vis_w * vis_h / (W * H),
        "object_center_x": 0.5 * (vis[0] + vis[2]),
        "object_center_y": 0.5 * (vis[1] + vis[3]),
        "bbox_x_offset": 0.5 * vis_w,
        "bbox_y_offset": 0.5 * vis_h,
        "body_in_frame_ratio": 100.0 * vis_w * vis_h / max(4.0 * w * h, 1e-9),
        "top_cut_frac": max(0.0, -full[1]) / span,
        "bot_cut_frac": max(0.0, full[3] - H) / span,
        "visible_frac": vis_h / span,
        "head_in_frame": 1.0 if full[1] >= -1e-9 else 0.0,
        "full_box": full,
        "visible_box": vis,
    }
    return out, notes


def extent_from_visible_box(cx: float, cy: float, hx: float, hy: float, top: float, bot: float,
                            *, aspect: float = PERSON_ASPECT, W: float = RENDER_WIDTH) -> dict[str, float]:
    """The image-side inverse: a MEASURED visible box plus cut fractions -> the full box's keys.

    The full height follows from the visible height and the fractions; the full width is the
    visible width unless the box touches a side of the frame, in which case the aspect supplies
    it. `body_in_frame_ratio` is the clipped-over-full area ratio the scorer defines.
    """
    vis_w, vis_h = 2.0 * hx, 2.0 * hy
    keep = max(1.0 - top - bot, 0.05)
    full_h = vis_h / keep
    touches_x = (cx - hx) <= 1.0 or (cx + hx) >= W - 1.0
    full_w = max(vis_w, full_h / aspect) if touches_x else vis_w
    return {
        "body_in_frame_ratio": 100.0 * (vis_w * vis_h) / max(full_w * full_h, 1e-9),
        "top_cut_frac": top, "bot_cut_frac": bot, "visible_frac": keep,
        "head_in_frame": 0.0 if top > 0.02 else 1.0,
    }


def _crop_request(vals: Mapping[str, float], exact: frozenset[str]) -> tuple[float, float, str, str | None]:
    """What the request says about the cut: (top, bot, source, note).

    Priority: explicit fractions (measured or category) > a side-only statement (1.0 marker ->
    the representative fraction of the milder word for that side) > `visible_frac` alone >
    `body_in_frame_ratio` (an area ratio; assumed to be a bottom cut, the photographic
    default) > uncut.
    """
    if "top_cut_frac" in vals or "bot_cut_frac" in vals:
        t, b = float(vals.get("top_cut_frac", 0.0)), float(vals.get("bot_cut_frac", 0.0))
        if t >= _SIDE_ONLY or b >= _SIDE_ONLY:
            label = crop_phrase(t, b)
            if label == "cropped below the waist":
                label = "cropped at the legs"            # side known, depth not: assume the milder cut
            t, b = vocab.CROP_SIDE[label]
            return t, b, "rule", f"crop side given without a fraction; assumed '{label}'"
        src = "given" if ({"top_cut_frac", "bot_cut_frac"} & exact) else "given"
        return t, b, src, None
    if "visible_frac" in vals:
        return 0.0, max(0.0, 1.0 - float(vals["visible_frac"])), "rule", "visible fraction given; assumed cut at the bottom"
    if "body_in_frame_ratio" in vals:
        bif = float(vals["body_in_frame_ratio"])
        if vocab._classify(bif, vocab.BODY_FRAMING) == "full body in frame":
            return 0.0, 0.0, "rule", None
        return 0.0, max(0.0, 1.0 - bif / 100.0), "rule", "body framing given; assumed cut at the bottom"
    return 0.0, 0.0, "default", None


class RuleCompleter:
    """Dataset-independent completion (see the module docstring)."""

    method = "rules"

    def __init__(self, *, aspect: float = PERSON_ASPECT, defaults: Mapping[str, float] | None = None):
        self.aspect = float(aspect)
        self.defaults = dict(RULE_DEFAULTS)
        if defaults:
            self.defaults.update(defaults)

    def complete(self, profile: GoalProfile) -> Completed:
        vals, exact = dict(profile.values), profile.exact
        out: dict[str, float] = {}
        src: dict[str, str] = {}
        warns: list[str] = []

        for k in (SUBJECT_BEARING_KEY, "cam_to_obj_elevation_deg"):
            if k in vals:
                out[k], src[k] = float(vals[k]), "given"
            else:
                out[k], src[k] = self.defaults[k], "default"

        top, bot, crop_src, note = _crop_request(vals, exact)
        if note:
            warns.append(note)

        box_known = all(k in exact for k in BOX_KEYS) and "occupancy" in exact
        if box_known:
            geom = extent_from_visible_box(vals["object_center_x"], vals["object_center_y"],
                                           vals["bbox_x_offset"], vals["bbox_y_offset"], top, bot,
                                           aspect=self.aspect)
            for k in ("occupancy", *BOX_KEYS):
                out[k], src[k] = float(vals[k]), "given"
            out["body_in_frame_ratio"] = (float(vals["body_in_frame_ratio"]) if "body_in_frame_ratio" in vals
                                          else geom["body_in_frame_ratio"])
            src["body_in_frame_ratio"] = "given" if "body_in_frame_ratio" in vals else "rule"
            for k in ("top_cut_frac", "bot_cut_frac", "visible_frac"):
                out[k] = geom[k]
                src[k] = "given" if (k in vals and crop_src == "given") else "rule"
        else:
            occ = float(vals.get("occupancy", self.defaults["occupancy"]))
            cx = float(vals.get("object_center_x", self.defaults["object_center_x"]))
            cy = float(vals.get("object_center_y", self.defaults["object_center_y"]))
            geom, notes = solve_person_box(occ, cx, cy, top, bot, aspect=self.aspect,
                                           keep_crop=(crop_src == "given"))
            warns.extend(notes)
            for k in ("occupancy", "object_center_x", "object_center_y"):
                out[k] = geom[k]
                src[k] = "given" if k in vals else "default"
            for k in ("bbox_x_offset", "bbox_y_offset", "body_in_frame_ratio"):
                out[k], src[k] = geom[k], "rule"
            for k in ("top_cut_frac", "bot_cut_frac", "visible_frac"):
                out[k], src[k] = geom[k], (crop_src if crop_src != "default" else "default")
            if "body_in_frame_ratio" in vals and abs(out["body_in_frame_ratio"] - float(vals["body_in_frame_ratio"])) > 5.0:
                warns.append(f"body_in_frame {float(vals['body_in_frame_ratio']):.0f}% requested, geometry gives "
                             f"{out['body_in_frame_ratio']:.0f}% (x-clipping or a conflicting crop)")
        out["head_in_frame"] = 1.0 if out["top_cut_frac"] <= 0.02 else 0.0
        keys = frozenset(out)
        return Completed(GoalProfile(out, keys, exact & keys), src, warns, None, None, self.method)


# --------------------------------------------------------------------------- #
# bank completer
# --------------------------------------------------------------------------- #
class BankCompleter:
    """Data-driven completion anchored on a real training goal (see the module docstring)."""

    method = "bank"
    # Dropped first when a request has too little support: the axes the policy is weakest on
    # and a user is least likely to insist on, down to the two that define the shot.
    RELAX_ORDER = ("body_in_frame_ratio", "object_center_y", "object_center_x", "crop",
                   "cam_to_obj_elevation_deg", "occupancy", SUBJECT_BEARING_KEY)

    def __init__(self, bank: GoalBank | None = None, *, min_support: int = 30, k: int = 25,
                 train_only: bool = True):
        self.bank = bank if bank is not None else GoalBank.load()
        self.min_support = int(min_support)
        self.k = int(k)
        self.train_only = bool(train_only)

    def _constraints(self, vals: Mapping[str, float], exact: frozenset[str]) -> dict[str, tuple[np.ndarray, str]]:
        b = self.bank
        cons: dict[str, tuple[np.ndarray, str]] = {}
        for k in BAND_KEYS:
            if k in vals:
                cons[k] = (b.band_mask(k, float(vals[k])), str(band_of(k, float(vals[k]))))
        if "top_cut_frac" in vals or "bot_cut_frac" in vals:
            t, bt = float(vals.get("top_cut_frac", 0.0)), float(vals.get("bot_cut_frac", 0.0))
            if t >= _SIDE_ONLY or bt >= _SIDE_ONLY:
                # side known, depth unknown: both words for that side qualify, the anchor decides
                labels = ({"cropped at the legs", "cropped below the waist"} if (bt > 0 and t < _SIDE_ONLY)
                          else {crop_phrase(t, bt)})
                mask = np.zeros(len(b), dtype=bool)
                for lab in labels:
                    mask |= b.crop_mask(lab)
                cons["crop"] = (mask, " | ".join(sorted(labels)))
            else:
                lab = crop_phrase(t, bt)
                cons["crop"] = (b.crop_mask(lab), lab)
        return cons

    def _anchor(self, near: np.ndarray, d_req: np.ndarray) -> int:
        """Among the k nearest rows: the one that is close to the REQUEST and typical of its
        neighbours. A pure medoid drifts off the requested value when the neighbourhood is
        wide (it optimises the unspecified keys); the pure nearest row can be an oddball on
        them. Both distances are normalized RMS, so their sum is a fair trade."""
        if len(near) == 1:
            return int(near[0])
        b = self.bank
        typ = np.zeros(len(near), dtype=np.float64)
        for i, r in enumerate(near):
            q = {k: float(b.values[k][r]) for k in NUMERIC_KEYS}
            typ[i] = b.distances(q, NUMERIC_KEYS, near).sum() / (len(near) - 1)
        return int(near[int(np.argmin(d_req + typ))])

    def complete(self, profile: GoalProfile) -> Completed:
        b = self.bank
        vals, exact = dict(profile.values), profile.exact
        base = b.train_mask() if self.train_only else np.ones(len(b), dtype=bool)
        cons = self._constraints(vals, exact)
        active = dict(cons)
        warns: list[str] = []

        def combined() -> np.ndarray:
            m = base.copy()
            for mk, _ in active.values():
                m &= mk
            return m

        mask = combined()
        relaxed: dict[str, tuple[str, int]] = {}          # key -> (requested word, support before)
        for k in self.RELAX_ORDER:
            if int(mask.sum()) >= self.min_support:
                break
            if k in active:
                n_before = int(mask.sum())
                _, desc = active.pop(k)
                relaxed[k] = (desc, n_before)
                mask = combined()
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise RuntimeError("goal bank has no rows for the split requested")

        query = {k: float(vals[k]) for k in NUMERIC_KEYS if k in vals}
        d = b.distances(query, list(query), idx) if query else np.zeros(len(idx))
        order = np.argsort(d, kind="stable")[: min(self.k, len(idx))]
        near = idx[order]
        anchor = self._anchor(near, d[order])
        row = b.row(anchor)
        # One line per relaxed constraint. A key given EXACTLY keeps its number whatever the
        # anchor says, so for it the relaxation only changes where the OTHER keys come from.
        for k, (desc, n_before) in relaxed.items():
            head = (f"{k}: '{desc}' with the other constraints matches only {n_before} training "
                    f"goals (< {self.min_support})")
            is_exact = (k == "crop" and {"top_cut_frac", "bot_cut_frac"} & exact) or (k in exact)
            if is_exact:
                warns.append(head + "; kept as given, the rest anchored on the nearest goals")
            else:
                got = (crop_phrase(row["top_cut_frac"], row["bot_cut_frac"]) if k == "crop"
                       else band_of(k, float(row[k])))
                warns.append(head + f"; the anchored training goal has '{got}'")

        out: dict[str, float] = {}
        src: dict[str, str] = {}
        for k in FULL_PROFILE_KEYS:
            if k in exact and k in vals:
                out[k], src[k] = float(vals[k]), "given"
            else:
                out[k] = float(row[k])
                src[k] = "bank(anchored)" if k in vals else "bank"
        # A side-only crop statement (1.0 marker) is never a fraction: the anchor's word wins.
        for k in CROP_FRACTION_KEYS:
            if k in vals and float(vals[k]) >= _SIDE_ONLY:
                out[k], src[k] = float(row[k]), "bank(anchored)"
        if crop_phrase(out["top_cut_frac"], out["bot_cut_frac"]) != crop_phrase(row["top_cut_frac"], row["bot_cut_frac"]) \
                and "visible_frac" not in exact:
            out["visible_frac"], src["visible_frac"] = float(row["visible_frac"]), "bank"
        out["head_in_frame"] = 1.0 if out["top_cut_frac"] <= 0.02 else 0.0

        for k, (mk, desc) in active.items():
            if not mk[anchor]:      # cannot happen by construction; keep the invariant loud
                raise AssertionError(f"anchor violates active constraint {k}={desc!r}")
        keys = frozenset(out)
        return Completed(GoalProfile(out, keys, exact & keys), src, warns, int(mask.sum()),
                         int(anchor), self.method)


def make_completer(method: str = "bank", **kwargs):
    if method == "bank":
        return BankCompleter(**kwargs)
    if method == "rules":
        return RuleCompleter(**kwargs)
    raise ValueError(f"unknown completion method {method!r} (bank | rules)")


__all__ = ["Completed", "RuleCompleter", "BankCompleter", "make_completer", "solve_person_box",
           "extent_from_visible_box", "PERSON_ASPECT", "RULE_DEFAULTS"]
