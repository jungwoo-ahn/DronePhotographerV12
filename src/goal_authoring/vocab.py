"""Cinematography vocabulary — the SINGLE SOURCE OF TRUTH mapping the geometric shot-profile
to/from a controlled category set, and to natural language.

Both goal-authoring front-ends share these tables:
  - Module 1 (natural language  -> goal profile): an LLM classifies user text into these categories,
    then `categories_to_profile` grounds them to numeric profile values.
  - Module 2 (reference image    -> goal profile): detection gives the geometric keys exactly; the
    semantic keys (bearing/elevation) are classified into these same categories.
  - The goal->prompt serializer (`profile_to_nl`) that conditions Cosmos is the inverse of the above,
    so profile <-> categories <-> NL round-trips consistently.

A category table is {label: (lo, hi, centroid)} in the profile key's RAW units. `_classify` maps a
value to its band label; the centroid maps a label back to a representative value. The bearing axis
reuses `facing.sector8` (subject-relative view) so there is one definition of front/side/back.
"""
from __future__ import annotations

from typing import Mapping

from src.common.facing import SECTOR8, sector3, sector8
from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH, SUBJECT_BEARING_KEY

# --- SHOT SIZE  (occupancy %, how much of the frame the subject fills) ---
SHOT_SIZE: dict[str, tuple[float, float, float]] = {
    "extreme wide shot": (0.0, 8.0, 4.0),
    "wide shot": (8.0, 20.0, 14.0),
    "medium-wide shot": (20.0, 38.0, 29.0),
    "medium shot": (38.0, 58.0, 48.0),
    "medium close-up": (58.0, 78.0, 68.0),
    "close-up": (78.0, 100.01, 88.0),
}
# --- BODY FRAMING  (body_in_frame_ratio %, how much of the subject's body is inside the frame) ---
BODY_FRAMING: dict[str, tuple[float, float, float]] = {
    "tightly cropped": (0.0, 30.0, 18.0),
    "partially cut off": (30.0, 60.0, 45.0),
    "mostly in frame": (60.0, 90.0, 75.0),
    "full body in frame": (90.0, 100.01, 96.0),
}
# --- ELEVATION  (cam_to_obj_elevation_deg; NEGATIVE = camera ABOVE the subject = high / looking down) ---
ELEVATION: dict[str, tuple[float, float, float]] = {
    "high angle": (-90.0, -20.0, -42.0),   # camera above, looking down
    "eye level": (-20.0, 15.0, -3.0),
    "low angle": (15.0, 90.0, 33.0),       # camera below, looking up
}
# --- PLACEMENT  (subject centre position as a fraction of frame width/height; y=0 is the TOP) ---
PLACE_X: dict[str, tuple[float, float, float]] = {
    "off-screen left": (-9.0, 0.0, -0.12),
    "left third": (0.0, 0.38, 0.19),
    "centered": (0.38, 0.62, 0.5),
    "right third": (0.62, 1.0, 0.81),
    "off-screen right": (1.0, 9.0, 1.12),
}
PLACE_Y: dict[str, tuple[float, float, float]] = {
    "off-screen top": (-9.0, 0.0, -0.12),
    "upper": (0.0, 0.38, 0.19),
    "mid": (0.38, 0.62, 0.5),
    "lower": (0.62, 1.0, 0.81),
    "off-screen bottom": (1.0, 9.0, 1.12),
}
# --- CROP SIDE  (which END of the subject the frame cuts) ---
# Two keys, not one, so this axis deliberately stays OUT of AXIS_KEY: that table is 1:1
# axis->key and drives `is_partial()`'s six-key contract. The labels must match
# `src.data.lerobot_export.crop_phrase` exactly, or an authored goal and a training goal
# describing the same framing would produce different sentences.
# Values are (top_cut_frac, bot_cut_frac) centroids chosen so `crop_phrase` reproduces the
# label verbatim (it cuts at 0.02 for "is it cut" and 0.35 for legs-vs-waist).
CROP_SIDE: dict[str, tuple[float, float]] = {
    "uncropped": (0.0, 0.0),
    "cropped at the legs": (0.0, 0.20),
    "cropped below the waist": (0.0, 0.50),
    "cropped above the head": (0.30, 0.0),
    "cropped at both the head and the feet": (0.30, 0.30),
}


def crop_label(top_cut_frac: float, bot_cut_frac: float) -> str:
    """(top, bot) fractions -> the same word `crop_phrase` would emit."""
    top, bot = float(top_cut_frac) > 0.02, float(bot_cut_frac) > 0.02
    if top and bot:
        return "cropped at both the head and the feet"
    if top:
        return "cropped above the head"
    if bot:
        return "cropped below the waist" if float(bot_cut_frac) > 0.35 else "cropped at the legs"
    return "uncropped"


# coarse subject-relative bearing centroids (when the user says only "front/side/back")
SECTOR3_CENTROID: dict[str, float] = {"front": 0.0, "side": 90.0, "back": 180.0}

# which profile key each category axis grounds to
AXIS_KEY = {
    "shot_size": "occupancy",
    "body_framing": "body_in_frame_ratio",
    "elevation": "cam_to_obj_elevation_deg",
    "bearing": SUBJECT_BEARING_KEY,
    "placement_x": "object_center_x",
    "placement_y": "object_center_y",
}
# geometry-only keys: exactly computable from a 2D bbox (Module 2) but not user-authored in NL.
# bbox_*_offset are the subject's apparent half-extents (redundant size cue, ~corr 0.68 w/ occupancy).
GEOMETRY_ONLY_KEYS = ("bbox_x_offset", "bbox_y_offset")


def _classify(value: float, table: dict[str, tuple[float, float, float]]) -> str:
    for label, (lo, hi, _c) in table.items():
        if lo <= value < hi:
            return label
    return next(iter(table)) if value < 0 else list(table)[-1]  # clamp to an edge band


def bearing_centroid(label: str) -> float:
    """Inverse of `sector8`: representative subject-bearing angle for an 8-way view label."""
    return float(SECTOR8.index(label) * 45)


# ---------------------------------------------------------------------------- #
# profile  <->  categories
# ---------------------------------------------------------------------------- #
def profile_to_categories(profile: Mapping[str, float]) -> dict[str, str]:
    """Classify a (possibly partial) numeric profile into cinematography categories."""
    cats: dict[str, str] = {}
    if "occupancy" in profile:
        cats["shot_size"] = _classify(float(profile["occupancy"]), SHOT_SIZE)
    if "body_in_frame_ratio" in profile:
        cats["body_framing"] = _classify(float(profile["body_in_frame_ratio"]), BODY_FRAMING)
    if "cam_to_obj_elevation_deg" in profile:
        cats["elevation"] = _classify(float(profile["cam_to_obj_elevation_deg"]), ELEVATION)
    if SUBJECT_BEARING_KEY in profile:
        cats["bearing"] = sector8(float(profile[SUBJECT_BEARING_KEY]))
    if "object_center_x" in profile:
        cats["placement_x"] = _classify(float(profile["object_center_x"]) / RENDER_WIDTH, PLACE_X)
    if "object_center_y" in profile:
        cats["placement_y"] = _classify(float(profile["object_center_y"]) / RENDER_HEIGHT, PLACE_Y)
    if "top_cut_frac" in profile or "bot_cut_frac" in profile:
        cats["crop_side"] = crop_label(profile.get("top_cut_frac", 0.0),
                                       profile.get("bot_cut_frac", 0.0))
    return cats


def categories_to_profile(cats: Mapping[str, str]) -> tuple[dict[str, float], frozenset[str]]:
    """Ground categories to numeric profile values (centroids). Returns (values, specified-keys).
    Only keys with a provided category are set -> the rest are UNSPECIFIED (partial goal)."""
    vals: dict[str, float] = {}
    if "shot_size" in cats:
        vals["occupancy"] = SHOT_SIZE[cats["shot_size"]][2]
    if "body_framing" in cats:
        vals["body_in_frame_ratio"] = BODY_FRAMING[cats["body_framing"]][2]
    if "elevation" in cats:
        vals["cam_to_obj_elevation_deg"] = ELEVATION[cats["elevation"]][2]
    if "bearing" in cats:
        b = cats["bearing"]
        vals[SUBJECT_BEARING_KEY] = SECTOR3_CENTROID[b] if b in SECTOR3_CENTROID else bearing_centroid(b)
    if "placement_x" in cats:
        vals["object_center_x"] = PLACE_X[cats["placement_x"]][2] * RENDER_WIDTH
    if "placement_y" in cats:
        vals["object_center_y"] = PLACE_Y[cats["placement_y"]][2] * RENDER_HEIGHT
    if "crop_side" in cats:
        t, b = CROP_SIDE[cats["crop_side"]]
        vals["top_cut_frac"], vals["bot_cut_frac"] = t, b
        vals["head_in_frame"] = 0.0 if t > 0.02 else 1.0
        if t <= 0.02 and b <= 0.02:
            vals["visible_frac"] = 1.0        # only knowable when nothing is cut
    return vals, frozenset(vals)


# ---------------------------------------------------------------------------- #
# profile  ->  natural language  (the conditioning-prompt serializer)
# ---------------------------------------------------------------------------- #
def profile_to_nl(
    profile: Mapping[str, float],
    specified: Mapping[str, float] | frozenset[str] | None = None,
    *,
    numbers: bool = True,
    coarse_bearing: bool = False,
) -> str:
    """Serialize a (partial) profile to a cinematography sentence — words + optional numbers.
    Only the `specified` keys are described (partial goal -> shorter prompt)."""
    spec = set(specified) if specified is not None else set(profile)
    parts: list[str] = []

    if "occupancy" in spec:
        occ = float(profile["occupancy"])
        s = _classify(occ, SHOT_SIZE)
        parts.append(f"a {s}" + (f" (subject fills ~{round(occ)}% of frame)" if numbers else ""))
    else:
        parts.append("a shot")

    if SUBJECT_BEARING_KEY in spec:
        b = float(profile[SUBJECT_BEARING_KEY])
        word = sector3(b) if coarse_bearing else sector8(b)
        parts.append(f"of the subject seen from the {word}"
                     + (f" ({round(b)}°)" if numbers and not coarse_bearing else ""))
    else:
        parts.append("of the subject")

    if "cam_to_obj_elevation_deg" in spec:
        parts.append(f"at {_classify(float(profile['cam_to_obj_elevation_deg']), ELEVATION)}")

    px = _classify(float(profile["object_center_x"]) / RENDER_WIDTH, PLACE_X) if "object_center_x" in spec else None
    py = _classify(float(profile["object_center_y"]) / RENDER_HEIGHT, PLACE_Y) if "object_center_y" in spec else None
    loc = " ".join(w for w in (py, px) if w)
    if loc:
        parts.append(f"subject positioned {loc} of the frame")

    if "body_in_frame_ratio" in spec:
        parts.append(_classify(float(profile["body_in_frame_ratio"]), BODY_FRAMING))

    return ", ".join(parts) + "."


# machine-readable JSON goal (Cosmos ActionPromptJsonFormatter-friendly alternative to prose)
def profile_to_json_goal(profile: Mapping[str, float],
                         specified: frozenset[str] | None = None) -> dict[str, str]:
    """Category dict for only the specified keys — a structured cinematography goal."""
    spec = set(specified) if specified is not None else set(profile)
    sub = {k: profile[k] for k in profile if k in spec}
    return profile_to_categories(sub)
