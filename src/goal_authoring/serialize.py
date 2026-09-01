"""The goal -> prompt contract: ONE serializer, shared by training and every authoring path.

`goal_prompt` is the sentence the policy was trained on (`src/data/lerobot_export.py` re-exports
it from here so the exporter and the eval scripts keep their import path). It used to live in
the exporter; it lives in the authoring package now because authoring's whole job is to produce
something this function can turn into a string the policy has seen before.

`TRAINING_PROMPT_RE` is the SHAPE of every prompt in the v5 export: all eight keys as word and
number, a crop phrase, and a `visible` number. Measured, not assumed — 44,560 / 44,560 unique task
strings in `runs/lerobot_v5/meta/tasks.parquet` match it. An authored prompt that does not match
is a prompt the policy has never been shown; `is_training_shape` is the cheap check for that, and
`parse_training_prompt` is the inverse (it is how the goal bank is built).
"""

from __future__ import annotations

import re
from typing import Collection, Mapping, Sequence

import numpy as np

from src.common.facing import sector8
from src.common.goal_space import DEFAULT_GOAL_KEYS, RENDER_HEIGHT, RENDER_WIDTH, SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.vocab import _classify

# The crop keys ride along with the eight goal keys: they are not in the goal VECTOR (they come
# from the signed bbox), but every training prompt carries them, so a profile is only complete
# when it has all eleven.
CROP_KEYS = ("top_cut_frac", "bot_cut_frac", "visible_frac")
FULL_PROFILE_KEYS = tuple(DEFAULT_GOAL_KEYS) + CROP_KEYS

PROMPT_PREFIX = "Move the camera to achieve this shot: "


def crop_phrase(top_cut: float, bot_cut: float) -> str:
    """Which END of the subject the frame cuts.

    This is the one composition fact no goal key carries: `occupancy` and
    `body_in_frame_ratio` are area ratios and `object_center_y` is clipped on screen,
    so "head cropped" and "legs cropped" are indistinguishable in the vector. It comes
    from the signed bbox instead (`src.common.annotations._apply_crop_extent`).
    """
    top, bot = top_cut > 0.02, bot_cut > 0.02
    if top and bot:
        return "cropped at both the head and the feet"
    if top:
        return "cropped above the head"
    if bot:
        return "cropped below the waist" if bot_cut > 0.35 else "cropped at the legs"
    # Not "the whole subject in frame" — BODY_FRAMING already says that in the preceding
    # clause, and the two would read as a stutter. They are different quantities (area
    # ratio vs vertical extent), so the word still has to be here; it just has to differ.
    return "uncropped"


def goal_prompt(
    goal_vec: np.ndarray,
    keys: Sequence[str] = tuple(DEFAULT_GOAL_KEYS),
    *,
    crop: Mapping[str, float] | None = None,
    specified: Collection[str] | None = None,
) -> str:
    """The goal as a cinematography instruction: every key, as a word AND as a number.

    The model receives the goal ONLY as this text — the goal vector never reaches it in
    numeric form — so a key omitted here is a key the policy can neither be asked for nor
    learn. The previous version serialized 3 of the 8 keys, and the axis probe showed
    exactly that consequence: `object_center_x` and `object_center_y`, the two absent
    ones, scored at chance (3/6 correct direction) because the request was never sent.

    Each word is paired with its number in the same clause. That pairing is what gives
    the number meaning: the words vary per sample, so they teach the mapping in a way a
    constant schema explanation could not — and a constant explanation would also be
    identical across all samples, carrying no signal for telling goals apart.

    Category words come from `src.goal_authoring.vocab`, the single source of truth.
    They used to be hand-copied here and had drifted (elevation cut at -25/+10 vs the
    vocab's -20/+15, label "medium close-up shot" vs "medium close-up"), so the same
    goal produced different sentences depending on which path serialized it.

    `crop` is optional because it is not part of the goal vector: it needs the signed
    bbox from the annotation (`top_cut_frac` / `bot_cut_frac`). Callers that synthesize
    a goal by perturbing a vector — the axis probe — have no such frame, and simply omit
    the clause rather than assert something they cannot know.
    """
    v = {k: float(goal_vec[i]) for i, k in enumerate(keys)}
    bearing = v[SUBJECT_BEARING_KEY]
    occ = v["occupancy"]
    body = v["body_in_frame_ratio"]
    elev = v["cam_to_obj_elevation_deg"]
    cx, cy = v["object_center_x"], v["object_center_y"]
    ox, oy = v["bbox_x_offset"], v["bbox_y_offset"]

    # `specified=None` means "every key is real" — the training export, where the goal comes
    # from an actual frame. A goal AUTHORED from language or a reference image constrains only
    # some axes, and the rest arrive as 0.0 from `gp.values.get(k, 0.0)`. Emitting those would
    # not be a harmless default: occupancy 0 reads as "extreme wide shot", body_in_frame 0 as
    # "tightly cropped", centre 0/0 as the top-left corner. The clause is dropped instead.
    have = (lambda k: True) if specified is None else (lambda k: k in specified)

    # The first two clauses are joined by a SPACE, the rest by ", " — "a wide shot of the
    # subject from the subject's back, at eye level, ...". Joining all of them with ", "
    # changes every training prompt (checked: 0/306 rebuilt prompts matched the exported set).
    head = (f"a {_classify(occ, vocab.SHOT_SIZE)} of the subject"
            if have("occupancy") else "a shot of the subject")
    if have(SUBJECT_BEARING_KEY):
        head += f" from the subject's {sector8(bearing)}"

    clauses, nums = [head], []
    if have("cam_to_obj_elevation_deg"):
        clauses.append(f"at {_classify(elev, vocab.ELEVATION)}")
    if have("object_center_x") and have("object_center_y"):
        clauses.append(f"{_classify(cx / RENDER_WIDTH, vocab.PLACE_X)} and "
                       f"{_classify(cy / RENDER_HEIGHT, vocab.PLACE_Y)} in the frame")
    elif have("object_center_x"):
        clauses.append(f"{_classify(cx / RENDER_WIDTH, vocab.PLACE_X)} in the frame")
    elif have("object_center_y"):
        clauses.append(f"{_classify(cy / RENDER_HEIGHT, vocab.PLACE_Y)} in the frame")
    if have("body_in_frame_ratio"):
        clauses.append(_classify(body, vocab.BODY_FRAMING))
    if crop is not None:
        clauses.append(crop_phrase(float(crop.get("top_cut_frac", 0.0)),
                                   float(crop.get("bot_cut_frac", 0.0))))

    if have(SUBJECT_BEARING_KEY):
        nums.append(f"bearing {bearing:.0f}°")
    if have("occupancy"):
        nums.append(f"occupancy {occ:.0f}%")
    if have("cam_to_obj_elevation_deg"):
        nums.append(f"elevation {elev:.0f}°")
    if have("body_in_frame_ratio"):
        nums.append(f"body_in_frame {body:.0f}%")
    if have("object_center_x") and have("object_center_y"):
        nums.append(f"center {cx:.0f}/{cy:.0f} px")
    if have("bbox_x_offset") and have("bbox_y_offset"):
        nums.append(f"half_size {ox:.0f}/{oy:.0f} px")
    if crop is not None and crop.get("visible_frac") is not None:
        nums.append(f"visible {float(crop['visible_frac']):.2f}")

    text = f"{PROMPT_PREFIX}{', '.join(clauses)}."
    return f"{text} ({', '.join(nums)})" if nums else text


def profile_to_prompt(values: Mapping[str, float], specified: Collection[str] | None = None) -> str:
    """`goal_prompt` for a key->value profile (the authoring-side entry point).

    `specified=None` serializes every key — only valid for a COMPLETE profile (`is_complete`);
    a partial one must pass its `specified` set, and the result is then a prompt shape the
    policy was never trained on (see `TRAINING_PROMPT_RE`). Crop keys are emitted whenever
    `top_cut_frac` / `bot_cut_frac` are present, exactly as the exporter does.
    """
    vec = np.array([float(values.get(k, 0.0)) for k in DEFAULT_GOAL_KEYS], dtype=np.float64)
    crop = None
    if "top_cut_frac" in values or "bot_cut_frac" in values:
        crop = {"top_cut_frac": float(values.get("top_cut_frac", 0.0)),
                "bot_cut_frac": float(values.get("bot_cut_frac", 0.0))}
        if values.get("visible_frac") is not None:
            crop["visible_frac"] = float(values["visible_frac"])
    return goal_prompt(vec, crop=crop, specified=specified)


def is_complete(values: Mapping[str, float]) -> bool:
    return all(k in values for k in FULL_PROFILE_KEYS)


# --------------------------------------------------------------------------- #
# the training shape, and its inverse
# --------------------------------------------------------------------------- #
_CROP_WORDS = "|".join(re.escape(w) for w in vocab.CROP_SIDE)
TRAINING_PROMPT_RE = re.compile(
    r"^" + re.escape(PROMPT_PREFIX)
    + r"a (?P<shot>[a-z\- ]+?) of the subject from the subject's (?P<sector>[a-z\-]+), "
    r"at (?P<elev_word>high angle|eye level|low angle), "
    r"(?P<px>[a-z\- ]+?) and (?P<py>[a-z\- ]+?) in the frame, "
    r"(?P<body_word>[a-z ]+?), (?P<crop>" + _CROP_WORDS + r")\. "
    r"\(bearing (?P<bearing>-?\d+)°, occupancy (?P<occupancy>\d+)%, "
    r"elevation (?P<elevation>-?\d+)°, body_in_frame (?P<body>\d+)%, "
    r"center (?P<cx>-?\d+)/(?P<cy>-?\d+) px, half_size (?P<hx>\d+)/(?P<hy>\d+) px, "
    r"visible (?P<visible>\d\.\d\d)\)$"
)


def is_training_shape(prompt: str) -> bool:
    """Does this prompt have the exact shape of every v5 training prompt?"""
    return TRAINING_PROMPT_RE.match(prompt) is not None


def parse_training_prompt(prompt: str) -> dict | None:
    """Training prompt -> the numeric profile it encodes (plus the crop label).

    Returns the eight goal keys, `visible_frac`, and `crop_label`; the crop FRACTIONS are
    not recoverable from the sentence (only the word is), so callers map the label back
    through `vocab.CROP_SIDE` when they need numbers. `None` if the shape does not match.
    """
    m = TRAINING_PROMPT_RE.match(prompt)
    if not m:
        return None
    return {
        SUBJECT_BEARING_KEY: float(m["bearing"]),
        "occupancy": float(m["occupancy"]),
        "cam_to_obj_elevation_deg": float(m["elevation"]),
        "body_in_frame_ratio": float(m["body"]),
        "object_center_x": float(m["cx"]),
        "object_center_y": float(m["cy"]),
        "bbox_x_offset": float(m["hx"]),
        "bbox_y_offset": float(m["hy"]),
        "visible_frac": float(m["visible"]),
        "crop_label": m["crop"],
    }


__all__ = [
    "CROP_KEYS", "FULL_PROFILE_KEYS", "PROMPT_PREFIX", "TRAINING_PROMPT_RE",
    "crop_phrase", "goal_prompt", "profile_to_prompt", "is_complete",
    "is_training_shape", "parse_training_prompt",
]
