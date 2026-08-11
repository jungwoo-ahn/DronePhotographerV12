"""Iterator over v7 placement annotations.

The v7 dataset (`docs/v7_handoff_jooyeol.md` on branch `v7_data_for_cosmos_policy`)
is the only format this project consumes. Layout:

    outputs/v7_stage2_renders/<scene>__<object>/
    ├── data.json
    ├── renders/pair_<pp>_frame_<ff>.jpg     (K_accepted × 32 JPEGs)
    ├── done.flag                            (Stage 2 complete)
    └── scored.flag                          (Stage 3 complete)

`data.json` carries Stage 1 (placement + accepted_pairs[].trajectory_32f) +
Stage 2 (render_records[][].path_rel/bbox/in_frame) + Stage 3 (render_records[][].scores
with 8 V5 keys per frame).

`iter_windows` slides a `chunk_size`-step window over each
`accepted_pairs[i].trajectory_32f` (length 32). Each yielded `TrajectoryWindow`
holds the start frame, end frame, and intermediate frames between them. The
action chunk + goal vector are computed downstream in `BasePolicyDataset`.
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH
from src.scoring.bbox_control import compute_v5_scores


@dataclass
class ViewRecord:
    """One rendered frame inside a v7 trajectory."""

    annotation_path: Path
    scene: str
    scene_file: str
    scene_scale: float
    object: str
    object_file: str
    pair_idx: int                       # which accepted_pair this frame belongs to
    frame_idx: int                      # 0..31 within trajectory_32f
    object_position: list[float]        # subject_foot (world frame)
    subject_center: list[float]         # subject bbox center (world frame) — for pose-based value
    subject_height: float               # subject height (m) — for pose-based apparent size
    image: str                          # absolute path to the rendered JPEG
    camera_position: list[float]
    camera_forward: list[float]
    camera_up: list[float]
    azimuth: float | None               # frame.yaw_deg (camera-side, not cam→obj)
    elevation: float | None             # frame.pitch_deg
    render_width: int                   # for pixel→angle conversion in the value metric
    render_height: int
    raw: dict                           # frame dict + injected Stage 3 scores


@dataclass
class TrajectoryWindow:
    """A K-step window from a trajectory: start frame, end frame, K-1 intermediates."""

    annotation_path: Path
    scene: str
    scene_file: str
    object: str
    object_file: str
    pair_idx: int
    start_frame_idx: int
    end_frame_idx: int                  # = start_frame_idx + chunk_size
    chunk_size: int
    start: ViewRecord
    end: ViewRecord
    intermediate: list[ViewRecord]      # length chunk_size - 1
    future: list[ViewRecord] = field(default_factory=list)
    # frames AFTER end on the same trajectory (end_frame_idx+1 .. 31) — the
    # HER-"future" goal candidate pool (goal = any frame in [end, 31]).
    keyframes: list[ViewRecord] = field(default_factory=list)
    # the chunk_size+1 frames whose consecutive pose deltas ARE the action chunk
    # (and whose per-frame poses feed the per-step value): [start, …, end].
    # sliding_window → contiguous start..end; multiscale_bidir → strided by
    # frame_step. `_compute_action_chunk` / the value sequence read this list.
    frame_step: int = 1                 # real trajectory frames spanned per action (1|2|3)
    direction: int = 1                  # +1 forward along the trajectory, -1 reversed
    goal_frame: ViewRecord | None = None
    # `goal_start` scheme only: the goal is a well-framed frame BEYOND the window's
    # end (delta 8..32 away), so the chunk is "the immediate steps toward the goal",
    # not the whole path to it. The other schemes leave this None and pin the goal to
    # `end` / the `future` pool.


def load_annotation(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_clamped_scores(raw: dict) -> bool:
    """The v7 stage-3 near-plane sentinel: occupancy 0 with a zero bbox size."""
    return raw.get("occupancy") == 0 and raw.get("bbox_y_offset") == 0


def _recover_clamped_goal(raw: dict) -> None:
    """Recover near-plane-clamped goal frames in place (no re-render).

    When the dolly closes in, the mesh-tight bbox blows up and the v7 stage-3
    scorer baked a zeroed sentinel (occupancy 0, bbox keys 0; az/el kept). Those
    frames are valid extreme close-ups — `occupancy_clipped`~1.0, `in_frame`
    True — that the old filter dropped. Recompute a bounded profile from the
    stored full bbox via the (now non-zeroing) `compute_v5_scores`, and take
    occupancy from the stored `occupancy_clipped` ground truth. No-op for
    non-sentinel frames and for genuine no-projection frames (bbox is None).
    """
    if not _is_clamped_scores(raw):
        return
    bbox = raw.get("bbox_xyxy_full")
    if not bbox:
        return  # genuine no-projection (subject behind camera) — leave zeroed
    az = float(raw.get("cam_to_obj_azimuth_deg") or 0.0)
    el = float(raw.get("cam_to_obj_elevation_deg") or 0.0)
    fixed = compute_v5_scores(int(RENDER_WIDTH), int(RENDER_HEIGHT), [float(v) for v in bbox], az, el)
    occ_clip = raw.get("occupancy_clipped")
    if occ_clip is not None:
        fixed["occupancy"] = max(0, min(100, int(round(100.0 * float(occ_clip)))))
    raw.update(fixed)


def _apply_visible_geometry(raw: dict, width: int, height: int) -> None:
    """Re-derive object_center / bbox_offset under the VISIBLE-bbox convention (no re-render).

    The v7 stage-3 scorer baked object_center / bbox_offset from the FULL projected bbox; for a
    subject cut off by the frame edge that center sits off-screen. `compute_v5_scores` now reads
    those from the frame-CLIPPED bbox, so recomputing from the stored `bbox_xyxy_full` here aligns
    the training goal with what Module 2 reads off a reference image (only the visible box exists
    there). Occupancy / body_in_frame / az / el are left as stored."""
    bbox = raw.get("bbox_xyxy_full")
    if not bbox:
        return
    fixed = compute_v5_scores(int(width or RENDER_WIDTH), int(height or RENDER_HEIGHT),
                              [float(v) for v in bbox],
                              float(raw.get("cam_to_obj_azimuth_deg") or 0.0),
                              float(raw.get("cam_to_obj_elevation_deg") or 0.0))
    for k in ("object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset"):
        raw[k] = fixed[k]
    _apply_crop_extent(raw, bbox, float(height or RENDER_HEIGHT))


def _apply_crop_extent(raw: dict, bbox, height: float) -> None:
    """Which END of the subject the frame cuts, and how much of it survives.

    Everything else in the profile is sign-destroying: `occupancy` and
    `body_in_frame_ratio` are area ratios and `object_center_y` is clipped on
    screen, so a beheaded subject and a chest-up portrait are indistinguishable.
    Measured consequence: the `body_in_frame_ratio >= 70` gate admitted goals that
    were 72.7% head-cropped, and rejected EVERY bust-extent frame (a chest-up shot
    shows 35-60% of the body, so it cannot clear 70 by construction).

    `bbox_xyxy_full` is the unclipped signed projection — it really does carry
    negative y0 and y1 beyond the frame — so the distinction is exact here, with no
    re-render, no keypoints and no detector.

    `visible_frac` is deliberately crop-side AGNOSTIC: it is the gate quantity, and
    gating on it preserves whatever top/bottom mix the pool happens to have instead
    of selecting one end. `top_cut_frac` / `bot_cut_frac` carry the side, for the
    prompt and for reporting.
    """
    try:
        y0, y1 = float(bbox[1]), float(bbox[3])
    except (TypeError, IndexError, ValueError):
        return
    span = y1 - y0
    if span <= 0:
        return
    raw["head_in_frame"] = bool(y0 >= 0.0)
    raw["top_cut_frac"] = max(0.0, -y0) / span
    raw["bot_cut_frac"] = max(0.0, y1 - height) / span
    raw["visible_frac"] = max(0.0, min(y1, height) - max(y0, 0.0)) / span


def _frame_to_view(
    doc: dict,
    annotation_path: Path,
    placement_dir: Path,
    pair_idx: int,
    frame_idx: int,
    frame: dict,
    render_record: dict | None,
) -> ViewRecord:
    """Convert a `trajectory_32f` frame + its `render_records[i][j]` into a `ViewRecord`.

    If `render_record` is absent (Stage 2/3 not yet run for this frame), we
    synthesize the expected `path_rel` and leave scores out — `goal_vector` will
    then return NaN and the sample will be filtered downstream.
    """
    if render_record is not None:
        image_rel = render_record.get("path_rel", f"renders/pair_{pair_idx:02d}_frame_{frame_idx:02d}.jpg")
        scores = render_record.get("scores") or {}
    else:
        image_rel = f"renders/pair_{pair_idx:02d}_frame_{frame_idx:02d}.jpg"
        scores = {}

    raw = dict(frame)
    raw.update(scores)
    raw["frame_idx"] = frame_idx
    # Placement-level, but carried per frame on purpose: `goal_vector` reads the goal
    # out of a single frame's `raw`, and threading a new argument through the six call
    # sites instead would let one be forgotten — which fails silently, as a wrong
    # bearing rather than an error. Absent on original (un-re-rendered) data, where 0
    # reproduces the previous behaviour exactly.
    raw["placement_yaw_deg"] = float(doc.get("placement_yaw_deg") or 0.0)
    if render_record is not None:
        raw["bbox_xyxy_full"] = render_record.get("bbox_xyxy_full")
        raw["in_frame"] = render_record.get("in_frame")
        raw["occupancy_clipped"] = render_record.get("occupancy_clipped")
        _recover_clamped_goal(raw)
        _apply_visible_geometry(raw, doc.get("render_width"), doc.get("render_height"))

    return ViewRecord(
        annotation_path=annotation_path,
        scene=str(Path(doc.get("scene_file", "")).parent.name) or doc.get("scene", ""),
        scene_file=doc.get("scene_file", ""),
        scene_scale=float(doc.get("scene_scale", 1.0)),
        object=doc.get("object", "") or doc.get("placement", "").split("__")[-1],
        object_file=doc.get("object_file", ""),
        pair_idx=pair_idx,
        frame_idx=frame_idx,
        object_position=list(doc.get("subject_foot") or [0.0, 0.0, 0.0]),
        subject_center=list(doc.get("subject_center") or doc.get("subject_foot") or [0.0, 0.0, 0.0]),
        subject_height=float(doc.get("subject_height") or 1.7),
        image=str(placement_dir / image_rel),
        camera_position=list(frame["pos"]),
        camera_forward=list(frame["forward"]),
        camera_up=list(frame["up"]),
        azimuth=frame.get("yaw_deg"),
        elevation=frame.get("pitch_deg"),
        render_width=int(doc.get("render_width") or 0),
        render_height=int(doc.get("render_height") or 0),
        raw=raw,
    )


def _pair_views(
    doc: dict,
    data_json_path: Path,
    placement_dir: Path,
    pair_idx: int,
    trajectory: list,
    render_records: list,
) -> list[ViewRecord]:
    """Build the per-frame `ViewRecord` list for one accepted_pair's trajectory."""
    recs = render_records[pair_idx] if pair_idx < len(render_records) else []
    recs_by_idx = {int(r.get("frame_idx", k)): r for k, r in enumerate(recs)}
    return [
        _frame_to_view(doc, data_json_path, placement_dir, pair_idx, j, trajectory[j], recs_by_idx.get(j))
        for j in range(len(trajectory))
    ]


def iter_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    stride: int = 1,
    reverse: bool = False,
) -> Iterator[TrajectoryWindow]:
    """Slide a chunk_size-step window over each accepted_pair's trajectory_32f.

    reverse=True plays each trajectory backward (camera path end->start): the action
    chunks become the inverse motion (dolly-OUT where the forward path dollies in) and
    the HER-"future" goals are FARTHER frames. Used to balance the dataset's strong
    far->near bias (~81% of trajectories dolly in). No re-render — the same rendered
    frames are reused in reverse order.
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        if len(trajectory) <= chunk_size:
            continue
        view_records = _pair_views(doc, data_json_path, placement_dir, pair_idx, trajectory, render_records)
        if reverse:
            view_records = list(reversed(view_records))
        n = len(view_records)
        for start_idx in range(0, n - chunk_size, stride):
            end_idx = start_idx + chunk_size
            yield TrajectoryWindow(
                annotation_path=data_json_path,
                scene=view_records[start_idx].scene,
                scene_file=view_records[start_idx].scene_file,
                object=view_records[start_idx].object,
                object_file=view_records[start_idx].object_file,
                pair_idx=pair_idx,
                start_frame_idx=view_records[start_idx].frame_idx,
                end_frame_idx=view_records[end_idx].frame_idx,
                chunk_size=chunk_size,
                start=view_records[start_idx],
                end=view_records[end_idx],
                intermediate=view_records[start_idx + 1 : end_idx],
                future=view_records[end_idx + 1 :],
                keyframes=view_records[start_idx : end_idx + 1],
                frame_step=1,
                direction=-1 if reverse else 1,
            )


def iter_multiscale_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    offsets: Iterable[int] = (8, 16, 24),
) -> Iterator[TrajectoryWindow]:
    """Bidirectional multi-scale endpoint windows — makes actions depend on the goal.

    For each start frame `p` and each signed offset `±o` (o in `offsets`) whose
    endpoint `p±o` exists in the trajectory, emit a window whose `chunk_size`
    actions traverse from `p` to that endpoint. The endpoint frame IS the goal
    (its profile), so the SAME start with DIFFERENT endpoints yields DIFFERENT
    action targets — forcing the policy to condition on the goal instead of
    collapsing to `f(state)` (the failure mode of the sliding-window scheme, where
    the action is pinned to the window while the HER goal varies independently).

    Each offset must be a positive multiple of `chunk_size`; the ratio is the
    per-action `frame_step` s (o=8→s=1, 16→2, 24→3 at chunk_size=8). The action
    chunk is re-encoded between the STRIDED keyframes `[p, p±s, …, p±(chunk_size·s)]`
    downstream — NOT by summing single-step deltas, which do not compose in the
    camera-local basis. Negative offsets play the path backward (dolly-OUT), so
    this subsumes the `reverse=` augmentation of `iter_windows`.

    On a 32-frame trajectory with offsets (8,16,24) this yields 96 windows/pair
    (24+16+8 forward + 24+16+8 reverse), before any downstream goal filtering.
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    offset_list = sorted({int(o) for o in offsets})
    for o in offset_list:
        if o <= 0 or o % chunk_size != 0:
            raise ValueError(
                f"offset {o} must be a positive multiple of chunk_size={chunk_size} "
                "(so it maps to an integer per-action frame_step)"
            )

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        n = len(trajectory)
        if n <= 1:
            continue
        view_records = _pair_views(doc, data_json_path, placement_dir, pair_idx, trajectory, render_records)
        for start_idx in range(n):
            for o in offset_list:
                step = o // chunk_size
                for direction in (1, -1):
                    end_idx = start_idx + direction * o
                    if end_idx < 0 or end_idx >= n:
                        continue
                    keyframes = [view_records[start_idx + direction * step * k] for k in range(chunk_size + 1)]
                    yield TrajectoryWindow(
                        annotation_path=data_json_path,
                        scene=keyframes[0].scene,
                        scene_file=keyframes[0].scene_file,
                        object=keyframes[0].object,
                        object_file=keyframes[0].object_file,
                        pair_idx=pair_idx,
                        start_frame_idx=keyframes[0].frame_idx,
                        end_frame_idx=keyframes[-1].frame_idx,
                        chunk_size=chunk_size,
                        start=keyframes[0],
                        end=keyframes[-1],
                        intermediate=keyframes[1:-1],
                        future=[],                      # goal is pinned to the endpoint; no HER pool
                        keyframes=keyframes,
                        frame_step=step,
                        direction=direction,
                    )


# Floor 10, not 20. The floor is a FRAMING filter in disguise: an uncropped subject is
# one shot from far enough away to fit, so it occupies little of the frame. At 20 the
# gate kept only 7.9% uncropped goals and doubled 'both ends cut' to 32.7%; at 10 the
# post-gate crop mix (17.4/42.3/20.5/19.8) tracks the pool's own (15.9/50.9/17.4/15.8).
DEFAULT_GOAL_OCCUPANCY_RANGE = (10.0, 80.0)
# Lower bound 0, not chunk_size: a start that already sits AT the goal yields an
# all-zero chunk, the only way the policy can learn to stop. `near_fraction` below
# controls how much of that appears, because letting it emerge from the raw pair
# counts would flood the set with near-goal starts and teach the policy to sit still.
DEFAULT_DELTA_RANGE = (0, 32)
DEFAULT_NEAR_FRACTION = 0.25
# Occupancy alone does NOT mean "well framed": it comes from the UNCLIPPED mesh-tight
# bbox, so a subject can fill 40% of the frame while hanging half outside it. But the
# old fix — body_in_frame_ratio >= 70 — was an AREA ratio, blind to which end of the
# subject the frame cuts. It selected goals that were 72.7% head-cropped and rejected
# every chest-up frame by construction (those show 35-60% of the body).
#
# 0.35 of the subject's VERTICAL extent is "at least a bust's worth is in frame",
# whichever end is cut. It is deliberately not a crop-side filter: the pool's own
# top/bottom mix is what we want to keep.
DEFAULT_MIN_GOAL_VISIBLE_FRAC = 0.35


_WARNED_NO_VISIBLE_FRAC = False


def _is_well_framed(
    raw: dict, min_visible_frac: float, require_center_on_screen: bool = False
) -> bool:
    """Composition gate a goal frame must pass on top of its occupancy band.

    Gates on `visible_frac` — how much of the subject's vertical extent survives the
    frame — rather than on `body_in_frame_ratio`. The old ratio is a 2D AREA ratio and
    therefore blind to WHICH end is cut, so `>= 70` scored a beheaded subject and a
    chest-up portrait identically and happened to select the beheaded one (72.7% of
    accepted goals). It also excluded every bust-extent frame by construction: a
    chest-up shot shows 35-60% of the body and can never reach 70.

    `visible_frac` is crop-side agnostic on purpose. The point is to stop the gate
    from PICKING a crop direction, so the pool's own top/bottom mix survives; the
    side itself lives in `top_cut_frac` / `bot_cut_frac` for the prompt.

    `require_center_on_screen` is retained only so existing callers keep working, and
    defaults off: `_apply_visible_geometry` recomputes the centre from the CLIPPED
    bbox, so it is on screen by construction and the check can now only reject a frame
    whose keys are missing entirely.
    """
    if not isinstance(raw, dict):
        return False
    if min_visible_frac > 0.0:
        vis = raw.get("visible_frac")
        if vis is None:
            # No signed bbox (no render record). Fall back to the old AREA ratio so a missing
            # key cannot silently empty the dataset — but say so once. Passing a raw on-disk
            # `scores` dict lands here, and then this gate silently measures something else
            # entirely (area, not vertical extent); two callers looked migrated for exactly
            # that reason. Enrich the dict via `_apply_crop_extent` instead.
            global _WARNED_NO_VISIBLE_FRAC
            if not _WARNED_NO_VISIBLE_FRAC:
                _WARNED_NO_VISIBLE_FRAC = True
                warnings.warn(
                    "_is_well_framed: no 'visible_frac' — falling back to body_in_frame_ratio, "
                    "which is an AREA ratio and cannot tell which end of the subject is cut. "
                    "Pass a raw enriched by _apply_crop_extent (needs bbox_xyxy_full).",
                    RuntimeWarning, stacklevel=2)
            body = raw.get("body_in_frame_ratio")
            if body is None or float(body) / 100.0 < min_visible_frac:
                return False
        elif float(vis) < min_visible_frac:
            return False
    if require_center_on_screen:
        cx, cy = raw.get("object_center_x"), raw.get("object_center_y")
        if cx is None or cy is None:
            return False
        if not (0.0 <= float(cx) <= RENDER_WIDTH and 0.0 <= float(cy) <= RENDER_HEIGHT):
            return False
    return True



def is_goal_frame(record: dict, *,
                  occupancy_range: tuple[float, float] | None = None,
                  min_visible_frac: float | None = None) -> bool:
    """Would this render record qualify as a training goal?

    One importable definition of "a usable shot", so the answer cannot drift per script. It
    was copy-pasted into eight of them with five different thresholds
    (`30 <= occupancy <= 92 and body_in_frame_ratio >= 45`, and 88/90/50/70 variants), all
    written against the retired area ratio.

    Takes the whole record, not just `record["scores"]`, because the gate needs the signed
    `bbox_xyxy_full` that sits beside them.
    """
    scores = record.get("scores")
    if not scores or not record.get("in_frame", True):
        return False
    lo, hi = occupancy_range or DEFAULT_GOAL_OCCUPANCY_RANGE
    occ = scores.get("occupancy")
    if occ is None or not (lo <= float(occ) <= hi):
        return False
    raw = dict(scores)
    bbox = record.get("bbox_xyxy_full")
    if bbox:
        _apply_crop_extent(raw, bbox, float(RENDER_HEIGHT))
    thr = DEFAULT_MIN_GOAL_VISIBLE_FRAC if min_visible_frac is None else min_visible_frac
    return _is_well_framed(raw, thr)

def iter_goal_start_windows(
    data_json_path: str | Path,
    *,
    chunk_size: int = 8,
    delta_range: tuple[int, int] = DEFAULT_DELTA_RANGE,
    near_fraction: float = DEFAULT_NEAR_FRACTION,
    goal_occupancy_range: tuple[float, float] = DEFAULT_GOAL_OCCUPANCY_RANGE,
    min_goal_visible_frac: float = DEFAULT_MIN_GOAL_VISIBLE_FRAC,
    require_goal_center_on_screen: bool = True,
    min_start_occupancy: float = 1.0,
    max_per_pair: int = 0,
    seed: int = 0,
) -> Iterator[TrajectoryWindow]:
    """Well-framed GOAL + a start some distance away; the chunk is the IMMEDIATE steps.

    Why not the trajectory's last frame: these are RANDOM camera motions, so the
    subject is best framed MID-trajectory and the camera then drifts past it — the
    terminal frame has median occupancy 0 and is empty in 52% of trajectories
    (measured over all 42840). A terminal goal would be mostly "subject not visible".

    So, per trajectory:
      * goal g — a WELL-FRAMED frame: occupancy in `goal_occupancy_range` (the upper
        bound drops saturated close-ups, the lower one the empty tail), at least
        `min_goal_visible_frac` of the subject's vertical extent inside the frame,
        the subject's centre actually on screen. The last two gates matter: occupancy
        is computed from the unclipped bbox, so on its own it admits subjects hanging
        and whichever end is cut (see DEFAULT_MIN_GOAL_VISIBLE_FRAC).
      * start s — any frame with `delta_range[0] <= |g - s| <= delta_range[1]` whose
        occupancy exceeds `min_start_occupancy`, so the policy never has to act from a
        frame where the subject is invisible. Both signs are used, which gives the
        dolly-out direction for free (no `reverse=` augmentation needed).
      * action chunk — the `chunk_size` steps from s TOWARD g, one trajectory frame
        each. Since `delta_range[0] >= chunk_size` the chunk never overshoots g: the
        policy learns "head toward the goal from here", not the whole path to it.

    `max_per_pair` caps (deterministically, seeded by placement/pair) how many of a
    trajectory's pairs are kept — the unrestricted scheme yields ~250 windows per
    trajectory (~10.8M over the dataset), far more than a training run needs.
    """
    data_json_path = Path(data_json_path)
    placement_dir = data_json_path.parent
    doc = load_annotation(data_json_path)
    accepted_pairs = doc.get("accepted_pairs") or []
    render_records = doc.get("render_records") or []

    d_min, d_max = int(delta_range[0]), int(delta_range[1])
    if d_min < 0 or d_max < d_min:
        raise ValueError(f"bad delta_range {delta_range}")
    occ_lo, occ_hi = float(goal_occupancy_range[0]), float(goal_occupancy_range[1])

    for pair_idx, pair in enumerate(accepted_pairs):
        trajectory = pair.get("trajectory_32f") or []
        n = len(trajectory)
        if n <= chunk_size:
            continue
        view_records = _pair_views(
            doc, data_json_path, placement_dir, pair_idx, trajectory, render_records
        )
        occupancy = [
            (v.raw.get("occupancy") if isinstance(v.raw, dict) else None) for v in view_records
        ]

        pairs_sg: list[tuple[int, int]] = []
        for g in range(n):
            og = occupancy[g]
            if og is None or not (occ_lo <= float(og) <= occ_hi):
                continue
            if not _is_well_framed(
                view_records[g].raw, min_goal_visible_frac, require_goal_center_on_screen
            ):
                continue
            for s in range(n):
                delta = abs(g - s)
                if delta < d_min or delta > d_max:
                    continue
                os_ = occupancy[s]
                if os_ is None or float(os_) <= min_start_occupancy:
                    continue
                pairs_sg.append((s, g))

        if max_per_pair and len(pairs_sg) > max_per_pair:
            rng = random.Random(f"{placement_dir.name}:{pair_idx}:{seed}")
            # Stratify by distance rather than sampling the pool flat. Starts within a
            # chunk of the goal vastly outnumber far ones (every g admits ~2*chunk_size
            # of them), so a flat draw would make most of the data "already there,
            # hold still" and the policy would learn to barely move.
            near = [sg for sg in pairs_sg if abs(sg[1] - sg[0]) < chunk_size]
            far = [sg for sg in pairs_sg if abs(sg[1] - sg[0]) >= chunk_size]
            n_near = min(len(near), int(round(max_per_pair * float(near_fraction))))
            n_far = min(len(far), max_per_pair - n_near)
            n_near = min(len(near), max_per_pair - n_far)      # backfill if far is short
            picked = rng.sample(near, n_near) + rng.sample(far, n_far)
            pairs_sg = sorted(picked)

        for s, g in pairs_sg:
            direction = 1 if g > s else -1
            # Walk toward g and CLAMP there. When delta < chunk_size the remaining
            # steps repeat the goal frame, so their action deltas are exactly zero —
            # that is the "you have arrived, hold still" supervision the old
            # delta >= chunk_size rule made impossible. The rule was meant to stop
            # labels overshooting the goal; in a rollout the camera reaches within a
            # chunk of the goal after a step or two and then has no idea what to do,
            # which is the overshoot we measure (88% of held-out episodes).
            idx = [s + direction * k for k in range(chunk_size + 1)]
            idx = [min(i, g) if direction > 0 else max(i, g) for i in idx]
            keyframes = [view_records[i] for i in idx]
            yield TrajectoryWindow(
                annotation_path=data_json_path,
                scene=keyframes[0].scene,
                scene_file=keyframes[0].scene_file,
                object=keyframes[0].object,
                object_file=keyframes[0].object_file,
                pair_idx=pair_idx,
                start_frame_idx=keyframes[0].frame_idx,
                end_frame_idx=keyframes[-1].frame_idx,
                chunk_size=chunk_size,
                start=keyframes[0],
                end=keyframes[-1],
                intermediate=keyframes[1:-1],
                future=[],                          # goal is explicit, not an HER pool
                keyframes=keyframes,
                frame_step=1,
                direction=direction,
                goal_frame=view_records[g],
            )


def list_annotation_files(roots: Iterable[str | Path]) -> list[Path]:
    """Find every `data.json` under each root.

    Each v7 placement contributes exactly one `data.json`, so we glob for that name.
    """
    out: list[Path] = []
    for r in roots:
        rp = Path(r)
        if rp.is_file() and rp.name == "data.json":
            out.append(rp)
        elif rp.is_dir():
            out.extend(sorted(rp.glob("**/data.json")))
    return out


__all__ = [
    "ViewRecord",
    "TrajectoryWindow",
    "iter_windows",
    "iter_multiscale_windows",
    "list_annotation_files",
    "load_annotation",
]
