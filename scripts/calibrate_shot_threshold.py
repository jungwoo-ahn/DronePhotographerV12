"""How close is close enough? Calibrate the rollout distance against "same shot".

`closed_loop_eval` reports `improvement = d_start - d_end` in radians, and a radian of
`src/common/reward.py` distance means nothing to a photographer. This turns it into a
statement about the picture: sample PAIRS of real frames inside one placement, measure the
same geometric distance between them, and ask whether the two frames carry the same
cinematography words — shot size, view sector, elevation band, placement x/y, crop side.

The answer is a curve, and the threshold read off it is what lets a rollout report
`success@tau` ("the frame it stopped on reads as the frame that was asked for") instead of
only "it got 0.45 closer".

Pairs come from the SAME placement, so the scene, the subject and its height are shared and
the distance is measured on exactly the quantity a rollout moves: the camera. Frames are
restricted to `is_goal_frame`, because a goal is what a rollout is aimed at, and drawn from
the held-out scenes so the number is not read off scenes the policy trained on.

    python scripts/calibrate_shot_threshold.py            # -> runs/eval/shot_threshold.json

venv: .venv-analysis (CPU, ~1 min).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np  # noqa: E402

from src.common.annotations import (_apply_crop_extent, _apply_visible_geometry,  # noqa: E402
                                    is_goal_frame)
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, DEFAULT_TRAJ_ROOT  # noqa: E402
from src.common.facing import front_azimuth, sector8  # noqa: E402
from src.common.goal_space import RENDER_HEIGHT, RENDER_WIDTH  # noqa: E402
from src.common.reward import _geometry_distance, pose_to_geometry  # noqa: E402
from src.goal_authoring.goal_bank import band_of  # noqa: E402
from src.goal_authoring.serialize import crop_phrase  # noqa: E402

DEFAULT_VAL_SCENES = str(Path(V12) / "configs" / "val_scenes.json")
DEFAULT_OUT = str(Path(V12) / "runs" / "eval" / "shot_threshold.json")

# The six axes a goal actually names. `bbox_*_offset` and `body_in_frame_ratio` are left out:
# they are size/area restatements of occupancy and the crop, so counting them would inflate
# agreement by counting the same fact more than once.
AXES = ("shot_size", "bearing", "elevation", "placement_x", "placement_y", "crop_side")


def shot_words(raw: dict, bearing: float) -> tuple[str, ...]:
    """The six category words a frame carries — the same tables the prompt is built from."""
    return (
        band_of("occupancy", raw["occupancy"]),
        sector8(bearing),
        band_of("cam_to_obj_elevation_deg", raw["cam_to_obj_elevation_deg"]),
        band_of("object_center_x", raw["object_center_x"]),
        band_of("object_center_y", raw["object_center_y"]),
        crop_phrase(raw["top_cut_frac"], raw["bot_cut_frac"]),
    )


def goal_frames(doc: dict, front_az: float) -> list[tuple[dict, dict, float]]:
    """(enriched scores, trajectory frame, subject bearing) for every usable goal frame."""
    out = []
    for pair_idx, pair in enumerate(doc.get("render_records", [])):
        for rec in pair:
            if not is_goal_frame(rec) or not rec.get("bbox_xyxy_full"):
                continue
            raw = dict(rec["scores"])
            _apply_visible_geometry(raw, int(RENDER_WIDTH), int(RENDER_HEIGHT))
            _apply_crop_extent(raw, rec["bbox_xyxy_full"], float(RENDER_HEIGHT))
            try:
                frame = doc["accepted_pairs"][pair_idx]["trajectory_32f"][rec["frame_idx"]]
            except (IndexError, KeyError):
                continue
            out.append((raw, frame, (front_az - float(raw["cam_to_obj_azimuth_deg"])) % 360.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(V12) / DEFAULT_TRAJ_ROOT))
    ap.add_argument("--val-scenes", default=DEFAULT_VAL_SCENES)
    ap.add_argument("--placements", type=int, default=80, help="held-out placements to draw from")
    ap.add_argument("--pairs-per-placement", type=int, default=60)
    ap.add_argument("--min-frames", type=int, default=6,
                    help="a placement with fewer usable goal frames says little about distance")
    ap.add_argument("--report-at", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="the cuts a rollout reports success at. Three rather than one: the "
                         "curve has no clean knee, so a single threshold would be a "
                         "presentation choice dressed up as a measurement.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    val = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
    root = Path(args.root)
    dirs = [d for d in sorted(os.listdir(root)) if d.split("__")[0] in val]
    rng = random.Random(args.seed)
    rng.shuffle(dirs)

    d_list: list[float] = []
    same_all: list[bool] = []
    n_axes: list[int] = []
    used = 0
    for name in dirs:
        if used >= args.placements:
            break
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        front = front_azimuth(obj)
        if front is None:
            continue
        path = root / name / "data.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        frames = goal_frames(doc, front)
        if len(frames) < args.min_frames:
            continue
        center = np.asarray(doc["subject_center"], dtype=np.float64)
        height = float(doc["subject_height"])
        for _ in range(args.pairs_per_placement):
            a, b = rng.sample(frames, 2)
            ga = pose_to_geometry(a[1]["pos"], a[1]["forward"], a[1]["up"],
                                  subject_center=center, subject_height=height)
            gb = pose_to_geometry(b[1]["pos"], b[1]["forward"], b[1]["up"],
                                  subject_center=center, subject_height=height)
            wa, wb = shot_words(a[0], a[2]), shot_words(b[0], b[2])
            d_list.append(float(_geometry_distance(ga, gb)))
            same_all.append(wa == wb)
            n_axes.append(sum(x == y for x, y in zip(wa, wb)))
        used += 1

    if not d_list:
        print("no pairs — is the trajectory root present?")
        return 1
    d = np.asarray(d_list)
    same = np.asarray(same_all)
    axes = np.asarray(n_axes)

    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.90, float("inf")]
    bins = []
    print(f"pairs: {len(d)} from {used} held-out placements\n")
    print(f"{'distance':>16} {'n':>6} {'same shot (6/6)':>17} {'mean axes /6':>14}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 20:
            continue
        bins.append({"lo": lo, "hi": None if hi == float("inf") else hi, "n": int(m.sum()),
                     "same_all": float(same[m].mean()), "mean_axes": float(axes[m].mean())})
        hi_s = "inf" if hi == float("inf") else f"{hi:.2f}"
        print(f"{lo:7.2f}-{hi_s:<8} {m.sum():>6} {100*same[m].mean():>16.0f}% {axes[m].mean():>13.2f}")

    # cumulative: what a success threshold would actually buy
    cuts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    cumulative = []
    print()
    for c in cuts:
        m = d < c
        if m.sum() < 20:
            continue
        cumulative.append({"tau": c, "n": int(m.sum()), "same_all": float(same[m].mean()),
                           "mean_axes": float(axes[m].mean())})
        print(f"  d < {c:.2f}: {100*same[m].mean():3.0f}% of pairs read as the SAME shot "
              f"on all six axes (n={m.sum()})")

    # One number for whoever wants one, by a rule that can be stated: the largest cut at which
    # the same shot is still the MAJORITY outcome. Reported as None when no cut reaches it,
    # rather than quietly falling back to the smallest -- a fallback that looks like a choice
    # is how a threshold ends up in a paper with nothing behind it.
    majority = [c["tau"] for c in cumulative if c["same_all"] >= 0.5]
    tau_majority = max(majority) if majority else None
    at = {c["tau"]: c["same_all"] for c in cumulative}
    print()
    for c in args.report_at:
        got = at.get(c)
        if got is None:
            print(f"  report_at {c:.2f}: not enough pairs below this distance to calibrate")
        else:
            print(f"  report_at {c:.2f}: a frame this close reads as the requested shot "
                  f"{100*got:.0f}% of the time")
    print(f"  tau_majority = {tau_majority}  (largest cut where 'same shot' is still >= 50%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_what": "Distance at which two frames of the same placement read as the SAME shot, "
                 "so a rollout can report success instead of only improvement.",
        "_rule": "No single threshold is chosen for you: `report_at` carries each cut with the "
                 "fraction of real frame pairs that read as the SAME shot at that distance, and "
                 "`tau_majority` is the largest cut where that is still the majority outcome. "
                 "The curve has no clean knee, so one number would be a presentation choice.",
        "axes": list(AXES),
        "root": str(root), "val_scenes": args.val_scenes, "seed": args.seed,
        "n_pairs": int(len(d)), "n_placements": used,
        "report_at": [{"tau": c, "same_all": at.get(c)} for c in args.report_at],
        "tau_majority": tau_majority,
        "bins": bins, "cumulative": cumulative,
    }, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
