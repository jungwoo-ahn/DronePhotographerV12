"""Build the shared closed-loop episode manifest, so two repos can evaluate the SAME shots.

The baselines (pi0.5 / GR00T / Diffusion Policy) live in another repo with its own port of this
protocol. The port is faithful — same `reward.py`, same horizon rule, same shoot-stop semantics —
but the two harnesses pick their episodes differently, and the score is dominated by how far the
start is: over 138 held-out rollouts `corr(d_start, improvement) = 0.66` with slope 0.82. Two
policies measured on two episode sets are not comparable no matter how identical the protocol,
and matching our episodes to the baselines' d_start band moved our own mean improvement from
0.549 to 0.270. This file removes that whole class of difference.

Selection: the eight held-out scenes of `configs/val_scenes.json`, balanced over the eight view
sectors (the pool is front-heavy — a 138-episode draw came out front 37 / front-right 32 /
left 6), spread over scenes and one episode per placement. Sector balancing follows
`scripts/export_lerobot.py:fill()`, including saying so out loud when a sector cannot fill: a
silent shortfall is balance that was never achieved.

Each episode is MATERIALISED — poses, subject geometry and the goal's enriched scores — so the
other repo replays the shot without having to reproduce our window enumeration. It builds its own
conditioning from `goal_raw` (they serialise a prompt, DP normalises a vector), which is why the
manifest carries the inputs and not a prompt.

`d_start` is the checksum. It is pure pose math from the materialised poses, so both repos
recompute it and refuse to run on a mismatch — the failure is loud, before any GPU is spent,
rather than a quiet difference discovered in the final table.

    python scripts/build_eval_episodes.py --per-sector 8 --out configs/eval_episodes_v1.json

venv: .venv-analysis (CPU, a few minutes).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np  # noqa: E402

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.dataset_base import (DEFAULT_EXCLUDE_OBJECTS,  # noqa: E402
                                     DEFAULT_TRAJ_ROOT, _window_object)
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import (DEFAULT_GOAL_KEYS,  # noqa: E402
                                   SUBJECT_BEARING_KEY, goal_vector)
from src.common.reward import _geometry_distance, pose_to_geometry  # noqa: E402

DEFAULT_VAL_SCENES = str(Path(V12) / "configs" / "val_scenes.json")
DEFAULT_OUT = str(Path(V12) / "configs" / "eval_episodes_v1.json")

SECTOR_ORDER = ("front", "front-right", "right", "back-right",
                "back", "back-left", "left", "front-left")

# The window pool depends on BOTH of these, not just on which window is then chosen, so they are
# part of the manifest rather than of whoever runs it. `max_per_pair=2` is what
# `closed_loop_eval.pick_episodes` has always used; the baselines' harness uses the uncapped pool,
# which is why a shared list of placement names would not have been enough.
CHUNK_SIZE = 8
MAX_PER_PAIR = 2


def episode_key(placement: str, pair_idx: int, start_idx: int, goal_idx: int) -> str:
    """Stable join key across repos and runs; also the resume key (real2sim_rollout.py's pattern,
    which is cleaner than keying on a triple of absolute image paths)."""
    return f"{placement}:p{pair_idx:02d}:s{start_idx:02d}:g{goal_idx:02d}"


def start_distance(window) -> float:
    """`closed_loop_eval.geometry_distance` at the start pose — the manifest's checksum."""
    gv = window.goal_frame
    achieved = pose_to_geometry(window.start.camera_position, window.start.camera_forward,
                                window.start.camera_up, subject_center=gv.subject_center,
                                subject_height=gv.subject_height)
    goal = pose_to_geometry(gv.camera_position, gv.camera_forward, gv.camera_up,
                            subject_center=gv.subject_center, subject_height=gv.subject_height)
    return float(_geometry_distance(achieved, goal))


def materialise(placement: str, path: Path, window, goal_vec, bearing: float, sector: str) -> dict:
    gv = window.goal_frame
    delta = abs(gv.frame_idx - window.start_frame_idx)
    return {
        "key": episode_key(placement, window.pair_idx, window.start_frame_idx, gv.frame_idx),
        "placement": placement,
        "scene": placement.split("__")[0],
        "object": _window_object(window),
        "data_json": str(path),
        "pair_idx": int(window.pair_idx),
        "start_frame_idx": int(window.start_frame_idx),
        "goal_frame_idx": int(gv.frame_idx),
        "start_pose": {"position": [float(v) for v in window.start.camera_position],
                       "forward": [float(v) for v in window.start.camera_forward],
                       "up": [float(v) for v in window.start.camera_up]},
        "goal_pose": {"position": [float(v) for v in gv.camera_position],
                      "forward": [float(v) for v in gv.camera_forward],
                      "up": [float(v) for v in gv.camera_up]},
        "subject_center": [float(v) for v in gv.subject_center],
        "subject_height": float(gv.subject_height),
        # The goal's scores as the loader enriched them (visible-bbox geometry + crop fractions).
        # Each repo builds its own conditioning from this: ours a prompt via
        # src/goal_authoring/serialize.goal_prompt, theirs the same through their goal_text, DP a
        # normalised vector. Sharing the inputs keeps the manifest from taking sides.
        "goal_raw": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in gv.raw.items() if not isinstance(v, (list, dict))},
        "goal_vec": [float(x) for x in goal_vec],
        "goal_keys": list(DEFAULT_GOAL_KEYS),
        "start_image": window.start.image,
        "goal_image": gv.image,
        "delta": int(delta),
        "n_chunks": int(np.ceil(delta / CHUNK_SIZE)),
        "sector": sector,
        "bearing": float(bearing),
        "d_start": start_distance(window),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(V12) / DEFAULT_TRAJ_ROOT))
    # Absolute, never a relative string literal: scripts that talk to Cosmos chdir into the
    # vendored framework, and a relative manifest default has already killed a training run and
    # then three rollouts at once. tests/test_shoot_and_scene_split.py scans for this.
    ap.add_argument("--val-scenes", default=DEFAULT_VAL_SCENES)
    ap.add_argument("--per-sector", type=int, default=8,
                    help="episodes per view sector; 8 x 8 sectors = 64")
    ap.add_argument("--per-scene", type=int, default=12,
                    help="cap per scene, so the set cannot collapse into one room the way an "
                         "alphabetical first-N does")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    val = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
    root = Path(args.root)
    dirs = [d for d in sorted(os.listdir(root))
            if d.split("__")[0] in val and (root / d / "data.json").exists()]
    rng = random.Random(args.seed)
    rng.shuffle(dirs)
    print(f"held-out placements: {len(dirs)} across {len({d.split('__')[0] for d in dirs})} scenes",
          flush=True)

    episodes: list[dict] = []
    per_sector: Counter = Counter()
    per_scene: Counter = Counter()
    t0 = time.time()
    for i, name in enumerate(dirs):
        if all(per_sector[s] >= args.per_sector for s in SECTOR_ORDER):
            break
        scene = name.split("__")[0]
        if per_scene[scene] >= args.per_scene:
            continue
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        path = root / name / "data.json"
        try:
            windows = list(iter_goal_start_windows(path, chunk_size=CHUNK_SIZE,
                                                   max_per_pair=MAX_PER_PAIR))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {name[:44]}: {exc}")
            continue
        for w in windows:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            bearing = float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
            sec = sector8(bearing)
            if per_sector[sec] >= args.per_sector:
                continue                      # this sector is full; try another window
            if not (Path(w.start.image).exists() and Path(w.goal_frame.image).exists()):
                continue                      # a policy has to be shown the start frame
            episodes.append(materialise(name, path, w, g, bearing, sec))
            per_sector[sec] += 1
            per_scene[scene] += 1
            break                             # one episode per placement
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(dirs)} placements scanned, {len(episodes)} episodes "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if not episodes:
        print("no episodes selected")
        return 1

    short = {s: per_sector[s] for s in SECTOR_ORDER if per_sector[s] < args.per_sector}
    print(f"\nepisodes {len(episodes)}  ({time.time()-t0:.0f}s)")
    print("sector mix:", {s: per_sector[s] for s in SECTOR_ORDER})
    print("scene mix :", dict(per_scene))
    if short:
        print(f"UNDER TARGET {args.per_sector}: {short}  — a real limit of the held-out data, "
              "not a bug; the manifest records it")
    ds = np.array([e["d_start"] for e in episodes])
    print(f"d_start: mean {ds.mean():.3f}  median {np.median(ds):.3f}  "
          f"range [{ds.min():.3f}, {ds.max():.3f}]")
    print(f"delta: {min(e['delta'] for e in episodes)}-{max(e['delta'] for e in episodes)}  "
          f"chunks: {Counter(e['n_chunks'] for e in episodes)}")

    doc = {
        "_what": "The shared closed-loop episode set. Every policy compared in one table must run "
                 "exactly these, because the score is dominated by how far the start is "
                 "(corr(d_start, improvement) = 0.66 over 138 held-out rollouts).",
        "_contract": "A MANIFEST, not a rule. Regenerating this file redefines the comparison and "
                     "makes earlier numbers incomparable — add a new version instead of editing "
                     "this one.",
        "_verify": "Recompute d_start from start_pose/goal_pose/subject_center/subject_height via "
                   "reward.pose_to_geometry + _geometry_distance and refuse to run if it differs "
                   "by more than 1e-6. That check is what proves two repos ran the same shots.",
        "version": 1,
        "source_root": str(root),
        "val_scenes": args.val_scenes,
        "scenes": sorted({e["scene"] for e in episodes}),
        "chunk_size": CHUNK_SIZE,
        "max_per_pair": MAX_PER_PAIR,
        "seed": args.seed,
        "per_sector_target": args.per_sector,
        "per_scene_cap": args.per_scene,
        "sector_mix": {s: per_sector[s] for s in SECTOR_ORDER},
        "scene_mix": dict(per_scene),
        "under_target": short,
        "n_episodes": len(episodes),
        "mean_d_start": float(ds.mean()),
        "episodes": episodes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
