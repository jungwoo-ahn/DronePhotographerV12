"""Export `goal_start` samples from the v7 trajectories into a LeRobot dataset.

    python scripts/export_lerobot.py --max-episodes 4000 --out runs/lerobot_v1

Samples are drawn round-robin across placements rather than by exhausting one at a time,
so a capped export still spans many scenes/objects (and, once the yaw re-render lands, many
view sectors) instead of being dominated by whichever placements come first alphabetically.
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

sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

import numpy as np

from src.common.annotations import iter_goal_start_windows
from src.common.dataset_base import (
    DEFAULT_EXCLUDE_OBJECTS,
    _compute_action_chunk,
    _window_object,
)
from src.common.facing import sector8
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector
from src.data.lerobot_export import EpisodeSpec, goal_prompt, write_lerobot_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=["data/trajectories"],
                help="trajectory dirs; add runs/rerender_yaw once the re-render lands")
ap.add_argument("--out", default="runs/lerobot_v1")
ap.add_argument("--max-episodes", type=int, default=4000)
ap.add_argument("--per-placement", type=int, default=4,
                help="samples taken per placement per round-robin pass")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--resize", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

placements: list[tuple[str, Path]] = []
for root in args.roots:
    r = Path(root)
    if not r.is_dir():
        print(f"  skip missing root {r}")
        continue
    for d in sorted(os.listdir(r)):
        p = r / d / "data.json"
        if p.exists():
            placements.append((d, p))
random.seed(args.seed)
random.shuffle(placements)
print(f"placements: {len(placements)} across {len(args.roots)} root(s)", flush=True)

episodes: list[EpisodeSpec] = []
sectors: Counter = Counter()
objects: Counter = Counter()
t0 = time.time()
for i, (name, path) in enumerate(placements):
    if len(episodes) >= args.max_episodes:
        break
    obj = name.split("__", 1)[1] if "__" in name else name
    if obj in DEFAULT_EXCLUDE_OBJECTS:
        continue
    try:
        windows = list(iter_goal_start_windows(
            path, chunk_size=args.chunk_size, max_per_pair=args.per_placement,
        ))
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {name[:40]}: {exc}")
        continue
    random.shuffle(windows)
    taken = 0
    for w in windows:
        if taken >= args.per_placement or len(episodes) >= args.max_episodes:
            break
        g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
        if not np.isfinite(g).all():
            continue
        frames = [Path(k.image) for k in w.keyframes]
        if not all(f.exists() for f in frames):
            continue
        episodes.append(EpisodeSpec(
            frame_paths=frames, actions=_compute_action_chunk(w), prompt=goal_prompt(g),
        ))
        sectors[sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]))] += 1
        objects[obj] += 1
        taken += 1
    if (i + 1) % 200 == 0:
        print(f"  ...{i+1} placements scanned, {len(episodes)} episodes, "
              f"{time.time()-t0:.0f}s", flush=True)

print(f"\nepisodes: {len(episodes)}  objects: {len(objects)}  ({time.time()-t0:.0f}s)")
tot = sum(sectors.values()) or 1
print("sector mix:", {k: f"{100*v/tot:.0f}%" for k, v in sectors.most_common()})

out = write_lerobot_dataset(
    episodes, args.out, fps=args.fps, overwrite=True, resize=args.resize,
)
info = json.load(open(Path(out) / "meta/info.json"))
mp4 = Path(out) / f"videos/observation.images.image/chunk-000/file-000.mp4"
print(f"\nwrote {out}: {info['total_episodes']} episodes / {info['total_frames']} frames / "
      f"{info['total_tasks']} tasks, mp4 {mp4.stat().st_size // (1024*1024)} MB "
      f"({time.time()-t0:.0f}s total)")
