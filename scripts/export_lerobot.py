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
    shoot_column,
)
from src.common.facing import sector8
from src.data.cosmos_camera_dataset import DEFAULT_VAL_SCENES

SECTOR_ORDER = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector
from src.data.lerobot_export import EpisodeSpec, goal_prompt, write_lerobot_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+",
                default=["data/trajectories/v7_stage2_renders_lookat075"],
                help="trajectory dirs; add runs/rerender_yaw once the re-render lands")
ap.add_argument("--out", default="runs/lerobot_v1")
ap.add_argument("--max-episodes", type=int, default=4000)
ap.add_argument("--per-placement", type=int, default=4,
                help="samples taken per placement per round-robin pass")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--resize", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--balance-sectors", type=int, default=1,
                help="fill each of the 8 view sectors to max_episodes/8 (1 = on). Off "
                     "reproduces the pool's ~36%% back / ~4%% left skew at any size.")
ap.add_argument("--episodes-per-video", type=int, default=20000,
                help="episodes per mp4. One file for everything makes each episode's "
                     "from_timestamp seek scale with total size — a 150k single-file "
                     "export halved training throughput.")
ap.add_argument("--val-scenes", default=DEFAULT_VAL_SCENES,
                help="manifest of scenes held out as val. Placements are partitioned by scene "
                     "BEFORE the fill and each side is balanced separately — filtering a "
                     "globally-balanced draw afterwards would re-skew both sides by whatever "
                     "those scenes happened to contain. Pass '' to export one undivided set.")
ap.add_argument("--val-fraction", type=float, default=0.10,
                help="share of --max-episodes spent on the val scenes")
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

# --- split the PLACEMENTS by scene before drawing anything ------------------------
# The alternative (draw globally, filter afterwards) cannot work: the draw is
# sector-balanced, and removing whichever scenes are val re-skews both sides by whatever
# those scenes happened to contain. Partition first, balance each side on its own.
val_scenes: frozenset[str] = frozenset()
if args.val_scenes:
    _m = json.loads(Path(args.val_scenes).read_text())
    val_scenes = frozenset(_m["scenes"])
    print(f"val scenes: {len(val_scenes)} held out whole (from {args.val_scenes})")

split_of = {name: ("val" if name.split("__")[0] in val_scenes else "train")
            for name, _ in placements}
n_val_pl = sum(1 for v in split_of.values() if v == "val")
print(f"  train placements {len(placements)-n_val_pl}  |  val placements {n_val_pl}")

near_count = 0
crop_counts: Counter = Counter()
t0 = time.time()


def fill(split: str, budget: int) -> tuple[list[EpisodeSpec], Counter, Counter]:
    """Draw up to `budget` sector-balanced episodes from this split's placements.

    Sector-balanced fill. The pool is ~36% back and ~4% left, so taking episodes in
    placement order reproduces that skew no matter how many are taken. Every sector has
    far more than budget/8 goals available (checked: the scarcest, `left`, has ~3100 goals
    across ~470 placements), so an even draw is feasible without inventing data — it just
    needs scanning to continue past the point where `back` is already full.
    """
    global near_count
    out: list[EpisodeSpec] = []
    sectors: Counter = Counter()
    objects: Counter = Counter()
    cap = (budget // 8) if args.balance_sectors else budget
    mine = [(n, p) for n, p in placements if split_of[n] == split]

    def _full() -> bool:
        if not args.balance_sectors:
            return len(out) >= budget
        return all(sectors[s] >= cap for s in SECTOR_ORDER)

    for i, (name, path) in enumerate(mine):
        if _full() or len(out) >= budget:
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
            if taken >= args.per_placement or len(out) >= budget:
                break
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            sec = sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]))
            if args.balance_sectors and sectors[sec] >= cap:
                continue                # this sector is full; keep the window for nobody
            frames = [Path(k.image) for k in w.keyframes]
            if not all(f.exists() for f in frames):
                continue
            # 9 pose dims + the shoot channel. Concatenated HERE rather than inside
            # `_compute_action_chunk`, so that function stays pose-only for its other
            # callers (fit_action_stats, the v11 path) and `encode_action_9d` keeps its
            # meaning — pose dims 0-8 remain comparable with the v4 run.
            pose = _compute_action_chunk(w)
            act = np.concatenate([pose, shoot_column(w)[:, None]], axis=1)
            out.append(EpisodeSpec(
                frame_paths=frames, actions=act,
                # crop comes from the goal frame's signed bbox, not the goal vector
                prompt=goal_prompt(g, crop=w.goal_frame.raw),
                placement=name,
            ))
            sectors[sec] += 1
            if abs(w.goal_frame.frame_idx - w.start_frame_idx) < args.chunk_size:
                near_count += 1
            _r = w.goal_frame.raw
            _t, _b = _r.get('top_cut_frac', 0.0) > 0.02, _r.get('bot_cut_frac', 0.0) > 0.02
            crop_counts['both' if (_t and _b) else 'top' if _t else 'bottom' if _b else 'none'] += 1
            objects[obj] += 1
            taken += 1
        if (i + 1) % 200 == 0:
            print(f"  [{split}] ...{i+1}/{len(mine)} placements scanned, {len(out)} episodes, "
                  f"{time.time()-t0:.0f}s", flush=True)

    t = sum(sectors.values()) or 1
    print(f"\n[{split}] episodes {len(out)}  objects {len(objects)}  ({time.time()-t0:.0f}s)")
    print(f"[{split}] sector mix:", {s_: f"{100*sectors[s_]/t:.1f}%" for s_ in SECTOR_ORDER})
    if args.balance_sectors:
        short = {s_: sectors[s_] for s_ in SECTOR_ORDER if sectors[s_] < cap}
        print(f"[{split}] per-sector target {cap}: "
              + ("all sectors filled" if not short else f"UNDER TARGET {short}"))
        # Said out loud on purpose: a sector that could not fill is a real limit of the
        # data, and a silent shortfall would read as balance that was never achieved.
    return out, sectors, objects


n_val_ep = int(args.max_episodes * args.val_fraction) if val_scenes else 0
val_eps, _, _ = fill("val", n_val_ep) if n_val_ep else ([], Counter(), Counter())
train_eps, sectors, objects = fill("train", args.max_episodes - len(val_eps))

# Train first, then val. Order no longer defines the split — `scene` does — but keeping
# each split contiguous keeps the mp4 shards from interleaving the two.
episodes: list[EpisodeSpec] = train_eps + val_eps
print(f"\ntotal episodes: {len(episodes)}  (train {len(train_eps)}, val {len(val_eps)})")
_tr_sc = {e.scene for e in train_eps}
_va_sc = {e.scene for e in val_eps}
assert not (_tr_sc & _va_sc), f"scene leaked across the split: {sorted(_tr_sc & _va_sc)[:5]}"
print(f"scene disjoint: train {len(_tr_sc)} scenes, val {len(_va_sc)} scenes, overlap 0")
print(f"near-goal episodes (delta < {args.chunk_size}): "
      f"{near_count}/{len(episodes)} = {100*near_count/max(1,len(episodes)):.0f}%")
# Crop mix, so a gate regression is visible HERE rather than after training: the
# old gate silently made 72.7% of goals head-cropped and nobody could see it.
_ct = sum(crop_counts.values()) or 1
print("crop mix: " + "  ".join(f"{k} {100*crop_counts[k]/_ct:.1f}%"
                               for k in ("none", "bottom", "top", "both")))

out = write_lerobot_dataset(
    episodes, args.out, fps=args.fps, overwrite=True, resize=args.resize,
    episodes_per_video=args.episodes_per_video,
)
info = json.load(open(Path(out) / "meta/info.json"))
mp4 = Path(out) / f"videos/observation.images.image/chunk-000/file-000.mp4"
print(f"\nwrote {out}: {info['total_episodes']} episodes / {info['total_frames']} frames / "
      f"{info['total_tasks']} tasks, mp4 {mp4.stat().st_size // (1024*1024)} MB "
      f"({time.time()-t0:.0f}s total)")
