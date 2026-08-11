"""Measure the 9D action + goal distributions under the `goal_start` scheme.

Produces the normalization constants for `src/common/action_repr.py` (the in-tree
ACTION_SCALE / ACTION_STD are stale: they were fit for the 5D multiscale scheme).

NORMALIZATION POLICY (matches Cosmos): only the 3 TRANSLATION dims are scaled.
The 6 rot6d dims are two columns of a rotation matrix — already bounded in [-1, 1],
and NOT zero-centred (a small relative rotation sits near the identity, so dims 3 and
7 hover near 1 while the rest hover near 0). Dividing those by a p99 of |x| would be
meaningless. Cosmos ships `skip_rotation_dims: [3..8]` for its own rot6d stats; we do
the same and report the rotation spread as a geodesic angle instead.

Usage (NAS is slow — run in background):
    python scripts/fit_action_stats.py --max-placements 150 --max-per-pair 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

import numpy as np

from src.common.annotations import iter_goal_start_windows
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _compute_action_chunk
from src.common.facing import sector8
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector
from src.utils.rotation_utils import matrix_from_rot6d
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=DEFAULT_TRAJ_ROOT)
ap.add_argument("--max-placements", type=int, default=150)
ap.add_argument("--max-per-pair", type=int, default=12)
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--out", default="runs/action_stats_9d.json")
args = ap.parse_args()

DIMS = ["d_right(m)", "d_up(m)", "d_fwd(m)",
        "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5"]

dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
random.seed(0)
random.shuffle(dirs)

actions: list[np.ndarray] = []
goals: list[np.ndarray] = []
scanned = 0
t0 = time.time()
for dn in dirs:
    if scanned >= args.max_placements:
        break
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if obj in DEFAULT_EXCLUDE_OBJECTS:
        continue
    path = os.path.join(args.root, dn, "data.json")
    if not os.path.exists(path):
        continue
    scanned += 1
    for w in iter_goal_start_windows(
        path, chunk_size=args.chunk_size, max_per_pair=args.max_per_pair
    ):
        actions.append(_compute_action_chunk(w))
        g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=obj)
        if np.isfinite(g).all():
            goals.append(g)
    if scanned % 25 == 0:
        print(f"...{scanned}/{args.max_placements} placements, "
              f"{len(actions)} chunks, {time.time()-t0:.0f}s", flush=True)

A = np.concatenate(actions, axis=0).astype(np.float64)      # (N, 9) per-STEP actions
G = np.stack(goals).astype(np.float64)                      # (M, 8)
print(f"\nper-step actions: {A.shape}   goals: {G.shape}   ({time.time()-t0:.0f}s)\n")

print("=== 9D ACTION per-dim ===")
print(f"{'dim':<12}{'mean':>9}{'std':>9}{'p1':>9}{'p50':>9}{'p99':>9}{'p99|x|':>9}{'max|x|':>9}")
for i, name in enumerate(DIMS):
    c = A[:, i]
    print(f"{name:<12}{c.mean():9.4f}{c.std():9.4f}{np.percentile(c,1):9.4f}"
          f"{np.percentile(c,50):9.4f}{np.percentile(c,99):9.4f}"
          f"{np.percentile(np.abs(c),99):9.4f}{np.abs(c).max():9.4f}")

trans_scale = np.percentile(np.abs(A[:, :3]), 99, axis=0)
trans_std = A[:, :3].std(axis=0)
print(f"\nTRANSLATION_SCALE (p99|x|) = [{', '.join(f'{v:.3f}' for v in trans_scale)}]")
print(f"TRANSLATION_STD            = [{', '.join(f'{v:.3f}' for v in trans_std)}]")
print("rot6d dims 3-8: NOT normalized (Cosmos skip_rotation_dims convention)")

# rotation magnitude as a geodesic angle — the interpretable check on rot6d spread
sample = A[np.random.default_rng(0).choice(len(A), size=min(20000, len(A)), replace=False)]
angles = []
for row in sample:
    rot = matrix_from_rot6d(row[3:])
    angles.append(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
angles = np.array(angles)
print(f"\n=== per-step relative ROTATION (geodesic deg) ===")
print(f"  mean={angles.mean():.2f}  p50={np.median(angles):.2f}  "
      f"p90={np.percentile(angles,90):.2f}  p99={np.percentile(angles,99):.2f}  max={angles.max():.2f}")
ident = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
print(f"  mean rot6d = [{', '.join(f'{v:+.3f}' for v in A[:,3:].mean(axis=0))}]  "
      f"(identity = [{', '.join(f'{v:+.0f}' for v in ident)}])")

print("\n=== GOAL distribution (new scheme) ===")
for i, k in enumerate(DEFAULT_GOAL_KEYS):
    c = G[:, i]
    print(f"  {k:<26} p1={np.percentile(c,1):8.1f} p50={np.percentile(c,50):8.1f} "
          f"p99={np.percentile(c,99):8.1f}")
bi = DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)
from collections import Counter
sec = Counter(sector8(b) for b in G[:, bi])
tot = sum(sec.values())
print("  bearing sector coverage: " +
      "  ".join(f"{k} {100*v/tot:.0f}%" for k, v in sorted(sec.items(), key=lambda x: -x[1])))

out = {
    "n_steps": int(A.shape[0]), "n_goals": int(G.shape[0]), "placements": scanned,
    "translation_scale_p99": [round(float(v), 4) for v in trans_scale],
    "translation_std": [round(float(v), 4) for v in trans_std],
    "rot6d_mean": [round(float(v), 4) for v in A[:, 3:].mean(axis=0)],
    "rot6d_normalized": False,
    "rotation_deg": {"mean": float(angles.mean()), "p50": float(np.median(angles)),
                     "p99": float(np.percentile(angles, 99)), "max": float(angles.max())},
    "goal_percentiles": {k: [float(np.percentile(G[:, i], q)) for q in (1, 50, 99)]
                         for i, k in enumerate(DEFAULT_GOAL_KEYS)},
    "bearing_sector_pct": {k: round(100 * v / tot, 1) for k, v in sec.items()},
}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
