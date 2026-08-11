"""Did the yaw re-render actually move the view-sector distribution?

The problem it was meant to fix: in the original data the camera almost always
ends up BEHIND the subject. Measured over the well-framed goals, ~69% sit in the
back family and front — the most useful shot in photography — is the rarest. That
is baked into how the data was generated: every placement used identity rotation,
so all subjects face the same world direction while the valid camera anchors sit
elsewhere.

The fix is cheap because the view angle relative to the subject is

    bearing = (front_az + yaw) - azimuth

so spinning the subject shifts a whole placement's bearings rigidly. The re-render
replays the same trajectories, same cameras, same scenes, with one number changed.

This script measures whether it worked, comparing the SAME placements before and
after. That pairing matters: comparing the re-rendered 300 against the original
3931 would confound the yaw change with whatever else differs between those two
sets of scenes. Same placements, one variable.

Counts goals the way the export does — `iter_goal_start_windows` with the same
chunk size, `goal_vector` for the bearing, `sector8` for the bucket — so the
numbers here are the distribution training would actually see.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
os.chdir(V12)

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)

SECTORS = ("front", "front-right", "right", "back-right",
           "back", "back-left", "left", "front-left")
FRONT_FAMILY = ("front", "front-left", "front-right")
BACK_FAMILY = ("back", "back-left", "back-right")

ap = argparse.ArgumentParser()
ap.add_argument("--new-root", default="runs/rerender_yaw")
# Deliberately the PRE-re-render tree: this script exists to measure what the
# re-render changed, so it must read the superseded renders. Everything that
# consumes current data uses dataset_base.DEFAULT_TRAJ_ROOT.
ap.add_argument("--old-root", default="data/trajectories")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--max-per-pair", type=int, default=4)
ap.add_argument("--out", default="runs/report/yaw_distribution.json")
args = ap.parse_args()


def count_sectors(root: Path, names: list[str]) -> tuple[Counter, int, int]:
    """Sector histogram over well-framed goals for the given placements."""
    hist: Counter = Counter()
    ok = 0
    for name in names:
        path = root / name / "data.json"
        if not path.exists():
            continue
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        try:
            windows = list(iter_goal_start_windows(
                path, chunk_size=args.chunk_size, max_per_pair=args.max_per_pair))
        except Exception:  # noqa: BLE001
            continue
        counted = False
        for w in windows:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            hist[sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]))] += 1
            counted = True
        ok += int(counted)
    return hist, ok, len(names)


def pct(hist: Counter) -> dict[str, float]:
    total = sum(hist.values()) or 1
    return {s: 100.0 * hist[s] / total for s in SECTORS}


def main() -> int:
    new_root, old_root = Path(args.new_root), Path(args.old_root)
    # Only placements the re-render actually finished — a half-rendered placement
    # would contribute a truncated trajectory and bias the histogram.
    done = sorted(p.parent.name for p in new_root.glob("*/done.flag"))
    print(f"re-rendered placements with done.flag: {len(done)}")
    if not done:
        print("nothing to measure yet")
        return 1

    after, ok_a, _ = count_sectors(new_root, done)
    before, ok_b, _ = count_sectors(old_root, done)   # SAME placements, original yaw
    print(f"placements contributing goals — before {ok_b}, after {ok_a}")

    pa, pb = pct(after), pct(before)
    print(f"\n{'sector':<13}{'before':>10}{'after':>10}{'delta':>10}")
    for s in SECTORS:
        print(f"  {s:<11}{pb[s]:9.1f}%{pa[s]:9.1f}%{pa[s]-pb[s]:+9.1f}")
    fb = sum(pb[s] for s in FRONT_FAMILY)
    fa = sum(pa[s] for s in FRONT_FAMILY)
    bb = sum(pb[s] for s in BACK_FAMILY)
    ba = sum(pa[s] for s in BACK_FAMILY)
    print(f"\n  {'front family':<11}{fb:9.1f}%{fa:9.1f}%{fa-fb:+9.1f}")
    print(f"  {'back family':<11}{bb:9.1f}%{ba:9.1f}%{ba-bb:+9.1f}")
    print(f"\n  goals counted — before {sum(before.values())}, after {sum(after.values())}")

    # Evenness: how far the histogram is from uniform. One number for "is the
    # distribution less lopsided", which the per-sector deltas alone do not answer.
    def gini(p: dict[str, float]) -> float:
        v = sorted(p[s] for s in SECTORS)
        n = len(v)
        s = sum(v) or 1.0
        return (2 * sum((i + 1) * x for i, x in enumerate(v)) / (n * s) - (n + 1) / n)

    print(f"  gini (0 = perfectly even) — before {gini(pb):.3f}, after {gini(pa):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "placements": len(done),
        "before": {"counts": dict(before), "pct": pb, "gini": gini(pb)},
        "after": {"counts": dict(after), "pct": pa, "gini": gini(pa)},
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
