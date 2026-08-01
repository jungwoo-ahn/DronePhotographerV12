"""Plan a yaw per placement so its well-framed goals land in UNDER-REPRESENTED view sectors.

Why: the v7 view-sector mix is severely skewed — measured over 400 placements, well-framed
goals run back 31% / back-left 26% / ... / front 4% / front-left 3% / left 1%. FRONT views,
the most useful photographic goal, are the rarest. The cause is in the data generation: every
placement uses rotation [0,0,0], so all 100 assets face the SAME world direction while the
valid camera anchors sit elsewhere.

Because bearing = (front_az + yaw) - azimuth, spinning the SUBJECT by `yaw` shifts a whole
trajectory's bearings rigidly. So we can re-render existing placements — same camera poses,
same scene, same everything else — and place their goals wherever we want:

    yaw = target_bearing + az_ref - front_az[asset]

`az_ref` is the circular mean of the cam->subject azimuth over the placement's WELL-FRAMED
frames, so the frames that will actually become goals are the ones that land on target.

Writes runs/yaw_plan.json = {placement_name: yaw_deg}, consumed by the shared repo's
scripts/v7_rerender_yaw.py --yaw-plan-path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

import numpy as np

from src.common.annotations import _is_well_framed
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS
from src.common.facing import front_azimuth, sector8

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/trajectories")
ap.add_argument("--out", default="runs/yaw_plan.json")
ap.add_argument("--n-placements", type=int, default=300, help="how many to re-render")
ap.add_argument("--targets", default="0,315,270",
                help="target bearings (deg) to fill, cycled over placements: "
                     "0=front, 315=front-left, 270=left")
ap.add_argument("--min-frames", type=int, default=4,
                help="skip placements with fewer well-framed frames than this")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

TARGETS = [float(t) for t in args.targets.split(",") if t.strip()]


def circular_mean_deg(angles: list[float]) -> float:
    r = np.radians(np.asarray(angles, dtype=np.float64))
    return float(np.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)


dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
random.seed(args.seed)
random.shuffle(dirs)

plan: dict[str, float] = {}
audit: list[dict] = []
skipped = 0
for dn in dirs:
    if len(plan) >= args.n_placements:
        break
    obj = dn.split("__", 1)[1] if "__" in dn else dn
    if obj in DEFAULT_EXCLUDE_OBJECTS:
        continue
    front = front_azimuth(obj)
    if front is None:
        continue
    path = os.path.join(args.root, dn, "data.json")
    if not os.path.exists(path):
        continue
    try:
        doc = json.load(open(path))
    except Exception:  # noqa: BLE001
        continue

    # azimuths of the frames that would qualify as goals today
    azs = [
        r["scores"]["cam_to_obj_azimuth_deg"]
        for rec in (doc.get("render_records") or [])
        for r in rec
        if r.get("scores") and _is_well_framed(r["scores"], 70.0, True)
        and 20.0 <= r["scores"]["occupancy"] <= 80.0
    ]
    if len(azs) < args.min_frames:
        skipped += 1
        continue

    az_ref = circular_mean_deg(azs)
    target = TARGETS[len(plan) % len(TARGETS)]
    yaw = (target + az_ref - front) % 360.0
    plan[dn] = round(yaw, 2)
    audit.append({
        "placement": dn, "object": obj, "front_az": front, "az_ref": round(az_ref, 1),
        "target_bearing": target, "yaw_deg": round(yaw, 2), "n_framed": len(azs),
        "bearing_now": round((front - az_ref) % 360, 1),
        "sector_now": sector8((front - az_ref) % 360), "sector_after": sector8(target),
    })

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(plan, open(args.out, "w"), indent=1)
json.dump(audit, open(args.out.replace(".json", "_audit.json"), "w"), indent=1)

print(f"planned {len(plan)} placements (skipped {skipped} with <{args.min_frames} framed frames)")
from collections import Counter  # noqa: E402
print("current sectors :", dict(Counter(a["sector_now"] for a in audit)))
print("targeted sectors:", dict(Counter(a["sector_after"] for a in audit)))
print(f"wrote {args.out}")
for a in audit[:3]:
    print(f"  e.g. {a['placement'][:44]:44s} front {a['front_az']:3.0f} az_ref {a['az_ref']:5.1f} "
          f"-> yaw {a['yaw_deg']:6.1f}  ({a['sector_now']} -> {a['sector_after']})")
