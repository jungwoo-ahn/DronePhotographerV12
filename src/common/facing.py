"""Per-asset subject facing: world-frame azimuth -> SUBJECT-frame bearing.

The stored `cam_to_obj_azimuth_deg` is a WORLD-frame angle (`src/scoring/projection.py`
:`cam_to_subject_angles`, `atan2(dy, dx)` of cam->subject). As a goal it is ambiguous:
the same number means "facing the camera" for one asset and "back turned" for another,
and nothing in the image tells the policy where the world frame points.

Each asset has its own baked canonical facing, recovered once (isolated turntable
renders + face detection + human verification) into `runs/facing_map_final.json`:

    {"<object key>": {"front_az": <deg>, "facing_world_deg": <deg>}, ...}

`front_az` is the camera azimuth from which the subject's FRONT is seen; the subject
looks toward `facing_world_deg == front_az + 180`.

The subject-frame bearing is

    bearing = (front_az - azimuth) mod 360

so 0 = seen from the front, 90 = from the subject's RIGHT, 180 = from behind, 270 =
from the subject's LEFT. (Sign verified geometrically and on renders: at bearing 90
the camera's right axis aligns with the facing direction, i.e. the subject appears
facing image-right.) This is scene- and asset-agnostic and readable straight off the
image, which is what makes it usable as a goal.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACING_MAP_PATH = REPO_ROOT / "runs" / "facing_map_final.json"


@lru_cache(maxsize=8)
def load_facing_map(path: str | Path | None = None) -> Mapping[str, dict]:
    """Load (and cache) the per-object facing map."""
    p = Path(path) if path is not None else DEFAULT_FACING_MAP_PATH
    with open(p) as fh:
        return json.load(fh)


def front_azimuth(object_key: str, path: str | Path | None = None) -> float | None:
    """Camera azimuth (deg) from which `object_key`'s front is seen, or None if unmapped."""
    entry = load_facing_map(path).get(object_key)
    if not entry or entry.get("front_az") is None:
        return None
    return float(entry["front_az"])


def subject_bearing_deg(
    world_azimuth_deg: float, object_key: str, path: str | Path | None = None,
    *, yaw_deg: float = 0.0,
) -> float | None:
    """World-frame `cam_to_obj_azimuth_deg` -> subject-frame bearing in [0, 360).

    `yaw_deg` is the placement's subject rotation (`placement_yaw_deg`), and it is
    applied HERE rather than to the azimuth. Spinning the subject does not move the
    camera, so the world azimuth is identical before and after a yaw re-render —
    verified on real data: a placement re-rendered at yaw 310.86 deg scored the same
    azimuth (26 deg) as the original. What the spin changes is where the subject's
    front points:

        effective_front_az = front_az[asset] + placement_yaw_deg

    Omitting it makes a yaw re-render a no-op in goal space: the frames show a rotated
    subject while every bearing stays exactly as it was.

    Returns None when the object has no facing entry, so callers can drop the sample
    rather than silently fall back to the (ambiguous) world angle.
    """
    front = front_azimuth(object_key, path)
    if front is None or world_azimuth_deg is None or not math.isfinite(world_azimuth_deg):
        return None
    return (front + float(yaw_deg) - float(world_azimuth_deg)) % 360.0


def world_azimuth_deg(
    bearing_deg: float, object_key: str, path: str | Path | None = None,
    *, yaw_deg: float = 0.0,
) -> float | None:
    """Inverse of `subject_bearing_deg` — needed to score a bearing goal against a
    world-frame achieved profile (eval / rollout).

    Takes the same `yaw_deg`; a one-sided fix would silently break round-tripping on
    re-rendered placements, which is where the eval scores its goals.
    """
    front = front_azimuth(object_key, path)
    if front is None or bearing_deg is None or not math.isfinite(bearing_deg):
        return None
    return (front + float(yaw_deg) - float(bearing_deg)) % 360.0


SECTOR8 = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)


def sector8(bearing_deg: float) -> str:
    """8-way view word for a subject-frame bearing (45-deg bins centred on the labels)."""
    return SECTOR8[int(((float(bearing_deg) + 22.5) % 360.0) // 45.0)]


def sector3(bearing_deg: float) -> str:
    """front / side / back — magnitude only, so it is immune to a mirrored asset."""
    off = abs(((float(bearing_deg) % 360.0) + 180.0) % 360.0 - 180.0)
    return "front" if off < 45.0 else ("back" if off > 135.0 else "side")


__all__ = [
    "DEFAULT_FACING_MAP_PATH",
    "load_facing_map",
    "front_azimuth",
    "subject_bearing_deg",
    "world_azimuth_deg",
    "sector8",
    "sector3",
    "SECTOR8",
]
