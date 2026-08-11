"""Plan yaws so the COMBINED pool comes out even — not so each placement hits a target.

The v1 planner cycled three fixed targets (front / front-left / left) across
placements. It worked in the narrow sense — those three sectors went from 4.7 / 2.5 /
1.3 % to 22.6 / 23.0 / 20.3 % within the re-rendered set — but it optimised the wrong
objective. What training sees is the *union* of the original 3931 placements and the
re-rendered ones, and at 300 placements the re-render is only ~8% of that union, so
the combined mix barely moved (front 4.3 -> 5.7 %, gini 0.436 -> 0.380).

This planner optimises the union directly:

  * the ~3931 original placements are a fixed, back-heavy background;
  * the 300 already re-rendered are fixed too (their frames exist);
  * every additional placement is chosen together with the yaw that most reduces the
    union's distance from a uniform 8-sector mix.

The per-placement effect is computed EXACTLY rather than assumed. Because
``bearing = (front_az + yaw) - azimuth``, a yaw shifts a placement's whole bearing set
rigidly, so for a candidate yaw we can bin its actual well-framed azimuths and get the
histogram it would contribute. No model of "aiming mostly works" is needed.

Greedy is appropriate here: each placement contributes a fixed histogram once chosen,
the objective is separable across sectors, and after each pick the deficit is
recomputed — so the selection naturally spreads across whichever sectors are still
short instead of piling onto one.

Writes runs/yaw_plan_v2.json = {placement: yaw_deg} for the NEW placements only, plus
runs/report/yaw_plan_v2_audit.json with the projected mix at several N.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

V12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V12))
os.chdir(V12)

from src.common.annotations import _apply_crop_extent, _is_well_framed  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS  # noqa: E402
from src.common.facing import front_azimuth, sector8  # noqa: E402

SECTORS = ("front", "front-right", "right", "back-right",
           "back", "back-left", "left", "front-left")
SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}


def _framed_raw(record: dict) -> dict:
    """scores + the crop keys `_is_well_framed` actually gates on.

    The on-disk `scores` dict has no `visible_frac` — only the signed `bbox_xyxy_full` sitting
    beside it does — so passing `scores` straight in silently fell back to an area ratio.
    """
    raw = dict(record.get("scores") or {})
    bbox = record.get("bbox_xyxy_full")
    if bbox:
        _apply_crop_extent(raw, bbox, float(RENDER_HEIGHT))
    return raw


ap = argparse.ArgumentParser()
# Deliberately the PRE-re-render tree: this script exists to measure what the
# re-render changed, so it must read the superseded renders. Everything that
# consumes current data uses dataset_base.DEFAULT_TRAJ_ROOT.
ap.add_argument("--root", default="data/trajectories")
ap.add_argument("--done-root", default="runs/rerender_yaw")
ap.add_argument("--candidates", type=int, default=1500,
                help="placements to scan as candidates (sampled from the un-re-rendered rest)")
ap.add_argument("--background-sample", type=int, default=700,
                help="placements sampled to estimate the fixed original distribution")
ap.add_argument("--max-n", type=int, default=900, help="largest plan size to evaluate")
ap.add_argument("--n", type=int, default=0,
                help="plan size to write (0 = pick where the gain flattens)")
ap.add_argument("--yaw-step", type=float, default=10.0, help="yaw grid in degrees")
ap.add_argument("--min-occ", type=float, default=20.0)
ap.add_argument("--max-occ", type=float, default=80.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/yaw_plan_v2.json")
ap.add_argument("--audit", default="runs/report/yaw_plan_v2_audit.json")
args = ap.parse_args()


def placement_azimuths(data_path: Path) -> tuple[list[float], str] | None:
    """Well-framed cam->subject azimuths for one placement, and its object key.

    Only frames that could actually BECOME goals are counted — the same
    well-framed + occupancy gate the dataset applies — so the projected histogram
    describes the goals training would see, not every rendered frame.
    """
    try:
        doc = json.loads(data_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    recs = doc.get("render_records") or []
    azs: list[float] = []
    for pair in recs:
        for r in (pair if isinstance(pair, list) else [pair]):
            sc = r.get("scores")
            if not sc:
                continue
            occ = sc.get("occupancy")
            if occ is None or not (args.min_occ <= float(occ) <= args.max_occ):
                continue
            if not _is_well_framed(_framed_raw(r), 0.35):
                continue
            az = sc.get("cam_to_obj_azimuth_deg")
            if az is not None and math.isfinite(float(az)):
                azs.append(float(az))
    if not azs:
        return None
    name = data_path.parent.name
    return azs, (name.split("__", 1)[1] if "__" in name else name)


def hist_for_yaws(azs: list[float], front: float, yaws: np.ndarray) -> np.ndarray:
    """(len(yaws), 8) histogram of this placement's goals at each candidate yaw."""
    a = np.asarray(azs, dtype=np.float64)
    # bearing = (front + yaw - az) mod 360, binned by sector8's 45-deg buckets
    bearings = (front + yaws[:, None] - a[None, :]) % 360.0
    idx = (((bearings + 22.5) % 360.0) // 45.0).astype(int)
    out = np.zeros((len(yaws), 8), dtype=np.float64)
    for i in range(len(yaws)):
        np.add.at(out[i], idx[i], 1.0)
    return out


def l1_to_uniform(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 2.0
    return float(np.abs(counts / total - 1.0 / 8).sum())


def gini(counts: np.ndarray) -> float:
    p = counts / (counts.sum() or 1.0)
    v = np.sort(p)
    n = len(v)
    return float(2 * np.sum((np.arange(1, n + 1)) * v) / (n * v.sum()) - (n + 1) / n)


def main() -> int:
    root, done_root = Path(args.root), Path(args.done_root)
    done = {p.parent.name for p in done_root.glob("*/done.flag")}
    all_names = sorted(d for d in os.listdir(root) if (root / d / "data.json").exists())
    rest = [n for n in all_names if n not in done]
    rng = random.Random(args.seed)

    print(f"placements: {len(all_names)} total, {len(done)} already re-rendered, "
          f"{len(rest)} candidates available")

    # --- fixed background: originals (sampled, scaled) + the finished re-render (exact)
    bg = np.zeros(8)
    sample = rest[:]
    rng.shuffle(sample)
    sample = sample[: args.background_sample]
    n_sampled = 0
    for name in sample:
        got = placement_azimuths(root / name / "data.json")
        if not got:
            continue
        azs, obj = got
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        front = front_azimuth(obj)
        if front is None:
            continue
        h = hist_for_yaws(azs, front, np.array([0.0]))[0]
        bg += h
        n_sampled += 1
    if n_sampled:
        bg *= len(all_names) / n_sampled          # scale the sample to every original
    print(f"background (originals, {n_sampled} sampled -> scaled to {len(all_names)}): "
          f"{bg.sum():.0f} goals, gini {gini(bg):.3f}")

    for name in sorted(done):
        got = placement_azimuths(done_root / name / "data.json")
        if not got:
            continue
        azs, obj = got
        front = front_azimuth(obj)
        if front is None:
            continue
        yaw = float(json.loads((done_root / name / "data.json").read_text())
                    .get("placement_yaw_deg") or 0.0)
        bg += hist_for_yaws(azs, front, np.array([yaw]))[0]
    print(f"+ finished re-render: total {bg.sum():.0f} goals, gini {gini(bg):.3f}, "
          f"L1 {l1_to_uniform(bg):.4f}")

    # --- candidates
    yaws = np.arange(0.0, 360.0, args.yaw_step)
    cand_names: list[str] = []
    cand_hists: list[np.ndarray] = []
    pool = [n for n in rest if n not in set(sample)] or rest
    rng.shuffle(pool)
    for name in pool:
        if len(cand_names) >= args.candidates:
            break
        got = placement_azimuths(root / name / "data.json")
        if not got:
            continue
        azs, obj = got
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        front = front_azimuth(obj)
        if front is None:
            continue
        cand_names.append(name)
        cand_hists.append(hist_for_yaws(azs, front, yaws))
    if not cand_names:
        print("no candidates")
        return 1
    H = np.stack(cand_hists)                       # (C, Y, 8)
    print(f"candidates scanned: {len(cand_names)}  (yaw grid {len(yaws)} steps)")

    # --- greedy selection
    cur = bg.copy()
    taken = np.zeros(len(cand_names), dtype=bool)
    plan: list[tuple[str, float]] = []
    curve = []
    limit = min(args.max_n, len(cand_names))
    for step in range(limit):
        totals = cur.sum() + H.sum(axis=2)                       # (C, Y)
        cand_counts = cur[None, None, :] + H                     # (C, Y, 8)
        l1 = np.abs(cand_counts / totals[:, :, None] - 1.0 / 8).sum(axis=2)
        l1[taken] = np.inf
        c, y = np.unravel_index(np.argmin(l1), l1.shape)
        taken[c] = True
        cur = cur + H[c, y]
        plan.append((cand_names[c], float(yaws[y])))
        if (step + 1) % 50 == 0 or step == 0:
            curve.append({"n": step + 1, "l1": l1_to_uniform(cur), "gini": gini(cur),
                          "pct": {s: 100 * cur[i] / cur.sum() for i, s in enumerate(SECTORS)}})

    # --- where does the gain flatten? (last n whose 50-step L1 gain still beats 2%)
    chosen_n = args.n
    if not chosen_n:
        chosen_n = curve[-1]["n"] if curve else limit
        for a, b in zip(curve, curve[1:]):
            if a["l1"] - b["l1"] < 0.02 * a["l1"]:
                chosen_n = a["n"]
                break

    print(f"\n{'N':>6}{'L1':>9}{'gini':>8}   " + "".join(f"{s[:5]:>7}" for s in SECTORS))
    for row in curve:
        print(f"{row['n']:>6}{row['l1']:>9.4f}{row['gini']:>8.3f}   "
              + "".join(f"{row['pct'][s]:6.1f}%" for s in SECTORS))
    print(f"\nrecommended N (gain flattens): {chosen_n}")

    out_plan = {n: y for n, y in plan[:chosen_n]}
    Path(args.out).write_text(json.dumps(out_plan, indent=1, sort_keys=True))
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps({
        "background_goals": bg.tolist(), "curve": curve, "chosen_n": chosen_n,
        "targets": Counter(sector8((y) % 360) for _, y in plan[:chosen_n]),
    }, indent=1, default=str))
    print(f"wrote {args.out} ({len(out_plan)} placements) and {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
