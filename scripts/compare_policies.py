"""One table, several policies, the same episodes — and paired statistics because of it.

Reads closed-loop result JSONs from this repo and from the baselines repo (whose harness is a
port of ours and writes the same per-episode fields), inner-joins them on the shared episode
`key`, and reports:

  * per-policy: mean / median improvement, frac positive, d_start -> d_end, and success@tau
    calibrated by scripts/calibrate_shot_threshold.py — the fraction whose FINAL frame is close
    enough that it reads as the shot that was asked for;
  * success at the closest point of the path as well, whose gap to the final one is the price of
    not knowing when to stop;
  * PAIRED comparisons against a reference policy: per-episode differences with a bootstrap CI
    and a sign test. Only a shared episode set allows this, and it is far more sensitive than
    comparing two means — the same-policy noise band for an 8-episode mean is [0.21, 0.84].

It refuses to join runs whose `d_start` disagrees on a shared key. That is the whole point of
the shared manifest: if the starts differ the episodes differ, and no amount of matching the
protocol makes the numbers comparable.

    python scripts/compare_policies.py \
        --run ours=runs/closed_loop/v5_shared.json \
        --run pi05=../baselines/runs/pi05_shared/summary.json \
        --ref ours

venv: .venv-analysis (CPU, instant).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np  # noqa: E402

DEFAULT_THRESHOLDS = str(Path(V12) / "runs" / "eval" / "shot_threshold.json")


def load_run(path: str) -> tuple[dict, dict]:
    """(summary, {key: episode}). Accepts either harness's output shape.

    Ours writes {"summary", "episodes"}; the baselines' `closed_loop_eval_baselines.py` writes a
    flat summary with the per-episode list under "results". Episodes older than the shared
    manifest have no `key`, so one is rebuilt from the image paths, which encode
    `pair_<pair>_frame_<idx>`.
    """
    doc = json.loads(Path(path).read_text())
    summary = doc.get("summary", doc)
    eps = doc.get("episodes") or doc.get("results") or []
    out = {}
    for e in eps:
        key = e.get("key") or _key_from_paths(e)
        if key is None:
            continue
        out[key] = e
    return summary, out


def _key_from_paths(e: dict) -> str | None:
    import re
    pl = e.get("placement")
    m_s = re.search(r"pair_(\d+)_frame_(\d+)", str(e.get("start_frame_image", "")))
    m_g = re.search(r"pair_(\d+)_frame_(\d+)", str(e.get("goal_frame_image", "")))
    if not (pl and m_s and m_g):
        return None
    return f"{pl}:p{int(m_g.group(1)):02d}:s{int(m_s.group(2)):02d}:g{int(m_g.group(2)):02d}"


def thresholds(path: str) -> list[float]:
    try:
        doc = json.loads(Path(path).read_text())
        return [float(r["tau"]) for r in doc.get("report_at", [])] or [0.05, 0.10, 0.20]
    except Exception:  # noqa: BLE001
        return [0.05, 0.10, 0.20]


def agreement_note(path: str) -> str:
    try:
        doc = json.loads(Path(path).read_text())
        parts = [f"{r['tau']:.2f}->{100*r['same_all']:.0f}%" for r in doc["report_at"]
                 if r.get("same_all") is not None]
        return ("two real frames this close carry the same six cinematography words "
                + ", ".join(parts) + f" (n={doc.get('n_pairs')} pairs)")
    except Exception:  # noqa: BLE001
        return "run scripts/calibrate_shot_threshold.py to calibrate these"


def stats(eps: list[dict], taus: list[float]) -> dict:
    imp = np.array([e["improvement"] for e in eps], dtype=np.float64)
    d_s = np.array([e["d_start"] for e in eps], dtype=np.float64)
    d_e = np.array([e["d_end"] for e in eps], dtype=np.float64)
    d_b = np.array([e.get("d_best", e["d_end"]) for e in eps], dtype=np.float64)
    return {
        "n": len(eps),
        "mean_improvement": float(imp.mean()),
        "median_improvement": float(np.median(imp)),
        "frac_positive": float((imp > 0).mean()),
        "mean_d_start": float(d_s.mean()),
        "mean_d_end": float(d_e.mean()),
        "mean_d_best": float(d_b.mean()),
        "median_closure": float(np.median(1.0 - d_e / np.maximum(d_s, 1e-9))),
        "success": {f"{t:.2f}": float((d_e <= t).mean()) for t in taus},
        "success_best": {f"{t:.2f}": float((d_b <= t).mean()) for t in taus},
    }


def paired(a: list[dict], b: list[dict], seed: int = 0) -> dict:
    """a - b per episode (same order), with a bootstrap CI and a sign test."""
    da = np.array([e["improvement"] for e in a], dtype=np.float64)
    db = np.array([e["improvement"] for e in b], dtype=np.float64)
    diff = da - db
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    wins = int((diff > 0).sum())
    ties = int((diff == 0).sum())
    n = len(diff) - ties
    # two-sided sign test, exact binomial
    from math import comb
    k = max(wins, n - wins)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)) if n else 1.0
    return {"mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": wins, "losses": n - wins, "ties": ties, "sign_p": float(p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH",
                    help="a result JSON, named for the table. Repeat.")
    ap.add_argument("--ref", default=None, help="policy to compare the others against (paired)")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--tol", type=float, default=1e-6, help="allowed d_start disagreement")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run wants NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        runs[name] = load_run(path)
        print(f"{name:16s} {len(runs[name][1]):>4} episodes  {path}")

    shared = set.intersection(*(set(e) for _s, e in runs.values()))
    if not shared:
        raise SystemExit("no episode key is present in every run — were they run from the same "
                         "manifest? (scripts/build_eval_episodes.py)")
    dropped = {n: len(e) - len(shared) for n, (_s, e) in runs.items() if len(e) != len(shared)}
    order = sorted(shared)

    # The check the shared manifest exists for.
    bad = []
    ref_name = args.ref if args.ref in runs else next(iter(runs))
    for k in order:
        vals = [runs[n][1][k]["d_start"] for n in runs]
        if max(vals) - min(vals) > args.tol:
            bad.append((k, {n: round(runs[n][1][k]["d_start"], 6) for n in runs}))
    if bad:
        for k, v in bad[:5]:
            print(f"  d_start MISMATCH {k}: {v}")
        raise SystemExit(f"{len(bad)}/{len(order)} shared keys disagree on d_start by more than "
                         f"{args.tol:g}. These are not the same episodes; the comparison would be "
                         "meaningless.")

    taus = thresholds(args.thresholds)
    print(f"\n{len(order)} shared episodes"
          + (f"  (dropped as unshared: {dropped})" if dropped else "")
          + f"\nsuccess@tau: {agreement_note(args.thresholds)}\n")

    table = {n: stats([runs[n][1][k] for k in order], taus) for n in runs}
    head = (f"{'policy':16s} {'n':>4} {'d_start':>8} {'d_end':>7} {'imp':>7} {'med':>7} "
            f"{'frac+':>6} {'closure':>8} " + " ".join(f"{'s@'+f'{t:.2f}':>7}" for t in taus)
            + " " + " ".join(f"{'best@'+f'{t:.2f}':>10}" for t in taus))
    print(head)
    for n, s in table.items():
        print(f"{n:16s} {s['n']:>4} {s['mean_d_start']:>8.3f} {s['mean_d_end']:>7.3f} "
              f"{s['mean_improvement']:>7.3f} {s['median_improvement']:>7.3f} "
              f"{s['frac_positive']:>6.2f} {s['median_closure']:>7.0%} "
              + " ".join(f"{s['success'][f'{t:.2f}']:>7.0%}" for t in taus) + " "
              + " ".join(f"{s['success_best'][f'{t:.2f}']:>10.0%}" for t in taus))

    pairs = {}
    others = [n for n in runs if n != ref_name]
    if others:
        print(f"\npaired against {ref_name} (per-episode difference in improvement):")
        print(f"{'policy':16s} {'mean diff':>10} {'95% CI':>20} {'w/l/t':>12} {'sign p':>8}")
        for n in others:
            pr = paired([runs[ref_name][1][k] for k in order],
                        [runs[n][1][k] for k in order])
            pairs[n] = pr
            ci = f"[{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]"
            wlt = "{}/{}/{}".format(pr["wins"], pr["losses"], pr["ties"])
            print(f"{n:16s} {pr['mean_diff']:>+10.3f} {ci:>20} {wlt:>12} {pr['sign_p']:>8.3f}")
        print(f"(positive = {ref_name} is closer to the requested shot on that episode)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"shared_episodes": len(order), "dropped": dropped, "thresholds": taus,
             "reference": ref_name, "table": table, "paired": pairs,
             "runs": {n: s for n, (s, _e) in runs.items()}}, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
