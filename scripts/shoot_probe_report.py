"""Read `shoot_probe.py`'s raw output and answer the three questions it was run for.

Kept separate from the probe on purpose: the probe holds a GPU, this does not, so thresholds
can be re-derived and the analysis re-argued without re-running inference. Everything printed
here comes from the recorded per-step values, so the numbers are reproducible from the JSON.

    python scripts/shoot_probe_report.py runs/eval/shoot_probe_iter40000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default="runs/eval/shoot_probe_iter40000.json")
ap.add_argument("--out", default="", help="write the chosen threshold + summary here")
args = ap.parse_args()

d = json.loads(Path(args.path).read_text())
rows = d["rows"]
if d.get("partial"):
    print(f"NOTE: partial file — {len(rows)} windows so far\n")

# (window, step) pairs. `pred_shoot` is (samples, chunk); average over draws so one noisy
# draw does not decide a threshold, and record the spread separately.
P = np.array([np.mean(r["pred_shoot"], axis=0) for r in rows])          # (N, chunk)
SPREAD = np.array([np.std(r["pred_shoot"], axis=0) for r in rows])
T = np.array([r["true_shoot"] for r in rows], dtype=float)              # (N, chunk)
p, t = P.ravel(), T.ravel()

print("=" * 74)
print(f"1. DISTRIBUTION of predicted dim 9   ({len(rows)} windows x {P.shape[1]} steps)")
print("=" * 74)
edges = [-np.inf, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, np.inf]
lab = ["<0.05", "0.05-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", ">0.95"]
h, _ = np.histogram(p, bins=edges)
for L, c in zip(lab, h):
    print(f"  {L:>9} {c:>6}  {100*c/len(p):5.1f}%  {'#' * int(60*c/max(1,h.max()))}")
mid = ((p > 0.4) & (p < 0.6)).mean()
print(f"\n  min {p.min():+.3f}  max {p.max():+.3f}  mean {p.mean():.3f}")
print(f"  piled in 0.4-0.6: {100*mid:.1f}%   <- if this is large, a 0.5 cut is arbitrary")
print(f"  mean per-draw spread: {SPREAD.mean():.4f}"
      + ("  (single draw — spread unmeasured)" if d.get("samples", 1) == 1 else ""))
print(f"  by true label: shoot=0 -> mean {p[t == 0].mean():.3f} | "
      f"shoot=1 -> mean {p[t == 1].mean():.3f}")

print("\n" + "=" * 74)
print("2. THRESHOLD from the curve, not from a literal")
print("=" * 74)
best = None
print(f"  {'thr':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'acc':>7}")
for thr in np.arange(0.05, 1.0, 0.05):
    pred = p >= thr
    tp = float((pred & (t == 1)).sum()); fp = float((pred & (t == 0)).sum())
    fn = float((~pred & (t == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = float((pred == (t == 1)).mean())
    if best is None or f1 > best[3]:
        best = (thr, pr, rc, f1, acc)
    mark = "  <--" if abs(thr - 0.5) < 1e-9 else ""
    print(f"  {thr:5.2f} {pr:7.3f} {rc:7.3f} {f1:7.3f} {acc:7.3f}{mark}")
print(f"\n  best F1 at thr={best[0]:.2f}: P {best[1]:.3f} R {best[2]:.3f} F1 {best[3]:.3f}")
# A threshold is only worth changing if it beats 0.5 by more than noise.
at50 = None
pred50 = p >= 0.5
tp = float((pred50 & (t == 1)).sum()); fp = float((pred50 & (t == 0)).sum())
fn = float((~pred50 & (t == 1)).sum())
pr50 = tp / (tp + fp) if tp + fp else 0.0
rc50 = tp / (tp + fn) if tp + fn else 0.0
f150 = 2 * pr50 * rc50 / (pr50 + rc50) if pr50 + rc50 else 0.0
print(f"  at the incumbent 0.50   : P {pr50:.3f} R {rc50:.3f} F1 {f150:.3f}")
print(f"  -> {'KEEP 0.50' if best[3] - f150 < 0.02 else f'USE {best[0]:.2f}'}"
      f"  (F1 delta {best[3]-f150:+.3f}; <0.02 is inside noise for this sample size)")

print("\n" + "=" * 74)
print("3. ARRIVAL STEP — does the first crossing land where it should")
print("=" * 74)
thr = best[0] if best[3] - f150 >= 0.02 else 0.5
pos = [i for i, r in enumerate(rows) if r["arrival"] >= 0]
neg = [i for i, r in enumerate(rows) if r["arrival"] < 0]
errs, fired = [], 0
for i in pos:
    cross = np.argmax(P[i] >= thr) if (P[i] >= thr).any() else None
    if cross is None:
        continue
    fired += 1
    errs.append(int(cross) - rows[i]["arrival"])
print(f"  windows WITH an arrival   : {len(pos)}   fired: {fired} ({100*fired/max(1,len(pos)):.0f}%)")
false_fire = sum(1 for i in neg if (P[i] >= thr).any())
print(f"  windows WITHOUT an arrival: {len(neg)}   fired anyway: {false_fire} "
      f"({100*false_fire/max(1,len(neg)):.0f}%)  <- these would stop the rollout early")
if errs:
    e = np.array(errs)
    print(f"  step error (pred - true): median {np.median(e):+.1f}  mean {e.mean():+.2f}  "
          f"|err|<=1 in {100*(np.abs(e) <= 1).mean():.0f}%")
    print(f"    distribution: {dict(sorted(Counter(e.tolist()).items()))}")

print("\n" + "=" * 74)
print("4. THE LEAK — same window, same seed, only idle_frame differs")
print("=" * 74)
sw = d.get("sweep") or []
if not sw:
    print("  (no sweep in this file)")
else:
    arms = [("0", "shoot_idle_0"), ("3", "shoot_idle_3"), ("8", "shoot_idle_8"),
            ("omitted", "shoot_idle_omitted")]
    base = None
    print(f"  {'idle_frame':>11} {'mean pred':>10} {'fired %':>9} {'vs idle=0':>11}")
    for label, key in arms:
        A = np.array([r[key] for r in sw if key in r])
        if not len(A):
            continue
        m = float(A.mean())
        f = 100 * float(np.mean([(a >= thr).any() for a in A]))
        if base is None:
            base = A
            print(f"  {label:>11} {m:10.3f} {f:9.1f} {'--':>11}")
        else:
            print(f"  {label:>11} {m:10.3f} {f:9.1f} {float(np.abs(A-base).mean()):11.3f}")
    a0 = np.array([r["shoot_idle_0"] for r in sw])
    a8 = np.array([r["shoot_idle_8"] for r in sw])
    shift = float(a8.mean() - a0.mean())
    tt = np.array([r["true_shoot"] for r in sw], dtype=float)
    # How much of the prediction is explained by the prompt field vs by the frame?
    print(f"\n  idle 0 -> 8 shifts the mean prediction by {shift:+.3f}")
    print(f"  correlation with the TRUE label: idle=0 {np.corrcoef(a0.ravel(), tt.ravel())[0,1]:+.3f}"
          f" | idle=8 {np.corrcoef(a8.ravel(), tt.ravel())[0,1]:+.3f}")
    verdict = ("READS THE LEAK — the channel tracks the prompt field, so every termination "
               "number is confounded" if shift > 0.25 else
               "mild prompt sensitivity — termination numbers stand, with the shift reported"
               if shift > 0.08 else
               "IGNORES the field — the channel is driven by the frame and the goal")
    print(f"\n  VERDICT: {verdict}")

print("\n" + "=" * 74)
print("5. BY SCENE and BY SECTOR (a single mean can hide one scene failing)")
print("=" * 74)
for field in ("scene", "sector"):
    print(f"\n  {field}:")
    for key in sorted({r[field] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[field] == key]
        pp, tv = P[idx].ravel(), T[idx].ravel()
        pred = pp >= thr
        acc = float((pred == (tv == 1)).mean())
        print(f"    {str(key)[:34]:<34} n={len(idx):>3}  acc {acc:.3f}  "
              f"mean pred {pp.mean():.3f}")

if args.out:
    Path(args.out).write_text(json.dumps({
        "source": args.path, "checkpoint": d.get("checkpoint"),
        "threshold": float(thr), "threshold_rule": "max-F1, kept 0.50 if delta < 0.02",
        "f1_at_threshold": float(best[3]), "f1_at_0.50": float(f150),
        "pile_0.4_0.6_frac": float(mid),
        "leak_shift_idle0_to_idle8": float(shift) if sw else None,
    }, indent=1) + "\n")
    print(f"\nwrote {args.out}  (the rollouts read the threshold from here)")
