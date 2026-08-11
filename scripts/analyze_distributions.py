"""Distribution analysis of v7 goal-profile (8-key V5) + action (5D) — evidence for v12 design.
Tests: (a) is the action/goal representation problematic? (b) is the current goal profile 'meh'?
Read-only, CPU. Samples placements to stay NAS-friendly."""
import os, sys, random, math
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np
from src.common.dataset_base import BasePolicyDataset
from src.common.goal_space import goal_keys as _gk
from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ROOT = DEFAULT_TRAJ_ROOT
dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
random.seed(0); random.shuffle(dirs)
sample_dirs = [os.path.join(ROOT, d) for d in dirs[:300]]  # ~300 placements

ds = BasePolicyDataset(sample_dirs, chunk_size=8, stride=2, sampling_scheme="multiscale_bidir",
                       offsets=(8, 16, 24), goal_sampling="end", filter_clamped_goals=True)
keys = list(ds.goal_keys)
print(f"placements sampled: {len(sample_dirs)} | windows: {len(ds)} | goal keys: {keys}")

G, A = [], []
N = min(len(ds), 25000)
idxs = list(range(len(ds))); random.shuffle(idxs); idxs = idxs[:N]
for i in idxs:
    s = ds[i]
    G.append(np.asarray(s.goal_vec, dtype=np.float64))
    A.append(np.asarray(s.action_chunk, dtype=np.float64))  # (8,5)
G = np.stack(G)              # (N,8)
A = np.stack(A)              # (N,8,5)
Af = A.reshape(-1, 5)        # (N*8,5) per-step
print(f"collected: goals {G.shape}, actions {A.shape}\n")

def q(x, p): return float(np.quantile(x, p))
print("===== GOAL PROFILE (8-key V5, raw units) =====")
print(f"{'key':26s} {'min':>8} {'p1':>8} {'p50':>8} {'p99':>8} {'max':>8} {'mean':>8} {'std':>8}  note")
for j, k in enumerate(keys):
    x = G[:, j]
    note = ""
    if k == "occupancy":
        note = f"sat@100:{100*np.mean(x>=99):.0f}%  <5%:{100*np.mean(x<5):.0f}%"
    elif k == "body_in_frame_ratio":
        note = f"sat@100:{100*np.mean(x>=99):.0f}%"
    elif "azimuth" in k:
        seam = np.mean((x < 15) | (x > 345))
        note = f"seam(0/360):{100*seam:.0f}%  range[{x.min():.0f},{x.max():.0f}]"
    if np.std(x) < 1e-6: note += " DEGENERATE(~const)"
    print(f"{k:26s} {x.min():8.1f} {q(x,.01):8.1f} {q(x,.5):8.1f} {q(x,.99):8.1f} {x.max():8.1f} {x.mean():8.1f} {x.std():8.1f}  {note}")

print("\n===== GOAL key-key correlation (redundancy check; |r|>0.7 = redundant) =====")
C = np.corrcoef(G.T)
short = [k[:10] for k in keys]
print("           " + " ".join(f"{s:>10}" for s in short))
for j, k in enumerate(short):
    row = " ".join(f"{C[j,i]:>10.2f}" for i in range(len(keys)))
    print(f"{k:10s} {row}")
print("high |r| pairs:")
for a in range(len(keys)):
    for b in range(a+1, len(keys)):
        if abs(C[a, b]) > 0.7:
            print(f"   {keys[a]} ~ {keys[b]}: r={C[a,b]:+.2f}")

print("\n===== ACTION (5D per-step, raw units; trans=m, rot=rad) =====")
adims = ["d_right(m)", "d_up(m)", "d_forward(m)", "d_yaw(rad)", "d_pitch(rad)"]
print(f"{'dim':14s} {'min':>8} {'p1':>8} {'p50':>8} {'p99':>8} {'max':>8} {'mean':>8} {'std':>8}  note")
for j, d in enumerate(adims):
    x = Af[:, j]
    note = f"|x|<0.01:{100*np.mean(np.abs(x)<0.01):.0f}%"
    if "yaw" in d:
        note += f"  in[-pi,pi]:{100*np.mean(np.abs(x)<=math.pi+1e-3):.0f}%  |x|>3.0:{100*np.mean(np.abs(x)>3.0):.1f}%"
    print(f"{d:14s} {x.min():8.3f} {q(x,.01):8.3f} {q(x,.5):8.3f} {q(x,.99):8.3f} {x.max():8.3f} {x.mean():8.3f} {x.std():8.3f}  {note}")

print("\n===== ACTION chunk-level: per-window mean |action| across 8 steps (is chunk near-constant?) =====")
mag = np.linalg.norm(A, axis=2)     # (N,8) per-step magnitude
print(f"per-step |a| : p50={np.median(mag):.3f}  p99={q(mag,.99):.3f}  frac steps |a|<0.02: {100*np.mean(mag<0.02):.0f}%")
within = A.std(axis=1).mean(axis=0)  # std across the 8 steps, per dim
print(f"within-chunk std per dim (small = chunk ~constant): {[round(float(v),3) for v in within]}")
print("\nDONE")
