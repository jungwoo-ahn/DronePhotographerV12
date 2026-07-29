"""Check whether camera ROLL about the world vertical axis is always ~0 in our trajectories.
If roll != 0 anywhere, the current 5D action (yaw+pitch only) is LOSSY there → a real error source,
and rot6d (full SO(3) relative rotation) would be strictly more faithful. Self-contained; reads only
the data symlink. Roll = camera-up's out-of-(vertical-plane) component."""
import os, sys, json, math, random
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

ROOT = "data/trajectories"
dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
random.seed(0); random.shuffle(dirs)
sample = dirs[:400]

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v / (n + 1e-12)

# world up axis: verify from data (mean of camera-up vectors should reveal it)
up_accum = np.zeros(3); nfr = 0
rolls = []           # roll angle (deg) per frame
near_vertical = 0    # frames where forward ~ vertical (roll ill-defined)
n_traj = 0
for dn in sample:
    p = os.path.join(ROOT, dn, "data.json")
    if not os.path.exists(p): continue
    try: d = json.load(open(p))
    except Exception: continue
    for pair in d.get("accepted_pairs", []):
        traj = pair.get("trajectory_32f") or []
        if not traj: continue
        n_traj += 1
        for f in traj:
            fwd = unit(f["forward"]); up = unit(f["up"])
            up_accum += up; nfr += 1
    if n_traj > 1500: break

world_up = unit(up_accum / max(1, nfr))
# snap to nearest axis for interpretation
axis = int(np.argmax(np.abs(world_up)))
print(f"frames={nfr} trajectories={n_traj}")
print(f"mean camera-up = [{world_up[0]:+.3f},{world_up[1]:+.3f},{world_up[2]:+.3f}]  -> world-up axis ~ {'XYZ'[axis]} (sign {'+' if world_up[axis]>0 else '-'})")
WUP = np.zeros(3); WUP[axis] = 1.0 if world_up[axis] > 0 else -1.0
print(f"using world_up = {WUP.tolist()}\n")

# roll per frame: horizontal 'right' = normalize(cross(world_up, forward)); roll = angle of camera-up
# out of the (world_up, forward) vertical plane = asin(up . horizontal_right).
for dn in sample:
    p = os.path.join(ROOT, dn, "data.json")
    if not os.path.exists(p): continue
    try: d = json.load(open(p))
    except Exception: continue
    for pair in d.get("accepted_pairs", []):
        for f in (pair.get("trajectory_32f") or []):
            fwd = unit(f["forward"]); up = unit(f["up"])
            h = np.cross(WUP, fwd)
            if np.linalg.norm(h) < 0.08:      # forward ~ vertical -> roll ill-defined
                near_vertical += 1; continue
            h = unit(h)
            roll = math.degrees(math.asin(np.clip(np.dot(up, h), -1, 1)))
            rolls.append(roll)

rolls = np.array(rolls)
ar = np.abs(rolls)
print("===== ROLL (deg, about world vertical) =====")
print(f"n={len(rolls)}  near-vertical(skipped)={near_vertical}")
print(f"|roll|: mean={ar.mean():.4f}  p50={np.median(ar):.4f}  p99={np.quantile(ar,.99):.4f}  max={ar.max():.4f}")
print(f"frac |roll|>0.5deg: {100*np.mean(ar>0.5):.3f}%   >2deg: {100*np.mean(ar>2):.3f}%   >5deg: {100*np.mean(ar>5):.3f}%")
if ar.max() < 0.5:
    print("=> ROLL ~ 0 everywhere: current yaw+pitch (2-DOF) rep is LOSSLESS on rotation; roll-free confirmed.")
else:
    print("=> ROLL is NONZERO on some frames: the current 5D (yaw+pitch) rep is LOSSY there; rot6d would capture it.")
print("\nDONE")
