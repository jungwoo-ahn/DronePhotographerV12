"""Build the goal bank from a LeRobot export and verify it reproduces every training prompt.

    python scripts/build_goal_bank.py                       # runs/lerobot_v5 -> assets/goal_bank_v5.npz
    python scripts/build_goal_bank.py --root runs/lerobot_v6 --out assets/goal_bank_v6.npz

The bank stores numbers only; every row is re-serialized through `goal_prompt` and compared
BYTE FOR BYTE with the task string it was parsed from. A single mismatch aborts the build,
because a row that does not round-trip is a goal the policy never saw. venv: .venv-analysis.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

import pandas as pd

from src.goal_authoring.goal_bank import DEFAULT_BANK_PATH, GoalBank

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/lerobot_v5")
ap.add_argument("--out", default=str(DEFAULT_BANK_PATH))
# Absolute on purpose (the same constant src/data/cosmos_camera_dataset.py anchors to the repo);
# that module is not imported here because it pulls in the cosmos framework, which the
# analysis venv does not have.
V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
args = ap.parse_args()

t0 = time.time()
val = json.load(open(args.val_scenes))["scenes"] if args.val_scenes else []
bank = GoalBank.from_lerobot(args.root, val_scenes=val)
print(f"parsed {len(bank)} goals (skipped {bank.meta['skipped']}) in {time.time()-t0:.0f}s")

tasks = pd.read_parquet(os.path.join(args.root, "meta", "tasks.parquet"))
strings = list(tasks["task"].astype(str)) if "task" in tasks.columns else list(tasks.index.astype(str))
assert len(strings) == len(bank) + bank.meta["skipped"]
bad = [i for i, s in enumerate(strings) if bank.prompt(i) != s]
if bad:
    i = bad[0]
    sys.exit(f"{len(bad)} rows do not round-trip; first:\n  src {strings[i]}\n  got {bank.prompt(i)}")
print(f"round-trip: {len(bank)}/{len(bank)} rows re-serialize to their source prompt byte for byte")

tr = bank.train_mask()
print(f"train-scene goals {int(tr.sum())}   val-only goals {int((~tr).sum())}")
sup = bank.support_table(tr)
for k, d in sup.items():
    tot = sum(d.values()) or 1
    print(f"  {k:26s} " + "  ".join(f"{lab}={n} ({100*n/tot:.1f}%)" for lab, n in d.items()))
for k in ("occupancy", "cam_to_obj_elevation_deg", "body_in_frame_ratio", "visible_frac"):
    q = bank.quantiles(k, [0.0, 0.005, 0.5, 0.995, 1.0], tr)
    print(f"  {k:26s} min {q[0]:.2f}  p0.5 {q[1]:.2f}  median {q[2]:.2f}  p99.5 {q[3]:.2f}  max {q[4]:.2f}")
bank.meta["built"] = time.strftime("%Y-%m-%d")
p = bank.save(args.out)
print(f"wrote {p} ({os.path.getsize(p)//1024} KB)")
