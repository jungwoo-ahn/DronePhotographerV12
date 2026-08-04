"""Demo Module 2: reference image -> GoalProfile. Runs the estimator on given images (default: a few
sample renders) and prints the recovered profile, cinematography categories, and the prompt it would
condition on. venv: .venv-analysis (GPU)."""
import argparse, glob, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="*", help="image paths (default: a few sample renders)")
args = ap.parse_args()

from src.goal_authoring.from_reference import ReferenceEstimator

images = args.images
if not images:  # grab a few varied sample renders
    import random
    dirs = [d for d in glob.glob("data/trajectories/*") if os.path.isdir(d)]
    random.seed(3); random.shuffle(dirs)
    for d in dirs[:6]:
        js = glob.glob(f"{d}/renders/pair_0*_frame_16.jpg")
        if js: images.append(js[0])

print("loading estimator (YOLO-pose + bearing model)...", flush=True)
est = ReferenceEstimator()

for ip in images:
    gp = est(ip)
    print(f"\nREFERENCE: {ip}")
    if not gp.specified:
        print("  (no subject detected)"); continue
    print(f"  categories: {gp.categories()}")
    print(f"  profile:    { {k: round(v,1) for k,v in gp.values.items()} }")
    print(f"  specified:  {sorted(gp.specified)}  (elevation left to the user)")
    print(f"  -> prompt:  {gp.to_nl()}")
