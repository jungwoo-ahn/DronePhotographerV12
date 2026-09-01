"""Demo Module 2 end to end: reference image -> measured profile -> COMPLETE goal -> prompt.

Runs the estimator on the given images (default: a few sample renders) and prints what was
measured (with its `exact` marks), then the completed goal and the training-shape prompt for the
bank and rule completers. venv: .venv-analysis (GPU if free, else CPU)."""
import argparse, glob, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

from src.common.dataset_base import DEFAULT_TRAJ_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="*", help="image paths (default: a few sample renders)")
ap.add_argument("--nl", default=None, help="words applied on top of the image, e.g. 'but from a low angle'")
ap.add_argument("--methods", default="bank,rules")
args = ap.parse_args()

from src.goal_authoring.author import author_goal
from src.goal_authoring.from_reference import ReferenceEstimator

images = args.images
if not images:  # grab a few varied sample renders
    import random
    dirs = [d for d in glob.glob(f"{DEFAULT_TRAJ_ROOT}/*") if os.path.isdir(d)]
    random.seed(3); random.shuffle(dirs)
    for d in dirs[:6]:
        js = glob.glob(f"{d}/renders/pair_0*_frame_16.jpg")
        if js: images.append(js[0])

print("loading estimator (YOLO-pose + bearing/elevation models)...", flush=True)
est = ReferenceEstimator()

for ip in images:
    print(f"\nREFERENCE: {ip}")
    gp = est(ip)
    if not gp.specified:
        print("  (no subject detected)"); continue
    print(f"  measured : { {k: round(v, 1) for k, v in gp.values.items()} }")
    print(f"  exact    : {sorted(gp.exact)}")
    for m in args.methods.split(","):
        g = author_goal(args.nl, ip, estimator=est, method=m.strip())
        print(f"  [{m.strip()}] -> {g.categories}")
        for w in g.warnings:
            print(f"       ! {w}")
        print(f"       prompt: {g.prompt}")
