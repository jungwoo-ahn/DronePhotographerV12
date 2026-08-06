"""Fit + save the camera-elevation regressor used by Module 2.

The VLM was at chance on elevation (34-36%); body-pose keypoints carry the pitch signal through
vertical foreshortening (CV MAE 8.2deg vs 13.7deg predict-median, corr 0.76). Reuses the cached
features from `bench_elevation_pose.py`. venv: .venv-analysis."""
import argparse, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.goal_authoring import vocab

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="runs/elevation_feats.npz")
ap.add_argument("--out", default="assets/models/elevation_pose_rf.joblib")
ap.add_argument("--trees", type=int, default=60)
ap.add_argument("--max-depth", type=int, default=12)
args = ap.parse_args()

if not os.path.exists(args.cache):
    sys.exit(f"missing {args.cache} — run scripts/bench_elevation_pose.py first (extracts features)")
d = np.load(args.cache); X, y = d["X"], d["y"]
print(f"features {X.shape}; GT elevation range [{y.min():.0f},{y.max():.0f}] median {np.median(y):.0f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

def make(): return RandomForestRegressor(args.trees, max_depth=args.max_depth, random_state=0, n_jobs=-1)

p = cross_val_predict(make(), X, y, cv=5)
mae = float(np.mean(np.abs(p - y)))
base = float(np.mean(np.abs(y - np.median(y))))
bands = np.array([vocab._classify(v, vocab.ELEVATION) for v in y])
pb = np.array([vocab._classify(v, vocab.ELEVATION) for v in p])
maj = max(set(bands), key=lambda b: (bands == b).sum())
print(f"5-fold CV: MAE={mae:.1f}deg (predict-median {base:.1f})  corr={np.corrcoef(p,y)[0,1]:.2f}  "
      f"band acc={100*np.mean(pb==bands):.0f}% (majority {100*np.mean(bands==maj):.0f}%)")
print("NOTE: the band metric is inflated by class imbalance — the v7 data is 68% high-angle / 3.5% "
      "low-angle. The continuous MAE is the meaningful number.")

import joblib
os.makedirs(os.path.dirname(args.out), exist_ok=True)
joblib.dump(make().fit(X, y), args.out)
print(f"saved {args.out}  ({os.path.getsize(args.out)//1024} KB)")
