"""How wrong is the guess? Held-out benchmark of the two completers.

Takes the goals that occur ONLY in the held-out scenes (never in the bank's train rows), throws
away everything but what a user would have said, completes, and compares with the truth:

    shot+bearing        "a wide shot from the front"
    +elevation          "... at eye level"
    +placement+crop     "... left third, lower, cropped at the legs"
    image:side-only     everything a photo pins (box, occupancy, bearing, elevation) + the cut SIDE
    image:fractions     the same with the cut FRACTIONS (body-extent fit succeeded)

For each unspecified key: normalized MAE (bearing cyclic) and whether the WORD the prompt would
carry matches the truth. The "rules" column is dataset-independent geometry; "bank" anchors on a
real training goal. Both face the same ambiguity — this measures which guesses better.
venv: .venv-analysis (CPU, a few minutes).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.goal_space import CYCLIC_GOAL_KEYS, DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY
from src.goal_authoring import vocab
from src.goal_authoring.author import complete_request
from src.goal_authoring.completion import BankCompleter, RuleCompleter
from src.goal_authoring.goal_bank import BAND_KEYS, NUMERIC_KEYS, SCALE, GoalBank, band_of
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.serialize import crop_phrase

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="targets (0 = all val-only goals)")
ap.add_argument("--out", default="runs/eval/completion_bench.json")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

bank = GoalBank.load()
targets = np.flatnonzero(bank.n_train == 0)
rng = np.random.default_rng(args.seed)
if args.n and args.n < len(targets):
    targets = rng.choice(targets, args.n, replace=False)
print(f"bank {len(bank)} goals; train rows {int(bank.train_mask().sum())}; held-out targets {len(targets)}")

MASKS = {
    "shot+bearing": ("shot_size", "bearing"),
    "+elevation": ("shot_size", "bearing", "elevation"),
    "+placement+crop": ("shot_size", "bearing", "elevation", "placement_x", "placement_y", "crop_side"),
    "image:side-only": "image_side",
    "image:fractions": "image_frac",
}
AXIS_OF_KEY = {"occupancy": "shot_size", SUBJECT_BEARING_KEY: "bearing", "cam_to_obj_elevation_deg": "elevation",
               "object_center_x": "placement_x", "object_center_y": "placement_y",
               "body_in_frame_ratio": "body_framing"}
IMG_KEYS = ("occupancy", "object_center_x", "object_center_y", "bbox_x_offset", "bbox_y_offset",
            SUBJECT_BEARING_KEY, "cam_to_obj_elevation_deg")


def gt_words(row):
    w = {AXIS_OF_KEY[k]: band_of(k, row[k]) for k in AXIS_OF_KEY}
    w["crop_side"] = row["crop_label"]
    return w


def fractions_from(label, vis):
    cut = max(0.0, 1.0 - vis)
    if label == "uncropped":
        return 0.0, 0.0
    if label == "cropped above the head":
        return cut, 0.0
    if label == "cropped at both the head and the feet":
        return cut / 2, cut / 2
    return 0.0, cut                                   # legs / waist


def request_for(row, mask):
    if isinstance(mask, tuple):
        return GoalProfile.from_categories({ax: gt_words(row)[ax] for ax in mask})
    vals = {k: float(row[k]) for k in IMG_KEYS}
    exact = set(IMG_KEYS)
    label, vis = row["crop_label"], float(row["visible_frac"])
    if mask == "image_frac" or label == "uncropped":
        t, b = fractions_from(label, vis)
        vals.update(top_cut_frac=t, bot_cut_frac=b, visible_frac=(1.0 - t - b))
        exact |= {"top_cut_frac", "bot_cut_frac", "visible_frac"}
    else:                                              # side only: 1.0 markers, no visible
        t, b = vocab.CROP_SIDE[label]
        vals.update(top_cut_frac=1.0 if t > 0 else 0.0, bot_cut_frac=1.0 if b > 0 else 0.0)
    return GoalProfile(vals, frozenset(vals), frozenset(exact))


def err(k, a, b):
    if k in CYCLIC_GOAL_KEYS:
        return abs(((a - b + 180.0) % 360.0) - 180.0) / SCALE[k]
    return abs(a - b) / SCALE[k]


completers = {"bank": BankCompleter(bank), "rules": RuleCompleter()}
results = {}
t0 = time.time()
for mname, mask in MASKS.items():
    results[mname] = {}
    for cname, comp in completers.items():
        acc = {k: [] for k in NUMERIC_KEYS}
        words = {ax: [] for ax in list(AXIS_OF_KEY.values()) + ["crop_side"]}
        n_warn = 0
        for i in targets:
            row = bank.row(int(i))
            req = request_for(row, mask)
            done = complete_request(req, comp)
            n_warn += bool(done.warnings)
            vals = done.profile.values
            for k in NUMERIC_KEYS:
                if k in req.specified and k in req.exact:
                    continue                            # given exactly -> not a guess
                acc[k].append(err(k, float(vals[k]), float(row[k])))
            got = gt_words({**{k: vals[k] for k in NUMERIC_KEYS}, "crop_label": crop_phrase(vals["top_cut_frac"], vals["bot_cut_frac"])})
            truth = gt_words(row)
            for ax in words:
                spec_axes = set(mask) if isinstance(mask, tuple) else set()
                if ax in spec_axes:
                    continue
                words[ax].append(got[ax] == truth[ax])
        results[mname][cname] = {
            "mae": {k: float(np.mean(v)) for k, v in acc.items() if v},
            "word_match": {ax: float(np.mean(v)) for ax, v in words.items() if v},
            "warned_frac": n_warn / len(targets),
        }
    print(f"[{mname}] done ({time.time()-t0:.0f}s)", flush=True)

# ---- report ----
def fmt(x):
    return f"{100*x:5.1f}"

print("\n=== word match of the GUESSED axes (%), bank | rules ===")
axes = ["shot_size", "bearing", "elevation", "placement_x", "placement_y", "body_framing", "crop_side"]
print(f"{'mask':18s} " + " ".join(f"{a[:11]:>13s}" for a in axes))
for mname in MASKS:
    cells = []
    for a in axes:
        b = results[mname]["bank"]["word_match"].get(a)
        r = results[mname]["rules"]["word_match"].get(a)
        cells.append(f"{'  --  ':>13s}" if b is None else f"{fmt(b)}|{fmt(r)}")
    print(f"{mname:18s} " + " ".join(cells))

print("\n=== normalized MAE of the guessed numbers (x100; 1 unit = full range), bank | rules ===")
keys = ["occupancy", "cam_to_obj_elevation_deg", "body_in_frame_ratio", "object_center_x", "object_center_y",
        "bbox_x_offset", "bbox_y_offset", "visible_frac"]
print(f"{'mask':18s} " + " ".join(f"{k[:11]:>13s}" for k in keys))
for mname in MASKS:
    cells = []
    for k in keys:
        b = results[mname]["bank"]["mae"].get(k)
        r = results[mname]["rules"]["mae"].get(k)
        cells.append(f"{'  --  ':>13s}" if b is None else f"{fmt(b)}|{fmt(r)}")
    print(f"{mname:18s} " + " ".join(cells))
print("\nwarned:", {m: {c: round(results[m][c]["warned_frac"], 2) for c in results[m]} for m in results})

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"targets": int(len(targets)), "results": results}, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
