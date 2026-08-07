"""Stage 3 of the true recon: re-measure the rolled-out frames with Module 2.

Deployment-realistic scoring — the achieved frame is judged the same way the reference was read, from
pixels only. Reports, per case, how far start and achieved sit from the requested shot on the three
attributes the policy is conditioned on, and whether the rollout closed the gap. venv: .venv-analysis."""
import argparse, json, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
import numpy as np

from src.common.goal_space import SUBJECT_BEARING_KEY
from src.goal_authoring.from_reference import ReferenceEstimator

ap = argparse.ArgumentParser()
ap.add_argument("--recon", default="runs/recon_ref/recon.json")
ap.add_argument("--out", default="runs/recon_ref/measured.json")
args = ap.parse_args()

est = ReferenceEstimator()
doc = json.loads(open(args.recon).read())
rows = []

def cyc(d): return abs(((d + 180) % 360) - 180)

for i, c in enumerate(doc["results"]):
    goal = c["goal"]
    out = {"case": i, "reference": c["ref_image"], "ref_object": c["ref_object"],
           "target": c["tgt_object"], "goal": goal,
           "start_frame": c["start_frame"], "achieved_frame": c["achieved_frame"]}
    def read(path):
        gp = est(path)
        if "occupancy" not in gp.specified:
            return None
        occ = float(gp.values["occupancy"])
        bear = float(gp.values.get(SUBJECT_BEARING_KEY, np.nan))
        elev = float(gp.values.get("cam_to_obj_elevation_deg", np.nan))
        return {
            "occupancy": round(occ, 1), "bearing": round(bear, 1), "elevation": round(elev, 1),
            "categories": gp.categories(),
            "d_occ": round(abs(occ - goal["occupancy"]), 1),
            "d_bear": (round(cyc(bear - goal["bearing"]), 1) if np.isfinite(bear) else None),
            "d_elev": (round(abs(elev - goal["elevation"]), 1) if np.isfinite(elev) else None),
        }

    # every intermediate view, so the trajectory is visible rather than just its endpoints
    steps = c.get("step_frames") or [c["start_frame"], c["achieved_frame"]]
    out["step_frames"] = steps
    out["trajectory"] = [read(p) for p in steps]
    out["start"] = out["trajectory"][0]
    out["achieved"] = out["trajectory"][-1]
    rows.append(out)

print("\n===== TRUE RECON: reference composition re-shot by the policy in another scene =====")
print(f"{'case':>4} {'reference -> target':38s} {'|Δocc| s→a':>13} {'|Δbearing| s→a':>16} {'|Δelev| s→a':>14}")
imp_o = imp_b = imp_e = n = 0
for r in rows:
    s, a = r.get("start"), r.get("achieved")
    if not s or not a:
        print(f"{r['case']:>4} {r['ref_object'][:18]+' -> '+r['target'][:16]:38s}  (no subject detected)")
        continue
    n += 1
    imp_o += (a["d_occ"] < s["d_occ"]);
    if s["d_bear"] is not None and a["d_bear"] is not None: imp_b += (a["d_bear"] < s["d_bear"])
    if s["d_elev"] is not None and a["d_elev"] is not None: imp_e += (a["d_elev"] < s["d_elev"])
    print(f"{r['case']:>4} {r['ref_object'][:18]+' -> '+r['target'][:16]:38s} "
          f"{s['d_occ']:5.0f}→{a['d_occ']:<5.0f} {str(s['d_bear']):>7}→{str(a['d_bear']):<7} "
          f"{str(s['d_elev']):>6}→{str(a['d_elev']):<6}")
print(f"\nclosed the gap (achieved nearer the requested shot than the start): "
      f"occupancy {imp_o}/{n}, bearing {imp_b}/{n}, elevation {imp_e}/{n}")
print("NOTE: policy is at iter 6000 and still training; this is an early-checkpoint reading.")
json.dump(rows, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
