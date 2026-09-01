"""Author a policy goal from natural language and/or a reference image, and show exactly what
the policy will be asked — every key, where it came from, and the prompt string.

    python scripts/author_goal.py --nl "a close-up from the front"
    python scripts/author_goal.py --nl "low angle full body from the front" --method rules
    python scripts/author_goal.py --image ref.jpg --nl "but tighter" --both
    python scripts/author_goal.py --nl "wide shot from behind" --json runs/goals/wide_back.json

`--both` runs the bank and the rule completer side by side. `--json` writes the record shape
`scripts/recon_from_reference.py` consumes (goal_vec / goal_specified / goal_crop / prompt).
venv: .venv-analysis (GPU only if --image is given).
"""
import argparse, json, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

from src.goal_authoring.author import author_goal
from src.goal_authoring.from_language import keyword_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--nl", default=None, help="natural-language request")
ap.add_argument("--image", default=None, help="reference image (path)")
ap.add_argument("--method", default="bank", choices=("bank", "rules"))
ap.add_argument("--both", action="store_true", help="show bank AND rules completions")
ap.add_argument("--no-clamp", action="store_true", help="rules: do not clamp to the trained support")
ap.add_argument("--json", default=None, help="write the authored record(s) here")
ap.add_argument("--llm", default=None,
                help="LLM model id for the hybrid classifier (keyword rules first, LLM fills gaps); "
                     "needs an env with transformers")
args = ap.parse_args()
if args.nl is None and args.image is None:
    ap.error("give --nl and/or --image")

clf = keyword_classifier
if args.llm:
    from functools import partial
    from src.goal_authoring.from_language import LLMClassifier, hybrid_classifier
    clf = partial(hybrid_classifier, llm=LLMClassifier(args.llm))

estimator = None
if args.image:
    from src.goal_authoring.from_reference import ReferenceEstimator
    estimator = ReferenceEstimator()

methods = ("bank", "rules") if args.both else (args.method,)
records = []
for m in methods:
    g = author_goal(args.nl, args.image, classifier=clf, estimator=estimator, method=m,
                    clamp_support=False if args.no_clamp else "auto")
    print(f"===== {m} =====")
    print(g.summary())
    print()
    records.append(g.to_record())

if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    json.dump({"nl": args.nl, "image": args.image, "goals": records}, open(args.json, "w"), indent=1)
    print(f"wrote {args.json}")
