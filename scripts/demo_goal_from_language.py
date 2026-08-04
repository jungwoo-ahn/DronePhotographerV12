"""Demo/validate Module 1: NL -> GoalProfile. Shows the keyword baseline and (with --llm) the real
LLM classifier on varied requests, printing categories, grounded profile, and the re-serialized prompt."""
import argparse, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from functools import partial

from src.goal_authoring.from_language import (
    LLMClassifier,
    hybrid_classifier,
    keyword_classifier,
    language_to_goal,
)

EXAMPLES = [
    "a dramatic low-angle close-up of the subject from the side",
    "wide establishing shot from behind, subject small in the lower left",
    "medium shot at eye level, facing the camera, centered",
    "shoot them from above, three-quarter front-right, full body",
    "an intimate portrait, tight on the face",
    "back view, subject in the upper right",
]

ap = argparse.ArgumentParser()
ap.add_argument("--llm", action="store_true", help="pure LLM classifier")
ap.add_argument("--hybrid", action="store_true", help="keyword-primary + LLM gap-fill (recommended)")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
args = ap.parse_args()

clf = keyword_classifier
if args.llm or args.hybrid:
    print(f"loading LLM classifier: {args.model}", flush=True)
    llm = LLMClassifier(args.model)
    clf = partial(hybrid_classifier, llm=llm) if args.hybrid else llm

for text in EXAMPLES:
    gp = language_to_goal(text, clf)
    print(f"\nREQUEST: {text}")
    print(f"  categories: {gp.categories()}")
    print(f"  profile:    { {k: round(v,1) for k,v in gp.values.items()} }")
    print(f"  specified:  {sorted(gp.specified)}  (partial={gp.is_partial()})")
    print(f"  -> prompt:  {gp.to_nl()}")
