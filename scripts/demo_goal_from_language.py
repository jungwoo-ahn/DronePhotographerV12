"""Demo Module 1 end to end: NL -> partial profile -> COMPLETE goal -> the training-shape prompt.

Shows, per request, what the classifier parsed, then the bank-completed and rule-completed goals
side by side with per-key provenance. `--llm MODEL` switches to the hybrid classifier (keyword
rules first, LLM fills what they miss; needs an env with transformers).
"""
import argparse, os, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")
from functools import partial

from src.goal_authoring.author import author_goal
from src.goal_authoring.from_language import keyword_classifier

EXAMPLES = [
    "a dramatic low-angle close-up of the subject from the side",
    "wide establishing shot from behind, subject small in the lower left",
    "medium shot at eye level, facing the camera, centered",
    "shoot them from above, three-quarter front-right, full body",
    "an intimate portrait, tight on the face",
    "back view, subject in the upper right",
]

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default=None, help="hybrid classifier with this LLM (e.g. Qwen/Qwen3-8B)")
ap.add_argument("--methods", default="bank,rules")
args = ap.parse_args()

clf = keyword_classifier
if args.llm:
    from src.goal_authoring.from_language import LLMClassifier, hybrid_classifier
    print(f"loading LLM classifier: {args.llm}", flush=True)
    clf = partial(hybrid_classifier, llm=LLMClassifier(args.llm))

for text in EXAMPLES:
    print(f"\nREQUEST: {text}")
    for m in args.methods.split(","):
        g = author_goal(text, classifier=clf, method=m.strip())
        print(f"  [{m.strip()}] parsed {g.request.categories()}")
        print(f"       -> {g.categories}")
        for w in g.warnings:
            print(f"       ! {w}")
        print(f"       prompt: {g.prompt}")
