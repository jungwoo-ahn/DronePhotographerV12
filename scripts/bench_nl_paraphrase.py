"""HONEST evaluation of Module 1, replacing the circular round-trip metric.

The round-trip number (profile -> our serializer -> our keyword parser) is inflated: the parser was
written against the serializer's own phrasing. Here we instead sample known category combinations,
have an LLM write DIVERSE user-style paraphrases of them (cached to disk), and then measure how well
each classifier recovers the source categories. That measures robustness to how people actually talk.

  python scripts/bench_nl_paraphrase.py --generate       # LLM writes the paraphrase set (GPU)
  python scripts/bench_nl_paraphrase.py                  # evaluate on the cached set (CPU for keyword)
  python scripts/bench_nl_paraphrase.py --eval-llm       # also score the LLM / hybrid classifiers (GPU)
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, "/home/nas_main/jungwooahn/projects/DronePhotographerV12")
os.chdir("/home/nas_main/jungwooahn/projects/DronePhotographerV12")

from src.goal_authoring import vocab
from src.goal_authoring.from_language import (
    ALLOWED, hybrid_classifier, keyword_classifier, validate_categories,
)

ap = argparse.ArgumentParser()
ap.add_argument("--generate", action="store_true")
ap.add_argument("--eval-llm", action="store_true")
ap.add_argument("--n", type=int, default=60, help="category combos to paraphrase")
ap.add_argument("--per", type=int, default=3, help="paraphrases per combo")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--set", default="assets/nl_gold_set.json",
                help="assets/nl_gold_set.json = hand-labelled user phrasings (the primary metric); "
                     "assets/nl_paraphrase_set.json = LLM-generated at scale (NOISY: the generator "
                     "often omits requested attributes, so recall there is a lower bound)")
args = ap.parse_args()

AXES = ["shot_size", "bearing", "elevation", "placement_x", "placement_y", "body_framing"]


def sample_combo(rng):
    """A realistic partial request: 1-4 axes stated."""
    k = rng.choice([1, 2, 2, 3, 3, 4])
    axes = rng.sample(AXES, k)
    return {a: rng.choice([l for l in ALLOWED[a] if "off-screen" not in l]) for a in axes}


def generate():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model); tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    rng = random.Random(0)
    out = []
    for i in range(args.n):
        cats = sample_combo(rng)
        desc = "; ".join(f"{a.replace('_',' ')}: {l}" for a, l in cats.items())
        prompt = (
            "You are a photographer talking to an assistant. Write "
            f"{args.per} DIFFERENT one-sentence requests that each mean EXACTLY this shot and nothing "
            f"more:\n{desc}\n"
            "Vary the wording naturally (casual, technical, terse). Do NOT mention any attribute that "
            "is not listed. Output one request per line, no numbering, no quotes."
        )
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=140, do_sample=True, temperature=0.9, top_p=0.95)
        raw = tok.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        lines = [re.sub(r'^[\-\d\.\)\s"]+', "", l).strip().strip('"') for l in raw.splitlines()]
        lines = [l for l in lines if 8 < len(l) < 160][: args.per]
        for l in lines:
            out.append({"categories": cats, "text": l})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.n} combos, {len(out)} paraphrases", flush=True)
    os.makedirs(os.path.dirname(args.set), exist_ok=True)
    json.dump(out, open(args.set, "w"), indent=1)
    print(f"wrote {args.set}  ({len(out)} paraphrases)")


def score(name, clf, data):
    """Per-axis recall (did we recover the stated label?) and precision (did we invent unstated axes?)."""
    ok = tot = 0
    hallucinated = 0; predicted = 0
    per_ax_ok, per_ax_tot = {}, {}
    exact = 0
    for row in data:
        gt, txt = row["categories"], row["text"]
        pred = validate_categories(clf(txt))
        predicted += len(pred)
        hallucinated += sum(1 for a in pred if a not in gt)
        hit = True
        for a, l in gt.items():
            per_ax_tot[a] = per_ax_tot.get(a, 0) + 1
            good = pred.get(a) == l
            # bearing: accept the coarse/fine equivalent (side <-> left/right, front <-> front-*)
            if not good and a == "bearing" and pred.get(a):
                from src.common.facing import sector3
                gb = vocab.SECTOR3_CENTROID.get(l, vocab.bearing_centroid(l) if l in vocab.SECTOR8 else None)
                pb = vocab.SECTOR3_CENTROID.get(pred[a], vocab.bearing_centroid(pred[a]) if pred[a] in vocab.SECTOR8 else None)
                good = gb is not None and pb is not None and sector3(gb) == sector3(pb)
            if good:
                ok += 1; per_ax_ok[a] = per_ax_ok.get(a, 0) + 1
            else:
                hit = False
            tot += 1
        exact += hit
    axes = " ".join(f"{a.split('_')[0]}:{100*per_ax_ok.get(a,0)/per_ax_tot[a]:.0f}%" for a in sorted(per_ax_tot))
    print(f"{name:10s} recall={100*ok/max(tot,1):5.1f}%  all-axes-correct={100*exact/len(data):5.1f}%  "
          f"hallucinated-axes={100*hallucinated/max(predicted,1):4.1f}%  | {axes}")
    return dict(recall=100*ok/max(tot,1), exact=100*exact/len(data),
                hallucinated=100*hallucinated/max(predicted,1))


if args.generate:
    generate(); sys.exit(0)

data = json.load(open(args.set))
print(f"paraphrase eval set: {len(data)} user-style requests over "
      f"{len({json.dumps(r['categories'],sort_keys=True) for r in data})} category combos\n")
results = {"keyword": score("keyword", keyword_classifier, data)}
if args.eval_llm:
    from functools import partial
    from src.goal_authoring.from_language import LLMClassifier
    llm = LLMClassifier(args.model)
    results["llm"] = score("llm", llm, data)
    results["hybrid"] = score("hybrid", partial(hybrid_classifier, llm=llm), data)
json.dump(results, open("runs/nl_paraphrase_results.json", "w"), indent=1)
print("\nNOTE: this replaces the circular round-trip metric (parser evaluated on its own serializer).")
