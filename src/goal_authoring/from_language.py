"""Module 1: natural language -> GoalProfile.

An LLM's only job is CLASSIFICATION into the controlled cinematography vocabulary (it must not invent
raw numbers — that is where VLMs are unreliable); `categories_to_profile` then grounds the categories
to numeric values deterministically. Anything the user did not state stays UNSPECIFIED (partial goal).

The classifier is pluggable:
  - `keyword_classifier` — deterministic phrase matching (no model): a baseline + what the unit tests use.
  - `LLMClassifier`      — a transformers instruct model emitting validated JSON over the allowed labels.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Mapping

from src.common.facing import SECTOR8
from src.goal_authoring import vocab
from src.goal_authoring.goal_profile import GoalProfile

# allowed labels per axis — the validation whitelist (bearing accepts sector8 OR coarse front/side/back)
ALLOWED: dict[str, tuple[str, ...]] = {
    "shot_size": tuple(vocab.SHOT_SIZE),
    "body_framing": tuple(vocab.BODY_FRAMING),
    "elevation": tuple(vocab.ELEVATION),
    "bearing": tuple(SECTOR8) + ("front", "side", "back"),
    "placement_x": tuple(vocab.PLACE_X),
    "placement_y": tuple(vocab.PLACE_Y),
}

Classifier = Callable[[str], Mapping[str, str]]


def validate_categories(cats: Mapping[str, str]) -> dict[str, str]:
    """Drop any axis/label not in the controlled vocabulary — the model can only pick real labels."""
    return {ax: lb for ax, lb in cats.items() if ax in ALLOWED and lb in ALLOWED[ax]}


def language_to_goal(text: str, classifier: Classifier, *, project_feasible: bool = True) -> GoalProfile:
    gp = GoalProfile.from_categories(validate_categories(classifier(text)))
    return gp.project_feasible() if project_feasible else gp


# --------------------------------------------------------------------------- #
# deterministic keyword classifier (no model) — baseline + test fixture
# --------------------------------------------------------------------------- #
# ordered (regex -> (axis, label)); first match per axis wins. Longer/more specific first.
_RULES: list[tuple[str, tuple[str, str]]] = [
    # shot size
    (r"\bextreme wide|establishing shot|very wide\b", ("shot_size", "extreme wide shot")),
    (r"\bwide shot|wide angle|full shot\b", ("shot_size", "wide shot")),
    (r"\bmedium[- ]wide|medium long\b", ("shot_size", "medium-wide shot")),
    (r"\bmedium close[- ]?up|med close\b", ("shot_size", "medium close-up")),
    (r"\bheadshot|extreme close[- ]?up\b", ("shot_size", "close-up")),
    (r"\btight on the (face|head)|just the (face|head)|tight on the subject\b", ("shot_size", "close-up")),
    (r"\bclose[- ]?up|closeup|tight shot\b", ("shot_size", "close-up")),
    (r"\bportrait\b", ("shot_size", "medium close-up")),
    (r"\bmedium shot|waist shot\b", ("shot_size", "medium shot")),
    # elevation
    (r"\bhigh[- ]angle|from above|looking down|bird'?s?[- ]eye|overhead\b", ("elevation", "high angle")),
    (r"\blow[- ]angle|from below|looking up|worm'?s?[- ]eye\b", ("elevation", "low angle")),
    (r"\beye[- ]level|eye height|straight on\b", ("elevation", "eye level")),
    # bearing (subject-relative)
    (r"\bfront[- ]right|three[- ]quarter front right\b", ("bearing", "front-right")),
    (r"\bfront[- ]left\b", ("bearing", "front-left")),
    (r"\bback[- ]right|rear right\b", ("bearing", "back-right")),
    (r"\bback[- ]left|rear left\b", ("bearing", "back-left")),
    (r"\bfrom behind|from the back|rear view|back view|back of|behind them\b", ("bearing", "back")),
    (r"\bprofile|from the side|side view|side[- ]on\b", ("bearing", "side")),
    (r"\bfrom the right\b", ("bearing", "right")),
    (r"\bfrom the left\b", ("bearing", "left")),
    (r"\bfrom the front|facing (the )?camera|head[- ]on|front view\b", ("bearing", "front")),
    # placement x
    (r"\b(upper|lower|top|bottom)[- ]?left\b", ("placement_x", "left third")),      # compound e.g. "lower left"
    (r"\b(upper|lower|top|bottom)[- ]?right\b", ("placement_x", "right third")),
    (r"\b(on the |to the )?left third|left side|on the left\b", ("placement_x", "left third")),
    (r"\b(on the |to the )?right third|right side|on the right\b", ("placement_x", "right third")),
    (r"\bcenter(ed)?|middle of (the )?frame|centred\b", ("placement_x", "centered")),
    # placement y
    (r"\bupper|top of (the )?frame|near the top\b", ("placement_y", "upper")),
    (r"\blower|bottom of (the )?frame|near the bottom\b", ("placement_y", "lower")),
    # body framing
    (r"\bfull body|full[- ]length|head to toe\b", ("body_framing", "full body in frame")),
    (r"\btightly cropped|cropped (in )?tight|just the (face|head)\b", ("body_framing", "tightly cropped")),
    (r"\bpartially (cut|cropped)|partly out of frame\b", ("body_framing", "partially cut off")),
]


def keyword_classifier(text: str) -> dict[str, str]:
    t = text.lower()
    cats: dict[str, str] = {}
    for pattern, (axis, label) in _RULES:
        if axis in cats:
            continue
        if re.search(pattern, t):
            cats[axis] = label
    return cats


def hybrid_classifier(text: str, llm: Classifier) -> dict[str, str]:
    """Deterministic keyword rules (precise on standard cinematography terms) take precedence; the LLM
    fills only the axes the rules left unset (robust to unusual phrasing). Best of both."""
    kw = keyword_classifier(text)
    ll = validate_categories(llm(text))
    return {**ll, **kw}


# --------------------------------------------------------------------------- #
# LLM classifier (transformers instruct model) — the deployment path
# --------------------------------------------------------------------------- #
_FEWSHOT = (
    'Request: "medium shot at eye level, facing the camera, centered"\n'
    'JSON: {"shot_size":["medium shot","medium shot"],"body_framing":null,'
    '"elevation":["eye level","at eye level"],"bearing":["front","facing the camera"],'
    '"placement_x":["centered","centered"],"placement_y":null}\n'
    'Request: "shoot them from above, three-quarter front-right, full body"\n'
    'JSON: {"shot_size":null,"body_framing":["full body in frame","full body"],'
    '"elevation":["high angle","from above"],"bearing":["front-right","three-quarter front-right"],'
    '"placement_x":null,"placement_y":null}\n'
    'Request: "an intimate portrait, tight on the face"\n'
    'JSON: {"shot_size":["close-up","tight on the face"],'
    '"body_framing":["tightly cropped","tight on the face"],"elevation":null,'
    '"bearing":null,"placement_x":null,"placement_y":null}\n'
)


def _prompt(text: str) -> str:
    axes = "\n".join(f'  "{ax}": [<one of {list(labels)}>, "<exact words from the request>"] or null'
                     for ax, labels in ALLOWED.items())
    return (
        "You convert a photographer's request into a fixed cinematography schema. Return ONLY a JSON "
        "object with EXACTLY these keys. For each attribute give BOTH the label AND the exact words "
        "from the request that state it — the quote is checked against the request, so an attribute "
        "with no literal supporting words must be null. Set a key to null unless the request "
        "EXPLICITLY states it; never guess. Use ONLY the listed labels verbatim. Bearing is the "
        "camera's view of the SUBJECT: front = we see their face, back = their back, side/left/right "
        "= a profile.\n"
        f"Schema:\n{{\n{axes}\n}}\n\n{_FEWSHOT}Request: {text!r}\nJSON:"
    )


class LLMClassifier:
    """Classify free text into cinematography categories with a cached instruct model."""

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda",
                 require_evidence: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.model_name = model
        try:  # VL checkpoints need the VL loader; fall back to a plain causal-LM
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(model)
            self.tok = self.proc.tokenizer
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map=device).eval()
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.device = device
        self.require_evidence = require_evidence

    def __call__(self, text: str) -> dict[str, str]:
        import torch

        msgs = [{"role": "user", "content": _prompt(text)}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=220, do_sample=False)
        raw = self.tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        return validate_categories(_parse_json_obj(raw, text if self.require_evidence else None))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _parse_json_obj(raw: str, source_text: str | None = None) -> dict[str, str]:
    """Parse the model's JSON. Values may be a bare label or [label, evidence].

    When `source_text` is given, an attribute is kept only if its EVIDENCE QUOTE actually occurs in
    the request (grounding check) — this is what suppresses the model asserting attributes the user
    never stated, the dominant LLM failure mode here (~30% of predicted axes were hallucinated)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    src = _norm(source_text) if source_text is not None else None
    out: dict[str, str] = {}
    for k, v in obj.items():
        label = evidence = None
        if isinstance(v, str):
            label = v
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            label = v[0]
            evidence = v[1] if len(v) > 1 and isinstance(v[1], str) else None
        if label is None:
            continue
        if src is not None:
            if not evidence:
                continue                              # unsupported claim -> drop
            ev = _norm(evidence)
            # accept if the quote (or any of its content words) is really in the request
            words = [w for w in ev.split() if len(w) > 2]
            if ev not in src and not (words and all(w in src for w in words)):
                continue
        out[k] = label
    return out
