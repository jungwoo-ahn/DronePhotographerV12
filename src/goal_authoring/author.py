"""`author_goal` — one entry point: natural language and/or a reference image -> a COMPLETE goal
profile and the exact prompt the policy was trained on.

    goal = author_goal(nl="a close-up from the front")
    goal = author_goal(image="ref.jpg", nl="but from a low angle")      # NL overrides the image
    goal.prompt        # 'Move the camera to achieve this shot: ...' — training shape, always
    goal.source        # per key: given / default / rule / bank / bank(anchored)
    goal.warnings      # what was assumed, relaxed or clamped

Pipeline: classify / detect -> partial `GoalProfile` (+ `exact` for measured numbers) -> complete
(`completion.BankCompleter` or `RuleCompleter`) -> optional clamp to the trained support -> serialize.
The result is asserted to be complete and to match `serialize.TRAINING_PROMPT_RE`; anything else
would be a prompt shape the policy has never seen, which is the failure this module replaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping

from src.common.goal_space import DEFAULT_GOAL_KEYS
from src.goal_authoring.completion import Completed, make_completer
from src.goal_authoring.from_language import Classifier, keyword_classifier, language_to_goal
from src.goal_authoring.goal_profile import GoalProfile
from src.goal_authoring.serialize import CROP_KEYS, is_training_shape

# Keys clamped to the trained support when the rule completer (which knows no data) is used.
# Quantiles rather than min/max: the p0.5 / p99.5 band still covers 99% of what the policy saw
# and drops the handful of stragglers that would otherwise define "possible".
SUPPORT_KEYS = ("occupancy", "cam_to_obj_elevation_deg", "body_in_frame_ratio", "visible_frac")
SUPPORT_QUANTILES = (0.005, 0.995)


@dataclass
class AuthoredGoal:
    request: GoalProfile              # what was asked (partial)
    profile: GoalProfile              # what will be sent (complete)
    prompt: str
    source: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    method: str = ""
    support: int | None = None
    anchor: int | None = None

    @property
    def categories(self) -> dict[str, str]:
        return self.profile.categories()

    def goal_vec(self) -> list[float]:
        return [float(self.profile.values[k]) for k in DEFAULT_GOAL_KEYS]

    def goal_crop(self) -> dict[str, float]:
        return {k: float(self.profile.values[k]) for k in CROP_KEYS}

    def to_record(self) -> dict:
        """JSON-able record in the shape `scripts/recon_from_reference.py` consumes
        (`goal_vec` / `goal_specified` / `goal_crop`), plus provenance."""
        return {
            "goal_keys": list(DEFAULT_GOAL_KEYS),
            "goal_vec": self.goal_vec(),
            "goal_specified": sorted(self.profile.specified),
            "goal_crop": self.goal_crop(),
            "prompt": self.prompt,
            "categories": self.categories,
            "request": {"values": dict(self.request.values), "specified": sorted(self.request.specified),
                        "exact": sorted(self.request.exact), "categories": self.request.categories()},
            "source": dict(self.source),
            "warnings": list(self.warnings),
            "method": self.method, "support": self.support, "anchor": self.anchor,
        }

    def summary(self) -> str:
        lines = [f"request    : {self.request.categories() or '(nothing parsed)'}",
                 f"method     : {self.method}"
                 + (f"  (support {self.support} training goals, anchor row {self.anchor})"
                    if self.support is not None else "")]
        for k in (*DEFAULT_GOAL_KEYS, *CROP_KEYS):
            v = self.profile.values[k]
            lines.append(f"  {k:26s} {v:9.2f}   {self.source.get(k, '?')}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        lines.append(f"prompt     : {self.prompt}")
        return "\n".join(lines)


def _support_ranges() -> dict[str, tuple[float, float]] | None:
    from src.goal_authoring.goal_bank import GoalBank
    try:
        bank = GoalBank.load()
    except FileNotFoundError:
        return None
    tr = bank.train_mask()
    return {k: tuple(bank.quantiles(k, SUPPORT_QUANTILES, tr)) for k in SUPPORT_KEYS}


def _clamp_request(request: GoalProfile, ranges: Mapping[str, tuple[float, float]],
                   warnings: list[str]) -> GoalProfile:
    """Pull the REQUESTED numbers into the trained range before the geometry is solved, so the
    derived keys (box, crop fractions) stay consistent with the clamped value. Clamping after
    the fact would leave a box sized for 88% next to an occupancy number that says 68%."""
    vals = dict(request.values)
    for k, (lo, hi) in ranges.items():
        if k not in vals:
            continue
        v = float(vals[k])
        c = min(max(v, lo), hi)
        if abs(c - v) > 1e-6:
            warnings.append(f"{k} {v:.2f} is outside the trained support [{lo:.2f}, {hi:.2f}]; clamped to {c:.2f}")
            vals[k] = c
    return GoalProfile(vals, request.specified, request.exact)


def _note_derived_outside_support(profile: GoalProfile, ranges: Mapping[str, tuple[float, float]],
                                  warnings: list[str]) -> None:
    for k, (lo, hi) in ranges.items():
        v = float(profile.values[k])
        if v < lo - 1e-6 or v > hi + 1e-6:
            warnings.append(f"derived {k} {v:.2f} falls outside the trained support [{lo:.2f}, {hi:.2f}] "
                            "(a geometric consequence of the request; not altered)")


def complete_request(request: GoalProfile, completer, *, clamp_support: bool | str = "auto") -> Completed:
    """The completion step exactly as `author_goal` runs it: for the rule completer, clamp the
    requested numbers to the trained support first (when a bank is available), complete, then
    note derived values that fall outside the support. Benchmarks call this directly."""
    warnings: list[str] = []
    ranges = None
    if completer.method == "rules" and clamp_support in ("auto", True):
        ranges = _support_ranges()
        if ranges is None:
            if clamp_support is True:
                raise FileNotFoundError("clamp_support=True but no goal bank is available")
            warnings.append("support clamp skipped: no goal bank available")
        else:
            request = _clamp_request(request, ranges, warnings)
    done: Completed = completer.complete(request)
    if ranges is not None:
        _note_derived_outside_support(done.profile, ranges, warnings)
    done.warnings = warnings + list(done.warnings)
    return done


def author_goal(
    nl: str | None = None,
    image=None,
    *,
    classifier: Classifier = keyword_classifier,
    estimator=None,
    method: str = "bank",
    completer=None,
    clamp_support: bool | str = "auto",
) -> AuthoredGoal:
    """Natural language and/or a reference image -> complete goal + training-shape prompt.

    `image` may be a path or an array (anything the YOLO estimator accepts); `estimator` is a
    `ReferenceEstimator` to reuse across calls (it loads models). NL is applied AFTER the image,
    so words override what was read from the picture ("like this, but from a low angle").

    `method` picks the completer: "bank" (data-driven, needs `assets/goal_bank_v5.npz`) or
    "rules" (dataset-independent). With "rules", `clamp_support` pulls the result into the
    trained range when a bank is available ("auto"), always (True) or never (False).
    """
    if nl is None and image is None:
        raise ValueError("author_goal needs natural language, a reference image, or both")

    request = GoalProfile({}, frozenset())
    warnings: list[str] = []
    if image is not None:
        if estimator is None:
            from src.goal_authoring.from_reference import ReferenceEstimator
            estimator = ReferenceEstimator()
        request = estimator(image)
        if not request.specified:
            warnings.append("no subject detected in the reference image; nothing taken from it")
    if nl:
        parsed = language_to_goal(nl, classifier, project_feasible=False)
        if not parsed.specified:
            warnings.append(f"no cinematography attribute recognised in {nl!r}")
        request = request.merge(parsed)

    completer = completer or make_completer(method)
    done = complete_request(request, completer, clamp_support=clamp_support)
    warnings.extend(done.warnings)
    profile = done.profile

    if not profile.is_complete():
        raise AssertionError(f"completion left keys missing: {profile.missing()}")
    prompt = profile.to_prompt()
    if not is_training_shape(prompt):
        raise AssertionError(f"authored prompt is not in the training shape: {prompt}")
    return AuthoredGoal(request, profile, prompt, dict(done.source), warnings, completer.method,
                        done.support, done.anchor)


__all__ = ["AuthoredGoal", "author_goal", "complete_request", "SUPPORT_KEYS", "SUPPORT_QUANTILES"]
