"""GoalProfile — the canonical (possibly PARTIAL) goal both authoring front-ends emit.

A goal profile carries numeric values only for the keys the user actually constrained
(`specified`); everything else is "don't care" — until it reaches the policy. The policy was
trained on FULL prompts only (every key, a crop phrase, a `visible` number: see
`serialize.TRAINING_PROMPT_RE`), so a partial profile is an intermediate: `completion` turns it
into a complete one before anything is serialized for inference. What the profile still buys
is that natural language ("a close-up") and reference images (which pin every geometric key)
share one representation, and that the completion step can see exactly what was asked for.

`exact` marks the specified keys whose NUMBER is a measurement (a detected bbox, an explicit
value) rather than a category centroid. A completer keeps exact numbers as they are and is
free to replace a centroid by a value the policy has actually seen in that category.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

from src.common.goal_space import CYCLIC_GOAL_KEYS, DEFAULT_V5_RANGES
from src.goal_authoring import vocab
from src.goal_authoring.serialize import FULL_PROFILE_KEYS, is_complete as _is_complete, profile_to_prompt


@dataclass(frozen=True)
class GoalProfile:
    values: dict[str, float]        # raw units; only the specified keys are present
    specified: frozenset[str]       # which profile keys the user constrained
    exact: frozenset[str] = frozenset()   # specified keys whose number is a measurement

    def __post_init__(self):
        object.__setattr__(self, "specified", frozenset(self.specified))
        # keep only specified keys in values (defensive)
        object.__setattr__(self, "values", {k: float(v) for k, v in self.values.items() if k in self.specified})
        object.__setattr__(self, "exact", frozenset(self.exact) & self.specified)

    # ---- constructors ----
    @classmethod
    def from_categories(cls, cats: Mapping[str, str]) -> "GoalProfile":
        vals, spec = vocab.categories_to_profile(cats)
        return cls(vals, spec)

    @classmethod
    def from_full_profile(cls, profile: Mapping[str, float], keys: frozenset[str] | None = None,
                          *, exact: bool = True) -> "GoalProfile":
        """Wrap a fully-computed profile (e.g. Module 2 detection output). `keys` restricts what
        counts as specified (default: everything present). Numbers are measurements, so they
        are `exact` unless told otherwise."""
        keys = frozenset(profile) if keys is None else frozenset(keys)
        spec = keys & frozenset(profile)
        return cls({k: float(profile[k]) for k in spec}, spec, spec if exact else frozenset())

    # ---- views ----
    def categories(self) -> dict[str, str]:
        return vocab.profile_to_categories(self.values)

    def to_prompt(self) -> str:
        """The training serializer's sentence for this profile.

        Complete profile -> the exact shape the policy was trained on. Partial profile -> the
        clauses for the specified keys only, which is NOT a shape the policy has seen; it is
        fine for display, and `completion` is the step that makes it a real prompt."""
        return profile_to_prompt(self.values, None if self.is_complete() else self.specified)

    def to_nl(self, **_ignored) -> str:
        """Deprecated alias of `to_prompt`. The old `vocab.profile_to_nl` sentence was never the
        one the policy trained on; keeping the name lets older scripts print the right thing."""
        warnings.warn("GoalProfile.to_nl() is deprecated; use to_prompt()", DeprecationWarning, stacklevel=2)
        return self.to_prompt()

    def to_json_goal(self) -> dict[str, str]:
        return vocab.profile_to_json_goal(self.values, self.specified)

    def is_partial(self) -> bool:
        return not frozenset(vocab.AXIS_KEY.values()).issubset(self.specified)

    def is_complete(self) -> bool:
        """Every key a training prompt carries (8 goal keys + top/bot cut + visible) is present."""
        return _is_complete(self.values)

    def missing(self) -> tuple[str, ...]:
        return tuple(k for k in FULL_PROFILE_KEYS if k not in self.specified)

    # ---- combination: `other` overrides on its specified keys (e.g. "like this ref, but tighter") ----
    def merge(self, other: "GoalProfile") -> "GoalProfile":
        vals = dict(self.values); vals.update(other.values)
        exact = (self.exact - other.specified) | other.exact
        return GoalProfile(vals, self.specified | other.specified, exact)

    def project_feasible(self, ranges: Mapping[str, tuple[float, float]] = DEFAULT_V5_RANGES) -> "GoalProfile":
        return GoalProfile(feasibility_project(self.values, ranges), self.specified, self.exact)


def feasibility_project(
    values: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]] = DEFAULT_V5_RANGES,
) -> dict[str, float]:
    """Clamp each value into the achievable range so an authored goal can't request the impossible.
    Cyclic keys (subject bearing) wrap mod 360 instead of clamping.

    `DEFAULT_V5_RANGES` is the SCORER's range (occupancy 0-100, elevation -90..90), not the
    trained one — a goal bank knows the latter (`completion` / `author` clamp to it)."""
    out: dict[str, float] = {}
    for k, v in values.items():
        v = float(v)
        if k in CYCLIC_GOAL_KEYS:
            out[k] = v % 360.0
        elif k in ranges:
            lo, hi = ranges[k]
            out[k] = min(max(v, lo), hi)
        else:
            out[k] = v
    return out
