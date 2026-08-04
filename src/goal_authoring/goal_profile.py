"""GoalProfile — the canonical (possibly PARTIAL) goal both authoring front-ends emit.

A goal profile carries numeric values only for the keys the user actually constrained
(`specified`); everything else is "don't care". Downstream this is (a) serialized to a
cinematography prompt describing only the specified attributes (partial goal -> shorter prompt),
and (b) scored in eval over only the specified keys. This is what lets natural-language goals
("a close-up") and reference images (which pin every geometric key) share one representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.common.goal_space import CYCLIC_GOAL_KEYS, DEFAULT_V5_RANGES
from src.goal_authoring import vocab


@dataclass(frozen=True)
class GoalProfile:
    values: dict[str, float]        # raw units; only the specified keys are present
    specified: frozenset[str]       # which profile keys the user constrained

    def __post_init__(self):
        object.__setattr__(self, "specified", frozenset(self.specified))
        # keep only specified keys in values (defensive)
        object.__setattr__(self, "values", {k: float(v) for k, v in self.values.items() if k in self.specified})

    # ---- constructors ----
    @classmethod
    def from_categories(cls, cats: Mapping[str, str]) -> "GoalProfile":
        vals, spec = vocab.categories_to_profile(cats)
        return cls(vals, spec)

    @classmethod
    def from_full_profile(cls, profile: Mapping[str, float], keys: frozenset[str] | None = None) -> "GoalProfile":
        """Wrap a fully-computed profile (e.g. Module 2 detection output). `keys` restricts what
        counts as specified (default: everything present)."""
        keys = frozenset(profile) if keys is None else frozenset(keys)
        return cls({k: float(profile[k]) for k in keys if k in profile}, keys & frozenset(profile))

    # ---- views ----
    def categories(self) -> dict[str, str]:
        return vocab.profile_to_categories(self.values)

    def to_nl(self, *, numbers: bool = True, coarse_bearing: bool = False) -> str:
        return vocab.profile_to_nl(self.values, self.specified, numbers=numbers, coarse_bearing=coarse_bearing)

    def to_json_goal(self) -> dict[str, str]:
        return vocab.profile_to_json_goal(self.values, self.specified)

    def is_partial(self) -> bool:
        return not frozenset(vocab.AXIS_KEY.values()).issubset(self.specified)

    # ---- combination: `other` overrides on its specified keys (e.g. "like this ref, but tighter") ----
    def merge(self, other: "GoalProfile") -> "GoalProfile":
        vals = dict(self.values); vals.update(other.values)
        return GoalProfile(vals, self.specified | other.specified)

    def project_feasible(self, ranges: Mapping[str, tuple[float, float]] = DEFAULT_V5_RANGES) -> "GoalProfile":
        return GoalProfile(feasibility_project(self.values, ranges), self.specified)


def feasibility_project(
    values: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]] = DEFAULT_V5_RANGES,
) -> dict[str, float]:
    """Clamp each value into the achievable range so an authored goal can't request the impossible.
    Cyclic keys (subject bearing) wrap mod 360 instead of clamping."""
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
