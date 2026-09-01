"""Goal-authoring front-ends: turn a user's natural language or a reference image into the
COMPLETE shot-profile goal — and the exact training-shape prompt — that conditions the Cosmos policy.

- `serialize`      — the goal->prompt contract (`goal_prompt`, the training shape regex, its inverse)
- `vocab`          — cinematography category tables (single source of truth) + profile<->categories
- `goal_profile`   — `GoalProfile` (values + specified mask + exact mask)
- `from_language`  (Module 1) — natural language -> partial GoalProfile (keyword + LLM classifier)
- `pose_bearing`   — subject bearing / camera elevation from body-pose keypoints (fitted regressors)
- `from_reference` (Module 2) — reference image -> partial GoalProfile (YOLO-pose bbox, bearing,
  elevation, crop side + body-proportion extent)
- `goal_bank`      — the trained goal distribution as data (parsed from the export's prompts)
- `completion`     — partial -> complete: `BankCompleter` (anchored on a real training goal) or
  `RuleCompleter` (dataset-independent person-box geometry)
- `author`         — `author_goal(nl=..., image=...)`, the one entry point
"""
from src.goal_authoring import vocab
from src.goal_authoring.goal_profile import GoalProfile, feasibility_project

__all__ = ["GoalProfile", "feasibility_project", "vocab"]


def author_goal(*args, **kwargs):
    """Lazy re-export of `src.goal_authoring.author.author_goal` (keeps package import light)."""
    from src.goal_authoring.author import author_goal as _author_goal
    return _author_goal(*args, **kwargs)
