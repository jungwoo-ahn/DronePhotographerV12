"""Goal-authoring front-ends: turn a user's natural language or a reference image into the
canonical (partial) shot-profile goal that conditions the Cosmos policy.

- `vocab`        — cinematography category tables (single source of truth) + profile<->categories<->NL
- `goal_profile` — `GoalProfile` (values + specified mask) + feasibility projection
- `from_language`  (Module 1) — natural language -> GoalProfile (keyword + LLM classifier)
- `pose_bearing`   — subject bearing from body-pose keypoints (feature extraction + fitted regressor)
- `from_reference` (Module 2) — reference image -> GoalProfile (YOLO-pose bbox + bearing; elevation
  left unspecified as it is not single-image-recoverable)
"""
from src.goal_authoring import vocab
from src.goal_authoring.goal_profile import GoalProfile, feasibility_project

__all__ = ["GoalProfile", "feasibility_project", "vocab"]
