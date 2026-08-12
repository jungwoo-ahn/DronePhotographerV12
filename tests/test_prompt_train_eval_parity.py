"""The eval prompt must be byte-identical to the one training saw.

This is the check that would have caught the missing `crop=`. Every rollout number
reported before it was measured by asking the policy in a prompt shape it had never
been trained on — the training string ends `..., cropped below the waist. (… visible
0.52)` and the eval string simply stopped after the composition clause.

Byte equality, not "looks equivalent": the failure was a dropped suffix, and before
that a comma. `goal_prompt` joins its first two clauses with a SPACE and the rest with
", ", so a uniform join reads fine to a human and matches 0/306 against the exporter.

Reads the real exported parquet when it is present and skips otherwise, so a checkout
without `runs/lerobot_v4/` still collects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.annotations import iter_goal_start_windows
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object
from src.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector
from src.data.lerobot_export import goal_prompt

DATASET = Path("runs/lerobot_v5")
TASKS = DATASET / "meta" / "tasks.parquet"


def _exported_prompts() -> set[str]:
    """LeRobot v3.0 writes the task STRING as the frame index and keeps only
    `task_index` as a column, so reading `df[df.columns[0]]` returns the integer ids
    and matches nothing. Read the index."""
    import pandas as pd
    df = pd.read_parquet(TASKS)
    if "task" in df.columns:
        return set(df["task"].astype(str))
    return set(df.index.astype(str))


def _roots() -> list[Path]:
    """The placement dirs the export enumerated, from its own recorded config."""
    info = DATASET / "meta" / "info.json"
    roots = []
    if info.exists():
        cfg = json.loads(info.read_text())
        roots = [Path(r) for r in cfg.get("source_roots", [])]
    if not roots:
        from src.common.dataset_base import DEFAULT_TRAJ_ROOT
        roots = [Path(DEFAULT_TRAJ_ROOT)]
    return roots


@pytest.mark.skipif(not TASKS.exists(), reason="no exported dataset in this checkout")
def test_eval_prompt_is_byte_identical_to_training():
    exported = _exported_prompts()
    assert exported, "exported task table is empty"

    checked = 0
    for root in _roots():
        if not root.is_dir():
            continue
        for placement in sorted(p for p in root.iterdir() if p.is_dir())[:40]:
            obj = placement.name.split("__", 1)[1] if "__" in placement.name else placement.name
            if obj in DEFAULT_EXCLUDE_OBJECTS:
                continue
            data = placement / "data.json"
            if not data.exists():
                continue
            for w in iter_goal_start_windows(data, chunk_size=8, max_per_pair=2):
                g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS,
                                object_key=_window_object(w))
                if not np.isfinite(g).all():
                    continue
                # The eval path: goal VECTOR + the goal frame's raw dict, exactly as
                # closed_loop_eval.make_prompt / gt_replay_eval build it.
                eval_prompt = goal_prompt(g, crop=w.goal_frame.raw)
                if eval_prompt in exported:
                    checked += 1
                    if checked >= 25:
                        return
    assert checked >= 25, (
        f"only {checked} eval-built prompts matched an exported training prompt; "
        "the two paths have diverged"
    )


def test_crop_clause_is_present_in_training_prompts():
    """Guards the specific regression: a prompt with no crop clause is not what
    training used, so an eval that produces one is asking out of distribution."""
    g = np.zeros(len(DEFAULT_GOAL_KEYS))
    g[DEFAULT_GOAL_KEYS.index("occupancy")] = 40.0
    raw = {"top_cut_frac": 0.0, "bot_cut_frac": 0.31, "visible_frac": 0.69}

    with_crop = goal_prompt(g, crop=raw)
    without = goal_prompt(g)

    assert "cropped" in with_crop and "visible" in with_crop
    assert with_crop != without
    assert with_crop.startswith(without.rstrip(". ").split(". (")[0][:20])
