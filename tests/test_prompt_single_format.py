"""The inference prompt must be single-formatted, exactly like training's.

`build_action_batch` runs `ActionPromptJsonFormatter` on whatever it is handed, and the eval
scripts hand it an ALREADY-formatted JSON. That wrapped it twice: the model received an
868-char JSON whose actions[0].description was the entire escaped 573-char first JSON — a
shape training never produced, on every rollout this project has reported.

A second, quieter divergence hid behind it: without an `action` tensor in the formatter's
input dict, `_get_total_frames` returns None and `idle_frame` reads "0." where training says
"0 out of 8.". Nine characters, invisible to eyeballing, found only by diffing field-by-field
against the dataset's own output.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest
import torch

DATASET = Path("runs/lerobot_v5")
_needs = pytest.mark.skipif(not (DATASET / "meta" / "info.json").exists(),
                            reason="no exported dataset in this checkout")


def _formatter():
    from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
    return ActionPromptJsonFormatter()


def _eval_side(description: str, idle: int, chunk: int = 8) -> str:
    """The assembly `closed_loop_eval.make_prompt` performs."""
    f = _formatter()
    d = {
        "ai_caption": description,
        "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(30),
        "image_size": torch.tensor([256, 256]),
        "mode": "policy",
        "idle_frames": torch.tensor(idle),
        "action": torch.zeros(chunk, 1),
        "video": torch.zeros(3, chunk + 1, 256, 256, dtype=torch.uint8),
    }
    out = f(d)[f.caption_key]
    return out if isinstance(out, str) else json.dumps(out)


def _train_samples(n: int):
    from src.data.cosmos_camera_dataset import get_camera_pose_sft_dataset
    ds = get_camera_pose_sft_dataset(root=str(DATASET), split="val",
                                     iterable_shuffle=False, format_prompt_as_json=True)
    for s in itertools.islice(iter(ds), n):
        cap = s["ai_caption"]
        yield cap if isinstance(cap, str) else json.dumps(cap)


@_needs
def test_eval_assembly_reproduces_the_training_prompt_exactly():
    """Structure parity. `idle_frames` is fed the sample's true value here — inference cannot
    know it, which is a separate (measured) problem; this pins everything else."""
    n = ok = 0
    for cap in _train_samples(20):
        a = json.loads(cap)["actions"][0]
        idle = int(re.match(r"(\d+)", a["idle_frame"]).group(1))
        n += 1
        ok += _eval_side(a["description"], idle) == cap
    assert n and ok == n, f"{ok}/{n} matched; the eval assembly has drifted from training"


@_needs
def test_description_is_not_itself_json():
    """The double-wrap signature: a JSON document nested inside `description`."""
    for cap in _train_samples(5):
        desc = json.loads(cap)["actions"][0]["description"]
        assert not desc.lstrip().startswith("{"), "description is itself JSON — double-wrapped"
        assert desc.startswith("Move the camera to achieve this shot:"), desc[:60]


@_needs
def test_missing_action_length_degrades_idle_frame_text():
    """Pins WHY the `action` tensor is in the formatter dict, so nobody removes it as unused."""
    cap = next(iter(_train_samples(1)))
    desc = json.loads(cap)["actions"][0]["description"]
    f = _formatter()
    d = {"ai_caption": desc, "viewpoint": "ego_view", "conditioning_fps": torch.tensor(30),
         "image_size": torch.tensor([256, 256]), "mode": "policy",
         "idle_frames": torch.tensor(0),
         "video": torch.zeros(3, 9, 256, 256, dtype=torch.uint8)}       # no "action"
    out = f(d)[f.caption_key]
    out = out if isinstance(out, str) else json.dumps(out)
    assert json.loads(out)["actions"][0]["idle_frame"] == "0."
    assert json.loads(_eval_side(desc, 0))["actions"][0]["idle_frame"] == "0 out of 8."
