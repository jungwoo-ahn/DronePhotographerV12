"""Cosmos 3 dataset for our `camera_pose` policy — reads the LeRobot export.

Lives here rather than inside `repos/cosmos-framework` so it stays version-controlled
(the vendored repo is gitignored); the training config points `L(...)` straight at
`get_camera_pose_sft_dataset` below, so nothing in the framework needs patching.

Shape of one sample (built by `ActionBaseDataset._build_result`):
    video   uint8 [C, chunk_length+1, H, W]
    action  float32 [chunk_length, 10]     raw; dims 0-8 pose, dim 9 shoot
    ai_caption str                         our goal prompt
    plus mode / domain_id / conditioning_fps / viewpoint / idle_frames

Actions are 10D = `build_action_spec(Pos(), Rot("rot6d"), Gripper())` — position, 6D
rotation, and the SHOOT channel on dim 9. The gripper slot is not a dummy pad: it carries a
latched 0/1 "I have arrived, take the photo" state (see `dataset_base.shoot_column`), because
the policy cannot otherwise signal termination — measured in docs/v4_session_changes.md
section 11, its final chunk moves as much as the previous one, so no threshold on action
magnitude separates "arrived" from "still travelling".

The two things an earlier version of this note warned about are now handled deliberately
rather than avoided:
  * `compute_idle_frames` gains a GRIPPER branch (`max |dgripper| < eps_g`). Idle is an AND
    across dim types, so this can only make it STRICTER. It matters because idle_frames
    reaches the model through the prompt in policy mode (json_formatter.py
    `_should_append_idle_frame_info`), so it was measured rather than assumed — over 800
    exported episodes: unchanged 80.4%, -1 frame 16.5%, **-3 frames 3.1%**. The -3 is not a
    bug and not the "at most one frame" this note first claimed: `min_streak=3` means the
    shoot transition landing inside an idle run of exactly 3 deletes the whole run. The
    stricter number is also the more correct one — the frame where the shoot state changes
    is genuinely not idle.
  * the raw-dim contract is satisfied by registering `camera_pose_shoot` below, instead of
    silently emitting 10 where the registry says 9.

`action_normalization=None` (raw) is deliberate, and matches Cosmos's own camera_pose usage
(translation_scale 1.0, rotation dims skipped). Measured on our data: rot6d sits at the
identity (dims 0/4 near 1 with std 0.002) because per-step rotations are small (3.5 deg
mean), so a p99 scale is meaningless for it; and scaling translation alone to +-1 would bury
rotation ~50x in the flow loss — the DOF that aims the camera. Raw, translation (std <=0.14)
and rotation (std ~0.04) sit within ~3.4x.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CF = Path(__file__).resolve().parents[2] / "repos" / "cosmos-framework"
if _CF.exists() and str(_CF) not in sys.path:
    sys.path.insert(0, str(_CF))

from cosmos_framework.data.generator.action.action_spec import (  # noqa: E402
    ActionSpec,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.datasets.base_dataset import (  # noqa: E402
    ActionBaseDataset,
)
from cosmos_framework.data.generator.action.domain_utils import (  # noqa: E402
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
)

V12_ROOT = Path(__file__).resolve().parents[2]
# Anchored to the REPO, not the cwd. `scripts/train_camera_policy.sh` does `cd $CF`
# (the cosmos-framework root) because Cosmos resolves several model configs by relative
# path, so a relative "configs/val_scenes.json" looks for it inside the framework
# checkout and dies. Caught by the 100-iter smoke, which is what that smoke is for.
DEFAULT_VAL_SCENES = str(V12_ROOT / "configs" / "val_scenes.json")

CAMERA_ACTION_DIM = 10                 # 3 pos + 6 rot6d + 1 shoot
VIDEO_KEY = "observation.images.image"
DOMAIN_NAME = "camera_pose_shoot"

# Registered from HERE rather than by editing the vendored registry. Adding a key leaves the
# stock `camera_pose` (9D) untouched and survives a framework update, where patching
# `EMBODIMENT_TO_RAW_ACTION_DIM["camera_pose"] = 10` would be reverted by the next sync and
# would also lie to anything else reading that table.
#
# The DOMAIN ID stays 2 — the same row `camera_pose` uses. That matters: `DomainAwareLinear`
# (mot/domain_aware_linear.py) stores per-domain weights as `nn.Embedding(num_domains,
# out*in)`, so `action2llm`/`llm2action` are a SEPARATE matrix per domain. Keeping id 2 means
# we keep fine-tuning the pretrained camera_pose row instead of starting a fresh one.
#
# The same fact is why the shoot channel inherits nothing from the gripper embodiments:
# whatever `droid_lerobot` (8) or `fractal` (20) learned at index 9 lives in THEIR rows. The
# 10D layout is NVIDIA's own (docs/action_policy_libero_posttrain.md: "10D = pos 3 + rot6d 6
# + gripper 1", "model emits [0,1]"), so we get the plumbing — not the weights.
EMBODIMENT_TO_DOMAIN_ID.setdefault(DOMAIN_NAME, EMBODIMENT_TO_DOMAIN_ID["camera_pose"])
EMBODIMENT_TO_RAW_ACTION_DIM.setdefault(DOMAIN_NAME, CAMERA_ACTION_DIM)


class CameraPoseLeRobotDataset(ActionBaseDataset):
    """One episode == one (start, goal) training sample; see `src/data/lerobot_export.py`."""

    def __init__(
        self,
        root: str,
        *,
        fps: float = 30.0,
        chunk_length: int = 8,
        image_size: int = 256,
        mode: str = "policy",
        action_normalization: str | None = None,
        tolerance_s: float = 1e-4,
        viewpoint: str = "ego_view",
        sample_stride: int = 1,
        video_key: str = VIDEO_KEY,
        video_backend: str | None = "pyav",
        split: str = "train",
        val_scenes: str | None = None,
        val_ratio: float | None = None,
    ) -> None:
        super().__init__(
            root=root,
            domain_name=DOMAIN_NAME,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        self._image_size = int(image_size)
        self._video_key = video_key
        # torchcodec is lerobot's default but needs FFmpeg *shared* libs (libavutil.so.*),
        # which this cluster image doesn't ship; PyAV bundles its own, so decode through it.
        self._video_backend = video_backend
        # Every episode is exactly chunk_length+1 frames, so episode index == sample index.
        all_episodes = sorted(self._episodes)
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if val_ratio is not None:
            raise ValueError(
                "val_ratio is gone. It took the last 5% of episode_index, which is "
                "placement-disjoint but scene-COMPLETE on both sides — measured at 88/88 "
                "scene overlap and 99/99 object overlap, so the val loss it produced was "
                "'a new camera path in a room the model already knows'. Use val_scenes."
            )

        # Scene-level holdout, read from the export's own provenance column.
        manifest = Path(val_scenes) if val_scenes else Path(DEFAULT_VAL_SCENES)
        if not manifest.is_absolute():
            manifest = V12_ROOT / manifest       # never relative to whatever cwd we are in
        val_set = frozenset(json.loads(manifest.read_text())["scenes"])
        missing = [i for i in all_episodes if "scene" not in self._episodes[i]]
        if missing:
            # Deliberately fatal. The tempting fallback — "no scene column, use the tail" —
            # is exactly the shape of bug that let two plan_rerender_yaw scripts evaluate a
            # retired gate for weeks: the run works, the number means something else.
            raise ValueError(
                f"{len(missing)} episodes in {root} have no 'scene' column, so a scene-level "
                "split cannot be built. This dataset predates provenance; re-export with "
                "scripts/export_lerobot.py (it writes 'scene' and 'placement')."
            )
        want_val = split == "val"
        self._episode_indices = [
            i for i in all_episodes
            if (self._episodes[i]["scene"] in val_set) == want_val
        ]
        if not self._episode_indices:
            raise ValueError(f"{split} split is empty (episodes={len(all_episodes)}, "
                             f"val_scenes={manifest})")
        self.split = split
        self.val_scenes = val_set

    # ---- ActionBaseDataset contract -------------------------------------------------
    @property
    def action_dim(self) -> int:
        return CAMERA_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())   # 10D, shoot on dim 9

    @classmethod
    def _stats_path(cls) -> Path:
        # Only consulted when action_normalization is not None; we feed raw actions.
        return (Path(__file__).resolve().parents[2]
                / "runs" / "camera_pose_action_stats.json")

    def __len__(self) -> int:
        return len(self._episode_indices)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Blocks for `ActionIterableShuffleDataset`, as ``(start, LENGTH)``.

        Not ``(start, end)`` — the consumer does ``range(start, start + length)``,
        so an end-exclusive pair silently walks off the end of the dataset.

        One block per sample here: each episode is exactly one training window, so
        there is no within-block sequence to preserve and every sample is free to
        be shuffled independently.
        """
        return [(i, 1) for i in range(len(self._episode_indices))]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_index = self._episode_indices[idx]
        episode = self._episodes[ep_index]
        start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])

        rows = self._rows[start:end]
        actions = np.stack([np.asarray(r["action"], dtype=np.float32) for r in rows])
        action = torch.from_numpy(actions[: self._chunk_length])       # drop the pad row

        timestamps = [float(r["timestamp"]) for r in rows]
        video = self._decode_video(episode, timestamps)                 # [T, C, H, W] in [0,1]

        task = episode.get("tasks") or [""]
        caption = task[0] if isinstance(task, (list, tuple)) else str(task)
        return self._build_result(
            mode=self._mode, video=video, action=action, ai_caption=caption,
            image_size=self._image_size,
        )

    # ---- helpers --------------------------------------------------------------------
    def _decode_video(self, episode: dict, timestamps: list[float]) -> torch.Tensor:
        """Frames for one episode out of the shared mp4.

        Episodes are packed into one file, so every timestamp is offset by this episode's
        `from_timestamp` — the same thing the LIBERO loader does.
        """
        from lerobot.datasets.video_utils import decode_video_frames

        path = self._video_path(episode, self._video_key)
        offset = float(episode.get(f"videos/{self._video_key}/from_timestamp", 0.0))
        frames = decode_video_frames(
            path, [t + offset for t in timestamps], self._tolerance_s,
            backend=self._video_backend,
        )
        if frames.shape[-1] != self._image_size or frames.shape[-2] != self._image_size:
            frames = torch.nn.functional.interpolate(
                frames, size=(self._image_size, self._image_size),
                mode="bilinear", align_corners=False,
            )
        return frames


def get_camera_pose_sft_dataset(
    root: str,
    *,
    chunk_length: int = 8,
    fps: float = 30.0,
    image_size: int = 256,
    mode: str = "policy",
    action_normalization: str | None = None,
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    format_prompt_as_json: bool = True,
    iterable_shuffle: bool = True,
    episode_shuffle_seed: int = 42,
    resolution: str | int | None = None,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    split: str = "train",
    val_scenes: str | None = None,
):
    """Factory the training config points at (mirrors `get_action_libero_sft_dataset`).

    `cfg_dropout_rate` defaults to 0: classifier-free guidance is OFF for this project.
    Blanking the caption teaches the policy to act WITHOUT the goal, which is the exact
    opposite of what we are trying to measure, and both v10 and v11 found guidance 1
    (i.e. no CFG) optimal anyway.
    """
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
        ActionIterableShuffleDataset,
        ActionSFTDataset,
    )

    base = CameraPoseLeRobotDataset(
        root=root, fps=fps, chunk_length=chunk_length, image_size=image_size,
        mode=mode, action_normalization=action_normalization,
        split=split, val_scenes=val_scenes,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        # Idle detection assumes metric translation (eps_t = 5e-3/fps). Our per-step
        # motion is ~0.08-0.14 m, far above it, so nothing is ever flagged idle — the
        # tag would just be noise in the prompt.
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(base, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


__all__ = [
    "CameraPoseLeRobotDataset",
    "get_camera_pose_sft_dataset",
    "CAMERA_ACTION_DIM",
    "DOMAIN_NAME",
]
