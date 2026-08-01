"""Cosmos 3 dataset for our `camera_pose` policy — reads the LeRobot export.

Lives here rather than inside `repos/cosmos-framework` so it stays version-controlled
(the vendored repo is gitignored); the training config points `L(...)` straight at
`get_camera_pose_sft_dataset` below, so nothing in the framework needs patching.

Shape of one sample (built by `ActionBaseDataset._build_result`):
    video   uint8 [C, chunk_length+1, H, W]
    action  float32 [chunk_length, 9]      raw (see `action_normalization` below)
    ai_caption str                         our goal prompt
    plus mode / domain_id / conditioning_fps / viewpoint / idle_frames

`camera_pose` is a registered embodiment (domain_id 2, raw action dim 9), and 9D =
`build_action_spec(Pos(), Rot("rot6d"))` — position + 6D rotation, NO gripper. Do not pad a
dummy gripper channel: it would send `compute_idle_frames` down the gripper branch and break
the raw_action_dim=9 contract the inference server assumes.

`action_normalization=None` (raw) is deliberate, and matches Cosmos's own camera_pose usage
(translation_scale 1.0, rotation dims skipped). Measured on our data: rot6d sits at the
identity (dims 0/4 near 1 with std 0.002) because per-step rotations are small (3.5 deg
mean), so a p99 scale is meaningless for it; and scaling translation alone to +-1 would bury
rotation ~50x in the flow loss — the DOF that aims the camera. Raw, translation (std <=0.14)
and rotation (std ~0.04) sit within ~3.4x.
"""

from __future__ import annotations

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
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.datasets.base_dataset import (  # noqa: E402
    ActionBaseDataset,
)

CAMERA_ACTION_DIM = 9
VIDEO_KEY = "observation.images.image"
DOMAIN_NAME = "camera_pose"


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
        self._episode_indices = sorted(self._episodes)

    # ---- ActionBaseDataset contract -------------------------------------------------
    @property
    def action_dim(self) -> int:
        return CAMERA_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"))      # 9D, no gripper

    @classmethod
    def _stats_path(cls) -> Path:
        # Only consulted when action_normalization is not None; we feed raw actions.
        return (Path(__file__).resolve().parents[2]
                / "runs" / "camera_pose_action_stats.json")

    def __len__(self) -> int:
        return len(self._episode_indices)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Contiguous blocks safe to shuffle — one per sample, since episodes don't overlap."""
        return [(i, i + 1) for i in range(len(self._episode_indices))]

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
