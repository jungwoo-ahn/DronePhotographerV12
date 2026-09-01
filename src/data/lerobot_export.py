"""Export `goal_start` samples as a LeRobot v3.0 dataset for Cosmos 3 action post-training.

MAPPING DECISION — one EPISODE per sample. A Cosmos action sample is
(video window, action chunk, one text prompt), and our goal is a frame elsewhere in the
trajectory that only ever reaches the model through the prompt. Since the same start frame
pairs with MANY different goals, a whole trajectory cannot be one episode without losing
that multiplicity. So each (start, goal) pair becomes its own `chunk_size + 1`-frame episode
whose task string is the goal prompt.

That would mean millions of tiny mp4s, which GPFS would hate, so many episodes are packed
into ONE mp4 and referenced by `videos/<key>/from_timestamp` — exactly what that field is
for (`libero_lerobot_dataset.py` adds it to every frame timestamp before decoding).

Layout written (schema verified against the shipped `*_lerobot_example` assets):

    <out>/meta/info.json
    <out>/meta/tasks.parquet                       task_index + task string
    <out>/meta/episodes/chunk-000/file-000.parquet episode_index, tasks, length,
                                                   dataset_from/to_index, video ptrs
    <out>/data/chunk-000/file-000.parquet          index, episode_index, frame_index,
                                                   timestamp, task_index, action[9]
    <out>/videos/observation.images.image/chunk-000/file-000.mp4

Actions are stored as the framewise-relative 9D delta (the LIBERO pattern: the dataset just
slices them), already in the OpenCV camera frame — `src.common.action_repr.encode_action_9d`
emits exactly what `pose_abs_to_rel(..., "rot6d", "backward_framewise")` would.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence

import numpy as np

from src.common.action_repr import ACTION_DIM

# What gets WRITTEN: the 9 pose dims plus the shoot channel. `ACTION_DIM` stays 9 because it
# describes `encode_action_9d`'s output, which is still pose-only — widening it there would
# silently change every caller of the pose encoder. Written width is a property of the
# dataset, so it is named here. Matches NVIDIA's own reference layout for this framework
# (docs/action_policy_libero_posttrain.md: "10D = pos 3 + rot6d 6 + gripper 1").
WRITTEN_ACTION_DIM = ACTION_DIM + 1

VIDEO_KEY = "observation.images.image"
DEFAULT_FPS = 30
# Cosmos reads the instruction from `ai_caption`, splitting on " | " and picking one at
# random per window — a free paraphrase-augmentation hook.
PARAPHRASE_SEP = " | "


# The serializer is the goal->prompt CONTRACT and now lives with the authoring code
# (`src/goal_authoring/serialize.py`); it is re-exported here so the exporter, the eval
# scripts and the parity tests keep the import path they already use. Same bytes, same
# function object — `tests/test_serialize_contract.py` pins that.
from src.goal_authoring.serialize import crop_phrase, goal_prompt  # noqa: E402,F401


@dataclass
class EpisodeSpec:
    """One training sample: the frames to show, the actions to predict, the goal prompt."""
    frame_paths: list[Path]          # chunk_size + 1 images
    actions: np.ndarray              # (chunk_size, 9) float32
    prompt: str
    # Where this sample came from, "<scene>__<object>". Carried so the split can be defined
    # on it: without provenance in the written dataset the only thing a loader can slice on
    # is episode_index, and a contiguous slice of that is placement-disjoint but
    # scene-complete on both sides — it was measured at 88/88 scene overlap. The value is
    # already in `frame_paths`; it was simply being thrown away here.
    placement: str = ""

    @property
    def scene(self) -> str:
        """Scene half of the placement name. `_split_key(level="scene")` uses the same rule."""
        return self.placement.split("__")[0]


def _ffmpeg_exe() -> str:
    """System ffmpeg, else the wheel-bundled one (the cluster image ships neither)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode_mp4(
    frames: Iterable[Path], out_path: Path, fps: int, resize: int | None = None
) -> None:
    """Concatenate frames into one h264/yuv420p mp4 (the codec the shipped assets use).

    `resize` writes square `resize x resize` frames. The dataset resizes to `image_size`
    anyway, so encoding at the training resolution avoids storing (and decoding) the full
    1024x768 renders — a large saving at hundreds of thousands of frames.
    `-g 1` makes every frame a keyframe, so seeking to an arbitrary episode is exact.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    listing = out_path.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in frames))
    cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-r", str(fps), "-i", str(listing)]
    if resize:
        cmd += ["-vf", f"scale={resize}:{resize}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1", str(out_path)]
    subprocess.run(cmd, check=True)
    listing.unlink()


def write_lerobot_dataset(
    episodes: Sequence[EpisodeSpec],
    out_dir: str | Path,
    *,
    fps: int = DEFAULT_FPS,
    robot_type: str = "blender_camera",
    overwrite: bool = False,
    resize: int | None = 256,
    episodes_per_video: int = 20000,
) -> Path:
    """Write `episodes` as a LeRobot v3.0 dataset rooted at `out_dir`."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (pass overwrite=True)")
        shutil.rmtree(out)
    if not episodes:
        raise ValueError("no episodes to write")

    chunk = len(episodes[0].actions)
    for ep in episodes:
        if len(ep.actions) != chunk or len(ep.frame_paths) != chunk + 1:
            raise ValueError("every episode must share chunk_size and have chunk+1 frames")

    # tasks: unique prompts -> task_index
    prompts: list[str] = []
    task_index: dict[str, int] = {}
    for ep in episodes:
        if ep.prompt not in task_index:
            task_index[ep.prompt] = len(prompts)
            prompts.append(ep.prompt)

    # Shard the video across several mp4s instead of one file for everything.
    # Every __getitem__ seeks into its episode by `from_timestamp`, and that seek gets
    # more expensive as the file grows: a 150k-episode export packed into a single
    # 3.7 GB / 1.35M-frame mp4 halved training throughput (240 iter/h vs 480 for the
    # 1.1 GB / 47k-episode one), with both ranks parked in futex_wait on the dataloader
    # while the GPUs sat at 0-39% util and CPU at 46%. Neither compute nor CPU-bound —
    # seek cost. Smaller files keep each seek local.
    n_files = max(1, math.ceil(len(episodes) / max(1, episodes_per_video)))
    file_of: list[int] = []
    for fi in range(n_files):
        group = episodes[fi * episodes_per_video : (fi + 1) * episodes_per_video]
        _encode_mp4([p for ep in group for p in ep.frame_paths],
                    out / f"videos/{VIDEO_KEY}/chunk-000/file-{fi:03d}.mp4",
                    fps, resize=resize)
        file_of.extend([fi] * len(group))

    rows, ep_rows = [], []
    global_index = 0
    # from_timestamp is relative to the episode's OWN mp4, so the cursor resets per file
    frame_cursor = 0
    cur_file = 0
    for ep_idx, ep in enumerate(episodes):
        if file_of[ep_idx] != cur_file:
            cur_file = file_of[ep_idx]
            frame_cursor = 0
        start_index = global_index
        for f in range(chunk + 1):
            action = (ep.actions[f] if f < chunk
                      else np.zeros(WRITTEN_ACTION_DIM, dtype=np.float32))  # pad the last frame
            rows.append({
                "index": global_index,
                "episode_index": ep_idx,
                "frame_index": f,
                "timestamp": np.float32(f / fps),
                "task_index": task_index[ep.prompt],
                "action": action.astype(np.float32).tolist(),
            })
            global_index += 1
        ep_rows.append({
            "episode_index": ep_idx,
            "tasks": [ep.prompt],
            "length": chunk + 1,
            # Provenance. LeRobot ignores unknown columns and the framework loader keeps
            # whole rows (`{int(row["episode_index"]): row}` in base_dataset.py), so these
            # ride along with no loader change — and give the dataset something to split on
            # that is not episode order.
            "placement": ep.placement,
            "scene": ep.scene,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start_index,
            "dataset_to_index": global_index,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            f"videos/{VIDEO_KEY}/chunk_index": 0,
            f"videos/{VIDEO_KEY}/file_index": file_of[ep_idx],
            # where this episode starts inside the shared mp4 — load-bearing
            f"videos/{VIDEO_KEY}/from_timestamp": float(frame_cursor / fps),
            f"videos/{VIDEO_KEY}/to_timestamp": float((frame_cursor + chunk + 1) / fps),
        })
        frame_cursor += chunk + 1

    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "data/chunk-000/file-000.parquet")

    (out / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows),
                   out / "meta/episodes/chunk-000/file-000.parquet")

    # v3.0 tasks.parquet: task string is the index, task_index the only real column
    pd.DataFrame({"task_index": list(range(len(prompts)))}, index=prompts).to_parquet(
        out / "meta/tasks.parquet"
    )

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(prompts),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [WRITTEN_ACTION_DIM],
                       "names": ["dx", "dy", "dz", "r0", "r1", "r2",
                                 "r3", "r4", "r5", "shoot"]},
            VIDEO_KEY: {"dtype": "video", "shape": [None, None, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    (out / "meta/info.json").write_text(json.dumps(info, indent=1))
    return out


__all__ = ["EpisodeSpec", "write_lerobot_dataset", "goal_prompt", "VIDEO_KEY"]
