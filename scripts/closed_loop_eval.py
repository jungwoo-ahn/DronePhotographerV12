"""Closed-loop evaluation: does the policy actually REACH the requested shot?

Training curves and the goal-dependence probe both say the policy *uses* the goal.
Neither says it *gets there*. v11 passed the first test and failed this one — its
world-model variant looked better on sampled action MSE while being worse in
rollout — so this is the measurement that decides anything.

Protocol, per episode:
  1. Pick a real (start, goal) pair from a held-out placement.
  2. Put the camera at the start pose in Blender and render what it sees.
  3. Ask the policy for an 8-step action chunk, conditioned on that frame and on
     the goal prompt.
  4. Execute the chunk in the Blender env, re-rendering as it goes.
  5. Measure how much closer to the goal pose the camera ended up.

The headline number is improvement over doing nothing:

    improvement = d(start, goal) − d(end, goal)

positive means the camera moved toward the requested shot. Reported against the
no-op baseline explicitly, because a policy that barely moves scores ~0 and a
policy that thrashes scores negative — both are failures that a raw distance
would flatter. `d` is the geometric pose distance from `src/common/reward.py`
(view angle, apparent size and aim, all in radians), the same metric the value
target uses.

Usage (see scripts/sbatch_closed_loop_eval.sh for the wrapper that sets the env):

    python scripts/closed_loop_eval.py \
        --checkpoint runs/train/.../checkpoints/iter_000005000 \
        --episodes 24 --out runs/closed_loop/iter5000.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)
os.chdir(V12)

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--episodes", type=int, default=24)
ap.add_argument("--chunks", type=int, default=1,
                help="action chunks executed per episode (each chunk is 8 steps)")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=30, help="flow-matching steps")
ap.add_argument("--guidance", type=float, default=1.0,
                help="1.0 = no CFG. Training used cfg_dropout_rate=0, so the model has "
                     "never seen an empty caption and a CFG uncond branch would be noise.")
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--root", default="data/trajectories")
ap.add_argument("--out", default="runs/closed_loop/eval.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true", help="load raw weights instead of EMA")
args = ap.parse_args()

# ---------------------------------------------------------------- policy
from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()  # must precede other cosmos imports

import torch  # noqa: E402

from cosmos_framework.data.generator.action.json_formatter import (  # noqa: E402
    ActionPromptJsonFormatter,
)
from cosmos_framework.inference.action import build_action_batch  # noqa: E402
from cosmos_framework.inference.args import OmniSetupOverrides  # noqa: E402
from cosmos_framework.inference.common.init import init_output_dir  # noqa: E402
from cosmos_framework.inference.inference import OmniInference  # noqa: E402
from cosmos_framework.scripts.action_policy_server_utils import (  # noqa: E402
    maybe_init_distributed,
)

from src.common.action_repr import apply_action_9d  # noqa: E402
from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS, goal_vector  # noqa: E402
from src.common.reward import CameraIntrinsics, pose_to_geometry, _geometry_distance  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402


def load_policy():
    maybe_init_distributed()
    wan = os.environ.get(
        "WAN_VAE_PATH",
        f"{V12}/repos/cosmos-framework/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth",
    )
    overrides = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint,
        "config_file": "cosmos_framework/configs/base/config.py",
        "experiment": "action_policy_camera_nano",
        "experiment_overrides": [f"model.config.tokenizer.vae_path={wan}"],
        "output_dir": str(Path(args.out).parent / "_infer"),
        "sampler": "unipc",
        "guardrails": False,          # a rollout loop must not pay for the safety models
        "use_ema_weights": not args.no_ema,
    })
    setup = overrides.build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec: np.ndarray) -> str:
    """The goal as the JSON string the model was trained on.

    Training prompts carry `idle_frame`, and the stock inference path drops it, so
    pass mode/idle_frames explicitly — otherwise the prompt drifts from training in
    a way that is invisible until the numbers come out wrong.
    """
    formatter = ActionPromptJsonFormatter()
    # __call__ takes the sample dict and REPLACES ai_caption with the JSON structure,
    # reading fps/size/idle from the same keys the dataset uses — hence `viewpoint`
    # and `conditioning_fps`, not `view_point`/`fps`.
    sample = {
        "ai_caption": goal_prompt(goal_vec),
        "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
        "image_size": torch.tensor([args.image_size, args.image_size]),
        "mode": "policy",
        "idle_frames": torch.tensor(0),
        "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                             dtype=torch.uint8),
    }
    out = formatter(sample)["ai_caption"]
    return out if isinstance(out, str) else json.dumps(out)


def predict_chunk(model, frame_uint8: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    """One 8x9 action chunk from the current observation + goal prompt.

    The batch is rebuilt every call on purpose: the model normalizes video in place
    and stamps `is_preprocessed`, so a reused dict silently takes the float branch.
    """
    video = torch.from_numpy(frame_uint8).permute(2, 0, 1).contiguous()      # [C,H,W]
    video = video.unsqueeze(1).repeat(1, args.chunk_size + 1, 1, 1)          # [C,T,H,W]
    batch = build_action_batch(
        video=video,
        action=torch.zeros(args.chunk_size, 64, dtype=torch.float32),
        raw_action_dim=9,
        prompt=prompt,
        view_point="ego_view",
        domain_name="camera_pose",
        model_mode="policy",
        action_chunk_size=args.chunk_size,
        fps=args.fps,
        input_video_key=model.config.input_video_key,
        batch_size=1,
        device="cuda",
    )
    with torch.no_grad():
        out = model.generate_samples_from_batch(
            batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed],
        )
    return out["action"][0].float().cpu().numpy()[:, :9]


def geometry_distance(position, forward, up, goal_view, intr) -> float:
    achieved = pose_to_geometry(
        position, forward, up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height,
    )
    goal = pose_to_geometry(
        goal_view.camera_position, goal_view.camera_forward, goal_view.camera_up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height,
    )
    return float(_geometry_distance(achieved, goal))


def pick_episodes() -> list:
    """Held-out (start, goal) pairs, one per placement for scene diversity."""
    dirs = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    rng = random.Random(args.seed)
    rng.shuffle(dirs)
    picked = []
    for name in dirs:
        if len(picked) >= args.episodes:
            break
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        path = os.path.join(args.root, name, "data.json")
        if not os.path.exists(path):
            continue
        try:
            windows = list(iter_goal_start_windows(path, chunk_size=args.chunk_size,
                                                   max_per_pair=2))
        except Exception:  # noqa: BLE001
            continue
        for w in windows:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if np.isfinite(g).all():
                picked.append((name, path, w, g))
                break
    return picked


def main() -> int:
    t0 = time.time()
    episodes = pick_episodes()
    print(f"[eval] {len(episodes)} episodes", flush=True)
    if not episodes:
        print("[eval] nothing to evaluate", flush=True)
        return 1

    model = load_policy()
    print(f"[eval] policy loaded ({time.time()-t0:.0f}s)", flush=True)
    intr = CameraIntrinsics.from_render(1024, 768)
    results = []

    for i, (name, path, window, goal_vec) in enumerate(episodes):
        try:
            renderer = SubprocessBlenderRenderer(repo_root=SHARED)
            env = BlenderRolloutEnv(run_info_path=path, renderer=renderer,
                                    object_position=window.start.object_position)
            start = window.start
            obs = env.reset(start.camera_position, start.camera_forward, start.camera_up)
            d0 = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)

            prompt = make_prompt(goal_vec)
            for c in range(args.chunks):
                frame = np.asarray(obs["image"])
                chunk = predict_chunk(model, frame, prompt, seed=args.seed + i)
                for step in chunk:
                    obs, _ = env.step(step, render=False)
                obs = env.reset(env.position, env.forward, env.up)   # render the new view

            d1 = geometry_distance(env.position, env.forward, env.up, window.goal_frame, intr)
            results.append({"placement": name, "d_start": d0, "d_end": d1,
                            "improvement": d0 - d1})
            print(f"[eval] {i+1}/{len(episodes)} {name[:44]:44s} "
                  f"d {d0:.4f} -> {d1:.4f}  improvement {d0-d1:+.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] {i+1} FAILED {name[:40]}: {type(exc).__name__}: {exc}", flush=True)

    if not results:
        print("[eval] every episode failed", flush=True)
        return 1
    imp = np.array([r["improvement"] for r in results])
    summary = {
        "checkpoint": args.checkpoint,
        "episodes": len(results),
        "mean_improvement_over_noop": float(imp.mean()),
        "median_improvement": float(np.median(imp)),
        "frac_positive": float((imp > 0).mean()),
        "mean_d_start": float(np.mean([r["d_start"] for r in results])),
        "mean_d_end": float(np.mean([r["d_end"] for r in results])),
        "guidance": args.guidance, "num_steps": args.num_steps,
        "chunks": args.chunks, "ema": not args.no_ema,
        "elapsed_s": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "episodes": results}, open(args.out, "w"), indent=1)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
