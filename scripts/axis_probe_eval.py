"""Does the policy move ONE goal axis at a time, or does everything come along?

The aggregate rollout number says the camera gets closer to the goal. It cannot say
whether the policy actually understands the goal's individual dimensions — a policy
that only learned "drift toward a nicer-looking framing" would score positively while
being useless to steer.

So: hold a real start frame, take its OWN achieved profile as the base goal, then ask
for the same thing with exactly one key changed. A working policy should move the
requested axis in the requested direction and leave the others roughly alone.

Conditions per start frame:

  null              goal == the frame's own profile. The camera is already there, so
                    the correct action is to hold still. This is the cleanest possible
                    test of the stopping failure — no distance to close, nothing to
                    confuse it — and needs no separate experiment.
  <axis> +/- delta  one key shifted, everything else identical.

Measurement is pure pose math via `pose_to_geometry` (az / el / size / aim_x / aim_y),
so no rendering or re-scoring is needed to read the outcome, and the numbers do not
inherit the scorer's off-screen clamp:

    occupancy               -> size      (apparent size)
    subject_bearing_deg     -> az        (orbit angle; compared modulo 360)
    cam_to_obj_elevation_deg-> el
    object_center_x         -> aim_x
    object_center_y         -> aim_y

Reported per axis: the response on the requested DOF, and the crosstalk on the rest.
Both matter — a large response with large crosstalk is not steering, it is drifting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)
CF_ROOT = f"{V12}/repos/cosmos-framework"
sys.path.insert(0, CF_ROOT)
os.chdir(CF_ROOT)

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--episodes", type=int, default=5, help="start frames to probe")
ap.add_argument("--chunks", type=int, default=2)
ap.add_argument("--samples", type=int, default=2, help="policy draws per chunk, averaged")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--root", default=f"{V12}/data/trajectories/v7_stage2_renders_lookat075")
ap.add_argument("--out", default=f"{V12}/runs/axis_probe/probe.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true")
args = ap.parse_args()

from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()

import torch  # noqa: E402

from cosmos_framework.data.generator.action.json_formatter import (  # noqa: E402
    ActionPromptJsonFormatter,
)
from cosmos_framework.inference.action import build_action_batch  # noqa: E402
from cosmos_framework.inference.args import ModelMode, OmniSetupOverrides  # noqa: E402
from cosmos_framework.inference.common.init import init_output_dir  # noqa: E402
from cosmos_framework.inference.inference import OmniInference  # noqa: E402
from cosmos_framework.scripts.action_policy_server_utils import (  # noqa: E402
    maybe_init_distributed,
)

from src.common.annotations import iter_goal_start_windows  # noqa: E402
from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402
from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.dataset_base import DEFAULT_EXCLUDE_OBJECTS, _window_object  # noqa: E402
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)
from src.common.reward import pose_to_geometry  # noqa: E402
from src.common.run_info import write_run_info  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402

# key -> (delta, geometry DOF it should move, sign of that DOF per +delta)
# The sign column is what makes "responded correctly" checkable rather than a guess:
# asking for MORE occupancy must make the subject apparently BIGGER, not merely change.
AXES = {
    "occupancy":                (20.0,  "size",  +1.0),
    "subject_bearing_deg":      (45.0,  "az",    -1.0),   # bearing = front - az
    "cam_to_obj_elevation_deg": (15.0,  "el",    +1.0),
    "object_center_x":          (150.0, "aim_x", +1.0),
    "object_center_y":          (100.0, "aim_y", +1.0),
}
DOFS = ("size", "az", "el", "aim_x", "aim_y")
CLIP = {"occupancy": (0.0, 100.0), "body_in_frame_ratio": (0.0, 100.0),
        "cam_to_obj_elevation_deg": (-89.0, 89.0),
        "object_center_x": (0.0, 1024.0), "object_center_y": (0.0, 768.0),
        "bbox_x_offset": (0.0, 1024.0), "bbox_y_offset": (0.0, 768.0)}


def load_policy():
    maybe_init_distributed()
    wan = os.environ.get("WAN_VAE_PATH",
                         f"{CF_ROOT}/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth")
    setup = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint,
        "config_file": "cosmos_framework/configs/base/config.py",
        "experiment": "action_policy_camera_nano",
        "experiment_overrides": [f"model.config.tokenizer.vae_path={wan}"],
        "output_dir": str(Path(args.out).parent / "_infer"),
        "sampler": "unipc", "guardrails": False, "use_ema_weights": not args.no_ema,
    }).build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec: np.ndarray) -> str:
    sample = {
        "ai_caption": goal_prompt(goal_vec), "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
        "image_size": torch.tensor([args.image_size, args.image_size]),
        "mode": "policy", "idle_frames": torch.tensor(0),
        # length hint only: _get_total_frames reads action.shape[0], and without it
        # idle_frame reads "0." where training says "0 out of 8."
        "action": torch.zeros(args.chunk_size, 1, dtype=torch.float32),
        "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                             dtype=torch.uint8),
    }
    out = ActionPromptJsonFormatter()(sample)["ai_caption"]
    return out if isinstance(out, str) else json.dumps(out)


def predict(model, frame, prompt, seed):
    from PIL import Image
    img = np.asarray(frame)
    if img.shape[0] != args.image_size:
        img = np.asarray(Image.fromarray(img).resize(
            (args.image_size, args.image_size), Image.BILINEAR))
    video = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    video = video.unsqueeze(1).repeat(1, args.chunk_size + 1, 1, 1)
    acc = []
    for j in range(args.samples):
        batch = build_action_batch(
            video=video.clone(), action=torch.zeros(args.chunk_size, 64),
            raw_action_dim=CAMERA_ACTION_DIM, prompt=prompt, view_point="ego_view",
            domain_name=DOMAIN_NAME, model_mode=ModelMode.POLICY,
            action_chunk_size=args.chunk_size, fps=args.fps,
            input_video_key=model.config.input_video_key, batch_size=1, device="cuda")
        # build_action_batch re-formats whatever it is handed, and `prompt` is already the
        # formatted JSON -> double-wrapped, a shape training never produced. See
        # closed_loop_eval.predict_chunk for the measurement.
        b["ai_caption"] = [prompt] * len(b["ai_caption"])
        with torch.no_grad():
            r = model.generate_samples_from_batch(
                batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed + j])
        acc.append(r["action"][0].float().cpu().numpy()[:, :CAMERA_ACTION_DIM])
    return np.stack(acc).mean(axis=0)


def geom(env, view) -> dict[str, float]:
    return pose_to_geometry(env.position, env.forward, env.up,
                            subject_center=view.subject_center,
                            subject_height=view.subject_height)


def ddeg(a: float, b: float) -> float:
    """Signed difference of two angles (radians in, degrees out), wrapped to +-180."""
    return math.degrees(math.atan2(math.sin(a - b), math.cos(a - b)))


def delta_dof(g1: dict, g0: dict) -> dict[str, float]:
    """Change per DOF. Angles wrap; size/aim are plain differences."""
    return {
        "az": ddeg(g1["az"], g0["az"]),
        "el": ddeg(g1["el"], g0["el"]),
        "size": math.degrees(g1["size"] - g0["size"]),
        "aim_x": math.degrees(g1["aim_x"] - g0["aim_x"]),
        "aim_y": math.degrees(g1["aim_y"] - g0["aim_y"]),
    }


def main() -> int:
    t0 = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)))
    random.Random(args.seed).shuffle(dirs)
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
            ws = list(iter_goal_start_windows(path, chunk_size=args.chunk_size,
                                              max_per_pair=2))
        except Exception:  # noqa: BLE001
            continue
        for w in ws:
            base = goal_vector(w.start.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(base).all():
                continue
            picked.append((name, path, w, base))
            break
    print(f"[probe] {len(picked)} start frames", flush=True)
    if not picked:
        return 1

    model = load_policy()
    print(f"[probe] policy loaded ({time.time()-t0:.0f}s)", flush=True)

    conditions = [("null", None, 0.0)]
    for k, (d, _dof, _sgn) in AXES.items():
        conditions += [(k, k, +d), (k, k, -d)]

    rows = []
    for ei, (name, path, w, base) in enumerate(picked):
        renderer = SubprocessBlenderRenderer(repo_root=SHARED)
        run_info = write_run_info(name, path, out.parent / "_run_info",
                                  shared_root=SHARED, resolution=args.image_size)
        for cond, key, delta in conditions:
            try:
                env = BlenderRolloutEnv(run_info_path=run_info, renderer=renderer,
                                        object_position=w.start.object_position)
                obs = env.reset(w.start.camera_position, w.start.camera_forward,
                                w.start.camera_up)
                g0 = geom(env, w.start)

                goal = base.copy()
                if key is not None:
                    i = DEFAULT_GOAL_KEYS.index(key)
                    v = float(goal[i]) + delta
                    if key == SUBJECT_BEARING_KEY:
                        v %= 360.0
                    elif key in CLIP:
                        lo, hi = CLIP[key]
                        v = min(max(v, lo), hi)
                    goal[i] = v
                prompt = make_prompt(goal)

                for c in range(args.chunks):
                    chunk = predict(model, np.asarray(obs["image"]), prompt,
                                    seed=args.seed + ei * 1000 + c)
                    for step in chunk:
                        obs, _ = env.step(step, render=False)
                    obs = env.reset(env.position, env.forward, env.up)
                g1 = geom(env, w.start)

                d = delta_dof(g1, g0)
                dof = AXES[key][1] if key else None
                sgn = AXES[key][2] if key else 0.0
                resp = d[dof] * math.copysign(1.0, delta) * sgn if key else 0.0
                cross = {k2: v for k2, v in d.items() if k2 != dof}
                rows.append({
                    "episode": ei, "placement": name, "condition": cond,
                    "key": key, "delta": delta, "dof": dof,
                    "response": resp,
                    "moved_total": float(np.sqrt(sum(v * v for v in d.values()))),
                    "crosstalk": float(np.sqrt(sum(v * v for v in cross.values()))),
                    "per_dof": {k2: round(v, 3) for k2, v in d.items()},
                })
                tag = f"{cond}{'' if key is None else f'{delta:+.0f}'}"
                print(f"[probe] ep{ei} {tag:<28} response {resp:+8.2f}  "
                      f"crosstalk {rows[-1]['crosstalk']:7.2f}  "
                      f"moved {rows[-1]['moved_total']:7.2f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] ep{ei} {cond} failed: {type(exc).__name__}: {exc}", flush=True)

    summary = {}
    nulls = [r for r in rows if r["condition"] == "null"]
    if nulls:
        summary["null_moved_mean"] = float(np.mean([r["moved_total"] for r in nulls]))
        summary["null_n"] = len(nulls)
    for k in AXES:
        rs = [r for r in rows if r["key"] == k]
        if not rs:
            continue
        summary[k] = {
            "n": len(rs),
            "response_mean": float(np.mean([r["response"] for r in rs])),
            "correct_sign_frac": float(np.mean([r["response"] > 0 for r in rs])),
            "crosstalk_mean": float(np.mean([r["crosstalk"] for r in rs])),
            "response_over_crosstalk": float(
                np.mean([r["response"] for r in rs])
                / (np.mean([r["crosstalk"] for r in rs]) or 1e-9)),
        }
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "summary": summary,
                               "rows": rows}, indent=1))
    print(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=1)}", flush=True)
    print(f"[probe] wrote {out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
