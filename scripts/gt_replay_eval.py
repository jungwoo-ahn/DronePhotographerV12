"""Can the policy reproduce a trajectory it was TRAINED on?

This is the sanity check that has to pass before a held-out rollout number means
anything. If the policy cannot reproduce the training data, a bad closed-loop
score tells you nothing about goal-conditioning — it only tells you the model
did not fit.

Two tests per episode:

  A. CHUNK REPRODUCTION (no rendering, no compounding)
     Feed the exact start frame and the exact goal prompt, and compare the
     predicted 8x9 chunk against `_compute_action_chunk(window)` — the literal
     supervision target for that sample. Errors here are pure fit: no
     autoregression, no render round-trip, no distribution shift.

  B. ROLLOUT TO GOAL (rendered, compounding)
     Roll the policy forward from the same start, re-rendering after each chunk,
     for as many chunks as the goal is away. Compare pose-by-pose against the GT
     path and track distance-to-goal.

The two differ in what they can blame. A small chunk error with a bad rollout is
compounding error / render mismatch. A large chunk error means the fit itself is
not there, and B is not worth reading.

Note on what "GT" means beyond step 8: the window's supervision covers start->end
(8 steps); the goal sits 8-32 frames further along the SAME recorded trajectory.
That recorded path is a *random* trajectory that happened to pass through a
well-framed view — it is a reference, not a correct answer. So B scores goal
attainment, and shows the GT path alongside for context.

Provenance: the export enumeration is deterministic given its seed, so replaying
it here recovers exactly which (placement, start, goal) became episode i. That is
verified against the exported parquet rather than assumed — see `_verify`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
# torch is imported after `init_script()` below, as the framework requires.

V12 = str(Path(__file__).resolve().parents[1])
SHARED = str(Path(V12).parent / "DronePhotographer")
CF_ROOT = f"{V12}/repos/cosmos-framework"
for p in (V12, CF_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--roots", nargs="+", default=["data/trajectories/v7_stage2_renders_lookat075"])
ap.add_argument("--lerobot", default="runs/lerobot_v1",
                help="exported dataset to verify the replay against")
ap.add_argument("--max-episodes", type=int, default=4000, help="must match the export")
ap.add_argument("--per-placement", type=int, default=4, help="must match the export")
ap.add_argument("--seed", type=int, default=0, help="must match the export")
ap.add_argument("--n", type=int, default=8, help="episodes to actually roll out")
ap.add_argument("--sector", default=None, help="restrict to one sector (e.g. back)")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--num-steps", type=int, default=35)
ap.add_argument("--samples", type=int, default=4,
                help="samples per chunk; the policy is stochastic, so a single draw "
                     "conflates sampling variance with fit error (v11's lesson)")
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--extra-chunks", type=int, default=1)
ap.add_argument("--out", default="runs/gt_replay/replay.json")
ap.add_argument("--rollout", action="store_true", default=True)
ap.add_argument("--no-rollout", dest="rollout", action="store_false",
                help="test A only (no Blender, much faster)")
args = ap.parse_args()

# Every path we were handed is relative to V12, but the framework must be imported
# and constructed from ITS OWN root — it resolves assets by relative path (e.g.
# model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json) and dies with
# FileNotFoundError otherwise. So absolutise ours first, then chdir.
args.roots = [str((Path(V12) / r).resolve()) for r in args.roots]
args.lerobot = str((Path(V12) / args.lerobot).resolve())
args.out = str((Path(V12) / args.out).resolve())
args.checkpoint = str(Path(args.checkpoint).resolve())
os.chdir(CF_ROOT)

from cosmos_framework.inference.common.init import init_script  # noqa: E402

init_script()  # must precede other cosmos imports

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
from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.dataset_base import (  # noqa: E402
    DEFAULT_EXCLUDE_OBJECTS, _compute_action_chunk, _window_object,
)
from src.common.facing import sector3, sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)
from src.common.reward import pose_to_geometry, _geometry_distance  # noqa: E402
from src.common.run_info import write_run_info  # noqa: E402
from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402
from src.utils.rotation_utils import matrix_from_rot6d  # noqa: E402


def abspath(p) -> Path:
    """Re-anchor an annotation path to the V12 root.

    `data.json` stores frame paths relative to V12 ("data/trajectories/.../x.jpg"),
    but we run from the framework root. Left relative, every `exists()` check below
    returns False and the enumeration yields ZERO episodes without raising — the
    failure would look like "no training data matched" rather than a path bug.
    """
    p = Path(p)
    return p if p.is_absolute() else Path(V12) / p


# ---------------------------------------------------------------- provenance
def replay_export() -> list[dict]:
    """Re-run the export's episode enumeration, keeping the window it discards.

    Mirrors `scripts/export_lerobot.py` call-for-call — same seed, same order of
    `random` consumption. Any divergence shows up in `_verify` as a prompt
    mismatch rather than as silently wrong "ground truth".
    """
    placements: list[tuple[str, Path]] = []
    for root in args.roots:
        r = Path(root)
        if not r.is_dir():
            continue
        for d in sorted(os.listdir(r)):
            p = r / d / "data.json"
            if p.exists():
                placements.append((d, p))
    random.seed(args.seed)
    random.shuffle(placements)

    episodes: list[dict] = []
    for name, path in placements:
        if len(episodes) >= args.max_episodes:
            break
        obj = name.split("__", 1)[1] if "__" in name else name
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        try:
            windows = list(iter_goal_start_windows(
                path, chunk_size=args.chunk_size, max_per_pair=args.per_placement,
            ))
        except Exception:  # noqa: BLE001
            continue
        random.shuffle(windows)
        taken = 0
        for w in windows:
            if taken >= args.per_placement or len(episodes) >= args.max_episodes:
                break
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            frames = [abspath(k.image) for k in w.keyframes]
            if not all(f.exists() for f in frames):
                continue
            episodes.append({
                "index": len(episodes), "placement": name, "data_path": str(path),
                "window": w, "goal_vec": g, "prompt": goal_prompt(g, crop=w.goal_frame.raw),
            })
            taken += 1
    return episodes


def _verify(episodes: list[dict]) -> dict:
    """Assert the replayed order matches the exported dataset, prompt for prompt.

    Without this the whole script is guesswork: an off-by-one in the enumeration
    would hand every episode a neighbour's goal and turn a correct policy into a
    broken-looking one.
    """
    meta = Path(args.lerobot) / "meta" / "episodes"
    files = sorted(meta.rglob("*.parquet"))
    if not files:
        return {"checked": 0, "note": f"no exported episodes under {meta}"}
    try:
        import pandas as pd
    except ImportError:
        return {"checked": 0, "note": "pandas unavailable"}
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
    n = min(len(df), len(episodes))
    match = 0
    first_bad = None
    for i in range(n):
        task = df.iloc[i]["tasks"]
        task = task[0] if isinstance(task, (list, tuple, np.ndarray)) else str(task)
        if str(task) == episodes[i]["prompt"]:
            match += 1
        elif first_bad is None:
            first_bad = {"i": i, "exported": str(task)[:120],
                         "replayed": episodes[i]["prompt"][:120]}
    return {"checked": n, "matched": match, "rate": match / n if n else 0.0,
            "first_mismatch": first_bad,
            "exported_episodes": int(len(df)), "replayed_episodes": len(episodes)}


# ---------------------------------------------------------------- policy
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
        "guardrails": False,
        "use_ema_weights": not args.no_ema,
    })
    setup = overrides.build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec: np.ndarray, crop=None) -> str:
    formatter = ActionPromptJsonFormatter()
    sample = {
        "ai_caption": goal_prompt(goal_vec, crop=crop),
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


def predict_chunks(model, frame_uint8: np.ndarray, prompt: str, seed: int,
                   k: int) -> np.ndarray:
    """`k` independent action chunks -> (k, chunk_size, 9).

    Drawing more than one is not optional here. v11 established that this policy's
    single-sample action error is dominated by multimodal sampling variance, so a
    one-draw comparison against GT would report variance and call it fit error.
    """
    from PIL import Image
    img = np.asarray(frame_uint8)
    if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
        img = np.asarray(Image.fromarray(img).resize(
            (args.image_size, args.image_size), Image.BILINEAR))
    video = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    video = video.unsqueeze(1).repeat(1, args.chunk_size + 1, 1, 1)
    out = []
    for j in range(k):
        batch = build_action_batch(
            video=video.clone(),
            action=torch.zeros(args.chunk_size, 64, dtype=torch.float32),
            raw_action_dim=CAMERA_ACTION_DIM, prompt=prompt, view_point="ego_view",
            # ModelMode, not "policy": the builder reads `model_mode.value`, and the
            # StrEnum compares equal to the string so nothing catches it earlier.
            domain_name=DOMAIN_NAME, model_mode=ModelMode.POLICY,
            action_chunk_size=args.chunk_size, fps=args.fps,
            input_video_key=model.config.input_video_key, batch_size=1, device="cuda",
        )
        with torch.no_grad():
            res = model.generate_samples_from_batch(
                batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed + j],
            )
        out.append(res["action"][0].float().cpu().numpy()[:, :CAMERA_ACTION_DIM])
    return np.stack(out)


# ---------------------------------------------------------------- metrics
def rot_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rot6d-encoded relative rotations."""
    Ra, Rb = matrix_from_rot6d(a), matrix_from_rot6d(b)
    tr = float(np.trace(Ra.T @ Rb))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))))


def chunk_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-step agreement between one predicted chunk and the GT chunk."""
    dt = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    gt_norm = np.linalg.norm(gt[:, :3], axis=1)
    cos = np.array([
        float(np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g)))
        if np.linalg.norm(p) > 1e-8 and np.linalg.norm(g) > 1e-8 else 0.0
        for p, g in zip(pred[:, :3], gt[:, :3])
    ])
    rot = np.array([rot_error_deg(p[3:], g[3:]) for p, g in zip(pred, gt)])
    return {
        "trans_err_mean": float(dt.mean()), "trans_err_per_step": dt.round(4).tolist(),
        "gt_step_norm_mean": float(gt_norm.mean()),
        "trans_err_rel": float(dt.mean() / gt_norm.mean()) if gt_norm.mean() > 1e-8 else float("nan"),
        "dir_cos_mean": float(cos.mean()), "dir_cos_per_step": cos.round(3).tolist(),
        "rot_err_deg_mean": float(rot.mean()), "rot_err_deg_per_step": rot.round(2).tolist(),
    }


def geometry_distance(position, forward, up, goal_view) -> float:
    achieved = pose_to_geometry(
        position, forward, up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height)
    goal = pose_to_geometry(
        goal_view.camera_position, goal_view.camera_forward, goal_view.camera_up,
        subject_center=goal_view.subject_center, subject_height=goal_view.subject_height)
    return float(_geometry_distance(achieved, goal))


def gt_path(window) -> list:
    """The recorded frames from start through the goal, in order."""
    records = list(window.keyframes)
    seen = {r.frame_idx for r in records}
    for r in window.future:
        if r.frame_idx not in seen and r.frame_idx <= window.goal_frame.frame_idx:
            records.append(r)
            seen.add(r.frame_idx)
    if window.goal_frame.frame_idx not in seen:
        records.append(window.goal_frame)
    return sorted(records, key=lambda r: r.frame_idx)


# ---------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Namespaced by the output file, NOT a shared "frames" dir. Two runs writing into
    # the same directory collide on ep{i}_{tag}.jpg and silently overwrite each
    # other's renders — the numbers stay correct (they come from env poses) but the
    # saved images end up belonging to a different scene, which only shows up as a
    # subject changing mid-strip in the report.
    frames_dir = out_path.parent / f"frames_{out_path.stem}"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("[replay] enumerating export ...", flush=True)
    episodes = replay_export()
    print(f"[replay] {len(episodes)} episodes replayed ({time.time()-t0:.0f}s)", flush=True)

    ver = _verify(episodes)
    print(f"[replay] provenance: {ver}", flush=True)
    if ver.get("checked") and ver.get("rate", 0) < 0.999:
        print("[replay] WARNING: replayed order does not match the export. "
              "GT below may belong to a different sample.", flush=True)

    # Spread the picks over placements and (if asked) one sector.
    picked = []
    seen_placements = set()
    for ep in episodes:
        if len(picked) >= args.n:
            break
        w = ep["window"]
        bearing = float(ep["goal_vec"][DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])
        sec = sector8(bearing)
        if args.sector and args.sector not in (sec, sector3(bearing)):
            continue
        if ep["placement"] in seen_placements:
            continue
        seen_placements.add(ep["placement"])
        ep["bearing"], ep["sector"] = bearing, sec
        picked.append(ep)
    print(f"[replay] rolling out {len(picked)} episodes", flush=True)

    model = load_policy()
    print(f"[replay] policy loaded ({time.time()-t0:.0f}s)", flush=True)

    from PIL import Image
    results = []
    for ep in picked:
        i, w = ep["index"], ep["window"]
        try:
            gt_chunk = _compute_action_chunk(w)                     # (8, 9) target
            start = w.start
            delta = abs(w.goal_frame.frame_idx - w.start_frame_idx)
            prompt = make_prompt(ep["goal_vec"], crop=w.goal_frame.raw)

            # ---- Test A: chunk reproduction, straight off the GT start frame.
            gt_start_img = np.asarray(Image.open(abspath(start.image)).convert("RGB"))
            preds = predict_chunks(model, gt_start_img, prompt,
                                   seed=1000 + i, k=args.samples)
            per_sample = [chunk_metrics(p, gt_chunk) for p in preds]
            mean_chunk = preds.mean(axis=0)
            best = int(np.argmin([m["trans_err_mean"] for m in per_sample]))
            test_a = {
                "gt_chunk": gt_chunk.round(4).tolist(),
                "pred_mean_chunk": mean_chunk.round(4).tolist(),
                "per_sample": per_sample,
                "mean_of_samples": chunk_metrics(mean_chunk, gt_chunk),
                "best_sample": per_sample[best],
                "spread_trans": float(np.std([m["trans_err_mean"] for m in per_sample])),
            }
            print(f"[replay] ep{i} A: trans_err {test_a['best_sample']['trans_err_mean']:.4f} "
                  f"(best of {args.samples}) | dir_cos {test_a['best_sample']['dir_cos_mean']:.3f} "
                  f"| rot {test_a['best_sample']['rot_err_deg_mean']:.2f}deg", flush=True)

            record = {
                "episode": i, "placement": ep["placement"], "object": _window_object(w),
                "sector": ep["sector"], "bearing": ep["bearing"], "delta": delta,
                "prompt": ep["prompt"],
                "start_frame_image": str(abspath(start.image)),
                "goal_frame_image": str(abspath(w.goal_frame.image)),
                "gt_path_images": [str(abspath(r.image)) for r in gt_path(w)],
                "test_a": test_a,
            }

            # ---- Test B: rendered rollout toward the goal.
            if args.rollout:
                renderer = SubprocessBlenderRenderer(repo_root=SHARED)
                run_info = write_run_info(ep["placement"], ep["data_path"],
                                          out_path.parent / "_run_info",
                                          shared_root=SHARED, resolution=args.image_size)
                env = BlenderRolloutEnv(run_info_path=run_info, renderer=renderer,
                                        object_position=start.object_position)
                obs = env.reset(start.camera_position, start.camera_forward, start.camera_up)
                d0 = geometry_distance(env.position, env.forward, env.up, w.goal_frame)
                trace, shots, poses = [d0], [], []

                def save(tag, image):
                    p = frames_dir / f"ep{i:03d}_{tag}.jpg"
                    Image.fromarray(np.asarray(image)).save(p, quality=82)
                    return str(p)

                shots.append({"tag": "start", "path": save("start", obs["image"]), "d": d0})
                n_chunks = int(np.ceil(delta / args.chunk_size)) + args.extra_chunks
                gt_records = gt_path(w)
                for c in range(n_chunks):
                    chunk = predict_chunks(model, np.asarray(obs["image"]), prompt,
                                           seed=2000 + i * 100 + c, k=args.samples).mean(axis=0)
                    for step in chunk:
                        obs, _ = env.step(step, render=False)
                    obs = env.reset(env.position, env.forward, env.up)
                    d = geometry_distance(env.position, env.forward, env.up, w.goal_frame)
                    trace.append(d)
                    poses.append(np.asarray(env.position).round(4).tolist())
                    # GT pose at the same step count, when the recorded path is that long.
                    k = (c + 1) * args.chunk_size
                    gt_pos = (np.asarray(gt_records[k].camera_position).round(4).tolist()
                              if k < len(gt_records) else None)
                    shots.append({"tag": f"chunk{c+1}", "path": save(f"chunk{c+1}", obs["image"]),
                                  "d": d, "gt_pos": gt_pos,
                                  "gt_image": str(abspath(gt_records[k].image)) if k < len(gt_records) else None})
                record["test_b"] = {
                    "chunks": n_chunks, "trace": [round(x, 4) for x in trace],
                    "d_start": d0, "d_end": trace[-1], "d_best": float(min(trace)),
                    "improvement": d0 - trace[-1], "best_improvement": d0 - float(min(trace)),
                    "shots": shots, "poses": poses,
                }
                print(f"[replay] ep{i} B: d {d0:.3f} -> {trace[-1]:.3f} "
                      f"(best {min(trace):.3f}) over {n_chunks} chunks", flush=True)

            results.append(record)
        except Exception as exc:  # noqa: BLE001
            print(f"[replay] ep{i} failed: {type(exc).__name__}: {exc}", flush=True)

    summary = {}
    if results:
        a_best = [r["test_a"]["best_sample"] for r in results]
        a_mean = [r["test_a"]["mean_of_samples"] for r in results]
        summary["test_a"] = {
            "n": len(results),
            "trans_err_best_mean": float(np.mean([m["trans_err_mean"] for m in a_best])),
            "trans_err_meanK_mean": float(np.mean([m["trans_err_mean"] for m in a_mean])),
            "trans_err_rel_best": float(np.mean([m["trans_err_rel"] for m in a_best])),
            "dir_cos_best_mean": float(np.mean([m["dir_cos_mean"] for m in a_best])),
            "dir_cos_meanK_mean": float(np.mean([m["dir_cos_mean"] for m in a_mean])),
            "rot_err_best_mean": float(np.mean([m["rot_err_deg_mean"] for m in a_best])),
            "gt_step_norm_mean": float(np.mean([m["gt_step_norm_mean"] for m in a_best])),
        }
        bs = [r["test_b"] for r in results if "test_b" in r]
        if bs:
            summary["test_b"] = {
                "n": len(bs),
                "mean_improvement": float(np.mean([b["improvement"] for b in bs])),
                "mean_best_improvement": float(np.mean([b["best_improvement"] for b in bs])),
                "frac_improved": float(np.mean([b["improvement"] > 0 for b in bs])),
                "mean_d_start": float(np.mean([b["d_start"] for b in bs])),
                "mean_d_end": float(np.mean([b["d_end"] for b in bs])),
            }

    out_path.write_text(json.dumps({
        "checkpoint": args.checkpoint, "provenance": ver,
        "frames_dir": str(frames_dir), "args": vars(args) | {"window": None},
        "summary": summary, "episodes": results,
    }, indent=2, default=str))
    print(f"\n=== SUMMARY ===\n{json.dumps(summary, indent=2)}", flush=True)
    print(f"[replay] wrote {out_path} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
