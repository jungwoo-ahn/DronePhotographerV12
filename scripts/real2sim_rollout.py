"""Roll the policy out inside a real reconstruction, and measure the real-to-sim gap.

Everything upstream of this has been checked: the reconstruction imports, stands at metric
scale, has a floor, renders at capture brightness once its baked lighting is driven as
Emission, accepts our subject asset at 1.893 m, and produces goal profiles that land inside
the training distribution on 6 of 7 keys (scripts/real2sim_probe.py). What has never been
tested is the only thing that matters: whether the policy can fly to a requested shot in a
room that was captured rather than designed.

Goals come from the scene itself. A pair of `is_camera_valid` anchors around one subject
placement gives (start, goal); the goal's profile is scored with the same
`project_verts_to_bbox` + `compute_v5_scores` the training data used, and serialised with the
same `goal_prompt`, so the policy is asked in exactly the words it was trained on.

The distance metric is `src/common/reward.py`'s geometric pose distance, identical to
closed_loop_eval, so the numbers here sit beside the 138-episode held-out result rather than
in their own units.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
sys.path.insert(0, V12)
CF_ROOT = f"{V12}/repos/cosmos-framework"
sys.path.insert(0, CF_ROOT)
os.chdir(CF_ROOT)

import numpy as np  # noqa: E402

from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--scene-blend", required=True, help="baked by scripts/recon_to_blend.py")
ap.add_argument("--scene-meta", required=True)
ap.add_argument("--object-blend", required=True)
ap.add_argument("--episodes", type=int, default=24)
ap.add_argument("--placements", type=int, default=8)
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--chunks", type=int, default=0, help="0 = adaptive from the anchor gap")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--stop-on-shoot", type=int, default=1)
ap.add_argument("--shoot-threshold", type=float, default=0.5)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--episodes-json", dest="episodes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--resume", type=int, default=1)
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

from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS  # noqa: E402
from src.common.reward import _geometry_distance, pose_to_geometry  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402

OUT = Path(args.out); OUT.parent.mkdir(parents=True, exist_ok=True)
FR = OUT.parent / f"frames_{OUT.stem}"; FR.mkdir(parents=True, exist_ok=True)


def load_policy():
    maybe_init_distributed()
    wan = os.environ.get("WAN_VAE_PATH", f"{CF_ROOT}/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth")
    ov = OmniSetupOverrides.model_validate({
        "checkpoint_path": args.checkpoint,
        "config_file": "cosmos_framework/configs/base/config.py",
        "experiment": "action_policy_camera_nano",
        "experiment_overrides": [f"model.config.tokenizer.vae_path={wan}"],
        "output_dir": str(OUT.parent / "_infer"),
        "sampler": "unipc", "guardrails": False, "use_ema_weights": not args.no_ema,
    })
    setup = ov.build_setup(); init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup); pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec, crop=None, *, specified=None, idle_frames=0) -> str:
    f = ActionPromptJsonFormatter()
    s = {"ai_caption": goal_prompt(np.asarray(goal_vec, dtype=np.float32), crop=crop,
                                   specified=specified),
         "viewpoint": "ego_view",
         "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
         "image_size": torch.tensor([args.image_size, args.image_size]),
         "mode": "policy",
         "action": torch.zeros(args.chunk_size, 1, dtype=torch.float32),
         "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                              dtype=torch.uint8)}
    if idle_frames is not None:
        s["idle_frames"] = torch.tensor(int(idle_frames))
    o = f(s)["ai_caption"]
    return o if isinstance(o, str) else json.dumps(o)


def predict_chunk(model, frame, prompt, seed):
    from PIL import Image
    img = np.asarray(frame)
    if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
        img = np.asarray(Image.fromarray(img).resize((args.image_size,)*2, Image.BILINEAR))
    video = torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(1).repeat(
        1, args.chunk_size + 1, 1, 1)
    batch = build_action_batch(
        video=video, action=torch.zeros(args.chunk_size, 64, dtype=torch.float32),
        raw_action_dim=CAMERA_ACTION_DIM, prompt=prompt, view_point="ego_view",
        domain_name=DOMAIN_NAME, model_mode=ModelMode.POLICY,
        action_chunk_size=args.chunk_size, fps=args.fps,
        input_video_key=model.config.input_video_key, batch_size=1, device="cuda")
    batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
    with torch.no_grad():
        draws = []
        for k in range(max(1, args.samples)):
            r = model.generate_samples_from_batch(
                batch, guidance=args.guidance, num_steps=args.num_steps,
                seed=[seed + 100_000 * k])
            draws.append(r["action"][0].float().cpu().numpy())
    chunk = (draws[0] if len(draws) == 1 else np.mean(draws, axis=0))[:, :CAMERA_ACTION_DIM]
    assert chunk.shape[1] == CAMERA_ACTION_DIM, chunk.shape
    return chunk


def main() -> int:
    from PIL import Image
    meta = json.loads(Path(args.scene_meta).read_text())
    # Episodes are precomputed by scripts/real2sim_make_episodes.py, which lives in the
    # DronePhotographer repo because it needs `bpy` and `src.blender.camera`; this process
    # has neither, so they arrive as JSON.
    epf = Path(args.episodes)
    if not epf.exists():
        print(f"[real2sim] missing {epf} — run scripts/real2sim_make_episodes.py first")
        return 1
    episodes = json.loads(epf.read_text())["episodes"]
    print(f"[real2sim] {len(episodes)} episodes from {meta['name']}", flush=True)

    run_info = {
        "input_scene": str(Path(args.scene_blend).resolve()),
        "input_object": str(Path(args.object_blend).resolve()),
        "scene_scale": 1.0, "scale": 1.0, "rotation_xyz_rad": [0.0, 0.0, 0.0],
        "options": {"object_position": None, "resolution": [args.image_size]*2},
    }
    rip = OUT.parent / f"_run_info_{OUT.stem}.json"

    model = load_policy()
    renderer = SubprocessBlenderRenderer(repo_root=SHARED)
    # Same construction closed_loop_eval.py:484 uses, and the same resolution the goals were
    # SCORED at (real2sim_make_episodes runs at 1024x768). The distance metric turns pixel
    # scores into angles through these focal lengths, so intrinsics that disagreed with the
    # scoring resolution would silently rescale every distance on this page.
    # `pose_to_geometry` is pure pose math and takes no intrinsics -- unlike
    # `profile_to_geometry`, which decodes bbox pixels and does. Passing them here raised
    # TypeError three minutes into the first GPU run.
    results = []
    done = set()
    part = OUT.with_suffix(".partial.json")
    if args.resume and part.exists():
        try:
            results = json.loads(part.read_text()).get("episodes", [])
            done = {r["key"] for r in results}
            print(f"[real2sim] resuming: {len(results)} done", flush=True)
        except Exception:
            results = []

    t0 = time.time()
    for i, ep in enumerate(episodes):
        if ep["key"] in done:
            continue
        run_info["options"]["object_position"] = ep["subject_foot"]
        rip.write_text(json.dumps(run_info))
        env = BlenderRolloutEnv(str(rip), renderer, object_position=ep["subject_centre"])
        obs = env.reset(ep["start_pos"], ep["start_fwd"], ep["start_up"])
        gpos, gfwd, gup = ep["goal_pos"], ep["goal_fwd"], ep["goal_up"]
        gg = pose_to_geometry(np.array(gpos), np.array(gfwd), np.array(gup),
                              subject_center=np.array(ep["subject_centre"]),
                              subject_height=ep["subject_height"])

        def dist():
            g = pose_to_geometry(env.position, env.forward, env.up,
                                 subject_center=np.array(ep["subject_centre"]),
                                 subject_height=ep["subject_height"])
            return float(_geometry_distance(g, gg))

        d0 = dist(); trace = [d0]
        shots = [{"tag": "start", "d": d0,
                  "path": str(FR / f"ep{i:03d}_start.jpg")}]
        Image.fromarray(np.asarray(obs["image"])).save(shots[-1]["path"], quality=88)
        prompt = make_prompt(ep["goal_vec"], crop=ep.get("goal_crop"),
                             specified=ep.get("goal_specified"))
        n_chunks = args.chunks or max(1, int(np.ceil(ep["delta_steps"] / args.chunk_size)))
        stop = None
        for c in range(n_chunks):
            chunk = predict_chunk(model, np.asarray(obs["image"]), prompt,
                                  seed=args.seed + 1000*i + c)
            sh = float(np.max(chunk[:, 9]))
            if stop is None and sh > args.shoot_threshold:
                stop = c
                if args.stop_on_shoot:
                    break
            for st in chunk[:, :9]:
                obs, _ = env.step(st, render=False)
            obs = env.reset(env.position, env.forward, env.up)
            d = dist(); trace.append(d)
            p = str(FR / f"ep{i:03d}_chunk{c+1}.jpg")
            Image.fromarray(np.asarray(obs["image"])).save(p, quality=88)
            shots.append({"tag": f"chunk{c+1}", "d": d, "path": p})
        results.append({"key": ep["key"], "scene": meta["name"], "sector": ep.get("sector"),
                        "delta_steps": ep["delta_steps"], "declared_stop": stop,
                        "d_start": d0, "d_end": trace[-1], "d_best": float(min(trace)),
                        "improvement": d0 - trace[-1],
                        "best_improvement": d0 - float(min(trace)),
                        "trace": [round(x, 4) for x in trace], "shots": shots})
        print(f"[real2sim] {i+1}/{len(episodes)} d {d0:.3f}->{trace[-1]:.3f} "
              f"(best {min(trace):.3f}) stop={stop}  {time.time()-t0:.0f}s", flush=True)
        part.write_text(json.dumps({"partial": True, "episodes": results}, indent=1))

    imp = np.array([r["improvement"] for r in results]) if results else np.array([0.0])
    ov = np.array([r["d_end"] - r["d_best"] for r in results]) if results else np.array([0.0])
    OUT.write_text(json.dumps({
        "summary": {"scene": meta["name"], "episodes": len(results),
                    "checkpoint": args.checkpoint, "samples": args.samples,
                    "stop_on_shoot": bool(args.stop_on_shoot),
                    "guidance": args.guidance, "num_steps": args.num_steps,
                    "mean_improvement": float(imp.mean()),
                    "frac_positive": float((imp > 0).mean()),
                    "mean_d_start": float(np.mean([r["d_start"] for r in results] or [0])),
                    "mean_d_end": float(np.mean([r["d_end"] for r in results] or [0])),
                    "mean_d_best": float(np.mean([r["d_best"] for r in results] or [0])),
                    "settle_frac": float((ov < 1e-6).mean()),
                    "declared_frac": float(np.mean([r["declared_stop"] is not None
                                                    for r in results] or [0])),
                    "frames_dir": str(FR)},
        "episodes": results}, indent=1))
    print(f"\nwrote {OUT}: {len(results)} episodes, improvement {imp.mean():.3f}")
    return 0


raise SystemExit(main())
