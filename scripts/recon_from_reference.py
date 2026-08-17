"""TRUE recon: a reference photo's composition, re-shot by the policy in a DIFFERENT scene.

The report's "recon" panel was retrieval — the nearest matching frame already in the dataset. This
closes the loop for real:

    reference image --Module 2--> goal profile --goal_prompt--> trained policy
        --> action chunks --> Blender rollout in another scene/subject --> achieved frame

Only the three attributes the policy was trained on are used (shot size / bearing / elevation), which
is exactly what Module 2 recovers. Success = the achieved frame lands on the same composition as the
reference, measured two ways: the geometric profile from the env pose, and Module 2 re-run on the
achieved render (fully image-based, the deployment-realistic check).

Run via sbatch (16B policy + Blender). venv: the cosmos-framework env.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
SHARED = "/home/nas_main/jungwooahn/projects/DronePhotographer"
# Cosmos resolves several model configs RELATIVE to the cwd, so the cosmos-framework root
# has to be the working directory (same reason closed_loop_eval.py does this) — every path
# of ours is therefore absolute.
CF_ROOT = f"{V12}/repos/cosmos-framework"
sys.path.insert(0, V12)
sys.path.insert(0, CF_ROOT)
os.chdir(CF_ROOT)

import numpy as np

# Imported up here, ahead of the torch/cosmos block below, purely so the `--root` default
# can reference it — the rest of the src imports stay after argparse to keep `--help` fast.
from src.common.dataset_base import DEFAULT_TRAJ_ROOT  # noqa: E402
from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--cases", type=int, default=6, help="reference images to re-shoot")
ap.add_argument("--chunks", type=int, default=8, help="action chunks per rollout")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--root", default=f"{V12}/{DEFAULT_TRAJ_ROOT}")
ap.add_argument("--out", default=f"{V12}/runs/recon_ref/recon.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--resume", action="store_true",
                help="skip cases already present in --out (share QOS gets preempted; without this "
                     "every restart redoes case 0 and the run never finishes)")
ap.add_argument("--goals-json", required=True,
                help="written by scripts/prepare_recon_goals.py in the analysis venv — keeps "
                     "Module 2 (ultralytics/sklearn) out of the cosmos environment")
args = ap.parse_args()

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
from PIL import Image  # noqa: E402

from src.common.blender_env import BlenderRolloutEnv, SubprocessBlenderRenderer  # noqa: E402
from src.common.facing import front_azimuth, sector3, sector8  # noqa: E402
from src.common.goal_space import DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY  # noqa: E402
from src.common.run_info import write_run_info as _write_run_info  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402

OUT = Path(args.out)
FRAMES = OUT.parent / "frames"
FRAMES.mkdir(parents=True, exist_ok=True)


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
        "output_dir": str(OUT.parent / "_infer"),
        "sampler": "unipc",
        "guardrails": False,
        "use_ema_weights": not args.no_ema,
    })
    setup = overrides.build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    pipe.model.eval()
    return pipe.model


def make_prompt(goal_vec: np.ndarray, specified=None, crop=None) -> str:
    """Exactly the JSON prompt the policy was trained on (see closed_loop_eval.make_prompt)."""
    formatter = ActionPromptJsonFormatter()
    sample = {
        # specified/crop must be threaded through: training prompts carry the crop clause and
        # omit nothing, so a prompt built without them is a different distribution.
        "ai_caption": goal_prompt(goal_vec, specified=specified, crop=crop),
        "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
        "image_size": torch.tensor([args.image_size, args.image_size]),
        "mode": "policy",
        # `action` is here ONLY so the formatter can count total frames: `_get_total_frames`
        # reads action.shape[0], and without it idle_frame reads "0." where training says
        # "0 out of 8." — a 9-character divergence that string-level inspection missed and
        # only a field-by-field diff against the dataset's own output surfaced. Zeros are
        # correct: policy mode never conditions on the action, only on its length.
        "action": torch.zeros(args.chunk_size, 1, dtype=torch.float32),
        "idle_frames": torch.tensor(0),
        "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                             dtype=torch.uint8),
    }
    out = formatter(sample)["ai_caption"]
    return out if isinstance(out, str) else json.dumps(out)


def predict_chunk(model, frame_uint8: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    video = torch.from_numpy(frame_uint8).permute(2, 0, 1).contiguous()
    video = video.unsqueeze(1).repeat(1, args.chunk_size + 1, 1, 1)
    batch = build_action_batch(
        video=video,
        action=torch.zeros(args.chunk_size, 64, dtype=torch.float32),
        raw_action_dim=CAMERA_ACTION_DIM,
        prompt=prompt,
        view_point="ego_view",
        domain_name=DOMAIN_NAME,
        model_mode=ModelMode.POLICY,
        action_chunk_size=args.chunk_size,
        fps=args.fps,
        input_video_key=model.config.input_video_key,
        batch_size=1,
        device="cuda",
    )
    # build_action_batch re-formats whatever it is handed, and `prompt` is already the
    # formatted JSON -> double-wrapped, a shape training never produced. See
    # closed_loop_eval.predict_chunk for the measurement.
    batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
    with torch.no_grad():
        out = model.generate_samples_from_batch(
            batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed],
        )
    chunk = out["action"][0].float().cpu().numpy()[:, :CAMERA_ACTION_DIM]
    assert chunk.shape[1] == CAMERA_ACTION_DIM, (
        f"predicted chunk is {chunk.shape[1]} wide, expected {CAMERA_ACTION_DIM}")
    return chunk


def goal_vec_from_profile(gp) -> np.ndarray:
    """Module 2's profile -> the goal vector `goal_prompt` reads (occupancy / bearing / elevation)."""
    v = np.zeros(len(DEFAULT_GOAL_KEYS), dtype=np.float32)
    for i, k in enumerate(DEFAULT_GOAL_KEYS):
        v[i] = float(gp.values.get(k, 0.0))
    return v


def profile_summary(gp) -> dict:
    return {
        "occupancy": round(float(gp.values.get("occupancy", float("nan"))), 1),
        "bearing": round(float(gp.values.get(SUBJECT_BEARING_KEY, float("nan"))), 1),
        "elevation": round(float(gp.values.get("cam_to_obj_elevation_deg", float("nan"))), 1),
        "categories": gp.categories(),
    }


def _unused_pick_cases(est, n):
    """DEAD CODE — `prepare_recon_goals.py` is the live path. Its inline filter
    (occupancy 30-88, body_in_frame >= 50) predates the visible_frac gate; use
    `src.common.annotations.is_goal_frame` if this is ever revived."""
    """(reference render, target placement) pairs — the target is a DIFFERENT scene AND subject."""
    root = Path(args.root)
    dirs = [d for d in root.iterdir() if d.is_dir()]
    random.seed(args.seed); random.shuffle(dirs)
    usable = []
    for d in dirs:
        obj = d.name.split("__", 1)[1] if "__" in d.name else d.name
        if front_azimuth(obj) is None:
            continue
        p = d / "data.json"
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue
        frames = [(r, s) for pair in doc.get("render_records", []) for r in pair
                  if (s := r.get("scores")) and r.get("in_frame")
                  and 30 <= s["occupancy"] <= 88 and s["body_in_frame_ratio"] >= 50]
        if not frames or not doc.get("accepted_pairs"):
            continue
        usable.append((d, doc, obj, frames))
        if len(usable) >= 3 * n:
            break

    cases = []
    for i in range(0, len(usable) - 1, 2):
        (rd, rdoc, robj, rframes) = usable[i]
        # target: different subject AND different scene
        rscene = rd.name.split("__", 1)[0]
        tgt = next(((d, doc, obj, fr) for (d, doc, obj, fr) in usable[i + 1:]
                    if obj != robj and d.name.split("__", 1)[0] != rscene), None)
        if tgt is None:
            continue
        r, s = random.choice(rframes)
        cases.append({
            "ref_image": str(rd / r["path_rel"]), "ref_scores": s, "ref_object": robj,
            "ref_dir": str(rd),
            "tgt_dir": str(tgt[0]), "tgt_doc": tgt[1], "tgt_object": tgt[2],
        })
        if len(cases) >= n:
            break
    return cases


def main():
    cases = json.loads(Path(args.goals_json).read_text())["cases"][: args.cases]
    results = []
    done = set()
    if args.resume and OUT.exists():
        try:
            results = json.loads(OUT.read_text()).get("results", [])
            done = {r["ref_image"] for r in results}
        except Exception:
            results, done = [], set()
    todo = [(i, c) for i, c in enumerate(cases) if c["ref_image"] not in done]
    print(f"cases: {len(cases)} total, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do", flush=True); return
    model = load_policy()
    renderer = SubprocessBlenderRenderer(repo_root=SHARED)

    for ci, c in todo:
        # 1) goal already extracted from the reference image by Module 2
        gvec = np.asarray(c["goal_vec"], dtype=np.float32)
        prompt = make_prompt(gvec, specified=c.get("goal_specified"), crop=c.get("goal_crop") or None)

        # 2) start in a DIFFERENT scene/subject
        tgt_dir = Path(c["tgt_dir"]); doc = json.loads((tgt_dir / "data.json").read_text())
        sp = c["start_pose"]        # visible start (subject in frame, but far from the requested shot)
        run_info = _write_run_info(tgt_dir.name, str(tgt_dir / "data.json"),
                                   OUT.parent / "_run_info",
                                   shared_root=SHARED, resolution=args.image_size)
        env = BlenderRolloutEnv(run_info_path=run_info, renderer=renderer,
                                object_position=doc.get("subject_foot") or doc.get("subject_center"))
        obs = env.reset(np.array(sp["pos"], dtype=np.float64),
                        np.array(sp["forward"], dtype=np.float64),
                        np.array(sp["up"], dtype=np.float64))
        start_png = FRAMES / f"case{ci}_start.jpg"
        obs["image"].save(start_png)

        # 3) roll the policy out toward the reference composition
        step_frames = [str(start_png)]
        for k in range(args.chunks):
            frame = np.asarray(obs["image"].convert("RGB").resize((args.image_size, args.image_size)),
                               dtype=np.uint8)
            chunk = predict_chunk(model, frame, prompt, seed=args.seed + ci * 100 + k)
            for step in chunk[:, :9]:      # env takes the 9 pose dims
                obs, _ = env.step(step, render=False)
            obs = env.reset(env.position, env.forward, env.up)      # re-render the new view
            # keep every intermediate view: the trajectory is the interesting part, and a single
            # end frame cannot tell "went straight there" from "wandered and came back"
            png = FRAMES / f"case{ci}_chunk{k+1:02d}.jpg"
            obs["image"].save(png)
            step_frames.append(str(png))
        end_png = FRAMES / f"case{ci}_achieved.jpg"
        obs["image"].save(end_png)

        # 4) frames + poses out; Module 2 re-measures them in the analysis venv (stage 3)
        rec = dict(c)
        rec.update({"prompt": prompt, "start_frame": str(start_png), "achieved_frame": str(end_png),
                    "step_frames": step_frames,
                    "end_pose": {"position": env.position.tolist(), "forward": env.forward.tolist(),
                                 "up": env.up.tolist()}})
        rec.pop("tgt_doc", None)
        results.append(rec)
        # write as we go: this runs on share QOS and can be preempted at any moment
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
        print(f"  [{ci}] {c['ref_object'][:20]} -> {c['tgt_object'][:20]} | goal {c['goal']['categories']}",
              flush=True)

    renderer.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    print(f"wrote {OUT} ({len(results)} cases)")


if __name__ == "__main__":
    main()
