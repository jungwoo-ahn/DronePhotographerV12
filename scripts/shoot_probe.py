"""What does the shoot channel actually predict, and is it reading the image or the leak?

Three questions, none of which need Blender or a rollout — so this is minutes of GPU per
hundred windows, and it has to run BEFORE the closed-loop sweeps because it decides how to
read them.

1. DISTRIBUTION. Dim 9 was trained under a continuous flow-matching objective against a
   binary target. Nothing guarantees the output is bimodal near {0,1}; it could pile up at
   0.5, in which case the 0.5 threshold `closed_loop_eval` uses has no evidence behind it.

2. THRESHOLD. Precision/recall against the true `shoot_column`, so whatever threshold the
   rollouts use is traceable to a curve instead of being a literal in the source.

3. THE LEAK. `idle_frame` reaches the model through the prompt in policy mode, and it is
   very nearly a copy of the shoot label: measured over 6000 exported episodes,
   `idle_frames > 0` implies shoot=1 in 896/896 (100%), while `idle_frames == 0` implies it
   in 11.3%. That is not a coincidence to be designed away -- post-arrival steps have
   translation exactly 0 and rot6d exactly identity, so they are idle by construction.

   Inference cannot know the true value, so `predict_chunk` feeds the training-modal 0 --
   which in training meant "89% chance you have NOT arrived". If the prediction tracks that
   field instead of the frame, the shoot channel is a copy of the prompt and every
   termination number downstream is confounded. Sweeping the field with everything else held
   fixed is the only way to tell, and it is cheap.

Windows come from the 8 held-out scenes in configs/val_scenes.json -- the same manifest the
training split used, so these are scenes the policy has never seen.

    sbatch --qos=own --gres=gpu:1 scripts/sbatch_shoot_probe.sh \
        --checkpoint runs/train/.../checkpoints/iter_000040000 --n 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)
CF_ROOT = f"{V12}/repos/cosmos-framework"
sys.path.insert(0, CF_ROOT)
os.chdir(CF_ROOT)

import numpy as np  # noqa: E402

from src.common.dataset_base import DEFAULT_TRAJ_ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--n", type=int, default=300, help="windows for the distribution / PR curve")
ap.add_argument("--n-sweep", type=int, default=100, help="windows for the idle_frame sweep")
ap.add_argument("--samples", type=int, default=1,
                help="draws per window. 1 is fine for the distribution; raise to separate "
                     "sampling variance from bias in the arrival step.")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--root", default=f"{V12}/{DEFAULT_TRAJ_ROOT}")
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--out", default=f"{V12}/runs/eval/shoot_probe.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-ema", action="store_true")
ap.add_argument("--resume", type=int, default=1,
                help="continue from a partial --out instead of redoing finished windows. "
                     "Needed on `share`, which is evicted the moment another account wants "
                     "the GPU; without it every eviction restarts from window 0 and a job "
                     "that takes longer than the gap between evictions never finishes.")
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
from src.common.dataset_base import (  # noqa: E402
    DEFAULT_EXCLUDE_OBJECTS, _window_object, shoot_column,
)
from src.common.facing import sector8  # noqa: E402
from src.common.goal_space import (  # noqa: E402
    DEFAULT_GOAL_KEYS, SUBJECT_BEARING_KEY, goal_vector,
)
from src.data.cosmos_camera_dataset import CAMERA_ACTION_DIM, DOMAIN_NAME  # noqa: E402
from src.data.lerobot_export import goal_prompt  # noqa: E402

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_policy():
    maybe_init_distributed()
    wan = os.environ.get(
        "WAN_VAE_PATH", f"{CF_ROOT}/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth")
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


def make_prompt(goal_vec: np.ndarray, crop=None, *, idle_frames: int | None = 0) -> str:
    """The prompt training produced, byte for byte (tests/test_prompt_single_format.py).

    `action` is a LENGTH hint only: `_get_total_frames` reads `action.shape[0]`, and without
    it `idle_frame` reads "0." where training says "0 out of 8.". `idle_frames=None` omits
    the field entirely, which is the leak ablation.
    """
    f = ActionPromptJsonFormatter()
    sample = {
        "ai_caption": goal_prompt(goal_vec, crop=crop),
        "viewpoint": "ego_view",
        "conditioning_fps": torch.tensor(args.fps, dtype=torch.long),
        "image_size": torch.tensor([args.image_size, args.image_size]),
        "mode": "policy",
        "action": torch.zeros(args.chunk_size, 1, dtype=torch.float32),
        "video": torch.zeros(3, args.chunk_size + 1, args.image_size, args.image_size,
                             dtype=torch.uint8),
    }
    if idle_frames is not None:
        sample["idle_frames"] = torch.tensor(int(idle_frames))
    out = f(sample)["ai_caption"]
    return out if isinstance(out, str) else json.dumps(out)


def predict(model, frame_uint8: np.ndarray, prompt: str, seed: int) -> np.ndarray:
    """One (chunk_size, CAMERA_ACTION_DIM) chunk, teacher-forced off a real frame."""
    from PIL import Image
    img = np.asarray(frame_uint8)
    if img.shape[0] != args.image_size or img.shape[1] != args.image_size:
        img = np.asarray(Image.fromarray(img).resize(
            (args.image_size, args.image_size), Image.BILINEAR))
    video = torch.from_numpy(img).permute(2, 0, 1).contiguous()
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
    # build_action_batch re-formats whatever it is handed; `prompt` is already formatted.
    batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
    with torch.no_grad():
        out = model.generate_samples_from_batch(
            batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed],
        )
    chunk = out["action"][0].float().cpu().numpy()[:, :CAMERA_ACTION_DIM]
    assert chunk.shape[1] == CAMERA_ACTION_DIM, (
        f"predicted chunk is {chunk.shape[1]} wide, expected {CAMERA_ACTION_DIM}; "
        "the shoot channel is missing")
    return chunk


def pick_windows(n: int) -> list[dict]:
    """Windows from the held-out scenes only, spread across placements and sectors."""
    val_scenes = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
    root = Path(args.root)
    dirs = [d for d in sorted(os.listdir(root))
            if (root / d / "data.json").exists() and d.split("__")[0] in val_scenes]
    rng = random.Random(args.seed)
    rng.shuffle(dirs)
    print(f"[probe] held-out pool: {len(dirs)} placements from {len(val_scenes)} scenes",
          flush=True)

    out: list[dict] = []
    # Deliberately balanced on whether the arrival is INSIDE the chunk. The natural rate is
    # ~25%, so an unbalanced draw would leave ~75 positives in 300 windows and the PR curve
    # would rest on very few points at the end that matters.
    want_pos = want_neg = n // 2
    for d in dirs:
        if want_pos <= 0 and want_neg <= 0:
            break
        obj = d.split("__", 1)[1] if "__" in d else d
        if obj in DEFAULT_EXCLUDE_OBJECTS:
            continue
        try:
            wins = list(iter_goal_start_windows(root / d / "data.json",
                                                chunk_size=args.chunk_size, max_per_pair=4))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {d[:40]}: {exc}")
            continue
        rng.shuffle(wins)
        for w in wins:
            g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
            if not np.isfinite(g).all():
                continue
            s = shoot_column(w)
            pos = bool(s.max() > 0)
            if pos and want_pos <= 0:
                continue
            if not pos and want_neg <= 0:
                continue
            img = Path(w.start.image)
            if not img.exists():
                continue
            out.append({
                "placement": d, "scene": d.split("__")[0], "object": obj,
                "start_image": str(img),
                "goal_vec": g.tolist(), "goal_raw": w.goal_frame.raw,
                "sector": sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)])),
                "delta": int(abs(w.goal_frame.frame_idx - w.start_frame_idx)),
                "true_shoot": s.tolist(),
                "arrival": int(np.argmax(s)) if pos else -1,
            })
            if pos:
                want_pos -= 1
            else:
                want_neg -= 1
            break                     # one window per placement -> maximum scene spread
    print(f"[probe] {len(out)} windows  "
          f"(arrival inside chunk: {sum(1 for e in out if e['arrival'] >= 0)})", flush=True)
    return out


def main() -> int:
    from PIL import Image
    eps = pick_windows(args.n)
    if not eps:
        print("[probe] no windows"); return 1
    model = load_policy()

    # ---- 1/2: distribution + PR, at the training-modal idle_frame=0 ----------------
    rows = []
    done_keys: set = set()
    if args.resume and OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            rows = prev.get("rows", [])
            done_keys = {r["placement"] for r in rows}
            print(f"[probe] resuming: {len(rows)} windows already done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] could not resume ({exc}); starting over", flush=True)
            rows = []

    for i, e in enumerate(eps):
        if e["placement"] in done_keys:
            continue
        img = np.asarray(Image.open(e["start_image"]).convert("RGB"))
        p = make_prompt(np.array(e["goal_vec"]), crop=e["goal_raw"], idle_frames=0)
        pred = np.stack([predict(model, img, p, seed=args.seed + 1000 * k + i)
                         for k in range(args.samples)])          # (S, chunk, D)
        rows.append({**{k: e[k] for k in ("placement", "scene", "object", "sector",
                                          "delta", "true_shoot", "arrival")},
                     "pred_shoot": pred[:, :, 9].round(5).tolist(),
                     "pred_trans_absmax": float(np.abs(pred[:, :, :3]).max())})
        if (i + 1) % 25 == 0:
            print(f"  [dist] {i+1}/{len(eps)}", flush=True)
            OUT.write_text(json.dumps({"partial": True, "rows": rows, "sweep": []}, indent=1))

    # ---- 3: the leak sweep, same window and seed, only idle_frame changes ----------
    sweep = []
    if args.resume and OUT.exists():
        try:
            sweep = json.loads(OUT.read_text()).get("sweep", []) or []
            if sweep:
                print(f"[probe] resuming sweep: {len(sweep)} done", flush=True)
        except Exception:  # noqa: BLE001
            sweep = []
    swept = {r["placement"] for r in sweep}

    for i, e in enumerate(eps[: args.n_sweep]):
        if e["placement"] in swept:
            continue
        img = np.asarray(Image.open(e["start_image"]).convert("RGB"))
        rec = {k: e[k] for k in ("placement", "scene", "sector", "delta", "arrival",
                                 "true_shoot")}
        for label, idle in (("0", 0), ("3", 3), ("8", 8), ("omitted", None)):
            p = make_prompt(np.array(e["goal_vec"]), crop=e["goal_raw"], idle_frames=idle)
            c = predict(model, img, p, seed=args.seed + i)        # SAME seed across arms
            rec[f"shoot_idle_{label}"] = c[:, 9].round(5).tolist()
        sweep.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  [sweep] {i+1}/{min(args.n_sweep, len(eps))}", flush=True)
            OUT.write_text(json.dumps({"partial": True, "rows": rows, "sweep": sweep}, indent=1))

    OUT.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "n": len(rows), "n_sweep": len(sweep), "samples": args.samples,
        "guidance": args.guidance, "num_steps": args.num_steps, "seed": args.seed,
        "val_scenes": str(args.val_scenes),
        "scenes": sorted({r["scene"] for r in rows}),
        "sector_mix": dict(Counter(r["sector"] for r in rows)),
        "rows": rows, "sweep": sweep,
    }, indent=1))
    print(f"\nwrote {OUT}  ({len(rows)} windows, {len(sweep)} swept)")
    print("Analyse with scripts/shoot_probe_report.py — thresholds are chosen there, from "
          "the curve, not here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
