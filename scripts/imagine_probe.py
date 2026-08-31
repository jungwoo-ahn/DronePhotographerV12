"""What does the policy IMAGINE, and does the imagination generalise to an unseen room?

`mode="policy"` in this framework means "predict both actions and video given the first
frame" (transforms.build_sequence_plan_from_mode). So every action chunk this model emits
comes with eight generated frames — and until now `predict_chunk` read `out["action"]` and
threw `out["vision"]` away. The imagination costs nothing extra to look at; it is already
computed in the same forward pass.

It is worth looking at because the training curves disagree with each other. Over 40k
iterations the ACTION loss on the held-out scenes fell the whole way (0.00265 -> 0.00066),
while the VISION loss on the same scenes went the other direction:

    heldout/train_flow_matching_loss_vision   0.1473 -> 0.0528
    heldout/val_flow_matching_loss_vision     0.1565 -> 0.2502

Actions generalise to rooms the model has never seen; frame prediction appears to overfit.
A loss number cannot say what that failure looks like, so this renders it.

The comparison is teacher-forced and needs no Blender: a window's real future frames are
already on disk (`window.keyframes[1:]`), so imagined frame k lines up with real frame k.

    sbatch -A share --qos=share scripts/sbatch_imagine_probe.sh \
        --checkpoint runs/train/.../checkpoints/iter_000040000 --n 120
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
ap.add_argument("--n", type=int, default=120, help="windows to imagine")
ap.add_argument("--chunk-size", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=30)
ap.add_argument("--guidance", type=float, default=1.0)
ap.add_argument("--image-size", type=int, default=256)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--root", default=f"{V12}/{DEFAULT_TRAJ_ROOT}")
ap.add_argument("--val-scenes", default=f"{V12}/configs/val_scenes.json")
ap.add_argument("--train-too", type=int, default=1,
                help="also sample windows from TRAINED scenes. Without them there is no "
                     "reference for what the imagination looks like when it works, and the "
                     "val-vision divergence cannot be attributed to the room being new.")
ap.add_argument("--out", default=f"{V12}/runs/eval/imagine_probe.json")
ap.add_argument("--frames-dir", default=f"{V12}/runs/eval/imagine_frames")
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
FR = Path(args.frames_dir)
OUT.parent.mkdir(parents=True, exist_ok=True)
FR.mkdir(parents=True, exist_ok=True)


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
    """Byte-identical to training (tests/test_prompt_single_format.py)."""
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


def imagine(model, frame_uint8: np.ndarray, prompt: str, seed: int):
    """One forward pass -> (imagined frames uint8 [T,H,W,3], action chunk).

    `out["vision"]` is a LATENT, not pixels — the model returns "List of vision latent
    tensors". Decoding is the step this project had never run: `model.decode` gives
    [B,C,T,H,W] in [-1,1], which the framework's own path rescales with (1+v)/2.
    """
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
    batch["ai_caption"] = [prompt] * len(batch["ai_caption"])
    with torch.no_grad():
        out = model.generate_samples_from_batch(
            batch, guidance=args.guidance, num_steps=args.num_steps, seed=[seed],
        )
        act = out["action"][0].float().cpu().numpy()[:, :CAMERA_ACTION_DIM]
        vis = out.get("vision")
        if vis is None:
            return None, act, "no vision key"
        lat = vis[0]
        if lat.dim() == 4:
            lat = lat.unsqueeze(0)                       # [C,T,H,W] -> [B,C,T,H,W]
        px = model.decode(lat.to(torch.bfloat16) if lat.dtype != torch.float32 else lat)
        px = ((1.0 + px.float()) / 2).clamp(0, 1)[0]     # [C,T,H,W] in [0,1]
    arr = (px.permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)  # [T,H,W,3]
    return arr, act, None


def pick_windows(n: int) -> list[dict]:
    """Balanced over (split, scene, sector) and over how far the goal is.

    'Maximum diversity' has to be built into the draw, not hoped for: the natural pool is
    lopsided by sector and by scene size, so an unbalanced sample would show the same room
    from the same side a dozen times.
    """
    val_scenes = frozenset(json.loads(Path(args.val_scenes).read_text())["scenes"])
    root = Path(args.root)
    allp = [d for d in sorted(os.listdir(root)) if (root / d / "data.json").exists()]
    rng = random.Random(args.seed)
    groups = {"val": [d for d in allp if d.split("__")[0] in val_scenes]}
    if args.train_too:
        groups["train"] = [d for d in allp if d.split("__")[0] not in val_scenes]
    for v in groups.values():
        rng.shuffle(v)
    print(f"[imagine] pools: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()), flush=True)

    per_split = n // len(groups)
    out: list[dict] = []
    for split, dirs in groups.items():
        seen: Counter = Counter()
        got = 0
        for d in dirs:
            if got >= per_split:
                break
            obj = d.split("__", 1)[1] if "__" in d else d
            if obj in DEFAULT_EXCLUDE_OBJECTS:
                continue
            try:
                wins = list(iter_goal_start_windows(root / d / "data.json",
                                                    chunk_size=args.chunk_size, max_per_pair=4))
            except Exception:  # noqa: BLE001
                continue
            rng.shuffle(wins)
            for w in wins:
                g = goal_vector(w.goal_frame.raw, DEFAULT_GOAL_KEYS, object_key=_window_object(w))
                if not np.isfinite(g).all():
                    continue
                sec = sector8(float(g[DEFAULT_GOAL_KEYS.index(SUBJECT_BEARING_KEY)]))
                sc = d.split("__")[0]
                delta = int(abs(w.goal_frame.frame_idx - w.start_frame_idx))
                band = "near" if delta < args.chunk_size else "mid" if delta < 20 else "far"
                cell = (sc, sec, band)
                if seen[cell] >= 1 or seen[(sc,)] >= 6:
                    continue
                real = [Path(k.image) for k in w.keyframes]
                if not all(p.exists() for p in real):
                    continue
                seen[cell] += 1
                seen[(sc,)] += 1
                out.append({
                    "split": split, "placement": d, "scene": sc,
                    "object": obj, "sector": sec, "delta": delta, "band": band,
                    "real": [str(p) for p in real],
                    "goal_image": str(Path(w.goal_frame.image)),
                    "goal_vec": g.tolist(), "goal_raw": w.goal_frame.raw,
                    "arrival": int(np.argmax(shoot_column(w))) if shoot_column(w).max() > 0 else -1,
                })
                got += 1
                break
    print(f"[imagine] {len(out)} windows  "
          f"({dict(Counter(o['split'] for o in out))})", flush=True)
    return out


def main() -> int:
    from PIL import Image
    eps = pick_windows(args.n)
    if not eps:
        print("[imagine] no windows"); return 1

    rows: list[dict] = []
    done: set = set()
    if args.resume and OUT.exists():
        try:
            rows = json.loads(OUT.read_text()).get("rows", [])
            done = {r["placement"] for r in rows}
            print(f"[imagine] resuming: {len(rows)} done", flush=True)
        except Exception:  # noqa: BLE001
            rows = []

    model = load_policy()
    shape_note = None
    for i, ep in enumerate(eps):
        if ep["placement"] in done:
            continue
        start = np.asarray(Image.open(ep["real"][0]).convert("RGB"))
        prompt = make_prompt(np.array(ep["goal_vec"]), crop=ep["goal_raw"], idle_frames=0)
        img, act, err = imagine(model, start, prompt, seed=args.seed + i)
        if img is None:
            print(f"[imagine] {i}: {err}", flush=True); return 1
        if shape_note is None:
            shape_note = list(img.shape)
            print(f"[imagine] decoded shape {shape_note} for a {args.chunk_size}-step chunk",
                  flush=True)

        # Align: the model conditions on frame 0 and generates the rest, so the LAST
        # k frames of the decode line up with real frames 1..k.
        k = min(img.shape[0], len(ep["real"]))
        imag = img[-k:]
        real = [np.asarray(Image.open(p).convert("RGB").resize(
            (args.image_size, args.image_size), Image.BILINEAR)) for p in ep["real"][-k:]]

        tag = f"{ep['split']}_{i:03d}"
        paths, l1s, psnrs = [], [], []
        for j, (a, b) in enumerate(zip(imag, real)):
            p = FR / f"{tag}_imag{j}.jpg"
            Image.fromarray(a).save(p, quality=88)
            paths.append(str(p))
            d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
            l1s.append(float(d))
            mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
            psnrs.append(float(10 * np.log10(255.0 ** 2 / max(mse, 1e-6))))

        rows.append({**{kk: ep[kk] for kk in ("split", "placement", "scene", "object",
                                              "sector", "delta", "band", "arrival",
                                              "real", "goal_image")},
                     "imagined": paths,
                     "l1": [round(x, 2) for x in l1s],
                     "psnr": [round(x, 2) for x in psnrs],
                     "l1_mean": round(float(np.mean(l1s)), 2),
                     "psnr_mean": round(float(np.mean(psnrs)), 2),
                     "shoot_pred": round(float(np.max(act[:, 9])), 4)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eps)}  last L1 {rows[-1]['l1_mean']:.1f} "
                  f"PSNR {rows[-1]['psnr_mean']:.1f}", flush=True)
            OUT.write_text(json.dumps({"partial": True, "rows": rows}, indent=1))

    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r["psnr_mean"])
    OUT.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n": len(rows), "decoded_shape": shape_note,
        "guidance": args.guidance, "num_steps": args.num_steps, "seed": args.seed,
        "frames_dir": str(FR),
        "psnr_by_split": {k: round(float(np.mean(v)), 2) for k, v in by.items()},
        "rows": rows,
    }, indent=1))
    print(f"\nwrote {OUT}  ({len(rows)} windows)")
    for k, v in by.items():
        print(f"  {k:<6} mean PSNR {np.mean(v):5.2f} dB  (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
