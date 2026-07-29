# DronePhotographer V12 — Cosmos 3 native world-action policy

v12 reframes the goal-conditioned camera policy on **NVIDIA Cosmos 3** (omnimodal two-tower
MoT), whose **`camera_pose` embodiment (9D = 3D translation + 6D rotation) and native `policy`
mode** map directly onto DronePhotographer's "imagine-before-act" thesis — replacing v11's
bolted-on action head. Start point: **Cosmos3-Edge**, reuse + modify the framework code.

See **`docs/cosmos3_understanding.md`** for the full Cosmos 3 deep-dive (architecture, action
modality, camera_pose, finetune recipe, the goal-conditioning crux) and the v12 design decisions.

## Layout
- `src/{common,scoring,utils}` — reusable data / shot-profile / rotation code (self-contained; copied, no cross-repo imports)
- `src/{model,data,train}` — v12 policy (to be built on Cosmos 3)
- `scripts/` — analysis & data tooling:
  - `analyze_distributions.py` — goal-profile + action distribution audit
  - `check_roll.py` — verifies camera roll ≈ 0 (justifies rot6d)
  - `facing_auto.py` / `verify_facing_html.py` — per-asset subject-facing recovery (auto + human-verify)
  - `viz_goal_prompt_gallery.py` — validate the cinematography goal descriptor against renders
- `configs/`, `tests/`, `docs/`
- `repos/` (gitignored) — cloned `NVIDIA/cosmos` + `NVIDIA/cosmos-framework`
- `data` (gitignored symlink) — original Blender trajectory dataset

## Environments
- `.venv-analysis` (gitignored) — light CPU analysis env (numpy / pillow / opencv-headless)
- Cosmos 3 training/inference env (CUDA 13 / transformers / uv) — TODO

## Status
Design phase. Cosmos 3 understood + cloned; v7 data audited; rotation rep (3+6d rot6d) and
goal-descriptor direction (cinematography text + numbers, staged goal conditioning) decided;
subject-facing recovery in progress (world-frame azimuth → subject-relative needs a per-asset
facing map). Nothing trained yet.
