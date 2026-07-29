# Cosmos 3 — deep understanding (for DronePhotographer v12)

Synthesized from 3 parallel investigations (2026-07): the technical report (arXiv 2606.02800,
released 2026-06-22, OpenMDW-1.1), the model code (`repos/cosmos-framework/cosmos_framework/`),
and the inference/finetune/data path (`repos/cosmos-framework` + `repos/cosmos/cookbooks/cosmos3`).
Paths below: `CF/` = `repos/cosmos-framework/cosmos_framework/`, `CB/` = `repos/cosmos/cookbooks/cosmos3/`.

---

## 1. What Cosmos 3 is

An **omnimodal world model** that jointly processes/generates text, image, video, audio, **and
action** in a single **two-tower Mixture-of-Transformers (MoT)**:

- **Reasoner tower** — an autoregressive VLM (**Qwen3-VL**: ViT understanding encoder + text model),
  causal self-attention, next-token decoding. Understands frames/text.
- **Generator tower** — a **diffusion transformer** (rectified flow / velocity prediction), full
  bidirectional attention, iterative denoising. Produces video/audio/**action**.
- **Coupling**: the two towers **share only the attention operation** — per layer everything else
  (MLP, layernorms, q/k/v/o projections) is duplicated (`_moe_gen` suffix). In each layer, reasoner
  tokens and generator tokens are concatenated into ONE joint self-attention (`PackedAttentionMoT`,
  `CF/model/generator/mot/unified_mot.py:468,584`). Information flows **one way: reasoner → generator**
  ("think, then act"). The generator always runs both towers; the reasoner can run alone.
- **Unified 3D mRoPE** (temporal/height/width) aligns video/action/audio on one physical time axis.
- **Sizes**: Edge **4B** (2B+2B), Nano **16B** (Qwen3-VL-8B + 8B gen expert), Super **64B** (32B+32B).
  "Nano" is NOT tiny — it's 16B.
- **Top-level code**: `OmniMoTModel` (`CF/model/generator/omni_mot_model.py:81`) →
  `Cosmos3VFMNetwork` (`CF/model/generator/mot/cosmos3_vfm_network.py:104`) → `unified_mot` MoT.
  The `packages/{transformers,vllm}-cosmos3` are **reasoner-tower-only shims** (they DROP all
  generation/action weights) — all action/policy logic is in the main framework.

## 2. How ACTION works (the crux) — code truth

- **Continuous, NOT discretized.** Actions live in the diffusion subsequence and are **denoised by
  the same rectified-flow sampler as video, in one joint loop** (`generate_samples_from_batch`,
  `omni_mot_model.py:2437`; velocity state = `[vision | action | sound]`). Not AR-decoded.
- **Action embedding** (`cosmos3_vfm_network.py:154-160`):
  `action2llm = DomainAwareLinear(action_dim, hidden, num_domains)` in,
  `llm2action = DomainAwareLinear(hidden, action_dim, num_domains)` out, plus a learned
  `action_modality_embed`. `DomainAwareLinear` keeps **per-embodiment weights** (nn.Embedding over
  domain_id) — each embodiment gets its own projection while sharing the transformer.
- **Action chunk** = `chunk_size × action_dim`, **one token per step** (temporal compression 1 for
  action vs 4 for video). Raw actions are per-channel affine-**normalized** then **zero-padded to
  `max_action_dim`=64 (Nano)**; the model masks padded channels via `raw_action_dim`.
- **CFG**: `v = uncond + guidance·(cond − uncond)`, default 1.5 but the **camera_pose example uses
  guidance=1.0** (cf. our own finding that CFG overshoots goal-conditioning).

### camera_pose = DronePhotographer's action space, natively
- Registered in `CF/data/generator/action/domain_utils.py`: **`camera_pose` → domain_id 2,
  raw_action_dim 9**. No registry edit needed.
- **9D = `Pos()` (3D translation) + `Rot("rot6d")` (6D rotation), NO gripper**
  (`action_spec.py:19`; robots add `Gripper()` → 10D). rot6d = first two columns of the rotation
  matrix; identity `[1,0,0,0,1,0]`.
- **Frame-wise RELATIVE (delta) poses**, backward-framewise: `pose_abs_to_rel(..., "rot6d",
  "backward_framewise")`. Not absolute, not intrinsics.
- Example action file = JSON `[T,9]` array (`CB/generator/action/assets/actions/camera_action.json`
  = `[60,9]`, row0 ≈ translation + near-identity rot6d).

### Three action modes (differ only by the sequence plan; `transforms.py:235`)
| Mode | Input | Output | DronePhotographer relevance |
|---|---|---|---|
| **policy** | image(frame0) + instruction | **action chunk + imagined video** | ← THE match: "imagine + act" |
| **forward_dynamics** | image + given action chunk | video | camera-controlled video; **works zero-shot on base Nano** |
| **inverse_dynamics** | video | action chunk | recover camera motion from video |

## 3. Direct mapping to DronePhotographer

DronePhotographer: (current frame, shot-profile goal) → next camera-action delta (chunk of 8 steps,
5D = Δright,Δup,Δforward,Δyaw,Δpitch). Maps onto:
- **Embodiment = `camera_pose`** (9D delta pose). Our 5D (3 translation + yaw/pitch, no roll) embeds
  cleanly into 3-translation + rot6d (roll simply 0). The project ALREADY uses a 6D orientation
  convention (`orientation_6d` = forward+up), so the 6D-rotation target is familiar (reconcile abs
  vs rel + which-2-columns).
- **Mode = `policy`** — condition on frame 0 + prompt, jointly denoise action chunk + imagined
  future frames. This **IS** the project's "counterfactual visuomotor policy / imagine-before-act"
  thesis, native.
- **chunk-native FITS** — we already predict an 8-step chunk (not single-step), matching Cosmos's
  chunked action design.

## 4. The finetune recipe = v11's A1, done natively

Reference: `CF/configs/base/experiment/action/posttrain_config/action_policy_{droid,libero}_nano.py`
+ `CB/finetune/`. To post-train a policy on a new dataset:
- **FROZEN**: the Qwen3-VL understanding/reasoner backbone.
- **TRAINED** (`keys_to_select`): `moe_gen` (diffusion expert), `time_embedder`, `vae2llm`, `llm2vae`,
  and the **action heads** `action2llm`, `llm2action`, `action_modality_embed`. Action heads are
  **init-fresh** (not loaded from base — "base has no DROID action heads") at **5× LR**.
- **Loss**: joint flow-matching, `action_loss_weight=10`, vision `loss_scale=10` (balances
  imagine + act heads). Data = **LeRobot v3.0** format.
- **This is structurally identical to v11's A1** (freeze backbone, jointly train world/imagine head
  + action head via flow-matching). The difference: Cosmos 3's backbone is **pretrained on 8M action
  samples across embodiments jointly with video** — so the "world model that already understands
  action" is given, not learned from scratch on our small Blender set.

Shipped policy checkpoints (**Cosmos3-Nano/Edge-Policy-DROID**) were produced this exact way on the
DROID robot dataset (8D joint-position actions). **There is NO shipped camera_pose *policy*
checkpoint, no camera training data, no camera normalizer-stats** — camera_pose is only demonstrated
for forward/inverse dynamics. **A camera policy is something we'd post-train first.**

## 5. THE crux (same as v11): goal conditioning

Cosmos policy conditions semantics ONLY through the **text/JSON `ai_caption` prompt** (Qwen3-VL
reasoner tokenizes it via `ActionPromptJsonFormatter` → `cinematography.framing`, etc.). There is **no
structured-goal input modality**. DronePhotographer's whole design philosophy (CLAUDE.md) rejects
language goals as underspecified in favor of a **structured geometric shot-profile** — so this is the
central integration decision, and it's the SAME binding constraint v11 hit (goal→action dependence):
- **Low-touch**: serialize the shot profile into the JSON prompt; rely on the frozen reasoner.
  No architecture change; goal precision bottlenecked by tokenized text.
- **Higher-fidelity**: add a structured-goal conditioning modality — a small projector into
  `hidden_size` + `goal_modality_embed`, scattered as extra condition tokens (analogous to
  `action2llm`/`action_modality_embed`). Touches `Cosmos3VFMNetwork._encode_*`, the sequence packer,
  mRoPE id assignment, and the dataset. More invasive; keeps the goal geometric.

## 6. Practical constraints
- **Compute**: reference recipe = 256 GPUs (GB200), global batch 8192, 10k iters. Single 8-GPU node
  possible via `data_parallel_replicate_degree=1` + `grad_accum_iter=32` (slow) or a smaller smoke
  batch. **Edge (4B) is the pragmatic first target** on an 8-GPU B200 `own` node; Nano is 16B.
- **Two dependency worlds**: cosmos-framework (`transformers>=4.57,<5`, CUDA13, torch 2.10,
  flash-attn3, Wan2.2 VAE, gated `nvidia/Cosmos-1.0-Guardrail`) vs cosmos-repo diffusers path
  (`transformers>=5.11`). NGC `pytorch:25.09-py3` base + NAS venv recommended.
- **Guardrails** required for Generator unless disabled (`--no-guardrails`).
- **Limitations (report)**: action-state drift, unstable camera motion in long/high-res rollouts,
  fixed predefined action dims (no turnkey new-embodiment recipe), approximate physics.

## 7. Candidate v12 plan (draft)
1. **Validate zero-shot**: run `camera_pose` **forward_dynamics** on base Cosmos3-Nano/Edge with a
   Blender camera trajectory → confirm the world model reproduces our camera dynamics
   (`CF/inputs/omni/action_forward_dynamics_camera.json` as template). Cheap, no training.
2. **Data conversion**: Blender trajectories → LeRobot v3.0, actions `[T,9]` via
   `pose_abs_to_rel(rot6d, backward_framewise)`, no gripper, `viewpoint=ego_view`; compute 9D
   normalizer stats. Template: `CB/finetune/data_processing_for_egocentric_hand_action.py`.
3. **Goal conditioning decision** (§5) — start low-touch (JSON prompt), measure goal→action
   dependence (our recurring bottleneck) before investing in a structured-goal modality.
4. **Post-train policy**: clone `action_policy_libero_nano.py` → camera config, frozen backbone +
   train gen/VAE/action heads, on 1×8-GPU node (Edge first). Export + closed-loop eval via a
   camera analog of `CF/scripts/action_policy_server_*.py` + `CF/simulation/.../closed_loop_eval.py`.

## 8. Biggest risks / open questions
1. **Goal representation** — the same goal→action-dependence wall as v11; text-prompt goals may be
   too weak, structured injection is off-recipe.
2. **No camera policy precedent** — we'd be first to post-train camera_pose policy; only FD/ID proven.
3. **Heavy for a low-D task** — even Edge is 4B + frozen video VAE/ViT; ~35-step diffusion + CFG per
   decision is much heavier than a single-forward action head. Worth it only if the pretrained
   omnimodal world knowledge actually lifts goal-conditioned camera control.
4. **Convention reconciliation** — Blender (+X right/+Y up/−Z fwd) + our orientation_6d/rotvec vs
   Cosmos meters + rot6d (relative, first-2-columns, backward-framewise); get abs→rel direction right.
5. **Env/ops** — CUDA13, Wan2.2 VAE, gated guardrail, LeRobot v3.0, two transformers versions.

## Cross-source confidence
- **High (all 3 agree + code)**: two-tower MoT, action = continuous flow-matching denoised, camera_pose
  9D=3+6 no gripper (domain_id 2), 3 modes, finetune freezes backbone + trains gen/action heads,
  goal = text-prompt-only.
- **Medium (report/cards, not re-derived from PDF)**: exact per-tower param split, DROID/RoboArena
  numbers, video VAE compression ratio, training data mix specifics. Verify against the PDF
  (technical-report.pdf exceeded the 10MB fetch limit) before citing numbers.

---

## Appendix A — v7 data distribution findings (2026-07, evidence for v12 design)

Measured on 300 sampled placements / 25k multiscale windows (`scripts/analyze_distributions.py`).

**ACTION (5D, raw) is CLEAN — not the source of high val loss.**
- All dims 0-centered, smooth; d_forward widest (std 0.27, dolly-dominant), d_pitch smallest (0.057).
- **d_yaw: 100% within [-π,π], 0% wrap artifacts** — the Δyaw delta IS correctly wrapped
  (`encode_action_5d`: `(az1-az0+π)%(2π)-π`). No action-encoding bug.
- within-chunk std 0.01–0.08 → the 8-step chunk is a smooth near-constant-velocity move.
- ⇒ switching the action rotation to rot6d is for **Cosmos compatibility**, not bug-fixing.

**GOAL PROFILE (8-key V5) is the problem — confirms "meh":**
1. **azimuth cyclic seam**: `cam_to_obj_azimuth_deg` ∈ [0,359], **14% sits within 15° of the 0/360 seam**
   → linear normalization maps these ambiguously to ±1 (discontinuous goal input). Hurts goal→action.
2. **occupancy saturates**: 10% of windows at 100 (can't distinguish close vs very-close).
3. **off-screen / poorly-framed goals**: object_center_x ∈ [-946,1788] (render width ~1024-1280),
   body_in_frame_ratio median only 27% → many goals have the subject largely OUT of frame (2D-bbox
   artifact, data-quality concern).
4. **redundancy**: occupancy ~ bbox_x_offset r=0.68, bbox_x ~ bbox_y 0.55 (apparent-size double-encoded).
5. azimuth & elevation ~uncorrelated with framing keys (viewing-direction ⟂ framing — good).

**Design implications:**
- The high val loss is on the GOAL side, not the action. Action rep is fine.
- Serializing the goal to NL (the chosen staged path) **naturally dissolves the azimuth seam** (words,
  not a wrapped angle) and lets us use cinematography descriptors.
- `reward.py` closed-loop distance is already pose-based / seam-safe (great-circle on the viewing
  sphere) — reuse it for eval/relabel; the seam only ever hurt the 8-key *input vector*.
- Goal-profile redesign should: fix azimuth continuity, de-saturate/curate framing, cut redundancy,
  reduce 2D-bbox dependence.

## v12 self-containment (rule, 2026-07)
All v12 code lives under `DronePhotographerV12/`; no imports from V11 or the original repo. Verified
reusable modules copied into `src/{common,scoring,utils}`; `data/` is a symlink to the original raw
data location (the only allowed external dependency). A dedicated v12 venv (CUDA13 / transformers /
uv, for Cosmos 3) is TODO; pure numpy/torch analysis uses the shared venv meanwhile.
