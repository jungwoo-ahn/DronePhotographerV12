# v4 session — what changed, why, and what it measured

Covers the work that produced `runs/lerobot_v4` and the `camera_policy_nano_v4` training run.
Every number below is measured on this repo's data, not estimated.

---

## 1. The dataset we had been training on was the wrong one

**Finding.** All training up to v3 used `data/trajectories/*` — 3,931 placements whose camera
aims at the subject's **mid-torso**. Measured over 200 placements:

```
(subject_center_z − subject_foot_z) / subject_height = 0.500   (range 0.495–0.507)
```

Combined with a pitch jitter of up to **+45°** at close range (`src/policy/data/sampling.py`,
`PITCH_LERP_NEAR = (1.0, -15.0, +45.0)`), that throws the face out of frame. Over 51,488 frames:

| | old (lookat 0.50) |
|---|---|
| head cropped | **72.4 %** |
| **both head AND feet cropped** | **55.6 %** |
| nothing cropped | 3.7 % |

A better dataset already existed and had **never been used**:
`data/trajectories/v7_stage2_renders_lookat075/` — 7,885 placements, `lookat_height_fraction
= 0.75`, already rendered and Stage-3 scored.

**Why it was invisible.** The export looks for `<root>/<dir>/data.json`; this dataset nests one
level deeper, so `if (r/d/"data.json").exists()` was False and it was skipped silently. Verified:
`export 가 본 배치: 3931, lookat075 포함? False`.

It is a **strict superset**: all 3,931 old placement names are in it, plus 3,954 more. Same 102
objects.

| | old | 075 |
|---|---|---|
| placements | 3,931 | **7,885** (7,668 with renders) |
| both ends cropped | 55.6 % | **15.5 %** |
| nothing cropped | 3.7 % | **16.8 %** |
| back-family sectors | 67.7 % | **49.3 %** |

**What did NOT need redoing** (each verified, not assumed):

- **Facing map** — 102/102 objects covered by `runs/facing_map_final.json`. It is built from
  isolated turntable renders, so it is a property of the asset, not of the trajectory dataset.
- **Azimuth convention** — recomputed `cam_to_obj_azimuth_deg` from camera pose + `subject_center`
  and compared to the stored value: median difference **0.17°** on both datasets (= int rounding).
- **rot6d / camera conventions** — derived from the absolute `pos/forward/up` in `trajectory_32f`.
- **RENDER_WIDTH/HEIGHT** — 1024×768 on both.
- **`ACTION_SCALE`, `VALUE_SCALE`** — checked and found **unused**: both config sites set
  `action_normalization=None` (`_stats_path` is "only consulted when action_normalization is not
  None"), and `VALUE_SCALE` is referenced only from `src/common/dataset_base.py`, the v11-lineage
  dataset. The v12 path (`src/data/cosmos_camera_dataset.py`) has no value channel. Refitting them
  was in the plan and was **removed after checking** — it would have been wasted work.

---

## 2. Re-scored 075 against `subject_center`

**Finding.** One field was *not* center-referenced. The 075 Stage-3 scores were computed against
`subject_lookat` (0.75 height):

```
075   |stored_el − el(center)| 5.48°   |stored_el − el(lookat)| 0.25°
old   |stored_el − el(center)| 0.24°   |stored_el − el(lookat)| 0.24°
```

Also confirmed the camera really does aim at `subject_lookat`: of 110 frames with near-zero
jitter, **110/110** pointed at it (angle 0.39° vs 6.75° to `subject_center`).

**Why it mattered.** The camera poses in `trajectory_32f` are absolute world-frame, so 0.75 was
only a *sampling* choice — everything downstream can stay `subject_center`-based. But the stored
elevation would then disagree with the pose-derived one by ~5.5°, desynchronising
`profile_to_geometry` from `pose_to_geometry`.

**Fix.** Re-ran `scripts/v7_stage3_score.py` (which computes `O = data["subject_center"]`) over
all of 075, 3.4 min:

```
ok=7668  pending=217  fail=0
after:  |el − el(center)| 0.25°   |el − el(lookat)| 5.92°   |az − az(center)| 0.25°
```

The 217 "pending" are placements with `K_accepted = 0` — the sampler produced no camera pairs, so
there is nothing to render or score. Not a processing gap; the plan's claim that re-scoring would
"fill" them was wrong.

---

## 3. Repointed the pipeline — and a path hazard

Changed the defaults in `scripts/export_lerobot.py`, `closed_loop_eval.py`, `gt_replay_eval.py`,
`axis_probe_eval.py` to `data/trajectories/v7_stage2_renders_lookat075`.

**Hazard:** `src/common/annotations.py:list_annotation_files` globs `**/data.json`
**recursively**. Pointing it at the parent `data/trajectories` yields 3,931 + 7,885 = **11,816
placements with 3,931 duplicated names**. Always name the subdirectory.

---

## 4. Replaced the framing gate

**Finding.** The gate was selecting *for* bad photographs. `_is_well_framed` required
`body_in_frame_ratio >= 70`, a 2-D **area** ratio that cannot tell which end of the subject the
frame cuts — so a beheaded subject and a chest-up portrait score identically. Measured on frames
that passed the gate: **72.7 % had the head cropped**. And a bust shot shows 35–60 % of the body,
so it can never clear 70: of 6,275 bust-extent frames in 075, **0 passed**.

Also found the second half of the gate was **dead code**: `require_center_on_screen` checks that
`object_center_x/y` are on screen, but `_apply_visible_geometry` recomputes those from the
*clipped* bbox, so they are on screen by construction. It could only ever reject a missing key.

**Fix** — `src/common/annotations.py`:

- New `_apply_crop_extent`, called from `_apply_visible_geometry` (which already holds the signed
  unclipped `bbox_xyxy_full` and the render dimensions):

  ```
  head_in_frame = y0 >= 0
  top_cut_frac  = max(0, -y0)    / span
  bot_cut_frac  = max(0, y1 - H) / span
  visible_frac  = (min(y1,H) - max(y0,0)) / span
  ```

- `_is_well_framed` now gates on `visible_frac >= 0.35` ("at least a bust's worth is in frame"),
  **deliberately crop-side agnostic** so the pool's own top/bottom mix survives. Falls back to the
  old area ratio when there is no signed bbox, so a missing key cannot silently empty the dataset.
- `DEFAULT_MIN_GOAL_BODY_IN_FRAME = 70.0` → `DEFAULT_MIN_GOAL_VISIBLE_FRAC = 0.35`; parameter
  renamed through `iter_goal_start_windows` → `src/common/dataset_base.py`, and in the two scripts
  that import `_is_well_framed` directly (`plan_rerender_yaw.py`, `plan_rerender_yaw_v2.py`).

**Effect on the crop mix** (post-gate, vs the pool's natural mix):

| | none | bottom | top | both |
|---|---|---|---|---|
| pool | 15.9 % | 50.9 % | 17.4 % | 15.8 % |
| old gate | 13.9 % | 34.3 % | **27.2 %** | 24.6 % |
| new gate, occupancy ≥ 20 | 7.9 % | 41.0 % | 18.4 % | **32.7 %** |
| new gate, occupancy ≥ 10 | 17.4 % | 42.3 % | 20.5 % | 19.8 % |

The head-cut bias is gone (27.2 → 18.4 %, pool is 17.4 %). But the occupancy **floor** turned out
to be a framing filter in disguise — an uncropped subject is shot from far enough away to fit, so
it occupies little of the frame. Lowered `DEFAULT_GOAL_OCCUPANCY_RANGE` from `(20, 80)` to
`(10, 80)`; the final export's mix is `none 22.4 / bottom 41.5 / top 19.7 / both 16.4 %`.

Gate pass rate: **5.6 % → 42.2 %** of frames.

---

## 5. Completed the prompt

**Finding.** `goal_prompt` serialized **3 of the 8 goal keys** (`occupancy`,
`subject_bearing_deg`, `cam_to_obj_elevation_deg`). `body_in_frame_ratio`, `object_center_x/y` and
`bbox_x/y_offset` never reached the model — and the model only ever sees the goal as **text**.

The axis probe on the old model confirmed the consequence exactly: the two keys absent from the
prompt scored at chance (`object_center_x` 3/6 correct direction, response −0.06; `object_center_y`
3/6, +0.01) — the request was never delivered.

**Fix** — `src/data/lerobot_export.py`:

- All 8 keys now appear as **word AND raw number**, plus a crop clause.
- Category words come from `src/goal_authoring/vocab.py` (`SHOT_SIZE`, `BODY_FRAMING`, `PLACE_X`,
  `PLACE_Y`, `ELEVATION`) instead of the hand-rolled copies, which had **drifted**: elevation cut
  at −25/+10 vs the vocab's −20/+15, and label `"medium close-up shot"` vs `"medium close-up"`. The
  same goal produced different sentences depending on which path serialized it.
- `crop` is an **optional** parameter, because crop is not in the goal vector — it needs the signed
  bbox. Callers that synthesize a goal by perturbing a vector (the axis probe) omit the clause
  rather than assert what they cannot know.

Example output (~90 tokens, vs ~46 before; `camera_pose` was trained on 35–44-token prose, and the
cap is 4096):

```
Move the camera to achieve this shot: a medium-wide shot of the subject from the subject's
front-right, at eye level, centered and lower in the frame, partially cut off, cropped below
the waist. (bearing 57°, occupancy 20%, elevation -8°, body_in_frame 52%, center 399/492 px,
half_size 140/276 px, visible 0.52)
```

**Deliberately NOT done: a 9th goal-vector dimension.** The vector reaches Cosmos only through
this function; widening it touches `V5_SCORE_KEYS`, `DEFAULT_V5_RANGES`, the index-keyed stats in
`fit_action_stats.py` and six tests, for a dimension the model never sees.

**Deliberately NOT done: a system/meta prompt.** It would be byte-identical across all 150 k
samples, so it carries no signal for telling goals apart, and it would push the conditioning text
well outside the short-prose distribution the base model saw. If `object_center_x/y` still fail the
axis probe, add a one-line meta prompt *then* and A/B it.

---

## 6. Stop supervision (delta range) and sector balance

Both were in place before the 075 switch but are part of what v4 trains on.

**Stop supervision.** `DEFAULT_DELTA_RANGE` was `(8, 32)` with a hard guard `d_min >= chunk_size`
"so the action chunk cannot overshoot the goal". Measured: **delta minimum 8, delta < 8 samples: 0**
— the policy had *never* seen a state closer than 8 frames from the goal, which is exactly where a
rollout ends up after one or two chunks. The rule meant to prevent label overshoot created the
blind spot that causes rollout overshoot.

Now `(0, 32)`, with the walk **clamped at the goal** so the trailing steps repeat the goal frame
and their action deltas are exactly zero. `DEFAULT_NEAR_FRACTION = 0.25` stratifies the sampling —
without it near-goal starts would flood the set (every goal admits ~2·chunk_size of them) and the
policy would learn to sit still. Verified: 25 % near, **340/340 of them contain zero-action steps**,
18 with delta = 0.

**Sector balance.** `--balance-sectors` fills each of the 8 sectors to `max_episodes/8` and logs
`UNDER TARGET` on a shortfall. On 075 this costs almost nothing — the natural mix is already
front 12.4 / front-right 15.6 / right 14.6 / back-right 13.0 / back 9.2 / back-left 14.0 /
left 8.2 / front-left 12.9 %, a 1.9× spread versus the old data's 22× (back 31 % vs left 1.4 %),
and every sector draws from 6,551–7,013 placements (old `left`: 93).

**Note:** the 300-placement yaw re-render does **not** carry over — 075 has no
`placement_yaw_deg`, so that work applies only to `runs/rerender_yaw/`.

---

## 7. mp4 sharding — training throughput

**Finding.** The first v4 run trained at **240 iter/h**, half of v3's 480. Diagnosis via `sjob` +
`sprobe`:

```
GPU[2] util 0 %   GPU[3] util 39 %   TC 8 %   CPU 45.8 %   D-load 0.0
both ranks: futex_wait_queue × 55–56 threads, plus pipe_read
```

Not compute-bound (TC 8 %), not CPU-bound (46 %), not disk-blocked (D-load 0). The change was the
video: 47,635 episodes / 1,149 MB → 150,000 episodes / **3,706 MB in a single mp4**. Every
`__getitem__` seeks into that one file by `from_timestamp`, and the seek cost scales with size.

**The first hypothesis was wrong** — "raise `--cpus-per-gpu`" would not have helped, and CPU at
46 % was the evidence against it.

**Fix** — `src/data/lerobot_export.py` gained `episodes_per_video` (default 20,000; 8 shards for
150 k), with `file_index` set per episode and the `from_timestamp` cursor **reset per file**.

**Verified by equivalence, not by inspection** — a silent off-by-one here would train on the wrong
video. Same 120 episodes exported single-file and 4-shard, then decoded and compared:

```
file_index {0:30, 1:30, 2:30, 3:30}
from_ts  ep35 → 1.500 s (=5×9/30)   ep70 → 3.000 s   ep115 → 7.500 s
max mean pixel difference 0.2288/255   mismatches 0/18   prompts identical
```

Result: **240 → 598 iter/h**, faster than v3's 480.

---

## 8. Evaluation infrastructure

Fixed in this session, in rough order of how much damage each was doing:

| fix | what was wrong |
|---|---|
| `ModelMode.POLICY` | `build_action_batch` reads `model_mode.value`; we passed the string `"policy"`. `ModelMode` is a **StrEnum**, so `"policy" == ModelMode.POLICY` is True and every check upstream passed — it only raised inside the builder, after a 106 s model load and a Blender render, **once per episode**. Every rollout had been failing. |
| `--held-out-only` | The eval's "held-out" claim was not enforced. The export consumes 4 episodes from each of a limited number of placements, and the eval drew from the same shuffled listing with the same seed — **12 of 12 evaluated placements were trained-on**. Now it replays the export enumeration and excludes them, and records `held_out_only` in the output. |
| namespaced `frames_dir` | Two concurrent runs wrote `ep000_start.jpg` into the same `frames/` directory and overwrote each other. 80 files were claimed by both. Numbers were unaffected (they come from env poses) but the report showed **two different scenes in one strip**. Now `frames_{out.stem}`, and the report refuses to display frames from a run that does not declare `frames_dir`. |
| sector exact-match | `sector3` returns `front`/`side`/`back`, and two of those **collide with sector8 names**, so `--sector back` also accepted back-left/back-right. Two "disjoint" shards shared 15 of 20 placements. Verified the fix over 0–355° in 5° steps: overlap 0, gaps 0. |
| partial writes | Results were written only at the end. Two ~16-minute preempted runs finished episodes that were unrecoverable. Now flushed after every episode via temp-file + `replace`. |
| `--resume` | A preempted run redid everything. At 80 episodes × ~3 min in a 30-minute preemption window it could never finish. Now skips episodes already in the partial. |
| self-requeue | `sbatch_closed_loop_eval.sh` traps SIGTERM, checks `sacct` State, and resubmits only on a real preemption. Handles the three documented pitfalls: no `exec` (it would destroy the trap), background child + `wait` (a foreground child defers the handler), and the `scancel` check. |
| trained-set cache | Replaying the export enumeration costs ~35 min over 7.7 k placements at `max_per_pair=24`; every shard paid it again, and one job spent its entire 21-minute slot on it and produced **0 episodes**. Cached, keyed by the parameters that define the trained set. |
| `--time 3 h → 12 h` | 160 episodes need ~8 h. The eval silently TIMED OUT at 63/160. |
| crop mix in the export report | The old gate made 72.7 % of goals head-cropped and **nobody could see it**. The export now prints the crop mix alongside the sector mix, so a regression shows up at export time rather than after training. |

**Checkpoint retention.** `keep_last=5` at 137 GB each means a checkpoint an eval is pinned to can
vanish while the job waits in the queue. `iter_5000` and `iter_13000` were copied to
`runs/keep_checkpoints/`; `iter_13000` was indeed deleted from the live directory before its axis
probe ran, and the backup is what let it run.

---

## 9. Held-out loss probe (`src/train/heldout_loss.py`)

Added because the framework's validation path is a stub — `OmniMoTModel.validation_step` is a bare
`pass`, so `run_validation=True` does not produce a number, it takes the run down, and `--dryrun`
cannot catch it (the config resolves; the stub is only reached at runtime).

Calls `training_step` under `no_grad` instead, with three choices that make the curve readable:

1. **Fixed batches**, drawn once at train start.
2. **Fixed noise** — the flow-matching loss samples a random sigma per call and that variance
   dwarfs the signal (the same effect swings `goal_dep/ratio` between 0.9 and 3.7 on a model that is
   genuinely using its goal). Seeding per batch index makes every evaluation ask the same question.
3. **A matched train control** through the identical procedure, so `heldout/gap` is a difference of
   comparable estimators rather than of estimator *types*.

`num_batches` defaults to 2: the val split packs into exactly 2 batches, and asking for more raised
`StopIteration` inside `on_train_start`, which the probe's own guard swallowed — it disabled itself
and logged nothing, silently.

This is what showed the v3 run was overfitting from iter 1,000 onward (gap 2.1× → 9.9×), which had
been invisible in every run before it.

---

## 10. What it produced

**Data** — `runs/lerobot_v4`: 150,000 episodes / 1,350,000 frames / 44,608 tasks, 8 mp4 shards.
Sector mix 12.5 % × 8, all filled. Near-goal 25 %. Crop mix
`none 22.4 / bottom 41.5 / top 19.7 / both 16.4 %`.

**Training** — `camera_policy_nano_v4`, 20,000 iterations, no overfitting at any point:

| run | episodes | val minimum | val at minimum | gap at iter 6 k |
|---|---|---|---|---|
| v2 | 4,000 | iter 1,000 | 0.00341 | **9.9×** |
| v3 | 47,635 | iter 5,000 | 0.00075 | 1.49× |
| **v4** | **150,000** | **iter 13,000** | **0.00026** | **0.90×** |

`goal_dep/ratio` median: v2 1.30 → v3 1.72 → **v4 3.19**.

**Held-out rollout** (`--held-out-only`, 8 sectors balanced):

| | n | mean improvement | given back | overshoot | settled | starts near | starts far |
|---|---|---|---|---|---|---|---|
| old data, iter 11,000 | 40 | +0.317 | 44 % | **88 %** | **12 %** | **+0.079** | +0.476 |
| v4 iter 5,000 | 80 | +0.286 | 56 % | 81 % | 19 % | +0.095 | +0.378 |
| v4 iter 13,000 * | 93 | +0.411 | 41 % | 70 % | 30 % | +0.159 | +0.499 |
| **v4 iter 20,000** | **158** | **+0.365** | 42 % | **68 %** | **32 %** | **+0.132** | +0.487 |

`*` sectors unbalanced (93/160) — not comparable, shown for completeness.

Sector means at iter 20,000 (20 each): front-left +0.655, front-right +0.517, front +0.405,
left +0.362, back-left +0.331, right +0.277, back +0.243, back-right +0.123. **The front family is
now the best-performing axis**, having been the rarest in the old data.

**Axis probe** (iter 13,000, 8 start frames, 16 conditions per axis):

| axis | before (response / correct sign) | v4 | response ÷ crosstalk |
|---|---|---|---|
| `subject_bearing_deg` | +6.80 · 4/8 | **+9.58 · 11/16** | 0.81 |
| `object_center_y` | +0.01 · 3/6 | +1.13 · 10/16 | 0.05 |
| `object_center_x` | −0.06 · 3/6 | +0.75 · 9/16 | 0.04 |
| `cam_to_obj_elevation_deg` | +0.23 · 4/8 | +2.04 · 9/16 | 0.10 |
| `occupancy` | −0.02 · 4/8 | +0.47 · 9/16 | 0.02 |
| null (goal = current state) | moved 54.4° | **moved 22.5°** | — |

Read honestly: only the null result and `bearing` are solid. At n = 16, 9/16 has a 0.40 chance of
arising by luck and 11/16 a 0.105 chance. Every axis moved off *exactly* 50 % and every response
turned positive — the old model gave **the same −15° for `occupancy +20` and `occupancy −20`**,
i.e. it did not read the request at all — but the effect sizes are small and the crosstalk ratios
(0.02–0.10 outside bearing) say the untargeted axes still move more than the targeted one.

---

## 11. Open — the stopping problem

Overshoot fell 88 % → 68 % and settling rose 12 % → 32 %, but the policy still does not stop.

The rollout has **no stopping criterion**: `n_chunks = ceil(delta / chunk_size) + extra_chunks`, and
`--extra-chunks 1` deliberately runs one chunk past the goal to see whether it settles. If the
policy emitted zeros there, that chunk would be a no-op. It is not:

```
                    last chunk motion   previous chunk motion
iter_5000                0.2773               0.2546
iter_13000               0.3148               0.2493
drift back after best    0.21–0.26
```

The final chunk moves as much as the one before it. This is not "overshoots slightly" — it is
**does not recognise it has arrived**. So the plan to use `‖action‖ < ε` as a termination test does
not work on this model; there is no threshold that separates 0.31 from motion.

Zero-action supervision (25 % near-goal samples) clearly helped the path — but it taught the policy
to *pass through* the goal more accurately, not to *end* there. The natural next step is a **shoot
action** (10-D, `build_action_spec(Pos(), Rot("rot6d"), Gripper())`) so termination is something the
policy declares rather than something we infer from action magnitude. Labels are free by hindsight:
`shoot = 1` on the window's goal frame, 0 elsewhere.

Caveat already on record in `src/data/cosmos_camera_dataset.py`: do not pad a *dummy* gripper
channel — it sends `compute_idle_frames` down the gripper branch and breaks the `raw_action_dim=9`
contract. A real binary signal may be exactly what that branch is for, but that has to be tested,
not assumed.

---

## 12. Module 2 (NL / reference image → goal) brought back in sync

Sections 4–5 changed what a goal *is* and how it serializes, but `src/goal_authoring/` had not
been touched. Three defects followed.

**A partial profile asserted things nobody asked for.** Once all 8 keys were emitted as
word+number, the `0.0` both flatteners write for an unspecified key stopped being invisible: a
profile specifying only occupancy / bearing / elevation printed `center 0/0 px` (the top-left
corner) and `body_in_frame 0%` (subject not in frame at all). `goal_prompt` now takes
`specified=`, and omits a clause *and* its number when the key is absent. `None` keeps the old
behaviour, so the exporter is untouched.

**Nothing produced the crop clause.** `grep` found zero hits for `top_cut_frac` / `bot_cut_frac` /
`visible_frac` inside `src/goal_authoring/`. `estimate_body_in_frame` detected the crop internally
and then threw the *side* away into one of three magic area numbers (100 / 55 / 60) — the exact
ambiguity §4 removed. Replaced by `crop_from_bbox`.

**Every eval prompt was missing the crop clause.** Training wrote it
(`export_lerobot.py:126 → goal_prompt(g, crop=w.goal_frame.raw)`); `closed_loop_eval`,
`gt_replay_eval` and `recon_from_reference` all called `goal_prompt(goal_vec)`. So **every rollout
number in §10 was measured with an out-of-distribution prompt** — a lower bound, not a neutral
measurement. All three now pass `crop=`, and `tests/test_prompt_train_eval_parity.py` asserts
byte-equality against `runs/lerobot_v4/meta/tasks.parquet` rather than trusting inspection.
(Reading that table needs care: LeRobot v3.0 stores the task string as the frame *index* and keeps
only `task_index` as a column, so the obvious `df[df.columns[0]]` returns integers and matches
nothing — it read 0/25 before that was noticed.)

### What a reference image can and cannot reveal

A training frame's crop comes from the UNCLIPPED projected bbox, so it knows the side *and* the
fraction. A photograph's bbox is already clipped at the frame edge: the subject's extent beyond it
is not observable. `crop_from_bbox` therefore reports the side, and reports `visible_frac` only
when nothing is cut (where it is exactly 1.0). This is the one place training has strictly more
information than inference, and fabricating a fraction would put an invented number straight into
the only channel the policy has.

### The crop round trip, measured

Rendered 96 training frames labelled by the exporter (24 per class) and asked the reference
estimator to recover the side.

| rule | overall | `bot` | `top` | `both` | `none` | prevalence-weighted |
|---|---|---|---|---|---|---|
| box edge only | 70/96 (73 %) | 23/24 | 21/24 | 20/24 | **6/24** | 77.9 % |
| + keypoint veto (shipped) | 81/96 (84 %) | 23/24 | 20/24 | 14/24 | **24/24** | **87.1 %** |

The `none` collapse had one cause, visible in the failing frames: on a shadowed floor the person
box runs to exactly `y1 = image_h` while the projected mesh ends 12–119 px higher, so a fully
visible subject reads as bottom-cropped. Sweeping `margin_frac` does not fix it — 0.002 → 0.05
moved the total 73 % → 68 % — so the margin default dropped to 0.002 and ankle keypoints carry the
rest. A shadow has no ankles.

The veto costs `both` (20/24 → 14/24): YOLO-pose still emits a confident ankle for a subject whose
feet are out of frame. Weighted by how often each class actually occurs among goal frames
(`bot` 42.8 %, `none` 19.6 %, `both` 19.1 %, `top` 18.4 %) the trade is clearly positive. The box
edge still *detects* the crop; keypoints may only cancel a false positive, never invent one —
`tests/test_crop_from_bbox.py` pins that direction explicitly.

This revises the plan's stated decision ("bbox edges, no keypoints"), on measurement rather than
preference. `kp=None` restores the box-only rule.

### Consolidation

- Three viz serializers re-implemented `shot_size` / `elevation_word` / `body_word` with drifted
  cuts (−25/+10 instead of −20/+15), the label `"medium close-up shot"` instead of
  `"medium close-up"`, and an invented `"mostly out of frame"` band — so the same goal read
  differently depending on which script rendered it. All three now import `vocab`.
- `30 <= occupancy <= 92 and body_in_frame_ratio >= 45` was copy-pasted into 6 scripts with three
  further variants (88, 90, 50). Replaced by `is_goal_frame`; the one survivor sits inside a helper
  now explicitly marked dead.
- **22 scripts read the superseded render tree.** The new data is nested *inside* the old one
  (`data/trajectories/v7_stage2_renders_lookat075/` under `data/trajectories/`) and 3931 of 7885
  placement names appear in both, so a script pointed at the parent silently reads the pre-yaw,
  mid-torso-look-at renders under the right-looking names. Added `DEFAULT_TRAJ_ROOT` /
  `LEGACY_TRAJ_ROOT` to `dataset_base`; 19 scripts migrated, and the 3 that exist to measure what
  the re-render changed are now annotated as deliberately historical.

### Regenerated

`runs/recon_ref5/goals.json` — 15 cases, built under the current gate, referencing the 075 tree,
carrying `goal_specified` and `goal_crop`. The prompts now read

```
... a wide shot of the subject from the subject's front-right, at eye level, centered and
lower in the frame, uncropped. (bearing 44°, occupancy 13%, elevation 3°, center 531/486 px,
half_size 106/241 px, visible 1.00)
```

with no `body_in_frame` clause or number — a reference photo cannot reveal it — and `visible`
present only on the uncropped case. The five earlier `runs/recon_ref*/goals.json` remain as the
record of what was measured before; they were built under `--goal-occ (20,80) --min-body 70` with
old-tree paths and should not be reused.

---

## Files touched

```
src/common/annotations.py          crop signals, visible_frac gate, occupancy floor, delta range
src/common/dataset_base.py         parameter rename passthrough
src/common/run_info.py             NEW — shared placement-transform adapter
src/data/lerobot_export.py         full 8-key prompt, vocab reuse, mp4 sharding
src/train/heldout_loss.py          NEW — held-out loss via training_step
configs/policy/action_policy_camera_nano.py   HeldOutLossProbe registration
scripts/export_lerobot.py          075 root, sector balance, crop-mix report, shard flag
scripts/closed_loop_eval.py        ModelMode, held-out-only + cache, per-sector, exact sector
                                   match, partial writes, resume, namespaced frames
scripts/gt_replay_eval.py          NEW — trained-trajectory reproduction (tests A and B)
scripts/axis_probe_eval.py         NEW — per-axis goal steering probe
scripts/build_eval_report.py       NEW — self-contained HTML report builder
scripts/v7_stage3_score.py         restored from history; re-scored 075
scripts/sbatch_closed_loop_eval.sh self-requeue on preemption, 12 h limit
scripts/sbatch_export_lerobot.sh   NEW — CPU-only export wrapper
scripts/sbatch_gt_replay.sh        NEW
scripts/sbatch_axis_probe.sh       NEW
scripts/plan_rerender_yaw_v2.py    NEW — deficit-driven yaw planner (concluded: not worth running)

— §12, Module 2 —
src/goal_authoring/from_reference.py   crop_from_bbox + keypoint veto; estimate_body_in_frame gone
src/goal_authoring/vocab.py            CROP_SIDE table + crop_label()
src/goal_authoring/from_language.py    crop_side axis + keyword rules
src/data/lerobot_export.py             goal_prompt(specified=) — omit unasked-for keys
src/common/dataset_base.py             DEFAULT_TRAJ_ROOT / LEGACY_TRAJ_ROOT
scripts/prepare_recon_goals.py         075 root, --min-visible, is_goal_frame gate
scripts/recon_from_reference.py        crop=, specified=, root default
scripts/closed_loop_eval.py            crop= on the eval prompt
scripts/gt_replay_eval.py              crop= on both prompt sites
scripts/viz_{goal_prompt_gallery,prompt_gallery_all,design_review_html}.py   import vocab
19 further scripts                     DEFAULT_TRAJ_ROOT instead of the superseded tree
tests/test_crop_from_bbox.py           NEW — box detects, keypoints only veto
tests/test_prompt_train_eval_parity.py NEW — eval prompt == training prompt, byte for byte
```
