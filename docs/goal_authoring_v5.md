# Goal authoring, resynced with the v5 policy (2026-09-01)

`src/goal_authoring/` turns a user's words and/or a reference photo into the goal that conditions
the trained camera policy. This note records what the v5 policy actually accepts, what the module
now does about it, and what was measured. Every number is from this repo's data.

## 1. The contract the policy imposes

The policy sees the goal only as text. `camera_policy_nano_v5` (iter 40 000, `runs/lerobot_v5`)
was trained on 44,560 unique prompts and **every one of them is complete**: eight keys as word +
number, a crop phrase, and a `visible` number —

```
Move the camera to achieve this shot: a wide shot of the subject from the subject's front, at eye
level, centered and mid in the frame, full body in frame, uncropped. (bearing 21°, occupancy 12%,
elevation -6°, body_in_frame 100%, center 600/430 px, half_size 96/246 px, visible 1.00)
```

`serialize.TRAINING_PROMPT_RE` is that shape; 44,560 / 44,560 match it and parse back into the
same string byte for byte (`tests/test_serialize_contract.py`). A prompt with a clause missing
is one the policy has never been shown.

The trained goal *support* is also narrower than the vocabulary (train-scene rows of the bank):

| key | trained range | vocabulary words with no / thin support |
|---|---|---|
| occupancy | 10 – 80 %, median 20 | extreme wide shot (<8 %): **0**; close-up (≥78 %): **22** (0.05 %); medium close-up: 713 (1.6 %) |
| elevation | −51 … +19°, median −10 | low angle (≥15°): **267** (0.6 %) |
| placement | on-screen only | off-screen left/right/top/bottom: **0** |
| crop | legs 26.6 / uncropped 21.7 / head 19.7 / waist 16.8 / both 15.2 % | — |

## 2. What was wrong before

1. Authored goals were **partial** and the serializer dropped the unspecified clauses
   (`goal_prompt(specified=)`), so every NL goal — and every image goal, which never carried
   `body_in_frame` / `visible` — was out of distribution.
2. `GoalProfile.to_nl()` built a **second sentence** (`vocab.profile_to_nl`) that was never the
   training one; demos, the report and a round-trip test displayed it as "the prompt".
3. Vocabulary centroids sat **outside the support**: close-up 88 %, extreme wide 4 %, low angle +33°.
4. Module 2 collapsed a bottom cut to `bot_cut_frac = 1.0` → always "cropped below the waist";
   "cropped at the legs", the most common training word, was unreachable.

## 3. What the module does now

```
NL ──keyword (+LLM) classifier──▶ partial GoalProfile ─┐
                                                       ├─▶ complete ──▶ profile_to_prompt ──▶ TRAINING_PROMPT_RE ✓
image ──YOLO-pose──▶ box, bearing, elevation, crop side + body-extent fit ─┘
```

* `serialize.py` — `goal_prompt` moved here (re-exported by `src.data.lerobot_export`, same
  function object), plus the shape regex and its inverse.
* `goal_bank.py` + `assets/goal_bank_v5.npz` (595 KB) — the 44,560 training goals as numbers,
  with train / held-out episode counts; band masks, distances, quantiles, support tables.
  Rebuild with `scripts/build_goal_bank.py` (asserts the byte-level round trip).
* `completion.py` — two completers, one interface, per-key provenance:
  * **`BankCompleter`** (default) filters the bank to the requested categories, ranks by distance
    to the requested numbers, and anchors on a training goal that is both close and typical.
    NL-only requests therefore yield a prompt that **is** a training prompt; measured numbers
    (`GoalProfile.exact`) are kept. Categories with < 30 matching goals are relaxed to
    nearest-by-value and reported.
  * **`RuleCompleter`** (dataset-independent) models the subject as one upright box of aspect
    2.4 (anthropometric; the export's uncropped goals happen to have median 2.4): occupancy
    sizes it on the *clipped* area, placement / crop position it, and centre, half-sizes,
    body_in_frame and the crop fractions follow. Bearing / elevation default to front / 0°.
    A requested crop word is honoured by widening the box; a defaulted one gives way to a cut.
    `author_goal` clamps rule requests to the bank's p0.5–p99.5 support when a bank exists.
* `from_reference.py` — `estimate_vertical_extent` fits the visible joint levels to standard
  segment heights (Drillis & Contini) and extrapolates vertex and soles, so a cut side becomes a
  cut *fraction* (and `visible`); the box edge still decides whether a side is cut. Paths are
  anchored to the repo.
* `from_language.py` — rules now read the training serializer's own words (`mid in the frame`,
  `mostly in frame`), `straight-on` is a bearing, "medium-wide" no longer matches "wide shot".
* `author.py` — `author_goal(nl=, image=, method=)` → `AuthoredGoal` (request, complete profile,
  prompt, per-key source, warnings, bank support). `scripts/author_goal.py` is the CLI
  (`--both` shows the two completers; `--json` writes the record `recon_from_reference` reads).

`GoalProfile.to_nl()` and `vocab.profile_to_nl` remain as deprecated aliases of the training
serializer, so older scripts print the right sentence.

## 4. Measured

### Completion on held-out goals (`scripts/bench_completion.py`, 1,162 goals that occur only in the 8 val scenes)

Word match of the *guessed* axes, bank | rules (%):

| request | elevation | placement_x | placement_y | body framing | crop |
|---|---|---|---|---|---|
| shot + bearing | 80.6 \| 85.7 | 46.4 \| 27.1 | 66.5 \| 56.8 | 49.1 \| 37.4 | 32.0 \| 27.8 |
| + elevation | — | 51.1 \| 27.1 | 60.5 \| 56.8 | 48.3 \| 37.4 | 35.2 \| 27.8 |
| + placement + crop | — | — | — | 80.1 \| 74.2 | — |
| image, cut side only | — | — | — | 88.6 \| 66.4 | 93.7 \| 81.6 |
| image, cut fractions | — | — | — | 94.7 \| 93.6 | 99.2 \| 99.2 |

Normalized MAE ×100 of the guessed numbers (bank | rules): for `shot + bearing`, body_in_frame
19.3 | 22.2, visible 14.6 | 19.3, centre-x 21.3 | 18.9; for `image, side only`, body_in_frame
**3.2 | 13.0**, visible **3.5 | 15.7**. Reading: from words alone both guess similarly (the words
simply do not say where the subject sits); once a photo pins the box, the bank's prior for
"legs vs waist given this box" is worth ~10 points of body_in_frame, and the extent fit closes
the rest.

### Module 2 on held-out renders (`scripts/bench_reference_completion.py`, 300 goal frames, 8 val scenes)

* Detected 209 / 300. The misses are small subjects (median occupancy 16 %), non-human assets
  (three snowmen), and frames whose subject is **occluded by scene geometry** — the projected
  bbox ignores occlusion, so e.g. `Modern-Warehouse__sitting-survivor/pair_00_frame_25.jpg`
  is a goal frame of a blank wall. That is a data issue upstream of this module.
* Geometry: occupancy MAE 3.0, shot-size word 84 %, centre MAE 13 / 9 px, placement words 97 / 95 %.
* Bearing (RF, still fitted on the pre-lookat075 tree): sector3 80 %, sector8 62 %, MAE 28°
  (front 33°, back 18°); 51 of 185 are off by one sector. Elevation MAE 9.7°, word 73 %.
* Crop side (box rule) 86.6 %. Body-extent fit available on 86 %: 5-class crop word 83.3 %,
  visible MAE 0.066 (median 0.037). Completed end to end: bank 80.9 % / visible 0.076 /
  body_in_frame MAE 8.9; rules 78.9 % / 0.084 / 9.0. Where the fit is unavailable the bank
  word is right 65.5 %, rules 51.7 %.
* Remaining crop confusion: "both" read as "above the head" (11 / 32) — the ankle veto cancels a
  real bottom cut when YOLO emits confident ankles for feet that are out of frame (the known
  trade from the v4 session).

### Tests

`tests/` — 144 pass in `.venv-analysis`; the seven framework-dependent modules pass in the
cosmos venv (`test_prompt_single_format`, `test_shoot_and_scene_split`,
`test_diagnostics_action_width`, plus the serializer contract): the move of `goal_prompt` did
not change a byte on the training side.

## 5. Not done here

* **Probe the policy's response** to authored prompts (teacher-forced, no Blender; the
  `shoot_probe.py` template): GT prompt vs bank-completed vs rule-completed on the same
  held-out windows. This decides whether completion is enough or a v6 with clause dropout is
  needed for true partial goals.
* Re-fit the bearing / elevation regressors on the lookat075 tree (feature extraction over
  thousands of renders → sbatch); sector8 62 % on held-out scenes says the old fit has slipped.
* An LLM environment for Module 1's hybrid classifier (`.venv-analysis` has no `transformers`).
* Resync `scripts/recon_from_reference.py` with the v5 rollout (shoot-stop, K samples) and
  run the recon on v5 with goals from `scripts/author_goal.py --json`.
* `scripts/viz_goal_profile.py` still carries its own copy of the person-box solver.
