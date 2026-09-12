# Retraining `camera_policy_nano_v6` — 100k iterations on 8 GPUs

## Why this run exists

v5 stopped at 40k because its LR schedule ended there, not because it converged. Continuing it
at `f_max=0.9` (4.5e-5) made it worse — held-out action loss went from **0.00073 at iteration
40000** to 0.0011–0.0018 over 46k–54k, with train moving the same way and the val–train gap
unchanged, which is an LR too large for the basin rather than overfitting. `iter_000040000`
was then deleted by a retention rule that had no way to know it mattered, and nothing better
survives it. So: one clean cycle, from the base, at the LR that is known to work.

## What changed from the run that produced the good 40k model

Only two things, deliberately:

| | v5 (40k) | v6 (100k) |
|---|---|---|
| `cycle_lengths` / `max_iter` | 40000 | **100000** |
| `warm_up_steps` | 100 | **500** (global batch is 4× larger) |
| `lr` | 5.0e-05 | 5.0e-05 — unchanged |
| GPUs / global batch | 2 / 2 | **8 / 8** |
| samples seen | ~200k | ~800k |

**8 GPUs do not make this faster.** The recipe is FSDP data-parallel at `batch_size=1` per
rank, so more GPUs multiply the global batch, not the iteration rate. 100k is ~7 days either
way; what 8 GPUs buy is 4× the data per step.

## Running it

The job needs `core-own` (8 GPU), so it is submitted from **jooyeol's** account. `jooyeolyun`
is in group `lab` and `runs/`, `runs/train/` are `drwxrwxr-x`, so it reads this repo, its venv,
`runs/lerobot_v5` and the base checkpoint, and writes checkpoints here — nothing is copied.

```bash
ssh davian-b200_jooyeol

# 1) smoke: 100 iterations, 8 GPUs (~15 min). The only thing that exercises FSDP at shard
#    degree 8 -- config load, dataset, sharding and the save path, which is where an 8-GPU
#    run actually fails.
sbatch --qos=own --export=ALL,NO_CHAIN=1 \
    /home/nas_main/jungwooahn/projects/DronePhotographerV12/scripts/sbatch_train_camera.sh \
    --sft-toml configs/policy/camera_policy_nano_100k.toml \
    job.name=camera_policy_nano_v6_smoke trainer.max_iter=100

#    NO_CHAIN=1 is required for anything that is not the real run: the successor is queued
#    BEFORE the first training step, so `banked >= max_iter` cannot suppress it at submission
#    time and a smoke would otherwise chain forever.
#
#    It writes ONE 137 GB checkpoint: the trainer saves at max_iter whatever save_iter says
#    (the v5 smoke did the same at save_iter=1000). Delete the directory when it passes:
#    rm -rf runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v6_smoke

# 2) the real run. The chain queues its own successor at start, so this is the only submit.
sbatch --qos=own /home/nas_main/jungwooahn/projects/DronePhotographerV12/scripts/sbatch_train_camera.sh \
    --sft-toml configs/policy/camera_policy_nano_100k.toml
```

`--gres`, `--mem`, `--cpus-per-gpu` and `--time` come from the script's `#SBATCH` defaults
(8 GPU / 768G / 8 CPU-per-GPU / 3 days) and are **propagated to every successor link** — pass
them on the command line only if you change the shape, and change `--gres` and `--mem`
together (admission caps a job at 244000 MiB × num_gpus).

The cluster often reads 120/120 GPUs used, but ~42 of core's 72 are held by preemptible
`extra` jobs, so an `own` job takes its 8 within the ~5-minute preemption grace.

## Checking on it

```bash
sjob <jobid>                 # 8 GPUs busy, no diagnosis lines = healthy
sjob -l <jobid>              # + last 30 lines of stdout
tail -f runs/camtrain_<jobid>.out
```

What to verify, and when:

| when | what |
|---|---|
| link 1, first log lines | `run: camera_policy_nano_v6`, `GPUs: 8`, `banked: 0 / 100000` |
| ~iteration 1000 (~2 h) | `checkpoints/iter_000001000` exists; record the measured iter/h — the 7-day estimate is extrapolated from 585 iter/h at 2 GPUs and has not been measured at 8 |
| ~iteration 6000 (~10 h) | first prune: `[ckpt_retention]` names the **v6** directory and the v5 directory is untouched |
| link 2 starts | `GPUs: 8`, not 2 — the successor inherits the allocation |

## Checkpoints kept

`keep_every=10000`, `keep_last=5`, `keep_best=5` (by held-out action loss) — at most 20 ×
137 GB = **2.7 TB**. `save_iter=1000` must stay a multiple of `HeldOutLossProbe(every_n=500)`:
retention credits a checkpoint with a metric only when the probe published at that exact
iteration, so a save cadence that misses the probe silently disables `keep_best`.

**To protect a checkpoint permanently**, put a `KEEP` file in it:

```bash
echo "why this one matters" > runs/train/.../checkpoints/iter_000040000/KEEP
```

Retention never deletes a directory containing one. Do this for any checkpoint you publish
numbers from — that is the kind of importance a loss curve cannot express, and its absence is
what cost `iter_000040000`. `camera_policy_nano_v5/checkpoints/iter_000049000` is already
pinned; it is the only trained model that still exists and the fallback if v6 disappoints.

## Stopping it

```bash
touch runs/train/.camtrain_stop_camera_policy_nano_v6   # every later link exits immediately
scancel <jobid>                                          # and the running one
```

The stop file is per-run, so this cannot affect any other chain.

## A note on trailing overrides

`job.name=`, `trainer.max_iter=` and friends on the command line **win over the TOML**
(`cosmos_framework/scripts/train.py`: "Trailing key.path=value positionals are applied last").
The chain reads them the same way the trainer does, via `_setting <hydra.path> <toml key>` —
it has to, because deriving the run name from the TOML alone puts the trainer in one directory
and the chain's bookkeeping in another. That mismatch is what made a smoke queue itself a
successor forever, and it is the same shape as the retention bug that cleaned an empty smoke
directory for weeks while the real run grew to 1.9 TB. If you add a new override that changes
where a run writes, teach `_setting` about it.
