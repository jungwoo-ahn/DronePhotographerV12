# Handoff — help run the v12 yaw re-render

**For:** anyone in group `lab` with spare B200 quota (written for @jooyeolyun)
**Contact:** jungwooahn
**Status when written:** 62 / 300 placements done, **238 remaining**

You can help by submitting **one sbatch array under your own account**. Nothing else is
needed — no code changes, no coordination messages, no dividing the list up between us.
The job already coordinates itself; §3 explains how, and §5 is the one thing you must not
skip.

---

## 1. What this is

DronePhotographer trains a goal-conditioned camera policy: given the current view and a
requested shot ("a medium shot of the subject from the front, at eye level"), predict the
camera motion that gets there. Training data is rendered in Blender.

The problem this re-render fixes: in the existing data the camera almost always ends up
**behind** the subject. Measured over 400 placements, well-framed goals break down as

| back | back-left | right | back-right | front-right | front | front-left | left |
|---|---|---|---|---|---|---|---|
| 31% | 26% | 13% | 12% | 10% | **4%** | 3% | **1%** |

~69% of goals are seen from behind, and *front* — the most useful shot in photography — is
the rarest. That is baked into how the data was generated: every placement used identity
rotation, so all 100 subjects face the same world direction while the valid camera anchors
sit elsewhere.

The fix is cheap because the view angle relative to the subject is

```
bearing = (front_az + yaw) − azimuth
```

so **spinning the subject** shifts a whole placement's bearings rigidly. We re-render the
same trajectories, same cameras, same scenes, with one number changed — the object's yaw —
chosen per placement to land its goals in an under-represented sector. `runs/yaw_plan.json`
holds that number for each of the 300 placements.

Verified on a pilot: rendering at yaw 0 reproduces the recorded subject centre to 2 mm, a
231° yaw moves the subject 1.2 cm (a spin, not an orbit), and that placement's goals went
from *back-right 7 / front-right 2 / back 1* to **front 6 / front-right 2 / left 2**.

---

## 2. What to take, and what not to

| Work | Take it? | Why |
|---|---|---|
| **Yaw re-render** (this doc) | ✅ **yes** | embarrassingly parallel, self-coordinating, no shared state beyond a claim file |
| Cosmos policy training | ❌ no | a single run with checkpoint/optimizer state; a second writer would corrupt it |
| Data export / analysis | ❌ no | fast, and it must run *after* the re-render finishes |

---

## 3. Why our jobs will not collide

Everything writes into one shared output directory, and three mechanisms keep that safe:

1. **Atomic claim.** Before touching a placement a worker creates `claim.lock` with
   `O_CREAT|O_EXCL`. Exactly one worker can win; everyone else moves on. `done.flag` alone
   would not be enough — it only appears when a placement *finishes*, so it cannot prevent
   two workers starting the same one.
2. **Heartbeat + staleness.** A live worker touches its claim after every pair. A claim
   untouched for 30 min is treated as abandoned (its task was preempted) and taken over, so
   a killed job does not park a placement forever.
3. **Frame-level resume.** Rendered `.jpg`s survive a killed task; only metadata is lost.
   A restarted worker skips frames already on disk. Preemption costs one frame (~4 s), not
   a placement (~19 min). This matters a lot here — most tasks *are* preempted.

So you do not need to be given a slice. Submit the array and your workers will pick up
whatever is unclaimed.

---

## 4. Submit

```bash
cd /home/nas_main/jungwooahn/projects/DronePhotographerV12

umask 002                                  # REQUIRED — see §5
sbatch --qos=own scripts/sbatch_rerender_yaw.sh
# or, to use spare capacity instead of your own quota:
# sbatch -A share --qos=share scripts/sbatch_rerender_yaw.sh
```

That is the whole thing. The script is a 16-task array, 1 GPU each, 6 h wall-clock, and it
already points at the shared paths (Blender binary, scene assets, yaw plan, output dir).

Resubmit the same command whenever tasks get preempted — `--resume` is already on, so a
resubmission continues rather than restarts.

**Tuning:** raise `--array` in the sbatch header if you have more quota than 16 GPUs. Do not
change `--out-dir` or `--yaw-plan-path`; the shared output dir is what makes the
coordination work.

---

## 5. The one thing you must not skip: `umask 002`

Files you create must stay **group-writable**. Our workers may need to take over a stale
claim that your (preempted) task left behind, and a claim created under the default
`umask 022` is not writable by us — the placement would sit blocked until you happen to
re-run it. The same applies to placement directories: if yours is `755`, our worker cannot
create `renders/` inside it.

The renderer now skips such a placement instead of crashing, but the work still stalls.
`umask 002` before `sbatch` avoids the whole class of problem.

If you use a shell rc file, `umask 002` there is the durable fix.

---

## 6. Checking progress

```bash
cd /home/nas_main/jungwooahn/projects/DronePhotographerV12
echo "done:   $(find runs/rerender_yaw -name done.flag | wc -l) / 300"
echo "frames: $(find runs/rerender_yaw -name '*.jpg' | wc -l)"
squeue -u $USER -o "%.14i %.10j %.9T %.10M %R"
```

A finished placement contains `data.json`, `done.flag`, and `renders/*.jpg`. `done.flag`
records how many frames were rendered versus resumed, e.g.
`rendered=0 resumed=192` means everything was already on disk.

---

## 7. Known noise, not bugs

- `Error: Unable to pack file … /g/products/…` and `/c/Users/…/render_ext/*.tif` — the
  scene `.blend`s carry the original artists' Windows paths and output nodes. Harmless, and
  present in the original v7 renders too.
- `could not get a list of mounted file-systems`, locale warnings — Blender on this image.
- A task exiting with placements left over is normal; it was preempted. Resubmit.

## 8. If something looks wrong

Ping jungwooahn. Useful to include: the job id, the placement name, and the tail of
`runs/reyaw_<jobid>_<task>.out`.

Do **not** delete anything under `runs/rerender_yaw/` to "clean up" — a partially rendered
placement is useful work that a later task will finish.
