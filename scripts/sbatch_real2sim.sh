#!/bin/bash
#SBATCH --job-name=r2sim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
# 3h was sized for 12-24 episodes. A 160-episode sweep runs ~8h at ~3 min each, and
# silently TIMED OUT at 63/160. Resume makes a restart cheap, but the limit should
# not be the thing that decides how many episodes finish.
#SBATCH --time=06:00:00
#SBATCH --output=runs/r2sim_%j.out
# Needs a C compiler for torch inductor (worker pods are driver-only), same as training.
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# Closed-loop evaluation of a trained camera policy.
#
#   sbatch --qos=extra scripts/sbatch_closed_loop_eval.sh \
#       --checkpoint runs/train/.../checkpoints/iter_000005000 \
#       --episodes 24 --out runs/closed_loop/iter5000.json
#
# Inference is single-GPU regardless of the training sharding: DCP reshards on load,
# so a 2-GPU-sharded checkpoint loads onto one device with no conversion step.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
CF="$V12/repos/cosmos-framework"
SP="$CF/.venv/lib/python3.13/site-packages"
cd "$V12"

bash "$V12/scripts/setup_cosmos_env.sh"      # cuda_cudart symlink + experiment registration

export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# Blender's own missing system libs, for the rollout renderer.
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$V12:$CF:${PYTHONPATH:-}"
export HF_HOME=/home/nas_main/.cache/huggingface
export WAN_VAE_PATH="$CF/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
# The recipe interpolates ${oc.env:CAMERA_ROOT} when the config is built. Inference
# never reads the dataset, but the interpolation still has to resolve or the config
# raises before the model is even constructed.
export CAMERA_ROOT="${CAMERA_ROOT:-$V12/runs/lerobot_v1}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$V12/runs/train}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$CF/examples/checkpoints/Cosmos3-Nano}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Self-requeue on preemption. This eval takes ~4.5 h and `share` has been cutting it at
# ~30 min, so without this it never finishes. Safe because closed_loop_eval resumes from
# <out>.partial.json — each attempt continues rather than restarting.
#
# Three pitfalls the cluster guide warns about, all handled:
#   - `exec` would replace this shell and destroy the trap  -> run python in background
#   - a FOREGROUND child defers the handler until it exits  -> `&` + `wait`
#   - `scancel` raises the same SIGTERM as a preemption      -> check sacct State first
_args=("$@")
_handle_term() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        PREEMPTED*|*CANCELLED*by\ 0*)
            echo "[requeue] preempted (state=$state) — resubmitting to finish the rest" >&2
            sbatch --qos=extra "$0" "${_args[@]}" >&2
            ;;
        *) echo "[requeue] state=$state — not a preemption, exiting" >&2 ;;
    esac
    exit 143
}
trap _handle_term SIGTERM

"$CF/.venv/bin/python" scripts/real2sim_rollout.py "$@" &
wait $!
