#!/bin/bash
#SBATCH --job-name=imagineprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=runs/imagineprobe_%j.out
# Needs a C compiler for torch inductor (worker pods are driver-only), same as training.
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# Decode what the policy IMAGINES. `mode="policy"` predicts video as well as actions, so
# every chunk already carries eight generated frames that `predict_chunk` was discarding.
# Needs the VAE (WAN_VAE_PATH) to turn the returned latents into pixels.
#
#   sbatch -A share --qos=share scripts/sbatch_imagine_probe.sh \
#       --checkpoint runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v5/checkpoints/iter_000040000 \
#       --n 120 --out runs/eval/imagine_probe.json
#
# `share`, not own/extra: this account's GPU pool is fully consumed by its other jobs, so
# both of those sit in AssocGrpGRES forever. Preemption is handled by the trap below.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
CF="$V12/repos/cosmos-framework"
SP="$CF/.venv/lib/python3.13/site-packages"
cd "$V12"

bash "$V12/scripts/setup_cosmos_env.sh"      # cuda_cudart symlink + experiment registration

export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$V12:$CF:${PYTHONPATH:-}"
export HF_HOME=/home/nas_main/.cache/huggingface
export WAN_VAE_PATH="$CF/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
# The recipe interpolates ${oc.env:CAMERA_ROOT} when the config is built. Inference never
# reads the dataset, but the interpolation still has to resolve or the config raises before
# the model is even constructed -- and it must point at the v5 export, whose action width is
# 10, or the config would describe a different model than the checkpoint.
export CAMERA_ROOT="${CAMERA_ROOT:-$V12/runs/lerobot_v5}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$V12/runs/train}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$CF/examples/checkpoints/Cosmos3-Nano}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Self-requeue on preemption. `share` is evicted the moment another account wants the GPU,
# and without this the job simply dies: the last attempt was preempted at 13 min and nothing
# resubmitted it, so the probe sat unfinished for a day. Safe because --resume skips windows
# already in the output file.
#   - `exec` would replace this shell and destroy the trap  -> run python in background
#   - a FOREGROUND child defers the handler until it exits  -> `&` + `wait`
#   - `scancel` raises the same SIGTERM as a preemption      -> check sacct State first
_args=("$@")
_handle_term() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        PREEMPTED*|*CANCELLED*by\ 0*)
            echo "[requeue] preempted (state=$state) — resubmitting to finish the rest" >&2
            sbatch -A share --qos=share "$0" "${_args[@]}" >&2
            ;;
        *) echo "[requeue] state=$state — not a preemption, exiting" >&2 ;;
    esac
    exit 143
}
trap _handle_term SIGTERM

"$CF/.venv/bin/python" scripts/imagine_probe.py "$@" &
wait $!
