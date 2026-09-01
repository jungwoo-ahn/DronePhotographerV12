#!/bin/bash
#SBATCH --job-name=goalrender
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=runs/goalrender_%j.out
#SBATCH --signal=B:SIGTERM@300
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# Render authored goals: NL -> goal -> pose -> Blender, and reference image -> goal -> the same
# composition in another scene with another subject. No policy involved; this grades the GOAL.
#
#   sbatch --qos=own scripts/sbatch_render_goals.sh --stage all --targets 6 --refs 8
#
# ~90 frames at 1024x768 / 32 samples: about 8 s each on GPU plus ~45 s per scene load, so
# well inside the 3 h limit. Every stage is resumable (geometry cached per placement, poses in
# solved.json, frames skipped when the jpg exists), which is what makes the self-requeue safe.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
cd "$V12"

# Blender ships its own libs here; the analysis venv supplies python/torch/ultralytics.
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$V12:${PYTHONPATH:-}"
export HF_HOME=/home/nas_main/.cache/huggingface
export YOLO_CONFIG_DIR="${TMPDIR:-/tmp}/ultralytics"

_args=("$@")
_handle_term() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        PREEMPTED*|*CANCELLED*by\ 0*)
            echo "[requeue] preempted (state=$state) — resubmitting; finished frames are kept" >&2
            sbatch -A share --qos=share "$0" "${_args[@]}" >&2 ;;
        *) echo "[requeue] state=$state — not a preemption, exiting" >&2 ;;
    esac
    exit 143
}
trap _handle_term SIGTERM

"$V12/.venv-analysis/bin/python" scripts/render_authored_goals.py "$@" &
wait $!
