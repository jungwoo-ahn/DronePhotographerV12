#!/bin/bash
#SBATCH --job-name=reyaw
#SBATCH --array=0-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/reyaw_%A_%a.out
# Re-render planned placements with the subject yawed so its well-framed goals land in
# under-represented view sectors (front / front-left / left). Everything else -- scene,
# placement position/scale, the stored camera poses, render settings -- is identical to
# the original v7 run, so the new frames stay in-distribution.
#
# The array slices runs/yaw_plan.json; --resume skips placements that already have a
# done.flag, so a preempted task (qos=extra) just needs resubmitting.
#
#   sbatch --qos=extra scripts/sbatch_rerender_yaw.sh
set -euo pipefail
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
cd "$SHARED"

"$SHARED/blender/blender" -b -P scripts/v7_rerender_yaw.py -- \
    --stage1-dir data/trajectories \
    --placements-v6-dir data/vlm_object_placing_v6_260428_061326 \
    --assets-root "$SHARED" \
    --out-dir "$V12/runs/rerender_yaw" \
    --yaw-plan-path "$V12/runs/yaw_plan.json" \
    --slice-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --slice-count "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --gpu-index 0 \
    --resume \
    "$@"
