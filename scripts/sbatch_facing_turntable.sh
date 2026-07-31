#!/bin/bash
#SBATCH --job-name=facing_tt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/facing_tt_%j.out
# Full facing turntable render (all 102 assets) on a worker GPU.
# Cycles + GPU; ~12 views x 512px each. Idempotent-ish: pass --skip-existing on a resubmit
# after preemption to skip already-rendered assets. Extra params after `--` override defaults.
set -euo pipefail
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
cd "$V12"
"$SHARED/blender/blender" --background \
    --python scripts/render_facing_turntable.py -- \
    --views 12 --res 512 --samples 24 --engine CYCLES --gpu "$@"
