#!/bin/bash
# Launcher for the facing turntable render (login smoke / generic).
# Blender is the shared repo's binary; its missing system libs (libXfixes/libGL/libxkbcommon)
# are supplied from blender/syslibs via LD_LIBRARY_PATH (same as sbatch_rollout_eval.sh).
#
# Smoke (login, cheap, CPU):   bash scripts/run_facing_turntable.sh --max-assets 2 --views 8 --res 384 --samples 8
# Full set goes through scripts/sbatch_facing_turntable.sh (GPU).
set -euo pipefail
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
cd "$V12"
exec "$SHARED/blender/blender" --background \
    --python scripts/render_facing_turntable.py -- "$@"
