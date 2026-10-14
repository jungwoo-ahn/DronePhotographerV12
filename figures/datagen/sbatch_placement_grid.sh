#!/bin/bash
#SBATCH --job-name=placegrid
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/home/nas_main/jungwooahn/projects/DronePhotographerV12/runs/placegrid_%j.out
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# CPU-only: re-run the placement floor-candidate search on Chill-Camping with Knit-and-Denim and dump every cell,
# plus a top-down and a scene-camera render, for the data-gen figure (panel b, "real grid" variant).
#   sbatch figures/datagen/sbatch_placement_grid.sh
set -euo pipefail
umask 002
OLD=/home/nas_main/jungwooahn/projects/DronePhotographer
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/dump_placement_grid.py" ] || HERE=/home/nas_main/jungwooahn/projects/DronePhotographerV12/.claude/worktrees/fig-overview/figures/datagen
export LD_LIBRARY_PATH="$OLD/blender/syslibs/lib:${LD_LIBRARY_PATH:-}"
cd "$OLD"
"$OLD/blender/blender" -b data/scenes/Chill-Camping_150d7263/Chill-Camping_150d7263.blend -t 16 \
    -P "$HERE/dump_placement_grid.py" -- \
    --object_file data/objects/Knit-and-Denim_82dd220b --scene_scale 2.0 \
    --out /home/nas_main/jungwooahn/projects/DronePhotographerV12/runs/placement_grid/Chill-Camping__Knit-and-Denim
