#!/bin/bash
#SBATCH --job-name=export
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=runs/export_%j.out
# CPU-only: enumerates every placement's windows and encodes the episode frames to mp4.
# No GPU requested — scoring is already done and this stage never touches the model.
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"
exec "$V12/repos/cosmos-framework/.venv/bin/python" scripts/export_lerobot.py "$@"
