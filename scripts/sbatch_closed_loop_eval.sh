#!/bin/bash
#SBATCH --job-name=cleval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=runs/cleval_%j.out
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec "$CF/.venv/bin/python" scripts/closed_loop_eval.py "$@"
