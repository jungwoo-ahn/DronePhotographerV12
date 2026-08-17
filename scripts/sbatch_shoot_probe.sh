#!/bin/bash
#SBATCH --job-name=shootprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/shootprobe_%j.out
# Needs a C compiler for torch inductor (worker pods are driver-only), same as training.
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# What does the shoot channel predict, and is it reading the frame or the idle_frame leak?
# No Blender and no rollout, so this is minutes per hundred windows -- and it has to run
# BEFORE the closed-loop sweeps, because it decides the threshold they use and whether their
# termination numbers mean anything.
#
#   sbatch --qos=own scripts/sbatch_shoot_probe.sh \
#       --checkpoint runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v5/checkpoints/iter_000040000 \
#       --n 300 --n-sweep 100 --out runs/eval/shoot_probe_iter40000.json
#
# `own` rather than `extra`: it is a one-hour single-GPU job and the free-group own quota is
# idle now that training finished, so there is no reason to take a preemption risk.
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

exec "$CF/.venv/bin/python" scripts/shoot_probe.py "$@"
