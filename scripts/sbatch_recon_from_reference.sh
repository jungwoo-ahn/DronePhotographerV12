#!/bin/bash
#SBATCH --job-name=reconref
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/reconref_%j.out
#SBATCH --signal=B:SIGTERM@300
#SBATCH --container=nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04
#
# TRUE recon: a reference photo's composition, re-shot by the policy in a DIFFERENT scene.
# Same environment as the closed-loop eval (16B policy + Blender rollout renderer).
#
#   sbatch --qos=extra scripts/sbatch_recon_from_reference.sh \
#       --checkpoint runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v2/checkpoints/iter_000006000 \
#       --cases 6 --out runs/recon_ref/recon.json
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
SHARED=/home/nas_main/jungwooahn/projects/DronePhotographer
CF="$V12/repos/cosmos-framework"
SP="$CF/.venv/lib/python3.13/site-packages"
cd "$V12"

bash "$V12/scripts/setup_cosmos_env.sh"

export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SHARED/blender/syslibs/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$V12:$CF:${PYTHONPATH:-}"
export HF_HOME=/home/nas_main/.cache/huggingface
export WAN_VAE_PATH="$CF/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth"
export CAMERA_ROOT="${CAMERA_ROOT:-$V12/runs/lerobot_v1}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$V12/runs/train}"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$CF/examples/checkpoints/Cosmos3-Nano}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Deliberately NOT adding .venv-analysis to PYTHONPATH: its torch would shadow the
# cosmos one (cuDNN 9.10 vs 9.20 -> hard failure at import). Module 2 runs in its own
# interpreter in stage 1/3; this stage only needs the policy + Blender.

# share QOS is preempted often; resubmit ourselves so the run finishes across several windows.
# --resume makes each restart pick up where the last one stopped.
_handle_preempt() {
    state=$(sacct -j "$SLURM_JOB_ID" -X -n -P -o State | head -1)
    if [[ "$state" == "PREEMPTED" || "$state" == *"CANCELLED"* ]]; then
        sbatch -A share --qos=share "$0" "$@"
    fi
    exit 143
}
trap '_handle_preempt "$@"' SIGTERM

"$CF/.venv/bin/python" scripts/recon_from_reference.py "$@" &
wait $!
