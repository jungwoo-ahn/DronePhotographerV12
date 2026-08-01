#!/bin/bash
# Launch the Cosmos3-Nano camera-policy SFT (configs/policy/action_policy_camera_nano.py).
#
#   bash scripts/train_camera_policy.sh                       # smoke: max_iter from the recipe (100)
#   bash scripts/train_camera_policy.sh trainer.max_iter=2000 # any Hydra override
#
# Env (all have defaults):
#   CAMERA_ROOT            LeRobot dir from scripts/export_lerobot.py
#   BASE_CHECKPOINT_PATH   Cosmos3-Nano DCP dir (convert_model_to_dcp)
#   WAN_VAE_PATH           Wan2.2_VAE.pth
#   NPROC_PER_NODE         GPUs to use
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
CF="$V12/repos/cosmos-framework"
VENV="$CF/.venv"
SP="$VENV/lib/python3.13/site-packages"

: "${CAMERA_ROOT:=$V12/runs/lerobot_v1}"
: "${BASE_CHECKPOINT_PATH:=$CF/examples/checkpoints/Cosmos3-Nano}"
: "${WAN_VAE_PATH:=$CF/examples/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
: "${IMAGINAIRE_OUTPUT_ROOT:=$V12/runs/train}"
: "${COSMOS_TB_DIR:=$IMAGINAIRE_OUTPUT_ROOT/tb}"
: "${NPROC_PER_NODE:=1}"
export CAMERA_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT COSMOS_TB_DIR
export HF_HOME=/home/nas_main/.cache/huggingface

# Vendored-checkout patches (cuda_cudart symlink + experiment registration). Idempotent.
bash "$V12/scripts/setup_cosmos_env.sh"

# The CUDA libs ship inside the venv but are not on the loader path.
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -maxdepth 2 -name lib -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# v12 must be importable: the recipe lives at configs/policy/ and imports
# src.data.cosmos_camera_dataset.
export PYTHONPATH="$V12:$CF:${PYTHONPATH:-}"

: "${SFT_TOML:=$V12/configs/policy/camera_policy_nano.toml}"
[ -f "$CAMERA_ROOT/meta/info.json" ] || {
    echo "ERROR: CAMERA_ROOT has no meta/info.json ($CAMERA_ROOT). Run scripts/export_lerobot.py first." >&2
    exit 1
}
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
echo "CAMERA_ROOT=$CAMERA_ROOT"
echo "BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
echo "SFT_TOML=$SFT_TOML  NPROC_PER_NODE=$NPROC_PER_NODE"
echo "TensorBoard: tensorboard --logdir $COSMOS_TB_DIR"

cd "$CF"
exec "$VENV/bin/torchrun" --nproc_per_node="$NPROC_PER_NODE" \
    -m src.train.run_with_tb --sft-toml "$SFT_TOML" "$@"
