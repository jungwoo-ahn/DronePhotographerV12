#!/bin/bash
#SBATCH --job-name=camtrain
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=runs/camtrain_%j.out
# Cosmos3-Nano camera-policy SFT.
#
# Single-GPU does NOT fit: with data_parallel_shard_degree=1 FSDP shards nothing, so the
# full parameters plus FusedAdam's fp32 master weights and moments sit on one device and
# the optimizer step OOMs a 178 GiB B200 (175 GiB already allocated). Shard across the
# GPUs instead — the reference LIBERO recipe runs HSDP 2x8 for the same reason.
#
#   sbatch --qos=own scripts/sbatch_train_camera.sh                       # 2 GPUs, smoke
#   sbatch --gres=gpu:8 --qos=extra scripts/sbatch_train_camera.sh \
#       trainer.max_iter=20000                                            # real run
set -euo pipefail
V12=/home/nas_main/jungwooahn/projects/DronePhotographerV12
cd "$V12"

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
export NPROC_PER_NODE="$NGPU"
export PYTORCH_ALLOC_CONF=expandable_segments:True   # optimizer step is the fragmentation peak

echo "GPUs: $NGPU  (sharding optimizer state across all of them)"
# NOTE the path: the TOML's [model.parallelism] table lands at model.CONFIG.parallelism.*,
# so a hydra tail override must use that longer path — `model.parallelism.…` raises
# "Key 'parallelism' is not in struct".
exec bash scripts/train_camera_policy.sh \
    model.config.parallelism.data_parallel_shard_degree="$NGPU" \
    model.config.parallelism.data_parallel_replicate_degree=1 \
    "$@"
