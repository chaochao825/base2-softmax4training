#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# WORLD_SIZE=3 RANK=0 MASTER_ADDR=server1 MASTER_PORT=23456 LOCAL_RANK=0 \
#   torchrun --nproc_per_node=1 --nnodes=3 --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
#   scripts/train.py --dataset CIFAR-10 --softmax base2 --epochs 50 --wandb

echo "Launching DDP training with torchrun..."
exec torchrun \
  --nproc_per_node=${NPROC_PER_NODE:-1} \
  --nnodes=${NNODES:-1} \
  --node_rank=${NODE_RANK:-0} \
  --master_addr=${MASTER_ADDR:-127.0.0.1} \
  --master_port=${MASTER_PORT:-23456} \
  scripts/train.py "$@"


