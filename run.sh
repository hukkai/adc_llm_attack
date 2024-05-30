#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth2
export NCCL_IB_DISABLE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p gpu --gres=gpu:8 --ntasks=64 --ntasks-per-node=8 \
    --cpus-per-task=16 --kill-on-bad-exit=1 --async \
    python3 -u single_attack.py ( ... train script args...)

