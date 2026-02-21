#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:8
#SBATCH --nodelist loki
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
conda activate flow-env

# Optimizations for NCCL (DDP Backend)
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1  # Often helps if Infiniband config is tricky
export OMP_NUM_THREADS=4  # Prevent CPU thread contention

# --nproc_per_node=4 : Spawns 3 processes (one per GPU)
# --rdzv_backend=c10d : Use PyTorch C10d backend for coordination
# --standalone : Tells torchrun we are on a single node (loki)

torchrun --standalone --nproc_per_node=8 Eyepacs_train_multiple_gpus.py