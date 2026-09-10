#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist lora
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
conda activate flow-env

# Optimizations for NCCL (DDP Backend)
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_DISABLE=1  
# export OMP_NUM_THREADS=4  

# torchrun --standalone --nproc_per_node=8 Eyepacs_train_multiple_gpus.py


# python compute_reference_knn.py
python visualise_messidor2_umap.py
# python compute_lpips.py