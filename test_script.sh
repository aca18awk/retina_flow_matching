#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist luna
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
conda activate flow-env

export NCCL_P2P_LEVEL=LOC
export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/outputs"


# python CM_real.py
# python CM_generation.py
python CM_LPIPS.py
# # python CM_train.py
# # python CM_test_colour.py
# # python CM_train_multiple_gpus.py

# # python Eyepacs_calculate_distances.py
# python Eyepacs_train_multiple_gpus.py
