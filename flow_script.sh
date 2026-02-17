#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist luna
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
# conda activate diffusion
conda activate flow-env

export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/outputs"

# inception_v3.tv_in1k
# efficientnet_b0 
# python conditional_retina.py
# python conditional_retina_CFG.py
# python MNIST_dataset.py
# python MNIST_embedding_flow.py
# python test_generation_col_mnist.py
python check_colour.py


# python test_generate_img_latent_embed.py
# python generate_images.py
# python test_generation.py
# python conditional_mnist.py