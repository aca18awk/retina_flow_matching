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
# python CM_generation_avg.py
# python CM_LPIPS.py
# # python CM_train.py
# python CM_test_colour.py
# # python CM_train_multiple_gpus.py

# # python Eyepacs_calculate_distances.py
# python Eyepacs_train_multiple_gpus.py
# python Eyepacs_experiment_guidance_scale.py
# python ditances_same_label.py
# python Eyepacs_assess_feature_embeddings_2.py
# python Eyepacs_feature_extraction_retFound.py



# python Messidor_tsne.py
# python Messidor_save_dataset.py
# python Messidor_real.py
# python distances.py
# python Messidor_real_data_for_comparison.py
# python Messidor_FID.py
# python messidor_feature_test.py
# python Messidr_weights_classifier.py
# python Messidor_test_dino.py
# python Messidor_classifier_on_dino_binary.py
# python Eyepacs_dist_ImageNet_normalised.py
# python distance_RetFound_DinoV2.py
# python distance_RetFound_MAE.py
# python distance_ImageNet_normalised.py

# python Eyepacs_experiment_guidance_scale.py
# python plot_datasets.py


python Eyepacs_generation.py
# python distance_generate_ALL_embeddings.py

# python assess_features.py
# python Eyepacs_assess_feature_embeddings.py
# python Messidor_assess_feature_embeddings.py