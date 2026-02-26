# Few-Shot Flow Matching with Stochastic Barycentric Sampling for Image Synthesis

> **Anonymous Submission for MICCAI 2026**
> Code repository for the paper: *"Few-Shot Flow Matching with Stochastic
Barycentric Sampling for Image Synthesis"*

<p align="center">
<img src="assets/diagram.png" width="600px">

This repository contains the PyTorch implementation of **Few-Shot Flow Matching (FSFM)**. FSFM is a generative adaptation framework that bridges extreme data-scarce regimes in medical imaging. By leveraging a frozen, label-free pre-trained prior and Stochastic Barycentric Sampling, FSFM adapts to novel local target domains using as few as 50 anchor images.

---

## ⚙️ Environment Setup & Dependencies

Our flow matching implementation is built upon the foundational work provided in the [conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) repository. Please ensure you follow their base environment guidelines if you encounter any OS-specific compilation issues.

To ensure reproducibility, we provide a `requirements.txt` mapping our working environment.

```bash
# Create and activate the conda environment
conda create -n fsfm_env python=3.10
conda activate fsfm_env

# Install dependencies from the requirements file
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

```

---

## 🔴🟢🔵 Colored MNIST (CM)

### Code Structure & Execution

**1. Data & Utilities**

* `CM_class.py`: Defines the dataset class for Coloured MNIST.
* `CM_real.py`: Extracts images from the validation subset and saves them in the same format as generated images to serve as upper-bound comparison data.

**2. Training**
To train the Coloured MNIST model, execute the main training script. The dataset will download automatically.

* `CM_train.py` / `CM_train_multiple_gpus.py`: Main training loops.
* `CM_get_fsfm_condition.py`: Takes a batch of images, passes them through a ResNet encoder, and dynamically calculates nearest neighbors on-the-fly during training.
* `CM_generate_images.py`: Utility for visualising generations during the training phase.

**3. Generation & Sampling**

* `CM_generation_barycentric.py`: Generates images using our proposed Stochastic Barycentric Sampling.
* `CM_generation_avg.py`: Generates images using standard average sampling (baseline).
* `CM_experiment_guidance_scale.py`: Generates image grids across different guidance scales for ablation studies.

**4. Evaluation**

* `CM_test_colour.py`: Calculates colour accuracy metrics.
* `calculate_LPIPS.py`: Computes structural fidelity/diversity metrics.

---

## 👁️ Retina

### 1. Dataset Preparation

Before running the scripts, ensure your source and target datasets are organized in the root directory. Please download [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) and [MESSIDOR-2](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess/data), and update the paths within the dataset classes.

* `Eyepacs_class.py`: Defines the EyePACS dataset class.
* `Messidor_class.py`: Defines the Messidor dataset class.
* `Messidor_save_dataset.py`: Automatically splits the dataset given the path to the original dataset and target indices.
* `hospital_B_file_list.csv`: Contains the specific, pre-selected indices for our 50-shot target domain. We ensured the selected indices are representative of the dataset and contain equal splits of each disease class.

Please structure your data as follows:

```text
data/
  ├── eyepacs/  
  │     ├── train/
  │     └── validation/
  └── messidor2/ 
        ├── hospital_b/
        ├── validation/
        ├── hidden_classifier_data/
        └── test/

```

### 2. Generating Feature Embeddings

The quality of the k-NN simplex heavily depends on the frozen feature extractor. Extract feature embeddings from the source dataset (EyePACS) and the target dataset by executing:

```bash
python distance_generate_ALL_embeddings.py

```

* `distance_generate_ALL_embeddings.py`: Master script that utilizes `distance_ImageNet.py`, `distance_RetFound_DinoV2.py`, and `distance_RetFound_MAE.py` to generate embeddings. Outputs `.pt` files required for k-NN retrieval.
* `Eyepacs_class_for_distance.py`: A simplified dataloader utilized strictly for rapid embedding calculations.
* `assess_features.py`: Trains a linear classifier on the generated embeddings to assess latent space robustness.

### 3. Phase 1: Label-Free Pre-Training

Train the continuous vector field entirely on unlabelled source data.

```bash
# Execute training across multiple GPUs
python Eyepacs_train_multiple_gpus.py

# Alternatively, use the provided shell script for cluster submission:
bash flow_script.sh

```

* `Eyepacs_generate_images.py`: Visualisation utility for monitoring the vector field during pre-training.

### 4. Phase 2: Few-Shot Generation (Barycentric Adaptation)

Using only 50 labeled anchor images from the target domain, we generate diverse synthetic samples via Stochastic Barycentric Sampling.

<p align="center">
<img src="assets/retina_neighbors_reference.png" width="600px">




<i>Anchor reference compared to its nearest neighbors in the learned manifold.</i>
</p>

To find the optimal guidance scale (w) for your target domain, execute:

```bash
python Eyepacs_experiment_guidance_scale.py 

```

Once optimal parameters are set, generate the final augmented dataset:

```bash
python Eyepacs_generation.py

```

Output files will be organized with samples and summary grids in their respective output directories.

<p align="center">
<img src="assets/retina_generated_grid.png" width="600px">




<i>Example grid of generated synthetic target domain samples.</i>
</p>

### 5. Evaluation & Downstream Classification

Evaluate visual fidelity and prepare for downstream classification.

* `Messidor_FID.py`: Calculates Frechet Inception Distance.
* `calculate_LPIPS.py`: Calculates LPIPS diversity and fidelity metrics.
* `Messidor_real_data_for_comparison.py`: Formats and saves real ground-truth data for direct comparison against synthetic outputs.

**Note on Downstream Classifiers:** To maintain strict modularity between the generative pipeline and the clinical evaluation, the downstream ResNet classification framework (including data loaders, training loops, and metrics for Tables 1 & 2) is maintained in a dedicated secondary repository.

---

## 📄 Reproducibility & Open Science

This repository is submitted strictly for double-blind peer review. To ensure compliance with MICCAI anonymity guidelines, model checkpoints have been temporarily withheld. Upon acceptance, we will release:

* Pre-trained Phase 1 model weights.
* The complete secondary classification repository, including all training scripts, validation loops, and downstream evaluation code.

---