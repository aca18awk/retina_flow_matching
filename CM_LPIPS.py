import glob
import itertools
import os

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

# --- Configuration ---
EXPERIMENT_ROOT = "models/15_Feb_Coloured_MNIST_FSFM_Latent/simulation_blue_GS_1_same_label"

# Derived Paths
SAMPLES_DIR = os.path.join(EXPERIMENT_ROOT, "samples")
SUMMARY_DIR = os.path.join(EXPERIMENT_ROOT, "summary")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize LPIPS Metric
# net_type='alex' is standard for LPIPS, but 'vgg' is also common. 'alex' is faster.
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

# LPIPS expects images in [0, 1] or [-1, 1].
# We will load as [0, 1] float tensors.
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Ensure size matches
        transforms.ToTensor(),  # Converts (0, 255) -> (0.0, 1.0)
    ]
)


def load_image(path):
    try:
        # Load RGB (LPIPS works on color)
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def main():
    print("\n--- Starting LPIPS Analysis ---")
    print(f"Scanning: {EXPERIMENT_ROOT}\n")

    if not os.path.exists(SAMPLES_DIR):
        print("Error: Samples directory not found.")
        return

    subfolders = [f.path for f in os.scandir(SAMPLES_DIR) if f.is_dir()]
    subfolders.sort(key=lambda f: int(os.path.basename(f)) if os.path.basename(f).isdigit() else f)

    # Global Accumulators
    all_fidelity_scores = []
    all_diversity_scores = []

    print(f"{'Folder':<8} | {'Fidelity (vs Orig)':<20} | {'Diversity (Pairwise)':<20}")
    print("-" * 60)

    for folder in subfolders:
        folder_name = os.path.basename(folder)

        # 1. Load the Original Anchor (Ground Truth)
        # Location: summary/{ID}/original.png
        orig_path = os.path.join(SUMMARY_DIR, folder_name, "original.png")

        if not os.path.exists(orig_path):
            print(f"Skipping {folder_name}: No original.png found.")
            continue

        original_tensor = load_image(orig_path)
        if original_tensor is None:
            continue

        # 2. Load Generated Samples
        sample_files = glob.glob(os.path.join(folder, "*.png")) + glob.glob(
            os.path.join(folder, "*.jpg")
        )
        # Filter out utility images
        sample_files = [
            f
            for f in sample_files
            if "grid" not in f and "original" not in f and "reference" not in f
        ]

        if len(sample_files) < 2:
            print(f"Skipping {folder_name}: Not enough samples for statistics.")
            continue

        # Load all samples into a list of tensors
        sample_tensors = []
        for f in sample_files:
            t = load_image(f)
            if t is not None:
                sample_tensors.append(t)

        if not sample_tensors:
            continue

        # Stack samples for batch processing if memory allows, or loop
        # For safety/clarity, we'll loop.

        # --- A. Calculate Fidelity (Dist to Original) ---
        # How far is each sample from the original?
        folder_fidelity_vals = []
        for sample in sample_tensors:
            with torch.no_grad():
                # LPIPS expects inputs in range [-1, 1] if normalize=True isn't set perfectly
                # But torchmetrics handles [0,1] well if initialized correctly.
                # We normalize inputs to [-1, 1] manually just to be 100% safe with LPIPS conventions
                # (0,1) * 2 - 1 = (-1, 1)
                val = lpips_metric(sample, original_tensor)
                folder_fidelity_vals.append(val.item())

        avg_fidelity = np.mean(folder_fidelity_vals)
        all_fidelity_scores.extend(folder_fidelity_vals)

        # --- B. Calculate Diversity (Pairwise Distance) ---
        # How far is sample_i from sample_j?
        folder_diversity_vals = []

        # We limit pairs to avoid N^2 explosion if you have 1000 images.
        # For < 50 images, all pairs is fine.
        # itertools.combinations(list, 2) gives unique pairs
        pairs = list(itertools.combinations(sample_tensors, 2))

        # If too many pairs, subsample (optional)
        if len(pairs) > 100:
            import random

            pairs = random.sample(pairs, 100)

        for s1, s2 in pairs:
            with torch.no_grad():
                val = lpips_metric(s1, s2)
                folder_diversity_vals.append(val.item())

        avg_diversity = np.mean(folder_diversity_vals) if folder_diversity_vals else 0.0
        all_diversity_scores.extend(folder_diversity_vals)

        print(f"{folder_name:<8} | {avg_fidelity:.4f}               | {avg_diversity:.4f}")

    # --- Summary ---
    print("-" * 60)
    print("OVERALL METRICS")
    print(
        f"Mean Fidelity (Sample vs Original): {np.mean(all_fidelity_scores):.4f} (Lower is better)"
    )
    if all_diversity_scores:
        print(
            f"Mean Diversity (Sample vs Sample):  {np.mean(all_diversity_scores):.4f} (Higher means more varied)"
        )
    print("-" * 60)


if __name__ == "__main__":
    main()
