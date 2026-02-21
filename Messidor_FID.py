import glob
import os

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

# --- Configuration ---
# EXPERIMENT_ROOT = "models/18_Feb_Eyepacs_DDP/simulation_GS_1.5_same_class_model_140"
EXPERIMENT_ROOT = "models/18_Feb_Eyepacs_DDP/simulation_GS_0_same_class_model_140"
# EXPERIMENT_ROOT = "models/18_Feb_Eyepacs_DDP/simulation_GS_1_same_class_model_120"
# EXPERIMENT_ROOT = "models/18_Feb_Eyepacs_DDP/hidden_class_samples_unique"

# Derived Paths
SAMPLES_DIR = os.path.join(EXPERIMENT_ROOT, "samples")
SUMMARY_DIR = os.path.join(EXPERIMENT_ROOT, "summary")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FID Metric
# By default, torchmetrics FID expects images as uint8 in the range [0, 255].
# feature=2048 is the standard Inception v3 pooling layer used for FID.
fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

# FID typically expects 299x299 images for the Inception network.
# PILToTensor converts the PIL Image to a uint8 tensor in [0, 255].
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.PILToTensor(),
    ]
)


def load_image(path):
    try:
        # Load RGB (Inception expects 3 channels)
        img = Image.open(path).convert("RGB")
        # Adds batch dimension: (1, 3, H, W)
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def main():
    print("\n--- Starting FID Analysis ---")
    print(f"Scanning: {EXPERIMENT_ROOT}\n")

    if not os.path.exists(SAMPLES_DIR):
        print("Error: Samples directory not found.")
        return

    subfolders = [f.path for f in os.scandir(SAMPLES_DIR) if f.is_dir()]
    subfolders.sort(key=lambda f: int(os.path.basename(f)) if os.path.basename(f).isdigit() else f)

    total_reals = 0
    total_fakes = 0

    print("Extracting features from folders. This may take a while...")

    for folder in subfolders:
        folder_name = os.path.basename(folder)

        # 1. Load the Original Anchor (Real Distribution)
        orig_path = os.path.join(SUMMARY_DIR, folder_name, "original.png")

        if os.path.exists(orig_path):
            original_tensor = load_image(orig_path)
            if original_tensor is not None:
                # Update real features
                fid_metric.update(original_tensor, real=True)
                total_reals += 1
        else:
            print(f"Warning: No original.png found for {folder_name}.")

        # 2. Load Generated Samples (Fake Distribution)
        sample_files = glob.glob(os.path.join(folder, "*.png")) + glob.glob(
            os.path.join(folder, "*.jpg")
        )

        # Filter out utility images
        sample_files = [
            f
            for f in sample_files
            if "grid" not in f and "original" not in f and "reference" not in f
        ]

        # Batch load samples for this folder to speed up Inception inference
        sample_tensors = []
        for f in sample_files:
            t = load_image(f)
            if t is not None:
                # Remove batch dim temporarily to stack them properly
                sample_tensors.append(t.squeeze(0))

        if sample_tensors:
            # Stack into a batch: (N, 3, 299, 299)
            batch_fakes = torch.stack(sample_tensors, dim=0).to(device)
            # Update fake features
            fid_metric.update(batch_fakes, real=False)
            total_fakes += len(sample_tensors)

    # --- Compute Final Score ---
    print("\n" + "-" * 60)
    print("OVERALL FID SCORE")
    print("-" * 60)
    print(f"Total Real Images Processed: {total_reals}")
    print(f"Total Fake Images Processed: {total_fakes}")

    if total_reals < 2 or total_fakes < 2:
        print("Error: Not enough images to compute covariance for FID.")
        return

    # Compute runs the final math over all accumulated features
    print("Computing Fréchet Inception Distance...")
    fid_score = fid_metric.compute()

    print(f"\nFID Score: {fid_score.item():.4f} (Lower is better)")
    print("-" * 60)


if __name__ == "__main__":
    main()
