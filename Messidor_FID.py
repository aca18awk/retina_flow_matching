import glob
import os

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

# --- Configuration ---
EXPERIMENT_ROOT = "models/23_Feb_Eyepacs_fixed_dinov2/simulation_GS_1_same_class_model_390"


# Derived Paths
SAMPLES_DIR = os.path.join(EXPERIMENT_ROOT, "samples")
SUMMARY_DIR = os.path.join(EXPERIMENT_ROOT, "summary")

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Initialize FID Metric
    # By default, torchmetrics FID expects images as uint8 in the range [0, 255].
    # feature=2048 is the standard Inception v3 pooling layer used for FID.
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

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
            # Update features
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
    fid_metric = fid_metric.to("cpu")
    fid_score = fid_metric.compute()

    print(f"\nFID Score: {fid_score.item():.4f} (Lower is better)")
    print("-" * 60)


if __name__ == "__main__":
    main()


# NOTE: FID for real data
# import os
# from pathlib import Path

# import torch
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchvision import transforms

# # --- Configuration ---
# # Update these paths. FID requires two distributions to compare: Reals and Fakes.
# BASE_DIR = "/vol/biomedic3/awk24/datasets/Messidor2_256"

# # Merging "this directory" and the "validation" subfolder for one of the distributions
# MERGED_DIRS = [
#     os.path.join(BASE_DIR, "hidden_classifier_data"),
#     os.path.join(BASE_DIR, "validation"),
# ]

# # TODO: Assign your directories to REAL and FAKE
# # (e.g., if MERGED_DIRS contains your generated images, set FAKE_DIRS = MERGED_DIRS)
# REAL_DIRS = [os.path.join(BASE_DIR, "hospital_b")]  # <-- ADD YOUR REAL IMAGE DIRECTORIES HERE
# FAKE_DIRS = MERGED_DIRS

# BATCH_SIZE = 32  # Adjust based on your GPU memory
# NUM_WORKERS = 4  # Speeds up image loading

# # --- Setup ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose(
#     [
#         transforms.Resize((299, 299)),
#         transforms.PILToTensor(),  # torchmetrics FID expects uint8 in [0, 255]
#     ]
# )


# class ImageDirectoryDataset(Dataset):
#     def __init__(self, directories, transform=None):
#         self.transform = transform
#         self.image_paths = []

#         valid_extensions = {".png", ".jpg", ".jpeg"}

#         # Gather all valid images from the provided list of directories
#         for directory in directories:
#             dir_path = Path(directory)
#             if not dir_path.exists() or not dir_path.is_dir():
#                 print(f"Warning: Directory {directory} not found or is not a dir.")
#                 continue

#             for file_path in dir_path.iterdir():
#                 if file_path.suffix.lower() in valid_extensions:
#                     name = file_path.name.lower()
#                     # Filter out utility images
#                     if "grid" not in name and "original" not in name and "reference" not in name:
#                         self.image_paths.append(file_path)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         try:
#             img = Image.open(path).convert("RGB")
#             if self.transform:
#                 img = self.transform(img)
#             return img
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             # Return a dummy tensor or handle better in production,
#             # but for a simple script, returning an empty tensor of right shape works
#             # if filtered out in the loop. For safety, we'll just return it and let DataLoader batch it.
#             return torch.zeros((3, 299, 299), dtype=torch.uint8)


# def process_dataset(fid_metric, dataloader, is_real, desc):
#     print(f"\nProcessing {desc} distribution...")
#     total_processed = 0

#     for batch in dataloader:
#         batch = batch.to(device)
#         fid_metric.update(batch, real=is_real)
#         total_processed += batch.shape[0]

#     print(f"Total {desc} images processed: {total_processed}")
#     return total_processed


# def main():
#     print("\n--- Starting FID Analysis ---")

#     if not REAL_DIRS or not FAKE_DIRS:
#         print("Error: You must provide paths for BOTH real and fake directories.")
#         return

#     # Initialize Datasets and DataLoaders
#     real_dataset = ImageDirectoryDataset(REAL_DIRS, transform=transform)
#     fake_dataset = ImageDirectoryDataset(FAKE_DIRS, transform=transform)

#     real_loader = DataLoader(
#         real_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
#     )
#     fake_loader = DataLoader(
#         fake_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
#     )

#     # Initialize FID Metric
#     fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

#     # Process Images
#     total_reals = process_dataset(fid_metric, real_loader, is_real=True, desc="REAL")
#     total_fakes = process_dataset(fid_metric, fake_loader, is_real=False, desc="FAKE")

#     # --- Compute Final Score ---
#     print("\n" + "-" * 60)
#     print("OVERALL FID SCORE")
#     print("-" * 60)

#     if total_reals < 2 or total_fakes < 2:
#         print("Error: Not enough images to compute covariance for FID (need at least 2 of each).")
#         return

#     print("Computing Fréchet Inception Distance...")
#     fid_metric = fid_metric.to("cpu")
#     fid_score = fid_metric.compute()

#     print(f"\nFID Score: {fid_score.item():.4f} (Lower is better)")
#     print("-" * 60)


# if __name__ == "__main__":
#     main()
