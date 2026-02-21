import os

import numpy as np
import torch
from Messidor_class import MessidorDataset
from torchvision.utils import make_grid, save_image


# --- 1. Helper to Organize Data by Label ---
def get_indices_by_label(dataset):
    """Returns a dict where keys are labels (0-4) and values are lists of indices."""
    indices_by_label = {i: [] for i in range(5)}  # Messidor has 5 classes
    print("Indexing dataset by label...")

    # We access the paths and labels directly to avoid loading all images into RAM
    for idx, img_path in enumerate(dataset.image_paths):
        filename = os.path.basename(img_path)

        # Match the label logic from MessidorDataset __getitem__
        if filename in dataset.labels:
            label = dataset.labels[filename].get("diagnosis", -1)
        elif os.path.splitext(filename)[0] in dataset.labels:
            label = dataset.labels[os.path.splitext(filename)[0]].get("diagnosis", -1)
        else:
            label = -1

        if 0 <= label <= 4:
            indices_by_label[label].append(idx)

    return indices_by_label


# --- 2. Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    IMG_SIZE = 128  # Updated to match your Messidor resolution
    BATCH_SIZE = 50  # We only need the first 50 anchors
    N_SAMPLES = 100  # How many real samples to draw per anchor

    # Paths
    base_savedir = "models/18_Feb_Eyepacs_DDP"
    experiment_dir = os.path.join(base_savedir, "real_labels")

    sample_folder_root = os.path.join(experiment_dir, "samples")
    summary_folder_root = os.path.join(experiment_dir, "summary")

    os.makedirs(sample_folder_root, exist_ok=True)
    os.makedirs(summary_folder_root, exist_ok=True)

    # Dataset
    val_dataset = MessidorDataset(
        purpose="hospital_b",
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=IMG_SIZE,
    )

    # Efficiently lookup indices by label
    indices_by_label = get_indices_by_label(val_dataset)

    print(f"Starting Real Data Extraction (Anchors=50, Samples per Anchor={N_SAMPLES})...")

    # Iterate through the first 50 anchors (indices 0 to 49)
    for anchor_idx in range(BATCH_SIZE):
        # 1. Setup Folders
        sample_folder = os.path.join(sample_folder_root, str(anchor_idx))
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, str(anchor_idx))
        os.makedirs(summary_folder, exist_ok=True)

        # 2. Get Anchor Image & Label (FIXED: Unpacking 3 variables)
        anchor_img, anchor_label, anchor_filename = val_dataset[anchor_idx]

        # Denormalize Anchor for saving: [-1, 1] -> [0, 1]
        anchor_img_denorm = torch.clamp((anchor_img + 1) / 2, 0, 1)
        save_image(anchor_img_denorm, os.path.join(summary_folder, "original.png"))

        # 3. Sample 100 REAL images of the same label
        possible_indices = indices_by_label[anchor_label].copy()

        # Exclude the anchor itself
        if anchor_idx in possible_indices:
            possible_indices.remove(anchor_idx)

        # FIXED: If we don't have 100 unique images (e.g., Class 4), we MUST sample with replacement
        replace_flag = len(possible_indices) < N_SAMPLES
        selected_indices = np.random.choice(possible_indices, N_SAMPLES, replace=replace_flag)

        # Load these images
        real_cohort_imgs = []
        for i, idx in enumerate(selected_indices):
            # FIXED: Unpacking 3 variables
            img, _, _ = val_dataset[idx]

            # Denormalize: [-1, 1] -> [0, 1]
            img_denorm = torch.clamp((img + 1) / 2, 0, 1)
            real_cohort_imgs.append(img_denorm)

            # Save individual sample
            save_image(img_denorm, os.path.join(sample_folder, f"{i}.png"))

        # 4. Save Cohort Grid
        real_cohort_tensor = torch.stack(real_cohort_imgs)
        grid_img = make_grid(real_cohort_tensor, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

        print(f"  Processed Anchor {anchor_idx} (Label: {anchor_label}, Replace={replace_flag})")

    print(f"Done. Data saved to: {experiment_dir}")
