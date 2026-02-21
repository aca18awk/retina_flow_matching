import os

import numpy as np
import torch
from Messidor_class import MessidorDataset
from torchvision.utils import make_grid, save_image


# --- 1. Helper to Organize Data by Label ---
def get_indices_by_label(dataset):
    """Returns a dict where keys are labels (0-4) and values are lists of indices."""
    indices_by_label = {i: [] for i in range(5)}
    print("Indexing dataset by label...")

    for idx, img_path in enumerate(dataset.image_paths):
        filename = os.path.basename(img_path)

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

    IMG_SIZE = 128
    N_SAMPLES = 50  # Target samples per disease class (will take less if not enough)

    # Paths
    base_savedir = "models/18_Feb_Eyepacs_DDP"
    experiment_dir = os.path.join(base_savedir, "hidden_class_samples_unique")

    sample_folder_root = os.path.join(experiment_dir, "samples")
    summary_folder_root = os.path.join(experiment_dir, "summary")

    os.makedirs(sample_folder_root, exist_ok=True)
    os.makedirs(summary_folder_root, exist_ok=True)

    # Dataset: "hidden" split
    val_dataset = MessidorDataset(
        purpose="hidden",
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=IMG_SIZE,
    )

    # Efficiently lookup indices by label
    indices_by_label = get_indices_by_label(val_dataset)

    print(
        f"Extracting up to {N_SAMPLES} unique samples per disease class from the 'hidden' split..."
    )

    # Iterate through the 5 disease classes
    for class_label in range(5):
        # 1. Setup Folders
        sample_folder = os.path.join(sample_folder_root, str(class_label))
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, str(class_label))
        os.makedirs(summary_folder, exist_ok=True)

        possible_indices = indices_by_label[class_label]

        if not possible_indices:
            print(f"  Class {class_label} has no images! Skipping.")
            continue

        # 2. Get pseudo-anchor (First available image of this class)
        anchor_idx = possible_indices[0]
        anchor_img, _, _ = val_dataset[anchor_idx]

        # Denormalize Anchor for saving: [-1, 1] -> [0, 1]
        anchor_img_denorm = torch.clamp((anchor_img + 1) / 2, 0, 1)
        save_image(anchor_img_denorm, os.path.join(summary_folder, "original.png"))

        # 3. Sample unique REAL images of this class
        # Target N_SAMPLES, but cap it at the total available if we fall short
        num_to_take = min(N_SAMPLES, len(possible_indices))
        selected_indices = np.random.choice(possible_indices, num_to_take, replace=False)

        # Load these images
        real_cohort_imgs = []
        for i, idx in enumerate(selected_indices):
            img, _, _ = val_dataset[idx]

            # Denormalize: [-1, 1] -> [0, 1]
            img_denorm = torch.clamp((img + 1) / 2, 0, 1)
            real_cohort_imgs.append(img_denorm)

            # Save individual sample (0.png up to num_to_take-1.png)
            save_image(img_denorm, os.path.join(sample_folder, f"{i}.png"))

        # 4. Save Cohort Grid
        real_cohort_tensor = torch.stack(real_cohort_imgs)
        # nrow=10 keeps it 10 wide, height will just dynamically shrink if < 50 images
        grid_img = make_grid(real_cohort_tensor, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

        print(
            f"  Processed Class {class_label} | Available: {len(possible_indices)} | Extracted: {num_to_take}"
        )

    print(f"\nDone. Unique data saved to: {experiment_dir}")
