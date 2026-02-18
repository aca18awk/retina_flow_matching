import os

import numpy as np
import torch
from CM_class import ColoredMNIST
from torchvision import transforms
from torchvision.utils import make_grid, save_image


# --- 2. Helper to Organize Data by Label ---
def get_indices_by_label(dataset):
    """Returns a dict where keys are labels (0-9) and values are lists of indices."""
    indices_by_label = {i: [] for i in range(10)}
    print("Indexing dataset by label...")
    for idx in range(len(dataset)):
        label = int(dataset.targets[idx])
        indices_by_label[label].append(idx)
    return indices_by_label


# --- 3. Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    IMG_SIZE = 32
    BATCH_SIZE = 50  # We only need the first 50 anchors
    N_SAMPLES = 100  # How many real samples to draw per anchor

    # Paths
    base_savedir = "models/15_Feb_Coloured_MNIST_FSFM_Latent"
    # We add _REAL to distinguish this from the generative experiment
    experiment_dir = os.path.join(base_savedir, "simulation_blue_REAL")

    sample_folder_root = os.path.join(experiment_dir, "samples")
    summary_folder_root = os.path.join(experiment_dir, "summary")

    os.makedirs(sample_folder_root, exist_ok=True)
    os.makedirs(summary_folder_root, exist_ok=True)

    # Transforms (Must match your generation script)
    # Normalization: (0.5, 0.5, 0.5) -> resulting range [-1, 1]
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Dataset (Force BLUE)
    val_dataset = ColoredMNIST(
        root="../../../../datasets/MNIST",
        train=False,
        download=True,
        transform=transform,
        set_color_idx=2,  # BLUE
    )

    # Efficiently lookup indices by label so we don't scan the dataset 5000 times
    indices_by_label = get_indices_by_label(val_dataset)

    print(
        f"Starting Real Data Extraction (Blue Only, Anchors=50, Samples per Anchor={N_SAMPLES})..."
    )

    # Iterate through the first 50 anchors (indices 0 to 49)
    for anchor_idx in range(BATCH_SIZE):
        # 1. Setup Folders
        sample_folder = os.path.join(sample_folder_root, str(anchor_idx))
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, str(anchor_idx))
        os.makedirs(summary_folder, exist_ok=True)

        # 2. Get Anchor Image & Label
        anchor_img, anchor_label = val_dataset[anchor_idx]

        # Denormalize Anchor for saving: [-1, 1] -> [0, 1]
        anchor_img_denorm = torch.clamp((anchor_img + 1) / 2, 0, 1)
        save_image(anchor_img_denorm, os.path.join(summary_folder, "original.png"))

        # 3. Sample 100 REAL images of the same label
        possible_indices = indices_by_label[anchor_label]

        # Exclude the anchor itself from the sampling pool if you want strict "others"
        # (Though with 1000+ samples, it rarely matters)
        if anchor_idx in possible_indices:
            possible_indices = [x for x in possible_indices if x != anchor_idx]

        # Randomly select N_SAMPLES
        selected_indices = np.random.choice(possible_indices, N_SAMPLES, replace=False)

        # Load these images
        real_cohort_imgs = []
        for i, idx in enumerate(selected_indices):
            img, _ = val_dataset[idx]

            # Denormalize: [-1, 1] -> [0, 1]
            img_denorm = torch.clamp((img + 1) / 2, 0, 1)
            real_cohort_imgs.append(img_denorm)

            # Save individual sample
            save_image(img_denorm, os.path.join(sample_folder, f"{i}.png"))

        # 4. Save Cohort Grid
        real_cohort_tensor = torch.stack(real_cohort_imgs)
        grid_img = make_grid(real_cohort_tensor, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

        print(f"  Processed Anchor {anchor_idx} (Label: {anchor_label})")

    print(f"Done. Data saved to: {experiment_dir}")
