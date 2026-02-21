import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from Eyepacs_class import EyepacsDataset

if __name__ == "__main__":
    # --- 1. Deterministic Seeding ---
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Ensure underlying algorithms are deterministic
    torch.backends.cudnn.deterministic = True

    # --- 2. Config ---
    ROOT_DIR = "/vol/biomedic3/awk24/datasets/EYEPACS_256"

    print(f"--- Running Deterministic Neighbor Diagnostic (Seed: {SEED}) ---")

    try:
        train_ds = EyepacsDataset(root=ROOT_DIR, purpose="train")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        exit()

    def show_fixed_neighbors(dataset, num_samples=5):
        print(f"Generating comparison for {num_samples} fixed images...")

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        # Instead of random, we can pick specific indices or
        # use the seeded generator to pick the same "random" ones
        idxs = torch.linspace(0, len(dataset) - 1, steps=num_samples).long()

        for row, idx in enumerate(idxs):
            idx = idx.item()
            query_path = dataset.image_paths[idx]

            # neighbor_idxs are the k-nearest neighbors found by your feature extractor
            neighbor_idxs = dataset.indices[idx]

            n1_idx = neighbor_idxs[0].item()
            n2_idx = neighbor_idxs[1].item()

            n1_path = dataset.image_paths[n1_idx]
            n2_path = dataset.image_paths[n2_idx]

            try:
                q_img = plt.imread(query_path)
                n1_img = plt.imread(n1_path)
                n2_img = plt.imread(n2_path)

                axes[row, 0].imshow(q_img)
                axes[row, 0].set_title(f"Query (ID: {idx})")
                axes[row, 0].axis("off")

                axes[row, 1].imshow(n1_img)
                axes[row, 1].set_title(f"Neighbor 1 (ID: {n1_idx})")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(n2_img)
                axes[row, 2].set_title(f"Neighbor 2 (ID: {n2_idx})")
                axes[row, 2].axis("off")

            except Exception as e:
                print(f"Error loading images for row {row}: {e}")

        plt.tight_layout()
        save_path = "check_neighbors.png"
        plt.savefig(save_path)
        print(f"\n[DONE] Saved fixed diagnostic to: {save_path}")

    show_fixed_neighbors(train_ds)
