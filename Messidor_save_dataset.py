import os
import random

import pandas as pd
from Messidor_class import MessidorDataset
from torchvision.utils import save_image
from tqdm import tqdm

# --- Configuration ---
original_root = "/vol/biomedic3/awk24/datasets/Messidor2"
csv_labels_path = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
hospital_b_list_csv = "hospital_B_file_list.csv"

# Output Paths
output_base = "/vol/biomedic3/awk24/datasets/Messidor2_256"
dir_hospital_b = os.path.join(output_base, "hospital_b")
dir_test = os.path.join(output_base, "test")
dir_hidden = os.path.join(output_base, "hidden_classifier_data")


def unnormalize(tensor):
    """
    The dataset normalizes to [-1, 1].
    We need to convert back to [0, 1] for saving as PNG.
    """
    return tensor * 0.5 + 0.5


if __name__ == "__main__":
    # 1. Create Output Directories
    os.makedirs(dir_hospital_b, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)
    os.makedirs(dir_hidden, exist_ok=True)
    print(f"Created output directories in {output_base}")

    # 2. Initialize Dataset (This handles the resizing/cropping/padding)
    print("Initializing Dataset...")
    dataset = MessidorDataset(root_dir=original_root, csv_path=csv_labels_path, img_size=256)

    # 3. Define Sets for Splitting
    # A. Hospital B (Fixed 50)
    df_b = pd.read_csv(hospital_b_list_csv)
    hospital_b_filenames = set(df_b["filename"].values)

    # B. Identify Remaining Pool
    all_filenames = [os.path.basename(p) for p in dataset.image_paths]
    remaining_pool = [f for f in all_filenames if f not in hospital_b_filenames]

    # Shuffle to ensure randomness for Test and Hidden
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(remaining_pool)

    # C. Split the Pool
    # We need 600 for test, 1000 for hidden
    if len(remaining_pool) < 1600:
        raise ValueError(
            f"Not enough images! Needed 1600, but only have {len(remaining_pool)} left."
        )

    test_filenames = set(remaining_pool[:600])
    hidden_filenames = set(remaining_pool[600:1600])

    print("Split Statistics:")
    print(f"  - Hospital B: {len(hospital_b_filenames)}")
    print(f"  - Test:       {len(test_filenames)}")
    print(f"  - Hidden:     {len(hidden_filenames)}")
    print(f"  - Unused:     {len(remaining_pool) - 1600}")

    # 4. Processing Loop
    print("Processing and saving images...")

    # We create a quick lookup map to speed up finding the folder
    # Map: filename -> target_directory
    file_destinations = {}
    for f in hospital_b_filenames:
        file_destinations[f] = dir_hospital_b
    for f in test_filenames:
        file_destinations[f] = dir_test
    for f in hidden_filenames:
        file_destinations[f] = dir_hidden

    count = 0

    for i in tqdm(range(len(dataset))):
        img_tensor, _, fname = dataset[i]

        # Check if this file is in one of our target lists
        if fname in file_destinations:
            target_dir = file_destinations[fname]
            save_path = os.path.join(target_dir, fname)

            # Un-normalize back to [0, 1] range
            img_save = unnormalize(img_tensor)

            save_image(img_save, save_path)
            count += 1

    print(f"Done! Saved {count} images to {output_base}")
