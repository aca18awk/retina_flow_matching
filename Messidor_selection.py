import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
from Messidor_class import MessidorDataset
from torchvision.utils import make_grid

# Import your class (assuming it's in the same file or imported)
# from messidor_dataset import MessidorDataset


def select_and_plot_diversity(dataset, output_plot="hospital_B_selection.png"):
    """
    Scans the dataset, picks 10 random samples per class (0-4),
    plots them, and returns the selected filenames.
    """
    print("Scanning dataset for labels... this might take a minute...")

    # 1. Map Indices to Classes
    # We iterate once to build a dictionary: { class_id: [index1, index2, ...] }
    class_indices = {0: [], 1: [], 2: [], 3: [], 4: []}

    # We access the internal list directly to avoid loading images (much faster)
    # But we need to ensure the labels are loaded in the dataset init first.

    missing_labels = 0

    for idx in range(len(dataset)):
        img_path = dataset.image_paths[idx]
        filename = os.path.basename(img_path)

        # Logic to find label (duplicating the logic inside __getitem__ for speed)
        # We don't want to load the image here, just the label.
        label = -1

        # Try exact match
        if filename in dataset.labels:
            label = dataset.labels[filename].get("diagnosis", -1)
        # Try without extension
        elif os.path.splitext(filename)[0] in dataset.labels:
            label = dataset.labels[os.path.splitext(filename)[0]].get("diagnosis", -1)

        if label in class_indices:
            class_indices[label].append(idx)
        else:
            missing_labels += 1

    print(f"Scan complete. Found labels for {len(dataset) - missing_labels} images.")
    print(f"Missing/Invalid labels: {missing_labels}")
    for k, v in class_indices.items():
        print(f"  Class {k}: {len(v)} candidates")

    # 2. Select 10 Samples per Class
    selected_indices = []
    selected_metadata = []  # To store filename and class

    samples_per_class = 10

    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]

        if len(indices) < samples_per_class:
            print(f"Warning: Class {cls} has only {len(indices)} samples. Taking all.")
            chosen = indices
        else:
            # Randomly sample to ensure diversity
            chosen = random.sample(indices, samples_per_class)

        selected_indices.extend(chosen)

        # Record metadata for the CSV later
        for idx in chosen:
            fname = os.path.basename(dataset.image_paths[idx])
            selected_metadata.append({"filename": fname, "class": cls, "original_index": idx})

    # 3. Load the Actual Images for Plotting
    print(f"Loading {len(selected_indices)} images for plotting...")

    batch_images = []

    # We reload these specific indices using __getitem__ to get the tensors
    for idx in selected_indices:
        img_tensor, label, fname = dataset[idx]
        batch_images.append(img_tensor)

    # Stack into a single tensor (B, C, H, W)
    batch_tensor = torch.stack(batch_images)

    # 4. Create Grid Plot
    # nrow=10 means 10 images per row. Since we sorted by class,
    # Row 1 will be Class 0, Row 2 will be Class 1, etc.
    grid = make_grid(batch_tensor, nrow=samples_per_class, padding=2, normalize=True)

    # Convert to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(20, 12))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Hospital B Selection: 10 Samples Per Diagnosis Class (rows=classes)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

    return selected_metadata


# --- Execution ---
if __name__ == "__main__":
    # 1. Setup Dataset
    # Make sure to point to your actual CSV file
    csv_path = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
    root_dir = "/vol/biomedic3/awk24/datasets/Messidor2"

    dataset = MessidorDataset(
        root_dir=root_dir, csv_path=csv_path, img_size=128
    )  # Keep size small for plotting grid

    # 2. Run Selection
    selected_data = select_and_plot_diversity(dataset)

    # 3. Save Selection to CSV (so you can move files later)
    df = pd.DataFrame(selected_data)
    df.to_csv("hospital_B_file_list.csv", index=False)
    print("Selection list saved to 'hospital_B_file_list.csv'")

    print("Done! Check the PNG to see if the samples look diverse enough.")
