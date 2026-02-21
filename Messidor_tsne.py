import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from Messidor_class import MessidorDataset
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Import your dataset class
# from messidor_dataset import MessidorDataset

# --- Configuration ---
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOSPITAL_B_CSV = "hospital_B_file_list.csv"
DATA_ROOT = "/vol/biomedic3/awk24/datasets/Messidor2"
LABELS_CSV = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"


def get_features_and_labels(dataset):
    """
    Passes all images through a pre-trained ResNet to get feature vectors.
    Returns: (features_array, labels_array, filenames_list)
    """
    print(f"Extracting features using ResNet18 on {DEVICE}...")

    # 1. Load Pre-trained Model
    # We remove the final classification layer (fc) to get the raw features
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # Replace final layer with identity
    model.to(DEVICE)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(loader):
            images = images.to(DEVICE)

            # Forward pass to get features (Batch, 512)
            feats = model(images)

            # Move to CPU and numpy
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_filenames.extend(filenames)

    # Concatenate all batches
    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.array(all_labels)

    return features_np, labels_np, all_filenames


def plot_tsne(features, labels, filenames, hospital_b_files):
    """
    Computes t-SNE and plots the results.
    """
    print("Computing t-SNE... (this uses CPU, might take a moment)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    # Create DataFrame
    df_plot = pd.DataFrame(
        {"x": tsne_results[:, 0], "y": tsne_results[:, 1], "label": labels, "filename": filenames}
    )

    # Identify Hospital B
    df_plot["is_hospital_b"] = df_plot["filename"].isin(hospital_b_files)

    plt.figure(figsize=(12, 10))

    # 1. Define Palette
    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))

    # 2. Plot background points (The main dataset)
    # We plot ONLY the points that are NOT in Hospital B as circles
    # This prevents the circle from hiding behind the X
    df_main = df_plot[~df_plot["is_hospital_b"]]
    sns.scatterplot(
        data=df_main, x="x", y="y", hue="label", palette=palette, s=60, alpha=0.5, legend="full"
    )

    # 3. Plot Hospital B points as 'X'
    # We use the SAME palette and hue logic so they match the class color
    df_hospital_b = df_plot[df_plot["is_hospital_b"]]
    sns.scatterplot(
        data=df_hospital_b,
        x="x",
        y="y",
        hue="label",  # Same color mapping
        palette=palette,  # Same palette
        marker="X",  # distinct shape
        s=200,  # Much larger
        edgecolor="black",  # Add a black outline to the X for visibility
        linewidth=1.5,
        legend=False,  # Don't duplicate the legend
    )

    plt.title("t-SNE of Messidor Dataset\n(X = Hospital B Selection)", fontsize=15)
    plt.tight_layout()
    plt.savefig("messidor_tsne_class_colored.png", dpi=300)
    print("Plot saved.")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load List of Hospital B files
    if not os.path.exists(HOSPITAL_B_CSV):
        print(f"Error: Could not find {HOSPITAL_B_CSV}. Run the selection script first.")
    else:
        df_b = pd.read_csv(HOSPITAL_B_CSV)
        hospital_b_set = set(df_b["filename"].values)
        print(f"Loaded {len(hospital_b_set)} filenames for Hospital B.")

        # 2. Initialize Dataset
        # Ensure your MessidorDataset class is available here
        dataset = MessidorDataset(
            root_dir=DATA_ROOT,
            csv_path=LABELS_CSV,
            img_size=224,  # ResNet expects 224x224 usually
        )

        # 3. Extract Features
        feats, labs, fnames = get_features_and_labels(dataset)

        # 4. Plot
        plot_tsne(feats, labs, fnames, hospital_b_set)
