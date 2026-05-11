import glob
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import umap

# Imports from your files
from Eyepacs_class_for_distance import EyepacsDataset
from Messidor_class import MessidorDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATED_SAMPLES_DIR = (
    "models/23_Feb_Eyepacs_fixed_dinov2/simulation_GS_2_same_class_model_390/samples"
)
NUM_EYEPACS_SAMPLES = 5000  # Subsample to keep the plot readable and fast


# --- 1. Custom Dataset for Generated Images ---
class GeneratedImagesDataset(Dataset):
    def __init__(self, root_dir):
        # Finds all PNGs inside the 50 subdirectories
        self.image_paths = glob.glob(os.path.join(root_dir, "*", "*.png"))
        print(f"Found {len(self.image_paths)} generated images in {root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = transforms.ToTensor()(img)
        # Map [0, 1] back to [-1, 1] to exactly match Messidor/Eyepacs preprocessing
        tensor = tensor * 2.0 - 1.0
        return tensor


# --- 2. Feature Extractor ---
def get_feature_encoder():
    """Returns the frozen ResNet18 encoder"""
    print("Loading ResNet18 Encoder...")
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder = encoder.to(DEVICE)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features(loader, encoder, desc="Extracting"):
    features_list = []

    resize_norm = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for batch in tqdm(loader, desc=desc):
        # Handle both (img, label) tuples and pure img batches
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(DEVICE)

        # Un-normalize from [-1, 1] back to [0, 1]
        imgs = imgs * 0.5 + 0.5

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        imgs = resize_norm(imgs)
        raw_feats = encoder(imgs)
        norm_feats = F.normalize(raw_feats, p=2, dim=1)

        features_list.append(norm_feats.cpu())

    return torch.cat(features_list, dim=0)


def main():
    encoder = get_feature_encoder()

    # --- Load Data ---
    print("\n--- Loading Datasets ---")

    # 1. Generated Data
    gen_dataset = GeneratedImagesDataset(GENERATED_SAMPLES_DIR)
    gen_loader = DataLoader(gen_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 2. Messidor (Target Distribution)
    messidor_dataset = MessidorDataset(
        purpose="hospital_b", img_size=224, useRetFoundPreprocessing=False
    )
    messidor_loader = DataLoader(messidor_dataset, batch_size=50, shuffle=False, num_workers=4)

    # 3. Eyepacs (Source Distribution - Subsampled)
    full_eyepacs_dataset = EyepacsDataset(purpose="train", img_size=224)
    # Randomly subsample Eyepacs so it doesn't overwhelm the plot
    indices = random.sample(range(len(full_eyepacs_dataset)), NUM_EYEPACS_SAMPLES)
    eyepacs_subset = Subset(full_eyepacs_dataset, indices)
    eyepacs_loader = DataLoader(eyepacs_subset, batch_size=128, shuffle=False, num_workers=4)

    # --- Extract Features ---
    print("\n--- Extracting Features ---")
    feat_gen = extract_features(gen_loader, encoder, desc="Generated Images")
    feat_mes = extract_features(messidor_loader, encoder, desc="Messidor (Hospital B)")
    feat_eye = extract_features(eyepacs_loader, encoder, desc="Eyepacs (Train Subset)")

    # --- Combine and Create Labels ---
    X = torch.cat([feat_eye, feat_mes, feat_gen], dim=0).numpy()

    y_labels = (
        ["Eyepacs (Source)"] * len(feat_eye)
        + ["Messidor (Target)"] * len(feat_mes)
        + ["Generated Cohort"] * len(feat_gen)
    )

    # --- Compute UMAP ---
    print("\n--- Computing UMAP (This may take a minute) ---")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    # --- Plotting ---
    print("\n--- Plotting ---")
    plt.figure(figsize=(10, 8))

    # We define a specific palette and plot order so Generated and Messidor pop out on top
    palette = {
        "Eyepacs (Source)": "#d3d3d3",  # Light Grey (Background)
        "Messidor (Target)": "#1f77b4",  # Solid Blue (Target)
        "Generated Cohort": "#ff7f0e",  # Bright Orange (Your model's output)
    }

    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y_labels,
        palette=palette,
        hue_order=["Eyepacs (Source)", "Generated Cohort", "Messidor (Target)"],
        s=25,
        alpha=0.7,
        edgecolor=None,
    )

    plt.title("Distribution Shift: Source vs Target vs Generated", fontsize=14, pad=15)
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = "umap_distribution_shift_2.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved UMAP visualization to {save_path}")


if __name__ == "__main__":
    main()
