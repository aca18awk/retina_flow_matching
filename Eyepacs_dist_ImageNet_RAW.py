import os

import torch
import torch.nn as nn
import torchvision.models as models
from Eyepacs_class import EyepacsDataset  # Ensure this matches your file name
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Configuration ---
CHUNK_SIZE = 1000  # How many images to process at once (VRAM safe)
K_NEIGHBORS = 10  # Save 10 neighbors (we can use fewer later)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_encoder():
    """Returns the frozen ResNet18 encoder"""
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder = encoder.to(DEVICE)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_all_features(loader, encoder):
    """
    Extracts features for the entire dataset first.
    We need ALL features in memory to find global neighbors.
    """
    print("Step 1: Extracting features from entire dataset...")
    features_list = []

    # We use a standard dataloader here just to get the images
    for imgs in tqdm(loader):
        imgs = imgs.to(DEVICE)

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        features = encoder(imgs)
        features_list.append(features.cpu())  # Store on CPU to save GPU VRAM

    return torch.cat(features_list, dim=0)  # [N, 512]


def find_neighbors_chunked(all_features, k=10, chunk_size=1000):
    """
    Calculates neighbors using a Query Chunk vs. Whole Database approach.
    """
    print(f"Step 2: Finding {k} Nearest Neighbors (Chunked)...")
    N = all_features.shape[0]

    # Move database to GPU (if it fits, otherwise we need more complex logic)
    # 50k images * 512 floats * 4 bytes = ~100MB. This fits easily on GPU.
    database = all_features.to(DEVICE)

    all_indices = []

    # Iterate through dataset in chunks
    for i in tqdm(range(0, N, chunk_size)):
        end_idx = min(i + chunk_size, N)

        # 1. Get Query Chunk
        query = database[i:end_idx]  # [Chunk, 512]

        # 2. Compute Distances (Chunk vs. All)
        # Result: [Chunk, N]
        dists = torch.cdist(query, database)

        # 3. Find Top-K (Smallest distances)
        # We grab k+1 because the closest match is always the image itself (dist=0)
        _, indices = torch.topk(dists, k=k + 1, largest=False, dim=1)

        # 4. Remove Self-Match
        # The first column (index 0) is the image itself. We take 1 to k+1.
        neighbor_indices = indices[:, 1:]  # [Chunk, K]

        all_indices.append(neighbor_indices.cpu())

    return torch.cat(all_indices, dim=0)  # [N, K]


if __name__ == "__main__":
    # 1. Dataset (No shuffle! We need index alignment)
    dataset = EyepacsDataset(purpose="validation", img_size=128)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    # 2. Extract Features
    encoder = get_feature_encoder()
    all_features = extract_all_features(loader, encoder)
    print(f"Total features extracted: {all_features.shape}")

    # 3. Find Indices
    neighbor_indices = find_neighbors_chunked(all_features, k=K_NEIGHBORS, chunk_size=CHUNK_SIZE)

    # 4. Save
    # We save BOTH features and indices.
    # The training loop will load these to do the barycentric sampling.
    torch.save(all_features, os.path.join("Eyepacs_val_features.pt"))
    torch.save(neighbor_indices, os.path.join("Eyepacs_val_indices.pt"))

    print(f"Features: {all_features.shape}")
    print(f"Indices: {neighbor_indices.shape}")
