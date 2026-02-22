import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Eyepacs_class_for_distance import EyepacsDataset
from Messidor_class import MessidorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
CHUNK_SIZE = 1000
K_NEIGHBORS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature_encoder():
    """Returns the frozen ResNet18 encoder"""
    encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder = encoder.to(DEVICE)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_all_features_and_labels(loader, encoder):
    print("Step 1: Extracting features and labels...")
    features_list = []
    labels_list = []

    # Define the transform expected by ResNet (ImageNet stats)
    resize_norm = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for batch in tqdm(loader):
        imgs = batch[0].to(DEVICE)
        labels = batch[1]

        # Un-normalize from [-1, 1] back to [0, 1]
        imgs = imgs * 0.5 + 0.5

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        imgs = resize_norm(imgs)
        raw_feats = encoder(imgs)

        # L2 Normalize
        norm_feats = F.normalize(raw_feats, p=2, dim=1)

        features_list.append(norm_feats.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)


def find_neighbors(all_features, all_labels, k=10, chunk_size=1000, same_label=True):
    """
    Finds k-NN while masking out self-matches AND enforcing same-class matches.
    Used for TRAINING.
    """
    print(f"Step 2: Finding {k} Nearest Neighbors (Class-Restricted)...")
    N = all_features.shape[0]
    database_features = all_features.to(DEVICE)
    database_labels = all_labels.to(DEVICE)
    all_indices = []

    for i in tqdm(range(0, N, chunk_size)):
        end_idx = min(i + chunk_size, N)
        current_chunk_size = end_idx - i

        query_feats = database_features[i:end_idx]
        dists = torch.cdist(query_feats, database_features)
        if same_label:
            query_labels = database_labels[i:end_idx]

            mismatch_mask = query_labels.unsqueeze(1) != database_labels.unsqueeze(0)
            dists.masked_fill_(mismatch_mask, float("inf"))

        row_indices = torch.arange(current_chunk_size, device=DEVICE)
        col_indices = torch.arange(i, end_idx, device=DEVICE)
        dists[row_indices, col_indices] = float("inf")

        _, indices = torch.topk(dists, k=k, largest=False, dim=1)
        all_indices.append(indices.cpu())

    return torch.cat(all_indices, dim=0)


def generate_ImageNet_Embeddings(
    SPLIT, datasetName, SAME_LABEL, useRetFoundPreprocessing, savedir=""
):
    if datasetName == "EYEPACS":
        dataset = EyepacsDataset(
            purpose=SPLIT,
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )
    else:
        dataset = MessidorDataset(
            purpose=SPLIT,
            root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
            csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )

    # 1. Setup Data
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # 2. Extract
    encoder = get_feature_encoder()
    all_features, all_labels = extract_all_features_and_labels(loader, encoder)

    # 3. Find Neighbors based on split
    if SPLIT == "validation" and datasetName == "EYEPACS":
        neighbor_indices = find_neighbors(
            all_features, all_labels, k=K_NEIGHBORS, chunk_size=CHUNK_SIZE, same_label=False
        )
    else:
        neighbor_indices = find_neighbors(
            all_features,
            all_labels,
            k=K_NEIGHBORS,
            chunk_size=CHUNK_SIZE,
            same_label=SAME_LABEL,
        )

    # 4. Save dynamically based on split name
    postFix = "_RetFoundPreprocessing" if useRetFoundPreprocessing else ""
    label = "_same_label" if SAME_LABEL else ""
    print(f"Saving {all_features.shape} features and indices...")
    # torch.save(all_features, f"{datasetName}_{SPLIT}_features_ImageNet{label}{postFix}.pt")
    # torch.save(neighbor_indices, f"{datasetName}_{SPLIT}_indices_ImageNet{label}{postFix}.pt")

    feat_name = f"features_ImageNet{label}{postFix}.pt"
    idx_name = f"indices_ImageNet{label}{postFix}.pt"

    torch.save(all_features, os.path.join(savedir, feat_name))
    torch.save(neighbor_indices, os.path.join(savedir, idx_name))
    print("Done!")


if __name__ == "__main__":
    datasetName = "Messidor"
    SPLIT = "hidden"
    useRetFoundPreprocessing = False
    SAME_LABEL = False

    generate_ImageNet_Embeddings(SPLIT, datasetName, SAME_LABEL, useRetFoundPreprocessing)
