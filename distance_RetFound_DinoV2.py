import os

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from Eyepacs_class_for_distance import EyepacsDataset
from Messidor_class import MessidorDataset

# from Eyepacs_class_for_distance import EyepacsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
CHUNK_SIZE = 1000
K_NEIGHBORS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InstanceNormalize:
    """
    Replicates the per-image, per-channel normalization used in the official RETFound repo.
    """

    def __call__(self, tensor):
        # tensor shape is [C, H, W]
        for c in range(tensor.shape[0]):
            mean = tensor[c].mean()
            std = tensor[c].std()
            if std > 0:
                tensor[c] = (tensor[c] - mean) / std
            else:
                tensor[c] = tensor[c] - mean
        return tensor


def get_retfound_encoder():
    print("Loading Domain-Specific Encoder (RETFound DINOv2)...")

    # DINOv2 Architecture, NO global_pool argument here
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
    )

    # Load the checkpoint dictionary (Make sure this is the file you just downloaded!)
    checkpoint = torch.load("RETFound_dinov2_meh.pth", map_location="cpu")

    # FIX 2: DINOv2 saves the best weights under the 'teacher' key
    state_dict = checkpoint.get("teacher", checkpoint.get("model", checkpoint))

    # Clean the keys (strip prefixes often added during distributed training)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("encoder.", "").replace("backbone.", "")
        cleaned_state_dict[clean_k] = v

    # Load the weights
    msg = model.load_state_dict(cleaned_state_dict, strict=False)

    # --- THE AUDIT ---
    print("\n" + "=" * 40)
    print("WEIGHT LOADING AUDIT:")
    print(f"Missing keys (In model, but not found in file): {len(msg.missing_keys)}")
    print(f"Unexpected keys (In file, but not used in model): {len(msg.unexpected_keys)}")
    print("=" * 40 + "\n")

    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def extract_all_features_domain_specific(loader, encoder, useRetFoundPreprocessing):
    print("Step 1: Extracting domain-specific features (No Labels Required)...")
    features_list = []

    norm = transforms.Compose([InstanceNormalize()])

    for batch in tqdm(loader):
        imgs = batch[0].to(DEVICE)

        if not useRetFoundPreprocessing:
            imgs = imgs * 0.5 + 0.5

        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)

        if not useRetFoundPreprocessing:
            imgs = norm(imgs)

        # Get ViT embeddings
        tokens = encoder.forward_features(imgs)

        # 1. Slice off CLS token and average the patches
        pooled_feats = tokens[:, 1:].mean(dim=1)

        # 2. Apply on-the-fly LayerNorm (Matches their exact DINOv2 code)
        ln = nn.LayerNorm(pooled_feats.shape[-1], eps=1e-6).to(DEVICE)
        raw_feats = ln(pooled_feats)

        # 3. Final L2 Normalize for Cosine Similarity distance
        # Note: original RETFOUND doesn't do that
        norm_feats = F.normalize(raw_feats, p=2, dim=1)
        features_list.append(norm_feats.cpu())

    return torch.cat(features_list, dim=0)


def find_neighbors_cross_set(query_features, ref_features, k=10, chunk_size=1000):
    """KNN from query set into a separate reference set (no self-distance masking)."""
    print(f"Finding {k} nearest neighbours in reference set of size {ref_features.shape[0]}...")
    ref = ref_features.to(DEVICE)
    all_indices = []

    for i in tqdm(range(0, query_features.shape[0], chunk_size)):
        query_chunk = query_features[i : i + chunk_size].to(DEVICE)
        dists = torch.cdist(query_chunk, ref)
        _, indices = torch.topk(dists, k=k, largest=False, dim=1)
        all_indices.append(indices.cpu())

    return torch.cat(all_indices, dim=0)


def find_neighbors_label_free(all_features, k=10, chunk_size=1000):
    print(f"Step 2: Finding {k} Nearest Neighbors (Label-Free!)...")
    N = all_features.shape[0]
    database_features = all_features.to(DEVICE)
    all_indices = []

    for i in tqdm(range(0, N, chunk_size)):
        end_idx = min(i + chunk_size, N)
        current_chunk_size = end_idx - i

        query_feats = database_features[i:end_idx]
        dists = torch.cdist(query_feats, database_features)

        # MASK SELF-DISTANCE ONLY
        row_indices = torch.arange(current_chunk_size, device=DEVICE)
        col_indices = torch.arange(i, end_idx, device=DEVICE)
        dists[row_indices, col_indices] = float("inf")

        _, indices = torch.topk(dists, k=k, largest=False, dim=1)
        all_indices.append(indices.cpu())

    return torch.cat(all_indices, dim=0)


def compare_with_original_retfoud():
    dataset = "MESSIDOR"
    useRetFoundPreprocessing = True

    if dataset == "EYEPACS":
        data_path = "/vol/biomedic3/awk24/datasets/EYEPACS_256/train"
        csv_path = "/vol/biomedic3/awk24/code/RETFound/Feature.csv"
        exact_dataset = EyepacsDataset(
            purpose="train",
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )
    else:
        data_path = "/vol/biomedic3/awk24/datasets/Messidor2_256/hidden_classifier_data"
        csv_path = "/vol/biomedic3/awk24/code/RETFound/hospital_b_hidden_dino_fixed.csv"
        exact_dataset = MessidorDataset(
            purpose="hidden",
            root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
            csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )

    # --- 1. Load the original RETFound features ---
    print(f"Loading reference features from: {csv_path}")
    df = pd.read_csv(csv_path)
    useRetFoundPreprocessing = True

    image_names = df["name"].values.tolist()
    reference_features = torch.tensor(df.drop(columns=["name"]).values, dtype=torch.float32)

    # --- 2. Load your custom timm encoder ---
    encoder = get_retfound_encoder()
    encoder.eval()

    exact_dataset.image_paths = [os.path.join(data_path, name) for name in image_names]
    exact_loader = DataLoader(exact_dataset, batch_size=10, shuffle=False)

    # --- 4. Extract features BYPASSING the transforms ---
    print("\nExtracting features using strict NumPy inputs...")
    custom_features = extract_all_features_domain_specific(
        exact_loader,
        encoder,
        useRetFoundPreprocessing=useRetFoundPreprocessing,
    )

    # --- 5. Compare the Tensors ---
    print("\n" + "=" * 40)
    print("STRICT NUMPY COMPARISON RESULTS:")

    diff = torch.abs(reference_features - custom_features)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Shape of Reference: {reference_features.shape}")
    print(f"Shape of Custom:    {custom_features.shape}")
    print(f"Max Absolute Error:  {max_diff:.8f}")
    print(f"Mean Absolute Error: {mean_diff:.8f}")

    # A tiny tolerance (1e-5) accounts for standard float32 GPU vs CPU rounding
    if torch.allclose(reference_features, custom_features, atol=1e-5):
        print("✅ SUCCESS! The timm model is a mathematically perfect match.")
    else:
        print("⚠️ NOTE: Differences still detected. Check your checkpoint path.")
    print("=" * 40 + "\n")

    # --- 3. Run k-NN on both sets ---
    TEST_K = 3
    print("\nCalculating Reference Neighbors...")
    ref_indices = find_neighbors_label_free(reference_features, k=TEST_K, chunk_size=10)

    print("Calculating Custom Neighbors...")
    custom_indices = find_neighbors_label_free(custom_features, k=TEST_K, chunk_size=10)

    # --- 4. Compare the Neighborhoods ---
    print("\n" + "=" * 50)
    print("NEAREST NEIGHBOR GEOMETRY TEST (k=3)")
    print("=" * 50)

    match_count = 0
    total_queries = len(image_names)

    for i in range(total_queries):
        ref_nns = ref_indices[i].tolist()
        cust_nns = custom_indices[i].tolist()

        if set(ref_nns) == set(cust_nns):
            match_count += 1
            status = "✅ MATCH"
        else:
            status = "❌ MISMATCH"

        print(f"Image {i:02d} | Ref NNs: {ref_nns} | Custom NNs: {cust_nns} | {status}")

    print("-" * 50)
    print(f"Total Matches: {match_count} / {total_queries}")

    if match_count == total_queries:
        print("CONCLUSION: The geometry is perfectly preserved!")
    else:
        print("CONCLUSION: The PyTorch preprocessing shifted the local neighborhoods.")
    print("=" * 50 + "\n")


def generate_DINO_embeddings(
    purpose,
    experiment="all",
    seed="seed_A",
    useRetFoundPreprocessing=False,
    savedir="",
):
    """Extract and save DINOv2 features + filenames for a Messidor split.

    Does NOT compute KNN — call find_neighbors_cross_set separately so KNN
    can be recomputed cheaply for different N values without re-running the
    expensive forward pass.
    """
    import json

    dataset = MessidorDataset(
        purpose=purpose,
        experiment=experiment,
        seed=seed,
        N=None,
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2",
        json_dir="/vol/biomedic3/awk24/datasets/Messidor2",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=224,
        useRetFoundPreprocessing=useRetFoundPreprocessing,
    )

    filenames = [os.path.basename(p) for p in dataset.image_paths]
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    encoder = get_retfound_encoder()
    all_features = extract_all_features_domain_specific(
        loader, encoder, useRetFoundPreprocessing=useRetFoundPreprocessing
    )

    os.makedirs(savedir, exist_ok=True)
    postfix = "_RetFoundPreprocessing" if useRetFoundPreprocessing else ""
    torch.save(all_features, os.path.join(savedir, f"features_dinov2{postfix}.pt"))
    with open(os.path.join(savedir, f"filenames_dinov2{postfix}.json"), "w") as f:
        json.dump(filenames, f)

    print(f"Saved {all_features.shape} features + {len(filenames)} filenames to {savedir}")
    return all_features, filenames


if __name__ == "__main__":
    # compare_with_original_retfoud()

    SPLIT = "hidden"
    useRetFoundPreprocessing = False
    datasetName = "EYEPACS"

    generate_DINO_embeddings(SPLIT, datasetName, useRetFoundPreprocessing)
