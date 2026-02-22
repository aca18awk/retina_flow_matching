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

    # FIX 1: DINOv2 Architecture, NO global_pool argument here
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
        # RETFOUND DOESNT DO THAT!
        norm_feats = F.normalize(raw_feats, p=2, dim=1)
        features_list.append(norm_feats.cpu())

    return torch.cat(features_list, dim=0)


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


if __name__ == "__main__":
    SPLIT = "validation"
    useRetFoundPreprocessing = True
    dataset = "EYEPACS"

    if dataset == "EYEPACS":
        dataset = EyepacsDataset(
            purpose=SPLIT,
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )
    else:
        dataset = MessidorDataset(
            purpose="hospital_b",
            root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
            csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
            img_size=224,
            useRetFoundPreprocessing=useRetFoundPreprocessing,
        )
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    encoder = get_retfound_encoder()
    all_features = extract_all_features_domain_specific(
        loader,
        encoder,
        useRetFoundPreprocessing=useRetFoundPreprocessing,
    )

    neighbor_indices = find_neighbors_label_free(
        all_features, k=K_NEIGHBORS, chunk_size=CHUNK_SIZE
    )

    print(f"Saving {all_features.shape} features and indices...")
    torch.save(all_features, f"{dataset}_{SPLIT}_features_RETFOUND_dinov2.pt")
    torch.save(neighbor_indices, f"{dataset}_{SPLIT}_indices_RETFOUND_dinov2.pt")
    print("Done!")
