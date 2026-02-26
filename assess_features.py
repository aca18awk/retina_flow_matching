import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from Eyepacs_class_for_distance import EyepacsDataset
from Messidor_class import MessidorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.distributions import Exponential
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = "embeddings"
# Set to True to train on the current split (e.g., 'hidden') and test on 'test'
# Set to False to just do a standard 80/20 train_test_split on the current split
USE_EXPLICIT_MESSIDOR_TEST = True

# --- BARYCENTRIC AUGMENTATION SETTINGS ---
APPLY_AUGMENTATION = True
K_NEIGHBORS = 2
N_SAMPLES_PER_ANCHOR = 20

IS_BINARY = False
BORDER_CLASS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(dataset_name, split):
    """Helper to load the correct dataset just for grabbing labels."""
    # We use img_size=128 and useRetFoundPreprocessing=False just for speed,
    # because we are throwing away the images and ONLY keeping the labels.
    if dataset_name.upper() == "EYEPACS":
        return EyepacsDataset(
            purpose=split,
            img_size=128,
        )
    else:
        return MessidorDataset(
            purpose=split,
            root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
            csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
            img_size=128,
            useRetFoundPreprocessing=False,
        )


def get_true_labels(dataset):
    """Quickly loop through the dataset to extract the true DR grades."""
    print(f"Fetching ground truth labels for split '{dataset.purpose}'...")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    labels_list = []
    for batch in tqdm(loader):
        labels = batch[1]  # Label is always the second item
        labels_list.append(labels.cpu())

    return torch.cat(labels_list, dim=0).numpy()


def generate_barycentric_samples(X, y, k=2, n_samples=20):
    """
    Takes Real Features (X) and Labels (y) and generates Synthetic vectors
    using Barycentric interpolation between same-class nearest neighbors.
    """
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long).to(DEVICE)

    # Calculate Distance Matrix & Masking inline
    dist = torch.cdist(X_t, X_t)
    label_match_mask = y_t.unsqueeze(0) == y_t.unsqueeze(1)
    dist = dist.masked_fill(~label_match_mask, float("inf"))

    # Find K_NEIGHBORS + 1 (since self is included at distance 0)
    actual_k = min(k + 1, X_t.shape[0])
    dists, indices = torch.topk(dist, k=actual_k, largest=False)

    z_gathered = X_t[indices]
    valid_masks = (dists != float("inf")).float().unsqueeze(-1)

    synthetic_vectors = []
    synthetic_labels = []

    for anchor_idx in range(len(X_t)):
        anchor_feats = z_gathered[anchor_idx]
        anchor_mask = valid_masks[anchor_idx].squeeze(-1)
        anchor_label = y[anchor_idx]

        # Barycentric generation sampling math
        raw_weights = (
            Exponential(torch.tensor(1.0)).sample((n_samples, anchor_feats.shape[0])).to(DEVICE)
        )
        masked_weights = raw_weights * anchor_mask.unsqueeze(0)
        weight_sum = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        final_weights = masked_weights / weight_sum

        # The generated synthetic condition vectors
        cond_batch = (final_weights.unsqueeze(-1) * anchor_feats.unsqueeze(0)).sum(dim=1)

        synthetic_vectors.append(cond_batch.cpu())
        synthetic_labels.extend([anchor_label] * n_samples)

    X_syn = torch.cat(synthetic_vectors, dim=0).numpy()
    y_syn = np.array(synthetic_labels)

    # Combine Real and Synthetic
    X_aug = np.concatenate([X, X_syn], axis=0)
    y_aug = np.concatenate([y, y_syn], axis=0)

    return X_aug, y_aug


def evaluate_features(
    name,
    X,
    y_valid,
    X_test_explicit=None,
    y_test_explicit=None,
    do_tsne=False,
    do_umap=False,
    augment=False,
    is_binary=False,
):
    print("\n" + "=" * 60)
    print(f"--- Evaluating {name} ---")
    print("=" * 60)

    # 1. Process Training Data
    valid_idx = y_valid != -1
    min_len = min(len(X), len(valid_idx))
    X_train_clean = X[:min_len][valid_idx[:min_len]]
    y_train_clean = y_valid[:min_len][valid_idx[:min_len]]

    # 2. Determine Train/Test Split
    if X_test_explicit is not None and y_test_explicit is not None:
        print("-> Using EXPLICIT test set.")
        test_valid_idx = y_test_explicit != -1
        min_len_test = min(len(X_test_explicit), len(test_valid_idx))

        X_train, y_train = X_train_clean, y_train_clean
        X_test = X_test_explicit[:min_len_test][test_valid_idx[:min_len_test]]
        y_test = y_test_explicit[:min_len_test][test_valid_idx[:min_len_test]]
    else:
        print("-> Using 80/20 train_test_split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_clean, y_train_clean, test_size=0.2, random_state=42
        )

    # --- MINIMAL CHANGE: BINARIZE LABELS ---
    # 0, 1 -> 0 (Non-Referable)
    # 2, 3, 4 -> 1 (Referable)
    if is_binary:
        y_train = (y_train >= BORDER_CLASS).astype(int)
        y_test = (y_test >= BORDER_CLASS).astype(int)

    # --- INJECT BARYCENTRIC AUGMENTATION ---
    if augment:
        print(
            f"\n-> Applying Barycentric Augmentation (K={K_NEIGHBORS}, Samples/Anchor={N_SAMPLES_PER_ANCHOR})..."
        )
        orig_len = len(X_train)
        X_train, y_train = generate_barycentric_samples(
            X_train, y_train, k=K_NEIGHBORS, n_samples=N_SAMPLES_PER_ANCHOR
        )
        print(
            f"-> Augmented Training Set: {len(X_train)} vectors ({orig_len} Real + {len(X_train) - orig_len} Synthetic)"
        )

    # ---------------------------------------------------------
    # Linear Probe: BALANCED CLASS WEIGHTS
    # ---------------------------------------------------------
    print("\nTraining Linear Probe (Balanced Weights)...")
    clf_balanced = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    clf_balanced.fit(X_train, y_train)
    preds_balanced = clf_balanced.predict(X_test)
    acc_balanced = accuracy_score(y_test, preds_balanced)

    print(f"-> Balanced Accuracy: {acc_balanced * 100:.2f}%")
    if is_binary:
        preds_proba = clf_balanced.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds_proba)
        score_text = f"AUC: {auc:.3f}"
        print(f"-> ROC AUC Score: {auc:.4f}  <-- THE MOST IMPORTANT METRIC")
        print("\nDetailed Binary Classification Report (Balanced Model):")
        print(
            classification_report(
                y_test,
                preds_balanced,
                target_names=["Non-Referable (0,1)", "Referable (2,3,4)"],
                zero_division=0,
            )
        )
    else:
        qwk = cohen_kappa_score(y_test, preds_balanced, weights="quadratic")
        score_text = f"QWK: {qwk:.3f}"
        print(f"-> Quadratic Weighted Kappa (QWK): {qwk:.4f}  <-- THE MOST IMPORTANT METRIC")
        print("\nDetailed Classification Report (Balanced Model):")
        print(classification_report(y_test, preds_balanced, zero_division=0))

    # ---------------------------------------------------------
    # 3. UMAP / t-SNE Visualization
    # ---------------------------------------------------------
    if do_umap:
        print("\nComputing UMAP (This may take a minute)...")
        subset = min(5000, len(X_train_clean))
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_train_clean[:subset])

        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=y_train_clean[:subset],
            palette="flare",
            s=15,
            alpha=0.8,
            linewidth=0,
        )
        plt.title(f"{name}\n{score_text} | Balanced Acc: {acc_balanced * 100:.1f}%")
        plt.legend(title="DR Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"umap_{name}.png", dpi=300)
        print(f"Saved UMAP plot to umap_{name}.png\n")

    if do_tsne:
        print("Computing t-SNE...")
        subset = min(5000, len(X_train_clean))
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
            X_train_clean[:subset]
        )

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=y_train_clean[:subset],
            palette="viridis",
            s=15,
            alpha=0.8,
        )
        plt.title(f"{name}\n{score_text}")
        plt.legend(title="DR Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"tsne_{name}.png", dpi=300)
        print(f"Saved t-SNE plot to tsne_{name}.png\n")


if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(
            f"Base directory '{BASE_DIR}' not found. Are you in the right folder?"
        )

    # Caching labels so we don't reload the dataset for every .pt file
    cached_labels = {}

    # Find all dataset folders inside 'embeddings/'
    datasets = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    # datasets = ["Messidor"]

    for dataset_name in datasets:
        dataset_dir = os.path.join(BASE_DIR, dataset_name)
        splits = [
            s for s in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, s))
        ]
        # splits = ["hospital_b"]

        for split in splits:
            split_dir = os.path.join(dataset_dir, split)

            # Find ONLY feature files (ignore indices.pt files)
            pt_files = glob.glob(os.path.join(split_dir, "*features*.pt"))
            if not pt_files:
                continue

            print(f"\n[{dataset_name} | {split}] Found {len(pt_files)} feature files.")

            # --- 1. Load Current Split Labels ---
            cache_key_train = f"{dataset_name}_{split}"
            if cache_key_train not in cached_labels:
                dataset_obj = get_dataset(dataset_name, split)
                cached_labels[cache_key_train] = get_true_labels(dataset_obj)
            y_train_full = cached_labels[cache_key_train]

            # --- 2. Load Explicit Test Labels (If requested for Messidor) ---
            y_test_full = None
            if dataset_name == "Messidor" and USE_EXPLICIT_MESSIDOR_TEST and split != "test":
                cache_key_test = "Messidor_test"
                if cache_key_test not in cached_labels:
                    test_dataset_obj = get_dataset("Messidor", "test")
                    cached_labels[cache_key_test] = get_true_labels(test_dataset_obj)
                y_test_full = cached_labels[cache_key_test]

            # --- 3. Evaluate Every File ---
            for pt_file in pt_files:
                name = os.path.basename(pt_file).replace(".pt", "")

                X_train = torch.load(pt_file, map_location="cpu").numpy()
                X_test = None
                y_test = None

                if y_test_full is not None:
                    test_pt_file = pt_file.replace(f"/{split}/", "/test/").replace(
                        f"_{split}_", "_test_"
                    )
                    if os.path.exists(test_pt_file):
                        X_test = torch.load(test_pt_file, map_location="cpu").numpy()
                        y_test = y_test_full
                    else:
                        print(
                            f"\n⚠️ WARNING: Requested explicit test split, but {test_pt_file} is missing!"
                        )
                        print("Falling back to standard 80/20 train_test_split for this file.")

                evaluate_features(
                    name=name,
                    X=X_train,
                    y_valid=y_train_full,
                    X_test_explicit=X_test,
                    y_test_explicit=y_test,
                    do_umap=False,
                    do_tsne=False,
                    augment=APPLY_AUGMENTATION,  # <--- Triggers the Barycentric Logic
                    is_binary=IS_BINARY,
                )
