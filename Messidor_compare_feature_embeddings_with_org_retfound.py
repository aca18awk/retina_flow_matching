import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from Messidor_class import MessidorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# # --- CONFIGURATION ---
# FEATURE_FILES = {
#     "ImageNet_raw": "Eyepacs_vectors/Eyepacs_train_features.pt",
#     "ImageNet_Double_normalised": "Eyepacs_vectors/Eyepacs_train_features_2.pt",
#     "ImageNet_same_class": "Eyepacs_vectors/Eyepacs_train_features_class_restricted.pt",
#     "ImageNet_correct": "Eyepacs_vectors/Eyepacs_train_features_ImageNet.pt",
#     "RETFOUND_Dinov2_preprocessed": "Eyepacs_vectors/Eyepacs_train_features_RETFOUND_dinov2_preprocessed.pt",
#     "RETFOUND_MAE": "Eyepacs_vectors/Eyepacs_train_features_RETFOUND_fixed_2.pt",
#     "RETFOUND_MAE_original_retfound": "Eyepacs_vectors/EYEPACS_train_features_RETFOUND_MAE_orgPreprocessing.pt",
# }


def evaluate_features(
    name,
    X,
    labels,
    do_tsne=False,
    do_umap=False,
):
    print("\n" + "=" * 50)
    print(f"--- Evaluating {name} ---")
    print("=" * 50)

    # Filter out missing labels (-1)
    valid_idx = labels != -1

    # Safety check in case the feature tensor and label tensor have a slight mismatch
    min_len = min(len(X), len(valid_idx))
    X = X[:min_len]
    valid_idx = valid_idx[:min_len]
    labels = labels[:min_len]

    X, y_valid = X[valid_idx], labels[valid_idx]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_valid, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 1. Linear Probe: NO CLASS WEIGHTS
    # ---------------------------------------------------------
    print("\nTraining Linear Probe (Unweighted)...")
    clf_unweighted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf_unweighted.fit(X_train, y_train)
    preds_unweighted = clf_unweighted.predict(X_test)
    acc_unweighted = accuracy_score(y_test, preds_unweighted)
    print(
        f"-> Unweighted Accuracy: {acc_unweighted * 100:.2f}% (Usually high due to majority Class 0)"
    )

    # ---------------------------------------------------------
    # 2. Linear Probe: BALANCED CLASS WEIGHTS
    # ---------------------------------------------------------
    print("\nTraining Linear Probe (Balanced Weights)...")
    clf_balanced = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    clf_balanced.fit(X_train, y_train)
    preds_balanced = clf_balanced.predict(X_test)
    acc_balanced = accuracy_score(y_test, preds_balanced)

    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(y_test, preds_balanced, weights="quadratic")

    print(f"-> Balanced Accuracy: {acc_balanced * 100:.2f}% (Expect this to be lower)")
    print(f"-> Quadratic Weighted Kappa (QWK): {qwk:.4f}  <-- THE MOST IMPORTANT METRIC")

    print("\nDetailed Classification Report (Balanced Model):")
    print(classification_report(y_test, preds_balanced, zero_division=0))

    # ---------------------------------------------------------
    # 3. UMAP Visualization
    # ---------------------------------------------------------
    if do_umap:
        print("\nComputing UMAP (This may take a minute)...")
        subset = min(5000, len(X))  # Keep it fast for visualization

        # UMAP preserves global structure better than t-SNE for disease progression
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X[:subset])

        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            x=X_umap[:, 0],
            y=X_umap[:, 1],
            hue=y_valid[:subset],
            palette="flare",  # "flare" or "magma" is great for sequential severity
            s=15,
            alpha=0.8,
            linewidth=0,
        )
        plt.title(
            f"{name} UMAP Projection\nQWK: {qwk:.3f} | Balanced Acc: {acc_balanced * 100:.1f}%"
        )
        plt.legend(title="DR Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_name = f"umap_{name}.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved UMAP plot to {save_name}\n")

    if do_tsne:
        # 2. t-SNE Plot
        print("Computing t-SNE...")
        subset = min(5000, len(X))  # Keep it fast
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X[:subset])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=y_valid[:subset],
            palette="viridis",
            s=15,
            alpha=0.8,
        )
        plt.title(f"{name}\nLinear Probe Accuracy: {acc_balanced * 100:.1f}%")
        plt.legend(title="DR Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        save_name = f"tsne_{name}.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved plot to {save_name}")


def get_true_labels(dataset):
    """Quickly loop through the dataset to extract true DR grades and names."""
    print("Fetching ground truth labels from dataset...")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    labels_list = []
    name_to_label = {}

    for batch in tqdm(loader):
        # Messidor yields: image, label, imagename
        labels = batch[1]
        names = batch[2]

        labels_list.append(labels.cpu())

        # Build the lookup dictionary
        for name, label in zip(names, labels):
            # Clean up the name (e.g., "/path/image1.jpg" -> "image1")
            clean_name = os.path.splitext(os.path.basename(name))[0]
            name_to_label[clean_name] = label.item()

    native_labels_array = torch.cat(labels_list, dim=0).numpy()
    return native_labels_array, name_to_label


if __name__ == "__main__":
    dataset = MessidorDataset(purpose="hidden", img_size=224)

    # Get both the native ordered labels and the lookup dictionary
    y_native, name_to_label = get_true_labels(dataset)

    # ---------------------------------------------------------
    # 1. Evaluate YOUR features (.pt file matches native order)
    # ---------------------------------------------------------
    our_features_path = "Messidor_hidden_features_RETFOUND_dinov2.pt"
    name_ours = "Our_dinoV2"

    print(f"\nLoading {our_features_path}...")
    X_ours = torch.load(our_features_path, map_location="cpu").numpy()
    evaluate_features(name_ours, X_ours, y_native)

    path2 = "Messidor_hidden_features_RETFOUND_dinov2_orgPreprocessing.pt"
    name_2 = "dinoV2 with RETFound preprocessing"

    print(f"\nLoading {path2}...")
    X_ours = torch.load(path2, map_location="cpu").numpy()
    evaluate_features(name_2, X_ours, y_native)

    # ---------------------------------------------------------
    # 2. Evaluate REFERENCE features (CSV order needs matching)
    # ---------------------------------------------------------
    csv_path = "/vol/biomedic3/awk24/code/RETFound/hospital_b_hidden_dino_fixed.csv"
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract features from CSV
    reference_features = torch.tensor(
        df.drop(columns=["name"]).values, dtype=torch.float32
    ).numpy()

    # Build the correctly ordered label array for the CSV
    y_csv = []
    for csv_name in df["name"]:
        clean_csv_name = os.path.splitext(os.path.basename(csv_name))[0]

        if clean_csv_name in name_to_label:
            y_csv.append(name_to_label[clean_csv_name])
        else:
            print(f"⚠️ WARNING: {csv_name} from CSV not found in Dataset. Labeling as -1.")
            y_csv.append(-1)

    y_csv = np.array(y_csv)

    evaluate_features("original_dinoV2", reference_features, y_csv)
