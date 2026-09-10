"""
Compute same-class self-KNN within a reference subset of size N.

Requires embeddings from generate_messidor_embeddings.py to exist first.

Usage:
    python compute_reference_knn.py --experiment all --seed seed_A --N 50

Output:
    embeddings_30_May/Messidor/{experiment}/{seed}/N{N}/indices_dinov2.pt
    embeddings_30_May/Messidor/{experiment}/{seed}/N{N}/ref_filenames_dinov2.json
"""

import argparse
import json
import os

import pandas as pd
import torch

BASE_DIR = "embeddings_30_May/Messidor"
JSON_DIR = "/vol/biomedic3/awk24/datasets/Messidor2"
CSV_PATH = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
K = 10

EXPERIMENT_TO_JSON = {
    "all":       "messidor2_splits_all.json",
    "dilated":   "messidor2_splits_dilated.json",
    "nondilated":"messidor2_splits_nondilated.json",
}

_CLASSES = ["G0", "G1", "G2", "G3", "G4"]


def select_reference_filenames(orderings, seed, N):
    seed_data = orderings[seed]
    if N is None:
        return [f for g in _CLASSES for f in seed_data.get(g, [])]
    per_class = N // len(_CLASSES)
    selected, surplus = [], {g: [] for g in _CLASSES}
    for g in _CLASSES:
        files = seed_data.get(g, [])
        take = min(per_class, len(files))
        selected.extend(files[:take])
        if len(files) > take:
            surplus[g] = files[take:]
    remaining = N - len(selected)
    for g in _CLASSES:
        if remaining <= 0:
            break
        if surplus[g]:
            extras = surplus[g][:remaining]
            selected.extend(extras)
            remaining -= len(extras)
    return selected


def compute_knn_same_class(features, labels, k):
    """Self-KNN with same-class constraint, no GPU needed for small N."""
    dist = torch.cdist(features, features)
    dist.fill_diagonal_(float("inf"))
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    dist = dist.masked_fill(~same_class, float("inf"))
    actual_k = min(k, features.shape[0] - 1)
    _, indices = torch.topk(dist, k=actual_k, largest=False, dim=1)
    return indices


EXPERIMENTS = ["all", "dilated", "nondilated"]
SEEDS       = ["seed_A", "seed_B", "seed_C"]
N_VALUES    = [20, 50, 100, 200, 300, 500]

# Labels shared across all combinations
df = pd.read_csv(CSV_PATH)
label_map = dict(zip(df["id_code"].str.strip(), df["diagnosis"]))

if __name__ == "__main__":
    for exp in EXPERIMENTS:
        # Load full embeddings once per experiment (shared across seeds)
        ref_dir = os.path.join(BASE_DIR, exp, "reference")
        features_full = torch.load(os.path.join(ref_dir, "features_dinov2.pt"), map_location="cpu")
        with open(os.path.join(ref_dir, "filenames_dinov2.json")) as f:
            filenames_full = json.load(f)
        filename_to_idx = {fn: i for i, fn in enumerate(filenames_full)}

        with open(os.path.join(JSON_DIR, EXPERIMENT_TO_JSON[exp])) as f:
            splits = json.load(f)

        for seed in SEEDS:
            for N in N_VALUES:
                out_dir  = os.path.join(BASE_DIR, exp, seed, f"N{N}")
                idx_path = os.path.join(out_dir, "indices_dinov2.pt")

                if os.path.exists(idx_path):
                    print(f"Skipping {exp}/{seed}/N{N} (already exists)")
                    continue

                ref_N_filenames = select_reference_filenames(
                    splits["hospital_b_orderings"], seed, N
                )
                valid = [(fn, filename_to_idx[fn]) for fn in ref_N_filenames
                         if fn in filename_to_idx]
                if len(valid) < N:
                    print(f"  Warning: only {len(valid)}/{N} files found in embeddings.")

                ref_N_filenames_found = [fn for fn, _ in valid]
                features_N = features_full[[i for _, i in valid]]
                labels_N = torch.tensor(
                    [label_map.get(os.path.splitext(fn)[0], label_map.get(fn, -1))
                     for fn in ref_N_filenames_found],
                    dtype=torch.long,
                )

                print(f"\n{'='*60}")
                print(f"KNN | experiment={exp}, seed={seed}, N={N}")
                print(f"Label distribution: { {int(l): int((labels_N==l).sum()) for l in labels_N.unique()} }")

                indices = compute_knn_same_class(features_N, labels_N, k=K)

                os.makedirs(out_dir, exist_ok=True)
                torch.save(indices, idx_path)
                with open(os.path.join(out_dir, "ref_filenames_dinov2.json"), "w") as f:
                    json.dump(ref_N_filenames_found, f)

                print(f"  Saved indices {indices.shape} → {out_dir}")

    print("\nAll done.")
