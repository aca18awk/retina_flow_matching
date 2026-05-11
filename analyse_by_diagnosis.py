"""
Per-diagnosis breakdown of neighbour subtype mixing and mean L2 distance,
across the four retrieval configurations.

For each diagnosis class and configuration reports:
  - n anchors in that class
  - % of anchors with ≥1 cross-protocol neighbour
  - mean L2 distance to the 2 nearest neighbours
"""

import os
from collections import defaultdict

import pandas as pd
import torch

ROOT_DIR   = "/vol/biomedic3/awk24/datasets/Messidor2_256"
EMBED_DIR  = "embeddings/Messidor"
CSV_PATH   = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

CONFIGS = [
    ("hospital_b    (N≈50)",      "hospital_b",    "features_dinov2.pt",                    "indices_dinov2.pt"),
    ("hospital_b_100 (N≈100)",    "hospital_b_100", "features_dinov2.pt",                   "indices_dinov2.pt"),
    ("hospital_b_200 (N≈200)",    "hospital_b_200", "features_dinov2.pt",                   "indices_dinov2.pt"),
    ("hospital_b_200 type-restr", "hospital_b_200", "features_dinov2_type_restricted.pt",   "indices_dinov2_type_restricted.pt"),
]


def build_image_meta(split: str, labels: dict) -> tuple[list[str], list[str], list[int]]:
    """Return (filenames, subtypes, diagnoses) sorted as MessidorDataset would."""
    data_dir = os.path.join(ROOT_DIR, split)
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()

    filenames, subtypes, diagnoses = [], [], []
    for p in paths:
        fname = os.path.basename(p)
        ext   = os.path.splitext(fname)[1].lower()
        filenames.append(fname)
        subtypes.append("JPG" if ext in {".jpg", ".jpeg"} else "PNG")

        diag = labels.get(fname, labels.get(os.path.splitext(fname)[0], -1))
        diagnoses.append(int(diag))

    return filenames, subtypes, diagnoses


def load_labels(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {str(row["id_code"]).strip(): row["diagnosis"] for _, row in df.iterrows()}


def analyse(label: str, split: str, feat_file: str, idx_file: str, labels: dict) -> None:
    features = torch.load(os.path.join(EMBED_DIR, split, feat_file), map_location="cpu")
    indices  = torch.load(os.path.join(EMBED_DIR, split, idx_file),  map_location="cpu")
    _, subtypes, diagnoses = build_image_meta(split, labels)

    n = features.shape[0]
    assert len(subtypes) == n == indices.shape[0]

    # per-diagnosis accumulators
    counts   = defaultdict(int)
    n_cross  = defaultdict(int)
    dist_sum = defaultdict(float)

    for i in range(n):
        diag        = diagnoses[i]
        anchor_type = subtypes[i]
        nn_idxs     = indices[i].tolist()
        nn_types    = [subtypes[j] for j in nn_idxs]
        is_cross    = any(t != anchor_type for t in nn_types)

        anchor_feat = features[i].unsqueeze(0)
        nn_feats    = features[nn_idxs]
        mean_dist   = torch.norm(anchor_feat - nn_feats, dim=1).mean().item()

        counts[diag]   += 1
        n_cross[diag]  += int(is_cross)
        dist_sum[diag] += mean_dist

    print(f"\n  {label}")
    print(f"  {'─' * 54}")
    print(f"  {'Diag':>5}  {'N':>5}  {'% cross':>8}  {'mean dist':>10}")
    print(f"  {'─' * 54}")

    for diag in sorted(counts):
        c     = counts[diag]
        cross = n_cross[diag]
        avg_d = dist_sum[diag] / c
        print(f"  {diag:>5}  {c:>5}  {cross/c*100:>7.1f}%  {avg_d:>10.4f}")

    total     = sum(counts.values())
    total_cross = sum(n_cross.values())
    total_dist  = sum(dist_sum.values()) / total
    print(f"  {'─' * 54}")
    print(f"  {'all':>5}  {total:>5}  {total_cross/total*100:>7.1f}%  {total_dist:>10.4f}")


def main() -> None:
    labels = load_labels(CSV_PATH)
    print("=" * 60)
    print("Per-diagnosis: % cross-protocol neighbours & mean L2 dist")
    print("=" * 60)
    for cfg in CONFIGS:
        analyse(*cfg, labels=labels)
    print()


if __name__ == "__main__":
    main()
