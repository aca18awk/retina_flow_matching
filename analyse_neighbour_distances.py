"""
For each of the four retrieval configurations, compute the mean L2 distance
from each anchor to its 2 nearest neighbours, broken down by the four
anchor×neighbour-type conditions.

Configurations:
  hospital_b            (N≈50,  unrestricted)
  hospital_b_100        (N≈100, unrestricted)
  hospital_b_200        (N≈200, unrestricted)
  hospital_b_200 (type) (N≈200, same-type-restricted)

Expected story: distances shrink as N grows (larger pool → closer neighbours);
type-restricted retrieval at N=200 should show *larger* distances than
unrestricted at N=200 because it sacrifices the closest cross-type matches.
"""

import os

import torch

ROOT_DIR   = "/vol/biomedic3/awk24/datasets/Messidor2_256"
EMBED_DIR  = "embeddings/Messidor"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

CONFIGS = [
    # (label,                    split,          feat_file,                      idx_file)
    ("hospital_b   (N≈50)",      "hospital_b",   "features_dinov2.pt",           "indices_dinov2.pt"),
    ("hospital_b_100 (N≈100)",   "hospital_b_100","features_dinov2.pt",           "indices_dinov2.pt"),
    ("hospital_b_200 (N≈200)",   "hospital_b_200","features_dinov2.pt",           "indices_dinov2.pt"),
    ("hospital_b_200 type-restr","hospital_b_200","features_dinov2_type_restricted.pt","indices_dinov2_type_restricted.pt"),
]


def build_subtypes(split: str) -> list[str]:
    """Return sorted list of 'PNG'/'JPG' labels matching the saved feature order."""
    data_dir = os.path.join(ROOT_DIR, split)
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    return ["JPG" if os.path.splitext(p)[1].lower() in {".jpg", ".jpeg"} else "PNG"
            for p in paths]


def mean_or_nan(values: list[float]) -> str:
    if not values:
        return "   n/a  "
    return f"{sum(values)/len(values):.4f} (n={len(values):>4})"


def analyse(label: str, split: str, feat_file: str, idx_file: str) -> None:
    features = torch.load(os.path.join(EMBED_DIR, split, feat_file), map_location="cpu")
    indices  = torch.load(os.path.join(EMBED_DIR, split, idx_file),  map_location="cpu")
    subtypes = build_subtypes(split)

    n = features.shape[0]
    assert len(subtypes) == n == indices.shape[0], "Dimension mismatch"

    dists_by_cond: dict[str, list[float]] = {
        "png_clean": [], "jpg_clean": [], "png_cross": [], "jpg_cross": []
    }

    for i in range(n):
        anchor_type = subtypes[i]
        nn_idxs     = indices[i].tolist()
        nn_types    = [subtypes[j] for j in nn_idxs]
        has_cross   = any(t != anchor_type for t in nn_types)

        # mean L2 distance to the k neighbours
        anchor_feat = features[i].unsqueeze(0)           # (1, D)
        nn_feats    = features[nn_idxs]                  # (k, D)
        mean_dist   = torch.norm(anchor_feat - nn_feats, dim=1).mean().item()

        if anchor_type == "PNG":
            key = "png_cross" if has_cross else "png_clean"
        else:
            key = "jpg_cross" if has_cross else "jpg_clean"

        dists_by_cond[key].append(mean_dist)

    print(f"\n  {label}")
    print(f"  {'─' * 56}")
    rows = [
        ("Anchor PNG, both nbrs PNG  (clean)", "png_clean"),
        ("Anchor JPG, both nbrs JPG  (clean)", "jpg_clean"),
        ("Anchor PNG, ≥1 nbr  JPG   (cross)",  "png_cross"),
        ("Anchor JPG, ≥1 nbr  PNG   (cross)",  "jpg_cross"),
    ]
    for desc, key in rows:
        print(f"    {desc:<38}  {mean_or_nan(dists_by_cond[key])}")


def main() -> None:
    print("=" * 66)
    print("Mean L2 distance: anchor → 2 nearest neighbours, by condition")
    print("=" * 66)
    for cfg in CONFIGS:
        analyse(*cfg)
    print()


if __name__ == "__main__":
    main()
