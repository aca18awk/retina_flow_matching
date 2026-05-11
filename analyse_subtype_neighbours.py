"""
Analyse nearest-neighbour subtype mixing for Messidor hospital_b splits.

For each split the script reports:
  1. How many main images are PNG vs JPG (sub-dataset breakdown)
  2. How many cases have a MIXED neighbourhood, i.e. the main image and/or
     at least one of its 2 nearest neighbours come from different sub-datasets.
  3. For every mixed case: image filename, its diagnosis, neighbour filenames
     and their diagnoses.

The image_paths list is reconstructed with the same logic as MessidorDataset
(os.walk → sort), so indices in indices_dinov2.pt map correctly.
"""

import os

import pandas as pd
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
SPLITS = ["hospital_b", "hospital_b_100", "hospital_b_200"]
ROOT_DIR = "/vol/biomedic3/awk24/datasets/Messidor2_256"
CSV_PATH = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
EMBEDDINGS_DIR = "embeddings/Messidor"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


def build_image_paths(split: str) -> list[str]:
    """Reproduce MessidorDataset's image_paths list (os.walk + sort)."""
    subfolder_map = {
        "hospital_b": "hospital_b",
        "hospital_b_100": "hospital_b_100",
        "hospital_b_200": "hospital_b_200",
    }
    data_dir = os.path.join(ROOT_DIR, subfolder_map[split])
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def subtype(filename: str) -> str:
    """Return 'JPG' or 'PNG' based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    return "JPG" if ext in {".jpg", ".jpeg"} else "PNG"


def load_labels(csv_path: str) -> dict[str, int]:
    """Return {id_code: diagnosis} mapping from CSV."""
    df = pd.read_csv(csv_path)
    return {str(row["id_code"]).strip(): int(row["diagnosis"]) for _, row in df.iterrows()}


def get_diagnosis(filename: str, labels: dict[str, int]) -> int:
    """Look up diagnosis by filename or stem."""
    if filename in labels:
        return labels[filename]
    stem = os.path.splitext(filename)[0]
    return labels.get(stem, -1)


def analyse_split(split: str, labels: dict[str, int]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Split: {split}")
    print(f"{'=' * 60}")

    # 1. Image paths (same order as during embedding)
    image_paths = build_image_paths(split)
    n = len(image_paths)
    filenames = [os.path.basename(p) for p in image_paths]
    subtypes = [subtype(f) for f in filenames]

    # 2. Load neighbour indices  shape: (N, K)
    idx_path = os.path.join(EMBEDDINGS_DIR, split, "indices_dinov2.pt")
    indices = torch.load(idx_path, map_location="cpu")

    k = indices.shape[1]
    assert indices.shape[0] == n, (
        f"Index count {indices.shape[0]} != image count {n}"
    )

    # 3. Sub-dataset counts for main images
    png_count = subtypes.count("PNG")
    jpg_count = subtypes.count("JPG")
    print(f"\nTotal images : {n}")
    print(f"  PNG (20051…_PP.png) : {png_count}")
    print(f"  JPG (IM….JPG)       : {jpg_count}")

    # 4. Categorise every anchor by (anchor_type, neighbour_types)
    counts = {
        "png_clean": 0,   # anchor PNG, all neighbours PNG
        "jpg_clean": 0,   # anchor JPG, all neighbours JPG
        "png_cross": 0,   # anchor PNG, ≥1 neighbour JPG
        "jpg_cross": 0,   # anchor JPG, ≥1 neighbour PNG
    }

    for i in range(n):
        nn_indices = indices[i].tolist()
        anchor = subtypes[i]
        nn_types = [subtypes[j] for j in nn_indices]
        has_opposite = any(t != anchor for t in nn_types)

        if anchor == "PNG":
            counts["png_cross" if has_opposite else "png_clean"] += 1
        else:
            counts["jpg_cross" if has_opposite else "jpg_clean"] += 1

    print(f"\n{'Category':<45} {'Count':>6}  {'/ total':>8}")
    print(f"{'─' * 62}")
    rows = [
        ("Anchor PNG, both neighbours PNG  (clean)",  counts["png_clean"]),
        ("Anchor JPG, both neighbours JPG  (clean)",  counts["jpg_clean"]),
        ("Anchor PNG, ≥1 neighbour JPG     (cross)",  counts["png_cross"]),
        ("Anchor JPG, ≥1 neighbour PNG     (cross)",  counts["jpg_cross"]),
    ]
    for label, c in rows:
        print(f"  {label:<43} {c:>6}  ({c/n*100:5.1f} %)")
    cross_total = counts["png_cross"] + counts["jpg_cross"]
    print(f"{'─' * 62}")
    print(f"  {'Total cross-protocol':43} {cross_total:>6}  ({cross_total/n*100:5.1f} %)")


def main() -> None:
    labels = load_labels(CSV_PATH)
    for split in SPLITS:
        analyse_split(split, labels)
    print()


if __name__ == "__main__":
    main()
