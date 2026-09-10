"""
Visualise the Messidor-2 dataset as a single 2-D UMAP projection of DINOv2
embeddings, coloured three ways on the same coordinates:

  1. Diagnosis           (G0–G4)
  2. Hospital + Dilation (colour = drops/no-drops; marker = which hospital)
  3. Image format        (PNG / JPG)

Run once on a GPU node; embeddings are cached so subsequent runs are fast.

    python visualise_messidor2_umap.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from distance_RetFound_DinoV2 import (
    extract_all_features_domain_specific,
    get_retfound_encoder,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = "/vol/biomedic3/awk24/datasets/Messidor2_256"
JSON_PATH = "/vol/biomedic3/awk24/datasets/Messidor2/messidor2_splits_all.json"
CSV_PATH = "/vol/biomedic3/awk24/datasets/Messidor2/messidor2_master.csv"
CACHE_DIR = "embeddings_30_May/Messidor/all_images"
FEAT_PATH = os.path.join(CACHE_DIR, "features_dinov2.pt")
FNAME_PATH = os.path.join(CACHE_DIR, "filenames_dinov2.json")
OUT_PATH = "umap_messidor2_3panel.png"

IMG_SIZE = 224
BATCH_SIZE = 32
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# ---------------------------------------------------------------------------
# Dataset — loads every image regardless of split
# ---------------------------------------------------------------------------

class AllMessidorDataset(Dataset):
    def __init__(self, root_dir: str, filenames: list[str]):
        # Build filename → absolute-path index; handles root-level files and
        # one level of subdirectories (same logic as MessidorDataset).
        file_index: dict[str, str] = {}
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                for fname in os.listdir(entry_path):
                    file_index[fname] = os.path.join(entry_path, fname)
            else:
                file_index[entry] = entry_path

        self.image_paths: list[str] = []
        self.filenames: list[str] = []
        for fname in filenames:
            if fname in file_index:
                self.image_paths.append(file_index[fname])
                self.filenames.append(fname)
            else:
                print(f"Warning: {fname} not found under {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize([IMG_SIZE, IMG_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print(f"AllMessidorDataset: {len(self.image_paths)} images loaded.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), 0, self.filenames[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all_filenames() -> list[str]:
    with open(JSON_PATH) as f:
        splits = json.load(f)
    filenames = splits["validation"] + splits["test"] + splits["reference_pool"]
    print(f"Total filenames across all splits: {len(filenames)}")
    return filenames


def get_or_create_embeddings(filenames: list[str]):
    if os.path.exists(FEAT_PATH) and os.path.exists(FNAME_PATH):
        print(f"Loading cached embeddings from {CACHE_DIR}")
        features = torch.load(FEAT_PATH, map_location="cpu")
        with open(FNAME_PATH) as f:
            cached_fnames = json.load(f)
        print(f"  shape={features.shape}, n_files={len(cached_fnames)}")
        return features, cached_fnames

    print("Generating embeddings (GPU required)…")
    dataset = AllMessidorDataset(ROOT_DIR, filenames)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    encoder = get_retfound_encoder()
    features = extract_all_features_domain_specific(
        loader, encoder, useRetFoundPreprocessing=False
    )
    cached_fnames = dataset.filenames

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(features, FEAT_PATH)
    with open(FNAME_PATH, "w") as f:
        json.dump(cached_fnames, f)
    print(f"Saved {features.shape} embeddings → {CACHE_DIR}")
    return features, cached_fnames


def align_metadata(cached_fnames: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(CSV_PATH).set_index("id_code")
    rows = []
    for fname in cached_fnames:
        if fname in meta.index:
            row = meta.loc[fname]
        else:
            base = os.path.splitext(fname)[0]
            row = meta.loc[base] if base in meta.index else None

        if row is not None:
            rows.append({
                "filename": fname,
                "diagnosis": int(row["diagnosis"]),
                "hospital": str(row["hospital"]),
                "dilation": int(row["with_eye_drops"]),
                "format": str(row["format"]),
            })
        else:
            print(f"Warning: no metadata for {fname}")
            rows.append({
                "filename": fname,
                "diagnosis": -1,
                "hospital": "Unknown",
                "dilation": -1,
                "format": "Unknown",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DIAG_PALETTE = {
    0: "#4393c3",
    1: "#74c476",
    2: "#fd8d3c",
    3: "#d6604d",
    4: "#8e0152",
}
DIAG_LABELS = {
    0: "G0 – No DR",
    1: "G1 – Mild",
    2: "G2 – Moderate",
    3: "G3 – Severe",
    4: "G4 – Proliferative",
}

# Hospital+dilation panel: colour = drops status, marker = which hospital.
# No-drops images all come from Brest; drops images from Lariboisière + Saint-Étienne.
_COLOR_NO_DROPS   = "#7b2d8b"   # purple
_COLOR_WITH_DROPS = "#f1a340"   # orange
_COLOR_WITH_DROPS_2 = "#cc4733"   # orange

ACQN_GROUPS = [
    # (label,                colour,            marker, dilation, hospital)
    ("Non-dilated – Brest",              _COLOR_NO_DROPS,   "o", 0, None),
    ("Dilated – Lariboisière",          _COLOR_WITH_DROPS, "o", 1, "Lariboisiere"),
    ("Dilated – Saint-Étienne",         _COLOR_WITH_DROPS_2, "^", 1, "Etienne"),
    # catch-all for any unexpected combinations
    ("Dilated – other",                 _COLOR_WITH_DROPS_2, "s", 1, None),
]

FMT_PALETTE = {"png": "#1b7837", "jpg": "#762a83"}
FMT_LABELS  = {"png": "PNG", "jpg": "JPG"}

_S = 8       # marker size
_A = 0.65   # alpha


def _panel_style(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    ax.legend(fontsize=11.5, markerscale=2.5, framealpha=0.8,
              handlelength=1.2, borderpad=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_diagnosis(ax, coords, df):
    for grade in sorted(DIAG_PALETTE):
        mask = df["diagnosis"].values == grade
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=DIAG_PALETTE[grade], label=f"{DIAG_LABELS[grade]} (n={mask.sum()})",
                   s=_S, alpha=_A, linewidths=0)
    _panel_style(ax, "Diagnosis (DR Grade)")


def plot_acquisition(ax, coords, df):
    """Colour by drops/no-drops; marker shape by hospital."""
    dil = df["dilation"].values
    hosp = df["hospital"].values

    for label, color, marker, drops, hospital in ACQN_GROUPS:
        if drops == 0:
            mask = dil == 0
        elif hospital is not None:
            mask = (dil == 1) & (hosp == hospital)
        else:
            # catch anything with drops not matched above
            known = {g[4] for g in ACQN_GROUPS if g[4] is not None}
            mask = (dil == 1) & ~np.isin(hosp, list(known))

        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, marker=marker, label=f"{label} (n={mask.sum()})",
                   s=_S, alpha=_A, linewidths=0)

    _panel_style(ax, "Acquisition (dilation · hospital)")


def plot_format(ax, coords, df):
    for fmt, color in FMT_PALETTE.items():
        mask = df["format"].values == fmt
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=f"{FMT_LABELS[fmt]} (n={mask.sum()})",
                   s=_S, alpha=_A, linewidths=0)
    _panel_style(ax, "Image Format")


def main():
    # 1. Embeddings
    filenames = load_all_filenames()
    features, cached_fnames = get_or_create_embeddings(filenames)

    # 2. Metadata
    df = align_metadata(cached_fnames)

    # 3. UMAP (fit once on all embeddings)
    print("Computing UMAP…")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(features.numpy())
    print(f"UMAP done → shape {coords.shape}")

    # 4. Three-panel figure (side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_diagnosis(axes[0], coords, df)
    plot_acquisition(axes[1], coords, df)
    plot_format(axes[2], coords, df)

    # fig.suptitle(
    #     "Messidor-2  ·  UMAP of DINOv2 embeddings  ·  1 744 images",
    #     fontsize=13, fontweight="bold", y=1.02,
    # )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
