"""
Compute LPIPS scores for generated cohorts.

For each anchor, computes:
  - fidelity: mean LPIPS(generated_i, original_anchor)
  - diversity: mean pairwise LPIPS across all pairs of generated images

Compares results_10_June_deterministic_centroid vs results_30_May,
for common (experiment, N) combinations.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GS_K = "GS1.5_K2"
EXPERIMENTS = ["all", "dilated", "nondilated"]
SEEDS = ["seed_A", "seed_B", "seed_C"]
# Only N values present in both dirs
COMMON_NS = [20, 50, 100]

BASE = Path("/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images")
DIRS = {
    "june_deterministic": BASE / "results_10_June_deterministic_centroid",
    "may_barycentric":    BASE / "results_30_May",
}


def load_img(path: Path) -> torch.Tensor:
    """Load image as [1, 3, H, W] in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)          # [3, H, W] in [0, 1]
    return (t * 2 - 1).unsqueeze(0).to(DEVICE)


def compute_lpips_for_anchor(loss_fn, samples_dir: Path, original_path: Path):
    """Return (mean_fidelity, mean_diversity) or (None, None) if <2 samples."""
    sample_paths = sorted(samples_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(sample_paths) == 0:
        return None, None

    original = load_img(original_path)

    fidelity_scores = []
    generated_tensors = []
    for sp in sample_paths:
        gen = load_img(sp)
        generated_tensors.append(gen)
        with torch.no_grad():
            fidelity_scores.append(loss_fn(gen, original).item())

    mean_fidelity = float(np.mean(fidelity_scores))

    diversity_scores = []
    if len(generated_tensors) >= 2:
        # subsample to at most 50 images to keep runtime reasonable
        tensors = generated_tensors[:50]
        for i, j in combinations(range(len(tensors)), 2):
            with torch.no_grad():
                diversity_scores.append(loss_fn(tensors[i], tensors[j]).item())
        mean_diversity = float(np.mean(diversity_scores))
    else:
        mean_diversity = None

    return mean_fidelity, mean_diversity


def main():
    loss_fn = lpips.LPIPS(net="alex").to(DEVICE)
    loss_fn.eval()

    rows = []

    for label, base_dir in DIRS.items():
        if not base_dir.exists():
            print(f"Skipping {label}: {base_dir} not found")
            continue
        print(f"\n=== {label} ===")

        for exp in EXPERIMENTS:
            for seed in SEEDS:
                for N in COMMON_NS:
                    leaf = base_dir / exp / seed / f"N{N}" / GS_K
                    samples_root = leaf / "samples"
                    summary_root = leaf / "summary"

                    if not samples_root.exists():
                        continue

                    anchor_dirs = sorted(samples_root.iterdir())
                    fidelities, diversities = [], []

                    for anchor_dir in tqdm(
                        anchor_dirs,
                        desc=f"{label} {exp} {seed} N{N}",
                        leave=False,
                    ):
                        orig_path = summary_root / anchor_dir.name / "original.png"
                        if not orig_path.exists():
                            continue
                        f, d = compute_lpips_for_anchor(loss_fn, anchor_dir, orig_path)
                        if f is not None:
                            fidelities.append(f)
                        if d is not None:
                            diversities.append(d)

                    if fidelities:
                        rows.append({
                            "run":          label,
                            "experiment":   exp,
                            "seed":         seed,
                            "N":            N,
                            "n_anchors":    len(fidelities),
                            "lpips_fidelity_mean":   np.mean(fidelities),
                            "lpips_fidelity_std":    np.std(fidelities),
                            "lpips_diversity_mean":  np.mean(diversities) if diversities else None,
                            "lpips_diversity_std":   np.std(diversities)  if diversities else None,
                        })

    df = pd.DataFrame(rows)
    out_csv = BASE / "lpips_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results → {out_csv}")

    # --- Summary: aggregate over seeds ---
    agg = (
        df.groupby(["run", "experiment", "N"])
        .agg(
            lpips_fidelity_mean=("lpips_fidelity_mean", "mean"),
            lpips_fidelity_std=("lpips_fidelity_mean", "std"),
            lpips_diversity_mean=("lpips_diversity_mean", "mean"),
            lpips_diversity_std=("lpips_diversity_mean", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )

    agg_csv = BASE / "lpips_comparison_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregated results → {agg_csv}\n")

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)
    print(agg.sort_values(["experiment", "N", "run"]).to_string(index=False))


if __name__ == "__main__":
    main()
