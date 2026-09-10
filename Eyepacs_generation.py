import argparse
import json
import math
import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from Messidor_class import MessidorDataset
from PIL import Image
from torch.distributions import Exponential
from torch.utils.data import DataLoader
from torchdiffeq import odeint
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet import UNetModel

# --- Configuration ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BASE_EMBEDDINGS_DIR = "embeddings_30_May/Messidor"


def load_img_tensor(path, img_size):
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [img_size, img_size])
    return TF.to_tensor(img)  # [3, H, W] in [0, 1]


def generate_cohorts(model, val_loader, ref_features, ref_labels, ref_filenames, ref_image_paths, knn_indices, k_neighbors, n_samples_per_class, guidance_scale, savedir):
    model.eval()

    # Create Output Directories
    sample_folder_root = os.path.join(savedir, "samples")
    os.makedirs(sample_folder_root, exist_ok=True)

    summary_folder_root = os.path.join(savedir, "summary")
    os.makedirs(summary_folder_root, exist_ok=True)

    ref_features = ref_features.to(device)
    ref_labels = ref_labels.to(device)

    # Define ODE Solver Vector Field
    def vector_field(t, x, cond, null_cond):
        t_vector = torch.ones(x.shape[0], device=device) * t
        # Model forward pass
        v_cond = model(t_vector, x, y=cond)
        v_uncond = model(t_vector, x, y=null_cond)
        # Classifier-Free Guidance
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    csv_rows = []
    img_size = None  # inferred from first batch

    global_idx = 0
    for batch_data in val_loader:
        val_batch, val_labels, val_filenames = batch_data
        val_batch = val_batch.to(device)
        batch_size = val_batch.shape[0]
        img_size = val_batch.shape[-1]

        # Load pre-computed neighbour indices for this batch [B, K]
        batch_knn = knn_indices[global_idx : global_idx + batch_size]

        print(f"Loaded batch of size {batch_size}. Generating cohorts...")

        for idx in range(batch_size):
            anchor_idx = global_idx + idx
            anchor_filename = val_filenames[idx]
            anchor_label = val_labels[idx].item()

            # Create subfolders for this specific patient
            folder_name = f"{anchor_idx}_{os.path.splitext(anchor_filename)[0]}"

            sample_folder = os.path.join(sample_folder_root, folder_name)
            os.makedirs(sample_folder, exist_ok=True)

            summary_folder = os.path.join(summary_folder_root, folder_name)
            os.makedirs(summary_folder, exist_ok=True)

            # Same-label constraint: keep only neighbours with matching diagnosis, limit to k_neighbors
            neighbour_indices = batch_knn[idx]
            same_label_mask = (ref_labels[neighbour_indices] == anchor_label)
            valid_indices = neighbour_indices[same_label_mask][:k_neighbors]

            # Triangle: anchor's own feature + up to K same-label neighbours
            anchor_own_feat = ref_features[anchor_idx].unsqueeze(0)  # [1, d]  (already on device)
            neighbour_feats = ref_features[valid_indices]             # [K_valid, d]
            anchor_feats = torch.cat([anchor_own_feat, neighbour_feats], dim=0)    # [1+K_valid, d]

            # Mask for barycentric sampling (all entries valid since we already filtered)
            anchor_mask = torch.ones(anchor_feats.shape[0], device=device)

            n_samples = n_samples_per_class[anchor_label]

            # --- SAVING ---

            # 1. Save Original Anchor (The "Patient Zero")
            original_img = (val_batch[idx] + 1) / 2
            save_image(original_img, os.path.join(summary_folder, "original.png"))

            # 2. Save neighbour reference images + neighbour summary grid
            neighbour_fns = [ref_filenames[vi.item()] for vi in valid_indices]
            neighbour_grid_imgs = [original_img.cpu()]
            for vi in valid_indices:
                path = ref_image_paths[vi.item()]
                if path is not None:
                    neighbour_grid_imgs.append(load_img_tensor(path, img_size))
            save_image(
                make_grid(torch.stack(neighbour_grid_imgs), nrow=len(neighbour_grid_imgs), padding=2),
                os.path.join(summary_folder, "neighbours_reference.png"),
            )

            if n_samples > 0:
                # --- Barycentric Sampling ---
                # Sample weights from Dirichlet/Exponential
                raw_weights = (
                    Exponential(torch.tensor(1.0)).sample(torch.Size([n_samples, anchor_feats.shape[0]])).to(device)
                )
                masked_weights = raw_weights * anchor_mask.unsqueeze(0)
                weight_sum = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                final_weights = masked_weights / weight_sum

                # Calculate condition vector
                cond_batch = (final_weights.unsqueeze(-1) * anchor_feats.unsqueeze(0)).sum(dim=1)
                null_cond_batch = torch.zeros_like(cond_batch)

                # Generate Noise
                x0 = torch.randn(n_samples, *val_batch.shape[1:], device=device)

                with torch.no_grad():
                    traj = odeint(
                        lambda t, x: vector_field(t, x, cond_batch, null_cond_batch),
                        x0,
                        torch.tensor([0.0, 1.0], device=device),
                        atol=1e-4,
                        rtol=1e-4,
                        method="dopri5",
                    )

                generated_imgs = traj[-1].clip(-1, 1)  # type: ignore[union-attr]
                generated_imgs_denorm = torch.clamp((generated_imgs + 1) / 2, 0, 1)

                for i in range(n_samples):
                    save_image(generated_imgs_denorm[i], os.path.join(sample_folder, f"{i}.png"))

                # 3. Save Cohort Grid (The synthetic images)
                grid_img = make_grid(generated_imgs_denorm, nrow=10, padding=2, normalize=False)
                save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

            # 4. Accumulate CSV row
            row = {"anchor": anchor_filename, "anchor_label": anchor_label}
            for k, fn in enumerate(neighbour_fns):
                row[f"neighbour_{k+1}"] = fn
            csv_rows.append(row)

            print(f"  Processed {folder_name} (Class {val_labels[idx].item()})")

        global_idx += batch_size

    pd.DataFrame(csv_rows).to_csv(os.path.join(savedir, "neighbours.csv"), index=False)
    print(f"Saved neighbours.csv → {savedir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",     default="all",    choices=["all", "dilated", "nondilated"])
    parser.add_argument("--seed",           default="seed_A", choices=["seed_A", "seed_B", "seed_C"])
    parser.add_argument("--N",              type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--k_neighbors",    type=int, default=2)
    args = parser.parse_args()

    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Setup Directories ---
    experiment_dir = os.path.join(
        "results_10_June_oversampling", args.experiment, args.seed, f"N{args.N}",
        f"GS{args.guidance_scale}_K{args.k_neighbors}",
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Params ---
    BATCH_SIZE = 50
    K_NEIGHBORS = args.k_neighbors
    GUIDANCE_SCALE = args.guidance_scale

    # Model Architecture
    NUM_CHANNELS_U_NET = 128
    NUM_RES_BLOCKS_U_NET = 2
    CHANNEL_MULT = (1, 2, 4, 8)  # Deep semantics: 128->256->512->1024
    ATTENTION_RESOLUTIONS = "32, 16, 8"
    IMG_SIZE = 128
    NO_OF_CHANNELS_IMG = 3

    # Load pre-computed embeddings and KNN indices
    seed_dir = os.path.join(BASE_EMBEDDINGS_DIR, args.experiment, args.seed)
    knn_indices = torch.load(
        os.path.join(seed_dir, f"N{args.N}", "indices_dinov2.pt"), map_location="cpu"
    )

    # ref_filenames: the N images for this seed (saved by compute_reference_knn.py)
    with open(os.path.join(seed_dir, f"N{args.N}", "ref_filenames_dinov2.json")) as f:
        ref_filenames = json.load(f)

    # Full embeddings are per-experiment (seed-independent); slice down to N
    full_ref_dir = os.path.join(BASE_EMBEDDINGS_DIR, args.experiment, "reference")
    features_full = torch.load(os.path.join(full_ref_dir, "features_dinov2.pt"), map_location="cpu")
    with open(os.path.join(full_ref_dir, "filenames_dinov2.json")) as f:
        filenames_full = json.load(f)
    filename_to_idx = {fn: i for i, fn in enumerate(filenames_full)}
    ref_features = features_full[[filename_to_idx[fn] for fn in ref_filenames]]

    feature_dim = ref_features.shape[-1]
    print(f"Reference features (N={args.N}): {ref_features.shape}  KNN indices: {knn_indices.shape}")
    df_labels = pd.read_csv("/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv")
    label_map = dict(zip(df_labels["id_code"].str.strip(), df_labels["diagnosis"]))
    ref_labels = torch.tensor(
        [label_map.get(os.path.splitext(fn)[0], label_map.get(fn, -1)) for fn in ref_filenames],
        dtype=torch.long,
    )

    GENERATED_IN_TOTAL = 5000
    # Per-class generation counts: target (1000+N)/5 total per class, distribute across anchors
    target_per_class = math.ceil((GENERATED_IN_TOTAL + args.N) / 5)
    class_counts = Counter(ref_labels.tolist())
    n_samples_per_class = {}
    for c in range(5):
        existing = class_counts.get(c, 0)
        to_generate = max(0, target_per_class - existing)
        n_samples_per_class[c] = math.ceil(to_generate / existing) if existing > 0 else 0

    print(f"Target per class: {target_per_class}")
    for c in range(5):
        existing = class_counts.get(c, 0)
        print(f"  Class {c}: {existing} existing → {n_samples_per_class[c]} synthetic per anchor")

    # Build filename → path index across Messidor2 folder for neighbour image loading
    raw_root = "/vol/biomedic3/awk24/datasets/Messidor2"
    file_index = {}
    for entry in os.listdir(raw_root):
        entry_path = os.path.join(raw_root, entry)
        if os.path.isdir(entry_path):
            for fname in os.listdir(entry_path):
                file_index[fname] = os.path.join(entry_path, fname)
        else:
            file_index[entry] = entry_path
    ref_image_paths = [file_index.get(fn) for fn in ref_filenames]

    # Initialize Dataset — reference pool of size N (the anchors)
    print("Loading reference dataset...")
    val_dataset = MessidorDataset(
        purpose="reference",
        experiment=args.experiment,
        seed=args.seed,
        N=args.N,
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2",
        json_dir="/vol/biomedic3/awk24/datasets/Messidor2",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=IMG_SIZE,
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNetModel(
        dim=(NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE),
        num_channels=NUM_CHANNELS_U_NET,
        num_res_blocks=NUM_RES_BLOCKS_U_NET,
        num_classes=1,
        class_cond=True,
        channel_mult=CHANNEL_MULT,
        attention_resolutions=ATTENTION_RESOLUTIONS,
    ).to(device)

    # Label embedding projection
    time_embed_dim = model.time_embed[-1].out_features
    model.label_emb = nn.Sequential(  # type: ignore
        nn.Linear(feature_dim, time_embed_dim),  # type: ignore
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
    ).to(device)

    # --- Load Weights ---
    model_path = "models/23_Feb_Eyepacs_dinov2/model_best_noble-hill-34.pth"

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"ERROR: Model file not found at {model_path}")
        exit()

    # --- Run Generation ---
    print(f"Starting Simulation (experiment={args.experiment}, seed={args.seed}, N={args.N}, "
          f"N_test={len(val_dataset)})...")

    with torch.no_grad():
        generate_cohorts(model, val_loader, ref_features, ref_labels,
                         ref_filenames, ref_image_paths, knn_indices,
                         K_NEIGHBORS, n_samples_per_class, GUIDANCE_SCALE, savedir=experiment_dir)

    print("Done.")
