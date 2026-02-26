import os

import torch
import torch.nn as nn
from Messidor_class import MessidorDataset
from torch.distributions import Exponential
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision.utils import save_image

from torchcfm.models.unet import UNetModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

FEATURE_FILE = "embeddings/Messidor/hospital_b/features_dinov2.pt"


def generate_row(model, x0, cond_batch, null_cond_batch, guidance_scale):
    """
    Generates a row of images for a specific Guidance Scale,
    using pre-computed noise and conditions to ensure alignment across different scales.
    """

    def vector_field(t, x):
        t_vector = torch.ones(x.shape[0], device=device) * t
        v_cond = model(t_vector, x, y=cond_batch)
        v_uncond = model(t_vector, x, y=null_cond_batch)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    with torch.no_grad():
        traj = odeint(
            vector_field,
            x0,
            torch.tensor([0.0, 1.0], device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )

    return traj[-1].clip(-1, 1)


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    folders_with_models = ["models/21_Feb_Eyepacs_dinov3_no_labels/"]

    models_to_evaluate = ["model_best_dauntless-tree-26.pth"]

    experiments_dir = ["model_best_gs_studies"]

    # --- Params ---
    BATCH_SIZE = 50
    K_NEIGHBORS = 2
    N_SAMPLES = 5  # Columns per row (Diversity)
    GUIDANCE_SCALES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10]

    NUM_PATIENTS_TO_EVALUATE = 1

    IMG_SIZE = 128
    NO_OF_CHANNELS_IMG = 3
    CHANNEL_MULT = (1, 2, 4, 8)
    ATTENTION_RESOLUTIONS = "32, 16, 8"
    NUM_CHANNELS_U_NET = 128
    NUM_RES_BLOCKS_U_NET = 2

    print("Loading Dataset...")
    val_dataset = MessidorDataset(
        purpose="hospital_b",
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=IMG_SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # --- PHASE 1: PREPARE BATCH & FEATURES ---
    # ==========================================
    val_batch, val_labels, val_filenames = next(iter(val_loader))
    val_batch = val_batch.to(device)
    val_labels = val_labels.to(device)

    # Load .pt files and group neighbors inline ---
    print(f"Loading pre-computed features from {FEATURE_FILE}...")
    z = torch.load(FEATURE_FILE, map_location=device)

    # Recreate the exact K-NN grouping geometry
    dist = torch.cdist(z, z)
    label_match_mask = val_labels.unsqueeze(0) == val_labels.unsqueeze(1)
    dist = dist.masked_fill(~label_match_mask, float("inf"))

    # Find K_NEIGHBORS + 1 (since the anchor itself is included at distance 0)
    actual_k = min(K_NEIGHBORS + 1, z.shape[0])
    dists, all_neighbor_indices = torch.topk(dist, k=actual_k, largest=False)

    z_gathered = z[all_neighbor_indices]
    valid_masks = (dists != float("inf")).float().unsqueeze(-1)
    feature_dim = z.shape[-1]  # Grabs 1024 for DINOv2, or 512 for ResNet

    for m_idx in range(len(folders_with_models)):
        savedir = folders_with_models[m_idx]
        model_path = os.path.join(savedir, models_to_evaluate[m_idx])
        experiment_dir = os.path.join(savedir, experiments_dir[m_idx])

        os.makedirs(experiment_dir, exist_ok=True)

        print("Initializing Model...")
        model = UNetModel(
            dim=(NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE),
            num_channels=NUM_CHANNELS_U_NET,
            num_res_blocks=NUM_RES_BLOCKS_U_NET,
            num_classes=1,
            class_cond=True,
            channel_mult=CHANNEL_MULT,
            attention_resolutions=ATTENTION_RESOLUTIONS,
        ).to(device)

        time_embed_dim = model.time_embed[-1].out_features

        # --- Dynamic input dimension ---
        model.label_emb = nn.Sequential(  # type: ignore
            nn.Linear(feature_dim, time_embed_dim),  # type: ignore
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
        ).to(device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Model missing at {model_path}!")
            continue

        model.eval()

        # ==========================================
        # --- PHASE 2: GENERATE PATIENT GRIDS ---
        # ==========================================
        for i in range(NUM_PATIENTS_TO_EVALUATE):
            anchor_idx = i
            filename = os.path.splitext(val_filenames[anchor_idx])[0]
            print(f"\nProcessing Patient: {filename} (Class {val_labels[anchor_idx].item()})")

            # --- 1. Build Row 1 (References & Padding) ---
            neighbor_imgs = val_batch[all_neighbor_indices[anchor_idx]]

            # Pad the rest of the row with white squares (1s) so it aligns with N_SAMPLES columns
            pad_count = N_SAMPLES - 1 - K_NEIGHBORS
            white_pad = torch.ones(pad_count, 3, IMG_SIZE, IMG_SIZE).to(device)

            ref_row = torch.cat(
                [neighbor_imgs, white_pad], dim=0
            )  # Shape: [N_SAMPLES, 3, 128, 128]
            grid_rows = [ref_row]

            # --- 2. Create FIXED Condition & Noise for this Patient ---
            anchor_feats = z_gathered[anchor_idx]
            anchor_mask = valid_masks[anchor_idx].squeeze(-1)

            # Sample N_SAMPLES different neighbor weightings for diversity
            raw_weights = (
                Exponential(torch.tensor(1.0))
                .sample((N_SAMPLES, anchor_feats.shape[0]))
                .to(device)
            )
            masked_weights = raw_weights * anchor_mask.unsqueeze(0)
            final_weights = masked_weights / masked_weights.sum(dim=1, keepdim=True).clamp(
                min=1e-6
            )

            cond_batch = (final_weights.unsqueeze(-1) * anchor_feats.unsqueeze(0)).sum(dim=1)
            null_cond_batch = torch.zeros_like(cond_batch)

            # Fix standard normal noise for all GS runs
            x0 = torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE, device=device)

            # --- 3. Iterate Guidance Scales ---
            for gs in GUIDANCE_SCALES:
                print(f"  Generating GS = {gs}...")
                gen_imgs = generate_row(model, x0, cond_batch, null_cond_batch, guidance_scale=gs)
                grid_rows.append(gen_imgs)

            # --- 4. Save Grid ---
            final_grid_tensor = torch.cat(grid_rows, dim=0)
            save_path = os.path.join(experiment_dir, f"{filename}_GS_study_all.png")

            save_image(
                final_grid_tensor,
                save_path,
                nrow=N_SAMPLES,
                normalize=True,
                value_range=(-1, 1),
                padding=2,
            )
            print(f"✅ Saved to: {save_path}")
