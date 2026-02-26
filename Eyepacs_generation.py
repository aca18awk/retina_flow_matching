import os

import torch
import torch.nn as nn

# --- Imports from your project structure ---
from Messidor_class import MessidorDataset
from torch.distributions import Exponential
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet import UNetModel

# --- Configuration ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

FEATURE_FILE = "embeddings/Messidor/hospital_b/features_dinov2.pt"


def generate_cohorts(model, val_loader, K_NEIGHBORS, guidance_scale, savedir):
    model.eval()

    # Create Output Directories
    sample_folder_root = os.path.join(savedir, "samples")
    os.makedirs(sample_folder_root, exist_ok=True)

    summary_folder_root = os.path.join(savedir, "summary")
    os.makedirs(summary_folder_root, exist_ok=True)

    # 1. Load Batch
    # MessidorDataset returns (image, label, filename)
    try:
        batch_data = next(iter(val_loader))
        val_batch, val_labels, val_filenames = batch_data
    except StopIteration:
        print("Loader is empty!")
        return

    val_batch = val_batch.to(device)
    val_labels = val_labels.to(device)
    batch_size = val_batch.shape[0]

    print(f"Loaded batch of size {batch_size}. Calculating components...")

    z = torch.load(FEATURE_FILE, map_location=device)

    dist = torch.cdist(z, z)
    label_match_mask = val_labels.unsqueeze(0) == val_labels.unsqueeze(1)
    dist = dist.masked_fill(~label_match_mask, float("inf"))

    actual_k = min(K_NEIGHBORS + 1, z.shape[0])
    dists, all_neighbor_indices = torch.topk(dist, k=actual_k, largest=False)

    z_gathered = z[all_neighbor_indices]
    valid_masks = (dists != float("inf")).float().unsqueeze(-1)

    random_indices = torch.arange(batch_size)

    # Define ODE Solver Vector Field
    def vector_field(t, x, cond, null_cond):
        t_vector = torch.ones(x.shape[0], device=device) * t
        # Model forward pass
        v_cond = model(t_vector, x, y=cond)
        v_uncond = model(t_vector, x, y=null_cond)
        # Classifier-Free Guidance
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    # 3. Iterate through each image in Hospital B
    print(f"Generating cohorts for {len(random_indices)} patients...")

    for idx in random_indices:
        anchor_idx = idx.item()
        anchor_filename = val_filenames[anchor_idx]

        # Create subfolders for this specific patient
        # We use the index + filename for clarity
        folder_name = f"{anchor_idx}_{os.path.splitext(anchor_filename)[0]}"

        sample_folder = os.path.join(sample_folder_root, folder_name)
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, folder_name)
        os.makedirs(summary_folder, exist_ok=True)

        anchor_feats = z_gathered[anchor_idx]
        anchor_mask = valid_masks[anchor_idx].squeeze(-1)

        # --- DYNAMIC SAMPLING ---
        n_samples = 20

        # Sample weights from Dirichlet/Exponential to mix neighbors
        # This creates the "Condition" (z) for the flow matching
        raw_weights = (
            Exponential(torch.tensor(1.0)).sample((n_samples, anchor_feats.shape[0])).to(device)
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

        generated_imgs = traj[-1].clip(-1, 1)
        generated_imgs_denorm = torch.clamp((generated_imgs + 1) / 2, 0, 1)

        for i in range(n_samples):
            save_image(generated_imgs_denorm[i], os.path.join(sample_folder, f"{i}.png"))

        # --- SAVING ---

        # 1. Save Original Anchor (The "Patient Zero")
        original_img = (val_batch[anchor_idx] + 1) / 2
        save_image(original_img, os.path.join(summary_folder, "original.png"))

        # 2. Save Neighbors Reference
        my_neighbor_indices = all_neighbor_indices[anchor_idx]
        neighbor_imgs = (val_batch[my_neighbor_indices] + 1) / 2
        save_image(
            make_grid(neighbor_imgs, nrow=len(my_neighbor_indices), padding=2),
            os.path.join(summary_folder, "neighbors_reference.png"),
        )

        # 3. Save Cohort Grid (The synthetic images)
        grid_img = make_grid(generated_imgs_denorm, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

        print(f"  Processed {folder_name} (Class {val_labels[anchor_idx].item()})")


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Setup Directories ---
    # Update this to where you want to save the results

    savedir = "models/21_Feb_Eyepacs_dinov3_no_labels/"
    experiment_dir = os.path.join(savedir, "simulation_GS_1.5_same_class_model_best")
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Params ---
    BATCH_SIZE = 50
    K_NEIGHBORS = 2
    GUIDANCE_SCALE = 1.5

    # Model Architecture
    NUM_CHANNELS_U_NET = 128
    NUM_RES_BLOCKS_U_NET = 2
    CHANNEL_MULT = (1, 2, 4, 8)  # Deep semantics: 128->256->512->1024
    ATTENTION_RESOLUTIONS = "32, 16, 8"
    IMG_SIZE = 128
    NO_OF_CHANNELS_IMG = 3

    # Initialize Dataset with purpose="hospital_b"
    print("Loading Hospital B dataset...")
    val_dataset = MessidorDataset(
        purpose="hospital_b",
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path="/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=IMG_SIZE,
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    temp_z = torch.load(FEATURE_FILE, map_location="cpu")
    feature_dim = temp_z.shape[-1]

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
    model_path = os.path.join(savedir, "model_best_dauntless-tree-26.pth")

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"ERROR: Model file not found at {model_path}")
        exit()

    # --- Run Generation ---
    print(f"Starting Simulation on Hospital B (N={len(val_dataset)})...")

    with torch.no_grad():
        generate_cohorts(model, val_loader, K_NEIGHBORS, GUIDANCE_SCALE, savedir=experiment_dir)

    print("Done.")
