import os

import torch
import torch.nn as nn

# --- Imports from your project structure ---
from CM_class import ColoredMNIST
from get_fsfm_condition import get_fsfm_condition
from torch.distributions import Exponential
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet import UNetModel

# --- Configuration ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_cohorts(model, val_loader, K_NEIGHBORS, guidance_scale, savedir):
    model.eval()
    sample_folder_root = os.path.join(savedir, "samples")
    os.makedirs(sample_folder_root, exist_ok=True)

    summary_folder_root = os.path.join(savedir, "summary")
    os.makedirs(summary_folder_root, exist_ok=True)

    # 1. Load Batch
    try:
        val_batch, val_labels = next(iter(val_loader))
    except StopIteration:
        print("Loader is empty!")
        return

    val_batch = val_batch.to(device)
    val_labels = val_labels.to(device)
    batch_size = val_batch.shape[0]

    print(f"Loaded batch of size {batch_size}. Calculating components...")

    # 2. Get Raw Components
    z_gathered, valid_masks, all_neighbor_indices = get_fsfm_condition(  # type: ignore
        val_batch, labels=val_labels, k=K_NEIGHBORS, ensure_same_label=True, return_components=True
    )

    random_indices = torch.arange(batch_size)

    # Define ODE Solver
    def vector_field(t, x, cond, null_cond):
        t_vector = torch.ones(x.shape[0], device=device) * t
        v_cond = model(t_vector, x, y=cond)
        v_uncond = model(t_vector, x, y=null_cond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    for idx in random_indices:
        anchor_idx = idx.item()

        sample_folder = os.path.join(sample_folder_root, str(anchor_idx))
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, str(anchor_idx))
        os.makedirs(summary_folder, exist_ok=True)

        anchor_feats = z_gathered[anchor_idx]
        anchor_mask = valid_masks[anchor_idx].squeeze(-1)

        # --- DYNAMIC SAMPLING ---
        n_samples = 100

        # Sample weights
        raw_weights = (
            Exponential(torch.tensor(1.0)).sample((n_samples, anchor_feats.shape[0])).to(device)
        )
        masked_weights = raw_weights * anchor_mask.unsqueeze(0)
        weight_sum = masked_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        final_weights = masked_weights / weight_sum

        # Calculate condition
        cond_batch = (final_weights.unsqueeze(-1) * anchor_feats.unsqueeze(0)).sum(dim=1)
        null_cond_batch = torch.zeros_like(cond_batch)

        # Generate
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
        save_image(original_img, os.path.join(summary_folder, "original.png"))  # <--- ADDED THIS

        # 2. Save Neighbors Reference
        my_neighbor_indices = all_neighbor_indices[anchor_idx]
        neighbor_imgs = (val_batch[my_neighbor_indices] + 1) / 2
        save_image(
            make_grid(neighbor_imgs, nrow=len(my_neighbor_indices), padding=2),
            os.path.join(summary_folder, "neighbors_reference.png"),
        )

        # 3. Save Cohort Grid
        grid_img = make_grid(generated_imgs_denorm, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "cohort_grid.png"))

        print(f"  Processed Anchor {anchor_idx}")


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Setup Directories ---
    # Adjust this path as needed
    savedir = "models/15_Feb_Coloured_MNIST_FSFM_Latent"
    experiment_dir = os.path.join(savedir, "simulation_blue_GS_7_same_label")
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Params ---
    BATCH_SIZE = 50
    K_NEIGHBORS = 2
    GUIDANCE_SCALE = 3

    # Model Architecture
    NUM_CHANNELS_U_NET = 64
    NUM_RES_BLOCKS_U_NET = 2
    CHANNEL_MULT = (1, 2, 4)
    ATTENTION_RESOLUTIONS = "16, 8"
    IMG_SIZE = 32
    NO_OF_CHANNELS_IMG = 3

    # --- Data ---
    if NO_OF_CHANNELS_IMG == 1:
        normalise = transforms.Normalize((0.5,), (0.5,))
    else:
        normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalise]
    )

    # NOTE: Set color idx=2 for BLUE ONLY experiment
    val_dataset = ColoredMNIST(
        root="../../../../datasets/MNIST",
        train=False,
        download=True,
        transform=transform,
        set_color_idx=2,  # ONLY BLUE
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Init ---
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
        nn.Linear(512, time_embed_dim),  # type: ignore
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
    ).to(device)

    # --- Load Weights ---
    model_path = os.path.join(savedir, "model_80_expressive-romance-10.pth")

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"ERROR: Model file not found at {model_path}")
        exit()

    # --- Run Generation ---
    print(f"Starting Hospital B Simulation (Blue Only, N=50, GS={GUIDANCE_SCALE})...")

    with torch.no_grad():
        generate_cohorts(model, val_loader, K_NEIGHBORS, GUIDANCE_SCALE, savedir=experiment_dir)

    print("Done.")
