import os

import torch
import torch.nn as nn

# --- Imports from your project structure ---
from CM_class import ColoredMNIST
from CM_get_fsfm_condition import get_fsfm_condition
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from torchcfm.models.unet import UNetModel

# --- Configuration ---
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_cohorts(model, val_loader, K_NEIGHBORS, guidance_scale, savedir):
    """
    Simulates Hospital B scenario:
    1. Takes the first batch (size 50).
    2. Selects 10 random images as 'anchors'.
    3. For each anchor, generates 50 new samples using cohort-restricted conditioning.
    """
    model.eval()

    sample_folder_root = os.path.join(savedir, "samples")
    os.makedirs(sample_folder_root, exist_ok=True)

    summary_folder_root = os.path.join(savedir, "summary")
    os.makedirs(summary_folder_root, exist_ok=True)

    # Replace the existing batch loading lines with this:
    try:
        val_batch, val_labels = next(iter(val_loader))  # Capture labels here
    except StopIteration:
        print("Loader is empty!")
        return

    val_batch = val_batch.to(device)
    val_labels = val_labels.to(device)

    batch_size = val_batch.shape[0]

    print(f"Loaded batch of size {batch_size}. Calculating embeddings...")

    # --- Pre-calculate Embeddings/Conditions for the WHOLE batch ---
    # This assumes get_fsfm_condition calculates neighbors within the provided batch
    # and returns the averaged conditioning vector for each item.
    # Shape: [50, Embedding_Dim] (e.g., 512)
    all_conditions, all_neighbor_indices = get_fsfm_condition(
        val_batch, labels=val_labels, k=K_NEIGHBORS, return_neigh=True, ensure_same_label=True
    )

    # 2. Pick 10 random indices from this batch to serve as anchors
    # num_anchors = 5
    # if batch_size < num_anchors:
    #     random_indices = torch.arange(batch_size)
    # else:
    #     # Fix seed for reproducibility of anchor selection
    #     g_cpu = torch.Generator()
    #     g_cpu.manual_seed(42)
    #     random_indices = torch.randperm(batch_size, generator=g_cpu)[:num_anchors]

    random_indices = torch.arange(batch_size)

    print(f"Generating cohorts for {len(random_indices)} anchors...")

    # Define the ODE function (Standard Flow Matching)
    def vector_field(t, x, cond, null_cond):
        t_vector = torch.ones(x.shape[0], device=device) * t
        v_cond = model(t_vector, x, y=cond)
        v_uncond = model(t_vector, x, y=null_cond)
        # CFG Formula
        v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
        return v_final

    for idx in random_indices:
        anchor_idx = idx.item()

        # --- Create Folder ---
        # Folder name: "figs_folder/12"
        sample_folder = os.path.join(sample_folder_root, str(anchor_idx))
        os.makedirs(sample_folder, exist_ok=True)

        summary_folder = os.path.join(summary_folder_root, str(anchor_idx))
        os.makedirs(summary_folder, exist_ok=True)

        my_neighbor_indices = all_neighbor_indices[anchor_idx]

        # --- Print Labels ---
        anchor_lbl = val_labels[anchor_idx].item()
        neighbor_lbls = val_labels[my_neighbor_indices].cpu().tolist()
        print(
            f"  [Anchor {anchor_idx}] Label: {anchor_lbl} | Neighbors (incl. self) Labels: {neighbor_lbls}"
        )

        # --- Save Neighbor Reference Image ---
        # This saves a grid containing: [Original Anchor, Neighbor 1, Neighbor 2]
        neighbor_imgs = (val_batch[my_neighbor_indices] + 1) / 2
        save_image(
            make_grid(neighbor_imgs, nrow=len(my_neighbor_indices), padding=2),
            os.path.join(summary_folder, "neighbors_reference.png"),
        )

        # --- Save Original Anchor ---
        # Denormalize [-1, 1] -> [0, 1]
        original_img = (val_batch[anchor_idx] + 1) / 2
        save_image(original_img, os.path.join(summary_folder, "original.png"))

        # --- Prepare Conditioning for this Anchor ---
        # We take the conditioning vector specific to this anchor
        # (which was formed by averaging its neighbors in the batch)
        anchor_cond = all_conditions[anchor_idx].unsqueeze(0)  # [1, 512]

        # We want to generate 50 samples
        n_samples = 100

        # Repeat condition 50 times
        cond_batch = anchor_cond.repeat(n_samples, 1)  # [50, 512]
        null_cond_batch = torch.zeros_like(cond_batch)  # [50, 512]

        # Start Noise for 50 samples
        x0 = torch.randn(n_samples, *val_batch.shape[1:], device=device)

        # --- Solve ODE ---
        with torch.no_grad():
            traj = odeint(
                lambda t, x: vector_field(t, x, cond_batch, null_cond_batch),
                x0,
                torch.tensor([0.0, 1.0], device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        generated_imgs = traj[-1].clip(-1, 1)  # [-1, 1]

        # Denormalize results
        generated_imgs_denorm = (generated_imgs + 1) / 2
        generated_imgs_denorm = torch.clamp(generated_imgs_denorm, 0, 1)

        # --- Save Individual Files ---
        for i in range(n_samples):
            save_image(generated_imgs_denorm[i], os.path.join(sample_folder, f"{i}.png"))

        # --- Save Grid ---
        # 5 rows x 10 cols
        grid_img = make_grid(generated_imgs_denorm, nrow=10, padding=2, normalize=False)
        save_image(grid_img, os.path.join(summary_folder, "grid.png"))

        print(f"  Saved Anchor {anchor_idx}: Original + 50 Samples + Grid -> {sample_folder}")


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Setup Directories ---
    # Adjust this path as needed
    savedir = "models/15_Feb_Coloured_MNIST_FSFM_Latent"
    experiment_dir = os.path.join(savedir, "simulation_blue_GS_5_same_label")
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Params ---
    BATCH_SIZE = 50
    K_NEIGHBORS = 2
    GUIDANCE_SCALE = 5

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
