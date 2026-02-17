import os

import torch
import torch.nn as nn
from CM_class import ColoredMNIST
from get_fsfm_condition import get_fsfm_condition
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision import transforms
from torchvision.utils import save_image

from torchcfm.models.unet import UNetModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_row(model, val_loader, K_NEIGHBORS, guidance_scale, seed=42):
    """
    Returns a tensor of generated images for a specific guidance scale.
    """
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def guided_vector_field(t, x):
        t_vector = torch.ones(x.shape[0], device=device) * t
        v_cond = model(t_vector, x, y=label)
        v_uncond = model(t_vector, x, y=y_uncond)
        v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
        diff = (v_cond - v_uncond).abs().mean().item()
        print(f"  [DEBUG] Guidance Strength (Diff): {diff:.6f}")
        return v_final

    # Fetch one fixed batch to be consistent across all GS values
    val_iter = iter(val_loader)
    val_batch = next(val_iter)

    full_imgs = val_batch[0].to(device)

    # We only take the first 8 images for the column width
    N_COLS = 8

    # Get condition (latent average)
    y_full = get_fsfm_condition(full_imgs, k=K_NEIGHBORS)
    label = y_full[:N_COLS]
    anchor_imgs = full_imgs[:N_COLS]
    y_uncond = torch.zeros_like(label)

    # Define the vector field for this specific GS

    x0 = torch.randn(
        *anchor_imgs.shape,
        device=device,
        generator=g,
    )

    traj = odeint(
        lambda t, x: guided_vector_field(t, x),
        x0,
        torch.tensor([0.0, 1.0], device=device),
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )

    generated_imgs = traj[-1].clip(-1, 1)  # Final state
    return anchor_imgs, generated_imgs


if __name__ == "__main__":
    savedir = "models/15_Feb_Coloured_MNIST_FSFM_Latent"
    os.makedirs(savedir, exist_ok=True)
    figs_dir = os.path.join(savedir, "guidance_scale_study")
    os.makedirs(figs_dir, exist_ok=True)

    VALIDATION_SEED = 42
    BATCH_SIZE = 128
    K_NEIGHBORS = 2

    # --- Model Params ---
    NUM_CHANNELS_U_NET = 64
    NUM_RES_BLOCKS_U_NET = 2
    CHANNEL_MULT = (1, 2, 4)
    ATTENTION_RESOLUTIONS = "16, 8"
    IMG_SIZE = 32
    NO_OF_CHANNELS_IMG = 3

    if NO_OF_CHANNELS_IMG == 1:
        normalise = transforms.Normalize((0.5,), (0.5,))
    else:
        normalise = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # --- Datasets ---
    transform = transforms.Compose(
        [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalise]
    )

    val_dataset = ColoredMNIST(
        root="../../../../datasets/MNIST", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- FSFM: 2. Model Initialization & Monkey-Patching ---
    # We initialize with num_classes=1 just to trigger the conditional logic in the UNet
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
    model.label_emb = nn.Sequential(  # type: ignore
        nn.Linear(512, time_embed_dim),  # type: ignore
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
    ).to(device)

    # --- Load Weights ---
    model_path = "models/15_Feb_Coloured_MNIST_FSFM_Latent/model_80_expressive-romance-10.pth"
    if os.path.exists(model_path):
        print(f"Loading {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        print("Model file not found! Please check path.")
        exit()

    GUIDANCE_SCALES = [0, 1, 3, 7, 20, 50]

    # --- COLLECTING THE GRID ---
    rows_to_stack = []
    is_original_added = False

    print("Generating rows...")
    with torch.no_grad():
        # 2. Generate for each GS
        for gs in GUIDANCE_SCALES:
            print(f"  Processing GS = {gs}...")
            anchors, gen_imgs = generate_row(
                model, val_loader, K_NEIGHBORS, guidance_scale=gs, seed=VALIDATION_SEED
            )
            if not is_original_added:
                rows_to_stack.append(anchors)
                is_original_added = True

            rows_to_stack.append(gen_imgs)

        # 3. Stack and Save
        # We have (1 + N_SCALES) tensors of shape [8, 3, 32, 32]
        # We want to stack them vertically.
        final_grid_tensor = torch.cat(rows_to_stack, dim=0)

        save_path = os.path.join(figs_dir, "guidance_scale_0_1_3_7_20_50.png")

        save_image(
            final_grid_tensor,
            save_path,
            nrow=8,  # Number of columns (same as N_COLS above)
            normalize=True,
            value_range=(-1, 1),
            padding=2,
        )

        print(f"\n✅ Big grid saved to: {save_path}")
        print("Row 1: Anchors")
        for i, gs in enumerate(GUIDANCE_SCALES):
            print(f"Row {i + 2}: GS {gs}")
