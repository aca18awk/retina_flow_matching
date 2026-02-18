import os

import torch
from torchdiffeq import odeint
from torchvision.utils import save_image


def generate_images(
    model, figs_dir, val_loader, K_NEIGHBORS, guidance_scale, epoch="test", seed=42
):
    # 1. Setup Device & Seed
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # 2. Get Data (Handle Tuple Unpacking)
    # The dataset now returns (image, condition_vector)
    val_batch = next(iter(val_loader))

    if isinstance(val_batch, (list, tuple)):
        full_imgs, full_conds = val_batch
    else:
        # Fallback if dataset somehow returns only images
        full_imgs = val_batch
        full_conds = torch.zeros(full_imgs.shape[0], 512)

    # Move to GPU
    full_imgs = full_imgs.to(device)
    full_conds = full_conds.to(device)

    # 3. Select Anchors (First 8 images)
    num_samples = 8
    anchor_imgs = full_imgs[:num_samples]

    # The condition is now the semantic vector from the dataset
    # We DO NOT calculate get_fsfm_condition here anymore
    label = full_conds[:num_samples]

    y_uncond = torch.zeros_like(label)

    # 4. Define Guided Vector Field
    def guided_vector_field(t, x):
        # Handle t broadcasting correctly for odeint
        # t is a scalar tensor coming from odeint, we need shape [B]
        t_vec = torch.ones(x.shape[0], device=device) * t

        # CFG: Uncond + w * (Cond - Uncond)
        v_cond = model(t_vec, x, y=label)
        v_uncond = model(t_vec, x, y=y_uncond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    # 5. Solve the ODE (Noise -> Data)
    x0 = torch.randn(
        *anchor_imgs.shape,
        device=device,
        generator=g,
    )

    traj = odeint(
        guided_vector_field,  # Cleaner lambda syntax
        x0,
        torch.tensor([0.0, 1.0], device=device),
        atol=1e-4,
        rtol=1e-4,
        method="dopri5",
    )

    generated_imgs = traj[-1]  # Final state at t=1

    # 6. Save Images
    # We clip to valid range [-1, 1] for safety
    final_images = generated_imgs.clip(-1, 1)

    # Concatenate: Top Row = Original Anchors, Bottom Row = Generated Reconstructions
    combined_images = torch.cat([anchor_imgs, final_images], dim=0)

    # Simplified filename (Vectors are too long to put in filename)
    save_filename = f"epoch_{epoch}_guidance_{guidance_scale}.png"
    save_path = os.path.join(figs_dir, save_filename)

    save_image(
        combined_images,
        save_path,
        nrow=num_samples,  # This makes it a 2-row grid (Row 1: Real, Row 2: Fake)
        normalize=True,  # Maps [-1, 1] -> [0, 1]
        value_range=(-1, 1),
        padding=2,
    )

    print(f"Saved visualization to {save_path}")
