import os

import torch
from CM_get_fsfm_condition import get_fsfm_condition
from torchdiffeq import odeint
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_images(
    model, figs_dir, val_loader, K_NEIGHBORS, guidance_scale, epoch="test", seed=42
):
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def guided_vector_field(t, x, gs=guidance_scale):
        t_vector = torch.full((x.shape[0],), fill_value=t, device=device)
        v_cond = model(t_vector, x, y=label)
        v_uncond = model(t_vector, x, y=y_uncond)
        v_final = v_uncond + gs * (v_cond - v_uncond)
        return v_final

    val_batch = next(iter(val_loader))
    full_imgs = val_batch[0].to(device)
    full_labels = val_batch[1]

    y_full = get_fsfm_condition(full_imgs, k=K_NEIGHBORS)

    label = y_full[:8]
    anchor_imgs = full_imgs[:8]
    anchor_labels = full_labels[:8].tolist()

    y_uncond = torch.zeros_like(label)

    # 4. Solve the ODE (Noise -> Data)
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

    generated_imgs = traj[-1]  # Final state at t=1

    label_str = "-".join(map(str, anchor_labels))
    save_filename = f"epoch_{epoch}_targets_{label_str}.png"
    save_path = os.path.join(figs_dir, save_filename)

    final_images = generated_imgs.clip(-1, 1)

    combined_images = torch.cat([anchor_imgs, final_images], dim=0)

    save_image(
        combined_images,
        save_path,
        nrow=8,
        normalize=True,  # Maps [-1, 1] -> [0, 1] for PNG saving
        value_range=(-1, 1),  # Explicitly tells function that min=-1, max=1
        padding=2,
    )
