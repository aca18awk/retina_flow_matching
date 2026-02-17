import os

import torch
import torchdiffeq
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

IMG_SIZE = 32
NUM_CLASSES = 11
NO_OF_CHANNELS_IMG = 1

n_samples_per_class = 8
total_samples = (NUM_CLASSES - 1) * n_samples_per_class
generated_class_list = torch.arange((NUM_CLASSES - 1), device=device).repeat_interleave(
    n_samples_per_class
)
no_label_list = torch.full_like(generated_class_list, fill_value=NUM_CLASSES - 1)
print(generated_class_list)
print(no_label_list)


def generate_images(model, figs_dir, guidance_scale, epoch="test", run_name="", seed=42):
    device = next(model.parameters()).device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def guided_vector_field(t, x, gs=guidance_scale):
        t_vector = torch.ones(x.shape[0], device=x.device) * t

        v_cond = model.forward(t_vector, x, generated_class_list)
        v_uncond = model.forward(t_vector, x, no_label_list)
        v_final = v_uncond + gs * (v_cond - v_uncond)
        return v_final

    model.eval()
    with torch.no_grad():
        x0_noise = torch.randn(
            total_samples, NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE, device=device, generator=g
        )

        traj = torchdiffeq.odeint(
            lambda t, x: guided_vector_field(t, x),
            x0_noise,
            torch.linspace(0, 1, 2, device=device),
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
        )

    final_images = traj[-1].view([-1, NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE])  # type: ignore

    grid = make_grid(
        final_images.clip(-1, 1),
        value_range=(-1, 1),
        normalize=True,
        padding=2,
        nrow=n_samples_per_class,
    )

    img = ToPILImage()(grid)
    image_filename = f"MNIST_gen_{epoch}_{run_name}.png"
    output_image_path = os.path.join(figs_dir, image_filename)

    img.save(output_image_path)
    print(f"Saved generated image to {output_image_path}")
