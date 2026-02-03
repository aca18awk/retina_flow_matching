import os
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchcfm.conditional_flow_matching import *
import torchdiffeq

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

IMG_SIZE = 128
NUM_CLASSES = 4
NO_OF_CHANNELS_IMG = 3

n_samples_per_class = 10
total_samples = (NUM_CLASSES -1) * n_samples_per_class 
generated_class_list = torch.arange((NUM_CLASSES -1), device=device).repeat_interleave(n_samples_per_class)
no_label_list = torch.full_like(generated_class_list, fill_value=3)


def generate_images(model, figs_dir, guidance_scale, epoch="test", run_name=""):

    def guided_vector_field(t, x, gs=guidance_scale):
        t_vector = torch.full((x.shape[0],), fill_value=t, device=x.device)
        v_cond = model.forward(t_vector, x, generated_class_list)
        v_uncond = model.forward(t_vector, x, no_label_list)
        v_final = v_uncond + gs * (v_cond - v_uncond)
        return v_final

    model.eval()
    with torch.no_grad():
        x0_noise = torch.randn(total_samples, NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE, device=device)
        traj = torchdiffeq.odeint(
                lambda t, x: guided_vector_field(t, x),
                x0_noise,
                torch.linspace(0, 1, 2, device=device),
                atol=1e-5, 
                rtol=1e-5,
                method="dopri5",
            )

    final_images = traj[-1].view([-1, NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE])

    grid = make_grid(
        final_images.clip(-1, 1), 
        value_range=(-1, 1),
        normalize=True, 
        padding=2, 
        nrow=n_samples_per_class
    )

    img = ToPILImage()(grid)
    image_filename = f"retina_gen_{epoch}_{run_name}.png"
    output_image_path = os.path.join(figs_dir, image_filename)

    img.save(output_image_path)
    print(f"Saved generated image to {output_image_path}")