import math
import os

import torch
import torchdiffeq
from torchvision.utils import save_image
from tqdm import tqdm  # Added for progress bars

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# --- Configuration ---

# Paths
best_model_path = "models/2_Feb/model_last_ancient-blaze-55.pth"
model_name = "ancient-blaze-55_dopri5"

# Generation Settings
GUIDANCE_SCALES = [3.0, 5.0, 10.0]
# GUIDANCE_SCALES = [3.0]

BATCH_SIZE = 10

# Sample counts per class
n_samples_dict = {"normal_control": 100, "early_glaucoma": 100, "advanced_glaucoma": 100}

class_name_to_number = {"normal_control": 2, "early_glaucoma": 1, "advanced_glaucoma": 0}

# Model Architecture Variables
IMG_SIZE = 128
NUM_CLASSES = 4
NO_OF_CHANNELS_IMG = 3
NUM_CHANNELS_U_NET = 128
NUM_RES_BLOCKS_U_NET = 2
CHANNEL_MULT = (1, 2, 4, 8)
ATTENTION_RESOLUTIONS = "32, 16, 8"

# --- Model Setup ---

model = UNetModel(
    dim=(NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE),
    num_channels=NUM_CHANNELS_U_NET,
    num_res_blocks=NUM_RES_BLOCKS_U_NET,
    num_classes=NUM_CLASSES,
    class_cond=True,
    channel_mult=CHANNEL_MULT,
    attention_resolutions=ATTENTION_RESOLUTIONS,
).to(device)

print(f"Loading model from {best_model_path}...")
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    for guidance_scale in GUIDANCE_SCALES:
        print(f"\n--- Processing Guidance Scale: {guidance_scale} ---")
        savedir = os.path.join(
            "/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/2_Feb/",
            f"{model_name}_GS_{guidance_scale}",
        )
        os.makedirs(savedir, exist_ok=True)

        for class_name, total_count in n_samples_dict.items():
            print(f"Generating {total_count} images for class: {class_name}...")

            class_save_path = os.path.join(savedir, class_name)
            os.makedirs(class_save_path, exist_ok=True)

            target_label = class_name_to_number[class_name]

            # Calculate number of batches needed
            num_batches = math.ceil(total_count / BATCH_SIZE)

            # Global counter for naming files correctly across batches
            generated_so_far = 0

            for batch_idx in tqdm(range(num_batches), desc=f"{class_name}"):
                # Determine current batch size (last batch might be smaller)
                current_batch_size = min(BATCH_SIZE, total_count - generated_so_far)

                # Prepare labels for this batch
                generated_class_list = torch.full(
                    (current_batch_size,), fill_value=target_label, device=device
                )
                no_label_list = torch.full_like(
                    generated_class_list, fill_value=3
                )  # Assuming 3 is null class

                # Define the guided vector field closure
                def guided_vector_field(t, x):
                    t_vector = torch.full((x.shape[0],), fill_value=t, device=x.device)
                    v_cond = model.forward(t_vector, x, generated_class_list)
                    v_uncond = model.forward(t_vector, x, no_label_list)
                    v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
                    return v_final

                # Generate noise for the current batch
                x0_noise = torch.randn(
                    current_batch_size, NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE, device=device
                )

                # Solve ODE
                traj = torchdiffeq.odeint(
                    lambda t, x: guided_vector_field(t, x),
                    x0_noise,
                    torch.linspace(0, 1, 2, device=device),
                    atol=1e-5,
                    rtol=1e-5,
                    method="dopri5",
                    # torch.linspace(0, 1, 100, device=device),
                    # atol=1e-5,
                    # rtol=1e-5,
                    # method="euler",
                )

                final_images = traj[-1]  # Shape: [batch_size, 3, 128, 128]

                # Save images
                for i, img in enumerate(final_images):
                    file_name = f"{generated_so_far + i + 1}.png"
                    file_path = os.path.join(class_save_path, file_name)
                    save_image(img.clip(-1, 1), file_path, normalize=True, value_range=(-1, 1))

                generated_so_far += current_batch_size

print("Generation complete.")
