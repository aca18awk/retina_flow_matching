import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from Eyepacs_class import EyepacsDataset
from Eyepacs_generate_images import generate_images
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_  # type: ignore
from torch.utils.data import DataLoader

from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# --- Configuration ---
savedir = "models/18_Feb_Eyepacs_multiple_GPUs"
os.makedirs(savedir, exist_ok=True)
figs_dir = os.path.join(savedir, "figs")
os.makedirs(figs_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
VALIDATION_SEED = 42
GUIDANCE_SCALE = 3.0

# training params
N_EPOCHS = 500
BATCH_SIZE = 126
MAX_PLATEAU = 50
LEARNING_RATE = 0.0001
K_NEIGHBORS = 2  # The FSFM "k"
IMG_SIZE = 128

# model params
NUM_CHANNELS_U_NET = 128
NUM_RES_BLOCKS_U_NET = 2
CHANNEL_MULT = (1, 2, 4, 8)  # Deep semantics: 128->256->512->1024
ATTENTION_RESOLUTIONS = "32, 16, 8"
NO_OF_CHANNELS_IMG = 3


# --- WandB Init ---
logger = wandb.init(
    project="flow_matching_eyepacs",
    config={
        "type": "FSFM_Latent_Conditioning",
        "k_neighbors": K_NEIGHBORS,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "mixed_precision": True,
    },
)
run_name = logger.name or "FSFM_Run"


# Define source and local target
source_dir = "/vol/biomedic3/awk24/datasets/EYEPACS_256"
# Use environment variable for cluster scratch if available, else /tmp
local_scratch = os.environ.get("TMPDIR", "/tmp")
dataset_loc = os.path.join(local_scratch, "EYEPACS_256")

# We remove the directory if it exists to ensure no partial/corrupted data remains
if os.path.exists(dataset_loc):
    print("data already copied")
    # print(f"Cleaning up existing data at {dataset_loc}...")
    # shutil.rmtree(dataset_loc)
else:
    print(f"Copying data from {source_dir} to {dataset_loc}...")
    start_time = time.time()

    # Option B: Copying the pre-processed 256px folder
    # This should now take < 30 seconds since we've shrunk the dataset!
    os.system(f"cp -r {source_dir} {local_scratch}")

    print(f"Data copied in {time.time() - start_time:.2f}s")

# UPDATE YOUR DATASET INIT TO USE THE LOCAL PATH
train_dataset = EyepacsDataset(
    root=dataset_loc,
    purpose="train",
    img_size=IMG_SIZE,
    k_neighbours=K_NEIGHBORS,
)

# train_dataset = EyepacsDataset(purpose="train", img_size=IMG_SIZE, k_neighbours=K_NEIGHBORS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_dataset = EyepacsDataset(
    root=dataset_loc, purpose="validation", img_size=IMG_SIZE, k_neighbours=K_NEIGHBORS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

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

# HACK: Replace the embedding layer with a Linear Projection for our 512-dim vector
# We check the internal dimension the UNet expects for time embeddings
time_embed_dim = model.time_embed[-1].out_features
model.label_emb = nn.Sequential(  # type: ignore
    nn.Linear(512, time_embed_dim),  # type: ignore
    nn.SiLU(),
    nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
).to(device)

print(f"Model patched! Accepting 512-dim latent vectors mapped to {time_embed_dim}-dim.")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # type: ignore

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # <--- 2. Initialize GradScaler
FM = TargetConditionalFlowMatcher(sigma=0.0)

# --- Training State ---
best_loss = float("inf")
plateau_count = 0
best_model = None

print(f"Starting FSFM training on {device}...")

for epoch in range(N_EPOCHS):
    model.train()
    print(f"Starting epoch {epoch}")

    train_avg_loss = []
    for i, data in enumerate(train_loader):
        print(">>>", i)
        optimizer.zero_grad()
        x1 = data[0].to(device)

        y1 = data[1].to(device)
        mask = torch.rand(x1.shape[0], device=device) > 0.1
        mask = mask.view(-1, 1)  # [B, 1] for broadcasting

        null_label = torch.zeros_like(y1)
        label = torch.where(mask, y1, null_label)

        # Standard Flow Matching
        x0 = torch.randn_like(x1)

        # <--- 3. Wrap Forward Pass in autocast
        with autocast():
            t, xt, ut, *_ = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, label)
            loss = torch.mean((vt - ut) ** 2)

        train_avg_loss.append(loss.item())

        # <--- 4. Scale Gradients & Step
        scaler.scale(loss).backward()

        # Unscale before clipping (important!)
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

    epoch_loss = np.mean(train_avg_loss)

    model.eval()
    val_avg_loss = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            x1 = data[0].to(device)
            y1 = data[1].to(device)
            x0 = torch.randn_like(x1)

            with autocast():
                t, xt, ut, *_ = FM.sample_location_and_conditional_flow(x0, x1)
                vt = model(t, xt, y1)
                loss = torch.mean((vt - ut) ** 2)

            val_avg_loss.append(loss.item())

    val_epoch_loss = np.mean(val_avg_loss)

    print(f"Epoch: {epoch} | Train: {epoch_loss:.4f} | Val: {val_epoch_loss:.4f}")
    logger.log({"epoch": epoch, "val_loss": val_epoch_loss, "train_loss": epoch_loss})

    # --- VISUALIZATION & GENERATION CHECK ---
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            eval_model = model.module if hasattr(model, "module") else model
            generate_images(
                eval_model,
                figs_dir,
                val_loader,
                K_NEIGHBORS,
                GUIDANCE_SCALE,
                epoch=str(epoch),
                seed=VALIDATION_SEED,
            )

            # Helper to get the underlying model whether wrapped or not
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(
                model_to_save.state_dict(),  # type: ignore
                os.path.join(savedir, f"model_{epoch}_{run_name}.pth"),
            )

    # --- Early Stopping ---
    if val_epoch_loss <= best_loss:
        plateau_count = 0
        best_loss = val_epoch_loss

        model_to_save = model.module if hasattr(model, "module") else model
        best_model = copy.deepcopy(model_to_save.state_dict())  # type: ignore
        print(f"  --> New Best Model! (Val Loss: {best_loss:.4f})")
    else:
        plateau_count += 1
        print(f"  --> No improvement. Patience: {plateau_count}/{MAX_PLATEAU}")

    if plateau_count >= MAX_PLATEAU:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")

eval_model = model.module if hasattr(model, "module") else model
generate_images(
    eval_model,
    figs_dir,
    val_loader,
    K_NEIGHBORS,
    GUIDANCE_SCALE,
    epoch="last",
    seed=VALIDATION_SEED,
)

model_filename = f"model_last_{run_name}.pth"
save_path = os.path.join(savedir, model_filename)
model_to_save = model.module if hasattr(model, "module") else model
torch.save(model_to_save.state_dict(), save_path)  # type: ignore

if best_model is not None:
    model_filename = f"model_best_{run_name}.pth"
    save_path = os.path.join(savedir, model_filename)

    torch.save(best_model, save_path)

    print(f"Model saved to {save_path}")

    eval_model.load_state_dict(best_model)  # type: ignore
    generate_images(
        eval_model,
        figs_dir,
        val_loader,
        K_NEIGHBORS,
        GUIDANCE_SCALE,
        epoch="last_after_loading",
        seed=VALIDATION_SEED,
    )


wandb.finish()
