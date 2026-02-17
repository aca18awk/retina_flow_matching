import copy
import os

import torch
import wandb
from MNIST_generate_images import generate_images
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms

# from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, TargetConditionalFlowMatcher
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

# --- Configuration ---
savedir = "models/13_Feb_MNIST_lr_0001_targetCFM"
os.makedirs(savedir, exist_ok=True)

figs_dir = os.path.join(savedir, "figs")
os.makedirs(figs_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
VALIDATION_SEED = 42
GUIDANCE_SCALE = 3

# training params
TRAIN_AUGMENTATION = True
N_EPOCHS = 50
BATCH_SIZE = 128
MAX_PLATEAU = 2000
LEARNING_RATE = 0.0001

# model params
NUM_CHANNELS_U_NET = 64
NUM_RES_BLOCKS_U_NET = 2
CHANNEL_MULT = (1, 2, 4)
ATTENTION_RESOLUTIONS = "16, 8"

# dataset params
IMG_SIZE = 32
NUM_CLASSES = 11
NO_OF_CHANNELS_IMG = 1


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


# --- WandB Init ---
logger = wandb.init(
    project="flow_matching_mnist",
    config={
        "epoch": N_EPOCHS,
        "img_size": IMG_SIZE,
        "max_plateau": MAX_PLATEAU,
        "batch_size": BATCH_SIZE,
        "num_channels": NUM_CHANNELS_U_NET,
        "num_res_blocks": NUM_RES_BLOCKS_U_NET,
        "lr": LEARNING_RATE,
        "channel_mult": CHANNEL_MULT,
        "attention_resolutions": ATTENTION_RESOLUTIONS,
        "data augmentation": "all",
        "classifier free guidance": True,
    },
    resume="allow",
)

run_name = logger.name or " "
print(f"Run Name: {run_name}")

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# --- Datasets ---
train_dataset = datasets.MNIST(
    "../../../../datasets/MNIST",
    train=True,
    download=True,
    transform=transform,
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = datasets.MNIST(
    "../../../../datasets/MNIST",
    train=False,
    download=True,
    transform=transform,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


#################################
#    Class Conditional CFM
#################################

sigma = 0.0
model = UNetModel(
    dim=(NO_OF_CHANNELS_IMG, IMG_SIZE, IMG_SIZE),
    num_channels=NUM_CHANNELS_U_NET,
    num_res_blocks=NUM_RES_BLOCKS_U_NET,
    num_classes=NUM_CLASSES,
    class_cond=True,
    channel_mult=CHANNEL_MULT,
    attention_resolutions=ATTENTION_RESOLUTIONS,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
FM = TargetConditionalFlowMatcher(sigma=0.0)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# --- Training State ---
best_loss = float("inf")
plateau_count = 0
best_model = None

print(f"Starting training on {device}...")

for epoch in range(N_EPOCHS):
    model.train()
    print(f"Starting epoch {epoch}")

    train_avg_loss = []
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data[0].to(device)

        real_label = data[1].to(device)

        mask = torch.rand(x1.shape[0], device=device) > 0.1
        null_label = torch.full_like(real_label, fill_value=NUM_CLASSES - 1)
        label = torch.where(mask, real_label, null_label)

        x0 = torch.randn_like(x1)

        t, xt, ut, *_ = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt, label)
        loss = torch.mean((vt - ut) ** 2)

        train_avg_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    epoch_loss = torch.mean(torch.tensor(train_avg_loss))

    # --- Validation ---
    model.eval()

    with torch.no_grad():
        val_avg_loss = []
        for i, data in enumerate(val_loader):
            x1 = data[0].to(device)
            label = data[1].to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut, *_ = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, label)
            loss = torch.mean((vt - ut) ** 2)

            val_avg_loss.append(loss.item())

        val_epoch_loss = torch.mean(torch.tensor(val_avg_loss))

    print(f"Epoch: {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
    logger.log({"epoch": epoch, "val_loss": val_epoch_loss, "train_loss": epoch_loss})

    if epoch % 10 == 0:
        print(f"Generating visualization for epoch {epoch}...")

        generate_images(model, figs_dir, GUIDANCE_SCALE, str(epoch), run_name)

        model_filename = f"model_{epoch}_{run_name}.pth"
        save_path = os.path.join(savedir, model_filename)
        save_model(model, save_path)

    # --- Early Stopping ---
    if val_epoch_loss <= best_loss:
        plateau_count = 0
        best_loss = val_epoch_loss
        best_model = copy.deepcopy(model.state_dict())
        print(f"  --> New Best Model! (Val Loss: {best_loss:.4f})")
    else:
        plateau_count += 1
        print(f"  --> No improvement. Patience: {plateau_count}/{MAX_PLATEAU}")

    if plateau_count >= MAX_PLATEAU:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")

generate_images(model, figs_dir, GUIDANCE_SCALE, "last", run_name)

model_filename = f"model_last_{run_name}.pth"
save_path = os.path.join(savedir, model_filename)
save_model(model, save_path)
if best_model is not None:
    model_filename = f"model_best_{run_name}.pth"
    save_path = os.path.join(savedir, model_filename)

    torch.save(best_model, save_path)

    print(f"Model saved to {save_path}")

    last_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(last_model)
    generate_images(model, figs_dir, GUIDANCE_SCALE, "last_after_loading", run_name)


wandb.finish()
