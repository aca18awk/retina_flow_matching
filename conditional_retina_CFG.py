import os
import torch
import copy
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchcfm.conditional_flow_matching import *
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet import UNetModel
import torchdiffeq
from torchdyn.core import NeuralODE
from retina_dataset import GlaucomaHarvardDataset
from torch.utils.data import DataLoader
import wandb
import numpy as np

# --- Configuration ---
savedir = "models/cond_retina" 
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 32
n_epochs = 200
IMG_SIZE = 128
NUM_CLASSES = 4
NO_OF_CHANNELS_IMG = 3
VALIDATION_SEED = 42 
MAX_PLATEAU = 200
NUM_CHANNELS_U_NET =128
NUM_RES_BLOCKS_U_NET=2
CHANNEL_MULT=(1, 2, 4, 8) # Deep semantics: 128->256->512->1024
ATTENTION_RESOLUTIONS="32, 16, 8"
TRAIN_AUGMENTATION = True

# --- WandB Init ---
logger = wandb.init(
    project="flow_matching",
    config={
        "epoch": n_epochs,
        "img_size": IMG_SIZE,
        "max_plateau": MAX_PLATEAU,
        "batch_size": batch_size,
        "num_channels": NUM_CHANNELS_U_NET, 
        "num_res_blocks": NUM_RES_BLOCKS_U_NET, 
        "lr": 1e-4,
        "channel_mult":CHANNEL_MULT, 
        "attention_resolutions":ATTENTION_RESOLUTIONS,
        "data augmentation": "geometric only",
        "classifier free guidance": True
    },
    resume="allow",
)

run_name = logger.name 
print(f"Run Name: {run_name}")

transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform1 = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
])

# --- Datasets ---
transform_train = transform1 if TRAIN_AUGMENTATION else transform
train_dataset = GlaucomaHarvardDataset("train", transform_train, augmentation=TRAIN_AUGMENTATION)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = GlaucomaHarvardDataset("validation", transform, augmentation=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
    channel_mult=CHANNEL_MULT, # Deep semantics: 128->256->512->1024
    attention_resolutions=ATTENTION_RESOLUTIONS, # Essential for global structure
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# FM = ConditionalFlowMatcher(sigma=sigma)
# Users can try target FM by changing the above line by
# FM = TargetConditionalFlowMatcher(sigma=sigma)
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# --- Training State ---
best_loss = float('inf')
plateau_count = 0
best_model = None

print(f"Starting training on {device}...")

for epoch in range(n_epochs):
    model.train() 
    print(f"Starting epoch {epoch}")
    
    train_avg_loss = []
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data[0].to(device)
        real_label = data[1].to(device)
        label = real_label if np.random.random() > 0.1 else torch.full_like(real_label, fill_value=3)
        x0 = torch.randn_like(x1)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt, label)
        loss = torch.mean((vt - ut) ** 2)
        
        train_avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = torch.mean(torch.tensor(train_avg_loss))

    # --- Validation ---
    torch.manual_seed(VALIDATION_SEED) 
    model.eval()
    
    with torch.no_grad():
        val_avg_loss = []
        for i, data in enumerate(val_loader):
            x1 = data[0].to(device)
            label = data[1].to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, label)
            loss = torch.mean((vt - ut) ** 2)
            
            val_avg_loss.append(loss.item())

        val_epoch_loss = torch.mean(torch.tensor(val_avg_loss))

    print(f"Epoch: {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
    logger.log({
        "epoch": epoch, 
        "val_loss": val_epoch_loss, 
        "train_loss": epoch_loss
    })

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

print("\nTraining complete. Loading best weights and generating images...")

model_filename = f"model_last_{run_name}.pth"
save_path = os.path.join(savedir, model_filename)
torch.save(model.state_dict(), save_path)
if best_model is not None:
    model_filename = f"model_best_{run_name}.pth"
    save_path = os.path.join(savedir, model_filename)
    
    torch.save(best_model, save_path)
    model.load_state_dict(best_model)
    print(f"Model saved to {save_path}")


n_samples_per_class = 10
total_samples = (NUM_CLASSES -1) * n_samples_per_class 
generated_class_list = torch.arange((NUM_CLASSES -1), device=device).repeat_interleave(n_samples_per_class)
no_label_list = torch.full_like(generated_class_list, fill_value=3)
GUIDANCE_SCALE = 5

def guided_vector_field(t, x):
    t_vector = torch.full((x.shape[0],), fill_value=t, device=x.device)
    v_cond = model.forward(t_vector, x, generated_class_list)
    v_uncond = model.forward(t_vector, x, no_label_list)
    v_final = v_uncond + GUIDANCE_SCALE * (v_cond - v_uncond)
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
image_filename = f"retina_gen_{run_name}.png"
output_image_path = os.path.join(savedir, image_filename)

img.save(output_image_path)
print(f"Saved generated image to {output_image_path}")

wandb.finish()