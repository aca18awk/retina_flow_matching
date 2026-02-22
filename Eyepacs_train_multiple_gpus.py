import copy
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from Eyepacs_class import EyepacsDataset
from Eyepacs_generate_images import generate_images
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import your custom modules
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from torchcfm.models.unet import UNetModel


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def main():
    setup()

    # 1. DDP Environment Variables (Set automatically by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- Configuration ---
    savedir = "models/21_Feb_Eyepacs_dinov3_no_labels"
    figs_dir = os.path.join(savedir, "figs")

    # Hyperparams
    VALIDATION_SEED = 42
    GUIDANCE_SCALE = 3.0
    N_EPOCHS = 400
    BATCH_SIZE = 42  # Per GPU (Total effective batch = 126 * 3 = 378)
    LEARNING_RATE = 0.0001
    K_NEIGHBORS = 2
    IMG_SIZE = 128

    NUM_CHANNELS_U_NET = 128
    NUM_RES_BLOCKS_U_NET = 2
    CHANNEL_MULT = (1, 2, 4, 8)  # Deep semantics: 128->256->512->1024
    ATTENTION_RESOLUTIONS = "32, 16, 8"
    NO_OF_CHANNELS_IMG = 3

    # Only Rank 0 creates directories/logs
    run_name = "FSFM_Run"
    if global_rank == 0:
        os.makedirs(figs_dir, exist_ok=True)
        # WandB only on Rank 0 to prevent 3 separate runs being logged
        logger = wandb.init(
            project="flow_matching_eyepacs",
            config={
                "type": "FSFM_Latent_Conditioning",
                "k_neighbors": K_NEIGHBORS,
                "lr": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "image_size": IMG_SIZE,
                "mixed_precision": True,
                "weight decay": False,
                "indices_file": "Eyepacs_train_indices_RETFOUND_dinov2",
            },
        )
        run_name = logger.name or "FSFM_Run"

    # --- Data Setup ---
    # Define paths
    source_dir = "/vol/biomedic3/awk24/datasets/EYEPACS_256"
    local_scratch = os.environ.get("TMPDIR", "/tmp")
    dataset_loc = os.path.join(local_scratch, "EYEPACS_256")

    # Sync Barrier: Wait for Rank 0 to copy data if needed
    if global_rank == 0:
        if not os.path.exists(dataset_loc):
            print(f"Rank 0: Copying data to {dataset_loc}...")
            os.system(f"cp -r {source_dir} {local_scratch}")
    dist.barrier()

    # Init Dataset (This will load data into RAM on each process)
    # Since we have 375GB RAM, loading 6GB three times (18GB) is trivial.
    train_dataset = EyepacsDataset(
        root=dataset_loc,
        purpose="train",
        img_size=IMG_SIZE,
        k_neighbours=K_NEIGHBORS,
    )
    # DDP Sampler (Crucial for splitting data correctly)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = EyepacsDataset(
        root=dataset_loc,
        purpose="validation",
        img_size=IMG_SIZE,
        k_neighbours=K_NEIGHBORS,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model Setup ---
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
        # nn.Linear(512, time_embed_dim),  # type: ignore
        # NOTE: FOR DINOV2 EMBEDDINGS
        nn.Linear(1024, time_embed_dim),  # type: ignore
        nn.SiLU(),
        nn.Linear(time_embed_dim, time_embed_dim),  # type: ignore
    ).to(device)

    # DDP Wrapper
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        # weight_decay=1e-5,
    )
    scaler = GradScaler()
    FM = TargetConditionalFlowMatcher(sigma=0.0)

    best_loss = float("inf")
    best_model = None

    start_epoch = 0
    # resume_path = os.path.join(savedir, "model_30.pth")  # Make sure this file exists!

    # if os.path.exists(resume_path):
    #     # We must use map_location to load correctly on DDP
    #     checkpoint = torch.load(resume_path, map_location=device)

    #     # Handle the "module." prefix that DDP adds
    #     # If your saved model has "module.conv..." keys, loading into model.module is redundant
    #     # But usually, just loading the state_dict works if keys match.
    #     try:
    #         model.module.load_state_dict(checkpoint)
    #     except:
    #         # Fallback if keys don't match exactly (sometimes DDP adds/removes prefixes)
    #         model.load_state_dict(checkpoint)

    #     start_epoch = 31  # Skip the first 20
    #     print(f"[Rank {global_rank}] Resuming from Epoch 20!")

    # print(f"[Rank {global_rank}] Ready to train.")

    for epoch in range(start_epoch, N_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)  # Essential for shuffling

        train_avg_loss = []

        for i, (x1, y1) in enumerate(train_loader):
            x1 = x1.to(device, non_blocking=True)
            y1 = y1.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Masking logic
            mask = torch.rand(x1.shape[0], device=device) > 0.1
            mask = mask.view(-1, 1)

            null_label = torch.zeros_like(y1)
            label = torch.where(mask, y1, null_label)

            x0 = torch.randn_like(x1)

            # Mixed Precision
            with autocast():
                t, xt, ut, *_ = FM.sample_location_and_conditional_flow(x0, x1)
                vt = model(t, xt, label)
                loss = torch.mean((vt - ut) ** 2)

            if torch.isnan(loss):
                print(
                    f"[Rank {global_rank}] WARNING: Loss is NaN at step {i}! Zeroing loss to avoid DDP hang."
                )
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if loss.item() > 0:
                train_avg_loss.append(loss.item())

        # This averages the loss across all 3 GPUs so WandB shows the TRUE global loss
        local_loss = torch.tensor(np.mean(train_avg_loss), device=device)
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        epoch_loss = local_loss.item() / world_size

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

        # Sync Validation Loss as well
        local_val = torch.tensor(np.mean(val_avg_loss), device=device)
        dist.all_reduce(local_val, op=dist.ReduceOp.SUM)
        val_epoch_loss = local_val.item() / world_size

        if global_rank == 0:
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Val: {val_epoch_loss:.4f}")
            wandb.log({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_epoch_loss})

            # Save Checkpoints & Generate Images
            if epoch % 10 == 0:
                print(">>>>", i)
                model.eval()
                # Access underlying model for saving
                model_to_save = model.module
                torch.save(model_to_save.state_dict(), os.path.join(savedir, f"model_{epoch}.pth"))

                # Validation Generation
                with torch.no_grad():
                    generate_images(
                        model_to_save,
                        figs_dir,
                        val_loader,
                        K_NEIGHBORS,
                        GUIDANCE_SCALE,
                        epoch=str(epoch),
                        seed=VALIDATION_SEED,
                    )
            # --- Early Stopping ---
            if val_epoch_loss <= best_loss:
                best_loss = val_epoch_loss

                model_to_save = model.module if hasattr(model, "module") else model
                best_model = copy.deepcopy(model_to_save.state_dict())  # type: ignore
                print(f"  --> New Best Model! (Val Loss: {best_loss:.4f})")

            if epoch % 100 == 0:
                if best_model is not None:
                    model_filename = f"model_best_{run_name}_{epoch}.pth"
                    save_path = os.path.join(savedir, model_filename)
                    torch.save(best_model, save_path)

    print("\nTraining complete.")

    if global_rank == 0:
        model_filename = f"model_last_{run_name}.pth"
        save_path = os.path.join(savedir, model_filename)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), save_path)  # type: ignore

        if best_model is not None:
            model_filename = f"model_best_{run_name}.pth"
            save_path = os.path.join(savedir, model_filename)

            torch.save(best_model, save_path)

            print(f"Model saved to {save_path}")

            eval_model = model.module if hasattr(model, "module") else model

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

    cleanup()


if __name__ == "__main__":
    main()
