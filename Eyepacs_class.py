import os
from typing import List, Literal, Optional

import numpy as np
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from torch import cat, distributions, empty, load, ones, randperm, stack, tensor
from torch.utils.data import Dataset
from torchvision import transforms

DatasetSplit = Literal["test", "train", "validation"]

# Default fallback (only used if no root is provided)
DEFAULT_ROOT_PATH = "/vol/biomedic3/awk24/datasets/EYEPACS_256"


class PadToSquare:
    """
    Pads the image with black pixels to make it a perfect square.
    """

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


class CropToFundus:
    """
    OPTIMIZED: Detects the eye content on a downscaled version (thumbnail)
    to save CPU cycles, then crops the original high-res image.
    """

    def __init__(self, tolerance=10, downscale_size=512):
        self.tolerance = tolerance
        self.downscale_size = downscale_size

    def __call__(self, img):
        # 1. Work on a tiny thumbnail to find the mask fast
        w, h = img.size

        # If image is already small, just process it normally
        if min(w, h) < self.downscale_size:
            scale_x, scale_y = 1.0, 1.0
            img_small = img
        else:
            # Resize uses NEAREST for speed, we just need the black/content boundary
            img_small = img.resize(
                (self.downscale_size, self.downscale_size), resample=Image.NEAREST
            )
            scale_x = w / self.downscale_size
            scale_y = h / self.downscale_size

        img_np = np.array(img_small)

        # 2. Create Mask (on small image)
        mask = img_np > self.tolerance
        if img_np.ndim == 3:
            has_content = mask.any(axis=2)
        else:
            has_content = mask

        if not has_content.any():
            return img

        # 3. Find Bounding Box (on small image)
        rows = np.any(has_content, axis=1)
        cols = np.any(has_content, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 4. Scale coordinates back to original image size
        # We add a small buffer/padding to ensure we don't cut too aggressively due to scaling
        real_cmin = int(np.floor(cmin * scale_x))
        real_rmin = int(np.floor(rmin * scale_y))
        real_cmax = int(np.ceil((cmax + 1) * scale_x))
        real_rmax = int(np.ceil((rmax + 1) * scale_y))

        # Clamp to image boundaries
        real_cmin = max(0, real_cmin)
        real_rmin = max(0, real_rmin)
        real_cmax = min(w, real_cmax)
        real_rmax = min(h, real_rmax)

        # 5. Crop the original high-res image
        return img.crop((real_cmin, real_rmin, real_cmax, real_rmax))


class EyepacsDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,  # <--- NEW ARGUMENT
        purpose: DatasetSplit = "train",
        img_size=128,
        k_neighbours=3,
    ):
        self.purpose = purpose
        self.image_paths: List[str] = []
        self.k_neighbours = k_neighbours

        # --- 1. Path Selection Logic ---
        # If 'root' is not passed, fall back to the global default
        base_path = root if root is not None else DEFAULT_ROOT_PATH

        if purpose == "train":
            folder_name = "train"
        else:
            folder_name = "validation"

        data_path = os.path.join(base_path, folder_name)

        # Load Cached Data (Features)
        # Note: Ensure these .pt files are in your working directory
        # or update the path to load them from 'base_path' if you move them too.
        if purpose == "train":
            self.features = load(
                "Eyepacs_vectors/Eyepacs_train_features_RETFOUND_dinov2.pt", map_location="cpu"
            )
            self.indices = load(
                "Eyepacs_vectors/Eyepacs_train_indices_RETFOUND_dinov2.pt", map_location="cpu"
            )
        else:
            self.features = load(
                "Eyepacs_vectors/Eyepacs_validation_features_RETFOUND_dinov2.pt",
                map_location="cpu",
            )
            self.indices = load(
                "Eyepacs_vectors/Eyepacs_validation_indices_RETFOUND_dinov2.pt",
                map_location="cpu",
            )

        # --- 2. File Collection ---
        valid_extensions = {".jpg", ".jpeg", ".png"}

        print(f"Scanning files in {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Could not find dataset at {data_path}. Did you copy it to scratch correctly?"
            )

        for root_dir, _, files in os.walk(data_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root_dir, file))

        self.image_paths.sort()

        # --- 3. Limit Validation Set ---
        if purpose == "validation":
            limit = 2000
            if len(self.image_paths) > limit:
                self.image_paths = self.image_paths[:limit]
                print(f"Validation mode: Limiting to first {limit} images.")

        print(f"Found {len(self.image_paths)} images for split '{purpose}'.")

        # --- 4. Pipeline Definitions ---
        self.pre_process = transforms.Compose(
            [
                # CropToFundus(tolerance=10, downscale_size=512),  # <--- Uses optimized version
                # PadToSquare(),
                transforms.Resize([img_size, img_size]),
            ]
        )

        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (128, 128))

        # 1. Base Preprocessing (Crop -> Pad -> Resize)
        image = self.pre_process(image)

        # 2. Convert to Tensor
        image = self.to_tensor_norm(image)

        # 3. Dynamic Barycentric Sampling
        if self.features is not None and self.indices is not None:
            neighbor_idxs = self.indices[index]
            num_neighbors_to_sample = self.k_neighbours

            # Safety check for k_neighbors
            if len(neighbor_idxs) < num_neighbors_to_sample:
                num_neighbors_to_sample = len(neighbor_idxs)

            perm = randperm(len(neighbor_idxs))[:num_neighbors_to_sample]
            selected_neighbor_idxs = neighbor_idxs[perm]

            self_feat = self.features[index].unsqueeze(0)
            neighbor_feats = self.features[selected_neighbor_idxs]

            vectors = cat([self_feat, neighbor_feats], dim=0)

            # 1. Sample the main image weight uniformly from [0.5, 0.9]
            w_main = empty(1).uniform_(0.5, 0.9)

            num_neighbors = len(vectors) - 1

            if num_neighbors > 0:
                # 2. Sample random weights for neighbors that sum to 1.0
                w_others = distributions.Dirichlet(ones(num_neighbors)).sample()

                # 3. Scale the neighbor weights so they sum to the remaining probability mass
                w_others = w_others * (1.0 - w_main)

                # 4. Combine the main weight and the neighbor weights
                weights = cat([w_main, w_others], dim=0)
            else:
                # Fallback in case a sample has 0 neighbors
                weights = tensor([1.0])

            condition = (vectors * weights.unsqueeze(1)).sum(dim=0)
            return image, condition

        return image


# --- Testing the Logic ---
if __name__ == "__main__":

    def save_preview(dataset, filename, num_images=5):
        """Helper to save a grid of the first N images from a dataset."""
        print(f"Saving preview to {filename}...")

        # 1. Collect first N images
        images = []
        for i in range(min(num_images, len(dataset))):
            item = dataset[i]

            # CHECK: If dataset returns (image, condition), unpack it
            if isinstance(item, tuple):
                img, cond = item
                images.append(img)
            else:
                images.append(item)

        if not images:
            print("No images found to save.")
            return

        # 2. Stack them into a single tensor (Batch, C, H, W)
        batch = stack(images)

        # 3. Un-normalize from [-1, 1] back to [0, 1] for viewing
        #    (img * 0.5 + 0.5)
        batch = batch * 0.5 + 0.5

        # 4. Create a grid and save
        #    nrow=5 means all 5 images in one row
        vutils.save_image(batch, filename, nrow=num_images, padding=2)
        print("Saved!")

    # --- 1. Test Training Split ---
    print("--- Loading Train ---")
    train_ds = EyepacsDataset(purpose="train")

    # Save preview
    save_preview(train_ds, "preview_train_cropToFundus_PadToSquare_final.png")

    if len(train_ds) > 0:
        img, cond = train_ds[0]
        print(f"Train Output Shape: {img.shape}")  # type: ignore
        print(f"Train Output Range: Min {img.min():.2f}, Max {img.max():.2f}")  # type: ignore
