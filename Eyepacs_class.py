import os
from typing import List, Literal

import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch import cat, distributions, load, ones, randperm, stack
from torch.utils.data import Dataset
from torchvision import transforms

DatasetSplit = Literal["test", "train", "validation"]

ROOT_PATH = "/vol/biomedic3/awk24/datasets/EYEPACS"

import torchvision.transforms.functional as TF


class PadToSquare:
    """
    Pads the image with black pixels to make it a perfect square.
    Crucial for 'cut' images so they don't get squashed when resizing.
    """

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)

        # Calculate padding needed to make it square
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top

        # Apply padding (fill=0 is black)
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)  # type: ignore


class CropToFundus:
    """
    Detects the non-black content (the eye) and crops the image to that bounding box.
    This unifies the 'zoom' level across all images.
    """

    def __init__(self, tolerance=10):
        self.tolerance = tolerance

    def __call__(self, img):
        # Convert PIL to Numpy
        if not isinstance(img, np.ndarray):
            img_np = np.array(img)
        else:
            img_np = img

        # Create a mask where pixels are greater than threshold
        # (We check all channels; if any channel > tolerance, it's part of the eye)
        mask = img_np > self.tolerance

        # If image is almost entirely black, return original (prevents crashing)
        if img_np.ndim == 3:
            has_content = mask.any(axis=2)
        else:
            has_content = mask

        if not has_content.any():
            return img

        # Find the bounding box of the 'True' values in the mask
        rows = np.any(has_content, axis=1)
        cols = np.any(has_content, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the image
        cropped_img = img_np[rmin : rmax + 1, cmin : cmax + 1]

        # Convert back to PIL
        return Image.fromarray(cropped_img)


class EyepacsDataset(Dataset):
    def __init__(self, purpose: DatasetSplit = "train", img_size=128, k_neighbours=3):
        self.purpose = purpose
        self.image_paths: List[str] = []
        self.k_neighbours = k_neighbours

        # --- 1. Path Selection Logic ---
        # If train -> use 'train' folder
        # If validation -> use 'test' folder (but only 2000 images)
        if purpose == "train":
            folder_name = "train"
        else:
            folder_name = "test"

        # Load Cached Data
        if purpose == "train":
            self.features = load("Eyepacs_train_features.pt", map_location="cpu")
            self.indices = load("Eyepacs_train_indices.pt", map_location="cpu")
        else:
            # self.features = None
            # self.indices = None
            self.features = load("Eyepacs_val_features.pt", map_location="cpu")
            self.indices = load("Eyepacs_val_indices.pt", map_location="cpu")

        data_path = os.path.join(ROOT_PATH, folder_name)

        # --- 2. File Collection (Recursive walk) ---
        # We manually walk the directory to find images.
        # This solves the issue if 'test' is flat and 'train' is structured.
        valid_extensions = {".jpg", ".jpeg", ".png"}

        print(f"Scanning files in {data_path}...")
        for root, _, files in os.walk(data_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))

        # Sort to ensure reproducibility across runs
        self.image_paths.sort()

        # --- 3. Limit Validation Set ---
        if purpose == "validation":
            # Only take the first 2000 images from the test folder
            limit = 2000
            if len(self.image_paths) > limit:
                self.image_paths = self.image_paths[:limit]
                print(f"Validation mode: Limiting to first {limit} images from {folder_name}.")

        print(f"Found {len(self.image_paths)} images for split '{purpose}'.")

        # --- 4. Pipeline Definitions ---
        self.pre_process = transforms.Compose(
            [
                CropToFundus(tolerance=10),  # <--- STEP 1: Remove black borders
                PadToSquare(),
                transforms.Resize([img_size, img_size]),
                # CLAHE_Transform(clip_limit=1.5), # Uncomment if desired
            ]
        )

        # Removing augmentations since I'm pre-calculating nearest neighbours for each image before training
        # self.train_augment = transforms.Compose(
        #     [
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomApply(
        #             [
        #                 transforms.ColorJitter(
        #                     brightness=0.1,  # type: ignore
        #                     contrast=0.1,  # type: ignore
        #                     saturation=0.1,  # type: ignore
        #                     hue=0.01,  # type: ignore
        #                 )
        #             ],
        #             p=0.8,
        #         ),
        #         # transforms.RandomApply(
        #         #     [transforms.RandomAffine(degrees=15, scale=(1.12, 1.2), shear=0)],
        #         #     p=0.5,
        #         # ),
        #     ]
        # )

        # Normalize to [-1, 1] for flow matching / diffusion
        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Load Image (Convert to RGB to handle grayscale issues)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image or handle error appropriately
            # Here we just generate a black image to prevent crashing
            image = Image.new("RGB", (128, 128))

        # 1. Base Preprocessing
        image = self.pre_process(image)

        # 3. Convert to Tensor [-1, 1]
        image = self.to_tensor_norm(image)

        # Dynamic Barycentric Sampling
        if self.features is not None and self.indices is not None:
            # 1. Get neighbors for this image
            neighbor_idxs = self.indices[index]  # [K]

            # 2. Select (K) neighbors randomly
            num_neighbors_to_sample = self.k_neighbours
            perm = randperm(len(neighbor_idxs))[:num_neighbors_to_sample]
            selected_neighbor_idxs = neighbor_idxs[perm]

            # 3. Gather Features: [Self] + [Neighbors]
            # self.features[index] is the feature of the current image
            self_feat = self.features[index].unsqueeze(0)  # [1, 512]
            neighbor_feats = self.features[selected_neighbor_idxs]  # [K-1, 512]

            # Combine them to form the simplex (triangle/tetrahedron)
            vectors = cat([self_feat, neighbor_feats], dim=0)  # [K, 512]

            # 4. Sample Dirichlet weights (Barycentric coords)
            weights = distributions.Dirichlet(ones(self.k_neighbours + 1)).sample()  # [3]

            # 5. Weighted Average
            condition = (vectors * weights.unsqueeze(1)).sum(dim=0)  # [512]

            return image, condition

        # RETURN ONLY IMAGE
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
