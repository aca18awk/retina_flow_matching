import csv
import os
from typing import List, Literal, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from torch import stack
from torch.utils.data import Dataset
from torchvision import transforms

DatasetSplit = Literal["test", "train", "validation"]

# Default fallback (only used if no root is provided)
DEFAULT_ROOT_PATH = "/vol/biomedic3/awk24/datasets/EYEPACS_256"


class RETFoundTransform:
    """
    Wraps the exact PIL resizing and NumPy population standard deviation
    math used by the original RETFound authors into a PyTorch transform.
    """

    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, img):
        # 1. PIL Resize
        img = img.resize((self.img_size, self.img_size))

        # 2. NumPy Conversion and [0, 1] scaling
        img_np = np.array(img) / 255.0

        # 3. Exact NumPy Population Std (ddof=0)
        for c in range(3):
            mean = img_np[..., c].mean()
            std = img_np[..., c].std()
            if std > 0:
                img_np[..., c] = (img_np[..., c] - mean) / std
            else:
                img_np[..., c] = img_np[..., c] - mean

        # 4. To PyTorch Tensor (CHW format)
        x = torch.tensor(img_np, dtype=torch.float32)
        x = torch.einsum("hwc->chw", x)
        return x


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


class BenGrahamPreprocessing:
    """
    Applies Ben Graham's color normalization and a strict circular mask.
    This was the winning preprocessing step in the Kaggle DR competition.
    """

    def __init__(self, sigmaX=10):
        self.sigmaX = sigmaX

    def __call__(self, img):
        # Convert PIL to cv2 numpy array (RGB)
        img_np = np.array(img)

        # 1. Apply Ben Graham's Local Average Subtraction
        blurred = cv2.GaussianBlur(img_np, (0, 0), self.sigmaX)
        # Formula: image*4 - blurred*4 + 128 (maps average to gray)
        img_graham = cv2.addWeighted(img_np, 4, blurred, -4, 128)

        # 2. Apply a strict circular mask to ensure uniform black borders
        h, w, _ = img_graham.shape
        center = (w // 2, h // 2)
        radius = min(center[0], center[1])

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)

        # Mask the image (sets everything outside the circle to pure black)
        img_masked = cv2.bitwise_and(img_graham, img_graham, mask=mask)

        # Convert back to PIL
        return Image.fromarray(img_masked)


class EyepacsDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        csv_path: str = "/vol/biomedic3/awk24/datasets/EYEPACS/trainLabels.csv",  # <--- NEW ARGUMENT
        purpose: DatasetSplit = "train",
        img_size=128,
        k_neighbours=3,
        useRetFoundPreprocessing=False,
    ):
        self.purpose = purpose
        self.useRetFoundPreprocessing = useRetFoundPreprocessing
        self.image_paths: List[str] = []
        self.k_neighbours = k_neighbours
        self.labels_map = {}

        # --- 0. Load Labels from CSV ---
        if os.path.exists(csv_path):
            print(f"Loading labels from {csv_path}...")
            with open(csv_path, mode="r") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    # Maps "10_left" -> 0
                    self.labels_map[row["image"]] = int(row["level"])
        else:
            print(f"WARNING: Label CSV not found at {csv_path}. Labels will default to -1.")

        # --- 1. Path Selection Logic ---
        base_path = root if root is not None else DEFAULT_ROOT_PATH

        if purpose == "train":
            folder_name = "train"
        else:
            folder_name = "validation"

        data_path = os.path.join(base_path, folder_name)

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
                # BenGrahamPreprocessing(),
                transforms.Resize([img_size, img_size]),
            ]
        )

        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.retfound_transform = RETFoundTransform(img_size=img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Get filename without extension (e.g., "13_right" from "13_right.png")
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # Look up label, default to -1 if missing
        label = self.labels_map.get(img_name, -1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        image = Image.open(img_path).convert("RGB")
        if self.useRetFoundPreprocessing:
            image = self.retfound_transform(image)
        else:
            image = self.pre_process(image)
            image = self.to_tensor_norm(image)

        return image, label_tensor


# --- Testing the Logic ---
if __name__ == "__main__":

    def save_preview(dataset, filename, num_images=5):
        """Helper to save a grid of the first N images from a dataset."""
        print(f"Saving preview to {filename}...")

        images = []
        for i in range(min(num_images, len(dataset))):
            item = dataset[i]

            # CHECK: Unpack tuple safely based on length
            if isinstance(item, tuple):
                if len(item) == 3:
                    img, cond, label = item
                else:
                    img, label = item
                images.append(img)
            else:
                images.append(item)

        if not images:
            print("No images found to save.")
            return

        batch = stack(images)
        batch = batch * 0.5 + 0.5
        vutils.save_image(batch, filename, nrow=num_images, padding=2)
        print("Saved!")

    # --- 1. Test Training Split ---
    print("--- Loading Train ---")
    train_ds = EyepacsDataset(purpose="train")

    save_preview(train_ds, "preview_train_BenGrahamPreprocessing_final.png")

    if len(train_ds) > 0:
        item = train_ds[0]
        if isinstance(item, tuple):
            if len(item) == 3:
                img, cond, label = item
                print(f"Train Output Shape: {img.shape}, Label: {label.item()}")
            else:
                img, label = item
                print(f"Train Output Shape: {img.shape}, Label: {label.item()}")
        else:
            print("Something is wrong, dataset didn't return a tuple.")
