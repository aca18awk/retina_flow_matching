import os

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --- Helper Classes (Same as before) ---


class PadToSquare:
    """Pads the image with black pixels to make it a perfect square."""

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


class CropToFundus:
    """Optimized cropping to remove black borders."""

    def __init__(self, tolerance=10, downscale_size=512):
        self.tolerance = tolerance
        self.downscale_size = downscale_size

    def __call__(self, img):
        w, h = img.size
        if min(w, h) < self.downscale_size:
            img_small = img
            scale_x, scale_y = 1.0, 1.0
        else:
            img_small = img.resize(
                (self.downscale_size, self.downscale_size), resample=Image.NEAREST
            )
            scale_x = w / self.downscale_size
            scale_y = h / self.downscale_size

        img_np = np.array(img_small)
        mask = img_np > self.tolerance
        if img_np.ndim == 3:
            has_content = mask.any(axis=2)
        else:
            has_content = mask

        if not has_content.any():
            return img

        rows = np.any(has_content, axis=1)
        cols = np.any(has_content, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        real_cmin = int(np.floor(cmin * scale_x))
        real_rmin = int(np.floor(rmin * scale_y))
        real_cmax = int(np.ceil((cmax + 1) * scale_x))
        real_rmax = int(np.ceil((rmax + 1) * scale_y))

        return img.crop(
            (max(0, real_cmin), max(0, real_rmin), min(w, real_cmax), min(h, real_rmax))
        )


# --- Main Messidor Class ---


class MessidorDataset(Dataset):
    def __init__(
        self,
        root_dir: str = "/vol/biomedic3/awk24/datasets/Messidor2",
        csv_path: str = None,
        img_size: int = 256,
    ):
        """
        Args:
            root_dir: Path to the folder containing images.
            csv_path: Path to the CSV with labels (id_code, diagnosis, etc.).
            img_size: Target size for resizing.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = {}  # Map filename -> dictionary of labels

        # 1. Load CSV Data (if provided)
        if csv_path and os.path.exists(csv_path):
            print(f"Loading labels from {csv_path}...")
            df = pd.read_csv(csv_path)
            # Ensure we can match filename (e.g. "IM002584.JPG" or just "IM002584") to the row
            # We assume the CSV 'id_code' column might match the filename
            for _, row in df.iterrows():
                # Store all column data for this ID
                key = str(row["id_code"]).strip()
                self.labels[key] = row.to_dict()

        # 2. Collect All Images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        print(f"Scanning files in {root_dir}...")

        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)

        self.image_paths.sort()
        print(f"Found {len(self.image_paths)} images.")

        # 3. Define Preprocessing Pipeline
        self.pre_process = transforms.Compose(
            [
                CropToFundus(tolerance=10, downscale_size=512),
                PadToSquare(),
                transforms.Resize([img_size, img_size]),
            ]
        )

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)

        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (128, 128))

        # 2. Preprocess
        image = self.pre_process(image)
        image_tensor = self.to_tensor(image)

        # 3. Retrieve Labels (if available)
        # We try to match exact filename, or filename without extension
        label_data = {}

        # Try finding key 'IM002584.JPG' or 'IM002584'
        if filename in self.labels:
            label_data = self.labels[filename]
        elif os.path.splitext(filename)[0] in self.labels:
            label_data = self.labels[os.path.splitext(filename)[0]]

        # Extract specific diagnosis if present, else default to -1
        diagnosis = label_data.get("diagnosis", -1)

        # Return: Image Tensor, The Diagnosis Label, The Filename (for moving files later)
        return image_tensor, diagnosis, filename


# --- Usage Example for your selection task ---
if __name__ == "__main__":
    # Example CSV path - update this to your actual CSV location
    csv_file = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"

    # Initialize
    ds = MessidorDataset(
        root_dir="/vol/biomedic3/awk24/datasets/Messidor2",
        csv_path=csv_file,  # Set to None if you don't have the CSV ready yet
        img_size=256,
    )

    # Iterate to plot or select
    # This loop is just to show how you access data to make your "hospital_B" selection
    print("Inspecting first 5 items...")
    for i in range(5):
        img, diagnosis, fname = ds[i]
        print(f"File: {fname} | Diagnosis: {diagnosis} | Shape: {img.shape}")
