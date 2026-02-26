import os
from typing import Literal

import pandas as pd
from Eyepacs_class_for_distance import RETFoundTransform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MessidorSplit = Literal["hospital_b", "test", "hidden", "validation"]


class MessidorDataset(Dataset):
    def __init__(
        self,
        purpose: MessidorSplit = "test",
        root_dir: str = "/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path: str = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=128,
        useRetFoundPreprocessing=False,
    ):
        """
        Args:
            purpose: One of "hospital_b", "test", or "hidden".
            root_dir: Path to the parent folder containing the processed subfolders.
            csv_path: Path to the original CSV with labels.
        """
        self.purpose = purpose
        self.image_paths = []
        self.labels = {}
        self.img_size = img_size
        self.useRetFoundPreprocessing = useRetFoundPreprocessing

        # 1. Select Subfolder based on Purpose
        if purpose == "hospital_b":
            subfolder = "hospital_b"
        elif purpose == "test":
            subfolder = "test"
        elif purpose == "hidden":
            subfolder = "hidden_classifier_data"
        elif purpose == "validation":
            subfolder = "validation"
        else:
            raise ValueError(
                f"Invalid purpose '{purpose}'. Must be 'hospital_b', 'test', or 'hidden'."
            )

        self.data_dir = os.path.join(root_dir, subfolder)

        # 2. Load CSV Data (Labels)
        # We still need the CSV to map the filename to the diagnosis
        if csv_path and os.path.exists(csv_path):
            print(f"Loading labels from {csv_path}...")
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                # We assume 'id_code' matches the filename stem (e.g. IM0001)
                key = str(row["id_code"]).strip()
                self.labels[key] = row.to_dict()
        else:
            print(f"Warning: CSV path {csv_path} not found. Labels will be -1.")

        # 3. Collect Images from the Specific Subfolder
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        print(f"Scanning files in {self.data_dir}...")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Directory {self.data_dir} does not exist. Did you run the split script?"
            )

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))

        self.image_paths.sort()
        print(f"Found {len(self.image_paths)} images for split '{purpose}'.")

        # 4. Define Transform
        # Images are already 256x256, Cropped and Padded.
        # We just need to convert to Tensor and Normalize to [-1, 1] range.
        self.transform = transforms.Compose(
            [
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.retfound_transform = RETFoundTransform(img_size=img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")

        if self.useRetFoundPreprocessing:
            image = self.retfound_transform(image)
        else:
            image = self.transform(image)

        # 3. Retrieve Label
        label_data = {}
        # Try finding key with or without extension
        if filename in self.labels:
            label_data = self.labels[filename]
        elif os.path.splitext(filename)[0] in self.labels:
            label_data = self.labels[os.path.splitext(filename)[0]]

        diagnosis = label_data.get("diagnosis", -1)

        return image, diagnosis, filename


# --- Testing the Logic ---
if __name__ == "__main__":
    # Configuration
    processed_root = "/vol/biomedic3/awk24/datasets/Messidor2_256"
    csv_file = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"

    print("--- Testing Hospital B Split ---")
    try:
        ds_b = MessidorDataset(purpose="hospital_b", root_dir=processed_root, csv_path=csv_file)
        if len(ds_b) > 0:
            img, label, fname = ds_b[0]
            print(f"Sample: {fname} | Label: {label} | Tensor Shape: {img.shape}")
            print(f"Min: {img.min():.2f} | Max: {img.max():.2f} (Should be approx -1 to 1)")
    except Exception as e:
        print(e)

    print("\n--- Testing Test Split ---")
    try:
        ds_test = MessidorDataset(purpose="test", root_dir=processed_root, csv_path=csv_file)
        print(f"Test Set Size: {len(ds_test)}")
    except Exception as e:
        print(e)
