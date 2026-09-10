import json
import os
from typing import Literal, Optional

import pandas as pd
from Eyepacs_class_for_distance import RETFoundTransform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MessidorSplit = Literal["validation", "reference", "test"]
MessidorExperiment = Literal["all", "dilated", "nondilated"]
MessidorSeed = Literal["seed_A", "seed_B", "seed_C"]

_EXPERIMENT_TO_JSON = {
    "all": "messidor2_splits_all.json",
    "dilated": "messidor2_splits_dilated.json",
    "nondilated": "messidor2_splits_nondilated.json",
}

_CLASSES = ["G0", "G1", "G2", "G3", "G4"]


class MessidorDataset(Dataset):
    def __init__(
        self,
        purpose: MessidorSplit = "test",
        experiment: MessidorExperiment = "all",
        root_dir: str = "/vol/biomedic3/awk24/datasets/Messidor2_256",
        json_dir: str = "/vol/biomedic3/awk24/datasets/Messidor2",
        csv_path: str = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size: int = 128,
        N: Optional[int] = None,
        seed: MessidorSeed = "seed_A",
        useRetFoundPreprocessing: bool = False,
    ):
        """
        Args:
            purpose: One of "validation", "reference", or "test".
            experiment: Which pupil-dilation subset to use — "all", "dilated", or "nondilated".
            root_dir: Root folder containing Messidor2_256 image subfolders.
            json_dir: Folder containing the messidor2_splits_*.json files.
            csv_path: CSV with id_code → diagnosis labels.
            img_size: Resize target (square).
            N: Number of reference images to sample (reference purpose only).
                Balanced across G0–G4; deficit classes are padded from surplus ones.
            seed: Ordering seed for reference sampling — "seed_A", "seed_B", or "seed_C".
            useRetFoundPreprocessing: Use RETFound transform instead of standard one.
        """
        self.purpose = purpose
        self.img_size = img_size
        self.useRetFoundPreprocessing = useRetFoundPreprocessing

        # Load split JSON
        json_path = os.path.join(json_dir, _EXPERIMENT_TO_JSON[experiment])
        with open(json_path) as f:
            splits = json.load(f)

        # Build filename → absolute path index: root dir itself + any subdirectories
        file_index: dict = {}
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                for fname in os.listdir(entry_path):
                    file_index[fname] = os.path.join(entry_path, fname)
            else:
                file_index[entry] = entry_path

        # Resolve filenames for this split
        if purpose == "validation":
            filenames = splits["validation"]
        elif purpose == "test":
            filenames = splits["test"]
        elif purpose == "reference":
            filenames = self._select_reference(splits["hospital_b_orderings"], seed, N)
        else:
            raise ValueError(
                f"Invalid purpose '{purpose}'. Must be 'validation', 'reference', or 'test'."
            )

        self.image_paths = []
        for fname in filenames:
            if fname in file_index:
                self.image_paths.append(file_index[fname])
            else:
                print(f"Warning: {fname} not found under {root_dir}")

        print(
            f"Found {len(self.image_paths)} images "
            f"(purpose='{purpose}', experiment='{experiment}'"
            + (f", N={N}, seed='{seed}'" if purpose == "reference" else "")
            + ")."
        )

        # Load labels from CSV
        self.labels: dict = {}
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                key = str(row["id_code"]).strip()
                self.labels[key] = row.to_dict()
        else:
            print(f"Warning: CSV path {csv_path} not found. Labels will be -1.")

        self.transform = transforms.Compose(
            [
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.retfound_transform = RETFoundTransform(img_size=img_size)

    @staticmethod
    def _select_reference(
        orderings: dict,
        seed: str,
        N: Optional[int],
    ) -> list:
        """Return up to N filenames from hospital_b_orderings, balanced across G0–G4.

        For each class we take the first N//5 entries. Classes with fewer images
        contribute all they have; the shortfall is filled from surplus classes
        (iterating G0 → G4 in order).
        """
        seed_data = orderings[seed]

        if N is None:
            return [f for g in _CLASSES for f in seed_data.get(g, [])]

        per_class = N // len(_CLASSES)
        selected: list = []
        surplus: dict = {g: [] for g in _CLASSES}

        for g in _CLASSES:
            files = seed_data.get(g, [])
            take = min(per_class, len(files))
            selected.extend(files[:take])
            if len(files) > take:
                surplus[g] = files[take:]

        remaining = N - len(selected)
        for g in _CLASSES:
            if remaining <= 0:
                break
            if surplus[g]:
                extras = surplus[g][:remaining]
                selected.extend(extras)
                remaining -= len(extras)

        return selected

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")

        if self.useRetFoundPreprocessing:
            image = self.retfound_transform(image)
        else:
            image = self.transform(image)

        label_data = {}
        if filename in self.labels:
            label_data = self.labels[filename]
        elif os.path.splitext(filename)[0] in self.labels:
            label_data = self.labels[os.path.splitext(filename)[0]]

        diagnosis = label_data.get("diagnosis", -1)
        return image, diagnosis, filename


# --- Testing the Logic ---
if __name__ == "__main__":
    processed_root = "/vol/biomedic3/awk24/datasets/Messidor2"
    json_dir = "/vol/biomedic3/awk24/datasets/Messidor2"
    csv_file = "/vol/biomedic3/awk24/datasets/Messidor2/messidor2_master.csv"

    for exp in ("all", "dilated", "nondilated"):
        print(f"\n=== experiment='{exp}' ===")

        ds_val = MessidorDataset(purpose="validation", experiment=exp,
                                 root_dir=processed_root, json_dir=json_dir, csv_path=csv_file)
        print(f"  validation: {len(ds_val)} images")

        ds_test = MessidorDataset(purpose="test", experiment=exp,
                                  root_dir=processed_root, json_dir=json_dir, csv_path=csv_file)
        print(f"  test:       {len(ds_test)} images")

        for seed in ("seed_A", "seed_B", "seed_C"):
            ds_ref = MessidorDataset(purpose="reference", experiment=exp, seed=seed, N=50,
                                     root_dir=processed_root, json_dir=json_dir, csv_path=csv_file)
            img, label, fname = ds_ref[0]
            print(f"  reference N=50 {seed}: {len(ds_ref)} images | "
                  f"first={fname} label={label} shape={img.shape}")
