import numbers
import os
from collections import Counter
from enum import Enum
from typing import Literal

import cv2

# import torchvision.transforms.v2 as T
# import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from torch import empty
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DatasetSplit = Literal["test", "train", "validation"]


class DatasetClass(Enum):
    normal_control = 2
    early_glaucoma = 1
    advanced_glaucoma = 0


path = "/vol/biomedic3/awk24/datasets/Glaucoma_fundus/"


class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""

    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, "gammacorrection")

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:  # type: ignore
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]  # type: ignore
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(name)
            )

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = (
            None if self.gamma is None else float(empty(1).uniform_(self.gamma[0], self.gamma[1]))
        )
        if gamma_factor is not None:
            img = transforms.functional.adjust_gamma(img, gamma_factor, gain=1)
        return img


# --- 1. NEW: CLAHE Transform to make veins pop ---
class CLAHE_Transform:
    """Applies Contrast Limited Adaptive Histogram Equalization to the L-channel"""

    def __init__(self, clip_limit=2.0, tile_grid_size=(12, 12)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Check if grayscale or RGB
        if len(img.shape) == 2:
            # Grayscale - apply directly
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            final = clahe.apply(img)
            return Image.fromarray(final)

        # RGB - Convert to LAB, apply to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)


# --- 2. Dataset Class ---
class GlaucomaHarvardDataset(Dataset):
    def __init__(
        self, purpose: DatasetSplit = "test", transform=None, augmentation=True, gen_data_path=None
    ):
        data_path = os.path.join(path, purpose)

        # We Initialize ImageFolder WITHOUT transform here to keep raw PIL images
        # We will apply transforms manually in __getitem__
        real_data = ImageFolder(data_path, transform=None)

        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx
        self.do_augment = augmentation

        data = [real_data]
        if gen_data_path:
            generated_data = ImageFolder(gen_data_path, transform=None)
            data.append(generated_data)
            self.data = ConcatDataset(data)
        else:
            self.data = real_data

        # --- PIPELINE DEFINITIONS ---

        # 1. Base Preprocessing (Always applied)
        # CLAHE must happen on PIL images before ToTensor
        self.pre_process = transforms.Compose(
            [
                transforms.Resize([128, 128]),
                # CLAHE_Transform(clip_limit=1.5),
            ]
        )

        self.train_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomApply([GammaCorrectionTransform(gamma=0.3)], p=0.5),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.2, hue=0.02
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=15,
                            scale=(1.12, 1.2),
                            shear=0,
                        )
                    ],
                    p=0.5,
                ),
            ]
        )

        # 3. Final Conversion (Tensor + Normalize)
        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get raw PIL image and label
        image, label = self.data[index]

        # 1. Apply Resize & CLAHE (Crucial for veins)
        image = self.pre_process(image)

        # 2. Apply Augmentations (Only if training)
        if self.do_augment:
            image = self.train_augment(image)

        # 3. Convert to Tensor and Normalize [-1, 1]
        image = self.to_tensor_norm(image)

        return (image, label)

    @property
    def classes(self):
        return self._classes

    @property
    def id_to_classes(self):
        return {v: k for k, v in self._class_to_idx.items()}

    def len_per_class(self):
        """Returns a dictionary with the count of samples per class."""
        all_targets = []
        if isinstance(self.data, ConcatDataset):
            for ds in self.data.datasets:
                if hasattr(ds, "targets"):
                    all_targets.extend(ds.targets)
        elif hasattr(self.data, "targets"):
            all_targets.extend(self.data.targets)

        counts = Counter(all_targets)
        class_counts = {self.id_to_classes[idx]: count for idx, count in counts.items()}
        for class_name in self.classes:
            if class_name not in class_counts:
                class_counts[class_name] = 0
        return class_counts


# --- Configuration ---
SAVE_DIR = "preview_augmentations"
SAMPLES_PER_CLASS = 3


def unnormalize(tensor):
    """Reverts the (0.5, 0.5, 0.5) normalization to [0, 1] for saving."""
    return tensor * 0.5 + 0.5


if __name__ == "__main__":
    # 1. Instantiate the dataset
    print("Initializing dataset...")
    dataset = GlaucomaHarvardDataset("train", augmentation=True)

    # 2. Print Dataset Statistics
    print(f"\nTotal images: {len(dataset)}")
    counts = dataset.len_per_class()
    print("Counts per class:", counts)
    print(dataset.id_to_classes)

    # # 3. Create output directories
    # os.makedirs(SAVE_DIR, exist_ok=True)
    # id_to_class = dataset.id_to_classes

    # for class_name in counts.keys():
    #     os.makedirs(os.path.join(SAVE_DIR, class_name), exist_ok=True)

    # # 4. Iterate Sequentially (No Shuffling)
    # print(f"\nSaving first {SAMPLES_PER_CLASS} samples per class to '{SAVE_DIR}'...")

    # samples_saved = defaultdict(int)
    # classes_needed = set(counts.keys())

    # # Loop through the dataset in order (0, 1, 2...)
    # # This matches the alphabetical order of files in your folders
    # for idx in range(len(dataset)):
    #     # Optimization: Stop if we have enough of all classes
    #     if not classes_needed:
    #         break

    #     img_tensor, label_idx = dataset[idx]
    #     class_name = id_to_class[label_idx]

    #     # Only save if we haven't reached the limit for this class
    #     if samples_saved[class_name] < SAMPLES_PER_CLASS:

    #         # Helper to find original filename (Works if data is ImageFolder)
    #         original_name = "unknown"
    #         if hasattr(dataset, 'data') and hasattr(dataset.data, 'samples'):
    #             # dataset.data.samples is a list of (path, class_index)
    #             full_path = dataset.data.samples[idx][0]
    #             original_name = os.path.basename(full_path)

    #         # Un-normalize back to [0,1] range
    #         img_visible = unnormalize(img_tensor)

    #         # Save file
    #         filename = f"{class_name}_{samples_saved[class_name]}.png"
    #         save_path = os.path.join(SAVE_DIR, class_name, filename)
    #         save_image(img_visible, save_path)

    #         print(f"[{class_name}] Saved {filename} (Source: {original_name})")

    #         samples_saved[class_name] += 1

    #         # Check if we are done with this class
    #         if samples_saved[class_name] >= SAMPLES_PER_CLASS:
    #             if class_name in classes_needed:
    #                 classes_needed.remove(class_name)

    # print("\nDone! Images saved.")
