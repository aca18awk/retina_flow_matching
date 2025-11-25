import os

from typing import Literal
from enum import Enum

from torch import unsqueeze, from_numpy, empty, Tensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from skimage.io import imread
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import numpy as np
import numbers
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from collections import Counter

DatasetSplit = Literal["test", "train", "validation"]

class DatasetClass(Enum):
    normal_control = 0
    early_glaucoma = 1
    advanced_glaucoma = 2

path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/'

AI_gen = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/2025/Nov_3_conditional_model/with_classifer_10'

# torch randaugment - do standard augmentations
#  colour, spatial rotations, not crazy rotations
# track macro AUC in torch metrics 
# early stopping based on AUC - save the best model based on that
# H-VAE, different diffusion models
# (common diffusion model, Flow matching model)

class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0: # type: ignore
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)] # type: ignore
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

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
        gamma_factor = None if self.gamma is None else float(empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img



transform_pipe = transforms.Compose([
    transforms.ToPILImage(), # Convert np array to PILImage
    
    # Resize image to 224 x 224 as required by most vision models
    transforms.Resize(
        size=(224, 224)
    ),
    
    # Convert PIL image to tensor with image values in [0, 1]
    transforms.ToTensor(),
    
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # ),
])

class GlaucomaHarvardDataset(Dataset):
    def __init__(self, purpose: DatasetSplit = "test", transform=None, augmentation=True, gen_data_path=None):
        data_path = os.path.join(path, purpose)
        real_data = ImageFolder(data_path, transform=transform)
        
        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx
        self.do_augment = augmentation
        
        data = [real_data]
        
        if gen_data_path is None:
            self.data = real_data
        else:
            generated_data = ImageFolder(gen_data_path, transform=transform) 
            data.append(generated_data)
            self.data = ConcatDataset(data)

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            T.RandomApply(transforms=[GammaCorrectionTransform(gamma=0.3)], p=0.5),
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        ])

        # geometric data augmentation
        self.geometric_augment = T.Compose([
            # T.RandomApply(transforms=[T.RandomPerspective(distortion_scale=0.2)], p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=(-30,30), scale=(1.15, 1.5))], p=0.7),
            # T.RandomApply(transforms=[T.RandomResizedCrop(scale=(0.8, 1.0), size=image_size)], p=0.5),
        ])

        # --- Processing Transforms ---
        self.processing_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if not self.do_augment:
            return self.data[index]
        else:
            image, label = self.data[index]
            
            # Apply Augmentations
            # image = self.photometric_augment(image) # Uncomment if needed
            image = self.geometric_augment(image)
            
            if not isinstance(image, Tensor):
                 image = T.ToTensor()(image)
                 
            image = self.processing_normalize(image)
            return (image, label)
    
    @property
    def classes(self):
        return self._classes
    
    @property
    def id_to_classes(self):
        return {v: k for k, v in self._class_to_idx.items()}

    def len_per_class(self):
        """
        Returns a dictionary with the count of samples per class.
        Handles both single ImageFolder and ConcatDataset.
        """
        all_targets = []

        # Case 1: self.data is a ConcatDataset (Real + Generated)
        if isinstance(self.data, ConcatDataset):
            for ds in self.data.datasets:
                if hasattr(ds, 'targets'):
                    all_targets.extend(ds.targets)
        
        # Case 2: self.data is a single ImageFolder (Real only)
        elif hasattr(self.data, 'targets'):
            all_targets.extend(self.data.targets)

        # Count occurrences of each class index
        counts = Counter(all_targets)

        # Map class indices to class names
        class_counts = {
            self.id_to_classes[idx]: count 
            for idx, count in counts.items()
        }
        
        # Ensure all classes are present (even if count is 0)
        for class_name in self.classes:
            if class_name not in class_counts:
                class_counts[class_name] = 0

        return class_counts
    
transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])


dataset1 = GlaucomaHarvardDataset("train", transform=transform)
print(len(dataset1))
# dataset1.classes
print(dataset1.id_to_classes)
counts = dataset1.len_per_class()
print("Counts per class:", counts)
