import os

from typing import Literal
from enum import Enum

from torch import unsqueeze, from_numpy, empty
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
    def __init__(self, purpose:DatasetSplit = "test", transform = None, augmentation = True, gen_data_path=None):
        data_path = os.path.join(path,purpose)
        real_data = ImageFolder(data_path, transform=transform)
        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx
        self.do_augment = augmentation
        
        data = [real_data]
        if gen_data_path == None:
            self.data = real_data
        else:
            generated_data = ImageFolder(data_path, transform=transform)
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
            T.RandomApply(transforms=[T.RandomPerspective(distortion_scale=0.2)], p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=(-30,30), scale=(0.8, 1.2))], p=0.5),
            # T.RandomApply(transforms=[T.RandomResizedCrop(scale=(0.8, 1.0), size=image_size)], p=0.5),
        ])

        # --- Processing Transforms ---
        # (Applied after augmentations, to convert PIL to normalized Tensor)
        self.processing_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if (not self.do_augment):
            return self.data[index]
        else:
            image, label = self.data[index]
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)
            image = T.ToTensor()(image)
            image = self.processing_normalize(image)
            return (image, label)
    
    @property
    def classes(self):
        return self._classes
    
    @property
    def id_to_classes(self):
        return {v: k for k, v in self._class_to_idx.items()}
    

transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])
dataset1 = GlaucomaHarvardDataset("train", transform=transform)
print(dataset1.__len__())
dataset1.classes
dataset1.id_to_classes


image, label = dataset1.__getitem__(0)
print(image.shape)
# Save the image to a file
save_image(image, "output.png")


# MORE MANUAL DEFINITION
class GlaucomaHarvardDatasetOld(Dataset):
    """
    Dataset class needs to have those 3 methods overwritten
    init - what to do when dataset is created
    len - model needs to know how big is the dataset
    getitem - to get specific item by using an id
    """
    
    def __init__(self, purpose:DatasetSplit = "test", transform=transform_pipe):

        dataPath = os.path.join(path,purpose)

        files = []
        labels = []
        for datasetClass in DatasetClass: 
            folderPath = os.path.join(dataPath, datasetClass.name)

            newfiles = [os.path.join(folderPath, filename) for filename in os.listdir(folderPath) if filename.endswith(".png")]
            files += newfiles
            labels += [datasetClass.value] * len(newfiles)
        
        self.images = files
        self.labels = labels
            
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = imread(img_path)
        
        if self.transform:
            img = self.transform(img)
            img = unsqueeze(img, 0)
        
        sample = {
            "image": img,
            "label": self.labels[idx],
            "id": os.path.basename(self.images[idx]).replace(".png", "")
        }

        return sample
    
    def __len__(self):
        return len(self.images)



# dataset = GlaucomaHarvardDataset("test")
# dataset.__getitem__(1)
# dataset.__len__()