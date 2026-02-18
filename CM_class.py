import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets, transforms


class ColoredMNIST(datasets.MNIST):
    def __init__(self, root, train=True, download=True, transform=None, set_color_idx=None):
        self.set_color_idx = set_color_idx
        super().__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        # 1. Get standard B&W image (PIL) and label
        img, target = self.data[index], int(self.targets[index])

        # 2. Convert to PIL Image for consistent transforming
        img = Image.fromarray(img.numpy(), mode="L")

        # 3. Create RGB container
        # We start with the grayscale image duplicated across 3 channels
        img_rgb = img.convert("RGB")
        img_rgb = np.array(img_rgb)  # [H, W, 3]

        # 4. Assign a deterministic color based on the index
        # This simulates "Patient A always has Scanner B"
        # 0=Red, 1=Green, 2=Blue
        state = np.random.RandomState(index)
        if self.set_color_idx is not None:
            color_idx = self.set_color_idx
        else:
            color_idx = state.randint(0, 3)

        # 5. Apply the Tint
        # If Green (1), we zero out Red (channel 0) and Blue (channel 2)
        if color_idx == 0:  # Red Only
            img_rgb[:, :, 1] = 0
            img_rgb[:, :, 2] = 0
        elif color_idx == 1:  # Green Only
            img_rgb[:, :, 0] = 0
            img_rgb[:, :, 2] = 0
        elif color_idx == 2:  # Blue Only
            img_rgb[:, :, 0] = 0
            img_rgb[:, :, 1] = 0

        # 6. Convert back to PIL for the transforms to work
        img_rgb = Image.fromarray(img_rgb)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)

        return img_rgb, target


# --- Sanity Check Block ---
if __name__ == "__main__":
    # Define the transforms we will use in training
    t = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Initialize
    ds = ColoredMNIST(root="../../../../datasets/MNIST", train=True, download=True, transform=t)

    print("len: ", len(ds))

    # Grab 3 samples
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for i in range(3):
        img, label = ds[i]
        # Un-normalize for display: x * 0.5 + 0.5
        img_disp = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
        axes[i].imshow(img_disp)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    print("Close the plot window to finish.")
    plt.savefig("color1.jpg")
