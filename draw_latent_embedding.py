import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import umap
from matplotlib.patches import Polygon
from sklearn.neighbors import NearestNeighbors


class LatentManifoldAnalyzer:
    def __init__(self, dataset, num_samples=100, device=None):
        """
        Initializes the analyzer, extracts latents (z) for a subset of the data, 
        and fits the base UMAP model.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.num_samples = num_samples
        self.dataset = dataset
        self.colors = sns.color_palette("tab10", 10)

        print(f"--- Initializing Latent GPS on {self.device} ---")

        # 1. Prepare Frozen ResNet18 Encoder
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. Extract Data & Latents
        self._extract_latents()

        # 3. Fit Base UMAP
        print("Fitting base UMAP on Z... (This sets our topographical map)")
        # Note: n_neighbors in UMAP can't exceed num_samples. 15 is fine for 100 samples.
        self.reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        self.Z_2d = self.reducer.fit_transform(self.Z)
        print("Initialization complete!\n")

    def _extract_latents(self):
        """Helper method to extract exactly 'num_samples' from the dataset."""
        print(f"Extracting {self.num_samples} samples and computing latents (z)...")
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256, shuffle=True)

        all_z, all_labels, all_imgs = [], [], []
        count = 0

        with torch.no_grad():
            for images, labels in dataloader:
                batch_size = images.size(0)
                if count + batch_size > self.num_samples:
                    images = images[:self.num_samples - count]
                    labels = labels[:self.num_samples - count]

                # Pre-process images for the interactive viewer
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                imgs_unnorm = (images * std) + mean
                all_imgs.append(imgs_unnorm.cpu())

                # Pass through encoder
                z = self.model(images.to(self.device))
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.numpy())

                count += len(images)
                if count >= self.num_samples:
                    break

        self.Z = np.vstack(all_z)
        self.labels = np.concatenate(all_labels)

        self.images = torch.cat(all_imgs, dim=0).permute(0, 2, 3, 1).numpy()
        self.images = np.clip(self.images, 0, 1)

    def _draw_knn_neighborhoods(self, ax, indices, points2d, alpha_fill=0.15):
        """
        Draws a polygon connecting each anchor point to its k nearest neighbors.
        """
        for i, neighbor_idx in enumerate(indices):
            # Get the 2D coordinates for the anchor and its neighbors
            pts = points2d[neighbor_idx]

            # Use the color of the anchor point's class
            anchor_label = self.labels[i]
            color = self.colors[anchor_label]

            # Draw the triangle (or n-sided polygon if k > 2)
            poly = Polygon(pts, closed=True, facecolor=color, alpha=alpha_fill, edgecolor=color, linewidth=0.8)
            ax.add_patch(poly)

    def visualize_and_save(self, save_path="umap_raw_z.png"):
        """Method 1: Plots the standard UMAP and saves it."""
        print("Generating Standard UMAP...")
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.scatterplot(
            x=self.Z_2d[:, 0], y=self.Z_2d[:, 1], hue=self.labels,
            palette="tab10", s=40, alpha=0.8, edgecolor="none", ax=ax
        )

        ax.set_title("UMAP of Raw Latents ($z$)", fontsize=14)
        ax.legend(title="Digit Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved to {save_path}")

    def interactive_plot(self):
        """Method 2: Launches the Interactive UMAP (Click to view image)."""
        print("Launching Interactive UMAP...")
        fig, (ax_umap, ax_img) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

        ax_umap.scatter(
            self.Z_2d[:, 0], self.Z_2d[:, 1], c=self.labels,
            cmap='tab10', s=40, alpha=0.8, picker=5
        )
        ax_umap.set_title("Interactive Manifold (Click a point!)", fontsize=14)

        ax_img.axis('off')
        ax_img.set_title("Awaiting Click...", fontsize=14)
        img_display = ax_img.imshow(np.zeros((32, 32, 3)))

        def on_pick(event):
            ind = event.ind[0]
            img_display.set_data(self.images[ind])
            ax_img.set_title(f"Class: {self.labels[ind]} | Index: {ind}", fontsize=14)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.tight_layout()
        plt.show()

    def compute_and_visualize_y(self, k_neighbors=2, save_path="umap_z_vs_y.png"):
        """Method 3: Computes y and plots side-by-side, showing neighborhood triangles."""
        Y = self.calculate_Y(k_neighbors)

        print("Projecting y into the existing UMAP topology...")
        Y_2d = self.reducer.transform(Y)

        print("Generating Side-by-Side Comparison...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)

        # Plot 1: Raw Z with Neighborhood Triangles
        sns.scatterplot(x=self.Z_2d[:, 0], y=self.Z_2d[:, 1], hue=self.labels, palette="tab10", s=30, alpha=0.8, ax=ax1, legend=False, zorder=5)
        self._draw_knn_neighborhoods(ax1, indices, self.Z_2d, alpha_fill=0.15)
        ax1.set_title(f"Raw Latents ($z$) with {k_neighbors}-NN Local Neighborhoods", fontsize=16)

        # Plot 2: Pooled Y (The triangles collapse to dots!)
        sns.scatterplot(x=Y_2d[:, 0], y=Y_2d[:, 1], hue=self.labels, palette="tab10", s=30, alpha=0.9, ax=ax2, zorder=5)

        formula = f"$y = \\frac{{1}}{{{k_neighbors + 1}}}(z_0"
        for i in range(1, k_neighbors + 1):
            formula += f" + z_{i}"
        formula += ")$"
        ax2.set_title(f"Pooled Latents ({formula})", fontsize=16)

        plt.legend(title="Digit Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved comparison to {save_path}")

    def calculate_Y(self, k_neighbors=2):
        print(f"Computing k-NN (k={k_neighbors}) in 512-D space...")
        knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        knn.fit(self.Z)
        _, indices = knn.kneighbors(self.Z)

        print("Calculating pooled latents (y)...")
        Y = np.mean(self.Z[indices], axis=1)
        return Y

    def compute_and_visualize_overlay(self, k_neighbors=2, save_path="umap_overlay_z_and_y.png"):
        """Method 4: Overlays z, its neighborhood triangles, and the resulting y on one plot."""
        Y = self.calculate_Y(k_neighbors)

        print("Projecting y into the existing UMAP topology...")
        Y_2d = self.reducer.transform(Y)

        print("Generating Overlay Plot...")
        fig, ax = plt.subplots(figsize=(14, 12))

        # 1. Draw the raw Z points (small, semi-transparent circles)
        sns.scatterplot(
            x=self.Z_2d[:, 0], y=self.Z_2d[:, 1], hue=self.labels,
            palette="tab10", s=30, alpha=0.4, ax=ax, legend=False, zorder=2
        )

        # 2. Draw the Neighborhood Triangles (faint background fill)
        self._draw_knn_neighborhoods(ax, indices, self.Z_2d, alpha_fill=0.1)

        # 3. Draw the pooled Y points (Large, prominent 'X' marks with black borders)
        sns.scatterplot(
            x=Y_2d[:, 0], y=Y_2d[:, 1], hue=self.labels,
            palette="tab10", s=150, marker='X', edgecolor='black',
            linewidth=1.2, alpha=1.0, ax=ax, zorder=4
        )

        # 4. Draw dotted "pull" lines from the Anchor (Z_0) to its new center of gravity (Y)
        for i in range(len(self.Z_2d)):
            z_0 = self.Z_2d[i] # The anchor point
            y_i = Y_2d[i]      # The pooled point

            # Draw a gray dotted line connecting them
            ax.plot(
                [z_0[0], y_i[0]], [z_0[1], y_i[1]],
                color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=3
            )

        # Formatting
        formula = f"$y = \\frac{{1}}{{{k_neighbors + 1}}}(z_0"
        for i in range(1, k_neighbors + 1):
            formula += f" + z_{i}"
        formula += ")$"
        ax.set_title("Latent GPS Overlay: Anchors ($z_0$, dots) pulled to Pooled Centers ($y$, X's)", fontsize=18)

        plt.legend(title="Digit Class (X = y, dot = z)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved overlay to {save_path}")


# --- Execution ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.MNIST(root="../../../../datasets/MNIST", train=False, download=True, transform=transform)

# Down to 100 samples for clear triangle visibility
analyzer = LatentManifoldAnalyzer(dataset=dataset, num_samples=100)

analyzer.visualize_and_save("generated_figs/my_raw_manifold.png")
# analyzer.interactive_plot()
analyzer.compute_and_visualize_y(k_neighbors=2, save_path="generated_figs/my_denoising_proof.png")

analyzer.compute_and_visualize_overlay(k_neighbors=2, save_path="generated_figs/my_overlay_proof.png")
