import torch
import torch.nn as nn
import torchvision.models as models

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
encoder.fc = nn.Identity()  # Remove classification head -> Output [B, 512]
encoder = encoder.to(device)
encoder.eval()  # Freeze batchnorm and dropout


def get_fsfm_condition(x_batch, k=2, return_neigh=False, ensure_same_label=False, labels=None):
    """
    Takes 1-channel MNIST batch, converts to 3-channel,
    extracts latents, and performs batch-wise k-NN pooling.

    If ensure_same_label=True, it masks out neighbors of different classes.
    If a sample has NO valid neighbors of the same class, it falls back to using only itself.
    """

    with torch.no_grad():
        # 1. Convert 1-ch to 3-ch for ImageNet ResNet
        if x_batch.shape[1] == 1:
            x_input = x_batch.repeat(1, 3, 1, 1)
        else:
            x_input = x_batch

        # 2. Extract latents z
        z = encoder(x_input)  # [B, 512]

        # 3. Calculate Distance Matrix
        dist = torch.cdist(z, z)  # [B, B]

        # Only mask if requested AND labels are provided
        if ensure_same_label and labels is not None:
            # Create a [B, B] boolean mask where True = same label
            label_match_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

            # Set distance to infinity where labels are different
            dist = dist.masked_fill(~label_match_mask, float("inf"))

        # 4. Find k+1 nearest neighbors
        actual_k = min(k + 1, z.shape[0])

        # Get values (dists) to check for infinity
        dists, indices = torch.topk(dist, k=actual_k, largest=False)

        # 5. Gather Features
        z_gathered = z[indices]  # [B, k+1, 512]

        # 6. Safe Averaging (Masks out Infs if they exist)
        # valid_mask is 1.0 for valid neighbors, 0.0 for "Inf" neighbors
        valid_mask = (dists != float("inf")).float().unsqueeze(-1)  # [B, k+1, 1]

        # Sum valid vectors
        numerator = (z_gathered * valid_mask).sum(dim=1)  # [B, 512]

        # Count valid neighbors (clamp min=1 to avoid div by zero if only self exists)
        denominator = valid_mask.sum(dim=1).clamp(min=1.0)  # [B, 1]

        y = numerator / denominator

        if return_neigh:
            return y, indices
        else:
            return y
