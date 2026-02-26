import glob
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image

# --- Configuration ---
EXPERIMENT_DIR = (
    "models/15_Feb_Coloured_MNIST_FSFM_Latent/simulation_blue_same_label_3_barycentric/samples"
)

TARGET_COLOR = "blue"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_image_color(image_path):
    """
    Simple heuristic: Load image, sum up pixel intensities per channel.
    The channel with the highest sum is the dominant color.
    Returns: 'red', 'green', or 'blue'
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)

        # img_arr shape is (H, W, 3) -> R=0, G=1, B=2
        r_score = np.sum(img_arr[:, :, 0])
        g_score = np.sum(img_arr[:, :, 1])
        b_score = np.sum(img_arr[:, :, 2])

        scores = {"red": r_score, "green": g_score, "blue": b_score}

        dominant_color = max(scores, key=scores.get)
        return dominant_color
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None


def main():
    print("--- Starting Color Analysis ---")
    print(f"Target Color: {TARGET_COLOR.upper()}")
    print(f"Scanning Directory: {EXPERIMENT_DIR}\n")

    if not os.path.exists(EXPERIMENT_DIR):
        print("Error: Directory not found.")
        return

    # Find all subfolders (assuming they are named '0', '1', '2' etc.)
    subfolders = [f.path for f in os.scandir(EXPERIMENT_DIR) if f.is_dir()]
    # Sort them numerically if possible, else alphabetically
    subfolders.sort(key=lambda f: int(os.path.basename(f)) if os.path.basename(f).isdigit() else f)

    total_images = 0
    total_correct = 0
    global_counts = Counter()

    # Store per-folder results to print a clean summary table later
    folder_stats = []

    for folder in subfolders:
        folder_name = os.path.basename(folder)

        # Get all png/jpg images in this folder
        image_files = glob.glob(os.path.join(folder, "*.png")) + glob.glob(
            os.path.join(folder, "*.jpg")
        )

        # Assuming generated files are numbers like "0.png", "1.png"
        # We skip files that contain "grid", "original", "reference"
        image_files = [
            f
            for f in image_files
            if "grid" not in f and "original" not in f and "reference" not in f
        ]

        if not image_files:
            continue

        folder_counts = Counter()

        for img_path in image_files:
            predicted_color = classify_image_color(img_path)
            if predicted_color:
                folder_counts[predicted_color] += 1
                global_counts[predicted_color] += 1

        count = len(image_files)
        correct = folder_counts[TARGET_COLOR]
        accuracy = (correct / count) * 100 if count > 0 else 0

        total_images += count
        total_correct += correct

        # Store stats for this folder
        folder_stats.append(
            {
                "name": folder_name,
                "count": count,
                "correct": correct,
                "acc": accuracy,
                "breakdown": dict(folder_counts),
            }
        )

        print(f"Processed Folder {folder_name}: {accuracy:.1f}% Accuracy ({correct}/{count})")

    # --- Final Summary Report ---
    print("\n" + "=" * 60)
    print(f"{'Folder':<10} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<10} | {'Breakdown'}")
    print("-" * 60)

    for stat in folder_stats:
        breakdown_str = ", ".join([f"{k}:{v}" for k, v in stat["breakdown"].items()])
        print(
            f"{stat['name']:<10} | {stat['count']:<8} | {stat['correct']:<8} | {stat['acc']:>6.1f}%   | {breakdown_str}"
        )

    print("-" * 60)

    if total_images > 0:
        total_acc = (total_correct / total_images) * 100
        print("OVERALL PERFORMANCE")
        print(f"Total Images Scanned: {total_images}")
        print(f"Total Correct ({TARGET_COLOR}): {total_correct}")
        print(f"Global Accuracy: {total_acc:.2f}%")
        print(f"Global Breakdown: {dict(global_counts)}")
    else:
        print("No images found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
