"""
Adds hospital_b_100 and hospital_b_200 subdirectories to the existing
Messidor2_256 folder. Uses symlinks so hidden_classifier_data is untouched.

Strategy:
- Always includes all 50 existing hospital_b images first.
- Draws extras from hidden_classifier_data (overlap is fine — hidden stays intact).
- If rare classes (3, 4) can't reach the per-class target, the deficit is
  distributed across classes 0-2 so the total hits exactly 100 / 200.

Usage:
    python create_hospital_b_splits.py

Outputs (inside /vol/biomedic3/awk24/datasets/Messidor2_256/):
    hospital_b_100/   (20/class target, exactly 100 total)
    hospital_b_200/   (40/class target, exactly 200 total)

Then use in MessidorDataset:
    purpose="hospital_b_100"  or  purpose="hospital_b_200"
"""

import os
import random
from collections import defaultdict

import pandas as pd

# --- Config ---
SEED = 42
ROOT = "/vol/biomedic3/awk24/datasets/Messidor2_256"
CSV_PATH = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"
TARGETS = {100: 20, 200: 40}  # {total_target: per_class_target}

random.seed(SEED)


def build_label_map(csv_path):
    df = pd.read_csv(csv_path)
    return {str(row["id_code"]).strip(): int(row["diagnosis"]) for _, row in df.iterrows()}


def inventory_images(split_dir, label_map, split_name):
    """Returns [(abs_path, class_id), ...] for all labelled images in a directory."""
    images = []
    for fname in sorted(os.listdir(split_dir)):
        if not any(fname.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(fname)[0]
        diag = label_map.get(fname, label_map.get(stem, -1))
        if diag == -1:
            print(f"  WARNING: no label for {fname} in {split_name}, skipping.")
            continue
        images.append((os.path.join(split_dir, fname), diag))
    return images


def select_images(hb_images, extra_images, per_class, total_target):
    """
    Pass 1: fill each class up to per_class (hb images always first).
    Pass 2: distribute any deficit from capped rare classes across
            majority classes that still have spare capacity.
    """
    extra_by_class = defaultdict(list)
    for path, cls in extra_images:
        extra_by_class[cls].append(path)
    for cls in extra_by_class:
        random.shuffle(extra_by_class[cls])

    hb_by_class = defaultdict(list)
    for path, cls in hb_images:
        hb_by_class[cls].append(path)

    all_classes = sorted(set(hb_by_class) | set(extra_by_class))

    # Pass 1
    selected = {}
    for cls in all_classes:
        need = max(0, per_class - len(hb_by_class[cls]))
        selected[cls] = list(hb_by_class[cls]) + extra_by_class[cls][:need]

    # Pass 2: top up from classes with spare capacity
    deficit = total_target - sum(len(v) for v in selected.values())
    if deficit > 0:
        spare = [
            cls for cls in all_classes
            if len(selected[cls]) >= per_class
            and len(extra_by_class[cls]) > len(selected[cls]) - len(hb_by_class[cls])
        ]
        while deficit > 0 and spare:
            top_up = max(1, deficit // len(spare))
            for cls in spare:
                if deficit <= 0:
                    break
                already = len(selected[cls]) - len(hb_by_class[cls])
                take = min(top_up, len(extra_by_class[cls]) - already, deficit)
                selected[cls] += extra_by_class[cls][already: already + take]
                deficit -= take
            spare = [
                cls for cls in spare
                if len(selected[cls]) - len(hb_by_class[cls]) < len(extra_by_class[cls])
            ]

    for cls in all_classes:
        n = len(selected[cls])
        flag = " (capped by availability)" if n < per_class else ""
        print(f"  Class {cls}: {n}{flag}")

    return selected


def create_subdir(selected, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for cls in sorted(selected):
        for src in selected[cls]:
            dst = os.path.join(out_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(src, dst)
            total += 1
    return total


def main():
    print("Loading CSV labels...")
    label_map = build_label_map(CSV_PATH)

    print("Reading hospital_b (base 50 images)...")
    hb_images = inventory_images(os.path.join(ROOT, "hospital_b"), label_map, "hospital_b")

    print("Reading hidden_classifier_data (extra pool)...")
    extra_images = inventory_images(os.path.join(ROOT, "hidden_classifier_data"), label_map, "hidden")

    print("\nPool per class:")
    hb_cnt = defaultdict(int)
    ex_cnt = defaultdict(int)
    for _, cls in hb_images:
        hb_cnt[cls] += 1
    for _, cls in extra_images:
        ex_cnt[cls] += 1
    for cls in sorted(set(hb_cnt) | set(ex_cnt)):
        print(f"  Class {cls}: {hb_cnt[cls]} hb + {ex_cnt[cls]} hidden = {hb_cnt[cls]+ex_cnt[cls]} available")

    for total_target, per_class in sorted(TARGETS.items()):
        subfolder = f"hospital_b_{total_target}"
        out_dir = os.path.join(ROOT, subfolder)
        print(f"\n{'='*50}")
        print(f"Creating {subfolder}/ (target {per_class}/class, {total_target} total)...")

        selected = select_images(hb_images, extra_images, per_class, total_target)
        n = create_subdir(selected, out_dir)
        print(f"  -> {out_dir}  ({n} symlinks)")

    print("\nDone. No existing directories were modified.")
    print("Update MessidorDataset with purpose='hospital_b_100' or purpose='hospital_b_200'.")


if __name__ == "__main__":
    main()
