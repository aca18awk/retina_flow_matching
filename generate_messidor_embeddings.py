"""
Generate DINOv2 embeddings for the full reference pool for each experiment.

Run once (GPU required). Output is used by compute_reference_knn.py.

Output:
    embeddings_30_May/Messidor/{experiment}/reference/features_dinov2.pt
    embeddings_30_May/Messidor/{experiment}/reference/filenames_dinov2.json
"""

import os

from distance_RetFound_DinoV2 import generate_DINO_embeddings

EXPERIMENTS = ["all", "dilated", "nondilated"]
USE_RETFOUND = False
BASE_DIR = "embeddings_30_May/Messidor"

for exp in EXPERIMENTS:
    ref_dir = os.path.join(BASE_DIR, exp, "reference")
    feat_path = os.path.join(ref_dir, "features_dinov2.pt")

    if os.path.exists(feat_path):
        print(f"Skipping {exp} (already exists)")
        continue

    print(f"\n{'='*60}")
    print(f"Generating reference embeddings: experiment={exp}")
    generate_DINO_embeddings(
        purpose="reference",
        experiment=exp,
        seed="seed_A",  # seed doesn't affect which files are in the full pool
        useRetFoundPreprocessing=USE_RETFOUND,
        savedir=ref_dir,
    )

print("\nDone. Run compute_reference_knn.py next.")
