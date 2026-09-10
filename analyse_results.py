"""
Analyse generation results in results_30_May/.

For each experiment/seed/N/GS run, checks:
  1. Number of anchors processed
  2. Images generated per anchor (expected: 1000 // N)
  3. Total images generated (expected: 1000)
  4. Whether anchors match the expected N-subset from the JSON splits
  5. Whether neighbours are from the same class as the anchor

Usage:
    python analyse_results.py
"""

import json
import os

import pandas as pd

RESULTS_DIR  = "results_30_May"
EMBEDDINGS_DIR = "embeddings_30_May/Messidor"
JSON_DIR     = "/vol/biomedic3/awk24/datasets/Messidor2"
CSV_PATH     = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"

EXPERIMENT_TO_JSON = {
    "all":        "messidor2_splits_all.json",
    "dilated":    "messidor2_splits_dilated.json",
    "nondilated": "messidor2_splits_nondilated.json",
}
_CLASSES = ["G0", "G1", "G2", "G3", "G4"]

# ── Load global label map ──────────────────────────────────────────────────────
df_labels = pd.read_csv(CSV_PATH)
label_map = dict(zip(df_labels["id_code"].str.strip(), df_labels["diagnosis"]))

def get_label(filename):
    return label_map.get(os.path.splitext(filename)[0], label_map.get(filename, -1))


def select_reference_filenames(orderings, seed, N):
    seed_data = orderings[seed]
    per_class = N // len(_CLASSES)
    selected, surplus = [], {g: [] for g in _CLASSES}
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
    return set(selected)


# ── Walk results ───────────────────────────────────────────────────────────────
rows = []

for exp in sorted(os.listdir(RESULTS_DIR)):
    exp_dir = os.path.join(RESULTS_DIR, exp)
    if not os.path.isdir(exp_dir):
        continue

    # Load JSON splits for this experiment once
    json_path = os.path.join(JSON_DIR, EXPERIMENT_TO_JSON.get(exp, ""))
    splits = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            splits = json.load(f)

    for seed in sorted(os.listdir(exp_dir)):
        seed_dir = os.path.join(exp_dir, seed)
        if not os.path.isdir(seed_dir):
            continue

        for n_folder in sorted(os.listdir(seed_dir)):
            if not n_folder.startswith("N"):
                continue
            try:
                N = int(n_folder[1:])
            except ValueError:
                continue

            n_dir = os.path.join(seed_dir, n_folder)

            for gs_folder in sorted(os.listdir(n_dir)):
                run_dir = os.path.join(n_dir, gs_folder)
                if not os.path.isdir(run_dir):
                    continue

                csv_path = os.path.join(run_dir, "neighbours.csv")
                samples_dir = os.path.join(run_dir, "samples")
                expected_per_anchor = max(1, 1000 // N)
                expected_total = N * expected_per_anchor

                row = {
                    "experiment": exp, "seed": seed, "N": N, "run": gs_folder,
                    "anchors_found": 0,
                    "total_images": 0,
                    "imgs_per_anchor_ok": None,
                    "imgs_per_anchor_bad": 0,
                    "anchor_subset_ok": None,
                    "anchor_wrong_subset": 0,
                    "neighbour_same_class": None,
                    "neighbour_wrong_class": 0,
                    "issues": [],
                }

                # ── 1. Count generated images ─────────────────────────────────
                if os.path.isdir(samples_dir):
                    anchor_folders = [f for f in os.listdir(samples_dir)
                                      if os.path.isdir(os.path.join(samples_dir, f))]
                    row["anchors_found"] = len(anchor_folders)
                    imgs_per_anchor = []
                    for af in anchor_folders:
                        n_imgs = len([f for f in os.listdir(os.path.join(samples_dir, af))
                                      if f.endswith(".png")])
                        imgs_per_anchor.append(n_imgs)
                    row["total_images"] = sum(imgs_per_anchor)
                    bad = sum(1 for n in imgs_per_anchor if n != expected_per_anchor)
                    row["imgs_per_anchor_ok"] = bad == 0
                    row["imgs_per_anchor_bad"] = bad
                    if bad:
                        row["issues"].append(f"{bad} anchors have wrong image count (expected {expected_per_anchor})")

                # ── 2 & 3. Check neighbours CSV ───────────────────────────────
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    neighbour_cols = [c for c in df.columns if c.startswith("neighbour_")]

                    wrong_class = 0
                    for _, r in df.iterrows():
                        anchor_lbl = int(r["anchor_label"])
                        for col in neighbour_cols:
                            nb = r[col]
                            if pd.isna(nb) or nb == "":
                                continue
                            nb_lbl = get_label(str(nb))
                            if nb_lbl != anchor_lbl:
                                wrong_class += 1

                    row["neighbour_same_class"] = wrong_class == 0
                    row["neighbour_wrong_class"] = wrong_class
                    if wrong_class:
                        row["issues"].append(f"{wrong_class} neighbour(s) have wrong class")

                    # ── Check anchors are from the correct N subset ───────────
                    if splits is not None:
                        expected_anchors = select_reference_filenames(
                            splits["hospital_b_orderings"], seed, N
                        )
                        wrong_subset = sum(
                            1 for fn in df["anchor"] if str(fn) not in expected_anchors
                        )
                        row["anchor_subset_ok"] = wrong_subset == 0
                        row["anchor_wrong_subset"] = wrong_subset
                        if wrong_subset:
                            row["issues"].append(f"{wrong_subset} anchor(s) not in expected N={N} subset")

                row["issues"] = "; ".join(row["issues"]) if row["issues"] else "OK"
                rows.append(row)

# ── Print summary ──────────────────────────────────────────────────────────────
summary = pd.DataFrame(rows)

print("\n" + "="*80)
print("GENERATION RESULTS ANALYSIS")
print("="*80)

for _, r in summary.iterrows():
    if r['total_images'] == 1000:
        continue
    status = "✓" if r["issues"] == "OK" else "✗"
    print(f"\n{status} {r['experiment']}/{r['seed']}/N{r['N']}/{r['run']}")
    print(f"   Anchors: {r['anchors_found']} (expected {r['N']})")
    print(f"   Total images: {r['total_images']} (expected {max(1, 1000//r['N']) * r['N']})")
    print(f"   Images/anchor uniform: {r['imgs_per_anchor_ok']}  ({r['imgs_per_anchor_bad']} bad)")
    print(f"   Anchors from correct subset: {r['anchor_subset_ok']}  ({r['anchor_wrong_subset']} wrong)")
    print(f"   Neighbours same class: {r['neighbour_same_class']}  ({r['neighbour_wrong_class']} wrong)")
    if r["issues"] != "OK":
        print(f"   ⚠ Issues: {r['issues']}")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "analysis_summary.csv")
summary.to_csv(out_path, index=False)
print(f"\n\nFull summary saved to {out_path}")
