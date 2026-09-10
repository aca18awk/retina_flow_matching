"""
Create stratified splits for 3 Messidor2 experiments:
  dilated    : with_eye_drops = 1  (385 patients)
  nondilated : with_eye_drops = 0  (489 patients)
  all        : all images           (874 patients)

Split ratios (at patient/pair level, stratified by worst-eye grade):
  reference_pool : 50%
  test           : 40%
  validation     : 10%

Both eyes of every patient land in the same split.
Outputs one JSON per experiment to OUT_DIR.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

MASTER  = Path("/vol/biomedic3/awk24/datasets/Messidor2/messidor2_master.csv")
OUT_DIR = Path("/vol/biomedic3/awk24/datasets/Messidor2")
SEED    = 7

# target fractions (must sum to 1.0)
FRACTIONS = {"validation": 0.10, "test": 0.40, "reference_pool": 0.50}

# Seeds for hospital_b orderings within reference_pool
ORDERING_SEEDS = {"seed_A": 11, "seed_B": 22, "seed_C": 33}

EXPERIMENTS = {
    "dilated":    1,
    "nondilated": 0,
    "all":        None,   # None = no filter
}


def stratified_patient_split(
    patients: pd.DataFrame,   # columns: pair_id, worst_grade, images (list)
    fractions: dict,
    seed: int,
) -> dict:
    """
    Stratify patients by worst_grade, then assign to splits proportionally.
    Returns {split_name: [image_id, ...]}.
    """
    rng = random.Random(seed)
    split_images: dict[str, list[str]] = defaultdict(list)

    for grade, group in patients.groupby("worst_grade"):
        pair_ids = group["pair_id"].tolist()
        rng.shuffle(pair_ids)
        n = len(pair_ids)

        # Compute per-split counts; rounding errors go to reference_pool
        counts: dict[str, int] = {}
        assigned = 0
        for split in ["validation", "test"]:  # assign smallest splits first
            k = max(1, round(n * fractions[split])) if n >= 3 else 1
            k = min(k, n - assigned - (len(fractions) - len(counts) - 1))
            counts[split] = k
            assigned += k
        counts["reference_pool"] = n - assigned

        start = 0
        for split in ["validation", "test", "reference_pool"]:
            k = counts[split]
            for pid in pair_ids[start : start + k]:
                imgs = patients.loc[patients["pair_id"] == pid, "images"].iloc[0]
                split_images[split].extend(imgs)
            start += k

    return dict(split_images)


def main() -> None:
    df = pd.read_csv(MASTER)

    # Patient-level summary: pair_id -> worst_grade + list of image ids
    patient_df = (
        df.groupby("pair_id")
        .apply(
            lambda g: pd.Series({
                "worst_grade":   int(g["diagnosis"].max()),
                "eye_drops":     int(g["with_eye_drops"].iloc[0]),
                "images":        g["id_code"].tolist(),
            }),
        )
        .reset_index()
    )

    for exp_name, eye_drops_val in EXPERIMENTS.items():
        if eye_drops_val is None:
            subset = patient_df
        else:
            subset = patient_df[patient_df["eye_drops"] == eye_drops_val]

        n_patients = len(subset)
        n_images   = subset["images"].apply(len).sum()

        splits = stratified_patient_split(subset, FRACTIONS, SEED)

        # ── Verify no overlap ─────────────────────────────────────────────────
        all_assigned = splits["validation"] + splits["test"] + splits["reference_pool"]
        assert len(all_assigned) == len(set(all_assigned)), "Duplicate image in splits!"
        assert len(all_assigned) == n_images, \
            f"Image count mismatch: {len(all_assigned)} assigned vs {n_images} total"

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"EXPERIMENT: {exp_name}  ({n_patients} patients, {n_images} images)")
        print(f"{'='*55}")
        img_lookup = df.set_index("id_code")["diagnosis"].to_dict()
        header = f"  {'Grade':<8}" + "".join(f"{s:>16}" for s in ["reference_pool", "test", "validation"])
        print(header)
        print(f"  {'-'*56}")
        grades = sorted(df["diagnosis"].dropna().unique().astype(int))
        totals = {s: 0 for s in FRACTIONS}
        for g in grades:
            row = f"  Grade {g:<2}"
            for split_name in ["reference_pool", "test", "validation"]:
                cnt = sum(1 for img in splits[split_name] if img_lookup.get(img) == g)
                totals[split_name] += cnt
                row += f"{cnt:>16}"
            print(row)
        print(f"  {'-'*56}")
        total_row = f"  {'TOTAL':<8}"
        for split_name in ["reference_pool", "test", "validation"]:
            n = len(splits[split_name])
            total_row += f"{n:>16}"
        print(total_row)

        # ── hospital_b orderings (reference_pool images shuffled per grade) ──
        ref_by_grade: dict[str, list[str]] = {}
        for img in splits["reference_pool"]:
            g = f"G{int(img_lookup[img])}"
            ref_by_grade.setdefault(g, []).append(img)

        hospital_b_orderings: dict[str, dict[str, list[str]]] = {}
        for seed_name, seed_val in ORDERING_SEEDS.items():
            rng = random.Random(seed_val)
            hospital_b_orderings[seed_name] = {
                grade: rng.sample(imgs, len(imgs))
                for grade, imgs in sorted(ref_by_grade.items())
            }

        splits["hospital_b_orderings"] = hospital_b_orderings

        # ── Save JSON ─────────────────────────────────────────────────────────
        out_path = OUT_DIR / f"messidor2_splits_{exp_name}.json"
        with open(out_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
