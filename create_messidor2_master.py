"""
Build a master CSV for the Messidor2 dataset combining:
  - messidor_data.csv     : diagnosis, adjudicated_dme
  - messidor_pairs.csv    : left/right eye pairs (pair_id, side)
  - messidor_original.csv : hospital, with_eye_drops  (PNG only)

For JPG images: hospital = "CHU de BREST", with_eye_drops = 0.

Output columns:
  id_code, format, diagnosis, adjudicated_dme,
  pair_id, side, hospital, with_eye_drops
"""

from pathlib import Path
import pandas as pd

ROOT = Path("/vol/biomedic3/awk24/datasets/Messidor2")
OUT  = ROOT / "messidor2_master.csv"

JPG_HOSPITAL   = "Brest"
JPG_EYE_DROPS  = 0


def main() -> None:
    # ── Load source CSVs ──────────────────────────────────────────────────────
    df_data = pd.read_csv(ROOT / "messidor_data.csv")
    df_data["id_code"] = df_data["id_code"].str.strip()

    df_pairs = pd.read_csv(ROOT / "messidor_pairs.csv", sep=";")
    df_pairs.columns = df_pairs.columns.str.strip()
    df_pairs["left"]  = df_pairs["left"].str.strip()
    df_pairs["right"] = df_pairs["right"].str.strip()
    # pair_id is 1-based row number
    df_pairs["pair_id"] = df_pairs.index + 1

    df_orig = pd.read_csv(ROOT / "messidor_original.csv", encoding="utf-8-sig")
    df_orig["Image name"] = df_orig["Image name"].str.strip()
    df_orig = df_orig.rename(columns={
        "Image name":              "id_code",
        "Ophthalmologic department": "hospital",
        "Is using eye drops":       "with_eye_drops",
    })[["id_code", "hospital", "with_eye_drops"]]

    # ── Build pair_id / side lookup ───────────────────────────────────────────
    left_map  = df_pairs.set_index("left")["pair_id"].to_dict()
    right_map = df_pairs.set_index("right")["pair_id"].to_dict()

    def get_pair_id(name: str) -> int | None:
        return left_map.get(name) or right_map.get(name)

    def get_side(name: str) -> str | None:
        if name in left_map:
            return "L"
        if name in right_map:
            return "R"
        return None

    # ── Collect images on disk ────────────────────────────────────────────────
    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    images = [p for p in sorted(ROOT.iterdir()) if p.suffix.lower() in IMAGE_EXTS]

    rows = []
    for p in images:
        name = p.name
        fmt  = "png" if p.suffix.lower() == ".png" else "jpg"
        rows.append({"id_code": name, "format": fmt})
    master = pd.DataFrame(rows)

    # ── Merge diagnosis + dme ─────────────────────────────────────────────────
    master = master.merge(
        df_data[["id_code", "diagnosis", "adjudicated_dme"]],
        on="id_code", how="left",
    )

    # ── Add pair_id / side ────────────────────────────────────────────────────
    master["pair_id"] = master["id_code"].map(get_pair_id)
    master["side"]    = master["id_code"].map(get_side)

    # ── Add hospital / with_eye_drops ─────────────────────────────────────────
    master = master.merge(df_orig, on="id_code", how="left")
    HOSPITAL_RENAME = {
        "LaTIM - CHU de BREST":               "Brest",
        "CHU de St Etienne":                   "Etienne",
        "Service Ophtalmologie Lariboisière":  "Lariboisiere",
    }
    master["hospital"] = master["hospital"].replace(HOSPITAL_RENAME)
    # Fill JPG values
    jpg_mask = master["format"] == "jpg"
    master.loc[jpg_mask, "hospital"]       = JPG_HOSPITAL
    master.loc[jpg_mask, "with_eye_drops"] = JPG_EYE_DROPS

    # ── Reorder columns ───────────────────────────────────────────────────────
    master = master[[
        "id_code", "format", "diagnosis", "with_eye_drops", "hospital",
        "pair_id", "side", "adjudicated_dme",
    ]]

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print(f"Total rows       : {len(master)}")
    print(f"Missing diagnosis: {master['diagnosis'].isna().sum()}")
    print(f"Missing pair_id  : {master['pair_id'].isna().sum()}")
    print(f"Missing side     : {master['side'].isna().sum()}")
    print(f"Missing hospital : {master['hospital'].isna().sum()}")
    print(f"Missing eye_drops: {master['with_eye_drops'].isna().sum()}")
    print()
    print("Format counts:")
    print(master["format"].value_counts().to_string())
    print()
    print("Hospital counts:")
    print(master["hospital"].value_counts().to_string())
    print()
    print("Side counts:")
    print(master["side"].value_counts(dropna=False).to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    master.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")
    print()
    print(master.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
