import os
import pandas as pd

BASE_DIR = "hf/pharmaco-explainer"
DATASETS = ["k3", "k4", "k5", "k4_2ar"]

KEEP_COLS = [
    "ID",
    "smiles",
    "y",
    "split",
    "split_distant_set",
    "split_close_set",
    "X_ecfp_2",
]

for ds in DATASETS:
    path = os.path.join(BASE_DIR, ds, f"{ds}_split.parquet")
    print(f"[INFO] Processing {path}")

    df = pd.read_parquet(path)

    existing_cols = [c for c in KEEP_COLS if c in df.columns]
    missing_cols = set(KEEP_COLS) - set(existing_cols)

    if missing_cols:
        print(f"[WARN] Missing columns in {ds}: {missing_cols}")

    df = df[existing_cols]

    df.to_parquet(path, index=False)
    print(f"[OK] Saved {ds}: shape={df.shape}")

print("[DONE] All datasets processed")
