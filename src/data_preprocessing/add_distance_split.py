import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "  export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return root

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--frac", type=float, default=0.05)
    args = parser.parse_args()

    root = get_project_root()
    preprocessing_dir = os.path.join(root, "data", args.dataset, "preprocessing")
    in_path = os.path.join(preprocessing_dir, f"{args.dataset}_tanimoto.parquet")
    out_path = os.path.join(root, "data", args.dataset, f"{args.dataset}_split.parquet")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[INFO] Loading: {in_path}")
    df = pd.read_parquet(in_path)

    required_cols = {"split", "tanimoto_to_positive"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["split_distant_set"] = np.nan
    df["split_close_set"] = np.nan

    pos_mask = df["y"] == 1
    df.loc[pos_mask, "split_distant_set"] = df.loc[pos_mask, "split"]
    df.loc[pos_mask, "split_close_set"] = df.loc[pos_mask, "split"]

    df_neg = df[~pos_mask].copy()
    test_mask = df_neg["split"] == "test"
    trainval_mask = ~test_mask

    df_neg_trainval = df_neg[trainval_mask].sort_values("tanimoto_to_positive")
    n_frac = max(int(len(df_neg_trainval) * args.frac), 1)
    easy_idx = df_neg_trainval.head(n_frac).index
    hard_idx = df_neg_trainval.tail(n_frac).index

    df.loc[easy_idx, "split_distant_set"] = df.loc[easy_idx, "split"]
    df.loc[hard_idx, "split_close_set"] = df.loc[hard_idx, "split"]

    df.loc[df_neg[test_mask].index, "split_distant_set"] = df.loc[df_neg[test_mask].index, "split"]
    df.loc[df_neg[test_mask].index, "split_close_set"]   = df.loc[df_neg[test_mask].index, "split"]

    df.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Saved: {out_path}")
    print(f"[SUCCESS] Total rows: {len(df)}")
    print(f"[INFO] Easy: {len(easy_idx)}, Hard: {len(hard_idx)}")

    print("\nValue counts for original split:")
    print(df["split"].value_counts())
    print("\nValue counts for split_distant_set:")
    print(df["split_distant_set"].value_counts())
    print("\nValue counts for split_close_set:")
    print(df["split_close_set"].value_counts())

if __name__ == "__main__":
    main()
