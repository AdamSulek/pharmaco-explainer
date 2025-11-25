import pandas as pd
import os
from glob import glob
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--num-chunks", type=int, default=50)
    parser.add_argument("--frac", type=float, default=0.05)
    args = parser.parse_args()

    base_dir = f"../../data/{args.dataset}"
    fgp_dir = os.path.join(base_dir, "fgp")
    tanimoto_dir = os.path.join(base_dir, "tanimoto")
    processed_dir = os.path.join(base_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    pos_file = os.path.join(fgp_dir, f"{args.dataset}_positive.parquet")
    neg_test_file = os.path.join(fgp_dir, f"{args.dataset}_negative_test.parquet")

    if not os.path.exists(pos_file):
        raise FileNotFoundError(f"Positive file missing: {pos_file}")
    if not os.path.exists(neg_test_file):
        raise FileNotFoundError(f"Negative test file missing: {neg_test_file}")

    print(f"[INFO] Loading positives:      {pos_file}")
    df_pos = pd.read_parquet(pos_file)

    print(f"[INFO] Loading negative test:  {neg_test_file}")
    df_neg_test = pd.read_parquet(neg_test_file)

    df_neg_test["split_easy"] = "test"
    df_neg_test["split_hard"] = "test"

    train_files = sorted(glob(os.path.join(
        tanimoto_dir,
        f"{args.dataset}_chunk_*_tanimoto.parquet"
    )))
    if not train_files:
        raise FileNotFoundError(f"No tanimoto train files in: {tanimoto_dir}")

    print(f"[INFO] Found {len(train_files)} tanimoto train chunks")
    df_train_val = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
    print(f"[INFO] Loaded tanimoto train size: {len(df_train_val)}")

    df_pos["split_easy"] = df_pos["split"]
    df_pos["split_hard"] = df_pos["split"]

    df_train_val["split_easy"] = np.nan
    df_train_val["split_hard"] = np.nan

    df_sorted = df_train_val.sort_values("tanimoto_to_positive")
    n_frac = max(int(len(df_sorted) * args.frac), 1)
    easy_idx = df_sorted.head(n_frac).index
    hard_idx = df_sorted.tail(n_frac).index
    df_train_val.loc[easy_idx, "split_easy"] = df_train_val.loc[easy_idx, "split"]
    df_train_val.loc[hard_idx, "split_hard"] = df_train_val.loc[hard_idx, "split"]

    df_all = pd.concat([df_pos, df_train_val, df_neg_test], ignore_index=True)

    df_all = df_all.sample(frac=1, random_state=123).reset_index(drop=True)

    df_all_no_X = df_all.copy()
    if "X_ecfp_2" in df_all_no_X.columns:
        df_all_no_X = df_all_no_X.drop(columns=["X_ecfp_2"])

    out_file = os.path.join(processed_dir, "final_dataset.parquet")
    df_all_no_X.to_parquet(out_file, index=False)
    print(f"[SUCCESS] Final dataset saved: {out_file}, total rows: {len(df_all_no_X)}")

    chunk_len = int(np.ceil(len(df_all) / args.num_chunks))
    print(f"[INFO] Splitting into {args.num_chunks} chunks, chunk size: {chunk_len}")

    for i in range(args.num_chunks):
        start = i * chunk_len
        end = min((i + 1) * chunk_len, len(df_all))
        df_chunk = df_all.iloc[start:end]
        out_chunk_file = os.path.join(processed_dir, f"final_dataset_part_{i:03d}.parquet")
        df_chunk.to_parquet(out_chunk_file, index=False)
        print(f"[INFO] Chunk {i:03d}: rows {len(df_chunk)} saved to {out_chunk_file}")

    print("[SUCCESS] All processing steps completed.")

if __name__ == "__main__":
    main()
