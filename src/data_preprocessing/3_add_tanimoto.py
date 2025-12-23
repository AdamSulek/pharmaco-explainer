import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from argparse import ArgumentParser
import os
import glob

def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run: export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return root

def to_bitvect(fp_list):
    if isinstance(fp_list, ExplicitBitVect):
        return fp_list
    if hasattr(fp_list, "tolist"):
        fp_list = fp_list.tolist()
    bv = ExplicitBitVect(len(fp_list))
    for i, bit in enumerate(fp_list):
        if int(bit):
            bv.SetBit(i)
    return bv

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()

    root = get_project_root()
    base_dir = os.path.join(root, "data", args.dataset)

    fps_dir = os.path.join(base_dir, "fps")
    tanimoto_dir = os.path.join(base_dir, "tanimoto")
    os.makedirs(tanimoto_dir, exist_ok=True)

    pos_path = os.path.join(fps_dir, f"{args.dataset}_positive.parquet")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"No positive file found: {pos_path}")

    print(f"[INFO] Loading positives: {pos_path}")
    df_pos = pd.read_parquet(pos_path)
    pos_fps = [to_bitvect(fp) for fp in df_pos["X_ecfp_2"]]
    pos_ids = df_pos["ID"].tolist()
    print(f"[INFO] Loaded {len(pos_fps)} positive molecules")

    neg_files = sorted(glob.glob(os.path.join(
        fps_dir, f"{args.dataset}_negative_train_chunk_*.parquet"
    )))

    if not neg_files:
        raise RuntimeError(f"No negative chunks found in {fps_dir}")

    print(f"[INFO] Found {len(neg_files)} negative chunks")

    for neg_path in neg_files:
        chunk_name = os.path.basename(neg_path)
        chunk_id = chunk_name.split("_")[-1].split(".")[0]

        print(f"\n[INFO] Processing chunk {chunk_id}: {neg_path}")
        df_neg = pd.read_parquet(neg_path)
        total = len(df_neg)
        print(f"[INFO] Negatives: {total}")

        results = []

        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            print(f"[INFO] Rows {start}–{end} / {total}")

            batch = df_neg.iloc[start:end].copy()
            batch_fps = batch["X_ecfp_2"].tolist()

            nearest_sims = []
            nearest_ids = []

            for fp in batch_fps:
                fp_rdk = to_bitvect(fp)
                sims = DataStructs.BulkTanimotoSimilarity(fp_rdk, pos_fps)
                best_idx = max(range(len(sims)), key=lambda i: sims[i])
                nearest_sims.append(sims[best_idx])
                nearest_ids.append(pos_ids[best_idx])

            batch["tanimoto_to_positive"] = nearest_sims
            batch["nearest_positive_ID"] = nearest_ids
            results.append(batch)

        df_out = pd.concat(results, ignore_index=True)

        out_path = os.path.join(
            tanimoto_dir,
            f"{args.dataset}_chunk_{chunk_id}_tanimoto.parquet"
        )
        df_out.to_parquet(out_path, index=False)

        print(f"[SUCCESS] Saved: {out_path}")
        print(f"[SUCCESS] Rows: {len(df_out)}")

if __name__ == "__main__":
    main()
