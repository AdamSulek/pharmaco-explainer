import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from argparse import ArgumentParser
import os

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
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()

    base_dir = f"../../data/{args.dataset}"
    fgp_dir = os.path.join(base_dir, "fgp")
    tanimoto_dir = os.path.join(base_dir, "tanimoto")
    os.makedirs(tanimoto_dir, exist_ok=True)

    pos_path = os.path.join(fgp_dir, f"{args.dataset}_positive.parquet")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"No positive file found: {pos_path}")
    print(f"[INFO] Loading positives from {pos_path}")
    df_pos = pd.read_parquet(pos_path)
    pos_fps = [to_bitvect(fp) for fp in df_pos["X_ecfp_2"]]
    pos_ids = df_pos["ID"].tolist()
    print(f"[INFO] Loaded {len(df_pos)} positive molecules")

    chunk_str = f"{args.part:03d}"
    neg_path = os.path.join(fgp_dir, f"{args.dataset}_negative_train_chunk_{chunk_str}.parquet")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"No negative chunk found: {neg_path}")
    print(f"[INFO] Loading negative chunk: {neg_path}")
    df_neg = pd.read_parquet(neg_path)
    print(f"[INFO] Loaded {len(df_neg)} negatives")

    results = []
    total = len(df_neg)

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        print(f"[INFO] Processing rows {start}–{end} / {total}")
        batch = df_neg.iloc[start:end].copy()
        batch_fps = batch["X_ecfp_2"].tolist()

        nearest_sims = []
        nearest_ids = []

        for fp in batch_fps:
            fp_rdk = to_bitvect(fp)
            sims = DataStructs.BulkTanimotoSimilarity(fp_rdk, pos_fps)
            max_idx = max(range(len(sims)), key=lambda i: sims[i])
            nearest_sims.append(sims[max_idx])
            nearest_ids.append(pos_ids[max_idx])

        batch["tanimoto_to_positive"] = nearest_sims
        batch["nearest_positive_ID"] = nearest_ids
        results.append(batch)

    df_out = pd.concat(results, ignore_index=True)
    out_path = os.path.join(tanimoto_dir, f"{args.dataset}_chunk_{args.part}_tanimoto.parquet")
    df_out.to_parquet(out_path, index=False)

    print(f"[SUCCESS] Saved output: {out_path}")
    print(f"[SUCCESS] Total processed rows: {len(df_out)}")

if __name__ == "__main__":
    main()
