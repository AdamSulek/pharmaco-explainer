import os
from argparse import ArgumentParser
import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "  export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return root

def to_bitvect(fp, n_bits=2048):
    if isinstance(fp, ExplicitBitVect):
        return fp
    if isinstance(fp, (bytes, bytearray)):
        arr = np.unpackbits(np.frombuffer(fp, dtype=np.uint8))[:n_bits]
        bv = ExplicitBitVect(n_bits)
        for i in np.nonzero(arr)[0]:
            bv.SetBit(int(i))
        return bv
    if hasattr(fp, "tolist"):
        fp = fp.tolist()
    bv = ExplicitBitVect(len(fp))
    for i, bit in enumerate(fp):
        if int(bit):
            bv.SetBit(i)
    return bv

def process_batch(batch_df, fp_col, pos_fps, pos_ids):
    nearest_sims = []
    nearest_ids = []
    for fp in batch_df[fp_col].tolist():
        fp_rdk = to_bitvect(fp)
        sims = DataStructs.BulkTanimotoSimilarity(fp_rdk, pos_fps)
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        nearest_sims.append(sims[best_idx])
        nearest_ids.append(pos_ids[best_idx])
    batch_df["tanimoto_to_positive"] = nearest_sims
    batch_df["nearest_positive_ID"] = nearest_ids
    return batch_df

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--fp-col", default="X_ecfp_2")
    args = parser.parse_args()

    root = get_project_root()
    preprocessing_dir = os.path.join(root, "data", args.dataset, "preprocessing")
    in_path = os.path.join(preprocessing_dir, f"{args.dataset}_ecfp.parquet")
    out_path = os.path.join(preprocessing_dir, f"{args.dataset}_tanimoto.parquet")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[INFO] Loading dataset: {in_path}")
    df = pd.read_parquet(in_path)

    if args.fp_col not in df.columns:
        raise RuntimeError(f"Fingerprint column not found: {args.fp_col}")

    # positives
    df_pos = df[df[args.label_col] == 1]
    if df_pos.empty:
        raise RuntimeError("No positive samples found")
    print(f"[INFO] Positives: {len(df_pos)}")
    pos_fps = [to_bitvect(fp) for fp in df_pos[args.fp_col]]
    pos_ids = df_pos["ID"].tolist()

    # ---- split dataset into batches
    batches = []
    total = len(df)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batches.append(df.iloc[start:end].copy())

    print(f"[INFO] Total batches: {len(batches)}, batch size: {args.batch_size}")

    results = []

    # ---- parallel processing
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_batch, batch, args.fp_col, pos_fps, pos_ids): i for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                res = future.result()
                results.append((batch_idx, res))
                print(f"[INFO] Completed batch {batch_idx}")
            except Exception as e:
                print(f"[ERROR] Batch {batch_idx} failed: {e}")

    # sort by batch order and concat
    results.sort(key=lambda x: x[0])
    df_out = pd.concat([r[1] for r in results], ignore_index=True)

    df_out.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Saved: {out_path}")
    print(f"[SUCCESS] Total processed rows: {len(df_out)}")

if __name__ == "__main__":
    main()
