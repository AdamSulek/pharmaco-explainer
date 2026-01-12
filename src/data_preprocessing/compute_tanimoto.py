import os
from argparse import ArgumentParser

import pandas as pd
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "  export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return root


def to_bitvect(fp):
    if isinstance(fp, ExplicitBitVect):
        return fp
    if hasattr(fp, "tolist"):
        fp = fp.tolist()

    bv = ExplicitBitVect(len(fp))
    for i, bit in enumerate(fp):
        if int(bit):
            bv.SetBit(i)
    return bv


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--label-col",
        default="y",
        help="Column defining positives (default: y == 1)",
    )
    parser.add_argument(
        "--fp-col",
        default="X_ecfp_2",
        help="Fingerprint column name",
    )
    args = parser.parse_args()

    root = get_project_root()
    preprocessing_dir = os.path.join(
        root, "data", args.dataset, "preprocessing"
    )

    in_path = os.path.join(
        preprocessing_dir,
        f"{args.dataset}_ecfp.parquet",
    )
    out_path = os.path.join(
        preprocessing_dir,
        f"{args.dataset}_tanimoto.parquet",
    )

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[INFO] Loading dataset: {in_path}")
    df = pd.read_parquet(in_path)

    if args.fp_col not in df.columns:
        raise RuntimeError(f"Fingerprint column not found: {args.fp_col}")

    # ---- positives -------------------------------------------------
    df_pos = df[df["y"] == 1]

    if df_pos.empty:
        raise RuntimeError("No positive samples found")

    print(f"[INFO] Positives: {len(df_pos)}")

    pos_fps = [to_bitvect(fp) for fp in df_pos[args.fp_col]]
    pos_ids = df_pos["ID"].tolist()

    # ---- compute tanimoto -----------------------------------------
    results = []
    total = len(df)

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        print(f"[INFO] Rows {start}–{end} / {total}")

        batch = df.iloc[start:end].copy()
        batch_fps = batch[args.fp_col].tolist()

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

    df_out.to_parquet(out_path, index=False)
    print(f"[SUCCESS] Saved: {out_path}")
    print(f"[SUCCESS] Rows: {len(df_out)}")


if __name__ == "__main__":
    main()
