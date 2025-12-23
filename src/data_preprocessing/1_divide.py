import argparse
import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def project_path(*parts):
    root = os.environ.get("PHARM_PROJECT_ROOT")
    if root is None:
        raise RuntimeError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "   export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return os.path.join(root, *parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--num-chunks", type=int, default=50)
    args = parser.parse_args()

    base_dir = project_path("data", args.dataset)
    splits_dir = project_path("data", args.dataset, "splits")

    os.makedirs(splits_dir, exist_ok=True)

    input_file = project_path("data", args.dataset, f"{args.dataset}_split.parquet")
    logging.info(f"Reading input file: {input_file}")

    df = pd.read_parquet(input_file)
    logging.info(f"Loaded {len(df)} rows")

    pos = df[df["y"] == 1].copy()
    pos_path = project_path("data", args.dataset, "splits", f"{args.dataset}_positive.parquet")
    pos.to_parquet(pos_path, index=False)
    logging.info(f"Saved positives: {len(pos)} rows")

    neg_test = df[(df["y"] == 0) & (df["split"] == "test")].copy()
    neg_test_path = project_path("data", args.dataset, "splits", f"{args.dataset}_negative_test.parquet")
    neg_test.to_parquet(neg_test_path, index=False)
    logging.info(f"Saved negative test: {len(neg_test)} rows")

    valid_names = ["val", "valid"]
    splits_valid = [s for s in valid_names if s in df["split"].unique()]

    train_val = df[(df["y"] == 0) & (df["split"].isin(["train"] + splits_valid))].copy()

    chunk_size = int(np.ceil(len(train_val) / args.num_chunks))
    logging.info(f"Negative train total: {len(train_val)}, chunk size: {chunk_size}")

    for i in range(args.num_chunks):
        chunk = train_val.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
        out_file = project_path("data", args.dataset, "splits",
                                f"{args.dataset}_negative_train_chunk_{i:03d}.parquet")
        chunk.to_parquet(out_file, index=False)
        logging.info(f"Saved chunk {i:03d}: {len(chunk)} rows")

    logging.info("DONE.")


if __name__ == "__main__":
    main()
