import argparse
import os
import pandas as pd
from skfp.fingerprints import ECFPFingerprint
import numpy as np

def project_path(*parts):
    root = os.environ.get("PHARM_PROJECT_ROOT")
    if root is None:
        raise RuntimeError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "   export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return os.path.join(root, *parts)


def process_file(in_path, out_path, fp_gen, fp_name, chunk_size=50_000):
    df = pd.read_parquet(in_path)

    if "smiles" not in df.columns:
        raise RuntimeError("Column 'smiles' not found in dataset")

    smiles = df["smiles"].tolist()


    fps = []
    for start in range(0, len(smiles), chunk_size):
        sub = smiles[start:start + chunk_size]
        out = fp_gen.transform(sub)

        for fp in out:
            arr = np.asarray(fp, dtype=np.uint8)
            packed = np.packbits(arr)
            fps.append(packed.tobytes())

    if len(fps) != len(df):
        raise RuntimeError(
            f"Fingerprint count mismatch: {len(fps)} != {len(df)}"
        )

    df[fp_name] = fps
    df.to_parquet(out_path, index=False)

    print(f"✔ Saved {out_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--fingerprint", default="X_ecfp_2")
    parser.add_argument("--n-proc", type=int, default=1)
    args = parser.parse_args()

    preprocessing_dir = project_path("data", args.dataset, "preprocessing")
    os.makedirs(preprocessing_dir, exist_ok=True)

    in_path = os.path.join(
        preprocessing_dir,
        f"{args.dataset}_scaffold_split.parquet",
    )

    out_path = os.path.join(
        preprocessing_dir,
        f"{args.dataset}_ecfp.parquet",
    )

    if not os.path.exists(in_path):
        raise RuntimeError(f"Input file not found: {in_path}")

    fp_gen = ECFPFingerprint(
        count=False,
        radius=2,
        n_jobs=args.n_proc,
    )

    print(f"Processing dataset: {in_path}")
    process_file(
        in_path=in_path,
        out_path=out_path,
        fp_gen=fp_gen,
        fp_name=args.fingerprint,
    )


if __name__ == "__main__":
    main()
