import argparse
import glob
import os
import pandas as pd
from skfp.fingerprints import ECFPFingerprint


def project_path(*parts):
    root = os.environ.get("PHARM_PROJECT_ROOT")
    if root is None:
        raise RuntimeError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "   export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return os.path.join(root, *parts)


def process_file(path, fp_gen, fp_name, out_dir, chunk_size=50000):
    df = pd.read_parquet(path)
    smiles = df["smiles"].tolist()

    fps = []
    for start in range(0, len(smiles), chunk_size):
        sub = smiles[start:start + chunk_size]
        fps.extend(fp_gen.transform(sub))

    if len(fps) != len(df):
        raise RuntimeError(
            f"Fingerprint count mismatch for {path}: {len(fps)} != {len(df)}"
        )

    df[fp_name] = fps

    out_path = os.path.join(out_dir, os.path.basename(path))
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--fingerprint", default="X_ecfp_2")
    parser.add_argument("--n-proc", type=int, default=1)
    args = parser.parse_args()

    base_dir = project_path("data", args.dataset)
    splits_dir = project_path("data", args.dataset, "splits")
    fps_dir = project_path("data", args.dataset, "fps")

    os.makedirs(fps_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(splits_dir, "*.parquet")))
    if not files:
        print(f"No split files found in {splits_dir}")
        return

    print(f"Found {len(files)} split files in {splits_dir}")

    fp_gen = ECFPFingerprint(count=False, radius=2, n_jobs=args.n_proc)

    for path in files:
        try:
            print(f"Processing {path}")
            process_file(path, fp_gen, args.fingerprint, fps_dir)
        except Exception as e:
            print(f"Failed {path}: {e}")


if __name__ == "__main__":
    main()
