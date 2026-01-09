import argparse
from pathlib import Path
import requests
import os

def project_path(*parts):
    root = os.environ.get("PHARM_PROJECT_ROOT")
    if root is None:
        raise RuntimeError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "   export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return os.path.join(root, *parts)

def download_file(url, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists, skipping: {dest_path}")
        return
    print(f"Downloading {url} -> {dest_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

def download_dataset(dataset_name):
    base_url = f"https://huggingface.co/datasets/klimczakjakubdev/pharmaco-explainer/resolve/main/{dataset_name}/"
    files = [f"{dataset_name}.parquet", f"{dataset_name}_split.parquet", f"{dataset_name}_labels.parquet"]

    dest_dir = Path(project_path("data", dataset_name))
    for file_name in files:
        url = base_url + file_name
        dest_path = dest_dir / file_name
        download_file(url, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="k3",
        choices=["k3", "k4", "k4_2ar", "k5"],
        help="Which dataset to download"
    )
    args = parser.parse_args()

    download_dataset(args.dataset)
