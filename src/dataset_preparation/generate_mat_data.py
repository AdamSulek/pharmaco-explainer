#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import logging

import pandas as pd
from rdkit import Chem  # kept in case you later need RDKit-based checks
from huggingmolecules import MatModel, MatFeaturizer, RMatModel, RMatFeaturizer

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Encode SMILES from a parquet part into (features, ID) pickles for MAT/RMAT."
    )
    parser.add_argument(
        "--model-type",
        choices=["mat", "rmat"],
        default="mat",
        help="Which model/featurizer to use for encoding.",
    )
    parser.add_argument(
        "--parquet_file_number",
        type=int,
        required=True,
        help="Index of the parquet file (e.g. 67 → 67.parquet).",
    )
    args = parser.parse_args()

    # --- Load model config & featurizer ---
    # Model itself is not used further here, but loading it ensures consistent config.
    if args.model_type == "mat":
        MatModel.from_pretrained("mat_masking_20M")
        featurizer = MatFeaturizer.from_pretrained("mat_masking_20M")
    else:
        RMatModel.from_pretrained("rmat_4M")
        featurizer = RMatFeaturizer.from_pretrained("rmat_4M")

    base_root = "/net/storage/pr3/plgrid/plggsanodrugs/pharmaco_explainer"

    # Parquet input directory depends on model_type (e.g. mat_parquet_parts / rmat_parquet_parts)
    parquet_dir = os.path.join(
        base_root,
        "pickle_dataloaders",
        f"{args.model_type}_parquet_parts"
    )
    parquet_path = os.path.join(parquet_dir, f"{args.parquet_file_number}.parquet")
    logging.info(f"Reading parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    ids = df["ID"].astype(str).tolist()
    smiles_list = df["smiles"].astype(str).tolist()
    logging.info(f"Loaded {len(smiles_list)} SMILES from parquet.")

    # For now: dummy labels (0) – only needed to satisfy featurizer API.
    y_list = [0] * len(smiles_list)

    # Encode SMILES using the chosen featurizer
    encoded = featurizer.encode_smiles_list(smiles_list, y_list)

    # Build list of (features, ID) tuples
    pickle_list = [(element, ids[i]) for i, element in enumerate(encoded)]

    # Output path depends on model type, e.g. .../pickle_dataloaders/rmat/rmat_parts/
    output_dir = os.path.join(
        base_root,
        "pickle_dataloaders",
        args.model_type,
        f"{args.model_type}_parts"
    )
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"{args.parquet_file_number}.p")
    with open(out_path, "wb") as f:
        pickle.dump(pickle_list, f)

    logging.info(f"Saved {len(pickle_list)} tuples (features, ID) -> {out_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
