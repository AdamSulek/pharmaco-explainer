#!/usr/bin/env python3
# Canonical SMILES → ~50 3D conformers per ID → single SDF file

"""
Reproducible conformer generation pipeline.

Input:
  - Single .parquet or .csv file with columns: ID, smiles

Output:
  - One SDF file containing multiple 3D conformers per molecule
    (one SDF record per conformer)

Reproducibility:
  - Deterministic conformer generation (seed derived from ID)
  - Identical input → identical SDF output

Example:
  python prepare_sdfs.py \
    --in-parquet molecules.parquet \
    --out-dir sdf_out \
    --part 000
"""

import os
import sys
import argparse
import hashlib
import tempfile
import shutil
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import get_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def md5_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def generate_conformers(
    mol: Chem.Mol,
    target_n: int = 50,
    prune_rms: float = 0.6,
    seed: int = 0,
):
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.maxAttempts = 1000

    n_try = max(target_n * 2, target_n + 10)
    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_try, params=params)

    if not conf_ids:
        return None

    if mol_h.GetNumConformers() > target_n:
        for cid in range(mol_h.GetNumConformers() - 1, target_n - 1, -1):
            mol_h.RemoveConformer(cid)

    return Chem.RemoveHs(mol_h)


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format (use .parquet or .csv)")

    if not {"ID", "smiles"}.issubset(df.columns):
        raise ValueError("Input must contain columns: ID and smiles")

    return df[["ID", "smiles"]].dropna().astype(str)


def _worker_batch(args):
    pairs, target_confs, prune_rms, tmp_dir = args

    tmp = tempfile.NamedTemporaryFile(
        prefix="sdf_chunk_", suffix=".sdf", dir=tmp_dir, delete=False
    )
    tmp_path = tmp.name
    tmp.close()

    writer = Chem.SDWriter(tmp_path)
    n_ids = 0
    n_records = 0

    for mol_id, smi in pairs:
        try:
            smi_can = canonical_smiles(smi)
            mol = Chem.MolFromSmiles(smi_can)
            if mol is None:
                continue

            seed = md5_seed(mol_id)
            mol_conf = generate_conformers(
                mol, target_n=target_confs, prune_rms=prune_rms, seed=seed
            )
            if mol_conf is None or mol_conf.GetNumConformers() == 0:
                continue

            mol_conf.SetProp("ID", mol_id)
            mol_conf.SetProp("smiles", smi_can)

            for i, conf in enumerate(mol_conf.GetConformers()):
                mol_conf.SetProp("_Name", f"{mol_id}#conf={i}")
                writer.write(mol_conf, confId=conf.GetId())
                n_records += 1

            n_ids += 1

        except Exception:
            continue

    writer.close()
    return tmp_path, n_ids, n_records


def merge_sdfs(temp_paths, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fout:
        for p in temp_paths:
            with open(p, "rb") as fin:
                shutil.copyfileobj(fin, fout)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--target-confs", type=int, default=50)
    ap.add_argument("--prune-rms", type=float, default=0.6)

    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--progress-step", type=int, default=10)

    return ap.parse_args()


def main():
    args = parse_args()

    logging.info(f"START part_{args.part}")
    logging.info(f"Input file: {args.in_parquet}")
    logging.info(f"Output dir: {args.out_dir}")
    logging.info(
        f"CPUs={args.n_proc}, batch_size={args.batch_size}, "
        f"target_confs={args.target_confs}"
    )

    try:
        df = load_table(args.in_parquet)
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        sys.exit(2)

    df = df.sort_values("ID").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"part_{args.part}_0001.sdf")
    logging.info(f"Total IDs: {len(df)}")
    logging.info(f"Output SDF: {out_path}")

    tmp_root = tempfile.mkdtemp(prefix=f"tmp_sdf_part_{args.part}_")
    logging.info(f"Temporary directory: {tmp_root}")

    try:
        pairs_all = list(df.itertuples(index=False, name=None))
        if not pairs_all:
            open(out_path, "wb").close()
            logging.warning("No input records found")
            return

        batches = [
            pairs_all[i:i + args.batch_size]
            for i in range(0, len(pairs_all), args.batch_size)
        ]

        worker_args = [
            (batch, args.target_confs, args.prune_rms, tmp_root)
            for batch in batches
        ]

        temp_paths = []
        total_ids = 0
        total_records = 0

        ctx = get_context("spawn")
        total_batches = len(worker_args)
        step = max(1, int((args.progress_step / 100) * total_batches))

        with ctx.Pool(processes=args.n_proc, maxtasksperchild=100) as pool:
            for i, (tmp_path, n_ids, n_rec) in enumerate(
                pool.imap_unordered(_worker_batch, worker_args), start=1
            ):
                temp_paths.append(tmp_path)
                total_ids += n_ids
                total_records += n_rec

                if i % step == 0 or i == total_batches:
                    logging.info(
                        f"Progress {i}/{total_batches} "
                        f"IDs={total_ids} records={total_records}"
                    )

        merge_sdfs(temp_paths, out_path)
        logging.info(
            f"DONE part_{args.part} | IDs={total_ids} records={total_records}"
        )

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logging.info(f"Temporary files removed: {tmp_root}")


if __name__ == "__main__":
    main()
