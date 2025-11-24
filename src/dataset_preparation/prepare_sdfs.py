#!/usr/bin/env python3
# prepare_sdfs.py — canonical SMILES → ~50 conformers per ID (props: ID, smiles; no energies)
# One run per input file → one output SDF. Parallel, clean logging.

import os
import sys
import argparse
import hashlib
import tempfile
import shutil
import datetime as dt
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import get_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def md5_seed(s: str) -> int:
    """Deterministic seed based on ID (stable conformers per ID)."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % (2**31 - 1)

def canon_smiles(smi: str) -> str:
    """Return canonical SMILES or the original string if parsing fails."""
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return smi
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def generate_conformers(mol: Chem.Mol, target_n=50, prune_rms=0.6, seed=0):
    """Generate up to target_n conformers (no energy optimization)."""
    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.maxAttempts = 1000

    n_try = max(target_n * 2, target_n + 10)
    cids = AllChem.EmbedMultipleConfs(molH, numConfs=n_try, params=params)
    if not cids:
        return None

    # prune excess conformers
    if molH.GetNumConformers() > target_n:
        for cid in range(molH.GetNumConformers() - 1, target_n - 1, -1):
            molH.RemoveConformer(cid)

    # remove hydrogens (lighter SDF)
    mol_noH = Chem.RemoveHs(molH)
    return mol_noH

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {ext} (use .parquet or .csv)")

    if not {"ID", "smiles"}.issubset(df.columns):
        raise ValueError("Input file must contain columns: ID, smiles")

    return df[["ID", "smiles"]].dropna().astype({"ID": str, "smiles": str})


def _worker_batch(args):
    """
    Worker processes its batch (list of (ID, smiles)), writes to a temporary SDF,
    and returns: (tmp_path, n_ids, n_records)
    """
    (pairs, target_confs, prune_rms, tmp_dir) = args
    tmp = tempfile.NamedTemporaryFile(prefix="sdf_chunk_", suffix=".sdf", dir=tmp_dir, delete=False)
    tmp_path = tmp.name
    tmp.close()

    writer = Chem.SDWriter(tmp_path)
    n_ids = 0
    n_records = 0

    for (ID, smi) in pairs:
        try:
            smi_can = canon_smiles(smi)
            mol = Chem.MolFromSmiles(smi_can)
            if mol is None:
                continue

            seed = md5_seed(ID)
            mol_conf = generate_conformers(mol, target_n=target_confs, prune_rms=prune_rms, seed=seed)
            if mol_conf is None or mol_conf.GetNumConformers() == 0:
                continue

            mol_conf.SetProp("ID", ID)
            mol_conf.SetProp("smiles", smi_can)

            for i, conf in enumerate(mol_conf.GetConformers()):
                mol_conf.SetProp("_Name", f"{ID}#conf={i}")
                writer.write(mol_conf, confId=conf.GetId())
                n_records += 1

            n_ids += 1

        except Exception:
            continue

    writer.close()
    return (tmp_path, n_ids, n_records)


def _merge_sdfs(temp_paths, out_sdf_path):
    os.makedirs(os.path.dirname(out_sdf_path), exist_ok=True)
    with open(out_sdf_path, "wb") as fout:
        for p in temp_paths:
            with open(p, "rb") as fin:
                shutil.copyfileobj(fin, fout)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build SDF with ~50 conformers per ID (ID + smiles properties; no energies)."
    )
    ap.add_argument("--in-parquet", required=True, help="Input .parquet or .csv (columns: ID, smiles)")
    ap.add_argument("--part", required=True, help="Part index, e.g. 000, 001, 002")
    ap.add_argument("--out-dir", required=True, help="Output directory for SDF (one file per part)")

    ap.add_argument("--target-confs", type=int, default=50)
    ap.add_argument("--prune-rms", type=float, default=0.6)

    ap.add_argument("--n-proc", type=int, default=1, help="Number of parallel processes")
    ap.add_argument("--batch-size", type=int, default=64, help="IDs per worker batch")
    ap.add_argument("--progress-step", type=int, default=10, help="Log progress every N%% (1–100)")

    return ap.parse_args()


def main():
    args = parse_args()

    logging.info(f"=== START part_{args.part} ===")
    logging.info(f"Input:  {args.in_parquet}")
    logging.info(f"Output: {args.out-dir}")
    logging.info(f"CPUs:   {args.n_proc} | batch_size={args.batch_size} | target_confs={args.target_confs}")

    try:
        df = load_table(args.in_parquet)
    except Exception as e:
        logging.error(f"Cannot load input file {args.in_parquet}: {e}")
        sys.exit(2)

    df = df.sort_values("ID").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)

    total_ids = len(df)
    out_path = os.path.join(args.out_dir, f"part_{args.part}_0001.sdf")

    logging.info(f"Total IDs: {total_ids}")
    logging.info(f"Output file: {out_path}")

    tmp_root = tempfile.mkdtemp(prefix=f"tmp_sdf_part_{args.part}_")
    logging.info(f"Using temporary directory: {tmp_root}")

    try:
        pairs_all = [(str(r.ID), str(r.smiles)) for r in df.itertuples(index=False)]
        if len(pairs_all) == 0:
            logging.warning("No input records – output will be empty.")
            open(out_path, "wb").close()
            return

        batches = [pairs_all[j:j + args.batch_size] for j in range(0, len(pairs_all), args.batch_size)]
        worker_args = [(b, args.target_confs, args.prune_rms, tmp_root) for b in batches]

        temp_paths = []
        n_ids_sum = 0
        n_rec_sum = 0

        ctx = get_context("spawn")
        total_batches = len(worker_args)
        step = max(1, int((args.progress_step / 100.0) * total_batches))

        logging.info(f"Starting pool: n_proc={args.n_proc}, batches={total_batches}")

        with ctx.Pool(processes=args.n_proc, maxtasksperchild=100) as pool:
            for k, (tmp_path, n_ids, n_rec) in enumerate(
                pool.imap_unordered(_worker_batch, worker_args, chunksize=1), start=1):

                temp_paths.append(tmp_path)
                n_ids_sum += n_ids
                n_rec_sum += n_rec

                if (k % step == 0) or (k == total_batches):
                    pct = int((k / total_batches) * 100)
                    logging.info(
                        f"Progress: {k}/{total_batches} batches ({pct}%) | "
                        f"IDs: {n_ids_sum} | records: {n_rec_sum}"
                    )

        _merge_sdfs(temp_paths, out_path)

        # cleanup
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        logging.info(f"[part_{args.part}] DONE | total IDs: {n_ids_sum} | total records: {n_rec_sum}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        logging.info(f"Temporary directory removed: {tmp_root}")


if __name__ == "__main__":
    main()
