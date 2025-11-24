#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import ast
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from rdkit import Chem, rdBase
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import multiprocessing as mp

# ----------------------------- LOGGING ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------- RDKit setup -----------------------------

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")

FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF)
INCLUDE_ONLY = ("Acceptor", "Donor")


def _get_donor_acceptor_features(mol):
    """
    Safely retrieve only Donor/Acceptor features; provides a fallback
    for older RDKit versions that do not support includeOnly.
    """
    try:
        return FACTORY.GetFeaturesForMol(mol, includeOnly=INCLUDE_ONLY)
    except Exception:
        return [
            f for f in FACTORY.GetFeaturesForMol(mol)
            if f.GetFamily() in INCLUDE_ONLY
        ]


# ----------------------------- UTILS -----------------------------------

def convert_str_to_list(x) -> List[int]:
    """
    Convert a string/number/list-like value into a list of ints (atom indices).

    Handles:
      - None -> []
      - list/tuple/ndarray -> list of ints (skipping NaN/None)
      - int/float -> [int(value)] (if not NaN)
      - string -> tries literal_eval, otherwise extracts all digits via regex
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x if v is not None and not pd.isna(v)]
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    if isinstance(x, (float, np.floating)) and not pd.isna(x):
        return [int(x)]
    if isinstance(x, str):
        s = x.strip()
        if s in ("", "[]", "None", "nan", "NaN"):
            return []
        try:
            val = ast.literal_eval(s)
            return convert_str_to_list(val)
        except Exception:
            return [int(m) for m in re.findall(r"\d+", s)]
    return []


def _roc_auc_binary_numpy(y: np.ndarray, p: np.ndarray) -> float:
    """
    ROC AUC implementation without SciPy/Scikit-learn. Handles ties via average rank.

    AUC formula:
        AUC = (sum_rank_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    where ranks are assigned based on ascending prediction scores.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)

    if y.size < 2:
        raise ValueError("Need at least 2 samples")
    u = np.unique(y)
    if u.size != 2:
        raise ValueError("Binary labels required")

    n_pos = int((y == 1).sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes must be present")

    # Stable sort to ensure deterministic behavior on ties
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.empty_like(p_sorted, dtype=np.float64)

    n = p_sorted.size
    r = 1.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and p_sorted[j] == p_sorted[i]:
            j += 1
        # assign average rank for ties
        avg_rank = (r + (r + (j - i) - 1)) * 0.5
        ranks[i:j] = avg_rank
        r += (j - i)
        i = j

    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    ranks_orig = ranks[inv]

    sum_rank_pos = ranks_orig[y == 1].sum()
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def per_row_auc(y_list: List[np.ndarray], p_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute ROC AUC per row: each element is (y_vector, p_vector) for one molecule.

    Returns:
        np.ndarray of shape (n_rows,), with NaN for invalid/degenerate cases.
    """
    n = len(y_list)
    out = np.full(n, np.nan, dtype=float)
    for i, (y, p) in enumerate(zip(y_list, p_list)):
        if y is None or p is None:
            continue
        y = np.asarray(y)
        p = np.asarray(p, dtype=float)
        if y.shape[0] != p.shape[0] or y.shape[0] < 2:
            continue
        if np.unique(y).size < 2:
            # AUC undefined if only one class is present
            continue
        try:
            out[i] = _roc_auc_binary_numpy(y, p)
        except Exception:
            # leave as NaN
            pass
    return out


# ------------------------ CORE -----------------------------------------

def _labels_and_benchmark_for_smiles(
    smiles: str,
    label_indices: Dict[str, List[int]]
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute atom-level labels and simple benchmark for a single SMILES.

    For a given SMILES:
      - Parse the molecule once.
      - n_atoms: number of atoms.
      - y_true: binary vector with 1 for atoms labelled in label_indices
                (e.g. HBA/HBD/aromatic indices).
      - benchmark: atoms that are either aromatic or part of any
                   Donor/Acceptor feature from the FeatureFactory.

    Returns:
        (n_atoms, y_true, benchmark) where y_true and benchmark are np.ndarray or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, None, None

    n = mol.GetNumAtoms()
    y_true = np.zeros(n, dtype=np.uint8)
    benchmark = np.zeros(n, dtype=np.uint8)

    # Ground-truth labels from provided index lists
    for idx_list in label_indices.values():
        for idx in idx_list:
            if 0 <= idx < n:
                y_true[idx] = 1

    # Benchmark: aromatic atoms
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            benchmark[atom.GetIdx()] = 1

    # Benchmark: Donor/Acceptor atoms
    for feat in _get_donor_acceptor_features(mol):
        for idx in feat.GetAtomIds():
            benchmark[idx] = 1

    return n, y_true, benchmark


# --------------------- PARALLEL DISPATCH -------------------------------

def _worker(
    batch: List[Tuple[str, Dict[str, List[int]]]]
) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Worker function for multiprocessing.

    For each (smiles, label_indices) pair in the batch, compute:
      (n_atoms, y_true, benchmark).

    On any exception it returns a safe sentinel (0, None, None)
    to avoid pickling issues such as MaybeEncodingError.
    """
    out = []
    append = out.append
    for smi, idx_dict in batch:
        try:
            append(_labels_and_benchmark_for_smiles(smi, idx_dict))
        except Exception:
            append((0, None, None))
    return out


def compute_labels_and_benchmark_parallel(
    smiles: List[str],
    label_dicts: List[Dict[str, List[int]]],
    batch_size: int = 8000,
    nprocs: Optional[int] = None
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Parallel computation of (n_atoms, y_true, benchmark) for many SMILES.

    Args:
        smiles: list of SMILES strings.
        label_dicts: list of dicts, each with keys like ["HBA","HBD","aromatic"]
                     and values being lists of atom indices.
        batch_size: number of molecules per worker chunk.
        nprocs: number of worker processes (default: CPU count - 1).

    Returns:
        n_atoms_list, y_true_list, benchmark_list
    """
    assert len(smiles) == len(label_dicts)
    nprocs = nprocs or max(1, mp.cpu_count() - 1)
    tasks = list(zip(smiles, label_dicts))

    if len(tasks) <= batch_size or nprocs == 1:
        res = _worker(tasks)
    else:
        chunks = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        with mp.Pool(processes=nprocs, maxtasksperchild=200) as pool:
            res = []
            extend = res.extend
            for part in pool.imap_unordered(_worker, chunks, chunksize=1):
                extend(part)

    n_atoms_list, y_true_list, benchmark_list = [], [], []
    for n_atoms, y_true, bench in res:
        n_atoms_list.append(n_atoms)
        y_true_list.append(y_true)
        benchmark_list.append(bench)
    return n_atoms_list, y_true_list, benchmark_list


# ------------------------------- MAIN ----------------------------------

def main(args):
    k = args.k
    parquet_in = f"{args.root_folder}/{k}/{k}.parquet"
    parquet_out = f"{args.root_folder}/{k}/{k}_with_benchmark.parquet"
    label_cols = ("HBA", "HBD", "aromatic")

    logging.info(f"Processing dataset: {k}")
    logging.info(f"Input: {parquet_in}")
    logging.info(f"Output: {parquet_out}")

    df = pd.read_parquet(parquet_in)
    df = df.loc[df["y"] == 1].copy().reset_index(drop=True)

    # Normalize label columns to lists of atom indices
    for col in label_cols:
        if col not in df.columns:
            df[col] = [[]] * len(df)
        else:
            df.loc[:, col] = df[col].apply(convert_str_to_list)

    smiles = df["smiles"].tolist()
    label_dicts = df[list(label_cols)].to_dict("records")

    logging.info("Computing labels and benchmark vectors...")

    if args.use_mp:
        # RDKit may spawn internal threads; restrict if needed
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        n_atoms, y_true, benchmark = compute_labels_and_benchmark_parallel(
            smiles, label_dicts, batch_size=args.batch_size, nprocs=args.nprocs
        )
    else:
        n_atoms, y_true, benchmark = [], [], []
        for smi, idxs in zip(smiles, label_dicts):
            try:
                n, y, b = _labels_and_benchmark_for_smiles(smi, idxs)
            except Exception:
                n, y, b = 0, None, None
            n_atoms.append(n)
            y_true.append(y)
            benchmark.append(b)

    df_out = pd.DataFrame({
        "ID": df["ID"].values,
        "smiles": smiles,
        "n_atoms": n_atoms,
        "y_true": y_true,
        "benchmark": benchmark,
    })

    logging.info("Computing per-molecule ROC AUC for the benchmark...")
    df_out["roc_per_molecule_benchmark"] = per_row_auc(
        df_out["y_true"].tolist(), df_out["benchmark"].tolist()
    )

    os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
    df_out.to_parquet(parquet_out, index=False)

    logging.info(f"Done! Saved {len(df_out)} rows to {parquet_out}")


# ------------------------------- ENTRYPOINT ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare atom-level labels and a simple benchmark for dataset k3/k4/k5 "
            "(no sklearn/scipy, multiprocessing-safe)."
        )
    )
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        choices=["k3", "k4", "k5"],
        help="Dataset key (k3/k4/k5).",
    )
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder where <k>/<k>.parquet is stored.",
    )
    parser.add_argument(
        "--use-mp",
        action="store_true",
        help="Use multiprocessing for RDKit SMILES processing.",
    )
    parser.add_argument(
        "--no-mp",
        dest="use_mp",
        action="store_false",
        help="Disable multiprocessing (single-process mode).",
    )
    parser.set_defaults(use_mp=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8000,
        help="Batch size per worker in multiprocessing mode.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1).",
    )
    args = parser.parse_args()
    main(args)
