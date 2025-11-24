#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_alignment_correctness.py

Diagnostic script for pharmacophore_alignment results.

For each positive SDF (single conformer) in:
    sdf_files/<k>_positive/<part-idx>/*.sdf

the script:

  1. Loads the hypothesis (3D points for HBA/HBD/aromatic) from JSON.
  2. Computes reference inter-point distances for the chosen pharmacophore mode:
       - k3       : 1×HBA + 1×HBD + 1×AR
       - k4_2hba  : 2×HBA + 1×HBD + 1×AR
       - k4_2ar   : 1×HBA + 1×HBD + 2×AR
       - k5       : 3×HBA + 1×HBD + 1×AR
  3. Enumerates all compatible atom combinations in the conformer
     (based on Donor/Acceptor features and aromatic atoms).
  4. For each combination, computes all relevant distances, compares to the
     hypothesis, and checks if they are within tolerances:
       tol_core – for “core” distances (without aromatic atoms)
       tol_ar   – for distances involving aromatic atoms
  5. For each SDF, keeps the best-fitting configuration (minimal total error)
     that passes all tolerances, and writes detailed diagnostics to CSV.

Output:
    diag_<k>_out/hipo_part_<part-idx>.csv
"""

import os
import glob
import json
import argparse
import logging
from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Check pharmacophore alignment distances for k3 / k4 / k4_2ar / k5 "
            "using single-conformer SDFs."
        )
    )
    ap.add_argument(
        "--k",
        required=True,
        choices=["k3", "k4", "k5"],
        help="Dataset key (k3, k4 or k5) – used for SDF directory naming."
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=["k3", "k4_2hba", "k4_2ar", "k5"],
        help=(
            "Pharmacophore mode to check:\n"
            "  k3      : 1×HBA + 1×HBD + 1×AR\n"
            "  k4_2hba : 2×HBA + 1×HBD + 1×AR (classic k4)\n"
            "  k4_2ar  : 1×HBA + 1×HBD + 2×AR\n"
            "  k5      : 3×HBA + 1×HBD + 1×AR"
        )
    )
    ap.add_argument(
        "--part-idx",
        type=int,
        required=True,
        help=(
            "Part index (e.g. SLURM_ARRAY_TASK_ID), corresponds to folder:\n"
            "  sdf_files/<k>_positive/<part-idx>/"
        )
    )
    ap.add_argument(
        "--hypo-json",
        required=True,
        help="Path to hypothesis JSON with HBA/HBD/aromatic points."
    )
    ap.add_argument(
        "--sdf-root",
        default="sdf_files",
        help="Root directory with SDFs (default: sdf_files)."
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSV diagnostics (default: diag_<k>_out)."
    )
    ap.add_argument(
        "--tol-core",
        type=float,
        default=1.0,
        help="Tolerance (Å) for core distances (no aromatic atoms)."
    )
    ap.add_argument(
        "--tol-ar",
        type=float,
        default=2.0,
        help="Tolerance (Å) for distances involving aromatic atoms."
    )
    return ap.parse_args()


# ----------------------------------------------------------------------
# Hypothesis / reference distances
# ----------------------------------------------------------------------

def load_hypothesis(path: str) -> Dict[str, np.ndarray]:
    """
    Load hypothesis with keys:
       HBA: list of 3D points
       HBD: list of 3D points
       aromatic: list of 3D points (>=1; for k4_2ar needs >=2)
    """
    with open(path, "r") as f:
        data = json.load(f)

    def to_array(key: str) -> np.ndarray:
        if key not in data:
            return np.zeros((0, 3), dtype=float)
        arr = np.array(data[key], dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    HBA = to_array("HBA")
    HBD = to_array("HBD")
    AROM = to_array("aromatic")

    return {"HBA": HBA, "HBD": HBD, "aromatic": AROM}


def distances_k3(hypo: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    k3: 1×HBA + 1×HBD + 1×AR
    Distances:
      A_D, R_A, R_D
    """
    if hypo["HBA"].shape[0] < 1 or hypo["HBD"].shape[0] < 1 or hypo["aromatic"].shape[0] < 1:
        logging.warning("Hypothesis for k3 has fewer than 1 HBA/HBD/aromatic point.")
    A = hypo["HBA"][0]
    D = hypo["HBD"][0]
    R = hypo["aromatic"][0]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return {
        "A_D": dist(A, D),
        "R_A": dist(R, A),
        "R_D": dist(R, D),
    }


def distances_k4_2hba(hypo: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    k4_2hba: 2×HBA + 1×HBD + 1×AR
    Distances:
      HBA1–HBD, HBA2–HBD, HBA1–HBA2,
      HBA1–AR, HBA2–AR, HBD–AR
    """
    if hypo["HBA"].shape[0] < 2 or hypo["HBD"].shape[0] < 1 or hypo["aromatic"].shape[0] < 1:
        logging.warning("Hypothesis for k4_2hba has fewer than 2 HBA / 1 HBD / 1 AR points.")

    HBA1 = hypo["HBA"][0]
    HBA2 = hypo["HBA"][1]
    HBD = hypo["HBD"][0]
    AR = hypo["aromatic"][0]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return {
        "HBA1_HBD": dist(HBA1, HBD),
        "HBA2_HBD": dist(HBA2, HBD),
        "HBA1_HBA2": dist(HBA1, HBA2),
        "HBA1_AR": dist(HBA1, AR),
        "HBA2_AR": dist(HBA2, AR),
        "HBD_AR": dist(HBD, AR),
    }


def distances_k4_2ar(hypo: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    k4_2ar: 1×HBA + 1×HBD + 2×AR
    Distances:
      A_D,
      R1_A, R1_D,
      R2_A, R2_D,
      R1_R2
    """
    if hypo["HBA"].shape[0] < 1 or hypo["HBD"].shape[0] < 1 or hypo["aromatic"].shape[0] < 2:
        logging.warning("Hypothesis for k4_2ar has fewer than 1 HBA / 1 HBD / 2 AR points.")

    A = hypo["HBA"][0]
    D = hypo["HBD"][0]
    R1 = hypo["aromatic"][0]
    R2 = hypo["aromatic"][1]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return {
        "A_D": dist(A, D),
        "R1_A": dist(R1, A),
        "R1_D": dist(R1, D),
        "R2_A": dist(R2, A),
        "R2_D": dist(R2, D),
        "R1_R2": dist(R1, R2),
    }


def distances_k5(hypo: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    k5: 3×HBA + 1×HBD + 1×AR
    Distances:
      HA1_HA2, HA1_HA3, HA2_HA3,
      HA1_HD,  HA2_HD,  HA3_HD,
      Ar_HA1,  Ar_HA2,  Ar_HA3,
      Ar_HD
    """
    if hypo["HBA"].shape[0] < 3 or hypo["HBD"].shape[0] < 1 or hypo["aromatic"].shape[0] < 1:
        logging.warning("Hypothesis for k5 has fewer than 3 HBA / 1 HBD / 1 AR points.")

    A1, A2, A3 = hypo["HBA"][0], hypo["HBA"][1], hypo["HBA"][2]
    D = hypo["HBD"][0]
    R = hypo["aromatic"][0]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return {
        "HA1_HA2": dist(A1, A2),
        "HA1_HA3": dist(A1, A3),
        "HA2_HA3": dist(A2, A3),
        "HA1_HD": dist(A1, D),
        "HA2_HD": dist(A2, D),
        "HA3_HD": dist(A3, D),
        "Ar_HA1": dist(R, A1),
        "Ar_HA2": dist(R, A2),
        "Ar_HA3": dist(R, A3),
        "Ar_HD": dist(R, D),
    }


# ----------------------------------------------------------------------
# Helpers on molecules
# ----------------------------------------------------------------------

def parse_id_conf_from_filename(fname: str) -> Tuple[str, int]:
    """
    Expect filenames of the form:
        <ID>_conf<NUM>.sdf

    Returns:
        (ID, conf_idx)
    """
    base = os.path.basename(fname)
    if not base.lower().endswith(".sdf"):
        raise ValueError(f"Not an SDF file: {fname}")
    stem = base[:-4]
    if "_conf" not in stem:
        raise ValueError(f"Missing '_conf' in file name: {fname}")
    id_part, conf_part = stem.split("_conf", 1)
    conf_idx = int(conf_part)
    return id_part, conf_idx


def get_hba_hbd_aromatic_atoms(mol: Chem.Mol) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract candidate atom indices:
      - HBA: from FeatureFactory (Acceptor)
      - HBD: from FeatureFactory (Donor)
      - aromatic: atoms with atom.GetIsAromatic() == True
    """
    feats = FACTORY.GetFeaturesForMol(mol)
    hba_atoms: List[int] = []
    hbd_atoms: List[int] = []
    for ftr in feats:
        fam = ftr.GetFamily()
        if fam == "Acceptor":
            hba_atoms.extend(ftr.GetAtomIds())
        elif fam == "Donor":
            hbd_atoms.extend(ftr.GetAtomIds())

    hba_atoms = sorted(set(hba_atoms))
    hbd_atoms = sorted(set(hbd_atoms))
    aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]

    return hba_atoms, hbd_atoms, aromatic_atoms


def atom_coord(conf: Chem.Conformer, idx: int) -> np.ndarray:
    """Return 3D coordinates of atom with given index from a conformer."""
    p = conf.GetAtomPosition(idx)
    return np.array([p.x, p.y, p.z], dtype=float)


# ----------------------------------------------------------------------
# Checkers for each mode
# ----------------------------------------------------------------------

def check_k3_configuration(
    hba_idx: int,
    hbd_idx: int,
    ar_idx: int,
    coords_hba: np.ndarray,
    coords_hbd: np.ndarray,
    coords_ar: np.ndarray,
    ref_dists: Dict[str, float],
    tol_core: float,
    tol_ar: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    k3: 1×HBA + 1×HBD + 1×AR.

    Distances checked:
      - A_D      vs ref["A_D"]      with tol_core
      - R_A, R_D vs ref["R_A"],ref["R_D"] with tol_ar
    """
    def dist(a, b):
        return float(np.linalg.norm(a - b))

    d_A_D = dist(coords_hba, coords_hbd)
    d_R_A = dist(coords_ar, coords_hba)
    d_R_D = dist(coords_ar, coords_hbd)

    delta_A_D = abs(d_A_D - ref_dists["A_D"])
    delta_R_A = abs(d_R_A - ref_dists["R_A"])
    delta_R_D = abs(d_R_D - ref_dists["R_D"])

    core_ok = delta_A_D <= tol_core
    ar_ok = (delta_R_A <= tol_ar) and (delta_R_D <= tol_ar)

    if not (core_ok and ar_ok):
        return False, {}

    total_err = delta_A_D + delta_R_A + delta_R_D

    result = {
        "HBA_atom_idx": hba_idx,
        "HBD_atom_idx": hbd_idx,
        "AR_atom_idx": ar_idx,
        "d_A_D_ref": ref_dists["A_D"],
        "d_A_D_mol": d_A_D,
        "delta_A_D": delta_A_D,
        "d_R_A_ref": ref_dists["R_A"],
        "d_R_A_mol": d_R_A,
        "delta_R_A": delta_R_A,
        "d_R_D_ref": ref_dists["R_D"],
        "d_R_D_mol": d_R_D,
        "delta_R_D": delta_R_D,
        "total_error": total_err,
    }
    return True, result


def check_k4_2hba_configuration(
    hba1_idx: int,
    hba2_idx: int,
    hbd_idx: int,
    coords_hba1: np.ndarray,
    coords_hba2: np.ndarray,
    coords_hbd: np.ndarray,
    arom_candidates: List[Tuple[int, np.ndarray]],
    ref_dists: Dict[str, float],
    tol_core: float,
    tol_ar: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    k4_2hba: 2×HBA + 1×HBD + 1×AR.

    First check the core (HBA–HBD, HBA–HBA) with tol_core; then
    search across all aromatic candidates for the best AR atom that
    fits (HBA1–AR, HBA2–AR, HBD–AR) within tol_ar, minimizing total error.
    """
    def dist(a, b):
        return float(np.linalg.norm(a - b))

    d_HBA1_HBD = dist(coords_hba1, coords_hbd)
    d_HBA2_HBD = dist(coords_hba2, coords_hbd)
    d_HBA1_HBA2 = dist(coords_hba1, coords_hba2)

    delta_HBA1_HBD = abs(d_HBA1_HBD - ref_dists["HBA1_HBD"])
    delta_HBA2_HBD = abs(d_HBA2_HBD - ref_dists["HBA2_HBD"])
    delta_HBA1_HBA2 = abs(d_HBA1_HBA2 - ref_dists["HBA1_HBA2"])

    core_ok = (
        delta_HBA1_HBD <= tol_core
        and delta_HBA2_HBD <= tol_core
        and delta_HBA1_HBA2 <= tol_core
    )
    if not core_ok:
        return False, {}

    best_ok = False
    best_err = float("inf")
    best_result: Dict[str, float] = {}

    for ar_idx, coord_ar in arom_candidates:
        d_HBA1_AR = dist(coords_hba1, coord_ar)
        d_HBA2_AR = dist(coords_hba2, coord_ar)
        d_HBD_AR = dist(coords_hbd, coord_ar)

        delta_HBA1_AR = abs(d_HBA1_AR - ref_dists["HBA1_AR"])
        delta_HBA2_AR = abs(d_HBA2_AR - ref_dists["HBA2_AR"])
        delta_HBD_AR = abs(d_HBD_AR - ref_dists["HBD_AR"])

        ar_ok = (
            delta_HBA1_AR <= tol_ar
            and delta_HBA2_AR <= tol_ar
            and delta_HBD_AR <= tol_ar
        )
        if not ar_ok:
            continue

        total_err = (
            delta_HBA1_HBD
            + delta_HBA2_HBD
            + delta_HBA1_HBA2
            + delta_HBA1_AR
            + delta_HBA2_AR
            + delta_HBD_AR
        )

        if total_err < best_err:
            best_err = total_err
            best_ok = True
            best_result = {
                "HBA1_atom_idx": hba1_idx,
                "HBA2_atom_idx": hba2_idx,
                "HBD_atom_idx": hbd_idx,
                "AR_atom_idx": ar_idx,
                "d_HBA1_HBD_ref": ref_dists["HBA1_HBD"],
                "d_HBA1_HBD_mol": d_HBA1_HBD,
                "delta_HBA1_HBD": delta_HBA1_HBD,
                "d_HBA2_HBD_ref": ref_dists["HBA2_HBD"],
                "d_HBA2_HBD_mol": d_HBA2_HBD,
                "delta_HBA2_HBD": delta_HBA2_HBD,
                "d_HBA1_HBA2_ref": ref_dists["HBA1_HBA2"],
                "d_HBA1_HBA2_mol": d_HBA1_HBA2,
                "delta_HBA1_HBA2": delta_HBA1_HBA2,
                "d_HBA1_AR_ref": ref_dists["HBA1_AR"],
                "d_HBA1_AR_mol": d_HBA1_AR,
                "delta_HBA1_AR": delta_HBA1_AR,
                "d_HBA2_AR_ref": ref_dists["HBA2_AR"],
                "d_HBA2_AR_mol": d_HBA2_AR,
                "delta_HBA2_AR": delta_HBA2_AR,
                "d_HBD_AR_ref": ref_dists["HBD_AR"],
                "d_HBD_AR_mol": d_HBD_AR,
                "delta_HBD_AR": delta_HBD_AR,
                "total_error": total_err,
            }

    return best_ok, best_result


def check_k4_2ar_configuration(
    hba_idx: int,
    hbd_idx: int,
    coords_hba: np.ndarray,
    coords_hbd: np.ndarray,
    arom_candidates: List[Tuple[int, np.ndarray]],
    ref_dists: Dict[str, float],
    tol_core: float,
    tol_ar: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    k4_2ar: 1×HBA + 1×HBD + 2×AR atoms.

    For each pair of aromatic atoms, check:
      - A_D vs ref["A_D"] with tol_core
      - R1_A, R1_D, R2_A, R2_D, R1_R2 vs ref[...] with tol_ar
    """
    from itertools import combinations

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    d_A_D = dist(coords_hba, coords_hbd)
    delta_A_D = abs(d_A_D - ref_dists["A_D"])
    if delta_A_D > tol_core:
        return False, {}

    best_ok = False
    best_err = float("inf")
    best_result: Dict[str, float] = {}

    for (ar1_idx, c_ar1), (ar2_idx, c_ar2) in combinations(arom_candidates, 2):
        d_R1_A = dist(c_ar1, coords_hba)
        d_R1_D = dist(c_ar1, coords_hbd)
        d_R2_A = dist(c_ar2, coords_hba)
        d_R2_D = dist(c_ar2, coords_hbd)
        d_R1_R2 = dist(c_ar1, c_ar2)

        delta_R1_A = abs(d_R1_A - ref_dists["R1_A"])
        delta_R1_D = abs(d_R1_D - ref_dists["R1_D"])
        delta_R2_A = abs(d_R2_A - ref_dists["R2_A"])
        delta_R2_D = abs(d_R2_D - ref_dists["R2_D"])
        delta_R1_R2 = abs(d_R1_R2 - ref_dists["R1_R2"])

        ar_ok = (
            delta_R1_A <= tol_ar
            and delta_R1_D <= tol_ar
            and delta_R2_A <= tol_ar
            and delta_R2_D <= tol_ar
            and delta_R1_R2 <= tol_ar
        )
        if not ar_ok:
            continue

        total_err = (
            delta_A_D
            + delta_R1_A
            + delta_R1_D
            + delta_R2_A
            + delta_R2_D
            + delta_R1_R2
        )

        if total_err < best_err:
            best_err = total_err
            best_ok = True
            best_result = {
                "HBA_atom_idx": hba_idx,
                "HBD_atom_idx": hbd_idx,
                "AR1_atom_idx": ar1_idx,
                "AR2_atom_idx": ar2_idx,
                "d_A_D_ref": ref_dists["A_D"],
                "d_A_D_mol": d_A_D,
                "delta_A_D": delta_A_D,
                "d_R1_A_ref": ref_dists["R1_A"],
                "d_R1_A_mol": d_R1_A,
                "delta_R1_A": delta_R1_A,
                "d_R1_D_ref": ref_dists["R1_D"],
                "d_R1_D_mol": d_R1_D,
                "delta_R1_D": delta_R1_D,
                "d_R2_A_ref": ref_dists["R2_A"],
                "d_R2_A_mol": d_R2_A,
                "delta_R2_A": delta_R2_A,
                "d_R2_D_ref": ref_dists["R2_D"],
                "d_R2_D_mol": d_R2_D,
                "delta_R2_D": delta_R2_D,
                "d_R1_R2_ref": ref_dists["R1_R2"],
                "d_R1_R2_mol": d_R1_R2,
                "delta_R1_R2": delta_R1_R2,
                "total_error": total_err,
            }

    return best_ok, best_result


def check_k5_configuration(
    hba_idxs: Tuple[int, int, int],
    hbd_idx: int,
    coords_hba1: np.ndarray,
    coords_hba2: np.ndarray,
    coords_hba3: np.ndarray,
    coords_hbd: np.ndarray,
    arom_candidates: List[Tuple[int, np.ndarray]],
    ref_dists: Dict[str, float],
    tol_core: float,
    tol_ar: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    k5: 3×HBA + 1×HBD + 1×AR.

    First check all core distances (HBA-HBA and HBA-HBD) with tol_core;
    then search over aromatic candidates for best AR atom.
    """
    def dist(a, b):
        return float(np.linalg.norm(a - b))

    # Core distances
    d_HA1_HA2 = dist(coords_hba1, coords_hba2)
    d_HA1_HA3 = dist(coords_hba1, coords_hba3)
    d_HA2_HA3 = dist(coords_hba2, coords_hba3)

    d_HA1_HD = dist(coords_hba1, coords_hbd)
    d_HA2_HD = dist(coords_hba2, coords_hbd)
    d_HA3_HD = dist(coords_hba3, coords_hbd)

    delta_HA1_HA2 = abs(d_HA1_HA2 - ref_dists["HA1_HA2"])
    delta_HA1_HA3 = abs(d_HA1_HA3 - ref_dists["HA1_HA3"])
    delta_HA2_HA3 = abs(d_HA2_HA3 - ref_dists["HA2_HA3"])

    delta_HA1_HD = abs(d_HA1_HD - ref_dists["HA1_HD"])
    delta_HA2_HD = abs(d_HA2_HD - ref_dists["HA2_HD"])
    delta_HA3_HD = abs(d_HA3_HD - ref_dists["HA3_HD"])

    core_ok = (
        delta_HA1_HA2 <= tol_core
        and delta_HA1_HA3 <= tol_core
        and delta_HA2_HA3 <= tol_core
        and delta_HA1_HD <= tol_core
        and delta_HA2_HD <= tol_core
        and delta_HA3_HD <= tol_core
    )
    if not core_ok:
        return False, {}

    best_ok = False
    best_err = float("inf")
    best_result: Dict[str, float] = {}

    for ar_idx, coord_ar in arom_candidates:
        d_Ar_HA1 = dist(coord_ar, coords_hba1)
        d_Ar_HA2 = dist(coord_ar, coords_hba2)
        d_Ar_HA3 = dist(coord_ar, coords_hba3)
        d_Ar_HD = dist(coord_ar, coords_hbd)

        delta_Ar_HA1 = abs(d_Ar_HA1 - ref_dists["Ar_HA1"])
        delta_Ar_HA2 = abs(d_Ar_HA2 - ref_dists["Ar_HA2"])
        delta_Ar_HA3 = abs(d_Ar_HA3 - ref_dists["Ar_HA3"])
        delta_Ar_HD = abs(d_Ar_HD - ref_dists["Ar_HD"])

        ar_ok = (
            delta_Ar_HA1 <= tol_ar
            and delta_Ar_HA2 <= tol_ar
            and delta_Ar_HA3 <= tol_ar
            and delta_Ar_HD <= tol_ar
        )
        if not ar_ok:
            continue

        total_err = (
            delta_HA1_HA2
            + delta_HA1_HA3
            + delta_HA2_HA3
            + delta_HA1_HD
            + delta_HA2_HD
            + delta_HA3_HD
            + delta_Ar_HA1
            + delta_Ar_HA2
            + delta_Ar_HA3
            + delta_Ar_HD
        )

        if total_err < best_err:
            best_err = total_err
            best_ok = True
            best_result = {
                "HBA1_atom_idx": hba_idxs[0],
                "HBA2_atom_idx": hba_idxs[1],
                "HBA3_atom_idx": hba_idxs[2],
                "HBD_atom_idx": hbd_idx,
                "AR_atom_idx": ar_idx,
                "d_HA1_HA2_ref": ref_dists["HA1_HA2"],
                "d_HA1_HA2_mol": d_HA1_HA2,
                "delta_HA1_HA2": delta_HA1_HA2,
                "d_HA1_HA3_ref": ref_dists["HA1_HA3"],
                "d_HA1_HA3_mol": d_HA1_HA3,
                "delta_HA1_HA3": delta_HA1_HA3,
                "d_HA2_HA3_ref": ref_dists["HA2_HA3"],
                "d_HA2_HA3_mol": d_HA2_HA3,
                "delta_HA2_HA3": delta_HA2_HA3,
                "d_HA1_HD_ref": ref_dists["HA1_HD"],
                "d_HA1_HD_mol": d_HA1_HD,
                "delta_HA1_HD": delta_HA1_HD,
                "d_HA2_HD_ref": ref_dists["HA2_HD"],
                "d_HA2_HD_mol": d_HA2_HD,
                "delta_HA2_HD": delta_HA2_HD,
                "d_HA3_HD_ref": ref_dists["HA3_HD"],
                "d_HA3_HD_mol": d_HA3_HD,
                "delta_HA3_HD": delta_HA3_HD,
                "d_Ar_HA1_ref": ref_dists["Ar_HA1"],
                "d_Ar_HA1_mol": d_Ar_HA1,
                "delta_Ar_HA1": delta_Ar_HA1,
                "d_Ar_HA2_ref": ref_dists["Ar_HA2"],
                "d_Ar_HA2_mol": d_Ar_HA2,
                "delta_Ar_HA2": delta_Ar_HA2,
                "d_Ar_HA3_ref": ref_dists["Ar_HA3"],
                "d_Ar_HA3_mol": d_Ar_HA3,
                "delta_Ar_HA3": delta_Ar_HA3,
                "d_Ar_HD_ref": ref_dists["Ar_HD"],
                "d_Ar_HD_mol": d_Ar_HD,
                "delta_Ar_HD": delta_Ar_HD,
                "total_error": total_err,
            }

    return best_ok, best_result


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    # Load hypothesis and reference distances for selected mode
    hypo = load_hypothesis(args.hypo_json)
    if args.mode == "k3":
        ref_dists = distances_k3(hypo)
    elif args.mode == "k4_2hba":
        ref_dists = distances_k4_2hba(hypo)
    elif args.mode == "k4_2ar":
        ref_dists = distances_k4_2ar(hypo)
    elif args.mode == "k5":
        ref_dists = distances_k5(hypo)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logging.info(f"Mode={args.mode} | reference distances: {ref_dists}")

    sdf_dir = os.path.join(args.sdf_root, f"{args.k}_positive", str(args.part_idx))
    if not os.path.isdir(sdf_dir):
        logging.error(f"SDF directory does not exist: {sdf_dir}")
        return

    out_dir = args.out_dir or f"diag_{args.k}_out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"hipo_part_{args.part_idx}_{args.mode}.csv")

    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    logging.info(
        f"Part {args.part_idx}: found {len(sdf_files)} SDF files in {sdf_dir}"
    )

    rows: List[Dict] = []

    from itertools import combinations

    for i, sdf_path in enumerate(sdf_files, start=1):
        try:
            mol_id, conf_idx = parse_id_conf_from_filename(sdf_path)
        except Exception as e:
            logging.warning(f"Skipping file {sdf_path}: {e}")
            continue

        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            logging.warning(f"No valid molecules in {sdf_path}")
            continue

        mol = mols[0]
        if mol.GetNumConformers() == 0:
            logging.warning(f"Molecule without conformers: {sdf_path}")
            continue

        conf = mol.GetConformer(0)

        hba_atoms, hbd_atoms, arom_atoms = get_hba_hbd_aromatic_atoms(mol)
        if not hbd_atoms or not arom_atoms or not hba_atoms:
            # Not enough pharmacophore elements to build a configuration
            continue

        # Precompute aromatic coordinates once
        arom_candidates = [(ar_idx, atom_coord(conf, ar_idx)) for ar_idx in arom_atoms]

        best_config_ok = False
        best_config_err = float("inf")
        best_row: Dict = {}

        if args.mode == "k3":
            # 1×HBA + 1×HBD + 1×AR
            for hba_idx in hba_atoms:
                c_hba = atom_coord(conf, hba_idx)
                for hbd_idx in hbd_atoms:
                    c_hbd = atom_coord(conf, hbd_idx)
                    for ar_idx, c_ar in arom_candidates:
                        ok, info = check_k3_configuration(
                            hba_idx,
                            hbd_idx,
                            ar_idx,
                            c_hba,
                            c_hbd,
                            c_ar,
                            ref_dists,
                            tol_core=args.tol_core,
                            tol_ar=args.tol_ar,
                        )
                        if not ok:
                            continue
                        err = info["total_error"]
                        if err < best_config_err:
                            best_config_err = err
                            best_config_ok = True
                            best_row = info.copy()

        elif args.mode == "k4_2hba":
            # 2×HBA + 1×HBD + 1×AR
            if len(hba_atoms) >= 2:
                for hba1_idx, hba2_idx in combinations(hba_atoms, 2):
                    c_hba1 = atom_coord(conf, hba1_idx)
                    c_hba2 = atom_coord(conf, hba2_idx)
                    for hbd_idx in hbd_atoms:
                        c_hbd = atom_coord(conf, hbd_idx)
                        ok, info = check_k4_2hba_configuration(
                            hba1_idx,
                            hba2_idx,
                            hbd_idx,
                            c_hba1,
                            c_hba2,
                            c_hbd,
                            arom_candidates,
                            ref_dists,
                            tol_core=args.tol_core,
                            tol_ar=args.tol_ar,
                        )
                        if not ok:
                            continue
                        err = info["total_error"]
                        if err < best_config_err:
                            best_config_err = err
                            best_config_ok = True
                            best_row = info.copy()

        elif args.mode == "k4_2ar":
            # 1×HBA + 1×HBD + 2×AR
            if len(arom_candidates) >= 2:
                for hba_idx in hba_atoms:
                    c_hba = atom_coord(conf, hba_idx)
                    for hbd_idx in hbd_atoms:
                        c_hbd = atom_coord(conf, hbd_idx)
                        ok, info = check_k4_2ar_configuration(
                            hba_idx,
                            hbd_idx,
                            c_hba,
                            c_hbd,
                            arom_candidates,
                            ref_dists,
                            tol_core=args.tol_core,
                            tol_ar=args.tol_ar,
                        )
                        if not ok:
                            continue
                        err = info["total_error"]
                        if err < best_config_err:
                            best_config_err = err
                            best_config_ok = True
                            best_row = info.copy()

        elif args.mode == "k5":
            # 3×HBA + 1×HBD + 1×AR
            if len(hba_atoms) >= 3:
                for hba1_idx, hba2_idx, hba3_idx in combinations(hba_atoms, 3):
                    c_hba1 = atom_coord(conf, hba1_idx)
                    c_hba2 = atom_coord(conf, hba2_idx)
                    c_hba3 = atom_coord(conf, hba3_idx)
                    for hbd_idx in hbd_atoms:
                        c_hbd = atom_coord(conf, hbd_idx)
                        ok, info = check_k5_configuration(
                            (hba1_idx, hba2_idx, hba3_idx),
                            hbd_idx,
                            c_hba1,
                            c_hba2,
                            c_hba3,
                            c_hbd,
                            arom_candidates,
                            ref_dists,
                            tol_core=args.tol_core,
                            tol_ar=args.tol_ar,
                        )
                        if not ok:
                            continue
                        err = info["total_error"]
                        if err < best_config_err:
                            best_config_err = err
                            best_config_ok = True
                            best_row = info.copy()

        if best_config_ok:
            row = {
                "ID": mol_id,
                "conf_idx": conf_idx,
                "sdf_path": sdf_path,
                "part_idx": args.part_idx,
                "mode": args.mode,
            }
            row.update(best_row)
            rows.append(row)

        if i % 100 == 0:
            logging.info(f"Part {args.part_idx}: processed {i} SDF files...")

    if not rows:
        logging.warning(
            f"Part {args.part_idx}, mode={args.mode}: no configurations matching hypothesis."
        )
        pd.DataFrame([]).to_csv(out_csv, index=False)
    else:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        logging.info(
            f"Part {args.part_idx}, mode={args.mode}: saved {len(df)} matches to {out_csv}"
        )


if __name__ == "__main__":
    main()
