import os
import glob
import json
import argparse
import logging
import re
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FDEF = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SDF conformers against a pharmacophore-like hypothesis (HBA/HBD/aromatic) "
            "using inter-point distance constraints."
        )
    )
    parser.add_argument("--k", required=True, choices=["k3", "k4", "k5"], help="Benchmark subset identifier.")
    parser.add_argument("--part-idx", type=int, required=True, help="Part index to process.")
    parser.add_argument("--hypo-json", required=True, help="Path to hypothesis JSON file.")
    parser.add_argument("--sdf-root", default="sdf_files", help="Root directory containing SDF files.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: diag_{k}_out).")
    parser.add_argument(
        "--tol-core",
        type=float,
        default=1.0,
        help="Distance tolerance (Å) for pairs not involving aromatic points.",
    )
    parser.add_argument(
        "--tol-ar",
        type=float,
        default=2.0,
        help="Distance tolerance (Å) for pairs involving aromatic points.",
    )
    return parser.parse_args()


def load_hypothesis(path: str) -> Dict[str, np.ndarray]:
    """
    Load hypothesis points (any number of points per type) for:
      - HBA: shape [n_hba, 3]
      - HBD: shape [n_hbd, 3]
      - aromatic: shape [n_ar, 3]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hba = np.array(data.get("HBA", []), dtype=float).reshape(-1, 3)
    hbd = np.array(data.get("HBD", []), dtype=float).reshape(-1, 3)
    arom = np.array(data.get("aromatic", []), dtype=float).reshape(-1, 3)

    if hba.size == 0 and hbd.size == 0 and arom.size == 0:
        raise ValueError(f"Hypothesis '{path}' appears to be empty (no HBA/HBD/aromatic points).")

    return {"HBA": hba, "HBD": hbd, "aromatic": arom}


def parse_id_conf_from_filename(fname: str) -> Tuple[str, int]:
    """
    Parse filenames of the form:
      <ID>_conf<NUM>.sdf
      <ID>_conf_<NUM>.sdf

    Example: 0003..._UN_conf_0.sdf
    """
    base = os.path.basename(fname)
    match = re.match(r"^(?P<id>.+?)_conf_?(?P<conf>\d+)\.sdf$", base, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to parse ID and conformer index from filename: {fname}")
    return match.group("id"), int(match.group("conf"))


def get_hba_hbd_aromatic_atoms(mol: Chem.Mol) -> Tuple[List[int], List[int], List[int]]:
    feats = FACTORY.GetFeaturesForMol(mol)

    hba_atoms: List[int] = []
    hbd_atoms: List[int] = []
    for feat in feats:
        fam = feat.GetFamily()
        if fam == "Acceptor":
            hba_atoms.extend(feat.GetAtomIds())
        elif fam == "Donor":
            hbd_atoms.extend(feat.GetAtomIds())

    hba_atoms = sorted(set(hba_atoms))
    hbd_atoms = sorted(set(hbd_atoms))
    aromatic_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    return hba_atoms, hbd_atoms, aromatic_atoms


def _label(ftype: str, idx: int) -> str:
    """Convert a point type + index into a stable label."""
    if ftype == "aromatic":
        return f"AR{idx + 1}"
    return f"{ftype}{idx + 1}"


def build_hypothesis_points(hypo: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert hypothesis arrays into a label->xyz mapping.

    Example:
      HBA1 -> [x,y,z]
      HBD1 -> [x,y,z]
      AR1  -> [x,y,z]
    """
    points: Dict[str, np.ndarray] = {}
    for ftype in ["HBA", "HBD", "aromatic"]:
        arr = hypo.get(ftype, np.zeros((0, 3)))
        for i in range(arr.shape[0]):
            points[_label(ftype, i)] = arr[i]
    return points


def compute_ref_distances(points: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """
    Compute reference distances for all pairs of hypothesis points.

    Key: (labelA, labelB) where labelA < labelB lexicographically.
    """
    labels = sorted(points.keys())
    ref: Dict[Tuple[str, str], float] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            ref[(a, b)] = float(np.linalg.norm(points[a] - points[b]))
    return ref


def _is_ar_label(lbl: str) -> bool:
    return lbl.startswith("AR")


def find_best_match(
    conf: Chem.Conformer,
    hba_atoms: List[int],
    hbd_atoms: List[int],
    arom_atoms: List[int],
    hypo_points: Dict[str, np.ndarray],
    ref_dists: Dict[Tuple[str, str], float],
    tol_core: float,
    tol_ar: float,
) -> Tuple[bool, Dict[str, float]]:
    """
    Match molecule atoms to hypothesis points by enumerating assignments and
    checking distance constraints. Returns the best assignment (minimum total error).
    """

    hba_labels = sorted([k for k in hypo_points.keys() if k.startswith("HBA")])
    hbd_labels = sorted([k for k in hypo_points.keys() if k.startswith("HBD")])
    ar_labels = sorted([k for k in hypo_points.keys() if k.startswith("AR")])

    need_hba = len(hba_labels)
    need_hbd = len(hbd_labels)
    need_ar = len(ar_labels)

    if len(hba_atoms) < need_hba or len(hbd_atoms) < need_hbd or len(arom_atoms) < need_ar:
        return False, {}

    def atom_coord(atom_idx: int) -> np.ndarray:
        p = conf.GetAtomPosition(atom_idx)
        return np.array([p.x, p.y, p.z], dtype=float)

    hba_coords = {idx: atom_coord(idx) for idx in hba_atoms}
    hbd_coords = {idx: atom_coord(idx) for idx in hbd_atoms}
    ar_coords = {idx: atom_coord(idx) for idx in arom_atoms}

    best_ok = False
    best_err = float("inf")
    best_out: Dict[str, float] = {}

    for hba_combo in combinations(hba_atoms, need_hba):
        for hba_perm in permutations(hba_combo, need_hba):
            map_hba = dict(zip(hba_labels, hba_perm))

            for hbd_combo in combinations(hbd_atoms, need_hbd):
                for hbd_perm in permutations(hbd_combo, need_hbd):
                    map_hbd = dict(zip(hbd_labels, hbd_perm))

                    assigned_idx = {**map_hba, **map_hbd}

                    ok_core = True
                    core_err = 0.0

                    for (a, b), d_ref in ref_dists.items():
                        if _is_ar_label(a) or _is_ar_label(b):
                            continue

                        ia = assigned_idx[a]
                        ib = assigned_idx[b]

                        if a.startswith("HBA"):
                            ca = hba_coords[ia]
                        elif a.startswith("HBD"):
                            ca = hbd_coords[ia]
                        else:
                            ok_core = False
                            break

                        if b.startswith("HBA"):
                            cb = hba_coords[ib]
                        elif b.startswith("HBD"):
                            cb = hbd_coords[ib]
                        else:
                            ok_core = False
                            break

                        d_mol = float(np.linalg.norm(ca - cb))
                        delta = abs(d_mol - d_ref)
                        if delta > tol_core:
                            ok_core = False
                            break

                        core_err += delta

                    if not ok_core:
                        continue

                    if need_ar == 0:
                        total_err = core_err
                        if total_err < best_err:
                            best_err = total_err
                            best_ok = True

                            out: Dict[str, float] = {}
                            for lbl, idx in map_hba.items():
                                out[f"{lbl}_atom_idx"] = float(idx)
                            for lbl, idx in map_hbd.items():
                                out[f"{lbl}_atom_idx"] = float(idx)

                            for (a, b), d_ref in ref_dists.items():
                                ia = assigned_idx[a]
                                ib = assigned_idx[b]
                                ca = hba_coords[ia] if a.startswith("HBA") else hbd_coords[ia]
                                cb = hba_coords[ib] if b.startswith("HBA") else hbd_coords[ib]
                                d_mol = float(np.linalg.norm(ca - cb))
                                pair = f"{a}_{b}"
                                out[f"d_{pair}_ref"] = float(d_ref)
                                out[f"d_{pair}_mol"] = float(d_mol)
                                out[f"delta_{pair}"] = float(abs(d_mol - d_ref))

                            out["total_err"] = float(total_err)
                            best_out = out
                        continue

                    for ar_combo in combinations(arom_atoms, need_ar):
                        for ar_perm in permutations(ar_combo, need_ar):
                            map_ar = dict(zip(ar_labels, ar_perm))
                            assigned_full = {**map_hba, **map_hbd, **map_ar}

                            ok_all = True
                            total_err = core_err
                            out: Dict[str, float] = {}

                            for lbl, idx in map_hba.items():
                                out[f"{lbl}_atom_idx"] = float(idx)
                            for lbl, idx in map_hbd.items():
                                out[f"{lbl}_atom_idx"] = float(idx)
                            for lbl, idx in map_ar.items():
                                out[f"{lbl}_atom_idx"] = float(idx)

                            for (a, b), d_ref in ref_dists.items():
                                ia = assigned_full[a]
                                ib = assigned_full[b]

                                if a.startswith("HBA"):
                                    ca = hba_coords[ia]
                                elif a.startswith("HBD"):
                                    ca = hbd_coords[ia]
                                else:
                                    ca = ar_coords[ia]

                                if b.startswith("HBA"):
                                    cb = hba_coords[ib]
                                elif b.startswith("HBD"):
                                    cb = hbd_coords[ib]
                                else:
                                    cb = ar_coords[ib]

                                d_mol = float(np.linalg.norm(ca - cb))
                                delta = abs(d_mol - d_ref)

                                tol = tol_ar if (_is_ar_label(a) or _is_ar_label(b)) else tol_core
                                if delta > tol:
                                    ok_all = False
                                    break

                                if _is_ar_label(a) or _is_ar_label(b):
                                    total_err += delta

                                pair = f"{a}_{b}"
                                out[f"d_{pair}_ref"] = float(d_ref)
                                out[f"d_{pair}_mol"] = float(d_mol)
                                out[f"delta_{pair}"] = float(delta)

                            if not ok_all:
                                continue

                            if total_err < best_err:
                                best_err = total_err
                                best_ok = True
                                out["total_err"] = float(total_err)
                                best_out = out

    return best_ok, best_out


def main() -> None:
    args = parse_args()

    hypothesis = load_hypothesis(args.hypo_json)
    hypo_points = build_hypothesis_points(hypothesis)
    ref_dists = compute_ref_distances(hypo_points)

    logging.info("Hypothesis points: %s", sorted(hypo_points.keys()))
    logging.info("Reference distances: %d pairs", len(ref_dists))

    sdf_dir = os.path.join(args.sdf_root, f"{args.k}_positive", str(args.part_idx))
    if not os.path.isdir(sdf_dir):
        logging.error("SDF directory does not exist: %s", sdf_dir)
        return

    out_dir = args.out_dir or f"diag_{args.k}_out"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"hipo_part_{args.part_idx}.csv")

    sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))
    logging.info("Part %s: found %d SDF files in %s", args.part_idx, len(sdf_files), sdf_dir)

    rows: List[Dict[str, float]] = []

    for i, sdf_path in enumerate(sdf_files, start=1):
        try:
            mol_id, conf_idx = parse_id_conf_from_filename(sdf_path)
        except Exception as exc:
            logging.warning("Skipping file %s: %s", sdf_path, exc)
            continue

        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            logging.warning("No valid molecules found in %s", sdf_path)
            continue

        mol = mols[0]
        if mol.GetNumConformers() == 0:
            logging.warning("Molecule has no conformers: %s", sdf_path)
            continue

        conf = mol.GetConformer(0)
        hba_atoms, hbd_atoms, arom_atoms = get_hba_hbd_aromatic_atoms(mol)

        ok, match = find_best_match(
            conf=conf,
            hba_atoms=hba_atoms,
            hbd_atoms=hbd_atoms,
            arom_atoms=arom_atoms,
            hypo_points=hypo_points,
            ref_dists=ref_dists,
            tol_core=args.tol_core,
            tol_ar=args.tol_ar,
        )

        if ok:
            row: Dict[str, float] = {
                "ID": mol_id,
                "conf_idx": float(conf_idx),
                "sdf_path": sdf_path,
                "part_idx": float(args.part_idx),
            }
            row.update(match)
            rows.append(row)

        if i % 100 == 0:
            logging.info("Part %s: processed %d SDF files...", args.part_idx, i)

    if not rows:
        logging.warning("Part %s: no conformers matched the hypothesis.", args.part_idx)
        pd.DataFrame([]).to_csv(out_csv, index=False)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logging.info("Part %s: wrote %d matches to %s", args.part_idx, len(df), out_csv)


if __name__ == "__main__":
    main()
