#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pharmacophore label builder (explicit modes)

Modes:
- k3:  1×HBA + 1×HBD + 1×aromatic
- k4:  auto-select between:
       (A) 2×HBA + 1×HBD + 1×aromatic
       (B) 1×HBA + 1×HBD + 2×aromatic
- k5:  3×HBA + 1×HBD + 1×aromatic

Definitions:
- Conformer: a specific 3D geometry of the same molecule; each SDF record may correspond to a different conformer.
- Configuration (assignment): a selection of atoms satisfying a pharmacophore schema for a given conformer.

Per-SDF procedure:
- Group records by molecule ID (across conformers).
- For each conformer, enumerate all candidate configurations and evaluate geometric constraints.
- Track unique matched configurations across all conformers for a given ID (atom-index signature).
- Label y=1 iff exactly one unique configuration matches across all conformers of the ID.
- Report winner_conformer_id (first conformer producing the first unique match) and matched_conformer_ids (all conformers with any match).

Output:
- One CSV row per ID: ID, smiles, y, atom indices for the first unique configuration,
  sdf_source, winner_conformer_id, winner_name, matched_conformer_ids, n_conformers, n_matches.

Example:
  python build_labels.py \
      --part-dir sdf_files/part_000 \
      --plots-root plots \
      --out-root labels_out \
      --pharm k4 \
      --cpus 8 \
      --hypo-json hypothesis/k4.json \
      --log-level DEBUG \
      --log-configs
"""

import os
import sys
import glob
import logging
import argparse
import json
import re
import itertools
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.Draw import rdMolDraw2D


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger().setLevel(lvl)


_ff = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
_HALOGENS = {9, 17, 35, 53}
_CONF_RE = re.compile(r"([^#]+)#conf=(\d+)")


class PharmacophoreSpecError(AssertionError):
    pass


def _is_tuple3(x) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) == 3
        and all(isinstance(v, (int, float)) for v in x)
    )


def _validate_coords_list(name: str, pts) -> None:
    if pts is None:
        return
    if not isinstance(pts, list):
        raise PharmacophoreSpecError(f"'{name}' must be a list of (x,y,z); got {type(pts).__name__}")
    for i, p in enumerate(pts):
        if not _is_tuple3(p):
            raise PharmacophoreSpecError(f"'{name}[{i}]' must be a 3-tuple of numbers; got {p}")


def validate_runtime_params(tol_core: float, tol_ar: float) -> None:
    if tol_core < 0 or tol_ar < 0:
        raise PharmacophoreSpecError(
            f"Tolerances must be >= 0 (got tol_core={tol_core}, tol_ar={tol_ar})."
        )


def ensure_has_conformer(mol: Chem.Mol, id_hint: str = "") -> None:
    if mol is None or mol.GetNumConformers() == 0:
        raise PharmacophoreSpecError(f"Molecule {id_hint or ''} has no conformer(s).")


def _p(conf: Chem.Conformer, atom_idx: int) -> np.ndarray:
    v = conf.GetAtomPosition(int(atom_idx))
    return np.array([v.x, v.y, v.z], dtype=float)


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def fused_aromatic_groups(mol: Chem.Mol) -> List[List[int]]:
    """Return atom-index groups for each fused aromatic system."""
    m = Chem.RemoveHs(mol)
    aro_atoms = {a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()}
    if not aro_atoms:
        return []

    from collections import defaultdict

    adj = defaultdict(list)
    for b in m.GetBonds():
        if b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in aro_atoms and j in aro_atoms:
                adj[i].append(j)
                adj[j].append(i)

    seen = set()
    groups: List[List[int]] = []
    for a in sorted(aro_atoms):
        if a in seen:
            continue
        comp: List[int] = []
        stack = [a]
        seen.add(a)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        groups.append(sorted(comp))
    return groups


def get_HBD_heavy(mol: Chem.Mol) -> List[int]:
    """Donors (N/O/S with attached H): heavy-atom indices."""
    mh = Chem.AddHs(mol)
    donors = set()
    for a in mh.GetAtoms():
        z = a.GetAtomicNum()
        if z in (7, 8, 16) and any(n.GetAtomicNum() == 1 for n in a.GetNeighbors()):
            donors.add(a.GetIdx())
    return sorted(donors)


def get_HBA_heavy(mol: Chem.Mol) -> List[int]:
    """Acceptors via RDKit FeatureFactory with simple filtering (no halogens, no positively charged N)."""
    mh = Chem.AddHs(mol)
    feats = _ff.GetFeaturesForMol(mh)
    acc = {int(f.GetAtomIds()[0]) for f in feats if f.GetFamily() == "Acceptor"}

    out: List[int] = []
    m = Chem.RemoveHs(mol)
    for i in sorted(acc):
        ai = m.GetAtomWithIdx(i)
        z = ai.GetAtomicNum()
        if z in _HALOGENS:
            continue
        if z == 7 and ai.GetFormalCharge() > 0:
            continue
        out.append(i)
    return out


def _euclid(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def distances_k3(hipo: Dict) -> Dict[str, float]:
    _validate_coords_list("HBA", hipo.get("HBA"))
    _validate_coords_list("HBD", hipo.get("HBD"))
    _validate_coords_list("aromatic", hipo.get("aromatic"))
    if len(hipo["HBA"]) < 1 or len(hipo["HBD"]) < 1 or len(hipo["aromatic"]) < 1:
        raise PharmacophoreSpecError("k3 requires >=1 HBA, >=1 HBD, >=1 aromatic.")
    a, d, r = hipo["HBA"][0], hipo["HBD"][0], hipo["aromatic"][0]
    return {"A_D": _euclid(a, d), "R_A": _euclid(r, a), "R_D": _euclid(r, d)}


def distances_k4_ha2(hipo: Dict) -> Dict[str, float]:
    if len(hipo["HBA"]) < 2 or len(hipo["HBD"]) < 1 or len(hipo["aromatic"]) < 1:
        raise PharmacophoreSpecError("k4(2HBA) requires >=2 HBA, >=1 HBD, >=1 aromatic.")
    a1, a2 = hipo["HBA"][0], hipo["HBA"][1]
    d = hipo["HBD"][0]
    r = hipo["aromatic"][0]
    return {
        "HA1_HA2": _euclid(a1, a2),
        "HA1_HD": _euclid(a1, d),
        "HA2_HD": _euclid(a2, d),
        "Ar_HA1": _euclid(r, a1),
        "Ar_HA2": _euclid(r, a2),
        "Ar_HD": _euclid(r, d),
    }


def distances_k4_ar2(hipo: Dict) -> Dict[str, float]:
    if len(hipo["HBA"]) < 1 or len(hipo["HBD"]) < 1 or len(hipo["aromatic"]) < 2:
        raise PharmacophoreSpecError("k4(2Ar) requires >=1 HBA, >=1 HBD, >=2 aromatic.")
    a = hipo["HBA"][0]
    d = hipo["HBD"][0]
    r1, r2 = hipo["aromatic"][0], hipo["aromatic"][1]
    return {
        "A_D": _euclid(a, d),
        "R1_A": _euclid(r1, a),
        "R1_D": _euclid(r1, d),
        "R2_A": _euclid(r2, a),
        "R2_D": _euclid(r2, d),
        "R1_R2": _euclid(r1, r2),
    }


def distances_k5_ha3(hipo: Dict) -> Dict[str, float]:
    if len(hipo["HBA"]) < 3 or len(hipo["HBD"]) < 1 or len(hipo["aromatic"]) < 1:
        raise PharmacophoreSpecError("k5 requires >=3 HBA, >=1 HBD, >=1 aromatic.")
    a1, a2, a3 = hipo["HBA"][0], hipo["HBA"][1], hipo["HBA"][2]
    d = hipo["HBD"][0]
    r = hipo["aromatic"][0]
    return {
        "HA1_HA2": _euclid(a1, a2),
        "HA1_HA3": _euclid(a1, a3),
        "HA2_HA3": _euclid(a2, a3),
        "HA1_HD": _euclid(a1, d),
        "HA2_HD": _euclid(a2, d),
        "HA3_HD": _euclid(a3, d),
        "Ar_HA1": _euclid(r, a1),
        "Ar_HA2": _euclid(r, a2),
        "Ar_HA3": _euclid(r, a3),
        "Ar_HD": _euclid(r, d),
    }


def enum_k3(mol: Chem.Mol) -> List[Dict[str, List[int]]]:
    hbd = get_HBD_heavy(mol)
    hba = get_HBA_heavy(mol)
    arom_groups = fused_aromatic_groups(mol)

    out: List[Dict[str, List[int]]] = []
    for a in hba:
        for d in hbd:
            if arom_groups:
                for ar in arom_groups:
                    out.append({"HBA": [a], "HBD": [d], "aromatic": ar})
            else:
                out.append({"HBA": [a], "HBD": [d], "aromatic": []})
    return out


def enum_k4_ha2(mol: Chem.Mol) -> List[Dict[str, List[int]]]:
    hbd = get_HBD_heavy(mol)
    hba = get_HBA_heavy(mol)
    arom_groups = fused_aromatic_groups(mol)

    out: List[Dict[str, List[int]]] = []
    for ha1, ha2 in itertools.combinations(hba, 2):
        for hd in hbd:
            if arom_groups:
                for ar in arom_groups:
                    out.append({"HBA": [ha1, ha2], "HBD": [hd], "aromatic": ar})
            else:
                out.append({"HBA": [ha1, ha2], "HBD": [hd], "aromatic": []})
    return out


def enum_k4_ar2(mol: Chem.Mol) -> List[Dict[str, List[int]]]:
    hbd = get_HBD_heavy(mol)
    hba = get_HBA_heavy(mol)
    arom_groups = fused_aromatic_groups(mol)

    out: List[Dict[str, List[int]]] = []
    if len(arom_groups) < 2:
        return out

    for a in hba:
        for d in hbd:
            for ar1, ar2 in itertools.combinations(arom_groups, 2):
                out.append({"HBA": [a], "HBD": [d], "aromatic_pair": (ar1, ar2)})
    return out


def enum_k5_ha3(mol: Chem.Mol) -> List[Dict[str, List[int]]]:
    hbd = get_HBD_heavy(mol)
    hba = get_HBA_heavy(mol)
    arom_groups = fused_aromatic_groups(mol)

    out: List[Dict[str, List[int]]] = []
    if len(hba) < 3:
        return out

    for ha1, ha2, ha3 in itertools.combinations(hba, 3):
        for hd in hbd:
            if arom_groups:
                for ar in arom_groups:
                    out.append({"HBA": [ha1, ha2, ha3], "HBD": [hd], "aromatic": ar})
            else:
                out.append({"HBA": [ha1, ha2, ha3], "HBD": [hd], "aromatic": []})
    return out


def check_k3(
    mol: Chem.Mol,
    dist: Dict[str, float],
    config: Dict[str, List[int]],
    tol_core: float = 1.0,
    tol_ar: float = 2.0,
) -> bool:
    m = Chem.RemoveHs(mol)
    ensure_has_conformer(m)
    conf = m.GetConformer()

    a = config["HBA"][0]
    d = config["HBD"][0]
    ar_list = config.get("aromatic", [])
    if not ar_list:
        return False

    coords = {i: _p(conf, i) for i in set([a, d] + ar_list)}
    for r in ar_list:
        if (
            abs(_d(coords[a], coords[d]) - dist["A_D"]) <= tol_core
            and abs(_d(coords[r], coords[a]) - dist["R_A"]) <= tol_ar
            and abs(_d(coords[r], coords[d]) - dist["R_D"]) <= tol_ar
        ):
            return True
    return False


def check_k4_ha2(
    mol: Chem.Mol,
    dist: Dict[str, float],
    config: Dict[str, List[int]],
    tol_core: float = 1.0,
    tol_ar: float = 2.0,
) -> bool:
    m = Chem.RemoveHs(mol)
    ensure_has_conformer(m)
    conf = m.GetConformer()

    ha = config["HBA"]
    d = config["HBD"][0]
    ar_list = config.get("aromatic", [])
    if not ar_list:
        return False

    coords = {i: _p(conf, i) for i in set(ha + [d] + ar_list)}

    def core_ok(h1: int, h2: int) -> bool:
        return (
            abs(_d(coords[h1], coords[h2]) - dist["HA1_HA2"]) <= tol_core
            and abs(_d(coords[h1], coords[d]) - dist["HA1_HD"]) <= tol_core
            and abs(_d(coords[h2], coords[d]) - dist["HA2_HD"]) <= tol_core
        )

    order = None
    if core_ok(ha[0], ha[1]):
        order = (ha[0], ha[1])
    elif core_ok(ha[1], ha[0]):
        order = (ha[1], ha[0])
    else:
        return False

    h1, h2 = order
    for r in ar_list:
        if (
            abs(_d(coords[r], coords[h1]) - dist["Ar_HA1"]) <= tol_ar
            and abs(_d(coords[r], coords[h2]) - dist["Ar_HA2"]) <= tol_ar
            and abs(_d(coords[r], coords[d]) - dist["Ar_HD"]) <= tol_ar
        ):
            return True
    return False


def check_k4_ar2(
    mol: Chem.Mol,
    dist: Dict[str, float],
    config: Dict[str, List[int]],
    tol_core: float = 1.0,
    tol_ar: float = 2.0,
) -> bool:
    m = Chem.RemoveHs(mol)
    ensure_has_conformer(m)
    conf = m.GetConformer()

    a = config["HBA"][0]
    d = config["HBD"][0]
    ar1_atoms, ar2_atoms = config["aromatic_pair"]

    coords = {i: _p(conf, i) for i in set([a, d] + ar1_atoms + ar2_atoms)}

    def any_match(ar_atoms: List[int], key_a: str, key_d: str) -> bool:
        for r in ar_atoms:
            if (
                abs(_d(coords[r], coords[a]) - dist[key_a]) <= tol_ar
                and abs(_d(coords[r], coords[d]) - dist[key_d]) <= tol_ar
                and abs(_d(coords[a], coords[d]) - dist["A_D"]) <= tol_core
            ):
                return True
        return False

    scheme1 = any_match(ar1_atoms, "R1_A", "R1_D") and any_match(ar2_atoms, "R2_A", "R2_D")
    scheme2 = any_match(ar1_atoms, "R2_A", "R2_D") and any_match(ar2_atoms, "R1_A", "R1_D")
    if not (scheme1 or scheme2):
        return False

    for r1 in ar1_atoms:
        for r2 in ar2_atoms:
            if abs(_d(coords[r1], coords[r2]) - dist["R1_R2"]) <= tol_ar:
                return True
    return False


def check_k5_ha3(
    mol: Chem.Mol,
    dist: Dict[str, float],
    config: Dict[str, List[int]],
    tol_core: float = 1.0,
    tol_ar: float = 2.0,
) -> bool:
    from itertools import permutations

    m = Chem.RemoveHs(mol)
    ensure_has_conformer(m)
    conf = m.GetConformer()

    ha = config["HBA"]
    d = config["HBD"][0]
    ar_list = config.get("aromatic", [])
    if not ar_list:
        return False

    coords = {i: _p(conf, i) for i in set(ha + [d] + ar_list)}

    for h1, h2, h3 in permutations(ha, 3):
        core_ok = (
            abs(_d(coords[h1], coords[h2]) - dist["HA1_HA2"]) <= tol_core
            and abs(_d(coords[h1], coords[h3]) - dist["HA1_HA3"]) <= tol_core
            and abs(_d(coords[h2], coords[h3]) - dist["HA2_HA3"]) <= tol_core
            and abs(_d(coords[h1], coords[d]) - dist["HA1_HD"]) <= tol_core
            and abs(_d(coords[h2], coords[d]) - dist["HA2_HD"]) <= tol_core
            and abs(_d(coords[h3], coords[d]) - dist["HA3_HD"]) <= tol_core
        )
        if not core_ok:
            continue

        for r in ar_list:
            if (
                abs(_d(coords[r], coords[h1]) - dist["Ar_HA1"]) <= tol_ar
                and abs(_d(coords[r], coords[h2]) - dist["Ar_HA2"]) <= tol_ar
                and abs(_d(coords[r], coords[h3]) - dist["Ar_HA3"]) <= tol_ar
                and abs(_d(coords[r], coords[d]) - dist["Ar_HD"]) <= tol_ar
            ):
                return True
    return False


def plot_molecule_with_highlights(mol: Chem.Mol, atom_labels: Dict[str, List[int]]) -> rdMolDraw2D.MolDraw2DCairo:
    colors = {
        "HBD_label": (1, 0.647, 0),
        "HBA_label": (0, 1, 0),
        "aromatic": (1, 0.75, 0.8),
    }

    mdraw = Chem.Mol(mol)
    rdMolDraw2D.PrepareMolForDrawing(mdraw)

    highlight_atoms: List[int] = []
    highlight_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_bonds: List[int] = []
    bond_colors: Dict[int, Tuple[float, float, float]] = {}

    colored_atoms = set()
    for label in ["HBD_label", "HBA_label"]:
        for atom in atom_labels.get(label, []):
            idx = int(atom)
            highlight_atoms.append(idx)
            highlight_colors[idx] = colors[label]
            colored_atoms.add(idx)

    aromatic_atoms = set(int(a) for a in atom_labels.get("aromatic", []))
    for idx in aromatic_atoms:
        if idx not in colored_atoms:
            highlight_atoms.append(idx)
            highlight_colors[idx] = colors["aromatic"]

    for bond in mdraw.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in aromatic_atoms and a2 in aromatic_atoms:
            bidx = bond.GetIdx()
            highlight_bonds.append(bidx)
            bond_colors[bidx] = colors["aromatic"]

    drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
    drawer.DrawMolecule(
        mdraw,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer


def plot_match(mol: Chem.Mol, config: Dict, out_png: str) -> None:
    if "aromatic_pair" in config:
        ar = list(set(config["aromatic_pair"][0] + config["aromatic_pair"][1]))
    else:
        ar = config.get("aromatic", [])

    atom_labels = {"HBD_label": config["HBD"], "HBA_label": config["HBA"], "aromatic": ar}
    drawer = plot_molecule_with_highlights(mol, atom_labels)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with open(out_png, "wb") as f:
        f.write(drawer.GetDrawingText())


def _parse_id_conf_from_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    if not name:
        return None, None
    m = _CONF_RE.match(name)
    if m:
        mol_id, conf_id = m.groups()
        return mol_id, int(conf_id)
    mol_id = (name.split("#", 1)[0]) if "#" in name else name
    return mol_id, None


def get_dict_id_with_mol_confs(sdf_path: str) -> Dict[str, List[Chem.Mol]]:
    """Return a mapping: ID -> list of conformer records (each SDF record is treated as one conformer)."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mol_list = [mol for mol in suppl if mol is not None]

    from collections import defaultdict

    id_to_mols: Dict[str, List[Chem.Mol]] = defaultdict(list)
    for mol in mol_list:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        mol_id = mol.GetProp("ID") if mol.HasProp("ID") else None
        if mol_id is None:
            mol_id, _ = _parse_id_conf_from_name(name)
        if mol_id is None:
            continue

        conf_idx: Optional[int] = None
        if mol.HasProp("conf_id"):
            try:
                conf_idx = int(mol.GetProp("conf_id"))
            except Exception:
                conf_idx = None

        if conf_idx is None and name:
            _, conf_idx = _parse_id_conf_from_name(name)

        if conf_idx is not None:
            mol.SetIntProp("conf_idx", int(conf_idx))

        if name:
            mol.SetProp("_orig_name", name)

        id_to_mols[mol_id].append(mol)

    return id_to_mols


def get_smiles_from_mol_list(mol_list: List[Chem.Mol]) -> Optional[str]:
    for m in mol_list:
        if m and m.HasProp("smiles"):
            return m.GetProp("smiles")

    try:
        if mol_list:
            m0 = Chem.Mol(mol_list[0])
            Chem.SanitizeMol(m0, catchErrors=True)
            m0 = Chem.RemoveHs(m0)
            return Chem.MolToSmiles(m0, canonical=True, isomericSmiles=True)
    except Exception:
        return None

    return None


def config_signature(config: Dict) -> Tuple:
    """Configuration signature for uniqueness across conformers (atom selection only)."""

    def norm_list(vals) -> Tuple[int, ...]:
        try:
            return tuple(sorted(set(int(v) for v in vals)))
        except Exception:
            try:
                return tuple(sorted(set(int(v[0]) for v in vals)))
            except Exception:
                return tuple()

    if "aromatic_pair" in config:
        a1 = norm_list(config["aromatic_pair"][0])
        a2 = norm_list(config["aromatic_pair"][1])
        return (norm_list(config.get("HBA", [])), norm_list(config.get("HBD", [])), ("pair", a1, a2))

    return (norm_list(config.get("HBA", [])), norm_list(config.get("HBD", [])), norm_list(config.get("aromatic", [])))


def build_df_labels_from_sdf(
    sdf_path: str,
    hypothesis_points: Dict[str, List[Tuple[float, float, float]]],
    pharm_kind: str,
    tol_core: float = 1.0,
    tol_ar: float = 2.0,
    make_plots: bool = False,
    plots_root: Optional[str] = None,
    max_plots_per_id: int = 10,
    strict: bool = False,
    log_configs: bool = False,
) -> pd.DataFrame:
    logging.info("Loading SDF: %s", sdf_path)
    validate_runtime_params(tol_core, tol_ar)
    src_path = os.path.abspath(sdf_path)

    if pharm_kind == "k3":
        distances = distances_k3(hypothesis_points)
        enum = enum_k3
        check = check_k3
        k4_variant = None
    elif pharm_kind == "k4":
        n_ha = len(hypothesis_points.get("HBA", []))
        n_ar = len(hypothesis_points.get("aromatic", []))
        if n_ar >= 2 and n_ha >= 1:
            distances = distances_k4_ar2(hypothesis_points)
            enum = enum_k4_ar2
            check = check_k4_ar2
            k4_variant = "2Ar"
        elif n_ha >= 2 and n_ar >= 1:
            distances = distances_k4_ha2(hypothesis_points)
            enum = enum_k4_ha2
            check = check_k4_ha2
            k4_variant = "2HBA"
        else:
            raise PharmacophoreSpecError(
                "k4 requires either (>=1 HBA, >=1 HBD, >=2 aromatic) or (>=2 HBA, >=1 HBD, >=1 aromatic)."
            )
    elif pharm_kind == "k5":
        distances = distances_k5_ha3(hypothesis_points)
        enum = enum_k5_ha3
        check = check_k5_ha3
        k4_variant = None
    else:
        raise PharmacophoreSpecError(f"Unknown pharm_kind: {pharm_kind}")

    id2mols = get_dict_id_with_mol_confs(sdf_path)
    all_ids = sorted(id2mols.keys())

    tail = f" (k4_variant={k4_variant})" if (pharm_kind == "k4" and k4_variant) else ""
    logging.info("Found %d IDs with conformers. Mode: %s%s", len(all_ids), pharm_kind, tail)

    rows: List[Dict] = []
    for i, mol_id in enumerate(all_ids, start=1):
        mol_list = id2mols[mol_id]

        seen_signatures = set()
        unique_config_first: Optional[Dict[str, List[int]]] = None
        winner_conf_id: Optional[int] = None
        winner_name: Optional[str] = None
        matched_conf_ids: List[int] = []
        plots_done = 0

        for mol in mol_list:
            name = mol.GetProp("_orig_name") if mol.HasProp("_orig_name") else (mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            conf_id = mol.GetIntProp("conf_idx") if mol.HasProp("conf_idx") else None
            if conf_id is None and name:
                _, conf_id = _parse_id_conf_from_name(name)

            configs = enum(mol)
            if strict and not configs:
                raise PharmacophoreSpecError(f"No candidate configurations for ID={mol_id} in mode={pharm_kind}.")

            any_match_this_conformer = False

            for config in configs:
                ok = check(mol, distances, config, tol_core=tol_core, tol_ar=tol_ar)

                if log_configs:
                    sig_dbg = config_signature(config)
                    logging.debug(
                        "[ID=%s | conformer=%s] config_sig=%s -> %s",
                        mol_id,
                        conf_id,
                        sig_dbg,
                        "MATCH" if ok else "no_match",
                    )

                if not ok:
                    continue

                sig = config_signature(config)
                is_dup = sig in seen_signatures
                if not is_dup:
                    seen_signatures.add(sig)
                    any_match_this_conformer = True

                    if unique_config_first is None:
                        def to_int_list(v) -> List[int]:
                            try:
                                return [int(x) for x in v]
                            except Exception:
                                try:
                                    return [int(x[0]) for x in v]
                                except Exception:
                                    return []

                        if "aromatic_pair" in config:
                            ar_for_save = list(set(config["aromatic_pair"][0] + config["aromatic_pair"][1]))
                        else:
                            ar_for_save = config.get("aromatic", [])

                        unique_config_first = {
                            "HBA": to_int_list(config.get("HBA", [])),
                            "HBD": to_int_list(config.get("HBD", [])),
                            "aromatic": to_int_list(ar_for_save),
                        }
                        winner_conf_id = int(conf_id) if conf_id is not None else None
                        winner_name = name

                if any_match_this_conformer and conf_id is not None:
                    matched_conf_ids.append(int(conf_id))

                if make_plots and plots_root and plots_done < max_plots_per_id:
                    out_png = os.path.join(plots_root, mol_id, f"match_{plots_done + 1:03d}.png")
                    try:
                        plot_match(mol, config, out_png)
                        plots_done += 1
                    except Exception as exc:
                        logging.warning("Plot failed for %s: %s", mol_id, exc)

        smiles = get_smiles_from_mol_list(mol_list)
        n_confs = len(mol_list)
        n_matches = len(set(matched_conf_ids))
        y = 1 if len(seen_signatures) == 1 else 0

        if y == 1:
            rows.append(
                {
                    "ID": mol_id,
                    "smiles": smiles,
                    "y": 1,
                    "HBA": unique_config_first.get("HBA", []) if unique_config_first else [],
                    "HBD": unique_config_first.get("HBD", []) if unique_config_first else [],
                    "aromatic": unique_config_first.get("aromatic", []) if unique_config_first else [],
                    "sdf_source": src_path,
                    "winner_conf_id": winner_conf_id,
                    "winner_name": winner_name,
                    "matched_conf_ids": sorted(set(matched_conf_ids)),
                    "n_confs": n_confs,
                    "n_matches": n_matches,
                }
            )
        else:
            rows.append(
                {
                    "ID": mol_id,
                    "smiles": smiles,
                    "y": 0,
                    "HBA": [],
                    "HBD": [],
                    "aromatic": [],
                    "sdf_source": src_path,
                    "winner_conf_id": None,
                    "winner_name": None,
                    "matched_conf_ids": sorted(set(matched_conf_ids)),
                    "n_confs": n_confs,
                    "n_matches": n_matches,
                }
            )

        if i % 100 == 0 or i == len(all_ids):
            logging.info("Processed %d/%d IDs", i, len(all_ids))

    df_labels = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "smiles",
            "y",
            "HBA",
            "HBD",
            "aromatic",
            "sdf_source",
            "winner_conf_id",
            "winner_name",
            "matched_conf_ids",
            "n_confs",
            "n_matches",
        ],
    )
    logging.info(
        "df_labels built: %d rows (y=1: %d, y=0: %d)",
        len(df_labels),
        int((df_labels["y"] == 1).sum()),
        int((df_labels["y"] == 0).sum()),
    )
    return df_labels


def process_one_sdf(
    sdf_path: str,
    hypothesis: Dict,
    pharm_kind: str,
    tol_core: float,
    tol_ar: float,
    make_plots: bool,
    plots_root_for_sdf: str,
    out_csv_for_sdf: str,
    strict: bool = False,
    log_configs: bool = False,
) -> Optional[str]:
    try:
        df = build_df_labels_from_sdf(
            sdf_path=sdf_path,
            hypothesis_points=hypothesis,
            pharm_kind=pharm_kind,
            tol_core=tol_core,
            tol_ar=tol_ar,
            make_plots=make_plots,
            plots_root=plots_root_for_sdf,
            max_plots_per_id=10,
            strict=strict,
            log_configs=log_configs,
        )
        if df is None or df.empty:
            logging.info("[%s] No rows to save (empty df_labels).", os.path.basename(sdf_path))
            return None

        os.makedirs(os.path.dirname(out_csv_for_sdf), exist_ok=True)
        df.to_csv(out_csv_for_sdf, index=False)
        logging.info("[%s] Saved %d rows -> %s", os.path.basename(sdf_path), len(df), out_csv_for_sdf)
        return out_csv_for_sdf
    except Exception as exc:
        logging.exception("Failed on %s: %s", sdf_path, exc)
        return None


def merge_part_csvs(part_out_dir: str, out_merged_path: str) -> int:
    files = sorted(glob.glob(os.path.join(part_out_dir, "*_labels.csv")))
    dfs: List[pd.DataFrame] = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            logging.warning("Skipping %s: %s", fpath, exc)

    if not dfs:
        logging.info("No CSVs to merge in %s.", part_out_dir)
        return 0

    cat = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(out_merged_path), exist_ok=True)
    cat.to_csv(out_merged_path, index=False)
    logging.info("Merged %d files -> %s (%d rows)", len(dfs), out_merged_path, len(cat))
    return int(len(cat))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labels for all SDF files in a part directory, in parallel.")
    parser.add_argument("--part-dir", required=True, help="Path to part directory containing SDF files.")
    parser.add_argument("--plots-root", required=True, help="Root directory for plots.")
    parser.add_argument("--out-root", required=True, help="Root directory for CSV outputs.")
    parser.add_argument("--pharm", choices=["k3", "k4", "k5"], required=True, help="Pharmacophore schema.")
    parser.add_argument("--cpus", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--tol-core", type=float, default=1.0, help="Core tolerance (Å).")
    parser.add_argument("--tol-ar", type=float, default=2.0, help="Aromatic tolerance (Å).")
    parser.add_argument("--hypo-json", type=str, default=None, help="Path to hypothesis JSON with HBA/HBD/aromatic.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a molecule has no candidate configurations.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO).")
    parser.add_argument("--log-configs", action="store_true", help="Log each configuration check (verbose).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    part_dir = args.part_dir
    part_name = os.path.basename(part_dir.rstrip("/"))
    os.makedirs(args.plots_root, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)

    if args.hypo_json:
        if not os.path.isfile(args.hypo_json):
            logging.error("Hypothesis JSON not found: %s", args.hypo_json)
            sys.exit(1)
        with open(args.hypo_json, "r", encoding="utf-8") as f:
            hypothesis_raw = json.load(f)
    else:
        hypothesis_raw = {
            "HBA": [[13.62, -35.69, 13.90]],
            "HBD": [[11.27, -35.87, 14.33]],
            "aromatic": [[14.14, -36.57, 12.96]],
        }

    def _as_tuple3(x) -> Tuple[float, float, float]:
        return (float(x[0]), float(x[1]), float(x[2]))

    def normalize_hypothesis(hypo_raw: Dict) -> Dict[str, List[Tuple[float, float, float]]]:
        out = {"HBA": [], "HBD": [], "aromatic": []}
        for key in ("HBA", "HBD", "aromatic"):
            if key in hypo_raw and isinstance(hypo_raw[key], list):
                out[key] = [_as_tuple3(p) for p in hypo_raw[key]]
        if "hydrophobic" in hypo_raw and isinstance(hypo_raw["hydrophobic"], list):
            out["aromatic"].extend(_as_tuple3(p) for p in hypo_raw["hydrophobic"])
        return out

    hypothesis = normalize_hypothesis(hypothesis_raw)

    if not os.path.isdir(part_dir):
        logging.error("Part directory not found: %s", part_dir)
        sys.exit(1)

    sdf_files = sorted(glob.glob(os.path.join(part_dir, "*.sdf")))
    logging.info("[%s] Found %d SDF files.", part_name, len(sdf_files))
    if not sdf_files:
        logging.warning("[%s] No SDF files found in %s.", part_name, part_dir)
        sys.exit(0)

    futures = []
    with ProcessPoolExecutor(max_workers=max(1, args.cpus)) as ex:
        for sdf in sdf_files:
            sdf_stem = os.path.splitext(os.path.basename(sdf))[0]
            plots_root_for_sdf = os.path.join(args.plots_root, part_name, sdf_stem)
            out_csv_for_sdf = os.path.join(args.out_root, part_name, f"{sdf_stem}_labels.csv")
            futures.append(
                ex.submit(
                    process_one_sdf,
                    sdf,
                    hypothesis,
                    args.pharm,
                    args.tol_core,
                    args.tol_ar,
                    True,
                    plots_root_for_sdf,
                    out_csv_for_sdf,
                    args.strict,
                    args.log_configs,
                )
            )

        done = 0
        for fut in as_completed(futures):
            _ = fut.result()
            done += 1
            if done % 10 == 0 or done == len(futures):
                logging.info("[%s] Progress: %d/%d SDF processed", part_name, done, len(futures))

    part_out_dir = os.path.join(args.out_root, part_name)
    out_merged = os.path.join(args.out_root, f"{part_name}_labels_merged.csv")
    merge_part_csvs(part_out_dir, out_merged)


if __name__ == "__main__":
    main()
