#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, pickle, argparse, logging
from rdkit import Chem
from huggingmolecules import MatModel, MatFeaturizer, RMatModel, RMatFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def patch_encode_mol_list():
    # tylko do trybu with_pos / with_y
    def encode_mol_list(self, mol_list, y_list=None):
        if y_list is None:
            y_list = [None] * len(mol_list)
        else:
            assert len(mol_list) == len(y_list), "mol_list and y_list must have the same length."

        enc = []
        for mol, y in zip(mol_list, y_list):
            if mol is None:
                continue
            enc.append(self._encode_mol(mol, y))
        return enc

    def _encode_mol_via_smiles(self, mol, y):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return self._encode_smiles(smiles, y)

    MatFeaturizer.encode_mol_list = encode_mol_list
    RMatFeaturizer.encode_mol_list = encode_mol_list

    if not hasattr(MatFeaturizer, "_encode_mol"):
        MatFeaturizer._encode_mol = _encode_mol_via_smiles
    if not hasattr(RMatFeaturizer, "_encode_mol"):
        RMatFeaturizer._encode_mol = _encode_mol_via_smiles


def load_featurizer(model_type):
    if model_type == "mat":
        MatModel.from_pretrained("mat_masking_20M")
        return MatFeaturizer.from_pretrained("mat_masking_20M")
    else:
        RMatModel.from_pretrained("rmat_4M")
        return RMatFeaturizer.from_pretrained("rmat_4M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["mat", "rmat"], default="mat")
    ap.add_argument("--mode", choices=["no_pos", "with_pos"], default="no_pos")
    ap.add_argument("--sdf-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-name", default=None)
    ap.add_argument("--y-value", type=int, default=1, help="used only in --mode with_pos")
    args = ap.parse_args()

    if args.output_name is None:
        args.output_name = f"featurized_{args.model_type}_{args.mode}.p"

    if args.mode == "with_pos":
        patch_encode_mol_list()

    featurizer = load_featurizer(args.model_type)

    sdf_files = sorted(glob.glob(os.path.join(args.sdf_dir, "*.sdf")))
    if not sdf_files:
        raise FileNotFoundError(f"No *.sdf in {args.sdf_dir}")

    mol_list, smiles_list, id_list = [], [], []

    for p in sdf_files:
        suppl = Chem.SDMolSupplier(p, removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue

            # ID (najprościej: z _Name albo fallback na nazwę pliku)
            mol_id = None
            if mol.HasProp("ID"):
                mol_id = mol.GetProp("ID")
            elif mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
                mol_id = name.split("#", 1)[0] if name else None
            if not mol_id:
                mol_id = os.path.basename(p).rsplit(".", 1)[0]

            mol_list.append(mol)
            smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
            id_list.append(str(mol_id))

    logging.info("Loaded %d molecules", len(id_list))

    if args.mode == "no_pos":
        # native HM path: smiles
        if hasattr(featurizer, "encode_smiles_list"):
            encoded = featurizer.encode_smiles_list(smiles_list)
        else:
            encoded = [featurizer._encode_smiles(smi, None) for smi in smiles_list]
    else:
        # with_pos path: mol_list + y
        y_list = [int(args.y_value)] * len(mol_list)
        encoded = featurizer.encode_mol_list(mol_list, y_list)

    if not encoded:
        raise RuntimeError("Empty encoding list.")
    if len(encoded) != len(id_list):
        raise RuntimeError(f"Count mismatch: enc={len(encoded)} ids={len(id_list)}")

    out = []
    if args.mode == "with_pos":
        for feat, mol_id in zip(encoded, id_list):
            feat.y = int(args.y_value)
            out.append((feat, mol_id))
    else:
        for feat, mol_id in zip(encoded, id_list):
            out.append((feat, mol_id))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
