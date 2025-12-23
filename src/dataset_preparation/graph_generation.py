import numpy as np
import pickle
import pandas as pd
from rdkit import Chem
import logging
import torch
from torch_geometric.data import Data
import argparse
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def one_of_k_encoding(x, allowable_set, default=None):
    if x not in allowable_set:
        if default is None:
            raise ValueError(f"{x} not in allowable set {allowable_set}:")
        else:
            x = default
    return [x == s for s in allowable_set]


def get_atom_features(atom):
    ELEMENT_SYMBOLS = ["Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Unknown"]
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]

    result = []
    result += one_of_k_encoding(atom.GetSymbol(), ELEMENT_SYMBOLS, default="Unknown")
    result += one_of_k_encoding(atom.GetDegree(), list(range(11)))
    result += one_of_k_encoding(atom.GetImplicitValence(), list(range(7)), default=6)
    result += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    result += one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES,
                                default=Chem.rdchem.HybridizationType.SP3D2)
    result += [atom.GetIsAromatic()]
    result += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], default=4)
    return result


def get_bond_features(bond):
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    return one_of_k_encoding(bond.GetBondType(), BOND_TYPES, default=Chem.rdchem.BondType.AROMATIC)


def get_edge(bond):
    return [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]


def make_ligand_graphs(mol):
    edge_features = []
    edges = []

    for bond in mol.GetBonds():
        ef = get_bond_features(bond)
        e = get_edge(bond)

        edge_features.append(ef)
        edges.append(e)

        edge_features.append(ef)
        edges.append(e[::-1])

    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]

    edges = np.array(edges, dtype=np.int64)
    edge_features = np.array(edge_features, dtype=np.int64)
    node_features = np.array(node_features)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.long)

    return x, edge_index, edge_attr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Path to input parquet file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save train/val/test pickle files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_parquet):
        raise FileNotFoundError(f"Input file not found: {args.input_parquet}")

    logging.info(f"Reading input: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    logging.info(f"Data shape: {df.shape}")

    train_pickle_data, val_pickle_data, test_pickle_data = [], [], []

    for index, row in df.iterrows():
        split = row['split']
        mol = Chem.MolFromSmiles(row['smiles'])
        ID = row['ID']
        y = row['y']

        if mol is None:
            logging.warning(f"Invalid SMILES at index {index}, ID {ID}")
            continue

        x, edge_index, edge_attr = make_ligand_graphs(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    y=torch.tensor([y], dtype=torch.float), id=ID)

        
        if split == 'train':
            train_pickle_data.append(data)
        elif split == 'val':
            val_pickle_data.append(data)
        elif split == 'test':
            test_pickle_data.append(data)
        else:
            logging.warning(f"Invalid split '{split}' at index {index}")

        if index % 1000 == 0:
            logging.info(f"Processed {index} rows")

    with open(f"{args.output_dir}/train.p", "wb") as f:
        pickle.dump(train_pickle_data, f)

    with open(f"{args.output_dir}/val.p", "wb") as f:
        pickle.dump(val_pickle_data, f)

    with open(f"{args.output_dir}/test.p", "wb") as f:
        pickle.dump(test_pickle_data, f)

    logging.info("Done.")
