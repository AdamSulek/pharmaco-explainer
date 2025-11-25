import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

def smiles_to_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""

def scaffold_split(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)
    scaffold_dict = {}
    for idx, row in df.iterrows():
        s = row["scaffold"]
        scaffold_dict.setdefault(s, []).append(idx)

    scaffolds = list(scaffold_dict.keys())
    random.shuffle(scaffolds)

    n = len(df)
    train_cut = train_frac * n
    val_cut = (train_frac + val_frac) * n

    train_idx = []
    val_idx = []
    test_idx = []

    count = 0
    for s in scaffolds:
        ids = scaffold_dict[s]
        if count < train_cut:
            train_idx.extend(ids)
        elif count < val_cut:
            val_idx.extend(ids)
        else:
            test_idx.extend(ids)
        count += len(ids)

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(f"../../data/{args.dataset}/{args.dataset}.parquet")
    df["scaffold"] = df["smiles"].apply(smiles_to_scaffold)
    df = scaffold_split(df, 0.8, 0.1, 0.1, seed=args.seed)

    df.to_parquet(f"../../data/{args.dataset}/{args.dataset}_split.parquet", index=False)

    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
