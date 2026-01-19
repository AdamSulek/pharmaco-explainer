import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MLP(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=128, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        layers = []
        last_dim = in_features

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_rf(path):
    return joblib.load(path)


def load_xgb(path):
    return joblib.load(path)


def load_mlp(path, fp_dim):
    checkpoint = torch.load(path, map_location="cpu")
    params = checkpoint["params"]

    model = MLP(
        in_features=checkpoint.get("in_features", fp_dim),
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
        dropout_rate=params["dropout_rate"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_fps(smiles_list, fp_size=2048, radius=2):
    X = np.zeros((len(smiles_list), fp_size), dtype=np.int8)
    bit_info_list = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bit_info_list.append({})
            continue

        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=fp_size, bitInfo=bit_info
        )

        arr = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)

        X[i] = arr
        bit_info_list.append({int(k): v for k, v in bit_info.items()})

    return X, bit_info_list


def shap_tree_values(model, X):
    import shap
    logging.info("Starting SHAP TreeExplainer (Full Matrix)...")
    
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)

    if isinstance(vals, list):
        vals = vals[1] if len(vals) == 2 else vals

    return np.asarray(vals, dtype=np.float32)


def shap_deep_values(model, X):
    import shap
    logging.info("Starting SHAP DeepExplainer...")
    bg_n = min(200, max(1, len(X) // 10))
    bg = torch.tensor(X[:bg_n], dtype=torch.float32)

    explainer = shap.DeepExplainer(model, bg)
    vals = explainer.shap_values(
        torch.tensor(X, dtype=torch.float32),
        check_additivity=False,
    )

    if isinstance(vals, list):
        vals = vals[0]

    return np.asarray(vals, dtype=np.float32)


def vanilla_gradients(model, X):
    logging.info("Starting Vanilla Gradients...")
    model.eval()

    grads = np.zeros_like(X, dtype=np.float32)

    for i in range(len(X)):
        x = torch.tensor(X[i:i + 1], dtype=torch.float32, requires_grad=True)
        out = model(x).sum()
        out.backward()

        grads[i] = x.grad.detach().cpu().numpy()[0]
        x.grad.zero_()

    return grads


def aggregate_atom_per_atom(bit_info, importance_vec):
    rows = []
    vec = np.asarray(importance_vec).flatten()
    
    for bit, atom_list in bit_info.items():
        bit_idx = int(bit)
        val = float(vec[bit_idx])
        for atom_idx, radius in atom_list:
            rows.append({
                "atom_index": int(atom_idx),
                "atom_importance": [val],
                "radiuses": [int(radius)],
            })
    return rows


def aggregate_atom_per_molecule(per_atom_rows, smiles_dict, aggregate="mean"):
    mol_dict = {}
    for row in per_atom_rows:
        ID = row["ID"]
        atom_idx = int(row["atom_index"])
        val = float(row["atom_importance"][0])
        mol_dict.setdefault(ID, {}).setdefault(atom_idx, []).append(val)

    out = {}
    for ID, atom_map in mol_dict.items():
        smi = smiles_dict[ID]
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms() if mol else 0
        
        atom_importances = []
        for i in range(num_atoms):
            if i in atom_map:
                vals = atom_map[i]
                if aggregate == "mean":
                    agg_val = float(np.mean(vals))
                elif aggregate == "max":
                    agg_val = float(np.max(vals))
                else:
                    agg_val = float(np.sum(vals))
            else:
                agg_val = 0.0
            atom_importances.append(agg_val)

        out[ID] = atom_importances
    return out

def run(dataset, model_name, split, aggregate):
    ROOT = os.environ.get("PHARM_PROJECT_ROOT")
    if ROOT is None:
        raise EnvironmentError("Set PHARM_PROJECT_ROOT")

    labels_path = os.path.join(ROOT, "data", dataset, f"{dataset}_labels.parquet")
    df_labels = pd.read_parquet(labels_path)
    allowed_ids = set(df_labels["ID"])

    split_path = os.path.join(ROOT, "data", dataset, f"{dataset}_split.parquet")
    df = pd.read_parquet(split_path)
    df = df[df["ID"].isin(allowed_ids)].reset_index(drop=True)

    smiles = df["smiles"].tolist()
    ids = df["ID"].tolist()

    X, bit_infos = compute_fps(smiles)

    ckpt_dir = os.path.join(ROOT, "results", "checkpoints", dataset)

    if model_name == "rf":
        model = load_rf(os.path.join(ckpt_dir, f"best_model_rf_{split}.joblib"))
        expl_vals = shap_tree_values(model, X)
    elif model_name == "xgb":
        model = load_xgb(os.path.join(ckpt_dir, f"best_model_xgb_{split}.joblib"))
        expl_vals = shap_tree_values(model, X)
    elif model_name == "mlp":
        model = load_mlp(os.path.join(ckpt_dir, f"best_model_mlp_{split}.pth"), X.shape[1])
        expl_vals = shap_deep_values(model, X)
    elif model_name == "mlp_vg":
        model = load_mlp(os.path.join(ckpt_dir, f"best_model_mlp_{split}.pth"), X.shape[1])
        expl_vals = vanilla_gradients(model, X)
    else:
        raise ValueError("Unknown model")

    if expl_vals.ndim == 1:
        logging.info(f"Reshaping expl_vals from {expl_vals.shape} to (1, -1)")
        expl_vals = expl_vals.reshape(1, -1)

    per_atom_rows = []
    for i in range(len(X)):
        rows = aggregate_atom_per_atom(bit_infos[i], expl_vals[i])
        for r in rows:
            r["ID"] = ids[i]
            per_atom_rows.append(r)

    out_dir = os.path.join(ROOT, "results", "shap", dataset)
    os.makedirs(out_dir, exist_ok=True)

    df_per_atom = pd.DataFrame(per_atom_rows)
    df_per_atom.to_parquet(
        os.path.join(out_dir, f"{model_name}_{split}_per_atom.parquet"),
        index=False,
    )

    smiles_dict = dict(zip(ids, smiles))
    agg_dict = aggregate_atom_per_molecule(per_atom_rows, smiles_dict, aggregate)

    df_agg = pd.DataFrame(
        [{"ID": ID, "atom_importances": vals} for ID, vals in agg_dict.items()]
    )

    df_agg.to_parquet(
        os.path.join(out_dir, f"{model_name}_{split}_aggregate_{aggregate}.parquet"),
        index=False,
    )

    logging.info(f"DONE. Processed {len(df_agg)} molecules without batching.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="k3")
    parser.add_argument("--model", type=str, default="rf")
    parser.add_argument("--split", type=str, default="split_close_set")
    parser.add_argument("--aggregate", type=str, default="max", choices=["mean", "max", "sum"])

    args = parser.parse_args()
    run(args.dataset, args.model, args.split, args.aggregate)