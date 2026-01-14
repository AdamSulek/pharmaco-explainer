import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MLP(torch.nn.Module):
    def __init__(self, in_features=2048, hidden_dim=128, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        layers = []
        last_dim = in_features
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(torch.nn.Linear(last_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_rf(path):
    return joblib.load(path)


def load_xgb(path):
    return joblib.load(path)


def load_mlp(path, fp_dim):
    checkpoint = torch.load(path, map_location="cpu")
    params = checkpoint["params"]
    m = MLP(
        in_features=checkpoint.get("in_features", fp_dim),
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
        dropout_rate=params["dropout_rate"]
    )
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m


def compute_fps(smiles_list, fp_size=2048, radius=2):
    X = np.zeros((len(smiles_list), fp_size), dtype=np.int8)
    bit_info_list = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bit_info_list.append({})
            continue
        bit_info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, bitInfo=bit_info)
        arr = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
        bit_info_list.append({int(k): v for k, v in bit_info.items()})
    return X, bit_info_list


def shap_tree_values(model, X):
    import shap
    logging.info("Starting SHAP TreeExplainer calculation...")
    explainer = shap.TreeExplainer(model)
    v = explainer.shap_values(X)
    logging.info("SHAP calculation finished.")
    if isinstance(v, list):
        if len(v) == 2:
            return np.array(v[1])
        return np.array(v)
    return np.array(v)


def shap_deep_values(torch_model, X):
    import shap
    logging.info("Starting SHAP DeepExplainer calculation...")
    N = X.shape[0]
    bg_n = min(200, max(1, N // 10))
    bg = torch.tensor(X[:bg_n], dtype=torch.float32)
    explainer = shap.DeepExplainer(torch_model, bg)
    sv = explainer.shap_values(torch.tensor(X, dtype=torch.float32), check_additivity=False)
    logging.info("SHAP calculation finished.")
    if isinstance(sv, list):
        return np.array(sv[0])
    if sv.ndim == 3 and sv.shape[1] == 1:
        return np.array(sv.squeeze(1))
    return np.array(sv)


def vanilla_gradients(torch_model, X):
    logging.info("Starting Vanilla Gradients calculation...")
    torch_model.eval()
    grads = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        x = torch.tensor(X[i:i + 1], dtype=torch.float32, requires_grad=True)
        out = torch_model(x).squeeze(-1).sum()
        out.backward()
        grads[i] = x.grad.detach().cpu().numpy()[0]
        x.grad.zero_()
    logging.info("Vanilla Gradients calculation finished.")
    return grads


def aggregate_atom_per_atom(bit_info, importance_vec):
    rows = []
    for bit, atom_list in bit_info.items():
        if bit >= len(importance_vec):
            val = 0.0
        else:
            val = float(importance_vec[bit])
        for atom_idx, radius in atom_list:
            rows.append({
                "atom_index": int(atom_idx),
                "atom_importance": [val],
                "radiuses": [int(radius)]
            })
    return rows

def aggregate_atom_per_molecule(per_atom_rows):
    agg = {}
    for row in per_atom_rows:
        ID = row["ID"]
        if ID not in agg:
            agg[ID] = []
        agg[ID].append(row["atom_importance"])
    return agg

def run(dataset, model_name, split):
    ROOT = os.environ.get("PHARM_PROJECT_ROOT")
    if ROOT is None:
        raise EnvironmentError("Please set the PHARM_PROJECT_ROOT environment variable!")

    df = pd.read_parquet(f"{ROOT}/data/{dataset}/{dataset}_split.parquet")
    smiles = df["smiles"].tolist()
    ids = df["ID"].tolist()

    X, bit_infos = compute_fps(smiles)

    ckpt = f"{ROOT}/results/checkpoints/{dataset}"
    if model_name == "rf":
        model = load_rf(f"{ckpt}/best_model_rf_{split}.joblib")
        expl = shap_tree_values(model, X)
    elif model_name == "xgb":
        model = load_xgb(f"{ckpt}/best_model_xgb_{split}.joblib")
        expl = shap_tree_values(model, X)
    elif model_name == "mlp":
        model = load_mlp(f"{ckpt}/best_model_mlp_{split}.pth", X.shape[1])
        expl = shap_deep_values(model, X)
    elif model_name == "mlp_vg":
        model = load_mlp(f"{ckpt}/best_model_mlp_{split}.pth", X.shape[1])
        expl = vanilla_gradients(model, X)
    else:
        raise ValueError("Unknown model")

    per_atom_rows = []
    for i in range(len(X)):
        rows = aggregate_atom_per_atom(bit_infos[i], expl[i])
        for r in rows:
            r["ID"] = ids[i]
            per_atom_rows.append(r)

    df_per_atom = pd.DataFrame(per_atom_rows)
    out_dir = f"{ROOT}/results/shap/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    atom_path = f"{out_dir}/{model_name}_{split}_per_atom.parquet"
    df_per_atom.to_parquet(atom_path, index=False)
    logging.info(f"Saved per-atom file: {atom_path}")

    agg_dict = aggregate_atom_per_molecule(per_atom_rows)
    df_agg = pd.DataFrame([
        {"ID": ID, "atom_importances": vals} for ID, vals in agg_dict.items()
    ])
    agg_path = f"{out_dir}/{model_name}_{split}_aggregate.parquet"
    df_agg.to_parquet(agg_path, index=False)
    logging.info(f"Saved aggregate file: {agg_path}")

    print("\n=== FIRST 5 ROWS PER ATOM ===")
    print(df_per_atom.head())
    print("\n=== FIRST 5 ROWS AGGREGATE ===")
    print(df_agg.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--model", default="rf")
    parser.add_argument("--split", default="split_close_set")
    args = parser.parse_args()

    run(args.dataset, args.model, args.split)
