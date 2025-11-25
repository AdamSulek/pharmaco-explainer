import argparse
import os
import glob
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
            # FIX: Removed inplace=True for gradient compatibility (SHAP, VG)
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


def aggregate_atom_from_bitinfo(bit_info, shap_vec):
    atom_map = {}
    for bit, atom_list in bit_info.items():
        for (atom_idx, radius) in atom_list:
            atom_idx = int(atom_idx)
            atom_map.setdefault(atom_idx, {"shap_values": [], "radiuses": []})
            val = float(shap_vec[int(bit)]) if int(bit) < len(shap_vec) else 0.0
            atom_map[atom_idx]["shap_values"].append(val)
            atom_map[atom_idx]["radiuses"].append(int(radius))
    rows = []
    for atom_idx, data in atom_map.items():
        rows.append({"atom_index": int(atom_idx), "shap_values": data["shap_values"], "radiuses": data["radiuses"]})
    return rows


def prepare_pred_fn_for_tree(model):
    if hasattr(model, "predict_proba"):
        def pred_fn(X):
            p = model.predict_proba(X)
            if p.ndim == 2:
                probs = p[:, 1]
            else:
                probs = p
            return (probs > 0.5).astype(int)
    else:
        def pred_fn(X):
            p = model.predict(X)
            p = np.array(p).reshape(-1)
            if p.dtype.kind == 'f':
                return (p > 0.5).astype(int)
            else:
                return p.astype(int)
    return pred_fn


def prepare_pred_fn_for_mlp(torch_model):
    def pred_fn(X):
        t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = torch_model(t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            return (probs > 0.5).astype(int)

    return pred_fn


def run(dataset, model_name, split):
    labels_path = f"../../../data/{dataset}/{dataset}_labels.parquet"
    logging.info("Loading labels")
    df_labels = pd.read_parquet(labels_path)
    allowed_ids = set(df_labels["ID"].tolist())

    files = sorted(glob.glob(f"../../../data/{dataset}/processed/final_dataset_part_*.parquet"))
    if len(files) == 0:
        raise FileNotFoundError("No data files found")

    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df = df[df["ID"].isin(allowed_ids)].reset_index(drop=True)
    # Sampling 1% of the data for explainability calculation speedup
    # df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
    if df.shape[0] == 0:
        raise ValueError("No records left after filtering")

    smiles = df["smiles"].tolist()
    X, bit_info_list = compute_fps(smiles)

    if model_name.lower() == "rf":
        logging.info("Loading RF model")
        model = load_rf(f"../../../results/checkpoints/{dataset}/best_model_rf_{split}.joblib")
        pred_fn = prepare_pred_fn_for_tree(model)
        expl_vals = shap_tree_values(model, X)  # This is the computationally heavy part

    elif model_name.lower() == "xgb":
        logging.info("Loading XGB model")
        model = load_xgb(f"../../../results/checkpoints/{dataset}/best_model_xgb_{split}.joblib")
        pred_fn = prepare_pred_fn_for_tree(model)
        expl_vals = shap_tree_values(model, X)

    elif model_name.lower() == "mlp":
        logging.info("Loading MLP model")
        torch_model = load_mlp(f"../../../results/checkpoints/{dataset}/best_model_mlp_{split}.pth", X.shape[1])
        pred_fn = prepare_pred_fn_for_mlp(torch_model)
        expl_vals = shap_deep_values(torch_model, X)

    elif model_name.lower() == "mlp_vg":
        logging.info("Loading MLP_VG model")
        torch_model = load_mlp(f"../../../results/checkpoints/{dataset}/best_model_mlp_{split}.pth", X.shape[1])
        pred_fn = prepare_pred_fn_for_mlp(torch_model)
        expl_vals = vanilla_gradients(torch_model, X)

    else:
        raise ValueError("Unknown model")

    preds = pred_fn(X)

    if expl_vals is None:
        expl_vals = np.zeros_like(X, dtype=float)
    if expl_vals.ndim == 3 and expl_vals.shape[0] == X.shape[0]:
        expl_vals = expl_vals.reshape(X.shape)

    masked_X = X.copy().astype(float)
    top_k = 5
    for i in range(X.shape[0]):
        vec = expl_vals[i]
        if np.isnan(vec).all():
            continue
        idxs = np.argsort(-np.abs(vec))[:top_k]
        for idx in idxs:
            masked_X[i, idx] = 1 - masked_X[i, idx]

    preds_mask = pred_fn(masked_X)

    acc = accuracy_score(preds, preds)
    masked_acc = accuracy_score(preds, preds_mask)
    fidelity = acc - masked_acc

    atom_rows = []
    for i in range(X.shape[0]):
        rows = aggregate_atom_from_bitinfo(bit_info_list[i], expl_vals[i])
        for r in rows:
            r["ID"] = df.iloc[i]["ID"]
            atom_rows.append(r)

    if len(atom_rows) == 0:
        atom_df = pd.DataFrame(columns=["ID", "atom_index", "shap_values", "radiuses"])
    else:
        atom_df = pd.DataFrame(atom_rows)[["ID", "atom_index", "shap_values", "radiuses"]]

    out_dir = f"../../../results/shap/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({
        "fidelity": [float(fidelity)],
        "accuracy": [float(acc)],
        "masked_accuracy": [float(masked_acc)]
    }).to_csv(f"{out_dir}/{model_name}_{split}_fidelity.csv", index=False)

    atom_df.to_parquet(f"{out_dir}/{model_name}_{split}_per_atom.parquet", index=False)

    logging.info(f"Fidelity: {fidelity:.6f}")
    logging.info("Results saved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--split", type=str, required=True)
    a = p.parse_args()
    run(a.dataset, a.model, a.split)


if __name__ == "__main__":
    main()