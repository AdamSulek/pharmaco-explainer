import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.metrics import accuracy_score

from gcn_model import GCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# -------- PATH UTILS ------------
# ================================

def project_path(*parts):
    root = os.environ.get("PHARM_PROJECT_ROOT")
    if root is None:
        raise RuntimeError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run:\n"
            "  export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return os.path.join(root, *parts)


# ================================
# -------- MODEL LOADING ---------
# ================================

def load_gcn_model(checkpoint_path, use_hooks):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hyperparams"]

    model = GCN(
        input_dim=42,
        model_dim=hparams["model_dim"],
        concat_conv_layers=hparams.get("concat_conv_layers", 1),
        n_layers=hparams["n_layers"],
        dropout_rate=hparams["dropout_rate"],
        fc_hidden_dim=hparams["fc_hidden_dim"],
        num_fc_layers=hparams["num_fc_layers"],
        use_hooks=use_hooks
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ================================
# -------- DATA LOADING ----------
# ================================

def load_graph_data(dataset):
    path = project_path("data", dataset, "graph_data", "test.p")
    with open(path, "rb") as f:
        return pickle.load(f)


# ==========================================
# -------- EXPLAINABILITY METHODS ----------
# ==========================================

def identify_influential_nodes_gcn_gradcam(model, data, top_k=5):
    model.eval()
    data = data.to(device)

    out = model(data)
    model.zero_grad()
    out.sum().backward(retain_graph=True)

    cam = (model.final_conv_grads * model.final_conv_acts).sum(dim=1)
    _, top_idx = torch.topk(cam, top_k)

    return top_idx.cpu().numpy(), cam.detach().cpu().numpy()


def identify_influential_nodes_gcn_vg(model, data):
    model.eval()
    data = data.to(device)
    data.x = data.x.clone().detach().requires_grad_(True)

    out = model(data)
    model.zero_grad()
    out.sum().backward()

    node_imp = torch.linalg.norm(data.x.grad, dim=1)
    return node_imp.detach().cpu().numpy()


# ================================
# ----------- MAIN RUN ----------
# ================================

def run(dataset, model_name, split):
    logging.info("Loading graph data...")
    graph_data_list = load_graph_data(dataset)

    logging.info("Loading model...")
    checkpoint_path = project_path(
        "results",
        "checkpoints_gcn",
        dataset,
        f"best_model_split_{split}.pth"
    )

    model = load_gcn_model(
        checkpoint_path,
        use_hooks=(model_name.lower() == "gcn")
    )

    top_k = 5
    preds = []
    preds_mask = []
    atom_rows = []

    for i, data in enumerate(graph_data_list):
        data = data.to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(data)).cpu().item()
        preds.append(1 if pred >= 0.5 else 0)

        modified_data = data.clone()
        modified_data.x = modified_data.x.clone()

        if model_name.lower() == "gcn":
            top_nodes, node_importance = identify_influential_nodes_gcn_gradcam(
                model, data, top_k
            )
        else:
            node_importance = identify_influential_nodes_gcn_vg(model, data)
            top_nodes = np.argsort(-np.abs(node_importance))[:top_k]

        mol_id = getattr(data, "id", f"mol_{i}")
        for atom_idx, imp in enumerate(node_importance):
            atom_rows.append({
                "ID": mol_id,
                "atom_index": atom_idx,
                "shap_values": [float(imp)],
                "radiuses": [0]
            })

        for idx in top_nodes:
            modified_data.x[idx] = 0.0

        with torch.no_grad():
            masked_pred = torch.sigmoid(model(modified_data)).cpu().item()
        preds_mask.append(1 if masked_pred >= 0.5 else 0)

    y_true = np.array([
        data.y.detach().cpu().view(-1).item()
        for data in graph_data_list
    ])

    fidelity = accuracy_score(y_true, preds) - accuracy_score(y_true, preds_mask)
    logging.info(f"FIDELITY = {fidelity:.6f}")

    out_dir = project_path("results", "shap", dataset)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"fidelity": [fidelity]}).to_csv(
        os.path.join(out_dir, f"{model_name}_{split}_fidelity.csv"),
        index=False
    )

    atom_df = pd.DataFrame(atom_rows)[
        ["ID", "atom_index", "shap_values", "radiuses"]
    ]
    atom_df.to_parquet(
        os.path.join(out_dir, f"{model_name}_{split}_per_atom.parquet"),
        index=False
    )

    logging.info(f"Saved explainability results to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    run(args.dataset, args.model, args.split)


if __name__ == "__main__":
    main()
