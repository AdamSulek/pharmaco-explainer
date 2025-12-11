#!/usr/bin/env python3
import argparse
import itertools
import logging
import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_GRID = {
    "learning_rate": [1e-4, 1e-5],
    "batch_size": [64],
    "num_hidden_layers": [2, 3],
    "dropout_rate": [0.15, 0.3],
    "hidden_dim": [64, 128]
}
PARAM_COMBINATIONS = list(itertools.product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError(
            "Environment variable PHARM_PROJECT_ROOT is not set.\n"
            "Run: export PHARM_PROJECT_ROOT=/path/to/project"
        )
    return root

class MLP(torch.nn.Module):
    def __init__(self, in_features=2048, hidden_dim=128, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        layers = []
        last_dim = in_features
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(torch.nn.Linear(last_dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MyDataset(Dataset):
    def __init__(self, X_list, y_list):
        arr = np.array(X_list, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2048:
            raise ValueError(f"Bad ECFP shape: {arr.shape}, expected (?, 2048)")
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_from_df(df):
    datasets = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}, "test": {"X": [], "y": []}}
    for _, row in df.iterrows():
        split = row["split"]
        if split in datasets:
            datasets[split]["X"].append(row["X_ecfp_2"])
            datasets[split]["y"].append(row['y'])
    return {k: MyDataset(v["X"], v["y"]) for k, v in datasets.items()}

def train_and_evaluate(dataset, split_choice, seed=123):
    root = get_project_root()
    input_dir = os.path.join(root, "data", dataset, "processed")
    checkpoint_dir = os.path.join(root, "results", "checkpoints", dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, "final_dataset_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logging.info(f"Loaded {len(df)} rows from {len(files)} files")

    if split_choice == "easy":
        df["split"] = df["split_easy"]
    elif split_choice == "hard":
        df["split"] = df["split_hard"]
    elif split_choice == "all":
        df["split"] = df["split_easy"]
    else:
        raise ValueError(f"Unknown split choice: {split_choice}")

    unique_splits = set(df["split"].dropna().unique())
    if "train" not in unique_splits or "test" not in unique_splits:
        raise ValueError(f"Missing train/test in splits: {unique_splits}")

    val_keys = {"val", "valid", "validation", "dev"}
    val_name = next((s for s in unique_splits if s in val_keys), None)
    if val_name is None:
        raise ValueError(f"No validation split found: {unique_splits}")
    df["split"] = df["split"].replace({val_name: "val"})
    logging.info(f"Using validation split: {val_name}")

    datasets = load_data_from_df(df)
    logging.info(f"Dataset sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}")

    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False)

    y_train = df[df["split"] == "train"]['y'].values
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32).to(device)

    best_overall = {"roc": -1, "params": None, "model_state": None}

    for comb in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, comb))
        logging.info(f"Training with params: {params}")

        train_loader = DataLoader(datasets["train"], batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=params["batch_size"], shuffle=False)

        model = MLP(
            in_features=2048,
            hidden_dim=params["hidden_dim"],
            num_hidden_layers=params["num_hidden_layers"],
            dropout_rate=params["dropout_rate"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_roc = -1
        patience = 7
        no_improve = 0

        for epoch in range(25):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb).view(-1)
                loss = criterion(logits, yb.view(-1))
                loss.backward()
                optimizer.step()

            model.eval()
            val_labels, val_preds = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    logits = model(Xb.to(device)).view(-1)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(yb.numpy())

            try:
                val_roc = roc_auc_score(val_labels, val_preds)
            except:
                val_roc = 0

            logging.info(f"[{params}] epoch {epoch+1} val_roc={val_roc:.4f}")

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                no_improve = 0
                best_state = model.state_dict()
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_val_roc > best_overall["roc"]:
            best_overall["roc"] = best_val_roc
            best_overall["params"] = params
            best_overall["model_state"] = best_state

            save_path = os.path.join(checkpoint_dir, f"best_model_mlp_{split_choice}.pth")
            torch.save({
                "model_state_dict": best_state,
                "params": params,
                "in_features": 2048
            }, save_path)
            logging.info(f"Saved BEST model → {save_path}")

    # ----- FINAL TEST EVAL -----
    params = best_overall["params"]
    model = MLP(
        in_features=2048,
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
        dropout_rate=params["dropout_rate"]
    ).to(device)
    model.load_state_dict(best_overall["model_state"])
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb.to(device)).view(-1)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(yb.numpy())

    test_roc = roc_auc_score(all_labels, all_preds)
    test_pr = average_precision_score(all_labels, all_preds)
    pred_labels = (np.array(all_preds) >= 0.5).astype(int)
    test_acc = accuracy_score(all_labels, pred_labels)
    test_f1 = f1_score(all_labels, pred_labels)
    cm = confusion_matrix(all_labels, pred_labels)

    logging.info("==== FINAL TEST METRICS ====")
    logging.info(f"ROC-AUC: {test_roc:.4f}")
    logging.info(f"PR-AUC: {test_pr:.4f}")
    logging.info(f"ACC: {test_acc:.4f}, F1: {test_f1:.4f}")
    logging.info(f"Confusion matrix:\n{cm}")
    logging.info(f"Best params:\n{params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3", choices=["k3","k4","k5"])
    parser.add_argument("--split", choices=["easy", "hard", "all"], default="easy")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    seed_everything(args.seed)
    train_and_evaluate(args.dataset, args.split, seed=args.seed)
