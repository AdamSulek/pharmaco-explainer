#!/usr/bin/env python3
import argparse
import itertools
import logging
import os
import random
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
        raise EnvironmentError("PHARM_PROJECT_ROOT not set")
    return root

def unpack_ecfp(fp, n_bits=2048):
    if isinstance(fp, (bytes, bytearray)):
        return np.unpackbits(np.frombuffer(fp, dtype=np.uint8))[:n_bits].astype(np.float32)
    return np.asarray(fp, dtype=np.float32)

class MLP(torch.nn.Module):
    def __init__(self, in_features=2048, hidden_dim=128, num_hidden_layers=2, dropout_rate=0.2):
        super().__init__()
        layers = []
        last = in_features
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(last, hidden_dim))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout_rate))
            last = hidden_dim
        layers.append(torch.nn.Linear(last, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MyDataset(Dataset):
    def __init__(self, X_list, y_list):
        X = np.stack([unpack_ecfp(x) for x in X_list])
        if X.ndim != 2 or X.shape[1] != 2048:
            raise ValueError(f"Bad ECFP shape: {X.shape}")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_from_df(df):
    out = {"train": {"X": [], "y": []}, "val": {"X": [], "y": []}, "test": {"X": [], "y": []}}
    for _, r in df.iterrows():
        if r["split"] in out:
            out[r["split"]]["X"].append(r["X_ecfp_2"])
            out[r["split"]]["y"].append(r["y"])
    return {k: MyDataset(v["X"], v["y"]) for k, v in out.items()}

def train_and_evaluate(dataset, split_choice, seed=123):
    root = get_project_root()
    input_dir = os.path.join(root, "data", dataset)
    checkpoint_dir = os.path.join(root, "results", "checkpoints", dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(input_dir, f"{dataset}_split.parquet"))

    if split_choice == "split_distant_set":
        df["split"] = df["split_distant_set"]
    elif split_choice == "split_close_set":
        df["split"] = df["split_close_set"]

    val_keys = {"val", "valid", "validation", "dev"}
    val_name = next(s for s in df["split"].unique() if s in val_keys)
    df["split"] = df["split"].replace({val_name: "val"})

    datasets = load_data_from_df(df)

    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False)

    y_train = df[df["split"] == "train"]["y"].values
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / max(pos, 1), device=device)

    best = {"roc": -1, "params": None, "state": None}

    for comb in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, comb))

        train_loader = DataLoader(datasets["train"], batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=params["batch_size"], shuffle=False)

        model = MLP(
            hidden_dim=params["hidden_dim"],
            num_hidden_layers=params["num_hidden_layers"],
            dropout_rate=params["dropout_rate"]
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val = -1
        patience = 7
        stall = 0

        for _ in range(25):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(Xb).view(-1), yb)
                loss.backward()
                opt.step()

            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    p = torch.sigmoid(model(Xb.to(device)).view(-1))
                    ps.extend(p.cpu().numpy())
                    ys.extend(yb.numpy())

            try:
                roc = roc_auc_score(ys, ps)
            except:
                roc = 0

            if roc > best_val:
                best_val = roc
                stall = 0
                best_state = model.state_dict()
            else:
                stall += 1
                if stall >= patience:
                    break

        if best_val > best["roc"]:
            best = {"roc": best_val, "params": params, "state": best_state}
            torch.save(
                {"model_state_dict": best_state, "params": params},
                os.path.join(checkpoint_dir, f"best_model_mlp_{split_choice}.pth")
            )

    params = best["params"]
    model = MLP(
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
        dropout_rate=params["dropout_rate"]
    ).to(device)
    model.load_state_dict(best["state"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            p = torch.sigmoid(model(Xb.to(device)).view(-1))
            ps.extend(p.cpu().numpy())
            ys.extend(yb.numpy())

    pred = (np.array(ps) >= 0.5).astype(int)

    logging.info(f"ROC-AUC: {roc_auc_score(ys, ps):.4f}")
    logging.info(f"PR-AUC: {average_precision_score(ys, ps):.4f}")
    logging.info(f"ACC: {accuracy_score(ys, pred):.4f}")
    logging.info(f"F1: {f1_score(ys, pred):.4f}")
    logging.info(f"CM:\n{confusion_matrix(ys, pred)}")
    logging.info(f"BEST PARAMS: {params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--split", default="easy")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    seed_everything(args.seed)
    train_and_evaluate(args.dataset, args.split, args.seed)
