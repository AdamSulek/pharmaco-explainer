import os
import random
import pickle
import logging
import argparse
import itertools

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from gcn_model import GCN

# ======================================================================
# LOGGING SETUP
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# UTILITIES
# ======================================================================

def get_project_root():
    root = os.getenv("PHARM_PROJECT_ROOT")
    if not root:
        raise EnvironmentError("PHARM_PROJECT_ROOT is not set. Run: export PHARM_PROJECT_ROOT=/path/to/project")
    return root

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def calculate_metrics(labels, predictions, threshold=0.5):
    pred_bin = (np.array(predictions) > threshold).astype(int)
    accuracy = accuracy_score(labels, pred_bin)
    roc_auc = roc_auc_score(labels, pred_bin)
    precision = precision_score(labels, pred_bin)
    recall = recall_score(labels, pred_bin)
    labels_arr = np.array(labels)
    TP = ((labels_arr == 1) & (pred_bin == 1)).sum()
    FP = ((labels_arr == 0) & (pred_bin == 1)).sum()
    TN = ((labels_arr == 0) & (pred_bin == 0)).sum()
    FN = ((labels_arr == 1) & (pred_bin == 0)).sum()
    return accuracy, roc_auc, precision, recall, TP, FP, TN, FN

def calculate_pos_weight(train_data, label_param):
    logging.info(f"[POS_WEIGHT] Calculating pos_weight for '{label_param}' on {len(train_data)} samples")
    labels = [float(getattr(e, label_param)) for e in train_data]
    pos = int(sum(labels))
    neg = int(len(labels) - pos)
    logging.info(f"[POS_WEIGHT] Positives={pos}, Negatives={neg}")
    pos_weight = neg / max(pos, 1)
    logging.info(f"[POS_WEIGHT] Computed pos_weight={pos_weight:.4f}")
    return pos_weight

# ======================================================================
# TRAIN & TEST
# ======================================================================

def train(model, loader, optimizer, criterion, threshold=0.5, label_param="label"):
    model.train()
    total_loss = 0
    labels_all = []
    preds_all = []

    for step, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze(-1)
        labels = getattr(data, label_param).float()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(out)
        total_loss += loss.item()
        labels_all.extend(labels.cpu().numpy())
        preds_all.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc, roc, prec, rec, TP, FP, TN, FN = calculate_metrics(labels_all, preds_all, threshold)
    return avg_loss, acc, roc, prec, rec, TP, FP, TN, FN

def test(model, loader, threshold=0.5, label_param="label"):
    model.eval()
    labels_all = []
    preds_all = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            labels = getattr(data, label_param).float()
            probs = torch.sigmoid(out)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(probs.detach().cpu().numpy().flatten())

    acc, roc, prec, rec, TP, FP, TN, FN = calculate_metrics(labels_all, preds_all, threshold)
    return acc, roc, prec, rec, TP, FP, TN, FN

# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat_conv_layers", type=int, default=1)
    parser.add_argument("--label_param", type=str, choices=["y", "activity", "label"], default="y")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--split_file_path", type=str, required=True)
    parser.add_argument("--split_type", type=str, choices=["split", "split_distant_set", "split_close_set"], default="split")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()

    seed_everything(123)
    root = get_project_root()

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    logging.info("[DATA] Loading datasets")
    with open(f"{args.input_dir}/train.p", "rb") as f:
        train_raw = pickle.load(f)
    with open(f"{args.input_dir}/val.p", "rb") as f:
        val_raw = pickle.load(f)
    with open(f"{args.input_dir}/test.p", "rb") as f:
        test_data = pickle.load(f)

    split_df = pd.read_parquet(args.split_file_path)[["ID", "split", args.split_type]]
    train_ids = split_df[split_df[args.split_type] == "train"]["ID"].tolist()
    val_ids = split_df[split_df[args.split_type] == "val"]["ID"].tolist()

    train_data = [d for d in train_raw if d.id in train_ids]
    val_data = [d for d in val_raw if d.id in val_ids]

    logging.info(f"[DATA] Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # -------------------------------
    # POS WEIGHT
    # -------------------------------
    pos_weight = calculate_pos_weight(train_data, args.label_param)

    # -------------------------------
    # DIRECTORIES
    # -------------------------------
    checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.k))
    result_dir = os.path.join(args.result_dir, str(args.k))
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # -------------------------------
    # HYPERPARAM GRID
    # -------------------------------
    lr_values = [0.001, 0.0001]
    batch_sizes = [32]
    conv_layers = [3, 4]
    model_dims = [512]
    dropout_rates = [0.0, 0.1]
    fc_hidden_dims = [128, 256]
    num_fc_layers = [1, 2]

    param_grid = list(itertools.product(
        lr_values, batch_sizes, conv_layers, model_dims,
        dropout_rates, fc_hidden_dims, num_fc_layers
    ))
    logging.info(f"[GRID] Total combinations: {len(param_grid)}")

    best_val_roc = float("-inf")
    best_hyperparams = None
    best_model_path = None

    # ==================================================================
    # GRID SEARCH
    # ==================================================================
    for lr, batch_size, n_layers, model_dim, dropout, fc_dim, fc_layers in param_grid:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = GCN(
            input_dim=42,
            model_dim=model_dim,
            concat_conv_layers=args.concat_conv_layers,
            n_layers=n_layers,
            dropout_rate=dropout,
            fc_hidden_dim=fc_dim,
            num_fc_layers=fc_layers
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

        patience = 20
        no_improve = 0

        for epoch in range(100):
            train_loss, train_acc, train_roc, *_ = train(
                model, train_loader, optimizer, criterion,
                threshold=0.5, label_param=args.label_param
            )
            val_acc, val_roc, *_ = test(
                model, val_loader, threshold=0.5, label_param=args.label_param
            )

            logging.info(
                f"[EPOCH {epoch+1}] "
                f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, ROC={train_roc:.4f} | "
                f"Val: Acc={val_acc:.4f}, ROC={val_roc:.4f}"
            )

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_hyperparams = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "n_layers": n_layers,
                    "model_dim": model_dim,
                    "dropout_rate": dropout,
                    "fc_hidden_dim": fc_dim,
                    "num_fc_layers": fc_layers,
                    "concat_conv_layers": args.concat_conv_layers
                }
                best_model_path = os.path.join(checkpoint_dir, f"best_model_gcn_{args.split_type}.pth")
                torch.save({"state_dict": model.state_dict(), "hyperparams": best_hyperparams}, best_model_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logging.info(f"[EARLY STOP] No improvement for {patience} epochs.")
                break

    # ==================================================================
    # LOAD BEST MODEL AND EVAL ON TEST
    # ==================================================================
    if best_model_path:
        logging.info(f"[BEST_MODEL] Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path)
        hyper = checkpoint["hyperparams"]

        best_model = GCN(
            input_dim=42,
            model_dim=hyper["model_dim"],
            concat_conv_layers=hyper["concat_conv_layers"],
            n_layers=hyper["n_layers"],
            dropout_rate=hyper["dropout_rate"],
            fc_hidden_dim=hyper["fc_hidden_dim"],
            num_fc_layers=hyper["num_fc_layers"],
            use_hooks=True
        ).to(device)
        best_model.load_state_dict(checkpoint["state_dict"])
        best_model.eval()

        test_loader = DataLoader(test_data, batch_size=hyper["batch_size"], shuffle=False)
        test_acc, test_roc, test_prec, test_rec, TP, FP, TN, FN = test(
            best_model, test_loader, threshold=0.5, label_param=args.label_param
        )

        logging.info("==== FINAL TEST METRICS ====")
        logging.info(f"Accuracy: {test_acc:.4f}")
        logging.info(f"ROC-AUC: {test_roc:.4f}")
        logging.info(f"Precision: {test_prec:.4f}")
        logging.info(f"Recall: {test_rec:.4f}")
        logging.info(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    else:
        logging.error("No best model was saved — training failed to produce any model.")

    logging.info("Training completed.")

if __name__ == "__main__":
    main()
