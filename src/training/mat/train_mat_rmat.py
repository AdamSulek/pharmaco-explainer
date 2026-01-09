#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from huggingmolecules import MatModel, MatFeaturizer, RMatModel, RMatFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def to_1d(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(-1) if t.dim() > 1 else t


def sigmoid_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).detach().cpu().numpy()


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    losses, y_true_all, y_prob_all = [], [], []

    for batch in loader:
        batch = batch.to(device)
        y = to_1d(batch.y).float()
        logits = to_1d(model(batch)).float()

        loss = loss_fn(logits, y)
        losses.append(loss.item())

        y_true_all.append(y.detach().cpu().numpy())
        y_prob_all.append(sigmoid_probs(logits))

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob) if y_true.size else float("nan")
    except Exception:
        auc = float("nan")

    y_pred = (y_prob >= 0.5).astype(int) if y_prob.size else np.array([])
    acc = accuracy_score(y_true, y_pred) if y_true.size else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0) if y_true.size else float("nan")

    return mean_loss, auc, acc, f1


@torch.no_grad()
def estimate_pos_weight(train_loader, device) -> float:
    n_pos = 0
    n_all = 0
    for batch in train_loader:
        y = to_1d(batch.y.to(device))
        n_pos += int((y > 0.5).sum().item())
        n_all += int(y.numel())

    n_neg = max(n_all - n_pos, 1)
    n_pos = max(n_pos, 1)
    return float(n_neg / n_pos)


def is_better(curr: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return (curr > best) if mode == "max" else (curr < best)


def load_pairs_from_dir(split_name: str, split_dir: str):
    """
    Loads samples stored as (feature_object, ID).
    Supports either a single file '<split>.p' or sharded files '<split>_*.p'.
    """
    single_path = os.path.join(split_dir, f"{split_name}.p")
    if os.path.exists(single_path):
        with open(single_path, "rb") as f:
            return pickle.load(f)

    pattern = os.path.join(split_dir, f"{split_name}_*.p")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing split file(s): {single_path} or {pattern}")

    out = []
    for pth in paths:
        with open(pth, "rb") as f:
            out.extend(pickle.load(f))
    return out


def filter_pairs_by_ids(pairs, allowed_ids):
    if allowed_ids is None:
        return pairs
    allowed = set(allowed_ids)
    return [(feat, ID) for (feat, ID) in pairs if ID in allowed]


def build_pos_dict(positive_pairs):
    """
    Builds ID -> feature mapping for positive samples.
    If duplicates exist, the last occurrence is used.
    """
    pos_by_id = {}
    dup = 0
    for feat_pos, ID in positive_pairs:
        if ID in pos_by_id:
            dup += 1
        pos_by_id[ID] = feat_pos
    if dup > 0:
        logging.warning("Duplicate IDs in positive pickle: %d (using last occurrence).", dup)
    return pos_by_id


def replace_positive_feats(pairs, pos_by_id):
    """
    Replaces feature objects for positive samples (y==1) by ID, if available in pos_by_id.
    """
    if not pos_by_id:
        return pairs

    replaced = 0
    out = []
    for feat, ID in pairs:
        y = getattr(feat, "y", None)
        if y is None:
            out.append((feat, ID))
            continue

        if isinstance(y, torch.Tensor):
            y_val = float(to_1d(y).detach().cpu().numpy().item())
        else:
            y_val = float(y)

        if y_val > 0.5 and ID in pos_by_id:
            out.append((pos_by_id[ID], ID))
            replaced += 1
        else:
            out.append((feat, ID))

    logging.info("Replaced positive features: %d", replaced)
    return out


def read_split_file(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser("Train MAT/RMAT on pre-featurized pickled datasets")

    ap.add_argument("--model", choices=["mat", "rmat"], required=True)
    ap.add_argument("--k", choices=["k3", "k4", "k5"], required=True)
    ap.add_argument("--subset", default="normal")

    ap.add_argument("--difficulty", choices=["normal", "easy", "hard", "none"], default="normal",
                    help="Dataset difficulty selection. 'easy'/'hard' filters samples using split_distant_set/split_close_set.")

    ap.add_argument("--data-root", dest="data_root",
                    default="/net/storage/pr3/plgrid/plggsanodrugs/pharmaco_explainer",
                    help="Project root containing 'pickle_dataloaders' and 'data'.")

    ap.add_argument("--data-split-file", dest="data_split_file", default=None,
                    help="Parquet/CSV with columns: ID and split_distant_set/split_close_set (and optionally val/test).")

    ap.add_argument("--positive-pickle-pos-path", dest="positive_pickle_pos_path", default=None,
                    help="Optional pickle: list of (feat_pos, ID) for positives. Replaces all y==1 features by ID.")

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-train", type=int, default=32)
    ap.add_argument("--batch-eval", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-every-batches", type=int, default=5000)

    ap.add_argument("--selection-metric", choices=["val_auc", "val_loss"], default="val_auc")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument("--results-pickle", default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    logging.info("Run: model=%s, k=%s, subset=%s, difficulty=%s", args.model, args.k, args.subset, args.difficulty)

    # --- Model and featurizer ---
    if args.model == "rmat":
        model = RMatModel.from_pretrained("rmat_4M")
        featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    else:
        model = MatModel.from_pretrained("mat_masking_20M")
        featurizer = MatFeaturizer.from_pretrained("mat_masking_20M")
    model.to(device)

    # --- Dataset paths ---
    pickle_root = os.path.join(args.data_root, "pickle_dataloaders", args.model, args.k, args.subset)
    train_dir = os.path.join(pickle_root, "train")
    val_dir = os.path.join(pickle_root, "val")
    test_dir = os.path.join(pickle_root, "test")

    train_pairs = load_pairs_from_dir("train", train_dir)
    val_pairs = load_pairs_from_dir("val", val_dir)
    test_pairs = load_pairs_from_dir("test", test_dir)

    # --- Difficulty-based filtering (optional) ---
    if args.difficulty in ("easy", "hard"):
        if args.data_split_file is None:
            k_int = int(args.k[1])
            auto_split = os.path.join(args.data_root, "data", args.k, f"ks{k_int}.parquet")
            if not os.path.exists(auto_split):
                raise FileNotFoundError(f"Missing split file: {auto_split}")
            args.data_split_file = auto_split

        df_split = read_split_file(args.data_split_file)
        id_col = "ID" if "ID" in df_split.columns else ("id" if "id" in df_split.columns else None)
        if id_col is None:
            raise KeyError("Split file must contain 'ID' or 'id' column.")

        split_col = f"split_{args.difficulty}"
        if split_col not in df_split.columns:
            raise KeyError(f"Split file must contain column: {split_col}")

        train_ids = df_split.loc[df_split[split_col] == "train", id_col].tolist()
        val_ids = df_split.loc[df_split[split_col] == "val", id_col].tolist()
        test_ids = df_split.loc[df_split[split_col] == "test", id_col].tolist()

        train_pairs = filter_pairs_by_ids(train_pairs, train_ids)
        val_pairs = filter_pairs_by_ids(val_pairs, val_ids)
        test_pairs = filter_pairs_by_ids(test_pairs, test_ids)

        logging.info("Filtered sizes: train=%d, val=%d, test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # --- Positive feature replacement (optional) ---
    if args.positive_pickle_pos_path is not None:
        if not os.path.exists(args.positive_pickle_pos_path):
            raise FileNotFoundError(f"Missing positive pickle: {args.positive_pickle_pos_path}")
        with open(args.positive_pickle_pos_path, "rb") as f:
            positive_pairs = pickle.load(f)
        pos_by_id = build_pos_dict(positive_pairs)

        train_pairs = replace_positive_feats(train_pairs, pos_by_id)
        val_pairs = replace_positive_feats(val_pairs, pos_by_id)
        test_pairs = replace_positive_feats(test_pairs, pos_by_id)

    train = [feat for (feat, _) in train_pairs]
    val = [feat for (feat, _) in val_pairs]
    test = [feat for (feat, _) in test_pairs]

    train_loader = featurizer.get_data_loader(train, batch_size=args.batch_train, shuffle=True)
    val_loader = featurizer.get_data_loader(val, batch_size=args.batch_eval, shuffle=False)
    test_loader = featurizer.get_data_loader(test, batch_size=args.batch_eval, shuffle=False)

    pos_w = estimate_pos_weight(train_loader, device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Checkpoint path ---
    if args.checkpoint_path is not None:
        best_ckpt = args.checkpoint_path
        ckpt_dir = os.path.dirname(best_ckpt) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        run_tag = f"{args.model}_{args.k}_{args.subset}_{args.difficulty}"
        ckpt_dir = args.ckpt_dir or os.path.join("checkpoints", "pharmaco", run_tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = os.path.join(ckpt_dir, "best_model.pth")

    mode = "max" if args.selection_metric == "val_auc" else "min"
    best_val = None
    no_improve = 0
    global_step = 0

    # --- Training ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            y = to_1d(batch.y).float()
            logits = to_1d(model(batch)).float()
            loss = loss_fn(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1

            if args.val_every_batches > 0 and (global_step % args.val_every_batches == 0):
                vloss, vauc, _, _ = evaluate(model, val_loader, device, loss_fn)
                score = vauc if args.selection_metric == "val_auc" else vloss
                if is_better(score, best_val, mode):
                    best_val = score
                    torch.save(model.state_dict(), best_ckpt)
                    no_improve = 0
                else:
                    no_improve += 1
                    if args.patience and no_improve >= args.patience:
                        break

        if args.patience and no_improve >= args.patience:
            break

        train_loss = running_loss / max(1, n_batches)
        vloss, vauc, _, _ = evaluate(model, val_loader, device, loss_fn)
        score = vauc if args.selection_metric == "val_auc" else vloss

        logging.info(
            "Epoch %d: train_loss=%.6f | val_loss=%.6f | val_auc=%.6f",
            epoch, train_loss, vloss, vauc
        )

        if is_better(score, best_val, mode):
            best_val = score
            torch.save(model.state_dict(), best_ckpt)
            no_improve = 0
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                break

    # --- Final test on best checkpoint ---
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_auc, test_acc, test_f1 = evaluate(model, test_loader, device, loss_fn)

    logging.info(
        "Test: loss=%.6f | auc=%.6f | acc=%.6f | f1=%.6f",
        test_loss, test_auc, test_acc, test_f1
    )

    if args.results_pickle is not None:
        os.makedirs(os.path.dirname(args.results_pickle) or ".", exist_ok=True)
        payload = {
            "k": args.k,
            "model": args.model,
            "subset": args.subset,
            "difficulty": args.difficulty,
            "selection_metric": args.selection_metric,
            "best_val": best_val,
            "test_loss": test_loss,
            "test_auc": test_auc,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
        with open(args.results_pickle, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
