#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pickle
import glob

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from huggingmolecules import RMatModel, RMatFeaturizer, MatModel, MatFeaturizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------- helpers ----------------
def to_1d(t: torch.Tensor) -> torch.Tensor:
    """Flatten tensor to shape (N,) if it has an extra last dimension."""
    return t.squeeze(-1) if t.dim() > 1 else t


def probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    """Convert raw logits to probabilities via sigmoid and move to CPU numpy."""
    return torch.sigmoid(logits).detach().cpu().numpy()


def evaluate(model, loader, device, loss_fn):
    """
    Evaluate model on a given DataLoader:
      - returns mean_loss, ROC AUC, accuracy, F1.
    """
    model.eval()
    losses, y_true_all, y_prob_all = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            y = to_1d(b.y).float()
            logits = to_1d(model(b)).float()
            loss = loss_fn(logits, y)
            losses.append(loss.item())
            y_true_all.append(y.cpu().numpy())
            y_prob_all.append(probs_from_logits(logits))

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    mean_loss = float(np.mean(losses)) if losses else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob) if y_true.size else float("nan")
    except Exception:
        auc = float("nan")

    y_pred = (y_prob >= 0.5).astype(int) if y_prob.size else np.array([])
    acc = accuracy_score(y_true, y_pred) if y_true.size else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0) if y_true.size else float("nan")
    return mean_loss, auc, acc, f1


def estimate_pos_weight(loader, device):
    """
    Estimate positive class weight for BCEWithLogitsLoss:
      pos_weight = n_neg / n_pos
    """
    n_pos = n_all = 0
    with torch.no_grad():
        for b in loader:
            y = to_1d(b.y.to(device))
            n_pos += int((y > 0.5).sum().item())
            n_all += int(y.numel())
    n_neg = max(n_all - n_pos, 1)
    n_pos = max(n_pos, 1)
    return n_neg / n_pos


def better(curr, best, mode: str) -> bool:
    """Return True if curr is better than best under 'max' or 'min' mode."""
    return best is None or (curr > best if mode == "max" else curr < best)


def filter_pairs_by_ids(pairs, allowed_ids):
    """
    Filter list of (feat, ID) pairs by allowed IDs.

    Args:
        pairs: list of (feat, ID).
        allowed_ids: collection of IDs (list/set) or None.

    Returns:
        Filtered list of (feat, ID); if allowed_ids is None, returns pairs.
    """
    if allowed_ids is None:
        return pairs
    allowed = set(allowed_ids)
    out = [(feat, ID) for (feat, ID) in pairs if ID in allowed]
    return out


def build_pos_dict(positive_pairs):
    """
    Build mapping from ID to positive features.

    Args:
        positive_pairs: list of (feat_pos, ID_pos).

    Returns:
        dict: ID -> feat_pos.
    """
    pos_by_id = {}
    dup = 0
    for feat_pos, ID in positive_pairs:
        if ID in pos_by_id:
            dup += 1
        pos_by_id[ID] = feat_pos
    if dup > 0:
        logging.warning(
            f"[POS] Found {dup} duplicated IDs in positive_pickle_pos_path – using last occurrence"
        )
    logging.info(f"[POS] Unique IDs in positive_pickle_pos_path: {len(pos_by_id)}")
    return pos_by_id


def replace_positive_feats(pairs, pos_by_id, split_name: str):
    """
    Replace features for positive samples (y==1) with positive features from pos_by_id.

    Args:
        pairs: list of (feat, ID).
        pos_by_id: dict ID -> feat_pos.
        split_name: split label for logging ('train'/'val'/'test').

    Returns:
        List of (feat, ID) where positives are replaced when ID is present in pos_by_id.
    """
    if not pos_by_id:
        logging.info(f"[POS-REPLACE][{split_name}] pos_by_id is empty – nothing to replace")
        return pairs

    replaced = 0
    missing = 0
    kept_nonpos = 0
    new_pairs = []

    for feat, ID in pairs:
        y = getattr(feat, "y", None)
        if y is None:
            new_pairs.append((feat, ID))
            continue

        if isinstance(y, torch.Tensor):
            y_val = float(to_1d(y).cpu().numpy().item())
        else:
            y_val = float(y)

        if y_val > 0.5:  # positive sample
            if ID in pos_by_id:
                feat_pos = pos_by_id[ID]
                y_pos = getattr(feat_pos, "y", None)
                if isinstance(y_pos, torch.Tensor):
                    y_pos_val = float(to_1d(y_pos).cpu().numpy().item())
                    if y_pos_val <= 0.5:
                        logging.warning(
                            f"[POS-REPLACE][{split_name}] ID={ID} has y_pos={y_pos_val}, expected ~1"
                        )
                new_pairs.append((feat_pos, ID))
                replaced += 1
            else:
                new_pairs.append((feat, ID))
                missing += 1
        else:
            new_pairs.append((feat, ID))
            kept_nonpos += 1

    logging.info(
        f"[POS-REPLACE][{split_name}] replaced_pos={replaced}, "
        f"missing_pos={missing}, kept_nonpos={kept_nonpos}, total={len(new_pairs)}"
    )
    return new_pairs


def load_pairs_from_dir(split_name: str, split_dir: str):
    """
    Load (feat, ID) pairs for a given split.

    Logic:
      1) If split_dir/<split_name>.p exists (e.g. train.p), load this single file.
      2) Otherwise, look for split_dir/<split_name>_*.p (e.g. train_0.p, train_1.p...)
         and concatenate all shards.

    Args:
        split_name: 'train' / 'val' / 'test'.
        split_dir: directory with pickle shards for this split.

    Returns:
        List of (feat, ID) pairs.
    """
    # 1) single file: train.p / val.p / test.p
    single_path = os.path.join(split_dir, f"{split_name}.p")
    if os.path.exists(single_path):
        logging.info(f"[LOAD] {split_name}: using single file {single_path}")
        with open(single_path, "rb") as f:
            pairs = pickle.load(f)
        logging.info(f"[LOAD] {split_name}: n={len(pairs)} from {single_path}")
        return pairs

    # 2) shards: train_*.p / val_*.p / test_*.p
    pattern = os.path.join(split_dir, f"{split_name}_*.p")
    paths = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(
            f"[LOAD] Neither {single_path} nor shards matching {pattern} were found"
        )

    logging.info(f"[LOAD] {split_name}: found {len(paths)} shard(s)")
    all_pairs = []
    for pth in paths:
        logging.info(f"[LOAD] {split_name}: shard {os.path.basename(pth)}")
        with open(pth, "rb") as f:
            part = pickle.load(f)
        all_pairs.extend(part)
        logging.info(f"[LOAD] {split_name}: +{len(part)} (total {len(all_pairs)})")

    logging.info(f"[LOAD] {split_name}: final total={len(all_pairs)}")
    return all_pairs


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Train MAT/RMAT on pharmaco_explainer pickles")

    ap.add_argument("--model", choices=["mat", "rmat"], required=True,
                    help="Model type to train: MAT or RMAT.")
    ap.add_argument("--k", choices=["k3", "k4", "k5"], required=True,
                    help="Pharmacophore dataset key (k3/k4/k5).")
    ap.add_argument("--subset", default="normal",
                    help="Subset name (e.g. 'normal').")

    ap.add_argument(
        "--difficulty",
        choices=["normal", "easy", "hard", "none"],
        default="normal",
        help=(
            "Data difficulty setting. The underlying train/val/test pickles are "
            "always the same; 'easy'/'hard' only filter IDs using split_easy/split_hard."
        ),
    )

    ap.add_argument(
        "--data-root", "--data_root",
        dest="data_root",
        default="/net/storage/pr3/plgrid/plggsanodrugs/pharmaco_explainer",
        help="Project root directory (the one containing 'pickle_dataloaders' and 'data')."
    )

    ap.add_argument(
        "--data-split-file", "--data_split_file",
        dest="data_split_file",
        default=None,
        help=(
            "Optional split file (parquet/csv) with columns: ID, split_easy, split_hard "
            "(and/or split). Used to filter easy/hard subsets."
        ),
    )

    ap.add_argument(
        "--positive-pickle-pos-path", "--positive_pickle_pos_path",
        dest="positive_pickle_pos_path",
        default=None,
        help=(
            "Optional pickle with positive features: list of (feat_pos, ID). "
            "If set, all samples with y==1 in train/val/test will have features "
            "replaced by the corresponding feat_pos (matched by ID)."
        ),
    )

    ap.add_argument("--ckpt-dir", default=None,
                    help="Optional directory to store checkpoints. Default: auto inside ./checkpoints.")
    ap.add_argument("--checkpoint-path", default=None,
                    help="Path to checkpoint file. If provided, training will save to this file.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-train", type=int, default=32)
    ap.add_argument("--batch-eval", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3,
                    help="Early stopping patience (in validation checks).")
    ap.add_argument(
        "--val-every-batches",
        type=int,
        default=5000,
        help="Validate every N training batches (0 or negative = only at epoch end).",
    )
    ap.add_argument(
        "--selection-metric",
        choices=["val_auc", "val_loss"],
        default="val_auc",
        help="Metric used to select and save the best checkpoint.",
    )

    args = ap.parse_args()

    diff = args.difficulty
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Args: model={args.model}, k={args.k}, subset={args.subset}, difficulty={diff}")
    logging.info(f"data_root={args.data_root}")
    logging.info(f"data_split_file={args.data_split_file}")
    logging.info(f"positive_pickle_pos_path={args.positive_pickle_pos_path}")

    # ---------------- Model / featurizer ----------------
    if args.model == "rmat":
        model = RMatModel.from_pretrained("rmat_4M")
        featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    else:
        model = MatModel.from_pretrained("mat_masking_20M")
        featurizer = MatFeaturizer.from_pretrained("mat_masking_20M")
    model.to(device)

    # ---------------- Data paths ----------------
    k_dir = args.k
    k_int = int(args.k[1])

    parquet_path = os.path.join(args.data_root, "data", k_dir, f"k{k_int}.parquet")
    logging.info(f"Parquet (info only): {parquet_path}")

    pickle_root = os.path.join(
        args.data_root,
        "pickle_dataloaders",
        args.model,   # 'mat' or 'rmat'
        args.k,       # 'k3' / 'k4' / 'k5'
        args.subset   # 'normal', etc.
    )

    train_dir = os.path.join(pickle_root, "train")
    val_dir   = os.path.join(pickle_root, "val")
    test_dir  = os.path.join(pickle_root, "test")

    logging.info(f"[PATHS] train_dir={train_dir}")
    logging.info(f"[PATHS] val_dir  ={val_dir}")
    logging.info(f"[PATHS] test_dir ={test_dir}")

    # ---------------- Load pickles (single file OR shards) ----------------
    # Assumption: each item is (feat, ID)
    train_pairs = load_pairs_from_dir("train", train_dir)
    val_pairs   = load_pairs_from_dir("val",   val_dir)
    test_pairs  = load_pairs_from_dir("test",  test_dir)

    logging.info(f"len(train_pairs)={len(train_pairs)}")
    logging.info(f"len(val_pairs)  ={len(val_pairs)}")
    logging.info(f"len(test_pairs) ={len(test_pairs)}")

    # ---------------- Filtering by data_split_file (easy/hard) ----------------
    if diff in ("easy", "hard"):
        if args.data_split_file is None:
            k_dir = args.k
            k_int = int(args.k[1])
            auto_split = os.path.join(
                args.data_root,
                "data",
                k_dir,
                f"ks{k_int}.parquet"
            )
            logging.info(
                f"[SPLIT] difficulty={diff}, no --data-split-file provided – "
                f"trying auto path for K={args.k}: {auto_split}"
            )
            if not os.path.exists(auto_split):
                raise FileNotFoundError(
                    f"[SPLIT] Auto split file for K={args.k} not found: {auto_split}"
                )
            args.data_split_file = auto_split
        else:
            logging.info(f"[SPLIT] Using provided split file: {args.data_split_file}")

        if args.data_split_file.endswith(".parquet"):
            df_split = pd.read_parquet(args.data_split_file)
        else:
            df_split = pd.read_csv(args.data_split_file)

        if "ID" in df_split.columns:
            id_col = "ID"
        elif "id" in df_split.columns:
            id_col = "id"
        else:
            raise KeyError(f"ID / id column not found in {args.data_split_file}")

        split_col = f"split_{diff}"  # split_easy or split_hard
        if split_col not in df_split.columns:
            raise KeyError(f"Column {split_col} not found in {args.data_split_file}")

        train_ids = df_split.loc[df_split[split_col] == "train", id_col].tolist()
        val_ids   = df_split.loc[df_split[split_col] == "val",   id_col].tolist()
        test_ids  = df_split.loc[df_split[split_col] == "test",  id_col].tolist()

        logging.info(
            f"[SPLIT] n_train_ids={len(train_ids)}, "
            f"n_val_ids={len(val_ids)}, n_test_ids={len(test_ids)}"
        )

        train_pairs = filter_pairs_by_ids(train_pairs, train_ids)
        val_pairs   = filter_pairs_by_ids(val_pairs,   val_ids)
        test_pairs  = filter_pairs_by_ids(test_pairs,  test_ids)

        logging.info(
            f"[FILTERED PAIRS] n_train={len(train_pairs)}, "
            f"n_val={len(val_pairs)}, n_test={len(test_pairs)}"
        )
    else:
        logging.info("[SPLIT] data_split_file is not used (difficulty=normal/none)")

    # ---------------- Positive feature replacement ----------------
    pos_by_id = {}
    if args.positive_pickle_pos_path is not None:
        if not os.path.exists(args.positive_pickle_pos_path):
            raise FileNotFoundError(
                f"positive_pickle_pos_path does not exist: {args.positive_pickle_pos_path}"
            )
        logging.info(f"[POS] Loading positive_pickle_pos_path: {args.positive_pickle_pos_path}")
        with open(args.positive_pickle_pos_path, "rb") as f:
            positive_pairs = pickle.load(f)
        logging.info(f"[POS] Number of records in positive_pickle_pos_path: {len(positive_pairs)}")
        pos_by_id = build_pos_dict(positive_pairs)

        train_pairs = replace_positive_feats(train_pairs, pos_by_id, "train")
        val_pairs   = replace_positive_feats(val_pairs,   pos_by_id, "val")
        test_pairs  = replace_positive_feats(test_pairs,  pos_by_id, "test")
    else:
        logging.info("[POS] positive_pickle_pos_path is None – not replacing positives")

    # ---------------- Extract features from (feat, ID) ----------------
    train = [feat for (feat, ID) in train_pairs]
    val   = [feat for (feat, ID) in val_pairs]
    test  = [feat for (feat, ID) in test_pairs]

    logging.info(f"[SIZES] final n_train={len(train)}, n_val={len(val)}, n_test={len(test)}")

    # ---------------- DataLoaders ----------------
    train_loader = featurizer.get_data_loader(train, batch_size=args.batch_train, shuffle=True)
    val_loader   = featurizer.get_data_loader(val,   batch_size=args.batch_eval,  shuffle=False)
    test_loader  = featurizer.get_data_loader(test,  batch_size=args.batch_eval,  shuffle=False)

    pos_w = estimate_pos_weight(train_loader, device)
    logging.info(f"Estimated pos_weight={pos_w:.4f}")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------- Checkpoint handling ----------------
    if args.checkpoint_path is not None:
        best_ckpt = args.checkpoint_path
        ckpt_dir = os.path.dirname(best_ckpt) or "."
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        run_tag = f"{args.model}_{args.k}_{args.subset}_{diff}"
        if args.positive_pickle_pos_path is not None:
            run_tag = f"{run_tag}_with_pos"
        else:
            run_tag = f"{run_tag}_normal"

        ckpt_dir = args.ckpt_dir or os.path.join("checkpoints", "pharmaco", run_tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = os.path.join(ckpt_dir, "best_model.pth")

    logging.info(f"Checkpoint path: {best_ckpt}")

    mode = "max" if args.selection_metric == "val_auc" else "min"
    best_val = None
    no_improve = 0
    global_step = 0

    # ---------------- Training loop ----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        batches = 0

        for b in train_loader:
            b = b.to(device)
            y = to_1d(b.y).float()
            logits = to_1d(model(b)).float()
            loss = loss_fn(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            run_loss += loss.item()
            batches += 1
            global_step += 1

            if args.val_every_batches > 0 and (global_step % args.val_every_batches == 0):
                vloss, vauc, _, _ = evaluate(model, val_loader, device, loss_fn)
                score = vauc if args.selection_metric == "val_auc" else vloss
                logging.info(
                    f"[MID] step={global_step} | val_auc={vauc:.5f} | val_loss={vloss:.5f}"
                )
                if better(score, best_val, mode):
                    best_val = score
                    torch.save(model.state_dict(), best_ckpt)
                    no_improve = 0
                    logging.info(f"✔ new best mid-run ({args.selection_metric}={best_val:.5f})")
                else:
                    no_improve += 1
                    if args.patience and no_improve >= args.patience:
                        logging.info("Early stop (mid-run)")
                        break

        if args.patience and no_improve >= args.patience:
            logging.info("Early stop triggered after epoch due to no improvement")
            break

        tloss = run_loss / max(1, batches)
        vloss, vauc, _, _ = evaluate(model, val_loader, device, loss_fn)
        score = vauc if args.selection_metric == "val_auc" else vloss
        logging.info(
            f"[EPOCH {epoch:03d}] train_loss={tloss:.5f} | val_loss={vloss:.5f} | val_auc={vauc:.5f}"
        )

        if better(score, best_val, mode):
            best_val = score
            torch.save(model.state_dict(), best_ckpt)
            no_improve = 0
            logging.info(f"✔ new best at epoch-end ({args.selection_metric}={best_val:.5f})")
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                logging.info("Early stop (epoch-end)")
                break

    logging.info(f"Training finished | best {args.selection_metric}={best_val}")

    # ---------------- Test on best checkpoint ----------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_auc, test_acc, test_f1 = evaluate(model, test_loader, device, loss_fn)
    logging.info(
        f"[TEST] loss={test_loss:.5f} | auc={test_auc:.5f} | "
        f"acc={test_acc:.4f} | f1={test_f1:.4f}"
    )
    # No result pickling here (by design).


if __name__ == "__main__":
    main()
