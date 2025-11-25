import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix
import joblib
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

PARAM_GRID = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def train_and_evaluate(df, split_name, checkpoint_dir):
    best_model, best_roc_auc, best_params = None, 0, None
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train = np.stack(train_df["X_ecfp_2"].values)
    y_train = train_df["y"].values
    X_val = np.stack(val_df["X_ecfp_2"].values)
    y_val = val_df["y"].values

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))
        model = RandomForestClassifier(n_jobs=32, **params)
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_proba)
        if roc_auc > best_roc_auc:
            best_roc_auc, best_model, best_params = roc_auc, model, params
            model_path = os.path.join(checkpoint_dir, f"best_model_rf_{split_name}.joblib")
            joblib.dump(best_model, model_path)
            logging.info(f"New best model! ROC-AUC={roc_auc:.4f}, Params={params}")
            logging.info(f"Saved to: {model_path}")

    X_test = np.stack(test_df["X_ecfp_2"].values)
    y_test = test_df["y"].values
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    logging.info(f"TEST ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    logging.info(f"TEST PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
    logging.info(f"Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"F1 Score:     {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Best Params: {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["k3","k4","k5"])
    parser.add_argument("--split", required=True, choices=["easy","hard","all"])
    args = parser.parse_args()

    input_dir = f"../../../data/{args.dataset}/processed"
    checkpoint_dir = f"../../../results/checkpoints/{args.dataset}/"
    seed_everything(123)

    files = sorted(glob(os.path.join(input_dir, "final_dataset_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No dataset parts found in: {input_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if args.split == "easy":
        df["split"] = df["split_easy"]
    elif args.split == "hard":
        df["split"] = df["split_hard"]

    logging.info(f"Loaded {len(df)} rows from dataset {args.dataset}")
    train_and_evaluate(df, split_name=args.split, checkpoint_dir=checkpoint_dir)
