import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from itertools import product
from glob import glob
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

PARAM_GRID = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 8],
    'n_estimators': [100, 300],
    'colsample_bytree': [0.7, 0.9],
}

PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def log_split_stats(train_df, val_df, test_df):
    def stats(df, name):
        return (name,
                len(df),
                (df['y'] == 1).sum(),
                (df['y'] == 0).sum())
    for name, total, pos, neg in [
        stats(train_df, "TRAIN"),
        stats(val_df,   "VAL"),
        stats(test_df,  "TEST")
    ]:
        logging.info(f"{name}: {total} rows | Positives: {pos}, Negatives: {neg}")

def train_and_evaluate(df, checkpoint_dir, split_name, tree_method="hist"):
    best_model, best_roc_auc, best_params = None, 0, None

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    logging.info("=== DATASET SPLIT SUMMARY ===")
    log_split_stats(train_df, val_df, test_df)

    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train, y_train = np.stack(train_df["X_ecfp_2"].values), train_df['y'].values
    X_val, y_val     = np.stack(val_df["X_ecfp_2"].values), val_df['y'].values

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))
        model = xgb.XGBClassifier(
            tree_method=tree_method,
            eval_metric="logloss",
            n_jobs=8,
            use_label_encoder=False,
            **params
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_proba)

        if roc_auc > best_roc_auc:
            best_roc_auc, best_model, best_params = roc_auc, model, params
            model_path = os.path.join(checkpoint_dir, f"best_model_xgb_{split_name}.joblib")
            joblib.dump(best_model, model_path)
            logging.info(f"New BEST model saved! ROC-AUC: {roc_auc:.4f}, Params: {params}")

    X_test, y_test = np.stack(test_df["X_ecfp_2"].values), test_df['y'].values
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_test_proba >= 0.5).astype(int)

    logging.info("=== FINAL TEST METRICS ===")
    logging.info(f"ROC-AUC:     {roc_auc_score(y_test, y_test_proba):.4f}")
    logging.info(f"PR-AUC:      {average_precision_score(y_test, y_test_proba):.4f}")
    logging.info(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"F1-score:    {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Precision:   {precision_score(y_test, y_pred):.4f}")
    logging.info(f"Recall:      {recall_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Best Params: {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--split", default="hard")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    PROJECT_ROOT = os.environ.get("PHARM_PROJECT_ROOT")
    if PROJECT_ROOT is None:
        raise EnvironmentError("Please set the PHARM_PROJECT_ROOT environment variable!")

    seed_everything(123)

    input_dir = os.path.join(PROJECT_ROOT, "data", args.dataset)
    checkpoint_dir = os.path.join(PROJECT_ROOT, "results", "checkpoints", args.dataset)

    df = pd.read_parquet(os.path.join(input_dir, f"{args.dataset}_split.parquet"))

    if args.split == "split_distant_set":
        df["split"] = df["split_distant_set"]
    elif args.split == "split_close_set":
        df["split"] = df["split_close_set"]

    logging.info(f"Training XGBoost on {args.split} split, total rows: {len(df)}")
    train_and_evaluate(df, checkpoint_dir, split_name=args.split, tree_method="hist")
