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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def unpack_ecfp(fp, n_bits=1024):
    if isinstance(fp, (bytes, bytearray)):
        return np.unpackbits(
            np.frombuffer(fp, dtype=np.uint8)
        )[:n_bits].astype(np.uint8)
    return np.asarray(fp, dtype=np.uint8)

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}

PARAM_COMBINATIONS = list(product(*PARAM_GRID.values()))
PARAM_KEYS = list(PARAM_GRID.keys())

def make_X(df):
    return np.stack(df["X_ecfp_2"].map(unpack_ecfp).values)

def train_and_evaluate(df, split_name, checkpoint_dir):
    best_model, best_roc_auc, best_params = None, 0, None

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    X_train = make_X(train_df)
    y_train = train_df["y"].values
    X_val = make_X(val_df)
    y_val = val_df["y"].values

    for param_values in PARAM_COMBINATIONS:
        params = dict(zip(PARAM_KEYS, param_values))
        model = RandomForestClassifier(n_jobs=32, **params)
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_val_proba)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_params = params
            model_path = os.path.join(
                checkpoint_dir, f"best_model_rf_{split_name}.joblib"
            )
            joblib.dump(best_model, model_path)
            logging.info(f"New best model! ROC-AUC={roc_auc:.4f}, Params={params}")
            logging.info(f"Saved to: {model_path}")

    X_test = make_X(test_df)
    y_test = test_df["y"].values
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    logging.info("==== FINAL TEST METRICS ====")
    logging.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    logging.info(f"PR-AUC:  {average_precision_score(y_test, y_proba):.4f}")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Best Params: {best_params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k3")
    parser.add_argument("--split", default="easy")
    args = parser.parse_args()

    PROJECT_ROOT = os.environ.get("PHARM_PROJECT_ROOT")
    if PROJECT_ROOT is None:
        raise EnvironmentError("Please set the PHARM_PROJECT_ROOT environment variable!")

    input_dir = os.path.join(PROJECT_ROOT, "data", args.dataset)
    checkpoint_dir = os.path.join(
        PROJECT_ROOT, "results", "checkpoints", args.dataset
    )

    seed_everything(123)

    df = pd.read_parquet(
        os.path.join(input_dir, f"{args.dataset}_split.parquet")
    )

    if args.split == "split_distant_set":
        df["split"] = df["split_distant_set"]
    elif args.split == "split_close_set":
        df["split"] = df["split_close_set"]

    logging.info(f"Loaded {len(df)} rows from dataset {args.dataset}")
    train_and_evaluate(df, split_name=args.split, checkpoint_dir=checkpoint_dir)
