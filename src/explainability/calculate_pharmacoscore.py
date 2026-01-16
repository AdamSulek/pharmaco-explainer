import pandas as pd
import numpy as np
import os
import ast
import argparse
from sklearn.metrics import roc_auc_score

def safe_parse(x):
    if isinstance(x, (list, np.ndarray)):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return None
    return None

def calculate_pharmacoscore(project_root, dataset, model_type, split, aggregate):
    aggregate_filename = f"{model_type}_{split}_aggregate_{aggregate}.parquet"
    aggregate_file = os.path.join(project_root, "results", "shap", dataset, aggregate_filename)
    labels_file = os.path.join(project_root, "data", dataset, f"{dataset}_labels.parquet")

    if not os.path.exists(aggregate_file):
        print(f"[ERROR] File not found: {aggregate_file}")
        return
    if not os.path.exists(labels_file):
        print(f"[ERROR] File not found: {labels_file}")
        return

    df_agg = pd.read_parquet(aggregate_file)
    df_lab = pd.read_parquet(labels_file)

    df = pd.merge(df_agg, df_lab[['ID', 'y_true']], on='ID', how='inner')
    scores = []
    
    for idx, row in df.iterrows():
        y_pred = safe_parse(row['atom_importances'])
        y_true = safe_parse(row['y_true'])

        if y_pred is None or y_true is None:
            continue
            
        L = min(len(y_true), len(y_pred))
        y_true_sub = y_true[:L]
        y_pred_sub = y_pred[:L]

        if len(set(y_true_sub)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true_sub, y_pred_sub)
            scores.append(auc)
        except Exception:
            continue

    if scores:
        final_mean = np.mean(scores)
        print(f"\n{'='*50}")
        print(f" FINAL PHARMACOSCORE REPORT")
        print(f"{'='*50}")
        print(f"Dataset:    {dataset}")
        print(f"Model:      {model_type} ({split})")
        print(f"Aggregate:  {aggregate}")
        print(f"Valid IDs:  {len(scores)}")
        print(f"Mean ROC:   {final_mean:.4f}")
        print(f"{'='*50}\n")
    else:
        print(f"Cannot calculate for {model_type} / {dataset}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PharmacoScore for ML models.")
    parser.add_argument("--project_root", default=os.environ.get("PHARM_PROJECT_ROOT"), help="Root path of the project")
    parser.add_argument("--dataset", default="k3", help="Dataset name (e.g., k3, k5)")
    parser.add_argument("--model", default="xgb", help="Model type (rf, xgb, mlp, mlp_vg)")
    parser.add_argument("--split", default="split_distant_set", help="Split type")
    parser.add_argument("--aggregate", default="max", help="Aggregation method (max, mean, sum)")

    args = parser.parse_args()
    
    if not args.project_root:
        print("[ERROR] PHARM_PROJECT_ROOT not set in environment or as argument.")
    else:
        calculate_pharmacoscore(args.project_root, args.dataset, args.model, args.split, args.aggregate)