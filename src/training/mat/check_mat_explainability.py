import argparse
import logging
import pickle
import torch
import pandas as pd

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Model / Featurizer loaders ---
def load_model_and_featurizer(model_type: str):
    """Load a pretrained MAT/RMAT model and its corresponding featurizer."""
    if model_type == "rmat":
        from huggingmolecules import RMatModel as Model, RMatFeaturizer as Featurizer
        model = Model.from_pretrained("rmat_4M")
        featurizer = Featurizer.from_pretrained("rmat_4M")
        return model, featurizer
    elif model_type == "mat":
        from huggingmolecules import MatModel as Model, MatFeaturizer as Featurizer
        model = Model.from_pretrained("mat_masking_20M")
        featurizer = Featurizer.from_pretrained("mat_masking_20M")
        return model, featurizer
    else:
        raise ValueError("model_type must be 'rmat' or 'mat'.")


# --- Safe checkpoint loading (two common formats) ---
def load_checkpoint_into_model(path: str, model: torch.nn.Module, device: torch.device):
    """
    Load checkpoint into a model and return metadata.

    Supported formats:
      1) dict with key 'model_state_dict' (+ optional metadata),
      2) raw state_dict.
    """
    ckpt = torch.load(path, map_location=device)
    meta = {"epoch": "?", "global_step": "?", "selection_metric": "?", "best_metric_value": "?"}

    # format 1: dict with "model_state_dict"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta.update(
            {
                "epoch": ckpt.get("epoch", "?"),
                "global_step": ckpt.get("global_step", "?"),
                "selection_metric": ckpt.get("selection_metric", "?"),
                "best_metric_value": ckpt.get("best_metric_value", "?"),
            }
        )
        logging.info(f"Loaded dict checkpoint with keys: {list(ckpt.keys())}")
    # format 2: bare state_dict
    else:
        model.load_state_dict(ckpt)
        logging.info("Loaded raw state_dict checkpoint (no metadata).")

    return meta


# --- Hooks for Grad-CAM ---
class GradCAMHook:
    """
    Utility class that attaches forward/backward hooks to the last feed-forward layer
    to collect activations and gradients for Grad-CAM.
    """

    def __init__(self, model, model_type: str):
        self.activations = {}
        self.grads = {}
        self.hooks = []

        # Try several possible attribute paths for the last feed-forward layer
        target_layer = None
        for attr in [
            "encoder.layers.-1.feed_forward",
            "encoder.layers.-1.ffn",
            "encoder.layers.-1.mlp",
            "layers.-1.feed_forward",
            "layers.-1.ffn",
        ]:
            try:
                target_layer = _resolve_attr(model, attr)
                if target_layer is not None:
                    break
            except Exception:
                continue

        if target_layer is None:
            raise RuntimeError("Could not find the last feed-forward layer for Grad-CAM.")

        def fwd_hook(module, inp, out):
            self.activations["value"] = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.grads["value"] = grad_output[0].detach()

        self.hooks.append(target_layer.register_forward_hook(fwd_hook))
        self.hooks.append(target_layer.register_backward_hook(bwd_hook))

    def close(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()


def _resolve_attr(obj, dotted: str):
    """
    Resolve a dotted attribute path, supporting list-like indexing by integer tokens.
    Example: 'encoder.layers.-1.feed_forward'
    """
    parts = dotted.split(".")
    cur = obj
    for p in parts:
        if p.isdigit() or (p.startswith("-") and p[1:].isdigit()):
            idx = int(p)
            cur = cur[idx]
        else:
            cur = getattr(cur, p)
    return cur


# --- Core computations (VG / Grad-CAM) ---
def compute_vg_for_batch(model, batch):
    """
    Compute Vanilla Gradient (VG) attribution per atom for a single-batch graph.

    - Gradients are taken w.r.t. node_features.
    - Aggregated by abs-sum across feature dimension.
    - Normalized to [0, 1].
    """
    batch = batch.to(next(model.parameters()).device)
    batch.node_features = batch.node_features.detach().requires_grad_()
    model.zero_grad()
    logits = model(batch).squeeze()
    logits.backward()
    grads = batch.node_features.grad.squeeze(0)  # [N_atoms, D]
    sal = grads.abs().sum(dim=1)                # [N_atoms]
    sal = sal / (sal.max() + 1e-12)
    return sal.detach().cpu()


def compute_gradcam_for_batch(model, batch, hook: GradCAMHook):
    """
    Compute Grad-CAM attribution per atom using activations and gradients
    from the hooked layer.

    - Weights are obtained by averaging gradients over feature dimension.
    - CAM = ReLU(weighted sum over channels).
    - Normalized to [0, 1].
    """
    batch = batch.to(next(model.parameters()).device)
    batch.node_features = batch.node_features.detach().requires_grad_()
    model.zero_grad()
    logits = model(batch).squeeze()
    logits.backward()
    A = hook.activations["value"].squeeze(0)  # [N_atoms, D]
    G = hook.grads["value"].squeeze(0)        # [N_atoms, D]
    weights = G.mean(dim=0)                   # [D]
    cam = torch.relu((A * weights).sum(dim=1))  # [N_atoms]
    cam = cam / (cam.max() + 1e-12)
    return cam.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="Run CAM (VG) or Grad-CAM on RMAT/MAT models.")
    parser.add_argument(
        "--method",
        choices=["vg", "gradcam"],
        required=True,
        help="Attribution method: 'vg' (Vanilla Gradient) or 'gradcam'.",
    )
    parser.add_argument(
        "--model",
        choices=["rmat", "mat"],
        required=True,
        help="Model type: 'rmat' or 'mat'.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth checkpoint (state_dict OR dict with 'model_state_dict').",
    )
    parser.add_argument(
        "--test-pickle",
        required=True,
        help="Path to test data pickle produced by the Featurizer "
             "(LIST of (data_obj, ID) in model order).",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output parquet path with per-atom attribution vectors.",
    )
    parser.add_argument(
        "--positive-pickle-pos-path",
        type=str,
        default=None,
        help=(
            "Optional pickle with positive features: list of (feat_pos, ID). "
            "If set, test features for y==1 and matching ID will be replaced by feat_pos."
        ),
    )
    parser.add_argument(
        "--only-positive",
        action="store_true",
        default=True,
        help=(
            "If set (default), run CAM only on positive samples (y==1). "
            "Otherwise, use all test samples."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load test list in model order ---
    with open(args.test_pickle, "rb") as f:
        test_data_full = pickle.load(f)
        # Expected format: list of (data_obj, ID)

    logging.info(f"Loaded test_data_full: {len(test_data_full)} examples.")

    # --- Optional: replace features for positives using a positive-feature pickle ---
    if args.positive_pickle_pos_path is not None:
        logging.info(f"Loading positive features from: {args.positive_pickle_pos_path}")
        with open(args.positive_pickle_pos_path, "rb") as f_pos:
            pos_list = pickle.load(f_pos)
            # pos_list is expected to be list of (feat_pos, ID) as in training

        pos_map = {}
        for feat_pos, mol_id in pos_list:
            pos_map[mol_id] = feat_pos
        logging.info(f"Positive feature map contains {len(pos_map)} IDs.")

        replaced = 0
        total_pos = 0
        new_test_data_full = []

        for data_obj, mol_id in test_data_full:
            y_val = int(data_obj.y)
            if y_val == 1:
                total_pos += 1
                if mol_id in pos_map:
                    # Replace features with feat_pos; keep the original label y
                    feat_pos = pos_map[mol_id]
                    try:
                        feat_pos.y = data_obj.y
                    except Exception:
                        pass
                    new_test_data_full.append((feat_pos, mol_id))
                    replaced += 1
                else:
                    new_test_data_full.append((data_obj, mol_id))
            else:
                new_test_data_full.append((data_obj, mol_id))

        test_data_full = new_test_data_full
        logging.info(
            f"Positive samples in test: {total_pos}, "
            f"features replaced for {replaced} IDs (from POS pickle)."
        )

    # --- Split into lists: y, ID, data objects ---
    y_list = [int(e[0].y) for e in test_data_full]
    id_list_all = [e[1] for e in test_data_full]
    data_list_all = [e[0] for e in test_data_full]

    # --- Optional restriction to positive samples only ---
    if args.only_positive:
        positive_idx = [i for i, y in enumerate(y_list) if y == 1]
        id_list = [id_list_all[i] for i in positive_idx]
        data_list = [data_list_all[i] for i in positive_idx]
        logging.info(
            f"Found {len(data_list)} positive samples out of {len(data_list_all)} total. "
            f"Running CAM only on y==1 because --only-positive is enabled."
        )
    else:
        id_list = id_list_all
        data_list = data_list_all
        logging.info(
            f"Running CAM on the entire test set: {len(data_list)} samples "
            f"(including {sum(y_list)} positives)."
        )

    # --- Model and checkpoint ---
    model, featurizer = load_model_and_featurizer(args.model)
    meta = load_checkpoint_into_model(args.checkpoint, model, device)
    model.to(device)
    model.eval()
    logging.info(
        f"Checkpoint loaded: epoch={meta.get('epoch','?')} | step={meta.get('global_step','?')} | "
        f"{meta.get('selection_metric','?')}={meta.get('best_metric_value','?')}"
    )

    test_loader = featurizer.get_data_loader(data_list, batch_size=1, shuffle=False)
    hook = GradCAMHook(model, args.model) if args.method == "gradcam" else None

    rows = []
    key_name = f"{args.method}_pred_proba"  # per-atom attribution vector (VG/GradCAM)

    for i, batch in enumerate(test_loader):
        mol_id = id_list[i]  # index in loader corresponds to index in id_list

        if args.method == "vg":
            vec = compute_vg_for_batch(model, batch)
        else:
            vec = compute_gradcam_for_batch(model, batch, hook)

        vec_np = vec.detach().cpu().numpy()
        rows.append({"ID": mol_id, key_name: vec_np.tolist()})

        if i % 200 == 0:
            logging.info(f"[idx {i} | id={mol_id}] processed")

    if hook is not None:
        hook.close()

    df_out = pd.DataFrame(rows)  # columns: ID, key_name
    df_out.to_parquet(args.output_file, index=False)
    logging.info(f"Saved {len(df_out)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
