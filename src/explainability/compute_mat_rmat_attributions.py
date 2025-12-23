#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_featurizer(model_type: str):
    if model_type == "rmat":
        from huggingmolecules import RMatFeaturizer as Featurizer
        from huggingmolecules import RMatModel as Model

        model = Model.from_pretrained("rmat_4M")
        featurizer = Featurizer.from_pretrained("rmat_4M")
        return model, featurizer

    if model_type == "mat":
        from huggingmolecules import MatFeaturizer as Featurizer
        from huggingmolecules import MatModel as Model

        model = Model.from_pretrained("mat_masking_20M")
        featurizer = Featurizer.from_pretrained("mat_masking_20M")
        return model, featurizer

    raise ValueError("model_type must be one of: rmat, mat")


def load_checkpoint_into_model(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    meta: Dict[str, Any] = {
        "epoch": "?",
        "global_step": "?",
        "selection_metric": "?",
        "best_metric_value": "?",
    }

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta["epoch"] = ckpt.get("epoch", "?")
        meta["global_step"] = ckpt.get("global_step", "?")
        meta["selection_metric"] = ckpt.get("selection_metric", "?")
        meta["best_metric_value"] = ckpt.get("best_metric_value", "?")
        LOGGER.info("Loaded checkpoint dict with keys: %s", list(ckpt.keys()))
        return meta

    model.load_state_dict(ckpt)
    LOGGER.info("Loaded raw state_dict checkpoint (no metadata).")
    return meta


class GradCAMHook:
    def __init__(self, model: torch.nn.Module):
        self.activations: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

        target_layer = None
        if hasattr(model, "encoder"):
            target_layer = getattr(model, "encoder")
        elif hasattr(model, "module") and hasattr(model.module, "encoder"):
            target_layer = getattr(model.module, "encoder")

        if target_layer is None:
            raise RuntimeError("Encoder layer not found (expected model.encoder or model.module.encoder).")

        def fwd_hook(_module, _inp, out):
            self.activations["value"] = out.detach()

        def bwd_hook(_module, _grad_input, grad_output):
            self.grads["value"] = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_backward_hook(bwd_hook))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _require_attr(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise AttributeError(f"Missing required attribute: {name}")
    return getattr(obj, name)


def _normalize_1d(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    m = x.max()
    return x / (m + eps)


def compute_vg_for_batch(model: torch.nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    batch = batch.to(device)
    node_features = _require_attr(batch, "node_features")
    batch.node_features = node_features.detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(batch).squeeze()
    logits.backward()

    grads = batch.node_features.grad
    if grads is None:
        raise RuntimeError("Gradient w.r.t. batch.node_features is None.")

    grads = grads.squeeze(0)
    sal = grads.abs().sum(dim=1)
    return _normalize_1d(sal).detach().cpu()


def compute_gradcam_for_batch(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    hook: GradCAMHook,
) -> torch.Tensor:
    batch = batch.to(device)
    node_features = _require_attr(batch, "node_features")
    batch.node_features = node_features.detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(batch).squeeze()
    logits.backward()

    if "value" not in hook.activations or "value" not in hook.grads:
        raise RuntimeError("Grad-CAM hook did not capture activations/gradients.")

    A = hook.activations["value"].squeeze(0)
    G = hook.grads["value"].squeeze(0)

    weights = G.mean(dim=0)
    cam = torch.relu((A * weights).sum(dim=1))
    return _normalize_1d(cam).detach().cpu()


def load_pickle(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def maybe_replace_positive_features(
    test_pairs: List[Tuple[Any, Any]],
    positive_pickle_pos_path: Optional[str],
) -> List[Tuple[Any, Any]]:
    if positive_pickle_pos_path is None:
        return test_pairs

    pos_list = load_pickle(positive_pickle_pos_path)
    pos_map: Dict[Any, Any] = {}
    for feat_pos, mol_id in pos_list:
        pos_map[mol_id] = feat_pos

    replaced = 0
    total_pos = 0
    out: List[Tuple[Any, Any]] = []

    for data_obj, mol_id in test_pairs:
        y_val = int(_require_attr(data_obj, "y"))
        if y_val == 1:
            total_pos += 1
            if mol_id in pos_map:
                feat_pos = pos_map[mol_id]
                try:
                    feat_pos.y = data_obj.y
                except Exception:
                    pass
                out.append((feat_pos, mol_id))
                replaced += 1
            else:
                out.append((data_obj, mol_id))
        else:
            out.append((data_obj, mol_id))

    LOGGER.info("Positive features map size: %d", len(pos_map))
    LOGGER.info("Test positives: %d | replaced: %d", total_pos, replaced)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-atom attributions (VG/Grad-CAM) for MAT/RMAT.")
    parser.add_argument("--method", choices=["vg", "gradcam"], required=True)
    parser.add_argument("--model", choices=["rmat", "mat"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-pickle", required=True, dest="test_pickle")
    parser.add_argument("--output-file", required=True, dest="output_file")
    parser.add_argument("--positive-pickle-pos-path", type=str, default=None)
    parser.add_argument("--only-positive", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    device = get_device()
    LOGGER.info("Device: %s", device)

    test_data_full = load_pickle(args.test_pickle)
    if not isinstance(test_data_full, list):
        raise ValueError("Expected test pickle to be a list of (data_obj, ID).")

    LOGGER.info("Loaded test pairs: %d", len(test_data_full))
    test_data_full = maybe_replace_positive_features(test_data_full, args.positive_pickle_pos_path)

    y_all = [int(_require_attr(pair[0], "y")) for pair in test_data_full]
    ids_all = [pair[1] for pair in test_data_full]
    data_all = [pair[0] for pair in test_data_full]

    if args.only_positive:
        idxs = [i for i, y in enumerate(y_all) if y == 1]
        ids = [ids_all[i] for i in idxs]
        data = [data_all[i] for i in idxs]
        LOGGER.info("Selected positives: %d / %d", len(data), len(data_all))
    else:
        ids = ids_all
        data = data_all
        LOGGER.info("Selected all test samples: %d (positives=%d)", len(data), sum(y_all))

    model, featurizer = load_model_and_featurizer(args.model)
    meta = load_checkpoint_into_model(args.checkpoint, model, device)
    model.to(device)
    model.eval()

    LOGGER.info(
        "Checkpoint meta: epoch=%s step=%s %s=%s",
        meta.get("epoch", "?"),
        meta.get("global_step", "?"),
        meta.get("selection_metric", "?"),
        meta.get("best_metric_value", "?"),
    )

    loader = featurizer.get_data_loader(data, batch_size=1, shuffle=False)

    hook = GradCAMHook(model) if args.method == "gradcam" else None
    key_name = f"{args.method}_pred_proba"

    rows: List[Dict[str, Any]] = []
    for i, batch in enumerate(loader):
        mol_id = ids[i]

        if args.method == "vg":
            vec = compute_vg_for_batch(model, batch, device)
        else:
            assert hook is not None
            vec = compute_gradcam_for_batch(model, batch, device, hook)

        rows.append({"ID": mol_id, key_name: vec.numpy().tolist()})

        if args.log_every > 0 and (i % args.log_every == 0):
            LOGGER.info("Processed %d / %d", i, len(ids))

    if hook is not None:
        hook.close()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_path, index=False)
    LOGGER.info("Saved rows: %d -> %s", len(df_out), out_path)


if __name__ == "__main__":
    main()
