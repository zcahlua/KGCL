"""Validation for KGCL configuration values."""
from __future__ import annotations

from typing import Mapping, Any
from .device import is_valid_device_request


def validate_config(values: Mapping[str, Any]) -> None:
    errors = []
    if str(values.get("dataset", "")).strip() == "":
        errors.append("dataset must be a non-empty string")
    if values.get("mode") not in {"train", "valid", "test"}:
        errors.append("mode must be one of: train, valid, test")
    for name in ("epochs", "n_heads", "mpn_size", "depth", "mlp_size", "patience", "max_clip", "max_steps", "num_workers"):
        if int(values.get(name, 0)) < (0 if name == "num_workers" else 1):
            errors.append(f"{name} must be {'non-negative' if name == 'num_workers' else 'at least 1'}")
    for name in ("dropout_mpn", "dropout_mlp"):
        value = float(values.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            errors.append(f"{name} must be between 0 and 1")
    if values.get("lr") is not None and float(values.get("lr", 0.0)) <= 0:
        errors.append("lr must be positive")
    if not 0.0 < float(values.get("factor", 0.0)) <= 1.0:
        errors.append("factor must be greater than 0 and at most 1")
    if float(values.get("thresh", 0.0)) < 0:
        errors.append("thresh must be non-negative")
    for name in ("preprocess_batch_size", "beam_size", "full_beam_size", "step_beam_size"):
        if int(values.get(name, 0)) < 1:
            errors.append(f"{name} must be at least 1")
    if int(values.get("train_batch_size", 0)) != 1:
        errors.append("train_batch_size is deprecated and must equal 1; use preprocess_batch_size for reaction shard size")
    if not is_valid_device_request(str(values.get("device", ""))):
        errors.append("device must be one of: auto, cpu, cuda, cuda:<index>")
    for name in ("checkpoint", "output_path", "forward_predictions_path", "resource_root"):
        if values.get(name) is not None and str(values.get(name)).strip() == "":
            errors.append(f"{name} must be non-empty when supplied")
    if errors:
        raise ValueError("Invalid KGCL configuration:\n- " + "\n- ".join(errors))
