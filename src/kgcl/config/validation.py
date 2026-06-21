"""Validation for KGCL configuration values."""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .device import is_valid_cuda_device_request


class ConfigScope(str, Enum):
    CANONICALIZE = "canonicalize"
    PREPROCESS = "preprocess"
    PREPARE_DATA = "prepare_data"
    TRAIN = "train"
    EVALUATE = "evaluate"
    EVALUATE_FULL = "evaluate_full"
    EVALUATE_ROUND_TRIP = "evaluate_round_trip"


def _positive(values: Mapping[str, Any], name: str, errors: list[str]) -> None:
    if int(values.get(name, 0)) < 1:
        errors.append(f"{name} must be at least 1")


def _non_empty_optional(values: Mapping[str, Any], name: str, errors: list[str]) -> None:
    if values.get(name) is not None and str(values.get(name)).strip() == "":
        errors.append(f"{name} must be non-empty when supplied")


def validate_config(values: Mapping[str, Any], *, scope: ConfigScope | None = None) -> None:
    errors: list[str] = []
    if str(values.get("dataset", "")).strip() == "":
        errors.append("dataset must be a non-empty string")
    if str(values.get("root_dir", "")).strip() == "":
        errors.append("root_dir must be a non-empty string")
    _non_empty_optional(values, "resource_root", errors)

    if scope in {ConfigScope.PREPROCESS, ConfigScope.PREPARE_DATA, ConfigScope.CANONICALIZE}:
        if values.get("mode") not in {"train", "valid", "test"}:
            errors.append("mode must be one of: train, valid, test")
    if scope in {ConfigScope.PREPROCESS, ConfigScope.PREPARE_DATA}:
        _positive(values, "preprocess_batch_size", errors)
        _positive(values, "preprocess_print_every", errors)
        _positive(values, "max_steps", errors)
    if scope == ConfigScope.TRAIN:
        if not is_valid_cuda_device_request(str(values.get("device", ""))):
            errors.append(
                f"KGCL model execution requires CUDA. Received device={values.get('device')!r}. "
                "Use --device cuda or --device cuda:<index>. No CPU fallback is provided."
            )
        for name in ("epochs", "n_heads", "mpn_size", "depth", "mlp_size", "patience", "max_clip"):
            _positive(values, name, errors)
        if int(values.get("num_workers", 0)) < 0:
            errors.append("num_workers must be non-negative")
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
        if int(values.get("train_batch_size", 0)) != 1:
            errors.append("train_batch_size is deprecated and must equal 1; use preprocess_batch_size for reaction shard size")
    if scope in {ConfigScope.EVALUATE, ConfigScope.EVALUATE_FULL, ConfigScope.EVALUATE_ROUND_TRIP}:
        if not is_valid_cuda_device_request(str(values.get("device", ""))):
            errors.append(
                f"KGCL model execution requires CUDA. Received device={values.get('device')!r}. "
                "Use --device cuda or --device cuda:<index>. No CPU fallback is provided."
            )
        for name in ("beam_size", "full_beam_size", "step_beam_size", "max_steps"):
            _positive(values, name, errors)
        for name in ("checkpoint", "output_path"):
            _non_empty_optional(values, name, errors)
        if scope == ConfigScope.EVALUATE_ROUND_TRIP:
            _non_empty_optional(values, "forward_predictions_path", errors)
    if scope is None:
        # Backward-compatible broad validation for direct load_config() callers.
        if values.get("lr") is not None and float(values.get("lr", 0.0)) <= 0:
            errors.append("lr must be positive")
        for name in ("preprocess_batch_size", "beam_size", "full_beam_size", "step_beam_size", "max_steps"):
            _positive(values, name, errors)
        if values.get("mode") not in {"train", "valid", "test"}:
            errors.append("mode must be one of: train, valid, test")
    if errors:
        raise ValueError("Invalid KGCL configuration:\n- " + "\n- ".join(errors))
