"""Runtime device resolution for KGCL."""
from __future__ import annotations

import re

_CUDA_RE = re.compile(r"^cuda(?::\d+)?$")


def is_valid_device_request(requested: str) -> bool:
    return requested in {"auto", "cpu", "cuda"} or bool(_CUDA_RE.match(requested or ""))


def resolve_device(requested: str = "auto") -> str:
    if not is_valid_device_request(requested):
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")
    if requested == "cpu":
        return "cpu"
    import torch
    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested.startswith("cuda") and not available:
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available. Use --device cpu or --device auto.")
    return requested
