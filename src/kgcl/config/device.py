"""CUDA-only runtime device resolution for KGCL model execution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CUDA_RE = re.compile(r"^cuda(?::([0-9]+))?$")


@dataclass(frozen=True)
class CUDADevice:
    requested: str
    torch_device: Any
    index: int
    name: str
    capability: tuple[int, int]


def is_valid_cuda_device_request(requested: str) -> bool:
    return bool(_CUDA_RE.match((requested or "").strip()))


def is_valid_device_request(requested: str) -> bool:
    """Backward-compatible name for CUDA-only validation."""
    return is_valid_cuda_device_request(requested)


def validate_cuda_device_request(requested: str) -> None:
    if not is_valid_cuda_device_request(requested):
        raise ValueError(
            f"KGCL model execution requires CUDA. Received device={requested!r}.\n"
            "Use --device cuda or --device cuda:<index>.\n"
            "No CPU fallback is provided."
        )


def resolve_cuda_device(requested: str = "cuda") -> CUDADevice:
    requested = (requested or "").strip()
    validate_cuda_device_request(requested)

    import torch

    if torch.version.cuda is None:
        raise RuntimeError(
            "KGCL requires a CUDA-enabled PyTorch build for model execution, but torch.version.cuda is None. "
            "Install a CUDA-enabled PyTorch build. No CPU fallback is provided."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "KGCL requires CUDA for model execution, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and run on an NVIDIA GPU. No CPU fallback is provided."
        )
    count = int(torch.cuda.device_count())
    if count <= 0:
        raise RuntimeError("KGCL requires at least one visible CUDA device. No CPU fallback is provided.")

    match = _CUDA_RE.match(requested)
    assert match is not None
    if match.group(1) is None:
        index = int(torch.cuda.current_device())
    else:
        index = int(match.group(1))
    if index >= count:
        raise ValueError(f"Requested CUDA device cuda:{index}, but only {count} CUDA device(s) are visible.")
    torch.cuda.set_device(index)
    device = torch.device(f"cuda:{index}")
    name = str(torch.cuda.get_device_name(index))
    capability = tuple(torch.cuda.get_device_capability(index))
    print(f"Resolved CUDA device: cuda:{index}")
    print(f"GPU: {name}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Capability: {capability[0]}.{capability[1]}")
    return CUDADevice(requested=requested, torch_device=device, index=index, name=name, capability=capability)  # type: ignore[arg-type]


def resolve_device(requested: str = "cuda") -> str:
    """Compatibility wrapper returning the selected CUDA device string."""
    return str(resolve_cuda_device(requested).torch_device)
