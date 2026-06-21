"""Resolve KGCL functional-group resources without moving binary assets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

@dataclass(frozen=True)
class FunctionalGroupResourceConfig:
    resource_root: Path | None = None
    root_dir: Path | None = None

@dataclass(frozen=True)
class FunctionalGroupResourcePaths:
    definitions: Path
    embeddings: Path

_CONFIG = FunctionalGroupResourceConfig()

def _normalize(path: str | Path | None) -> Path | None:
    return None if path in (None, '') else Path(path).expanduser().resolve(strict=False)

def configure_functional_group_resources(*, resource_root: str | Path | None, root_dir: str | Path | None) -> None:
    global _CONFIG
    new = FunctionalGroupResourceConfig(_normalize(resource_root), _normalize(root_dir))
    if new != _CONFIG:
        _CONFIG = new

def get_functional_group_resource_config() -> FunctionalGroupResourceConfig:
    return _CONFIG

def _candidate_roots(resource_root: Path | None = None, root_dir: Path | None = None) -> list[Path]:
    if resource_root is not None:
        return [resource_root]
    env_root = os.environ.get('KGCL_RESOURCE_ROOT')
    if env_root:
        return [_normalize(env_root)]  # type: ignore[list-item]
    roots: list[Path] = []
    if root_dir is not None:
        roots.append(root_dir)
    roots.append(Path(__file__).resolve().parents[3])
    roots.append(Path(__file__).resolve().parents[2])
    return roots

@lru_cache(maxsize=8)
def _resolve_cached(use_rxn_class: bool, resource_root_key: str | None, root_dir_key: str | None, env_key: str | None) -> FunctionalGroupResourcePaths:
    resource_root = Path(resource_root_key) if resource_root_key else None
    root_dir = Path(root_dir_key) if root_dir_key else None
    directory_name = 'KGembedding_2' if use_rxn_class else 'KGembedding'
    attempted: list[tuple[Path, Path]] = []
    for root in _candidate_roots(resource_root, root_dir):
        bundle_dir = root / directory_name
        definitions = bundle_dir / 'funcgroup.txt'
        embeddings = bundle_dir / 'fg2emb.pkl'
        attempted.append((definitions, embeddings))
        if definitions.is_file() and embeddings.is_file():
            return FunctionalGroupResourcePaths(definitions=definitions.resolve(), embeddings=embeddings.resolve())
    formatted = '\n'.join(f'- definitions: {d}; embeddings: {e}' for d, e in attempted)
    raise FileNotFoundError(
        'Could not locate KGCL functional-group resources for '
        f'use_rxn_class={use_rxn_class}. Expected both files at one of:\n{formatted}\n'
        'Resolution order: explicit resource_root, KGCL_RESOURCE_ROOT, root_dir, source-checkout inference.'
    )

def resolve_functional_group_resources(*, use_rxn_class: bool, resource_root: Path | None = None, root_dir: Path | None = None) -> FunctionalGroupResourcePaths:
    cfg = _CONFIG
    rr = _normalize(resource_root) or cfg.resource_root
    rd = _normalize(root_dir) or cfg.root_dir
    env = os.environ.get('KGCL_RESOURCE_ROOT')
    return _resolve_cached(bool(use_rxn_class), str(rr) if rr else None, str(rd) if rd else None, str(_normalize(env)) if env and rr is None else None)
