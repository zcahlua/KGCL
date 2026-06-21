"""Resolve KGCL functional-group resources without moving binary assets.

The research embedding bundles intentionally remain at their historical tracked
locations (``KGembedding`` and ``KGembedding_2``).  This resolver never relies
on the process current working directory; callers may provide an explicit root,
set ``KGCL_RESOURCE_ROOT``, or use a source checkout where the repository root
can be inferred from this file's location.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionalGroupResourcePaths:
    definitions: Path
    embeddings: Path


def _candidate_roots(resource_root: Path | None = None) -> list[Path]:
    if resource_root is not None:
        return [Path(resource_root).expanduser()]
    env_root = os.environ.get("KGCL_RESOURCE_ROOT")
    if env_root:
        return [Path(env_root).expanduser()]
    roots: list[Path] = []
    # src/kgcl/resources/functional_groups.py -> repo root is parents[3]
    roots.append(Path(__file__).resolve().parents[3])
    # Installed-data location placeholder: if a distribution installs external
    # assets beside the package root in the future, this path can already be
    # discovered without making wheel self-containment claims in this task.
    roots.append(Path(__file__).resolve().parents[2])
    return roots


def resolve_functional_group_resources(
    *,
    use_rxn_class: bool,
    resource_root: Path | None = None,
) -> FunctionalGroupResourcePaths:
    """Return paths for one immutable functional-group resource bundle.

    Resolution order is explicit ``resource_root``, ``KGCL_RESOURCE_ROOT``, a
    source-checkout repository root inferred from this file, then a documented
    installed-data fallback if those historical directories already exist.
    """
    directory_name = "KGembedding_2" if use_rxn_class else "KGembedding"
    attempted: list[tuple[Path, Path]] = []
    for root in _candidate_roots(resource_root):
        bundle_dir = root / directory_name
        definitions = bundle_dir / "funcgroup.txt"
        embeddings = bundle_dir / "fg2emb.pkl"
        attempted.append((definitions, embeddings))
        if definitions.is_file() and embeddings.is_file():
            return FunctionalGroupResourcePaths(
                definitions=definitions.resolve(),
                embeddings=embeddings.resolve(),
            )

    formatted = "\n".join(
        f"- definitions: {definitions}; embeddings: {embeddings}"
        for definitions, embeddings in attempted
    )
    raise FileNotFoundError(
        "Could not locate KGCL functional-group resources for "
        f"use_rxn_class={use_rxn_class}. Expected both files at one of:\n{formatted}\n"
        "Provide resource_root or set KGCL_RESOURCE_ROOT to the repository root "
        "containing KGembedding/ and KGembedding_2/."
    )
