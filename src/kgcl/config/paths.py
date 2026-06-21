from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    def dataset_dir(self, dataset: str) -> Path:
        return self.root / "data" / dataset.lower()
    def split_dir(self, dataset: str, mode: str) -> Path:
        return self.dataset_dir(dataset) / mode
    def serialized_reactions_file(self, dataset: str, mode: str, kekulize: bool = True) -> Path:
        suffix = ".file.kekulized" if kekulize else ".file"
        return self.split_dir(dataset, mode) / f"{mode}{suffix}"
    def vocab_dir(self, dataset: str) -> Path:
        return self.split_dir(dataset, "train")
    def prepared_shard_dir(self, dataset: str, mode: str, use_rxn_class: bool) -> Path:
        return self.split_dir(dataset, mode) / ("with_rxn_class" if use_rxn_class else "without_rxn_class")
    def experiment_dir(self, dataset: str, use_rxn_class: bool, experiment: str) -> Path:
        return self.root / "experiments" / dataset.lower() / ("with_rxn_class" if use_rxn_class else "without_rxn_class") / experiment

DEFAULT_CHECKPOINTS = {
    ("uspto_50k", False): "epoch_132.pt",
    ("uspto_50k", True): "epoch_128.pt",
    ("uspto_full", False): "epoch_168.pt",
}

def historical_checkpoint_name(dataset: str, use_rxn_class: bool) -> str:
    key = (dataset.lower(), bool(use_rxn_class))
    if key not in DEFAULT_CHECKPOINTS:
        raise ValueError(f"No historical checkpoint default for dataset={dataset!r}, use_rxn_class={use_rxn_class!r}; pass --checkpoint.")
    return DEFAULT_CHECKPOINTS[key]

def resolve_checkpoint(paths: ProjectPaths, dataset: str, use_rxn_class: bool, experiment: str, checkpoint: str | None = None) -> Path:
    exp_dir = paths.experiment_dir(dataset, use_rxn_class, experiment)
    candidate = Path(checkpoint) if checkpoint else Path(historical_checkpoint_name(dataset, use_rxn_class))
    resolved = candidate if candidate.is_absolute() else exp_dir / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved
