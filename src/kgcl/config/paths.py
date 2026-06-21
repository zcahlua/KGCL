from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root).expanduser().resolve(strict=False))
    def _resolve_user_path(self, value: str | Path, *, relative_to: Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve(strict=False) if candidate.is_absolute() else (relative_to / candidate).resolve(strict=False)
    def dataset_dir(self, dataset: str) -> Path:
        return self.root / 'data' / dataset.lower()
    def raw_csv(self, dataset: str, mode: str) -> Path:
        return self.dataset_dir(dataset) / f'raw_{mode}.csv'
    def canonicalized_csv(self, dataset: str, mode: str) -> Path:
        return self.dataset_dir(dataset) / f'canonicalized_{mode}.csv'
    def split_dir(self, dataset: str, mode: str) -> Path:
        return self.dataset_dir(dataset) / mode
    def serialized_reactions_file(self, dataset: str, mode: str, kekulize: bool = True) -> Path:
        return self.split_dir(dataset, mode) / (f'{mode}.file.kekulized' if kekulize else f'{mode}.file')
    def vocab_dir(self, dataset: str) -> Path:
        return self.split_dir(dataset, 'train')
    def vocab_file(self, dataset: str, name: str) -> Path:
        return self.vocab_dir(dataset) / name
    def prepared_shard_dir(self, dataset: str, mode: str, use_rxn_class: bool) -> Path:
        return self.split_dir(dataset, mode) / ('with_rxn_class' if use_rxn_class else 'without_rxn_class')
    def experiment_dir(self, dataset: str, use_rxn_class: bool, experiment: str) -> Path:
        return self.root / 'experiments' / dataset.lower() / ('with_rxn_class' if use_rxn_class else 'without_rxn_class') / experiment
    def historical_checkpoint(self, dataset: str, use_rxn_class: bool, experiment: str) -> Path:
        return self.experiment_dir(dataset, use_rxn_class, experiment) / historical_checkpoint_name(dataset, use_rxn_class)
    def explicit_checkpoint(self, dataset: str, use_rxn_class: bool, experiment: str, checkpoint: str) -> Path:
        return self._resolve_user_path(checkpoint, relative_to=self.experiment_dir(dataset, use_rxn_class, experiment))
    def prediction_report(self, dataset: str, use_rxn_class: bool, experiment: str, output_path: str | None = None, default_name: str = 'pred_results.txt') -> Path:
        if output_path:
            return self._resolve_user_path(output_path, relative_to=self.root)
        return self.experiment_dir(dataset, use_rxn_class, experiment) / default_name
    def round_trip_prediction_dir(self, dataset: str, use_rxn_class: bool, experiment: str, output_path: str | None = None) -> Path:
        if output_path:
            return self._resolve_user_path(output_path, relative_to=self.root)
        return self.experiment_dir(dataset, use_rxn_class, experiment) / 'pred_text1'
    def forward_prediction_input(self, dataset: str, use_rxn_class: bool, experiment: str, forward_predictions_path: str | None = None) -> Path:
        if forward_predictions_path:
            return self._resolve_user_path(forward_predictions_path, relative_to=self.root)
        return self.experiment_dir(dataset, use_rxn_class, experiment) / 'forward_predictions_50k_top50.txt'

DEFAULT_CHECKPOINTS = {('uspto_50k', False): 'epoch_132.pt', ('uspto_50k', True): 'epoch_128.pt', ('uspto_full', False): 'epoch_168.pt'}

def historical_checkpoint_name(dataset: str, use_rxn_class: bool) -> str:
    key = (dataset.lower(), bool(use_rxn_class))
    if key not in DEFAULT_CHECKPOINTS:
        raise ValueError(f'No historical checkpoint default for dataset={dataset!r}, use_rxn_class={use_rxn_class!r}; pass --checkpoint.')
    return DEFAULT_CHECKPOINTS[key]

def resolve_checkpoint(paths: ProjectPaths, dataset: str, use_rxn_class: bool, experiment: str, checkpoint: str | None = None) -> Path:
    resolved = paths.explicit_checkpoint(dataset, use_rxn_class, experiment, checkpoint) if checkpoint else paths.historical_checkpoint(dataset, use_rxn_class, experiment)
    if not resolved.exists():
        raise FileNotFoundError(f'Checkpoint not found: {resolved}')
    return resolved
