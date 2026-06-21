"""Shared evaluation setup helpers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from kgcl.config.device import CUDADevice, resolve_cuda_device
from kgcl.config.paths import ProjectPaths, resolve_checkpoint
from kgcl.resources.functional_groups import configure_functional_group_resources

@dataclass(frozen=True)
class EvaluationContext:
    paths: ProjectPaths
    cuda: CUDADevice
    checkpoint_path: Path
    test_data_path: Path
    experiment_dir: Path
    output_path: Path

def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 1
    while True:
        candidate = path.with_name(f'{stem}_{idx}{suffix}')
        if not candidate.exists():
            return candidate
        idx += 1

def evaluation_setup(config, *, output_kind='standard', default_name='pred_results.txt') -> EvaluationContext:
    paths = ProjectPaths(Path(config.root_dir))
    configure_functional_group_resources(resource_root=config.resource_root, root_dir=paths.root)
    cuda = resolve_cuda_device(config.device)
    checkpoint = resolve_checkpoint(paths, config.dataset, config.use_rxn_class, config.experiments, config.checkpoint)
    data_file = paths.serialized_reactions_file(config.dataset, 'test', config.kekulize)
    if not data_file.exists():
        raise FileNotFoundError(f'Test data file not found: {data_file}')
    exp_dir = paths.experiment_dir(config.dataset, config.use_rxn_class, config.experiments)
    if output_kind == 'round_trip_dir':
        output = paths.round_trip_prediction_dir(config.dataset, config.use_rxn_class, config.experiments, config.output_path)
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = paths.prediction_report(config.dataset, config.use_rxn_class, config.experiments, config.output_path, default_name)
        if output_kind == 'full' and not config.output_path:
            output = next_available_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Resolved device: {cuda.torch_device}')
    print(f'Resolved checkpoint: {checkpoint}')
    print(f'Resolved test data: {data_file}')
    print(f'Resolved output: {output}')
    return EvaluationContext(paths, cuda, checkpoint, data_file, exp_dir, output)
