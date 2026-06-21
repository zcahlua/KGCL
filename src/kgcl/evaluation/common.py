"""Shared evaluation setup helpers."""
from __future__ import annotations
from pathlib import Path
from kgcl.config.device import resolve_device
from kgcl.config.paths import ProjectPaths, resolve_checkpoint
from kgcl.resources.functional_groups import resolve_functional_group_resources


def setup_paths(config):
    return ProjectPaths(Path(config.root_dir))


def configure_resources(config):
    return resolve_functional_group_resources(
        use_rxn_class=config.use_rxn_class,
        resource_root=Path(config.resource_root) if config.resource_root else None,
        root_dir=Path(config.root_dir),
    )


def evaluation_setup(config, *, output_kind='report', default_name='pred_results.txt'):
    paths = setup_paths(config)
    device = resolve_device(config.device)
    checkpoint = resolve_checkpoint(paths, config.dataset, config.use_rxn_class, config.experiments, config.checkpoint)
    data_file = paths.serialized_reactions_file(config.dataset, 'test', config.kekulize)
    if output_kind == 'round_trip_dir':
        output = paths.round_trip_prediction_dir(config.dataset, config.use_rxn_class, config.experiments, config.output_path)
    else:
        output = paths.prediction_report(config.dataset, config.use_rxn_class, config.experiments, config.output_path, default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    return {'paths': paths, 'device': device, 'checkpoint': checkpoint, 'data_file': data_file, 'output': output}
