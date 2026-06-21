from __future__ import annotations

import argparse
from collections.abc import Sequence

from kgcl.config import KGCLConfig, add_arguments, add_config_argument, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train KGCL model')
    add_config_argument(parser)
    add_arguments(parser, ['dataset', 'root_dir', 'resource_root', 'device', 'use_rxn_class', 'atom_message', 'use_attn', 'n_heads', 'epochs', 'mpn_size', 'depth', 'dropout_mpn', 'mlp_size', 'dropout_mlp', 'lr', 'patience', 'factor', 'thresh', 'max_clip', 'train_batch_size', 'print_every', 'num_workers', 'kekulize'])
    return parser


def parse_config(argv: Sequence[str] | None = None) -> KGCLConfig:
    parsed = build_parser().parse_args(argv)
    overrides = vars(parsed).copy()
    config_file = overrides.pop('config_file')
    return load_config(config_file=config_file, cli_overrides=overrides)


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    from kgcl.training.runner import run_training
    return int(run_training(config) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
