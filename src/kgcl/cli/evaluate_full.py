from __future__ import annotations

import argparse
from collections.abc import Sequence

from kgcl.config import KGCLConfig, add_arguments, add_config_argument, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate KGCL USPTO-FULL model')
    add_config_argument(parser)
    add_arguments(parser, ['dataset', 'root_dir', 'resource_root', 'device', 'use_rxn_class', 'experiments', 'full_beam_size', 'step_beam_size', 'checkpoint', 'output_path', 'max_steps', 'kekulize'])
    parser.set_defaults(dataset='uspto_full')
    return parser


def parse_config(argv: Sequence[str] | None = None) -> KGCLConfig:
    parsed = build_parser().parse_args(argv)
    overrides = vars(parsed).copy()
    config_file = overrides.pop('config_file')
    return load_config(config_file=config_file, cli_overrides=overrides)


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    from kgcl.evaluation.full import run
    return int(run(config) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
