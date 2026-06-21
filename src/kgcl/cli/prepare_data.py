from __future__ import annotations

import argparse
from collections.abc import Sequence

from kgcl.config import ConfigScope, KGCLConfig, add_arguments, add_config_argument, parse_command_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare KGCL tensors from preprocessed reactions')
    add_config_argument(parser)
    add_arguments(parser, ['dataset', 'root_dir', 'resource_root', 'mode', 'use_rxn_class', 'preprocess_batch_size', 'max_steps', 'preprocess_print_every', 'kekulize'])
    return parser


def parse_config(argv: Sequence[str] | None = None) -> KGCLConfig:
    return parse_command_config(build_parser(), argv, scope=ConfigScope.PREPARE_DATA)


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    from kgcl.data.preparation import run
    return int(run(config) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
