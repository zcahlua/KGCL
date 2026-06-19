"""Lightweight public workflow parsers for compatibility tests.

The helpers in this module are retained for scripts that need to answer
``--help`` before importing optional scientific dependencies.
"""
from __future__ import annotations

import argparse

from kgcl.config import add_arguments, add_config_argument


def parser_for(script: str) -> argparse.ArgumentParser:
    if script == "train.py":
        parser = argparse.ArgumentParser(description="Train KGCL model")
        add_config_argument(parser)
        add_arguments(parser, ['dataset', 'root_dir', 'device', 'use_rxn_class', 'atom_message', 'use_attn', 'n_heads', 'epochs', 'mpn_size', 'depth', 'dropout_mpn', 'mlp_size', 'dropout_mlp', 'lr', 'patience', 'factor', 'thresh', 'max_clip', 'train_batch_size', 'print_every', 'num_workers'])
        return parser
    if script == "preprocess.py":
        parser = argparse.ArgumentParser(description="Preprocess KGCL reaction CSV files")
        add_config_argument(parser)
        add_arguments(parser, ['dataset', 'root_dir', 'mode', 'preprocess_print_every', 'kekulize', 'max_steps'])
        return parser
    if script == "prepare_data.py":
        parser = argparse.ArgumentParser(description="Prepare KGCL tensors from preprocessed reactions")
        add_config_argument(parser)
        add_arguments(parser, ['dataset', 'root_dir', 'mode', 'use_rxn_class', 'preprocess_batch_size', 'max_steps', 'preprocess_print_every', 'kekulize'])
        return parser
    if script == "eval-full.py":
        parser = argparse.ArgumentParser(description="Evaluate KGCL on USPTO-FULL")
        add_config_argument(parser)
        add_arguments(parser, ['dataset', 'root_dir', 'device', 'use_rxn_class', 'experiments', 'full_beam_size', 'max_steps'])
        parser.set_defaults(dataset='uspto_full')
        return parser
    if script == "eval-rtacc.py":
        parser = argparse.ArgumentParser(description="Evaluate KGCL round-trip accuracy")
        add_config_argument(parser)
        add_arguments(parser, ['dataset', 'root_dir', 'device', 'use_rxn_class', 'experiments', 'beam_size', 'max_steps'])
        return parser
    parser = argparse.ArgumentParser(description="Evaluate KGCL model")
    add_config_argument(parser)
    add_arguments(parser, ['dataset', 'root_dir', 'device', 'use_rxn_class', 'experiments', 'beam_size', 'max_steps'])
    return parser


def maybe_print_help(script: str, argv: list[str]) -> None:
    if any(arg in {"-h", "--help"} for arg in argv[1:]):
        parser_for(script).parse_args(["--help"])
