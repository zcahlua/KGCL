from __future__ import annotations
import sys
from .help import maybe_print_help


def _run(script: str, module: str) -> int:
    maybe_print_help(script, [script, *sys.argv[1:]])
    mod = __import__(module, fromlist=["main"])
    return int(mod.main(sys.argv[1:]) or 0)


def train() -> int:
    return _run("train.py", "kgcl.cli.train")

def preprocess() -> int:
    return _run("preprocess.py", "kgcl.cli.preprocess")

def prepare_data() -> int:
    return _run("prepare_data.py", "kgcl.cli.prepare_data")

def evaluate() -> int:
    return _run("eval.py", "kgcl.cli.evaluate")

def evaluate_full() -> int:
    return _run("eval-full.py", "kgcl.cli.evaluate_full")

def evaluate_round_trip() -> int:
    return _run("eval-rtacc.py", "kgcl.cli.evaluate_round_trip")

def canonicalize() -> int:
    return _run("canonicalize_prod.py", "kgcl.cli.canonicalize")
