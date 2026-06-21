"""Authoritative user-adjustable configuration for KGCL workflows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, get_type_hints


@dataclass
class KGCLConfig:
    # data and paths
    dataset: str = "uspto_50k"
    mode: str = "train"
    root_dir: str = "./"
    experiments: str = "BEST"
    use_rxn_class: bool = False
    kekulize: bool = True
    # model architecture
    atom_message: bool = False
    use_attn: bool = False
    n_heads: int = 8
    mpn_size: int = 256
    depth: int = 10
    dropout_mpn: float = 0.15
    mlp_size: int = 512
    dropout_mlp: float = 0.2
    # training and optimization
    epochs: int = 200
    lr: float | None = None
    patience: int = 5
    factor: float = 0.8
    thresh: float = 0.01
    max_clip: int = 10
    train_batch_size: int = 1
    preprocess_batch_size: int = 256
    # evaluation
    beam_size: int = 50
    full_beam_size: int = 10
    step_beam_size: int = 10
    checkpoint: str | None = None
    output_path: str | None = None
    forward_predictions_path: str | None = None
    max_steps: int = 9
    # runtime/logging
    num_workers: int = 24
    print_every: int = 200
    preprocess_print_every: int = 1000
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ALIASES = {
    "batch_size": "preprocess_batch_size",
}

FIELD_TYPES = get_type_hints(KGCLConfig)
FIELD_TYPES["lr"] = float
FIELD_TYPES["checkpoint"] = str
FIELD_TYPES["output_path"] = str
FIELD_TYPES["forward_predictions_path"] = str
DEFAULTS = KGCLConfig()

PARAMETER_HELP = {
    "dataset": "Dataset name, normalized to lowercase (for example uspto_50k or uspto_full).",
    "mode": "Dataset split/workflow mode: train, valid, or test.",
    "root_dir": "Repository/workflow root used to resolve data and experiment paths.",
    "experiments": "Experiment/checkpoint directory name used by evaluation commands.",
    "use_rxn_class": "Whether reaction classes are included as model/input conditions.",
    "kekulize": "Whether preprocessing kekulizes molecules.",
    "atom_message": "Use atom-level message passing instead of the default bond-level setting.",
    "use_attn": "Enable global attention in the model.",
    "n_heads": "Number of attention heads when attention is enabled.",
    "mpn_size": "Message passing network hidden dimension.",
    "depth": "Number of message passing iterations.",
    "dropout_mpn": "Dropout rate in the message passing network.",
    "mlp_size": "MLP hidden dimension.",
    "dropout_mlp": "Dropout rate in MLP layers.",
    "epochs": "Maximum training epochs.",
    "lr": "Learning rate; train.py preserves dataset-specific overrides.",
    "patience": "ReduceLROnPlateau patience.",
    "factor": "ReduceLROnPlateau multiplicative decay factor.",
    "thresh": "ReduceLROnPlateau absolute improvement threshold.",
    "max_clip": "Maximum gradient norm for clipping.",
    "train_batch_size": "Training DataLoader batch size; current default preserves historical behavior.",
    "preprocess_batch_size": "Number of reactions per serialized preprocessing shard.",
    "beam_size": "Beam-search width for standard evaluation.",
    "full_beam_size": "Beam-search width for USPTO-FULL evaluation script.",
    "step_beam_size": "Beam-search step expansion width.",
    "checkpoint": "Checkpoint filename or absolute path. When unset, historical dataset defaults are used.",
    "output_path": "Optional prediction output path.",
    "forward_predictions_path": "Optional forward prediction file for round-trip evaluation.",
    "max_steps": "Maximum number of graph edit steps.",
    "num_workers": "DataLoader worker processes.",
    "print_every": "Training progress print interval.",
    "preprocess_print_every": "Preprocessing progress print interval.",
    "device": "Runtime device: auto, cpu, cuda, or cuda device string.",
}
