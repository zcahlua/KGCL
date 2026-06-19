# KGCL Architecture

## Before this cleanup

```text
.
└── KGCL-main/
    ├── README.md
    ├── pyproject.toml
    ├── src/kgcl/config/
    ├── tests/
    ├── docs/
    ├── models/
    ├── utils/
    ├── KGembedding/
    ├── KGembedding_2/
    ├── data/
    └── experiments/
```

## Current repository tree

```text
.
├── README.md
├── pyproject.toml
├── train.py / preprocess.py / prepare_data.py / eval*.py
├── src/kgcl/
│   ├── config/        # schema, coercion, validation, parser generation
│   ├── cli/           # lightweight public workflow parser helpers
│   ├── chemistry/     # shared chemistry/domain operations
│   ├── data/          # reserved package namespace for future data migration
│   ├── models/        # reserved package namespace for future model migration
│   ├── training/      # reserved package namespace for future training migration
│   └── evaluation/    # reserved package namespace for future evaluation migration
├── models/            # historical serialization-sensitive model import path
├── utils/             # historical serialization-sensitive data/action import path
├── configs/           # default and example configuration files
├── tests/             # lightweight regression tests
├── KGembedding/       # shipped functional-group embedding assets
├── KGembedding_2/     # alternate shipped functional-group embedding assets
├── data/              # datasets and generated preprocessing artifacts
└── experiments/       # checkpoints and prediction outputs
```

`models/` and `utils/` remain at their historical top-level import paths because existing torch/joblib artifacts may refer to modules such as `models.KGCL`, `utils.rxn_graphs`, and `utils.reaction_actions`.

## Configuration precedence and parsing

Configuration values are resolved in this order:

```text
dataclass defaults < config file < KGCL_* environment variables < CLI arguments
```

The schema in `src/kgcl/config/schema.py` is the Python type authority. Config files are parsed as JSON or with `yaml.safe_load`; malformed files and unknown keys are rejected. Boolean values accept common true/false spellings. CLI booleans support explicit false forms such as `--no-kekulize`.

Relative paths are resolved by workflow code beneath `root_dir`. Training, preprocessing, and tensor preparation now consume `root_dir`; evaluation scripts retain their historical runtime implementation and still need deeper migration.

## Parameter-consumer matrix

| Parameter | Type | CLI / env | Runtime consumer | Test coverage |
| --- | --- | --- | --- | --- |
| dataset | str | `--dataset`, `KGCL_DATASET` | train/preprocess/prepare/eval path and policy selection | parser/config tests |
| mode | str | `--mode`, `KGCL_MODE` | preprocess and prepare split selection | parser/config tests |
| root_dir | str | `--root_dir`, `KGCL_ROOT_DIR` | train/preprocess/prepare path roots | parser/help tests |
| experiments | str | `--experiments`, `KGCL_EXPERIMENTS` | evaluation experiment directory | parser/help tests |
| use_rxn_class | bool | `--use_rxn_class` / `--no-use_rxn_class`, env | data/model path and feature behavior | bool/parser tests |
| kekulize | bool | `--kekulize` / `--no-kekulize`, env | preprocessing molecule handling | bool/parser tests |
| model dimensions/dropouts | int/float | matching flags/env | model config construction | parser/type tests |
| epochs, patience, factor, thresh, max_clip | int/float | matching flags/env | training loop/scheduler | parser/type tests |
| lr | float or unset | `--lr`, `KGCL_LR` | training optimizer; unset keeps dataset policy | parser/type tests |
| train_batch_size | int | matching flag/env | training/validation loaders | parser/help tests |
| preprocess_batch_size | int | `--preprocess_batch_size`, legacy `--batch_size`, env | prepare-data shard flushing | alias tests |
| beam_size/full_beam_size/max_steps | int | matching flags/env | evaluation/beam-search and prep limits | parser/help tests |
| num_workers/print intervals/device | int/str | matching flags/env | loaders, logging, training device | parser/help tests |

## Compatibility matrix

| Surface | Compatibility status |
| --- | --- |
| Top-level commands | `train.py`, `preprocess.py`, `prepare_data.py`, `eval.py`, `eval-full.py`, and `eval-rtacc.py` remain present. |
| Historical flags | Existing underscore flags remain accepted; canonical `--preprocess_batch_size` is added while legacy `--batch_size` remains accepted. |
| Boolean flags | True and false forms are accepted for generated config flags. |
| Historical imports | `models.*` and `utils.*` remain importable for checkpoints and joblib objects. |
| `prepare_data.apply_edit_to_mol` | Still importable, re-exported from shared chemistry logic. |
| Checkpoints/joblib | File locations and state-dict keys were not intentionally changed; representative fixture loading was not available. |

## Where to change what

- Add user-facing configuration fields in `src/kgcl/config/schema.py`, validation in `src/kgcl/config/validation.py`, and parser behavior in `src/kgcl/config/loader.py`.
- Add lightweight CLI-only parser changes in `src/kgcl/cli/help.py` so `--help` remains dependency-light.
- Put shared molecule-edit chemistry in `src/kgcl/chemistry/edit_application.py`.
- Keep compatibility-sensitive model/data classes under top-level `models/` and `utils/` unless fixtures prove safe migration.
