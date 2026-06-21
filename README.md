# KGCL

KGCL is a retrosynthesis workflow with lightweight command wrappers, centralized configuration, immutable external functional-group resources, and historical compatibility shims.

## Environment requirements
Python 3.11 is recommended. Runtime workflows need PyTorch, NumPy, RDKit, pandas, tqdm, and joblib.

## Installation
Lightweight development:
```bash
python -m pip install -e ".[test,dev]"
```
Runtime development:
```bash
python -m pip install -e ".[runtime,test,dev]"
```

## External resource requirements
Functional-group embeddings are immutable external assets in `KGembedding/` and `KGembedding_2/`. Source checkouts infer them from the repository root. Installed workflows should pass `--resource-root /path/to/root` or set `KGCL_RESOURCE_ROOT=/path/to/root`.

## Repository map
- `src/models/KGCL.py`, `src/models/encoder.py`, `src/models/beam_search.py`: canonical model code.
- `src/utils/attn_layer.py`, `src/utils/datasets.py`: canonical utility code.
- `src/kgcl/training/`: training orchestration.
- `src/kgcl/evaluation/`: evaluation orchestration.
- `src/kgcl/data/`: canonicalization, preprocessing, and preparation.
- `src/kgcl/config/`: schema, parsing, validation, and `ProjectPaths`.
- `models/`, `utils/`: historical source-checkout shims only.

## Where to change what
Change CLI options and defaults in `src/kgcl/config/` and `src/kgcl/cli/`. Change workflow path rules in `src/kgcl/config/paths.py`. Do not change checkpoint formats, state-dict keys, model equations, prediction output formats, or binary assets.

## Data layout
```text
data/<dataset>/raw_<mode>.csv
data/<dataset>/canonicalized_<mode>.csv
data/<dataset>/<mode>/<mode>.file.kekulized
```

## Canonicalization
```bash
python canonicalize_prod.py --dataset uspto_full --mode test
```

## Preprocessing
```bash
python preprocess.py --dataset uspto_50k --mode train
python preprocess.py --dataset uspto_full --mode train
```

## Data preparation
```bash
python prepare_data.py --dataset uspto_50k --mode train
```

## Training
```bash
python train.py --dataset uspto_50k
python train.py --dataset uspto_full
```

## Evaluation
```bash
python eval.py --dataset uspto_50k
python eval-full.py --dataset uspto_full
python eval-rtacc.py --dataset uspto_50k
```

## Configuration precedence
Precedence is schema defaults < command-specific defaults < config file < `KGCL_*` environment variables < explicit CLI arguments.

## Checkpoint selection
When `--checkpoint` is omitted, evaluation uses historical checkpoint defaults from `ProjectPaths`. Pass an absolute path or experiment-relative filename with `--checkpoint` to override.

## Output-path behavior
Evaluation creates output parents. Relative `--output-path` values resolve under `root_dir`; absolute paths are used as provided.

## Compatibility commands
```bash
python -c "from models.KGCL import KGCL"
python -c "from models.beam_search import BeamSearch"
python -c "from utils.rxn_graphs import MolGraph, RxnGraph, Vocab"
python -c "from utils.generate_edits import ReactionData"
```

## Tests
```bash
python -m pytest -q
python -m pytest tests/unit -q
python -m pytest tests/smoke -q
python -m pytest tests/compatibility -q
python -m pytest tests/runtime -q
python -m compileall src
ruff check src/kgcl tests
```
