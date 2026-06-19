# KGCL contributor/agent notes

## Setup

- Recommended Python: 3.11.8.
- Install runtime dependencies matching the README: PyTorch, NumPy, RDKit, pandas, tqdm, and joblib.
- Install test dependency: `python -m pip install pytest`.

## Checks

- Unit/smoke tests: `python -m pytest`.
- Import/config syntax check: `python -m py_compile src/kgcl/config/*.py`.
- CLI help checks: `python train.py --help`, `python preprocess.py --help`, `python prepare_data.py --help`.

## Architecture rules

- Keep command-line scripts thin orchestration layers.
- Put user-adjustable defaults, descriptions, aliases, and validation in `src/kgcl/config`.
- Do not change scientific algorithms, checkpoint formats, prediction output formats, or public commands without compatibility handling.
- Prefer compatibility shims over breaking import paths.
- Do not manually edit generated data under `data/` or generated experiment outputs under `experiments/` except when intentionally updating examples.
