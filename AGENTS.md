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

## Refactor conventions

- Root workflow scripts are compatibility wrappers; put parser/config changes in `src/kgcl/cli` and reusable path policy in `src/kgcl/config`.
- Keep help parsing lightweight: `--help` must not require PyTorch, RDKit, pandas, NumPy, or joblib.
- Historical imports under `models.*` and `utils.*` are serialization compatibility surfaces; keep them installable and avoid renaming checkpoint/joblib classes.
- Functional-group embedding files are package resources under `src/kgcl/resources/functional_groups`; load them with `importlib.resources` and do not depend on the current working directory.
- Generated outputs (`build/`, `dist/`, caches, generated prediction files, generated preprocessing shards) should remain untracked unless deliberately added as reference artifacts.

## Structural refactor rules

- Each command module in `src/kgcl/cli` owns exactly one `build_parser()` and `main()` must use it.
- CLI imports must remain lightweight: no torch, RDKit, NumPy, pandas, joblib, tqdm, `models`, or `utils` before parsing.
- Canonical historical implementation sources are `src/models` and `src/utils`; root `models/` and `utils/` are shims only.
- Preserve historical serialization names, checkpoint keys, joblib class names, and binary assets.
- Treat `experiments/**/*.pt`, `*.pth`, `*.joblib`, `*.npy`, `*.npz`, and KGembedding pickle files as immutable.
- Functional-group embeddings are external immutable research resources, not wheel package data; use `--resource-root` or `KGCL_RESOURCE_ROOT` outside a source checkout.
- Run `python -m pytest -q`, `python -m compileall src`, root help commands, and binary hash verification before finishing.
