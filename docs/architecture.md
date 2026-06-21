# KGCL architecture

KGCL keeps scientific implementations under `src/models/` and `src/utils/`. Root `models/` and `utils/` packages are source-checkout shims for historical imports and serialized objects.

Directory tree: root wrappers (`train.py`, `preprocess.py`, `prepare_data.py`, `canonicalize_prod.py`, `eval*.py`) call `src/kgcl/cli/`; parsers load `src/kgcl/config/`; runtime work lives in `src/kgcl/data/`, `src/kgcl/training/`, and `src/kgcl/evaluation/`; reusable path policy lives in `src/kgcl/config/paths.py`; functional-group resource resolution lives in `src/kgcl/resources/functional_groups.py`.

CLI modules are lightweight and must not import PyTorch, RDKit, pandas, joblib, `models`, or `utils` before parsing. Runtime modules configure functional-group resources first, then import `models` and graph utilities. This prevents stale or wrong resource roots.

`ProjectPaths` resolves the workflow root once and constructs data, vocabulary, prepared shard, experiment, checkpoint, and prediction paths without duplicating relative roots.

Configuration precedence is schema defaults, command defaults, config file, environment, then CLI. Checkpoint names and state keys are compatibility surfaces and must remain unchanged. Binary assets, including checkpoints, embeddings, joblib files, arrays, images, PDFs, archives, and generated shards, are immutable.

Testing layers: unit tests cover configuration, paths, resource cache, validation, and docs; smoke tests cover command help; compatibility tests cover historical imports and binary hashes; runtime tests exercise real scientific dependencies and resources; wheel checks verify installed console scripts.
