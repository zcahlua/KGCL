# KGCL configuration files

`default.yaml` mirrors the historical command-line defaults. Use `--config path/to/file.yaml` with command scripts, then override individual values on the command line.

Config files must be flat YAML or JSON mappings matching `KGCLConfig` fields. Nested sections are intentionally rejected because KGCL currently keeps a flat backward-compatible schema and nested keys such as `training.device` and `evaluation.device` could otherwise collide silently.

Model execution is CUDA-only. Training and evaluation commands accept `device: cuda` or `device: cuda:<visible-index>`; CPU and automatic fallback are not supported. Data-only preprocessing and canonicalization do not require CUDA.
