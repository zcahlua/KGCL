# KGCL architecture

Runtime entry points live in `src/kgcl/cli` and own argparse parsers. Runtime modules in `src/kgcl/data`, `src/kgcl/training`, and `src/kgcl/evaluation` consume `KGCLConfig` and do not build command-line parsers. Canonical scientific implementation modules are `src/models` and `src/utils`; root `models/` and `utils/` are import shims for historical checkpoints and imports.

Functional-group resources must be configured before model or graph imports. `ProjectPaths` resolves `root_dir` once to an absolute path and all generated workflow paths are derived from that root.
