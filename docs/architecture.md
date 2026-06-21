# KGCL architecture

KGCL uses a `src/` layout. Command modules in `src/kgcl/cli` own the single authoritative parser for each command and defer scientific imports until after parsing. Root scripts are compatibility wrappers that insert `src/` and call the matching CLI module.

Canonical serialization-sensitive implementations remain in historical packages `src/models` and `src/utils` so existing checkpoints and joblib objects continue to import names such as `models.KGCL` and `utils.generate_edits.ReactionData`. Root `models/` and `utils/` are source-checkout shims only. `kgcl.models` is a public facade over those historical modules.

Workflow implementation lives in `kgcl.training`, `kgcl.data`, `kgcl.chemistry`, and `kgcl.evaluation`. Path and checkpoint construction is centralized in `kgcl.config.paths`; runtime devices are resolved by `kgcl.config.device`.

Functional-group embedding binaries are externally located immutable research resources in `KGembedding/` and `KGembedding_2/`. They are not packaged in wheels. Source checkouts are discovered automatically; installed workflows should pass `--resource-root` or set `KGCL_RESOURCE_ROOT`.
