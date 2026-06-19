# KGCL architecture

## Final directory tree

```text
KGCL-main/
├── configs/              # User-editable defaults and example overrides.
├── docs/                 # Contributor architecture and configuration guides.
├── src/kgcl/config/      # Authoritative configuration schema, loading, and validation.
├── models/               # KGCL neural model, encoder, beam search, and model utilities.
├── utils/                # Chemistry, graph, dataset, collation, loss, and reaction action logic.
├── data/                 # Expected local dataset root; large generated files are not committed.
├── experiments/          # Checkpoints and generated prediction files.
├── preprocess.py         # CSV-to-reaction-object preprocessing entry point.
├── prepare_data.py       # Reaction-object-to-training-tensor entry point.
├── train.py              # Training orchestration entry point.
├── eval.py               # USPTO-50K exact/MaxFrag evaluation entry point.
├── eval-full.py          # USPTO-FULL evaluation entry point.
└── eval-rtacc.py         # Round-trip accuracy evaluation entry point.
```

## Execution flow

1. `preprocess.py` reads canonicalized CSV files from `data/<dataset>/`, generates edit vocabularies and serialized reaction objects, and writes split-specific files.
2. `prepare_data.py` reads those reaction objects, builds graph/edit tensors via `utils.collate_fn` and `utils.rxn_graphs`, and writes batch shards.
3. `train.py` loads vocabularies and training shards, builds `models.KGCL`, trains with the existing loss functions, and writes checkpoints/logs under `experiments/`.
4. `eval.py`, `eval-full.py`, and `eval-rtacc.py` load checkpoints, run `models.BeamSearch`, and write or report prediction metrics.

## Boundaries and dependency directions

- Configuration is centralized in `src/kgcl/config`; workflow scripts parse CLI options and then pass plain values to existing implementation functions.
- Orchestration stays in the top-level scripts for backward-compatible commands.
- Scientific model and chemistry logic remains in `models/` and `utils/` to avoid numerical or public import behavior changes.
- Data and experiment I/O paths are documented and configurable without changing output formats.

## Extension points

- Model architecture: `models/KGCL.py`, `models/encoder.py`, `utils/attn_layer.py`.
- Data loading: `utils/datasets.py` and `utils/collate_fn.py`.
- Preprocessing/edit generation: `preprocess.py`, `utils/generate_edits.py`, `utils/reaction_actions.py`.
- Training loop/objective: `train.py` and `utils/ADNCE.py`.
- Evaluation/search metrics: `eval.py`, `eval-full.py`, `eval-rtacc.py`, `models/beam_search.py`.
- Configuration defaults and validation: `src/kgcl/config/schema.py` and `src/kgcl/config/validation.py`.

## Example command path

`python train.py --config configs/default.yaml --epochs 5` loads defaults from `src/kgcl/config/schema.py`, overlays `configs/default.yaml`, applies CLI `epochs=5`, validates the result, then calls `main(args)` in `train.py` to construct datasets, `models.KGCL`, optimizer, scheduler, and checkpoint outputs.
