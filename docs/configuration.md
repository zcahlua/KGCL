# KGCL configuration

Configuration precedence is built-in defaults, optional config file, `KGCL_*` environment variables, then CLI flags.

| Parameter | Type | Default | CLI | Environment | Runtime consumer | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | str | uspto_50k | --dataset | KGCL_DATASET | all workflows | non-empty |
| mode | str | train | --mode | KGCL_MODE | data workflows | train/valid/test |
| root_dir | str | ./ | --root-dir | KGCL_ROOT_DIR | ProjectPaths | non-empty |
| resource_root | str? | null | --resource-root | KGCL_RESOURCE_ROOT | resource resolver | non-empty if set |
| device | str | auto | --device | KGCL_DEVICE | training/evaluation | auto/cpu/cuda/cuda:N |
| train_batch_size | int | 1 | --train-batch-size | KGCL_TRAIN_BATCH_SIZE | training compatibility | must equal 1 |
| preprocess_batch_size | int | 256 | --preprocess-batch-size | KGCL_PREPROCESS_BATCH_SIZE | data preparation shards | positive |
| beam_size | int | 50 | --beam-size | standard/round-trip eval | KGCL_BEAM_SIZE | positive |
| full_beam_size | int | 10 | --full-beam-size | full eval | KGCL_FULL_BEAM_SIZE | positive |
| step_beam_size | int | 10 | --step-beam-size | all evaluators | KGCL_STEP_BEAM_SIZE | positive |
| checkpoint | str? | null | --checkpoint | KGCL_CHECKPOINT | evaluators | non-empty if set |
| output_path | str? | null | --output-path | KGCL_OUTPUT_PATH | evaluators | non-empty if set |
| forward_predictions_path | str? | null | --forward-predictions-path | KGCL_FORWARD_PREDICTIONS_PATH | round-trip eval | non-empty if set |
| kekulize | bool | true | --kekulize/--no-kekulize | KGCL_KEKULIZE | data/evaluation file selection | boolean |
