# KGCL configuration

Precedence is schema defaults, command defaults, config file, `KGCL_*` environment variables, then explicit CLI values. YAML files are loaded with `yaml.safe_load`; JSON files are loaded with `json.loads`.

| Parameter | Type | Default | CLI | Environment | Runtime consumer | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | str | uspto_50k | --dataset | KGCL_DATASET | all workflows | supported dataset |
| root_dir | str | . | --root-dir | KGCL_ROOT_DIR | ProjectPaths | path-like |
| resource_root | str? | None | --resource-root | KGCL_RESOURCE_ROOT | resource resolver | path-like |
| mode | str | train | --mode | KGCL_MODE | data workflows | train/valid/test |
| device | str | auto | --device | KGCL_DEVICE | training/evaluation | auto/cpu/cuda |
| use_rxn_class | bool | false | --use-rxn-class | KGCL_USE_RXN_CLASS | model/data/eval | boolean |
| output_path | str? | None | --output-path | KGCL_OUTPUT_PATH | evaluation | path-like |
| checkpoint | str? | None | --checkpoint | KGCL_CHECKPOINT | evaluation | path-like |
| forward_predictions_path | str? | None | --forward-predictions-path | KGCL_FORWARD_PREDICTIONS_PATH | round-trip eval | path-like |
| train_batch_size | int | 1 | --train-batch-size | KGCL_TRAIN_BATCH_SIZE | training guard | must be 1 |
| preprocess_batch_size | int | 1000 | --preprocess-batch-size/--batch_size | KGCL_PREPROCESS_BATCH_SIZE | data sharding | positive int |
| kekulize | bool | true | --kekulize/--no-kekulize | KGCL_KEKULIZE | data/eval | boolean |
