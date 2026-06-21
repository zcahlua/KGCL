# KGCL configuration

Configuration precedence is: schema defaults < command-specific defaults < configuration file < `KGCL_*` environment variables < explicit CLI arguments.

Command-specific defaults: `canonicalize_prod.py` uses `dataset=uspto_full` and `mode=test`; `eval-full.py` uses `dataset=uspto_full`; `eval-rtacc.py` uses `dataset=uspto_50k`.

| Parameter | Type | Default | CLI | Environment | Runtime consumer | Validation |
|---|---:|---|---|---|---|---|
| dataset | str | uspto_50k | `--dataset` | KGCL_DATASET | all workflows | normalized lowercase |
| mode | str | train | `--mode` | KGCL_MODE | data workflows | train/valid/test |
| root_dir | str | ./ | `--root-dir` | KGCL_ROOT_DIR | paths | resolved by ProjectPaths |
| resource_root | str? | None | `--resource-root` | KGCL_RESOURCE_ROOT | runtime resources | must contain KGembedding dirs when used |
| experiments | str | BEST | `--experiments` | KGCL_EXPERIMENTS | train/eval | path segment |
| use_rxn_class | bool | False | `--use-rxn-class` | KGCL_USE_RXN_CLASS | model/resources | boolean |
| kekulize | bool | True | `--kekulize` | KGCL_KEKULIZE | preprocessing/eval | boolean |
| atom_message | bool | False | `--atom-message` | KGCL_ATOM_MESSAGE | model | boolean |
| use_attn | bool | False | `--use-attn` | KGCL_USE_ATTN | model | boolean |
| n_heads | int | 8 | `--n-heads` | KGCL_N_HEADS | model | int |
| mpn_size | int | 256 | `--mpn-size` | KGCL_MPN_SIZE | model | int |
| depth | int | 10 | `--depth` | KGCL_DEPTH | model | int |
| dropout_mpn | float | 0.15 | `--dropout-mpn` | KGCL_DROPOUT_MPN | model | float |
| mlp_size | int | 512 | `--mlp-size` | KGCL_MLP_SIZE | model | int |
| dropout_mlp | float | 0.2 | `--dropout-mlp` | KGCL_DROPOUT_MLP | model | float |
| epochs | int | 200 | `--epochs` | KGCL_EPOCHS | training | int |
| lr | float? | None | `--lr` | KGCL_LR | training | dataset fallback if unset |
| patience | int | 5 | `--patience` | KGCL_PATIENCE | training | int |
| factor | float | 0.8 | `--factor` | KGCL_FACTOR | training | float |
| thresh | float | 0.01 | `--thresh` | KGCL_THRESH | training | float |
| max_clip | int | 10 | `--max-clip` | KGCL_MAX_CLIP | training | int |
| train_batch_size | int | 1 | `--train-batch-size` | KGCL_TRAIN_BATCH_SIZE | training | must be 1 |
| preprocess_batch_size | int | 256 | `--preprocess-batch-size`, `--batch_size` | KGCL_PREPROCESS_BATCH_SIZE | preparation | int |
| beam_size | int | 50 | `--beam-size` | KGCL_BEAM_SIZE | eval | int |
| full_beam_size | int | 10 | `--full-beam-size` | KGCL_FULL_BEAM_SIZE | eval-full | int |
| step_beam_size | int | 10 | `--step-beam-size` | KGCL_STEP_BEAM_SIZE | eval | int |
| checkpoint | str? | None | `--checkpoint` | KGCL_CHECKPOINT | eval | default historical when unset |
| output_path | str? | None | `--output-path` | KGCL_OUTPUT_PATH | eval | parent created |
| forward_predictions_path | str? | None | `--forward-predictions-path` | KGCL_FORWARD_PREDICTIONS_PATH | rt eval | must exist |
| max_steps | int | 9 | `--max-steps` | KGCL_MAX_STEPS | preprocessing/eval | int |
| num_workers | int | 24 | `--num-workers` | KGCL_NUM_WORKERS | loaders | int |
| print_every | int | 200 | `--print-every` | KGCL_PRINT_EVERY | training | int |
| preprocess_print_every | int | 1000 | `--preprocess-print-every` | KGCL_PREPROCESS_PRINT_EVERY | preprocessing | int |
| device | str | auto | `--device` | KGCL_DEVICE | runtime | auto/cpu/cuda |
