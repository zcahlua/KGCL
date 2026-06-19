# KGCL configuration

## Precedence

For scripts that use the centralized loader (`train.py`, `preprocess.py`, and `prepare_data.py`), values are resolved as:

1. Built-in defaults in `src/kgcl/config/schema.py`.
2. Optional YAML/JSON file passed with `--config`.
3. Environment variables named `KGCL_<PARAMETER>`.
4. Explicit command-line arguments.

Evaluation scripts keep their historical CLI-only behavior for compatibility; their parameters are still inventoried in `schema.py`.

## Parameter groups

- Data/paths: `dataset`, `mode`, `root_dir`, `experiments`, `use_rxn_class`, `kekulize`.
- Model: `atom_message`, `use_attn`, `n_heads`, `mpn_size`, `depth`, `dropout_mpn`, `mlp_size`, `dropout_mlp`.
- Training/optimization: `epochs`, `lr`, `patience`, `factor`, `thresh`, `max_clip`, `train_batch_size`.
- Preprocessing/evaluation: `preprocess_batch_size`, `beam_size`, `full_beam_size`, `max_steps`.
- Runtime/logging: `num_workers`, `print_every`, `preprocess_print_every`, `device`.

## Validation

The loader rejects invalid values with actionable messages: positive learning rate, non-empty dataset, allowed mode, positive epoch/model sizes, probabilities in `[0, 1]`, and scheduler factor in `(0, 1]`.

## Examples

```bash
python train.py --config configs/default.yaml --epochs 5 --num_workers 0
KGCL_EPOCHS=5 python train.py --config configs/default.yaml
python preprocess.py --config configs/examples/minimal_train.yaml --mode valid
python prepare_data.py --dataset uspto_50k --batch_size 128
```

## Migration notes

The historical `--batch_size` preprocessing option is accepted and mapped to `preprocess_batch_size`. Existing top-level commands remain available.
