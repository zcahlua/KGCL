# KGCL configuration

KGCL uses `src/kgcl/config/schema.py` as the authoritative configuration schema. Values are loaded with this precedence:

```text
defaults < config file < KGCL_* environment variables < command-line arguments
```

Config files may be JSON or YAML parsed with `yaml.safe_load`; unknown keys and malformed files fail fast. Environment values are coerced to the schema type, including integers, floats, and booleans. Boolean text accepts `1/0`, `true/false`, `yes/no`, `on/off`, `y/n`, and `t/f`.

Generated CLI booleans use explicit true and false forms, for example:

```bash
python preprocess.py --kekulize
python preprocess.py --no-kekulize
```

Preprocessing batch size supports both spellings:

```bash
python prepare_data.py --preprocess_batch_size 256
python prepare_data.py --batch_size 256  # legacy alias
```

`lr` defaults to an unset value in the shared schema. Training resolves the historical dataset policy only when no explicit file, environment, or CLI learning rate is supplied: `0.001` for USPTO-50K and `0.0001` for USPTO-FULL.

## Added evaluation/path fields

| Canonical key | Legacy key/flag | Type | Default | Consumer | Validation |
| --- | --- | --- | --- | --- | --- |
| `step_beam_size` | `--step-beam-size`, `--step_beam_size`, `KGCL_STEP_BEAM_SIZE` | int | 10 | Evaluation beam-search construction | must be at least 1 |
| `checkpoint` | `--checkpoint`, `KGCL_CHECKPOINT` | string/null | null | Checkpoint resolution policy | explicit path must exist before loading |
| `output_path` | `--output-path`, `--output_path`, `KGCL_OUTPUT_PATH` | string/null | null | Prediction writers | parent directory is created by consumers |
| `forward_predictions_path` | `--forward-predictions-path`, `--forward_predictions_path`, `KGCL_FORWARD_PREDICTIONS_PATH` | string/null | null | Round-trip evaluation | file must exist before consuming |
