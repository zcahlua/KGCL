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
