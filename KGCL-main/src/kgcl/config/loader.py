"""Load KGCL configuration from defaults, files, environment, and CLI overrides."""
from __future__ import annotations

import argparse, json, os, warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .schema import ALIASES, DEFAULTS, FIELD_TYPES, KGCLConfig, PARAMETER_HELP
from .validation import validate_config

ENV_PREFIX = "KGCL_"


def _coerce(name: str, value: Any) -> Any:
    if name in ALIASES:
        warnings.warn(f"Configuration key '{name}' is deprecated; use '{ALIASES[name]}' instead.", DeprecationWarning, stacklevel=2)
        name = ALIASES[name]
    target = FIELD_TYPES[name]
    if isinstance(value, str):
        value = value.strip()
    if target is bool:
        if isinstance(value, bool): return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    if target is int: return int(value)
    if target is float: return float(value)
    return value


def _flatten(prefix: str, data: Mapping[str, Any], out: Dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, Mapping):
            _flatten(key, value, out)
        else:
            out[key] = value


def load_config_file(path: str | os.PathLike[str]) -> Dict[str, Any]:
    text = Path(path).read_text()
    try:
        raw = json.loads(text)
        if not isinstance(raw, Mapping):
            raise json.JSONDecodeError("configuration root must be an object", text, 0)
    except json.JSONDecodeError:
        raw = {}
        stack = [raw]
        for line in text.splitlines():
            line = line.split("#", 1)[0].rstrip()
            if not line: continue
            if line.endswith(":"):
                key = line[:-1].strip()
                raw[key] = {}
                stack = [raw, raw[key]]
                continue
            if ":" in line:
                key, value = [part.strip() for part in line.split(":", 1)]
                if value.lower() in {"true", "false"}: parsed = value.lower() == "true"
                else:
                    try: parsed = int(value)
                    except ValueError:
                        try: parsed = float(value)
                        except ValueError: parsed = value.strip('"\'')
                stack[-1][key] = parsed
    flat: Dict[str, Any] = {}
    _flatten("", raw, flat)
    return flat


def load_config(config_file: str | None = None, cli_overrides: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None) -> KGCLConfig:
    values = DEFAULTS.to_dict()
    if config_file:
        values.update(load_config_file(config_file))
    env = os.environ if environ is None else environ
    for name in list(values):
        env_name = ENV_PREFIX + name.upper()
        if env_name in env:
            values[name] = env[env_name]
    if cli_overrides:
        values.update({k: v for k, v in cli_overrides.items() if v is not None})
    normalized: Dict[str, Any] = {}
    for key, value in values.items():
        mapped = ALIASES.get(key, key)
        if key in ALIASES:
            warnings.warn(f"Configuration key '{key}' is deprecated; use '{mapped}' instead.", DeprecationWarning, stacklevel=2)
        if mapped not in FIELD_TYPES:
            raise ValueError(f"Unknown KGCL configuration key: {key}")
        normalized[mapped] = _coerce(mapped, value)
    normalized["dataset"] = str(normalized["dataset"]).strip().lower()
    validate_config(normalized)
    return KGCLConfig(**normalized)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", dest="config_file", default=None, help="Optional KGCL YAML/JSON config file. Precedence: defaults < config < KGCL_* env < CLI.")


def add_arguments(parser: argparse.ArgumentParser, names: Sequence[str]) -> None:
    for name in names:
        flag = "--" + ("batch_size" if name == "preprocess_batch_size" else name)
        kwargs = {"default": None, "help": PARAMETER_HELP.get(name, name)}
        typ = FIELD_TYPES[name]
        if typ is bool:
            parser.add_argument(flag, dest=name, action="store_true", default=None, help=kwargs["help"])
        else:
            parser.add_argument(flag, dest=name, type=typ, **kwargs)
