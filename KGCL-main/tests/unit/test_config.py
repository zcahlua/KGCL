import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from kgcl.config import load_config


def test_load_default_configuration():
    cfg = load_config()
    assert cfg.dataset == "uspto_50k"
    assert cfg.epochs == 200


def test_config_file_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("epochs: 3\ndropout_mpn: 0.25\n")
    cfg = load_config(config_file=config)
    assert cfg.epochs == 3
    assert cfg.dropout_mpn == 0.25


def test_precedence_config_env_cli(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("epochs: 3\n")
    cfg = load_config(config_file=config, environ={"KGCL_EPOCHS": "4"}, cli_overrides={"epochs": 5})
    assert cfg.epochs == 5


def test_invalid_config_values():
    with pytest.raises(ValueError, match="lr must be positive"):
        load_config(cli_overrides={"lr": 0})


def test_backward_compatible_batch_size_alias():
    cfg = load_config(cli_overrides={"batch_size": 7})
    assert cfg.preprocess_batch_size == 7
