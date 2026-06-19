import argparse
import subprocess
import sys

import pytest

from kgcl.config import add_arguments, load_config, load_config_file
from kgcl.config.schema import FIELD_TYPES, KGCLConfig


def test_resolved_field_types_are_callable():
    assert FIELD_TYPES["epochs"] is int
    assert FIELD_TYPES["lr"] is float
    assert FIELD_TYPES["use_rxn_class"] is bool


def test_parser_construction_and_help_for_selected_fields(capsys):
    parser = argparse.ArgumentParser()
    add_arguments(parser, ["epochs", "lr", "use_rxn_class"])
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    assert "--epochs" in capsys.readouterr().out


def test_parser_construction_for_every_config_field(capsys):
    parser = argparse.ArgumentParser()
    add_arguments(parser, list(FIELD_TYPES))
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    assert "--no-kekulize" in capsys.readouterr().out


def test_environment_integer_and_float_conversion():
    cfg = load_config(environ={"KGCL_EPOCHS": "4", "KGCL_LR": "0.25"})
    assert cfg.epochs == 4
    assert isinstance(cfg.epochs, int)
    assert cfg.lr == 0.25
    assert isinstance(cfg.lr, float)


@pytest.mark.parametrize("text", ["1", "true", "yes", "on", "y", "t", "TRUE"])
def test_true_boolean_spellings(text):
    assert load_config(environ={"KGCL_USE_RXN_CLASS": text}).use_rxn_class is True


@pytest.mark.parametrize("text", ["0", "false", "no", "off", "n", "f", "FALSE"])
def test_false_boolean_spellings(text):
    assert load_config(environ={"KGCL_USE_RXN_CLASS": text}).use_rxn_class is False


def test_invalid_boolean_text_rejected():
    with pytest.raises(ValueError, match="Invalid boolean"):
        load_config(environ={"KGCL_USE_RXN_CLASS": "maybe"})


def test_config_file_env_cli_precedence(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("epochs: 2\n")
    cfg = load_config(config_file=path, environ={"KGCL_EPOCHS": "3"}, cli_overrides={"epochs": 4})
    assert cfg.epochs == 4


def test_explicit_false_cli_override():
    cfg = load_config(environ={"KGCL_KEKULIZE": "true"}, cli_overrides={"kekulize": False})
    assert cfg.kekulize is False


def test_preprocess_batch_size_canonical_and_legacy_flags():
    parser = argparse.ArgumentParser()
    add_arguments(parser, ["preprocess_batch_size"])
    assert parser.parse_args(["--preprocess_batch_size", "5"]).preprocess_batch_size == 5
    assert parser.parse_args(["--batch_size", "6"]).preprocess_batch_size == 6


def test_unknown_config_key_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("not_a_field: 1\n")
    with pytest.raises(ValueError, match="Unknown KGCL configuration key"):
        load_config(config_file=path)


def test_malformed_configuration_file_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("dataset: [unterminated\n")
    with pytest.raises(ValueError, match="Malformed configuration file"):
        load_config_file(path)


def test_default_yaml_matches_schema_defaults():
    assert load_config_file("configs/default.yaml") == KGCLConfig().to_dict()


@pytest.mark.parametrize("script", ["train.py", "preprocess.py", "prepare_data.py", "eval.py", "eval-full.py", "eval-rtacc.py"])
def test_public_workflow_help(script):
    result = subprocess.run([sys.executable, script, "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
