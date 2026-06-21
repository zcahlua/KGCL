from kgcl.cli import canonicalize, evaluate_full, evaluate_round_trip


def test_canonicalize_historical_defaults():
    config = canonicalize.parse_config([])
    assert config.dataset == "uspto_full"
    assert config.mode == "test"


def test_canonicalize_config_file_overrides_command_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: uspto_50k\nmode: valid\n")
    config = canonicalize.parse_config(["--config", str(path)])
    assert config.dataset == "uspto_50k"
    assert config.mode == "valid"


def test_canonicalize_environment_overrides_command_defaults(monkeypatch):
    monkeypatch.setenv("KGCL_DATASET", "uspto_50k")
    monkeypatch.setenv("KGCL_MODE", "train")
    config = canonicalize.parse_config([])
    assert config.dataset == "uspto_50k"
    assert config.mode == "train"


def test_canonicalize_cli_has_highest_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: uspto_full\nmode: valid\n")
    monkeypatch.setenv("KGCL_MODE", "test")
    config = canonicalize.parse_config([
        "--config", str(path), "--dataset", "uspto_50k", "--mode", "train",
    ])
    assert config.dataset == "uspto_50k"
    assert config.mode == "train"


def test_eval_full_command_default_has_low_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: uspto_50k\n")
    assert evaluate_full.parse_config([]).dataset == "uspto_full"
    assert evaluate_full.parse_config(["--config", str(path)]).dataset == "uspto_50k"
    monkeypatch.setenv("KGCL_DATASET", "uspto_50k")
    assert evaluate_full.parse_config([]).dataset == "uspto_50k"
    assert evaluate_full.parse_config(["--dataset", "uspto_full"]).dataset == "uspto_full"


def test_eval_round_trip_command_default_has_low_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("dataset: uspto_full\n")
    assert evaluate_round_trip.parse_config([]).dataset == "uspto_50k"
    assert evaluate_round_trip.parse_config(["--config", str(path)]).dataset == "uspto_full"
    monkeypatch.setenv("KGCL_DATASET", "uspto_full")
    assert evaluate_round_trip.parse_config([]).dataset == "uspto_full"
    assert evaluate_round_trip.parse_config(["--dataset", "uspto_50k"]).dataset == "uspto_50k"
