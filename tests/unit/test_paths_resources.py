
from kgcl.config.paths import ProjectPaths, historical_checkpoint_name, resolve_checkpoint


def test_serialized_reaction_filename_kekulized_and_plain(tmp_path):
    paths = ProjectPaths(tmp_path)
    assert paths.serialized_reactions_file("USPTO_50k", "valid", True) == tmp_path / "data" / "uspto_50k" / "valid" / "valid.file.kekulized"
    assert paths.serialized_reactions_file("USPTO_50k", "valid", False) == tmp_path / "data" / "uspto_50k" / "valid" / "valid.file"


def test_vocab_dir_always_train(tmp_path):
    paths = ProjectPaths(tmp_path)
    assert paths.vocab_dir("uspto_50k") == tmp_path / "data" / "uspto_50k" / "train"


def test_historical_checkpoint_policy_and_explicit_resolution(tmp_path):
    paths = ProjectPaths(tmp_path)
    assert historical_checkpoint_name("uspto_50k", False) == "epoch_132.pt"
    exp = paths.experiment_dir("uspto_50k", False, "BEST")
    exp.mkdir(parents=True)
    ckpt = exp / "custom.pt"
    ckpt.write_bytes(b"x")
    assert resolve_checkpoint(paths, "uspto_50k", False, "BEST", "custom.pt") == ckpt
