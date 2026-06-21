import pytest


pytestmark = pytest.mark.runtime


def test_evaluation_setup_resolves_paths(tmp_path):
    joblib = pytest.importorskip("joblib")
    from kgcl.config import KGCLConfig
    from kgcl.evaluation.common import evaluation_setup
    data_dir = tmp_path / "data" / "uspto_50k" / "test"
    data_dir.mkdir(parents=True)
    joblib.dump([], data_dir / "test.file.kekulized")
    exp = tmp_path / "experiments" / "uspto_50k" / "without_rxn_class" / "BEST"
    exp.mkdir(parents=True)
    ckpt = exp / "epoch_132.pt"
    ckpt.write_bytes(b"placeholder")
    cfg = KGCLConfig(root_dir=str(tmp_path), checkpoint=str(ckpt), output_path="pred.txt", device="cpu")
    context = evaluation_setup(cfg)
    assert str(context.device) == "cpu"
    assert context.test_data_path == data_dir / "test.file.kekulized"
    assert context.output_path == tmp_path / "pred.txt"
