import pytest

from kgcl.config import KGCLConfig
from kgcl.training.runner import run_training

pytestmark = pytest.mark.runtime


def test_training_rejects_unsupported_batch_size_before_imports(tmp_path):
    with pytest.raises(ValueError, match="train_batch_size"):
        run_training(KGCLConfig(root_dir=str(tmp_path), train_batch_size=2))
