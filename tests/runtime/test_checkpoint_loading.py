from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

pytestmark = pytest.mark.runtime


def test_committed_checkpoints_load_read_only():
    checkpoints = sorted(Path("experiments").glob("**/*.pt")) + sorted(Path("experiments").glob("**/*.pth"))
    assert checkpoints
    for path in checkpoints:
        obj = torch.load(path, map_location="cpu")
        assert isinstance(obj, dict), path
        assert "state" in obj, path
        if "BEST" in path.parts:
            assert "saveables" in obj, path
