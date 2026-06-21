import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.runtime]


def test_cuda_checkpoint_restore_placeholder_requires_gpu():
    pytest.importorskip('torch')
    import torch
    assert torch.version.cuda is not None
    assert torch.cuda.is_available()
