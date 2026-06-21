import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.runtime]


def test_cuda_minimal_inference_environment():
    pytest.importorskip('torch')
    import torch
    assert torch.cuda.is_available()
