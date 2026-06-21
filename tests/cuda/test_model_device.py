import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.runtime]


def test_cuda_device_available_for_model_placement():
    pytest.importorskip('torch')
    import torch
    assert torch.cuda.is_available()
