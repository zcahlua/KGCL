import sys
import types

import pytest

from kgcl.config.device import is_valid_cuda_device_request, resolve_cuda_device, validate_cuda_device_request


class FakeDevice:
    def __init__(self, text):
        self.text = text
        self.type = text.split(':')[0]

    def __str__(self):
        return self.text


def install_fake_torch(monkeypatch, *, cuda_version='12.1', available=True, count=1):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        current_device=lambda: 0,
        set_device=lambda index: None,
        get_device_name=lambda index: 'Fake CUDA GPU',
        get_device_capability=lambda index: (8, 0),
    )
    torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=cuda_version), cuda=cuda, device=FakeDevice)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return torch


def test_cuda_device_validation_rejects_cpu_auto():
    assert is_valid_cuda_device_request('cuda')
    assert is_valid_cuda_device_request('cuda:1')
    for value in ['auto', 'cpu', 'mps', 'gpu', 'cuda:-1', 'cuda:abc']:
        assert not is_valid_cuda_device_request(value)
        with pytest.raises(ValueError, match='No CPU fallback'):
            validate_cuda_device_request(value)


def test_cuda_unavailable(monkeypatch):
    install_fake_torch(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match='No CPU fallback'):
        resolve_cuda_device('cuda')


def test_cpu_only_build(monkeypatch):
    install_fake_torch(monkeypatch, cuda_version=None)
    with pytest.raises(RuntimeError, match='CUDA-enabled PyTorch'):
        resolve_cuda_device('cuda')


def test_invalid_index(monkeypatch):
    install_fake_torch(monkeypatch, count=1)
    with pytest.raises(ValueError, match='only 1 CUDA'):
        resolve_cuda_device('cuda:1')


def test_valid_cuda(monkeypatch):
    install_fake_torch(monkeypatch, count=2)
    resolved = resolve_cuda_device('cuda:0')
    assert str(resolved.torch_device) == 'cuda:0'
    assert resolved.index == 0
    assert resolved.name == 'Fake CUDA GPU'
