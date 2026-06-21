import sys
import types

import pytest

from kgcl.config.device import is_valid_device_request, resolve_device


class FakeDevice:
    def __init__(self, text): self.text = text
    def __str__(self): return self.text


def fake_torch(available):
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 1 if available else 0,
        current_device=lambda: 0,
        set_device=lambda index: None,
        get_device_name=lambda index: 'Fake CUDA',
        get_device_capability=lambda index: (8, 0),
    )
    return types.SimpleNamespace(version=types.SimpleNamespace(cuda='12.1'), cuda=cuda, device=FakeDevice)


def test_device_validation():
    assert is_valid_device_request('cuda')
    assert is_valid_device_request('cuda:1')
    assert not is_valid_device_request('auto')
    assert not is_valid_device_request('cpu')
    assert not is_valid_device_request('gpu')


def test_auto_is_rejected_before_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(False))
    with pytest.raises(ValueError, match='No CPU fallback'):
        resolve_device('auto')


def test_explicit_unavailable_cuda_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(False))
    with pytest.raises(RuntimeError, match='No CPU fallback'):
        resolve_device('cuda')
