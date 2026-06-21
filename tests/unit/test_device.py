import sys, types, pytest
from kgcl.config.device import is_valid_device_request, resolve_device


def fake_torch(available):
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: available))


def test_device_validation():
    assert is_valid_device_request('auto')
    assert is_valid_device_request('cuda:1')
    assert not is_valid_device_request('gpu')


def test_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(False))
    assert resolve_device('auto') == 'cpu'


def test_explicit_unavailable_cuda_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', fake_torch(False))
    with pytest.raises(RuntimeError):
        resolve_device('cuda')
