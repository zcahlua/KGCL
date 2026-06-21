import pytest

from kgcl.config.device import validate_cuda_device_request

pytestmark = [pytest.mark.cuda, pytest.mark.runtime]


def test_real_runtime_rejects_cpu_before_torch_cuda_resolution():
    with pytest.raises(ValueError, match='No CPU fallback'):
        validate_cuda_device_request('cpu')
