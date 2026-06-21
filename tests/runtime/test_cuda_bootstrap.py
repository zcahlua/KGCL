import pytest

from kgcl.config.device import validate_cuda_device_request

pytestmark = pytest.mark.runtime


def test_cuda_bootstrap_rejects_cpu_without_importing_torch():
    with pytest.raises(ValueError, match='No CPU fallback'):
        validate_cuda_device_request('cpu')
