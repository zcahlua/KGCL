import pytest

from kgcl.config import ConfigScope, load_config

pytestmark = pytest.mark.runtime


def test_training_scope_accepts_cuda_syntax_without_resolving_gpu():
    cfg = load_config(cli_overrides={'device': 'cuda:0'}, environ={}, scope=ConfigScope.TRAIN)
    assert cfg.device == 'cuda:0'
