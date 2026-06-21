import pytest

from kgcl.config import ConfigScope, load_config


def test_eval_scope_ignores_training_batch_env():
    cfg = load_config(environ={'KGCL_TRAIN_BATCH_SIZE': '2'}, scope=ConfigScope.EVALUATE)
    assert cfg.train_batch_size == 2


def test_training_scope_rejects_training_batch_env():
    with pytest.raises(ValueError, match='train_batch_size'):
        load_config(environ={'KGCL_TRAIN_BATCH_SIZE': '2'}, scope=ConfigScope.TRAIN)


def test_eval_scope_rejects_cpu_device():
    with pytest.raises(ValueError, match='requires CUDA'):
        load_config(cli_overrides={'device': 'cpu'}, environ={}, scope=ConfigScope.EVALUATE)
