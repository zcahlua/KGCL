
def test_config_public_import():
    from kgcl.config import KGCLConfig, load_config
    assert KGCLConfig().dataset == load_config().dataset
