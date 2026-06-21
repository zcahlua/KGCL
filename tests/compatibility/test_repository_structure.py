from pathlib import Path


def test_root_models_and_utils_are_shims_only():
    assert sorted(p.name for p in Path('models').iterdir() if p.is_file()) == ['__init__.py']
    assert sorted(p.name for p in Path('utils').iterdir() if p.is_file()) == ['__init__.py']
