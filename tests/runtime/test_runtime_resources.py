from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch")

from kgcl.resources.functional_groups import configure_functional_group_resources, resolve_functional_group_resources
from utils.rxn_graphs import clear_functional_group_resource_cache, load_functional_group_resources

pytestmark = pytest.mark.runtime


def test_real_resource_bundles_load_from_explicit_root():
    root = Path.cwd()
    configure_functional_group_resources(resource_root=root, root_dir=None)
    clear_functional_group_resource_cache()
    for use_rxn_class in (False, True):
        paths = resolve_functional_group_resources(use_rxn_class=use_rxn_class)
        resources = load_functional_group_resources(use_rxn_class)
        assert paths.definitions.exists()
        assert paths.embeddings.exists()
        assert resources.smarts
        assert set(resources.smart2name.values()).issubset(resources.embeddings.keys())


def test_resource_loading_survives_cwd_change(tmp_path, monkeypatch):
    root = Path.cwd()
    monkeypatch.chdir(tmp_path)
    configure_functional_group_resources(resource_root=root, root_dir=None)
    clear_functional_group_resource_cache()
    assert load_functional_group_resources(False).smarts


def test_resource_loading_uses_environment(monkeypatch):
    monkeypatch.setenv("KGCL_RESOURCE_ROOT", str(Path.cwd()))
    configure_functional_group_resources(resource_root=None, root_dir=None)
    clear_functional_group_resource_cache()
    assert load_functional_group_resources(False).smarts
