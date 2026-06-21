import pickle

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("torch")

from kgcl.resources.functional_groups import configure_functional_group_resources
from utils.rxn_graphs import clear_functional_group_resource_cache, load_functional_group_resources


def make_root(tmp_path, name, value):
    root = tmp_path / name
    bundle = root / "KGembedding"
    bundle.mkdir(parents=True)
    (bundle / "funcgroup.txt").write_text("fg [C]\n")
    with (bundle / "fg2emb.pkl").open("wb") as fh:
        pickle.dump({"fg": value}, fh)
    rc = root / "KGembedding_2"
    rc.mkdir()
    (rc / "funcgroup.txt").write_text("fg2 [O]\n")
    with (rc / "fg2emb.pkl").open("wb") as fh:
        pickle.dump({"fg2": value}, fh)
    return root


@pytest.fixture(autouse=True)
def clear(monkeypatch):
    monkeypatch.delenv("KGCL_RESOURCE_ROOT", raising=False)
    configure_functional_group_resources(resource_root=None, root_dir=None)
    clear_functional_group_resource_cache()
    yield
    configure_functional_group_resources(resource_root=None, root_dir=None)
    clear_functional_group_resource_cache()


def test_same_root_returns_cached_object(tmp_path):
    root = make_root(tmp_path, "a", [1])
    configure_functional_group_resources(resource_root=root, root_dir=None)
    first = load_functional_group_resources(False)
    assert load_functional_group_resources(False) is first


def test_environment_change_selects_new_cache_entry(tmp_path, monkeypatch):
    a = make_root(tmp_path, "a", [1])
    b = make_root(tmp_path, "b", [2])
    monkeypatch.setenv("KGCL_RESOURCE_ROOT", str(a))
    first = load_functional_group_resources(False)
    monkeypatch.setenv("KGCL_RESOURCE_ROOT", str(b))
    second = load_functional_group_resources(False)
    assert second is not first


def test_explicit_resource_root_wins_over_environment(tmp_path, monkeypatch):
    a = make_root(tmp_path, "a", [1])
    b = make_root(tmp_path, "b", [2])
    monkeypatch.setenv("KGCL_RESOURCE_ROOT", str(a))
    configure_functional_group_resources(resource_root=b, root_dir=None)
    assert "2" in str(load_functional_group_resources(False).embeddings["fg"])


def test_reaction_class_modes_are_independent(tmp_path):
    root = make_root(tmp_path, "a", [1])
    configure_functional_group_resources(resource_root=root, root_dir=None)
    assert load_functional_group_resources(False) is not load_functional_group_resources(True)


def test_missing_resources_raise_actionable_error(tmp_path):
    configure_functional_group_resources(resource_root=tmp_path / "missing", root_dir=None)
    with pytest.raises(FileNotFoundError, match="functional-group resources"):
        load_functional_group_resources(False)
