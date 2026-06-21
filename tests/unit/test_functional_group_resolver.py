from pathlib import Path

import pytest

from kgcl.resources.functional_groups import resolve_functional_group_resources


def test_resolve_existing_repository_resources_from_checkout():
    paths = resolve_functional_group_resources(use_rxn_class=False)
    assert paths.definitions.name == "funcgroup.txt"
    assert paths.embeddings == Path("KGembedding/fg2emb.pkl").resolve()


def test_resolve_existing_repository_resources_after_chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = resolve_functional_group_resources(use_rxn_class=True)
    assert paths.embeddings == (Path(__file__).resolve().parents[2] / "KGembedding_2" / "fg2emb.pkl").resolve()


def test_resource_root_override(tmp_path):
    (tmp_path / "KGembedding").mkdir()
    (tmp_path / "KGembedding" / "funcgroup.txt").write_text("fg [C]\n")
    (tmp_path / "KGembedding" / "fg2emb.pkl").write_bytes(b"temporary-test-binary")
    paths = resolve_functional_group_resources(use_rxn_class=False, resource_root=tmp_path)
    assert paths.definitions == (tmp_path / "KGembedding" / "funcgroup.txt").resolve()
    assert paths.embeddings == (tmp_path / "KGembedding" / "fg2emb.pkl").resolve()


def test_environment_resource_root_override(tmp_path, monkeypatch):
    (tmp_path / "KGembedding_2").mkdir()
    (tmp_path / "KGembedding_2" / "funcgroup.txt").write_text("fg [N]\n")
    (tmp_path / "KGembedding_2" / "fg2emb.pkl").write_bytes(b"temporary-test-binary")
    monkeypatch.setenv("KGCL_RESOURCE_ROOT", str(tmp_path))
    paths = resolve_functional_group_resources(use_rxn_class=True)
    assert paths.embeddings == (tmp_path / "KGembedding_2" / "fg2emb.pkl").resolve()


def test_missing_resources_error_lists_attempted_paths(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_functional_group_resources(use_rxn_class=False, resource_root=tmp_path)
    message = str(excinfo.value)
    assert "KGembedding" in message
    assert "funcgroup.txt" in message
    assert "fg2emb.pkl" in message
    assert "KGCL_RESOURCE_ROOT" in message
